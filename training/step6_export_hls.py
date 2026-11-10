"""
STEP 6 -- Export the trained GreenCNN for HLS (FPGA).

  1. fold BatchNorm into the conv weights
  2. quantize: weights int8 (per output channel), activations uint8, bias int32
  3. requantize between layers with integer multiply + shift (no floats in hardware)
  4. run an INTEGER-ONLY golden model in numpy -> must match what the HLS C++ computes
  5. write hls_export/cnn_weights.h  and  hls_export/cnn_test_vectors.h  (for Vitis HLS)
  6. write deploy/cnn_meta.json, cnn_test_vectors.npz, cnn_int8_params.npz (for PYNQ / PC sim)

Integer maths per conv layer (exactly what the FPGA will do):
    acc  = bias_q[c] + sum( w_q[c,k,ky,kx] * a_in[k, y+ky-1, x+kx-1] )    (int32)
    a    = clip( (max(acc,0) * M_q[c] + 2^(SHIFT-1)) >> SHIFT , 0, 255 )  (uint8)
    then 2x2 max-pool
Final: GAP = sum over 4x4 (no divide), logits = fc_b_q + fc_w_q . GAP, class = argmax.

Run:  python step6_export_hls.py
"""
import json
import os

import numpy as np
import torch

import config as C
from cnn_model import GreenCNN
from common import center_crop, load_cache, metrics

SHIFT = 24          # requantization shift
N_VECTORS = 16      # test images written into the HLS testbench header


# --------------------------------------------------------------------------- float model pieces
def fold_bn(conv, bn):
    w = conv.weight.detach().numpy().astype(np.float64)
    g = bn.weight.detach().numpy() / np.sqrt(bn.running_var.detach().numpy() + bn.eps)
    w_f = w * g[:, None, None, None]
    b_f = bn.bias.detach().numpy() - bn.running_mean.detach().numpy() * g
    return w_f, b_f


def conv3x3(x, w, b):
    """float or int conv, x: C x H x W, w: O x C x 3 x 3, pad 1."""
    c, h, wd = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    cols = np.lib.stride_tricks.sliding_window_view(xp, (3, 3), axis=(1, 2))  # C,H,W,3,3
    out = np.einsum("chwij,ocij->ohw", cols, w, optimize=True)
    return out + b[:, None, None]


def maxpool2(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def float_forward(layers, fc_w, fc_b, img_u8, record=None):
    x = img_u8.transpose(2, 0, 1).astype(np.float64) / 255.0
    for i, (w, b) in enumerate(layers):
        x = np.maximum(conv3x3(x, w, b), 0)
        if record is not None:
            record[i].append(x.ravel())
        x = maxpool2(x)
    return fc_w @ x.mean(axis=(1, 2)) + fc_b


# --------------------------------------------------------------------------- integer model
def int_forward(q, img_u8):
    a = img_u8.transpose(2, 0, 1).astype(np.int64)
    for L in q["conv"]:
        acc = conv3x3(a, L["w"].astype(np.int64), L["b"].astype(np.int64))
        acc = np.maximum(acc, 0)
        a = (acc * L["m"][:, None, None] + (1 << (SHIFT - 1))) >> SHIFT
        a = maxpool2(np.clip(a, 0, 255))
    gap = a.sum(axis=(1, 2))
    return q["fc_b"].astype(np.int64) + q["fc_w"].astype(np.int64) @ gap


def c_array(name, arr, ctype):
    flat = ", ".join(str(int(v)) for v in arr.ravel())
    dims = "".join(f"[{d}]" for d in arr.shape)
    return f"static const {ctype} {name}{dims} = {{ {flat} }};\n"


def main():
    ck = torch.load(os.path.join(C.MODEL_DIR, "dl_best.pt"), map_location="cpu")
    model = GreenCNN(widths=tuple(ck["widths"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()

    mods = list(model.features)
    layers = [fold_bn(mods[i], mods[i + 1]) for i in range(0, len(mods), 4)]
    widths = [w.shape[0] for w, _ in layers]
    if len(layers) != 4 or any(c % 8 for c in widths) or C.IMG_SIZE != 64:
        raise SystemExit(f"HLS kernel expects 4 conv layers with channels divisible by 8 and "
                         f"64x64 input; got {widths}, {C.IMG_SIZE}. Keep CNN_WIDTHS like (16,32,48,64).")
    fc_w = model.fc.weight.detach().numpy().astype(np.float64)
    fc_b = model.fc.bias.detach().numpy().astype(np.float64)

    data = load_cache()
    Xtr = center_crop(data["train"][0])
    Xte, yte = center_crop(data["test"][0]), data["test"][1]

    # ---- calibrate activation ranges on training images
    rng = np.random.default_rng(C.SEED)
    calib = Xtr[rng.choice(len(Xtr), size=min(400, len(Xtr)), replace=False)]
    rec = [[] for _ in layers]
    for img in calib:
        float_forward(layers, fc_w, fc_b, img, rec)
    s_act = [1.0 / 255.0]
    for r in rec:
        s_act.append(max(np.percentile(np.concatenate(r), 99.99), 1e-6) / 255.0)

    # ---- quantize
    q = {"conv": []}
    for i, (w, b) in enumerate(layers):
        s_in, s_out = s_act[i], s_act[i + 1]
        # floor 1e-4 keeps "dead" channels from creating huge int32 biases
        s_w = np.maximum(np.abs(w).reshape(w.shape[0], -1).max(1), 1e-4) / 127.0
        w_q = np.clip(np.round(w / s_w[:, None, None, None]), -127, 127).astype(np.int8)
        b_q = np.round(b / (s_in * s_w)).astype(np.int64)
        M = s_in * s_w / s_out
        assert M.max() < 128, "requant multiplier too large"
        m_q = np.round(M * (1 << SHIFT)).astype(np.int64)
        # the FPGA accumulates in int32 -> make sure nothing can overflow
        worst = np.abs(b_q) + 127 * 255 * w[0].size
        assert worst.max() < 2 ** 31 - 1, f"layer {i + 1}: int32 accumulator could overflow"
        assert m_q.max() < 2 ** 31 - 1, f"layer {i + 1}: requant multiplier too large"
        q["conv"].append({"w": w_q, "b": b_q, "m": m_q})
    s4 = s_act[-1]
    s_fc = max(np.abs(fc_w).max(), 1e-6) / 127.0
    q["fc_w"] = np.clip(np.round(fc_w / s_fc), -127, 127).astype(np.int8)
    q["fc_b"] = np.round(fc_b / (s_fc * s4 / 16.0)).astype(np.int64)   # GAP is a sum of 16
    logit_scale = float(s_fc * s4 / 16.0)     # float_logit = int_logit * logit_scale

    # ---- accuracy: float (folded) vs int8
    pf = np.array([np.argmax(float_forward(layers, fc_w, fc_b, x)) for x in Xte])
    pi = np.array([np.argmax(int_forward(q, x)) for x in Xte])
    mf, mi = metrics(yte, pf), metrics(yte, pi)
    agree = float((pf == pi).mean()) if len(pf) else 0.0
    print(f"float  test acc {mf['accuracy']:.3f}  macro-F1 {mf['macro_f1']:.3f}")
    print(f"int8   test acc {mi['accuracy']:.3f}  macro-F1 {mi['macro_f1']:.3f}")
    print(f"float/int8 prediction agreement {agree:.3f}")
    with open(os.path.join(C.HLS_EXPORT_DIR, "int8_metrics.json"), "w") as f:
        json.dump({"float": mf, "int8": mi, "agreement": agree}, f, indent=2)
    meta = {"classes": C.CLASS_NAMES, "img_size": C.IMG_SIZE, "logit_scale": logit_scale,
            "req_shift": SHIFT, "widths": widths, "int8_test": mi}
    for d in (C.HLS_EXPORT_DIR, C.DEPLOY_DIR):
        with open(os.path.join(d, "cnn_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ---- headers for HLS
    h = ["// Auto-generated by step6_export_hls.py -- GreenCNN int8 weights",
         "#ifndef CNN_WEIGHTS_H", "#define CNN_WEIGHTS_H", "",
         f"#define IMG_SIZE {C.IMG_SIZE}", f"#define NUM_CLASSES {C.NUM_CLASSES}",
         f"#define REQ_SHIFT {SHIFT}", "#define PAR 8   // output channels computed in parallel"]
    cin = 3
    for i, w in enumerate(widths, 1):
        h.append(f"#define L{i}_CIN {cin}\n#define L{i}_COUT {w}")
        cin = w
    h.append("")
    body = ""
    for i, L in enumerate(q["conv"], 1):
        body += c_array(f"W{i}", L["w"], "signed char")
        body += c_array(f"B{i}", L["b"], "int")
        body += c_array(f"M{i}", L["m"], "int")
    body += c_array("FC_W", q["fc_w"], "signed char")
    body += c_array("FC_B", q["fc_b"], "int")
    names = ", ".join(f'"{n}"' for n in C.CLASS_NAMES)
    with open(os.path.join(C.HLS_EXPORT_DIR, "cnn_weights.h"), "w") as f:
        f.write("\n".join(h) + "\n" + body +
                f"#ifndef __SYNTHESIS__   // names are only used by the testbench\n"
                f"static const char *CLASS_NAMES[NUM_CLASSES] = {{ {names} }};\n#endif\n\n#endif\n")

    # ---- test vectors: images + expected integer logits (bit-exact check for HLS)
    # pick test images round-robin over the classes so every class is checked
    by_cls = [list(np.flatnonzero(yte == k)) for k in range(C.NUM_CLASSES)]
    sel = []
    while len(sel) < min(N_VECTORS, len(Xte)) and any(by_cls):
        for lst in by_cls:
            if lst and len(sel) < N_VECTORS:
                sel.append(lst.pop(0))
    sel = np.array(sel, dtype=int)
    n = len(sel)
    imgs = Xte[sel].transpose(0, 3, 1, 2)                    # N x 3 x 64 x 64 (CHW)
    logits = (np.stack([int_forward(q, x) for x in Xte[sel]]) if n
              else np.zeros((0, C.NUM_CLASSES)))
    with open(os.path.join(C.HLS_EXPORT_DIR, "cnn_test_vectors.h"), "w") as f:
        f.write("// Auto-generated test vectors: uint8 CHW images + expected int logits\n")
        f.write("#ifndef CNN_TEST_VECTORS_H\n#define CNN_TEST_VECTORS_H\n")
        f.write(f"#define N_VECTORS {n}\n")
        f.write(c_array("TV_IMG", imgs.astype(np.uint8), "unsigned char"))
        f.write(c_array("TV_LOGITS", logits.astype(np.int64), "long long"))
        f.write(c_array("TV_LABEL", yte[sel].astype(np.int64), "int"))
        f.write("#endif\n")
    for d in (C.HLS_EXPORT_DIR, C.DEPLOY_DIR):       # same vectors for the PYNQ board test
        np.savez(os.path.join(d, "cnn_test_vectors.npz"), img=imgs.astype(np.uint8),
                 logits=logits.astype(np.int64), label=yte[sel].astype(np.int64))

    params = dict(**{f"W{i}": L["w"] for i, L in enumerate(q["conv"], 1)},
                  **{f"B{i}": L["b"] for i, L in enumerate(q["conv"], 1)},
                  **{f"M{i}": L["m"] for i, L in enumerate(q["conv"], 1)},
                  FC_W=q["fc_w"], FC_B=q["fc_b"], SHIFT=SHIFT)
    np.savez(os.path.join(C.DEPLOY_DIR, "cnn_int8_params.npz"), **params)   # for --sim on PC
    np.savez(os.path.join(C.HLS_EXPORT_DIR, "cnn_int8_params.npz"),
             **{f"W{i}": L["w"] for i, L in enumerate(q["conv"], 1)},
             **{f"B{i}": L["b"] for i, L in enumerate(q["conv"], 1)},
             **{f"M{i}": L["m"] for i, L in enumerate(q["conv"], 1)},
             FC_W=q["fc_w"], FC_B=q["fc_b"], SHIFT=SHIFT)
    nparam = sum(L["w"].size for L in q["conv"]) + q["fc_w"].size
    print(f"\nint8 weights: {nparam:,} bytes  -> {C.HLS_EXPORT_DIR}")
    print("  cnn_weights.h, cnn_test_vectors.h  -> copy both into the HLS project folder")
    print(f"  cnn_meta.json, cnn_test_vectors.npz, cnn_int8_params.npz -> {C.DEPLOY_DIR}")
    print("STEP 6 OK")


if __name__ == "__main__":
    main()
