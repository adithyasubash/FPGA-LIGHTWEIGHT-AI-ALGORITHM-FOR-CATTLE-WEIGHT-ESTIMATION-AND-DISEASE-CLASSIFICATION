"""
STEP 8 -- Fuse the two models (exactly like the PYNQ board will):

  p_cnn  = softmax( int8 GreenCNN logits * logit_scale )    <- bit-exact FPGA model
  p_mnv2 = softmax( MobileNetV2 logits )                    <- ARM CPU model
  p      = w * p_cnn + (1 - w) * p_mnv2                     <- final decision

The weight w is chosen on the VALIDATION set, then all three are reported on the TEST set.
Writes deploy/fusion.json (used by the camera app) and outputs/fusion/ (confusion matrices).

Run:  python step8_fusion.py
"""
import json
import os

import numpy as np
import torch

import config as C
from common import center_crop, load_cache, metrics, save_report
from step6_export_hls import int_forward
from step7_train_mobilenetv2 import MobileNetV2Head, predict_proba

OUT = os.path.join(C.OUT_DIR, "fusion")


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cnn_int8_proba(X, params, scale):
    q = {"conv": [{"w": params[f"W{i}"], "b": params[f"B{i}"], "m": params[f"M{i}"]}
                  for i in range(1, 5)],
         "fc_w": params["FC_W"], "fc_b": params["FC_B"]}
    logits = np.stack([int_forward(q, x) for x in center_crop(X)]).astype(np.float64)
    return softmax(logits * scale)


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(C.DEPLOY_DIR, "cnn_meta.json")) as f:
        meta = json.load(f)
    params = dict(np.load(os.path.join(C.DEPLOY_DIR, "cnn_int8_params.npz")))

    d_cnn, d_mnv2 = load_cache(), load_cache(C.MNV2_CACHE_NPZ)
    model = MobileNetV2Head(pretrained=False)
    model.load_state_dict(torch.load(os.path.join(C.MODEL_DIR, "mnv2_best.pt"),
                                     map_location="cpu")["state_dict"])

    probs = {}
    for s in ("val", "test"):
        print(f"scoring {s} set ...")
        probs[s] = (cnn_int8_proba(d_cnn[s][0], params, meta["logit_scale"]),
                    predict_proba(model, d_mnv2[s][0]), d_cnn[s][1])

    pc, pm, yv = probs["val"]
    best_w, best_key = 0.5, None
    for w in np.linspace(0, 1, 21):
        p = w * pc + (1 - w) * pm
        f1 = metrics(yv, p.argmax(1))["macro_f1"]
        nll = -np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-9))   # tie-break: better confidence
        key = (round(f1, 4), -nll)
        if best_key is None or key > best_key:
            best_w, best_key = float(w), key
    print(f"best CNN weight on validation: w = {best_w:.2f}  (val macro-F1 {best_key[0]:.3f})")

    pc, pm, yt = probs["test"]
    res = {}
    for name, p in (("GreenCNN_int8_FPGA", pc), ("MobileNetV2_ARM", pm),
                    ("Fused_CNN+MobileNetV2", best_w * pc + (1 - best_w) * pm)):
        m, _ = save_report(name, yt, p.argmax(1), OUT)
        res[name] = m
        print(f"TEST {name:22s} acc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}")

    fusion = {"w_cnn": best_w, "classes": C.CLASS_NAMES, "test": res,
              "confidence_threshold": 0.60}
    with open(os.path.join(C.DEPLOY_DIR, "fusion.json"), "w") as f:
        json.dump(fusion, f, indent=2)
    print(f"-> {os.path.join(C.DEPLOY_DIR, 'fusion.json')}")
    print("STEP 8 OK")


if __name__ == "__main__":
    main()
