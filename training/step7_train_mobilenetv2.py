"""
STEP 7 -- MobileNetV2 + custom CNN head (transfer learning), for the PYNQ ARM CPU.

Model:  MobileNetV2 backbone (ImageNet weights)
          -> custom head: GAP -> Dropout 0.3 -> Dense 128 + ReLU -> Dropout 0.2 -> Dense NUM_CLASSES
Input:  112 x 112 RGB (fast enough on the PYNQ-Z2 Cortex-A9 with OpenCV DNN)
Train:  phase 1 -- backbone frozen, only the head learns        (MNV2_EPOCHS_HEAD)
        phase 2 -- unfreeze top blocks, fine-tune at low LR     (MNV2_EPOCHS_FINE)
Output: models/mnv2_best.pt
        deploy/mobilenetv2.onnx   <- runs on the board with cv2.dnn
        outputs/mnv2/ (curves, confusion matrix)

Run:  python step7_train_mobilenetv2.py
      python step7_train_mobilenetv2.py --no-pretrained   (if the ImageNet download is blocked)
CPU time: roughly 40-90 min for 5,200 images. Needs torchvision.
"""
import argparse
import json
import os
import time
import warnings

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

import config as C
from common import center_crop, load_cache, metrics, save_report
from step4_train_dl import augment, to_tensor

OUT = os.path.join(C.OUT_DIR, "mnv2")
MEAN = torch.tensor(C.IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(C.IMAGENET_STD).view(1, 3, 1, 1)


class MobileNetV2Head(nn.Module):
    def __init__(self, pretrained=True, num_classes=C.NUM_CLASSES):
        super().__init__()
        w = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v2(weights=w).features          # -> N x 1280 x 4 x 4 at 112 px
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 128), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(128, num_classes))

    def forward(self, x):                      # x: N x 3 x 112 x 112, ImageNet-normalised
        return self.head(self.pool(self.backbone(x)).flatten(1))


def norm(x):
    return (x - MEAN) / STD


def set_trainable(model, phase):
    for p in model.backbone.parameters():
        p.requires_grad = False
    if phase == 2:
        for blk in model.backbone[C.MNV2_UNFREEZE_FROM:]:
            for p in blk.parameters():
                p.requires_grad = True


@torch.no_grad()
def predict_proba(model, X, bs=64):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = norm(to_tensor(center_crop(X[i:i + bs], C.MNV2_IMG_SIZE)))
        out.append(torch.softmax(model(x), 1).numpy())
    return np.concatenate(out) if out else np.zeros((0, C.NUM_CLASSES))


def run_phase(model, phase, epochs, lr, data, cls_w, hist, best, best_path):
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    set_trainable(model, phase)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"\n--- phase {phase}: {sum(p.numel() for p in params):,} trainable params, lr {lr}")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        if phase == 1:
            model.backbone.eval()              # keep frozen BatchNorm statistics
        perm = np.random.permutation(len(Xtr))
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            xb = norm(augment(Xtr[idx], C.MNV2_IMG_SIZE))
            yb = torch.from_numpy(ytr[idx]).long()
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(idx)
        sched.step()
        vm = metrics(yva, predict_proba(model, Xva).argmax(1))
        hist["train_acc"].append(correct / n)
        hist["val_acc"].append(vm["accuracy"])
        hist["val_f1"].append(vm["macro_f1"])
        flag = ""
        if vm["macro_f1"] > best["f1"]:
            best.update(f1=vm["macro_f1"], epoch=len(hist["val_f1"]))
            torch.save({"state_dict": model.state_dict(), "classes": C.CLASS_NAMES,
                        "img_size": C.MNV2_IMG_SIZE}, best_path)
            flag = "  * saved"
        print(f"P{phase} epoch {ep:2d}/{epochs}  loss {tot / n:.3f}  train-acc {correct / n:.3f}  "
              f"val-acc {vm['accuracy']:.3f}  val-F1 {vm['macro_f1']:.3f}  ({time.time() - t0:.0f} s){flag}")


def export_onnx(model, path):
    """Export with the ImageNet normalisation left OUTSIDE (done in the PYNQ app, same as here)."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    model.eval()
    dummy = torch.zeros(1, 3, C.MNV2_IMG_SIZE, C.MNV2_IMG_SIZE)
    try:
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                          opset_version=11, dynamo=False)
    except TypeError:                      # older torch without the dynamo argument
        torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["logits"],
                          opset_version=11)
    # verify OpenCV DNN (the engine used on the PYNQ board) gives the same answer
    net = cv2.dnn.readNetFromONNX(path)
    x = torch.rand(2, 3, C.MNV2_IMG_SIZE, C.MNV2_IMG_SIZE)
    xn = norm(x)
    with torch.no_grad():
        ref = model(xn).numpy()
    worst = 0.0
    for k in range(2):
        net.setInput(xn[k:k + 1].numpy().astype(np.float32))
        worst = max(worst, float(np.abs(net.forward() - ref[k:k + 1]).max()))
    print(f"ONNX -> {path}  ({os.path.getsize(path) / 1e6:.1f} MB), OpenCV-DNN vs PyTorch max diff {worst:.2e}")
    if worst > 1e-2:
        print("WARNING: OpenCV DNN output differs from PyTorch -- tell me this number.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs-head", type=int, default=C.MNV2_EPOCHS_HEAD)
    ap.add_argument("--epochs-fine", type=int, default=C.MNV2_EPOCHS_FINE)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.set_num_threads(C.NUM_THREADS)

    data = load_cache(C.MNV2_CACHE_NPZ)
    ytr = data["train"][1]
    print(f"train {len(ytr)}  val {len(data['val'][1])}  test {len(data['test'][1])}")
    counts = np.bincount(ytr, minlength=C.NUM_CLASSES).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / (C.NUM_CLASSES * np.maximum(counts, 1)))

    try:
        model = MobileNetV2Head(pretrained=not args.no_pretrained)
    except Exception as e:
        print(f"Could not download ImageNet weights ({e}); training from random weights instead.")
        model = MobileNetV2Head(pretrained=False)

    hist = {"train_acc": [], "val_acc": [], "val_f1": []}
    best = {"f1": -1.0, "epoch": 0}
    best_path = os.path.join(C.MODEL_DIR, "mnv2_best.pt")
    run_phase(model, 1, args.epochs_head, C.MNV2_LR_HEAD, data, cls_w, hist, best, best_path)
    run_phase(model, 2, args.epochs_fine, C.MNV2_LR_FINE, data, cls_w, hist, best, best_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["train_acc"], label="train acc")
    ax.plot(hist["val_acc"], label="val acc")
    ax.plot(hist["val_f1"], label="val macro-F1")
    ax.axvline(args.epochs_head - 0.5, ls="--", c="gray", label="fine-tune starts")
    ax.set_xlabel("epoch")
    ax.set_title("MobileNetV2 + custom head")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=120)

    model.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
    Xte, yte = data["test"]
    tm, rep = save_report("DL_MobileNetV2", yte, predict_proba(model, Xte).argmax(1), OUT)
    print(f"\nBest epoch {best['epoch']}  (val macro-F1 {best['f1']:.3f})")
    print(rep)
    print(f"TEST  acc {tm['accuracy']:.3f}  macro-F1 {tm['macro_f1']:.3f}")

    export_onnx(model, os.path.join(C.DEPLOY_DIR, "mobilenetv2.onnx"))
    with open(os.path.join(C.DEPLOY_DIR, "mnv2_meta.json"), "w") as f:
        json.dump({"classes": C.CLASS_NAMES, "img_size": C.MNV2_IMG_SIZE,
                   "mean": C.IMAGENET_MEAN, "std": C.IMAGENET_STD, "test": tm}, f, indent=2)
    print("STEP 7 OK")


if __name__ == "__main__":
    main()
