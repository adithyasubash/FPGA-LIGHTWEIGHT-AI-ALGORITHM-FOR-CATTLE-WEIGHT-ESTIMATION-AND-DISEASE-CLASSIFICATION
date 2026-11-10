"""
STEP 4 -- Deep Learning: train GreenCNN from scratch (random weights, no pretrained model).

  * augmentation: random crop 64 of 72, horizontal flip, small rotation,
    brightness/contrast jitter  (pure PyTorch, no torchvision needed)
  * class-weighted loss (FMD has fewer images than Healthy/Lumpy)
  * AdamW + OneCycle learning rate, early stopping on validation macro-F1
  * saves models/dl_best.pt, outputs/dl/training_curves.png, confusion matrix

Run:  python step4_train_dl.py            (CPU, roughly 15-40 min for 5,200 images)
      python step4_train_dl.py --epochs 5 (quick smoke test)
"""
import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from cnn_model import GreenCNN, count_params
from common import center_crop, load_cache, metrics, save_report

OUT = os.path.join(C.OUT_DIR, "dl")


def to_tensor(X):
    """uint8 NHWC -> float NCHW in [0,1]"""
    return torch.from_numpy(X).permute(0, 3, 1, 2).float().div_(255.0)


def augment(xb, s=C.IMG_SIZE):
    """xb: uint8 N x 72 x 72 x 3 -> float N x 3 x s x s in [0,1] (random crop/flip/rotate/colour)."""
    n = xb.shape[0]
    out = np.empty((n, s, s, 3), dtype=np.uint8)
    for i in range(n):
        oy, ox = np.random.randint(0, xb.shape[1] - s + 1, size=2)
        img = xb[i, oy:oy + s, ox:ox + s]
        if np.random.rand() < 0.5:
            img = img[:, ::-1]
        out[i] = img
    x = to_tensor(out)
    # small random rotation (+-15 deg) with affine grid
    ang = (torch.rand(n) - 0.5) * (math.pi / 6)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)
    # brightness / contrast jitter
    b = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.4
    c = 1 + (torch.rand(n, 1, 1, 1) - 0.5) * 0.4
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * c + mean * b).clamp_(0, 1)


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        preds.append(model(to_tensor(center_crop(X[i:i + bs]))).argmax(1).numpy())
    return np.concatenate(preds) if preds else np.array([], dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.set_num_threads(C.NUM_THREADS)

    data = load_cache()
    Xtr, ytr = data["train"]
    Xva, yva = data["val"]
    Xte, yte = data["test"]
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")

    counts = np.bincount(ytr, minlength=C.NUM_CLASSES).astype(np.float32)
    cls_w = torch.tensor(counts.sum() / (C.NUM_CLASSES * np.maximum(counts, 1)))
    print("class weights:", dict(zip(C.CLASS_NAMES, cls_w.numpy().round(2))))

    model = GreenCNN()
    print(f"GreenCNN parameters: {count_params(model):,}")
    opt = torch.optim.AdamW(model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY)
    steps = math.ceil(len(Xtr) / C.BATCH_SIZE)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=C.LEARNING_RATE,
                                                epochs=args.epochs, steps_per_epoch=steps)
    loss_fn = nn.CrossEntropyLoss(weight=cls_w, label_smoothing=0.05)

    hist = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": []}
    best_f1, best_ep, wait = -1, 0, 0
    best_path = os.path.join(C.MODEL_DIR, "dl_best.pt")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        perm = np.random.permutation(len(Xtr))
        tot, correct, n = 0.0, 0, 0
        for i in range(0, len(perm), C.BATCH_SIZE):
            idx = perm[i:i + C.BATCH_SIZE]
            xb = augment(Xtr[idx])
            yb = torch.from_numpy(ytr[idx]).long()
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(idx)

        vm = metrics(yva, predict(model, Xva))
        hist["train_loss"].append(tot / n)
        hist["train_acc"].append(correct / n)
        hist["val_acc"].append(vm["accuracy"])
        hist["val_f1"].append(vm["macro_f1"])
        flag = ""
        if vm["macro_f1"] > best_f1:
            best_f1, best_ep, wait = vm["macro_f1"], ep, 0
            torch.save({"state_dict": model.state_dict(), "widths": C.CNN_WIDTHS,
                        "classes": C.CLASS_NAMES, "img_size": C.IMG_SIZE}, best_path)
            flag = "  * saved"
        else:
            wait += 1
        print(f"epoch {ep:3d}/{args.epochs}  loss {tot / n:.3f}  train-acc {correct / n:.3f}  "
              f"val-acc {vm['accuracy']:.3f}  val-F1 {vm['macro_f1']:.3f}  "
              f"({time.time() - t0:.0f} s){flag}")
        if wait >= C.PATIENCE:
            print(f"early stop (no improvement for {C.PATIENCE} epochs)")
            break

    # curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(hist["train_loss"])
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("epoch")
    ax[1].plot(hist["train_acc"], label="train acc")
    ax[1].plot(hist["val_acc"], label="val acc")
    ax[1].plot(hist["val_f1"], label="val macro-F1")
    ax[1].axvline(best_ep - 1, ls="--", c="gray", label=f"best epoch {best_ep}")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=120)
    with open(os.path.join(OUT, "history.json"), "w") as f:
        json.dump(hist, f)

    # final test with best weights
    model.load_state_dict(torch.load(best_path, map_location="cpu")["state_dict"])
    tm, rep = save_report("DL_GreenCNN", yte, predict(model, Xte), OUT)
    print(f"\nBest epoch {best_ep}  (val macro-F1 {best_f1:.3f})")
    print(rep)
    print(f"TEST  acc {tm['accuracy']:.3f}  macro-F1 {tm['macro_f1']:.3f}")
    print(f"model -> {best_path}")
    print("STEP 4 OK")


if __name__ == "__main__":
    main()
