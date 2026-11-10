"""common.py -- shared helpers (data loading, metrics, plots)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score)

import config as C


def load_cache(path=C.CACHE_NPZ):
    if not os.path.isfile(path):
        raise SystemExit(f"ERROR: {path} not found -- run step2_prepare_data.py first.")
    d = np.load(path)
    out = {}
    for s in ("train", "val", "test"):
        m = d["split"] == s
        out[s] = (d["X"][m], d["y"][m])
    return out


def center_crop(X, size=C.IMG_SIZE):
    o = (X.shape[1] - size) // 2
    return X[:, o:o + size, o:o + size, :]


def metrics(y_true, y_pred):
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def save_report(name, y_true, y_pred, folder):
    """Confusion matrix PNG + text report + JSON metrics."""
    os.makedirs(folder, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=range(C.NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(C.NUM_CLASSES), C.CLASS_NAMES)
    ax.set_yticks(range(C.NUM_CLASSES), C.CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(C.NUM_CLASSES):
        for j in range(C.NUM_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    m = metrics(y_true, y_pred)
    ax.set_title(f"{name}\nacc {m['accuracy']:.3f}  macro-F1 {m['macro_f1']:.3f}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"cm_{name}.png"), dpi=120)
    plt.close(fig)

    rep = classification_report(y_true, y_pred, labels=range(C.NUM_CLASSES),
                                target_names=C.CLASS_NAMES, digits=3, zero_division=0)
    with open(os.path.join(folder, f"report_{name}.txt"), "w") as f:
        f.write(rep)
    with open(os.path.join(folder, f"metrics_{name}.json"), "w") as f:
        json.dump(m, f, indent=2)
    return m, rep
