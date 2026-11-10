"""
STEP 3 -- Classical ML (machine learning) from scratch on hand-made image features.

Features per 64x64 image (no neural network):
  * HSV colour histogram   3 x 16 bins   -> skin colour / redness of lesions
  * HOG (shape/edges)      1764 values    -> lumps, blisters, outlines
  * LBP texture histogram  10 bins x 2    -> skin roughness / nodules
Models (each tuned on the validation set):
  Logistic Regression, SVM (RBF), Random Forest, K-Nearest Neighbours
Best model by validation macro-F1 is saved to models/ml_best.joblib.

Run:  python step3_train_ml.py
"""
import json
import os
import time

import joblib
import numpy as np
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as C
from common import center_crop, load_cache, metrics, save_report

OUT = os.path.join(C.OUT_DIR, "ml")


def features_one(img):
    hsv = rgb2hsv(img)
    hist = [np.histogram(hsv[..., c], bins=16, range=(0, 1))[0] for c in range(3)]
    color = np.concatenate(hist).astype(np.float32) / (img.shape[0] * img.shape[1])

    gray = rgb2gray(img)
    h = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
            feature_vector=True).astype(np.float32)

    g8 = (gray * 255).astype(np.uint8)
    lbp = []
    for r in (1, 2):
        codes = local_binary_pattern(g8, P=8, R=r, method="uniform")
        lbp.append(np.histogram(codes, bins=10, range=(0, 10))[0])
    lbp = np.concatenate(lbp).astype(np.float32) / g8.size
    return np.concatenate([color, h, lbp])


def features(X):
    return np.stack([features_one(x) for x in X])


def candidates(n_train):
    p1, p2 = min(200, n_train - 1), min(100, n_train - 1)
    """(name, list of models to try). Each is a full pipeline."""
    return {
        "LogReg": [make_pipeline(StandardScaler(), LogisticRegression(
            C=c, max_iter=3000, class_weight="balanced")) for c in (0.01, 0.1, 1.0)],
        "SVM_RBF": [make_pipeline(StandardScaler(), PCA(n_components=p1, random_state=C.SEED),
                                  SVC(C=c, gamma="scale", class_weight="balanced"))
                    for c in (1, 5, 20)],
        "RandomForest": [RandomForestClassifier(n_estimators=n, max_depth=d,
                                                class_weight="balanced_subsample",
                                                n_jobs=-1, random_state=C.SEED)
                         for n, d in ((300, None), (300, 20))],
        "KNN": [make_pipeline(StandardScaler(), PCA(n_components=p2, random_state=C.SEED),
                              KNeighborsClassifier(n_neighbors=k, weights="distance"))
                for k in (3, 7, 15)],
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    data = load_cache()
    feats = {}
    for s in ("train", "val", "test"):
        t = time.time()
        X, y = data[s]
        feats[s] = (features(center_crop(X)), y)
        print(f"features {s:5s}: {feats[s][0].shape}  ({time.time() - t:.0f} s)")
    Xtr, ytr = feats["train"]
    Xva, yva = feats["val"]
    Xte, yte = feats["test"]

    summary = {}
    best = (None, -1, None)
    for name, models in candidates(len(Xtr)).items():
        best_here = (None, -1)
        for m in models:
            t = time.time()
            m.fit(Xtr, ytr)
            f1 = metrics(yva, m.predict(Xva))["macro_f1"]
            print(f"  {name:12s} {str(m.get_params().get('steps', [m])[-1])[:60]:60s} "
                  f"val-F1 {f1:.3f}  ({time.time() - t:.0f} s)")
            if f1 > best_here[1]:
                best_here = (m, f1)
        m, vf1 = best_here
        test_m, rep = save_report(f"ML_{name}", yte, m.predict(Xte), OUT)
        summary[name] = {"val_macro_f1": vf1, **{f"test_{k}": v for k, v in test_m.items()}}
        print(f"==> {name}: test acc {test_m['accuracy']:.3f}  macro-F1 {test_m['macro_f1']:.3f}\n")
        if vf1 > best[1]:
            best = (name, vf1, m)

    joblib.dump({"model": best[2], "name": best[0], "classes": C.CLASS_NAMES},
                os.path.join(C.MODEL_DIR, "ml_best.joblib"))
    with open(os.path.join(OUT, "ml_summary.json"), "w") as f:
        json.dump({"best": best[0], "models": summary}, f, indent=2)

    print("ML summary (test set):")
    for n, s in summary.items():
        print(f"  {n:12s} acc {s['test_accuracy']:.3f}  macro-F1 {s['test_macro_f1']:.3f}"
              f"{'   <- best (val)' if n == best[0] else ''}")
    print(f"\nBest ML model saved -> {os.path.join(C.MODEL_DIR, 'ml_best.joblib')}")
    print("STEP 3 OK")


if __name__ == "__main__":
    main()
