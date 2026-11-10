r"""
predict.py -- classify one image (or a folder) with the trained ML and DL models.

    python predict.py "C:\path\to\cow.jpg"
    python predict.py "C:\path\to\folder"
"""
import os
import sys

import joblib
import numpy as np
import torch

import config as C
from cnn_model import GreenCNN
from step2_prepare_data import load_square
from step3_train_ml import features_one


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    target = sys.argv[1]
    files = ([os.path.join(target, f) for f in sorted(os.listdir(target))
              if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
             if os.path.isdir(target) else [target])

    dl = ml = None
    p = os.path.join(C.MODEL_DIR, "dl_best.pt")
    if os.path.isfile(p):
        ck = torch.load(p, map_location="cpu")
        dl = GreenCNN(widths=tuple(ck["widths"]))
        dl.load_state_dict(ck["state_dict"])
        dl.eval()
    p = os.path.join(C.MODEL_DIR, "ml_best.joblib")
    if os.path.isfile(p):
        ml = joblib.load(p)
    if dl is None and ml is None:
        sys.exit("No trained models found -- run step3 / step4 first.")

    for f in files:
        try:
            img = load_square(f, C.IMG_SIZE)
        except Exception as e:
            print(f"{os.path.basename(f)}: cannot open ({e})")
            continue
        line = os.path.basename(f)[:40].ljust(42)
        if dl is not None:
            with torch.no_grad():
                x = torch.from_numpy(img.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)
                prob = torch.softmax(dl(x), 1)[0].numpy()
            k = int(prob.argmax())
            line += f"DL: {C.CLASS_NAMES[k]:8s} {prob[k] * 100:5.1f}%   "
        if ml is not None:
            k = int(ml["model"].predict(features_one(img)[None])[0])
            line += f"ML({ml['name']}): {C.CLASS_NAMES[k]}"
        print(line)


if __name__ == "__main__":
    main()
