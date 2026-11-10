"""
STEP 1 -- Check the dataset.
  * reads labels.csv, checks every image exists and opens
  * prints images per class
  * writes outputs/bad_images.csv (images that are missing/corrupt; they are skipped later)
  * saves outputs/sample_grid.png so you can eyeball the classes

Run:  python step1_check_dataset.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

import config as C


def main():
    print(f"Dataset folder : {C.DATASET_DIR}")
    if not os.path.isfile(C.LABELS_CSV):
        sys.exit(f"ERROR: {C.LABELS_CSV} not found. Set GV_DATASET or edit config.py.")

    df = pd.read_csv(C.LABELS_CSV)
    need = {"filename", "disease_label"}
    if not need.issubset(df.columns):
        sys.exit(f"ERROR: labels.csv needs columns {need}, found {list(df.columns)}")
    df = df[["filename", "disease_label"]].dropna()
    df["disease_label"] = df["disease_label"].astype(int)

    bad = []
    for i, row in df.iterrows():
        path = os.path.join(C.IMAGES_DIR, row["filename"])
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                im.convert("RGB").load()
        except Exception as e:  # missing, truncated, not really an image ...
            bad.append({"filename": row["filename"], "error": str(e)[:120]})
        if (i + 1) % 500 == 0:
            print(f"  checked {i + 1}/{len(df)}")

    pd.DataFrame(bad, columns=["filename", "error"]).to_csv(
        os.path.join(C.OUT_DIR, "bad_images.csv"), index=False)
    good = df[~df["filename"].isin({b["filename"] for b in bad})]

    print("\nImages per class (usable):")
    for k, name in enumerate(C.CLASS_NAMES):
        n = int((good["disease_label"] == k).sum())
        print(f"  {k} {name:15s} {n:5d}  {'#' * (n // 50)}")
    print(f"  total             {len(good):5d}   (bad/missing: {len(bad)})")

    # sample grid: 6 images per class
    fig, axes = plt.subplots(C.NUM_CLASSES, 6, figsize=(12, 2.2 * C.NUM_CLASSES))
    for k, name in enumerate(C.CLASS_NAMES):
        sub = good[good["disease_label"] == k].sample(
            n=min(6, int((good["disease_label"] == k).sum())), random_state=C.SEED)
        for j in range(6):
            ax = axes[k, j]
            ax.axis("off")
            if j < len(sub):
                ax.imshow(Image.open(os.path.join(C.IMAGES_DIR, sub.iloc[j]["filename"])).convert("RGB"))
                if j == 0:
                    ax.set_title(name, loc="left", fontsize=11)
    plt.tight_layout()
    out = os.path.join(C.OUT_DIR, "sample_grid.png")
    plt.savefig(out, dpi=110)
    print(f"\nSample grid -> {out}")
    print("STEP 1 OK")


if __name__ == "__main__":
    main()
