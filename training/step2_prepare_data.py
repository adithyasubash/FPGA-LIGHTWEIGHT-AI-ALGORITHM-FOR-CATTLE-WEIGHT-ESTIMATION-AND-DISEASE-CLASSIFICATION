"""
STEP 2 -- Split + cache.
  * stratified split 70 / 15 / 15  (same split is used by ML and DL -> fair comparison)
  * every image: center-crop to square, resize to 72x72 (GreenCNN/ML) and 128x128 (MobileNetV2)
  * writes work/split.csv, work/dataset_cache.npz and work/dataset_cache_mnv2.npz

Run:  python step2_prepare_data.py
"""
import os

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import config as C


def square_pil(path):
    """Open an image and center-crop it to a square (same as the PYNQ camera app does)."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = min(w, h)
        return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


def load_square(path, size):
    return np.asarray(square_pil(path).resize((size, size), Image.BILINEAR), dtype=np.uint8)


def main():
    df = pd.read_csv(C.LABELS_CSV)[["filename", "disease_label"]].dropna()
    df["disease_label"] = df["disease_label"].astype(int)
    bad_csv = os.path.join(C.OUT_DIR, "bad_images.csv")
    if os.path.isfile(bad_csv):
        bad = set(pd.read_csv(bad_csv)["filename"])
        df = df[~df["filename"].isin(bad)]
    df = df.reset_index(drop=True)

    tr, rest = train_test_split(df, test_size=C.VAL_RATIO + C.TEST_RATIO,
                                stratify=df["disease_label"], random_state=C.SEED)
    va, te = train_test_split(rest, test_size=C.TEST_RATIO / (C.VAL_RATIO + C.TEST_RATIO),
                              stratify=rest["disease_label"], random_state=C.SEED)
    df["split"] = ""
    df.loc[tr.index, "split"] = "train"
    df.loc[va.index, "split"] = "val"
    df.loc[te.index, "split"] = "test"
    df.to_csv(C.SPLIT_CSV, index=False)

    print("Split (rows = class, cols = split):")
    print(pd.crosstab(df["disease_label"].map(dict(enumerate(C.CLASS_NAMES))), df["split"]))

    # two caches from one pass: 72x72 for GreenCNN/ML, 128x128 for MobileNetV2
    X = np.zeros((len(df), C.CACHE_SIZE, C.CACHE_SIZE, 3), dtype=np.uint8)
    X2 = np.zeros((len(df), C.MNV2_CACHE_SIZE, C.MNV2_CACHE_SIZE, 3), dtype=np.uint8)
    keep = np.ones(len(df), dtype=bool)
    for i, fn in enumerate(df["filename"]):
        try:
            sq = square_pil(os.path.join(C.IMAGES_DIR, fn))
            X[i] = np.asarray(sq.resize((C.CACHE_SIZE,) * 2, Image.BILINEAR), dtype=np.uint8)
            X2[i] = np.asarray(sq.resize((C.MNV2_CACHE_SIZE,) * 2, Image.BILINEAR), dtype=np.uint8)
        except Exception as e:
            keep[i] = False
            print(f"  skip {fn}: {e}")
        if (i + 1) % 500 == 0:
            print(f"  cached {i + 1}/{len(df)}")

    meta = dict(y=np.asarray(df["disease_label"], dtype=np.int64)[keep],
                split=np.array(df["split"].tolist(), dtype="U5")[keep],
                filename=np.array(df["filename"].tolist(), dtype=str)[keep])
    np.savez_compressed(C.CACHE_NPZ, X=X[keep], **meta)
    np.savez_compressed(C.MNV2_CACHE_NPZ, X=X2[keep], **meta)
    print(f"\nCache -> {C.CACHE_NPZ}  ({keep.sum()} images, {X[keep].nbytes / 1e6:.0f} MB raw)")
    print(f"Cache -> {C.MNV2_CACHE_NPZ}  ({X2[keep].nbytes / 1e6:.0f} MB raw)")
    print("STEP 2 OK")


if __name__ == "__main__":
    main()
