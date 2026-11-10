"""
STEP 0 -- Build ONE combined cow + goat dataset inside C:\\LivestockAI\\dataset

Sources (found automatically, from the earlier projects on this PC):
  COWS : C:\\AI-Environment_OLD\\dataset\\images + labels.csv   (0 healthy, 1 fmd, 2 lumpy)
  GOATS: C:\\Users\\VLSI\\Downloads\\archive (4)\\healthy_goat / unhealthy_goat
         (or C:\\AI-Environment_OLD\\goat_health_project\\raw_dataset\\Healthy / Unhealthy)

Result:
  C:\\LivestockAI\\dataset\\images\\      all images, renamed  cow_fmd_00001.jpg, goat_healthy_00001.jpg ...
  C:\\LivestockAI\\dataset\\labels.csv    filename, disease_label, species
  C:\\LivestockAI\\dataset\\classes.json  class names in label order (read by config.py)

Classes:  0 Cow_Healthy  1 Cow_FMD  2 Cow_Lumpy  3 Goat_Healthy  4 Goat_Unhealthy
(the goat source only says healthy / unhealthy, it does not name the goat disease)

Run once:  python step0_build_dataset.py
Different folders?  python step0_build_dataset.py --cow "D:\\x\\dataset" --goat-healthy "..." --goat-unhealthy "..."
Originals are only COPIED, never moved or changed.
"""
import argparse
import csv
import json
import os
import shutil
import sys

OUT_DIR = r"C:\LivestockAI\dataset"
COW_CANDIDATES = [r"C:\AI-Environment_OLD\dataset", r"C:\AI-Environment\dataset"]
GOAT_CANDIDATES = [
    (r"C:\Users\VLSI\Downloads\archive (4)\healthy_goat",
     r"C:\Users\VLSI\Downloads\archive (4)\unhealthy_goat"),
    (r"C:\AI-Environment_OLD\goat_health_project\raw_dataset\Healthy",
     r"C:\AI-Environment_OLD\goat_health_project\raw_dataset\Unhealthy"),
    (r"C:\AI-Environment\goat_health_project\raw_dataset\Healthy",
     r"C:\AI-Environment\goat_health_project\raw_dataset\Unhealthy"),
]
COW_CLASSES = ["Cow_Healthy", "Cow_FMD", "Cow_Lumpy"]       # old labels 0, 1, 2
GOAT_CLASSES = ["Goat_Healthy", "Goat_Unhealthy"]
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder):
    """All image files in a folder, including sub-folders."""
    out = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, f))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cow", help="folder with images\\ and labels.csv")
    ap.add_argument("--goat-healthy")
    ap.add_argument("--goat-unhealthy")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    cow_dir = args.cow or next((p for p in COW_CANDIDATES
                                if os.path.isfile(os.path.join(p, "labels.csv"))), None)
    if args.goat_healthy and args.goat_unhealthy:
        goat = (args.goat_healthy, args.goat_unhealthy)
    else:
        goat = next(((h, u) for h, u in GOAT_CANDIDATES
                     if os.path.isdir(h) and os.path.isdir(u)), None)

    print(f"Cow source : {cow_dir or 'NOT FOUND'}")
    print(f"Goat source: {goat[0] + '  +  ' + goat[1] if goat else 'NOT FOUND'}")
    if not cow_dir and not goat:
        sys.exit("ERROR: no source found. Pass --cow / --goat-healthy / --goat-unhealthy.")

    img_out = os.path.join(args.out, "images")
    if os.path.isdir(img_out) and os.listdir(img_out):
        ans = input(f"{img_out} already has files. Delete and rebuild? (y/n): ").strip().lower()
        if ans != "y":
            sys.exit("Stopped, nothing changed.")
        shutil.rmtree(img_out)
    os.makedirs(img_out, exist_ok=True)

    classes, rows, counts = [], [], {}

    def add(src, cls_name, species):
        if cls_name not in classes:
            classes.append(cls_name)
        counts[cls_name] = counts.get(cls_name, 0) + 1
        ext = os.path.splitext(src)[1].lower()
        fn = f"{cls_name.lower()}_{counts[cls_name]:05d}{ext}"
        shutil.copy2(src, os.path.join(img_out, fn))
        rows.append({"filename": fn, "disease_label": classes.index(cls_name), "species": species})

    if cow_dir:
        with open(os.path.join(cow_dir, "labels.csv"), newline="") as f:
            cow_rows = list(csv.DictReader(f))
        for c in COW_CLASSES:          # keep label order 0,1,2 even if a class is empty
            classes.append(c)
        for i, r in enumerate(cow_rows):
            src = os.path.join(cow_dir, "images", r["filename"])
            if os.path.isfile(src) and r.get("disease_label", "") != "":
                add(src, COW_CLASSES[int(float(r["disease_label"]))], "cow")
            if (i + 1) % 1000 == 0:
                print(f"  cows copied {i + 1}/{len(cow_rows)}")
    if goat:
        for folder, cls_name in zip(goat, GOAT_CLASSES):
            imgs = list_images(folder)
            for i, src in enumerate(imgs):
                add(src, cls_name, "goat")
            print(f"  {cls_name}: {len(imgs)} images")

    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "disease_label", "species"])
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "classes.json"), "w") as f:
        json.dump(classes, f, indent=2)

    print("\n" + "=" * 50)
    print("COMBINED DATASET READY")
    print("=" * 50)
    for k, c in enumerate(classes):
        print(f"  {k}  {c:15s} {counts.get(c, 0):5d}")
    print(f"  total             {len(rows):5d}")
    print(f"\n-> {args.out}\\images, labels.csv, classes.json")
    if not goat:
        print("NOTE: goat images not found -- cow-only dataset. Re-run with --goat-healthy/--goat-unhealthy.")
    print("STEP 0 OK")


if __name__ == "__main__":
    main()
