"""
config.py -- every path and setting for the GREEN_VLSI training pipeline.
Edit only this file if your folders are somewhere else.
"""
import os

# --------------------------------------------------------------------------- paths
# Combined cow + goat dataset made by step0_build_dataset.py (C:\LivestockAI\dataset).
# The first folder that has a labels.csv is used. Override with:  set GV_DATASET=D:\my\dataset
_CANDIDATES = [
    os.environ.get("GV_DATASET", ""),
    r"C:\LivestockAI\dataset",
    r"C:\AI-Environment_OLD\dataset",
    r"C:\AI-Environment\dataset",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"),
]
DATASET_DIR = next((p for p in _CANDIDATES
                    if p and os.path.isfile(os.path.join(p, "labels.csv"))), _CANDIDATES[1])
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_CSV = os.path.join(DATASET_DIR, "labels.csv")

ROOT = os.path.dirname(os.path.abspath(__file__))
WORK_DIR = os.path.join(ROOT, "work")          # cached arrays + split
OUT_DIR = os.path.join(ROOT, "outputs")        # plots, reports
MODEL_DIR = os.path.join(ROOT, "models")       # trained models
HLS_EXPORT_DIR = os.path.join(ROOT, "hls_export")
for _d in (WORK_DIR, OUT_DIR, MODEL_DIR, HLS_EXPORT_DIR):
    os.makedirs(_d, exist_ok=True)

CACHE_NPZ = os.path.join(WORK_DIR, "dataset_cache.npz")
SPLIT_CSV = os.path.join(WORK_DIR, "split.csv")

# --------------------------------------------------------------------------- classes
# Read from dataset\classes.json (written by step0): label number -> name, e.g.
#   0 Cow_Healthy  1 Cow_FMD  2 Cow_Lumpy  3 Goat_Healthy  4 Goat_Unhealthy
# Without classes.json the old cow-only names are used.
# Everything (ML, CNN, MobileNetV2, HLS, PYNQ app) follows this list automatically.
_CLASSES_JSON = os.path.join(DATASET_DIR, "classes.json")
if os.path.isfile(_CLASSES_JSON):
    import json as _json
    with open(_CLASSES_JSON) as _f:
        CLASS_NAMES = _json.load(_f)
else:
    CLASS_NAMES = ["Healthy", "FMD", "Lumpy"]
NUM_CLASSES = len(CLASS_NAMES)

# --------------------------------------------------------------------------- images
CACHE_SIZE = 72      # images stored at 72x72 ...
IMG_SIZE = 64        # ... and the model sees 64x64 (random crop in training)
SEED = 42
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15

# --------------------------------------------------------------------------- DL training
BATCH_SIZE = 32
EPOCHS = 120
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 25                    # early stopping on val macro-F1
CNN_WIDTHS = (24, 48, 64, 96)    # channels of the 4 conv blocks (each a multiple of 8 for HLS)
NUM_THREADS = max(1, (os.cpu_count() or 2) - 1)

# --------------------------------------------------------------------------- MobileNetV2 (runs on PYNQ ARM CPU)
MNV2_CACHE_SIZE = 128            # second cache for MobileNetV2 ...
MNV2_IMG_SIZE = 112              # ... model sees 112x112 (fast enough on the Cortex-A9)
MNV2_CACHE_NPZ = os.path.join(WORK_DIR, "dataset_cache_mnv2.npz")
MNV2_EPOCHS_HEAD = 8             # phase 1: backbone frozen, train custom head
MNV2_EPOCHS_FINE = 12            # phase 2: unfreeze top blocks, fine-tune
MNV2_LR_HEAD = 1e-3
MNV2_LR_FINE = 1e-4
MNV2_UNFREEZE_FROM = 14          # unfreeze MobileNetV2 feature blocks 14..18
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --------------------------------------------------------------------------- deployment files (copied to the PYNQ board)
DEPLOY_DIR = os.path.join(ROOT, "deploy")
os.makedirs(DEPLOY_DIR, exist_ok=True)
