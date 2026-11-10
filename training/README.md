# GREEN_VLSI – training from scratch (ML + DL)

Dataset (from earlier chats): `C:\AI-Environment_OLD\dataset\images` + `labels.csv`
(filename, disease_label: 0 Healthy, 1 FMD, 2 Lumpy). Change it in `config.py` or `set GV_DATASET=...`.

## Setup (once)
```
cd C:\LivestockAI
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r training\requirements.txt
cd training
```

## Run (step by step)
| Step | Command | Output |
|---|---|---|
| 1 | `python step1_check_dataset.py` | class counts, outputs\sample_grid.png, bad_images.csv |
| 2 | `python step2_prepare_data.py` | 70/15/15 split + work\dataset_cache.npz |
| 3 | `python step3_train_ml.py` | LogReg / SVM / RandomForest / KNN, models\ml_best.joblib |
| 4 | `python step4_train_dl.py` | GreenCNN from scratch, models\dl_best.pt, curves |
| 6 | `python step6_export_hls.py` | int8 weights for HLS: hls_export\cnn_weights.h |
| 5 | `python step5_compare.py` | outputs\comparison.png (ML vs DL vs int8) |

Or all at once: `run_all.bat`.  Test any photo: `python predict.py "C:\path\cow.jpg"`.
