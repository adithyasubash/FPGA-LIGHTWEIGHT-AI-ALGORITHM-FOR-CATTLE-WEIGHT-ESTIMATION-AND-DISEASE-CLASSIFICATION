"""
STEP 5 -- Compare ML vs DL on the same test set.
Writes outputs/comparison.png and outputs/comparison.csv (for your report/slides).

Run:  python step5_compare.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C


def main():
    rows = []
    ml_json = os.path.join(C.OUT_DIR, "ml", "ml_summary.json")
    if os.path.isfile(ml_json):
        with open(ml_json) as f:
            for name, s in json.load(f)["models"].items():
                rows.append({"model": f"ML: {name}", "type": "ML",
                             "accuracy": s["test_accuracy"], "macro_f1": s["test_macro_f1"]})
    dl_json = os.path.join(C.OUT_DIR, "dl", "metrics_DL_GreenCNN.json")
    if os.path.isfile(dl_json):
        with open(dl_json) as f:
            s = json.load(f)
        rows.append({"model": "DL: GreenCNN", "type": "DL", **s})
    mn_json = os.path.join(C.OUT_DIR, "mnv2", "metrics_DL_MobileNetV2.json")
    if os.path.isfile(mn_json):
        with open(mn_json) as f:
            rows.append({"model": "DL: MobileNetV2 (ARM)", "type": "DL", **json.load(f)})
    fu_json = os.path.join(C.DEPLOY_DIR, "fusion.json")
    if os.path.isfile(fu_json):
        with open(fu_json) as f:
            s = json.load(f)["test"]["Fused_CNN+MobileNetV2"]
        rows.append({"model": "DL: Fused CNN+MobileNetV2", "type": "DL", **s})
    q_json = os.path.join(C.HLS_EXPORT_DIR, "int8_metrics.json")
    if os.path.isfile(q_json):
        with open(q_json) as f:
            s = json.load(f)["int8"]
        rows.append({"model": "DL: GreenCNN int8 (FPGA)", "type": "DL", **s})
    if not rows:
        raise SystemExit("Nothing to compare yet -- run step3 and step4 first.")

    df = pd.DataFrame(rows).sort_values("macro_f1")
    df.to_csv(os.path.join(C.OUT_DIR, "comparison.csv"), index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(df) + 1.2))
    y = range(len(df))
    colors = ["#4C78A8" if t == "ML" else "#F58518" for t in df["type"]]
    ax.barh(y, df["macro_f1"], color=colors)
    for i, (f1, acc) in enumerate(zip(df["macro_f1"], df["accuracy"])):
        ax.text(f1 + 0.005, i, f"F1 {f1:.3f}  acc {acc:.3f}", va="center", fontsize=9)
    ax.set_yticks(list(y), df["model"])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Test macro-F1 (blue = ML, orange = DL)")
    ax.set_title("Cattle disease classification: ML vs DL (same test set)")
    plt.tight_layout()
    out = os.path.join(C.OUT_DIR, "comparison.png")
    plt.savefig(out, dpi=130)
    print(f"\nchart -> {out}")
    print("STEP 5 OK")


if __name__ == "__main__":
    main()
