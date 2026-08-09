"""Phase 4 visualizations - localization figures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_LABELS = {
    "efficientnet_b3_seed42": "EfficientNet-B3",
    "convnext_tiny_seed42": "ConvNeXt-Tiny",
    "vit_b16_seed42": "ViT-B/16",
    "swin_tiny_seed42": "Swin-Tiny",
}
TEST_ACC = {
    "efficientnet_b3_seed42": 0.816,
    "convnext_tiny_seed42": 0.837,
    "vit_b16_seed42": 0.754,
    "swin_tiny_seed42": 0.814,
}

plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})


def load_summaries(xai_dir, method, runs):
    data = {}
    for run in runs:
        f = xai_dir / f"{run}_test_{method}_summary.json"
        if f.exists():
            data[run] = json.loads(f.read_text())
    return data


def localization_comparison(xai_dir, method, out_dir, runs):
    data = load_summaries(xai_dir, method, runs)
    if not data:
        return None
    order = sorted(data.keys(), key=lambda r: -data[r]["pointing_game"])
    labels = [RUN_LABELS.get(r, r) for r in order]
    metrics = [("pointing_game", "Pointing game"),
               ("mean_lesion_frac", "Lesion attention fraction"),
               ("mean_iou_p20", "IoU (top-20%)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(order))
    for ax, (key, name) in zip(axes, metrics):
        vals = [data[r].get(key, np.nan) for r in order]
        bars = ax.bar(x, vals, color="#2471a3", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(name)
        ax.set_ylim(0, max(vals) * 1.25 if vals else 1)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Lesion localization by model (Grad-CAM++, test subset)", fontsize=13)
    fig.tight_layout()
    p = out_dir / "localization_comparison.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def accuracy_vs_localization(xai_dir, method, out_dir, runs):
    data = load_summaries(xai_dir, method, runs)
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for run in data:
        acc = TEST_ACC.get(run, np.nan)
        pg = data[run]["pointing_game"]
        ax.scatter(acc, pg, s=140, color="#c0392b", zorder=3)
        ax.annotate(RUN_LABELS.get(run, run), (acc, pg),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("test accuracy")
    ax.set_ylabel("pointing game (lesion focus)")
    ax.set_title("Accuracy does not imply lesion focus")
    ax.axhline(0.5, ls=":", color="gray", lw=1)
    fig.tight_layout()
    p = out_dir / "accuracy_vs_localization.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xai-dir", default="results/xai")
    ap.add_argument("--method", default="gradcam++")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--runs", nargs="+", default=list(RUN_LABELS.keys()))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in (localization_comparison, accuracy_vs_localization):
        p = fn(Path(args.xai_dir), args.method, out_dir, args.runs)
        if p:
            made.append(p)
    print("Figures written:")
    for p in made:
        print(" ", p)


if __name__ == "__main__":
    main()
