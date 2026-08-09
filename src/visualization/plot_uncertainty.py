"""Phase 3 visualizations - uncertainty figures."""
import argparse
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
SCORE_LABELS = {
    "neg_confidence": "confidence",
    "predictive_entropy": "pred. entropy",
    "mutual_information": "mutual info",
    "variance": "variance",
}
COLORS = {"neg_confidence": "#7f8c8d", "predictive_entropy": "#2471a3",
          "mutual_information": "#c0392b", "variance": "#27ae60"}

plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})


def error_detection_bars(unc_dir, out_dir, runs, split="test"):
    scores = ["neg_confidence", "predictive_entropy", "mutual_information", "variance"]
    data = {}
    for run in runs:
        f = unc_dir / f"{run}_{split}_uncertainty.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).set_index("score")
        data[run] = {s: df.loc[s, "err_auroc"] for s in scores if s in df.index}
    if not data:
        return None
    runs_present = list(data.keys())
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(runs_present))
    w = 0.2
    for i, s in enumerate(scores):
        vals = [data[r].get(s, np.nan) for r in runs_present]
        ax.bar(x + (i - 1.5) * w, vals, w, label=SCORE_LABELS[s], color=COLORS[s], alpha=0.9)
    ax.axhline(0.5, ls="--", color="gray", lw=1, label="random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS.get(r, r) for r in runs_present], fontsize=9)
    ax.set_ylabel("error-detection AUROC")
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Can uncertainty detect errors? (test set, higher = better)")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    fig.tight_layout()
    p = out_dir / "error_detection.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def risk_coverage_curves(unc_dir, out_dir, runs, split="test", score="predictive_entropy"):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, run in zip(axes.flat, runs):
        f = unc_dir / "risk_coverage" / f"{run}_{split}_{score}.csv"
        base_f = unc_dir / "risk_coverage" / f"{run}_{split}_neg_confidence.csv"
        if not f.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(f)
        ax.plot(df["coverage"], df["risk"], "-", color="#2471a3", lw=1.8, label="pred. entropy")
        if base_f.exists():
            b = pd.read_csv(base_f)
            ax.plot(b["coverage"], b["risk"], "--", color="#7f8c8d", lw=1.5, label="confidence")
        ax.set_title(RUN_LABELS.get(run, run))
        ax.set_xlabel("coverage (fraction predicted)")
        ax.set_ylabel("risk (error rate)")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8, loc="upper left")
    fig.suptitle("Risk-coverage curves: abstaining on uncertain cases lowers error",
                 y=0.995, fontsize=12)
    fig.tight_layout()
    p = out_dir / "risk_coverage.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unc-dir", default="results/uncertainty")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--split", default="test")
    ap.add_argument("--runs", nargs="+", default=list(RUN_LABELS.keys()))
    args = ap.parse_args()
    unc_dir = Path(args.unc_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    p1 = error_detection_bars(unc_dir, out_dir, args.runs, args.split)
    if p1:
        made.append(p1)
    p2 = risk_coverage_curves(unc_dir, out_dir, args.runs, args.split)
    if p2:
        made.append(p2)
    print("Figures written:")
    for p in made:
        print(" ", p)


if __name__ == "__main__":
    main()
