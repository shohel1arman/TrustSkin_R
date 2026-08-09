"""Phase 2 calibration figures."""
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
plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})

def reliability_diagrams(cal_dir, out_dir, runs):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, run in zip(axes.flat, runs):
        pre_f = cal_dir / "reliability" / f"{run}_test_pre.csv"
        post_f = cal_dir / "reliability" / f"{run}_test_post.csv"
        if not pre_f.exists():
            ax.set_visible(False); continue
        pre = pd.read_csv(pre_f).dropna(subset=["avg_conf"])
        post = pd.read_csv(post_f).dropna(subset=["avg_conf"])
        ax.plot([0,1],[0,1],"--",color="gray",lw=1,label="perfect",zorder=1)
        ax.bar(pre["avg_conf"], pre["accuracy"], width=0.05, alpha=0.5,
               color="#c0392b", label="uncalibrated", zorder=2, edgecolor="none")
        ax.plot(post["avg_conf"], post["accuracy"], "-o", color="#2471a3",
                ms=4, lw=1.5, label="temp-scaled", zorder=3)
        ax.set_title(RUN_LABELS.get(run, run)); ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy"); ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_aspect("equal"); ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("Reliability diagrams (test set)", y=0.995, fontsize=13)
    fig.tight_layout(); p = out_dir / "reliability_diagrams.png"
    fig.savefig(p); plt.close(fig); return p

def calibration_comparison(cal_dir, out_dir):
    df = pd.read_csv(cal_dir / "calibration_summary.csv")
    df["label"] = df["run"].map(lambda r: RUN_LABELS.get(r, r))
    metrics = [("ece","ece_ts","ECE"),("brier","brier_ts","Brier"),("nll","nll_ts","NLL")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    x = np.arange(len(df)); w = 0.38
    for ax, (pre_c, post_c, name) in zip(axes, metrics):
        ax.bar(x-w/2, df[pre_c], w, label="uncalibrated", color="#c0392b", alpha=0.85)
        ax.bar(x+w/2, df[post_c], w, label="temp-scaled", color="#2471a3", alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=30, ha="right", fontsize=8)
        ax.set_title(name); ax.set_ylabel(name)
        if name == "ECE": ax.legend(fontsize=8)
    fig.suptitle("Calibration before vs after temperature scaling (test set)", fontsize=13)
    fig.tight_layout(); p = out_dir / "calibration_comparison.png"
    fig.savefig(p); plt.close(fig); return p

def accuracy_vs_calibration(cal_dir, out_dir):
    df = pd.read_csv(cal_dir / "calibration_summary.csv")
    df["label"] = df["run"].map(lambda r: RUN_LABELS.get(r, r))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["accuracy"], df["ece"], s=90, color="#c0392b", label="uncalibrated", zorder=3)
    ax.scatter(df["accuracy"], df["ece_ts"], s=90, color="#2471a3", marker="^", label="temp-scaled", zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["label"], (r["accuracy"], r["ece"]), textcoords="offset points",
                    xytext=(6,4), fontsize=8)
    ax.set_xlabel("test accuracy"); ax.set_ylabel("ECE (lower = better)")
    ax.set_title("Does accuracy predict calibration?"); ax.legend()
    fig.tight_layout(); p = out_dir / "accuracy_vs_calibration.png"
    fig.savefig(p); plt.close(fig); return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cal-dir", default="results/calibration")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--runs", nargs="+", default=list(RUN_LABELS.keys()))
    args = ap.parse_args()
    cal_dir = Path(args.cal_dir); out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in [reliability_diagrams(cal_dir, out_dir, args.runs),
              calibration_comparison(cal_dir, out_dir),
              accuracy_vs_calibration(cal_dir, out_dir)]:
        print("wrote", f)

if __name__ == "__main__":
    main()
