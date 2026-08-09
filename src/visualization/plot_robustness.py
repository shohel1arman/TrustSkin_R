"""Phase 5 visualizations - robustness & explanation stability."""
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
RUNS = list(RUN_LABELS.keys())
FAM_COLORS = {
    "gaussian_noise": "#c0392b", "gaussian_blur": "#e67e22",
    "brightness": "#f1c40f", "contrast": "#27ae60",
    "jpeg": "#2471a3", "occlusion": "#8e44ad", "rotation": "#7f8c8d",
}

plt.rcParams.update({"font.size": 10, "figure.dpi": 300, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False})


def robustness_curves(rob_dir, out_dir, split="test"):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, run in zip(axes.flat, RUNS):
        f = rob_dir / f"{run}_{split}_prediction_robustness.csv"
        if not f.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(f)
        clean = df[df.family == "clean"]["accuracy"].iloc[0]
        for fam in FAM_COLORS:
            sub = df[df.family == fam].sort_values("severity")
            if len(sub):
                ax.plot([0] + list(sub.severity), [clean] + list(sub.accuracy),
                        "-o", ms=3, lw=1.3, color=FAM_COLORS[fam], label=fam)
        ax.set_title(RUN_LABELS.get(run, run))
        ax.set_xlabel("severity")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, max(0.9, clean + 0.05))
        ax.set_xticks(range(6))
    axes.flat[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Prediction robustness: accuracy vs perturbation severity", fontsize=13)
    fig.tight_layout()
    p = out_dir / "robustness_curves.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def robustness_heatmap(rob_dir, out_dir, split="test"):
    mat, fams, labels = [], None, []
    for run in RUNS:
        f = rob_dir / f"{run}_{split}_robustness_summary.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).sort_values("family")
        fams = list(df["family"])
        mat.append(list(df["acc_drop_sev5"]))
        labels.append(RUN_LABELS.get(run, run))
    if not mat:
        return None
    mat = np.array(mat)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(mat, cmap="Reds", aspect="auto", vmin=0, vmax=max(0.5, mat.max()))
    ax.set_xticks(range(len(fams)))
    ax.set_xticklabels(fams, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black" if mat[i, j] < 0.35 else "white")
    fig.colorbar(im, ax=ax, label="accuracy drop @ severity 5")
    ax.set_title("Where each model breaks (higher = more fragile)")
    fig.tight_layout()
    p = out_dir / "robustness_heatmap.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def stability_gap(rob_dir, out_dir, split="test"):
    rows = []
    for run in RUNS:
        pf = rob_dir / f"{run}_{split}_robustness_summary.csv"
        sf = rob_dir / f"{run}_{split}_explanation_stability_overall.json"
        if not (pf.exists() and sf.exists()):
            continue
        pred_consistency = pd.read_csv(pf)["mean_consistency"].mean()
        expl = json.loads(sf.read_text())["mean_ssim"]
        rows.append((RUN_LABELS.get(run, run), pred_consistency, expl))
    if not rows:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, pc, ex in rows:
        ax.scatter(pc, ex, s=140, color="#c0392b", zorder=3)
        ax.annotate(label, (pc, ex), textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="equal stability")
    ax.set_xlabel("prediction stability (consistency)")
    ax.set_ylabel("explanation stability (SSIM)")
    ax.set_title("Stable predictions do not imply stable explanations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "stability_gap.png"
    fig.savefig(p)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rob-dir", default="results/robustness")
    ap.add_argument("--out-dir", default="results/figures")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rob_dir = Path(args.rob_dir)
    made = []
    for fn in (robustness_curves, robustness_heatmap, stability_gap):
        p = fn(rob_dir, out_dir, args.split)
        if p:
            made.append(p)
    print("Figures written:")
    for p in made:
        print(" ", p)


if __name__ == "__main__":
    main()
