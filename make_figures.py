#!/usr/bin/env python3
"""TrustSkin - generate all publication figures from committed results.

Run from the repo root (where results/ lives):
    python make_figures.py

Reads only committed result files; every number in every figure traces to
those files. Writes figures/ as 300-dpi PNG + vector PDF.

Figures:
  fig2_classification   - accuracy & macro-F1, mean+/-std, 5 models
  fig4_epistemic        - error-detection AUROC: total vs epistemic (the finding)
  fig5_localization     - pointing game & IoU vs lesion masks, mean+/-std
  fig7_edas             - EDAS faithfulness gap, mean+/-std
  fig9_decoupling       - accuracy vs trustworthiness scatter (the thesis)
"""
import os
import re
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 8.5, "figure.dpi": 120,
})

SEEDS = [42, 43, 44]
MODELS = ["resnet50", "efficientnet_b3", "convnext_tiny", "vit_b16", "swin_tiny"]
LABELS = {"resnet50": "ResNet-50", "efficientnet_b3": "EfficientNet-B3",
          "convnext_tiny": "ConvNeXt-T", "vit_b16": "ViT-B/16", "swin_tiny": "Swin-T"}
IS_CNN = {"resnet50": True, "efficientnet_b3": True,
          "convnext_tiny": False, "vit_b16": False, "swin_tiny": False}
C_CNN, C_ATT = "#C44E52", "#4C72B0"
os.makedirs("figures", exist_ok=True)


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"figures/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved figures/{name}.pdf + .png")


def parse_ms(s):
    """'0.6605 ± 0.0130' -> (0.6605, 0.0130); tolerates plain floats."""
    s = str(s)
    m = re.findall(r"[-+]?\d*\.?\d+", s)
    return (float(m[0]), float(m[1]) if len(m) > 1 else 0.0)


# ---------------------------------------------------------------- data readers
def read_uncertainty(score):
    means, stds = [], []
    for m in MODELS:
        vals = []
        for s in SEEDS:
            d = "results/uncertainty" if s == 42 else f"results/uncertainty_seed{s}"
            f = f"{d}/{m}_seed{s}_test_uncertainty.csv"
            df = pd.read_csv(f)
            vals.append(float(df[df["score"] == score]["err_auroc"].iloc[0]))
        means.append(np.mean(vals)); stds.append(np.std(vals, ddof=1))
    return np.array(means), np.array(stds)


def _find_json(dirpath, model, seed, kind):
    """kind: 'gradcam' or 'edas'. Returns the summary json path or None."""
    pats = glob.glob(f"{dirpath}/{model}_seed{seed}_*{'gradcam' if kind=='gradcam' else 'edas'}*.json")
    return pats[0] if pats else None


def read_localization(key):
    means, stds = [], []
    for m in MODELS:
        vals = []
        for s in SEEDS:
            d = "results/xai" if s == 42 else f"results/xai_seed{s}"
            f = _find_json(d, m, s, "gradcam")
            if f is None:
                vals.append(np.nan); continue
            vals.append(float(json.load(open(f))[key]))
        vals = [v for v in vals if not np.isnan(v)]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def read_edas():
    means, stds = [], []
    for m in MODELS:
        vals = []
        for s in SEEDS:
            d = "results/edas" if s == 42 else f"results/edas_seed{s}"
            f = _find_json(d, m, s, "edas")
            if f is None:
                continue
            j = json.load(open(f))
            vals.append(float(j.get("mean_edas", j.get("mean_edas_norm", np.nan))))
        vals = [v for v in vals if v == v]
        means.append(np.mean(vals) if vals else np.nan)
        stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
    return np.array(means), np.array(stds)


def read_classification(col):
    df = pd.read_csv("results/aggregate/classification_meanstd.csv").set_index("model")
    means = np.array([parse_ms(df.loc[m, col])[0] for m in MODELS])
    stds = np.array([parse_ms(df.loc[m, col])[1] for m in MODELS])
    return means, stds


def bar_colors():
    return [C_CNN if IS_CNN[m] else C_ATT for m in MODELS]


# ---------------------------------------------------------------- figures
def fig2_classification():
    acc_m, acc_s = read_classification("accuracy")
    f1_m, f1_s = read_classification("f1_macro")
    x = np.arange(len(MODELS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, acc_m, w, yerr=acc_s, capsize=3, color="#55A868",
           edgecolor="black", lw=0.5, label="Accuracy")
    ax.bar(x + w/2, f1_m, w, yerr=f1_s, capsize=3, color="#8172B3",
           edgecolor="black", lw=0.5, label="Macro-F1")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("Score (HAM10000 test, mean $\\pm$ std over 3 seeds)")
    ax.set_ylim(0.4, 0.95); ax.legend(frameon=False)
    ax.set_title("Classification performance")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig2_classification")


def fig4_epistemic():
    tot_m, tot_s = read_uncertainty("predictive_entropy")
    epi_m, epi_s = read_uncertainty("mutual_information")
    x = np.arange(len(MODELS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w/2, tot_m, w, yerr=tot_s, capsize=3, color="#4C72B0",
           edgecolor="black", lw=0.5, label="Total uncertainty (predictive entropy)")
    ax.bar(x + w/2, epi_m, w, yerr=epi_s, capsize=3, color="#C44E52",
           edgecolor="black", lw=0.5, label="Epistemic uncertainty (mutual information)")
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.text(len(MODELS)-0.5, 0.505, "random", ha="right", va="bottom", fontsize=8, color="grey")
    ax.axvline(1.5, ls=":", lw=1, color="black", alpha=0.4)
    ax.text(0.5, 1.03, "CNN", ha="center", fontsize=9, style="italic")
    ax.text(3.0, 1.03, "Attention-era", ha="center", fontsize=9, style="italic")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("Error-detection AUROC")
    ax.set_ylim(0.40, 1.10); ax.set_yticks(np.arange(0.4, 1.01, 0.1))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, ncol=1)
    ax.set_title("Epistemic uncertainty is informative only for attention-era backbones")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig4_epistemic")


def fig5_localization():
    pg_m, pg_s = read_localization("pointing_game")
    iou_m, iou_s = read_localization("mean_iou")
    x = np.arange(len(MODELS)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w/2, pg_m, w, yerr=pg_s, capsize=3, color="#DD8452",
           edgecolor="black", lw=0.5, label="Pointing game")
    ax.bar(x + w/2, iou_m, w, yerr=iou_s, capsize=3, color="#937860",
           edgecolor="black", lw=0.5, label="Mean IoU")
    ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in MODELS], rotation=15, ha="right")
    ax.set_ylabel("Localization vs lesion mask (mean $\\pm$ std)")
    ax.legend(frameon=False)
    ax.set_title("Lesion localization of Grad-CAM++ explanations")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig5_localization")


def fig7_edas():
    m, s = read_edas()
    x = np.arange(len(MODELS))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(x, m, 0.6, yerr=s, capsize=3, color=bar_colors(), edgecolor="black", lw=0.5)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([LABELS[mm] for mm in MODELS], rotation=15, ha="right")
    ax.set_ylabel("EDAS faithfulness gap (mean $\\pm$ std)")
    ax.set_title("Explanation-Decision Alignment: positive = faithful")
    ax.annotate("faithful", xy=(0.01, 0.98), xycoords="axes fraction", va="top", fontsize=8, color="green")
    ax.annotate("unfaithful", xy=(0.01, 0.02), xycoords="axes fraction", va="bottom", fontsize=8, color="red")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig7_edas")


def fig9_decoupling():
    acc_m, _ = read_classification("accuracy")
    iou_m, _ = read_localization("mean_iou")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for i, mm in enumerate(MODELS):
        ax.scatter(acc_m[i], iou_m[i], s=140,
                   color=(C_CNN if IS_CNN[mm] else C_ATT),
                   edgecolor="black", zorder=3)
        ax.annotate(LABELS[mm], (acc_m[i], iou_m[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    ax.set_xlabel("Accuracy (HAM10000 test)")
    ax.set_ylabel("Lesion localization (mean IoU)")
    ax.set_title("Accuracy and trustworthiness are decoupled")
    ax.scatter([], [], color=C_CNN, edgecolor="black", label="CNN")
    ax.scatter([], [], color=C_ATT, edgecolor="black", label="Attention-era")
    ax.legend(frameon=False, loc="lower left")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    save(fig, "fig9_decoupling")




def fig3_reliability():
    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5))
    for ax, m in zip(axes.flat, ["efficientnet_b3", "convnext_tiny", "vit_b16", "swin_tiny"]):
        f = f"results/calibration/reliability/{m}_seed42_test_post.csv"
        df = pd.read_csv(f).dropna(subset=["avg_conf", "accuracy"])
        ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        ax.bar(df["avg_conf"], df["accuracy"], width=0.05, color="#4C72B0",
               edgecolor="black", lw=0.4, alpha=0.85)
        ax.bar(df["avg_conf"], df["avg_conf"] - df["accuracy"], width=0.05,
               bottom=df["accuracy"], color="#C44E52", alpha=0.4, edgecolor="none")
        ax.set_title(LABELS[m], fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Accuracy")
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.suptitle("Reliability diagrams (post temperature scaling, HAM10000 test)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, "fig3_reliability")


def fig8_external():
    import json as _json
    ms = ["efficientnet_b3", "convnext_tiny", "vit_b16", "swin_tiny"]

    def ham(m, metric):
        return float(pd.read_csv(f"results/csv/{m}_seed42_test_metrics.csv").iloc[0][metric])

    def ext(m, metric):
        # prefer the committed external metrics CSV; fall back to the uncertainty meta json (accuracy only)
        p = f"results/csv/{m}_seed42_external_metrics.csv"
        if os.path.exists(p):
            return float(pd.read_csv(p).iloc[0][metric])
        if metric == "accuracy":
            mj = f"results/uncertainty_external/{m}_seed42_external_meta.json"
            if os.path.exists(mj):
                return float(_json.load(open(mj))["accuracy"])
        return np.nan

    x = np.arange(len(ms)); w = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, metric, title in [(ax1, "accuracy", "Accuracy"), (ax2, "f1_macro", "Macro-F1")]:
        h = [ham(m, metric) for m in ms]
        e = [ext(m, metric) for m in ms]
        ax.bar(x - w/2, h, w, color="#55A868", edgecolor="black", lw=0.5,
               label="HAM10000 (in-distribution)")
        ax.bar(x + w/2, e, w, color="#C44E52", edgecolor="black", lw=0.5,
               label="ISIC-2019 (external, OOD)")
        # mark any model whose external metric is unavailable
        for i, val in enumerate(e):
            if np.isnan(val):
                ax.text(x[i] + w/2, 0.02, "n/a", ha="center", va="bottom", fontsize=7, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([LABELS[m] for m in ms], rotation=15, ha="right")
        ax.set_ylabel(title); ax.set_ylim(0, 1); ax.set_title(title)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax1.legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Performance drops sharply under distribution shift", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "fig8_external")


if __name__ == "__main__":
    print("Generating TrustSkin figures...")
    for fn in (fig2_classification, fig3_reliability, fig4_epistemic, fig5_localization, fig7_edas, fig8_external, fig9_decoupling):
        try:
            fn()
        except Exception as e:
            print(f"  !! {fn.__name__} failed: {e}")
    print("Done. Figures in figures/")
