# TrustSkin

Code for **"Beyond accuracy: a multidimensional trustworthiness evaluation of deep models for skin lesion classification"** (Arman & Ahmed, manuscript under review).

The question is simple: when two skin lesion classifiers reach similar accuracy, are they equally trustworthy? We train five backbones under one fixed, leakage-controlled protocol on HAM10000, repeat everything over three seeds (42, 43, 44), and evaluate each model on four axes beyond accuracy.

| Axis | Question | Method |
|---|---|---|
| Calibration | Does confidence match correctness? | ECE, MCE, Brier, NLL; temperature scaling fit on validation only |
| Predictive uncertainty | Does the model know when it may be wrong? | MC dropout (T=30); total vs epistemic uncertainty, scored as error-detection AUROC |
| Explanation localization | Does the explanation point at the lesion? | Grad-CAM++ vs expert lesion masks (IoU, pointing game) |
| Explanation faithfulness | Does the decision actually depend on the highlighted region? | EDAS: masking the salient region vs a same-size control region |
| Distribution shift | Does any of this survive new data? | De-contaminated ISIC-2019 external set (14,885 images) |

Backbones: ResNet-50, EfficientNet-B3 (convolutional); ConvNeXt-Tiny, ViT-B/16, Swin-Tiny (attention-era).

## Main findings

- Accuracy and trustworthiness are decoupled. The most accurate in-distribution models are among the least lesion-focused and least faithful.
- Epistemic uncertainty from MC dropout is near-random for the CNNs (error-detection AUROC ~0.50) but informative for the attention-era models (~0.83).
- Under distribution shift, calibration does not transfer and melanoma recall drops to 0.26-0.37.
- A learned score combining all axes does not beat plain confidence at flagging errors, so trust is reported as a profile, not a single number.

## Repository layout

```
TrustSkin_R/
├── configs/                  # one YAML per backbone (+ smoke.yaml for a quick pipeline check)
├── data/splits/              # locked lesion-grouped splits + de-contaminated external CSV
├── src/
│   ├── data/                 # split preparation and dataset
│   ├── data_prep/            # ISIC-2019 de-contamination
│   ├── models/               # timm backbones + MC-dropout hook
│   ├── training/             # training loop
│   ├── evaluation/           # classification metrics, saved logits
│   ├── calibration/          # temperature scaling (in-distribution and external)
│   ├── uncertainty/          # MC dropout
│   ├── xai/                  # Grad-CAM++ localization vs lesion masks
│   ├── edas/                 # counterfactual faithfulness (EDAS)
│   ├── trust/                # learned trust-score combiner (negative result)
│   ├── robustness/           # exploratory, not reported in the paper
│   └── visualization/
├── results/                  # per-seed CSVs and results/aggregate/ (mean ± std)
├── figures/                  # paper figures
└── make_figures.py
```

Checkpoints and raw logits are not committed (too large).

## Setup

Python 3.10+ and [uv](https://github.com/astral-sh/uv):

```
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

- **HAM10000**: Kaggle `kmader/skin-cancer-mnist-ham10000` or Harvard Dataverse (DOI `10.7910/DVN/DBW86T`). Put it under `data/ham10000/`.
- **Lesion masks** (localization and EDAS): Tschandl et al. HAM10000 lesion segmentations.
- **ISIC-2019** (external test only): put it under `data/isic2019/`. It contains all of HAM10000, so it must be de-contaminated before use (step 1 below).

## Reproducing the pipeline

**0. Splits.** HAM10000 has several photos of the same lesion, so we split by `lesion_id`. The split CSVs in `data/splits/` are the exact ones used in the paper.

```
uv run python -m src.data.prepare_splits \
    --metadata data/ham10000/HAM10000_metadata.csv \
    --image-dirs data/ham10000/HAM10000_images_part_1 data/ham10000/HAM10000_images_part_2 \
    --out-dir data/splits --seed 42
```

**1. External set.** Removes every ISIC-2019 image that originates from HAM10000 (exactly 10,015), plus the SCC and unknown classes.

```
uv run python -m src.data_prep.decontaminate_isic
```

**2. Train.** Repeat with `--seed 43` and `--seed 44`.

```
for m in resnet50 efficientnet convnext vit swin; do
  uv run python -m src.training.train --config configs/$m.yaml --seed 42
done
```

**3. Evaluate** (saves logits used by later stages):

```
uv run python -m src.evaluation.evaluate \
    --checkpoint results/checkpoints/<run>/best.pt \
    --split-csv data/splits/test.csv --split-name test
```

**4. Trust axes:**

```
# calibration (temperature fit on val, applied to test)
uv run python -m src.calibration.calibrate --runs <run> [<run> ...]
uv run python -m src.calibration.calibrate_external --runs <run> [<run> ...]

# MC-dropout uncertainty
uv run python -m src.uncertainty.mc_dropout --checkpoint <ckpt> --split-csv data/splits/test.csv

# Grad-CAM++ localization
uv run python -m src.xai.gradcam_localize --checkpoint <ckpt> --split-csv data/splits/test.csv --mask-dir <masks>

# EDAS faithfulness
uv run python -m src.edas.edas_counterfactual --checkpoint <ckpt> --split-csv data/splits/test.csv

# learned trust score
uv run python -m src.trust.learned_trust --run <run>
```

Note: `calibrate.py` overwrites the shared `calibration_summary.csv` with only the runs you pass, so pass all runs together.

**5. Figures:**

```
uv run python make_figures.py
```

## Rules we kept

- The test split is locked; nothing is tuned or selected on it.
- The external set is never used for training, model selection, or fitting temperatures.
- Every number in the paper is mean ± std over seeds 42/43/44.

## Scope

This is a research evaluation, not a clinical tool. HAM10000 and ISIC do not cover every population, device, or clinic, and the clinical implications in the paper were not validated with dermatologists.
