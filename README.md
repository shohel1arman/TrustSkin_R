# TrustSkin

A framework for checking whether a skin-lesion image classifier is actually *trustworthy*, not just accurate.

A model can be confident and still wrong. TrustSkin looks at a prediction from several angles at once — is it correct, does its confidence match reality, does it know when it might be wrong, does its explanation point at the lesion rather than the background, does it stay stable when the image is perturbed, and does it hold up on data from a different source — and combines these into a single per-image trust score.

The four backbones (EfficientNet-B3, ConvNeXt-Tiny, ViT-B/16, Swin-T) are just representative models to test the framework on. The interesting question is whether the most accurate model is also the most trustworthy — often it isn't.

---

## What gets measured

| Dimension | The question | How |
|---|---|---|
| Correctness | Is the prediction right? | Accuracy, balanced accuracy, macro-F1, MCC, AUROC, AUPRC |
| Calibration | Does confidence match correctness? | ECE, MCE, Brier, NLL, reliability diagrams, temperature scaling |
| Uncertainty | Does the model know when it might be wrong? | MC Dropout, deep ensembles, risk–coverage, AURC |
| Explanation quality | Does the explanation point at the lesion? | Grad-CAM/++, Integrated Gradients, attention rollout; IoU, Dice, pointing game |
| Robustness | Do predictions survive realistic image changes? | 7 perturbation families × 5 severities |
| Explanation stability | Do explanations survive the same changes? | SSIM, cosine, IoU/Dice, Spearman on clean vs perturbed maps |
| Alignment (EDAS) | Do the "important" regions actually drive the decision? | Paired counterfactual masking of important vs control regions |
| Distribution shift | Do the trust signals degrade sensibly on outside data? | External ISIC evaluation on a locked set |

---

## How it's organized

The work moves in stages. First: leakage-controlled data splits, baseline training for the four backbones, and full classification metrics. Then calibration, uncertainty, explainability and localization, robustness and explanation stability, the EDAS counterfactual test, and finally the combined trust score with external-shift evaluation and ablations. Later stages reuse the saved probabilities from the baseline stage, so they don't need to re-run inference.

---

## Repository layout

```
TrustSkin_R/
├── configs/                 # one YAML per backbone + a smoke config
│   ├── efficientnet.yaml
│   ├── convnext.yaml
│   ├── vit.yaml
│   ├── swin.yaml
│   └── smoke.yaml           # tiny resnet18 pipeline check, not for real runs
├── src/
│   ├── data/
│   │   ├── prepare_splits.py   # lesion-grouped, leakage-checked train/val/test
│   │   └── dataset.py          # HAM10000 dataset + train/eval transforms
│   ├── models/
│   │   └── factory.py          # timm backbones + MC-Dropout hook
│   ├── training/
│   │   └── train.py            # AdamW, cosine, AMP, class-weighted loss, early stop
│   └── evaluation/
│       └── evaluate.py         # all classification metrics + saved logits/probs
├── data/
│   ├── ham10000/            # dataset (gitignored)
│   └── splits/              # locked split CSVs (committed)
├── results/
│   ├── csv/                 # metrics / per-class / confusion
│   ├── logits/              # saved probabilities, reused later (gitignored)
│   └── checkpoints/         # best.pt, history.csv, config, env (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

Needs Python 3.10+ and [`uv`](https://github.com/astral-sh/uv). If conda is active, run `conda deactivate` first.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch, timm, sklearn, pandas, PIL, yaml; print('imports OK')"
```

---

## Dataset

**HAM10000** — 10,015 dermoscopic images, 7 classes. The class balance is skewed (NV dominates), so the project leans on macro metrics, balanced accuracy, MCC, and per-class numbers rather than raw accuracy.

| Code | Category | Type |
|---|---|---|
| MEL | Melanoma | Malignant |
| NV | Melanocytic nevi | Benign |
| BCC | Basal cell carcinoma | Malignant |
| AKIEC | Actinic keratoses / intraepithelial carcinoma | Pre-/malignant |
| BKL | Benign keratosis | Benign |
| DF | Dermatofibroma | Benign |
| VASC | Vascular lesion | Benign |

Grab it from Kaggle (`kmader/skin-cancer-mnist-ham10000`) or Harvard Dataverse (DOI `10.7910/DVN/DBW86T`):

```bash
mkdir -p data/ham10000 && cd data/ham10000
kaggle datasets download kmader/skin-cancer-mnist-ham10000
unzip -q skin-cancer-mnist-ham10000.zip
cd ../..
```

After unzipping you should have `HAM10000_metadata.csv` and images in `HAM10000_images_part_1/` and `HAM10000_images_part_2/`. If your unzip drops everything into one image folder instead of two, just pass that single folder to `--image-dirs`.

External data (ISIC 2018/2019) is used only for the out-of-domain check and is kept away from all training and model selection.

---

## Running it

**1. Make the splits.** HAM10000 has several images of the same lesion, so splitting by image leaks. This groups by `lesion_id`, stratifies roughly by class, and stops with an error if any lesion ends up in two splits. Roughly 70/10/20.

```bash
uv run python -m src.data.prepare_splits \
    --metadata data/ham10000/HAM10000_metadata.csv \
    --image-dirs data/ham10000/HAM10000_images_part_1 data/ham10000/HAM10000_images_part_2 \
    --out-dir data/splits --seed 42
```

Commit `data/splits/*.csv` — the exact split matters for reproducing results.

**2. Train the backbones.** Best checkpoint is picked on validation macro-F1. Add `--seed 43` / `--seed 44` for the multi-seed runs.

```bash
uv run python -m src.training.train --config configs/efficientnet.yaml
uv run python -m src.training.train --config configs/convnext.yaml
uv run python -m src.training.train --config configs/vit.yaml
uv run python -m src.training.train --config configs/swin.yaml
```

**3. Evaluate on val and test.**

```bash
uv run python -m src.evaluation.evaluate \
    --checkpoint results/checkpoints/efficientnet_b3_seed42/best.pt \
    --split-csv data/splits/test.csv --split-name test
```

Or loop over all four:

```bash
for m in efficientnet_b3 convnext_tiny vit_b16 swin_tiny; do
  uv run python -m src.evaluation.evaluate --checkpoint results/checkpoints/${m}_seed42/best.pt --split-csv data/splits/val.csv  --split-name val
  uv run python -m src.evaluation.evaluate --checkpoint results/checkpoints/${m}_seed42/best.pt --split-csv data/splits/test.csv --split-name test
done
```

Everything lands in `results/csv/` (metrics, per-class, confusion), `results/logits/` (raw probabilities, reused later), and `results/checkpoints/<run>/` (checkpoint, training history, config, environment).

**Quick pipeline check** before spending GPU time:

```bash
uv run python -m src.training.train --config configs/smoke.yaml
```

---

## On the GPU node

Run training in `tmux` so a dropped SSH connection doesn't kill it. Rebuild the splits on the node with the same seed so the file paths point to node-local images (same lesion assignment, just different paths).

```bash
tmux new -s trustskin
uv run python -m src.data.prepare_splits \
    --metadata data/ham10000/HAM10000_metadata.csv \
    --image-dirs data/ham10000/HAM10000_images_part_1 data/ham10000/HAM10000_images_part_2 \
    --out-dir data/splits --seed 42
uv run python -m src.training.train --config configs/efficientnet.yaml && \
uv run python -m src.training.train --config configs/convnext.yaml && \
uv run python -m src.training.train --config configs/vit.yaml && \
uv run python -m src.training.train --config configs/swin.yaml
# detach: Ctrl-b then d   |   reattach: tmux attach -t trustskin
```

---

## Quick sanity check

On a lesion-grouped split, test macro-F1 around **0.70–0.80** is normal. If it's above ~0.90, that usually means leakage, not a great model — check the split. If one backbone is way behind the others, it's probably a learning-rate mismatch in its config.

---

## A few rules worth keeping

- The test split is locked once made — never tune or model-select on it.
- External ISIC data stays unseen during development.
- Every result and checkpoint carries its `<model>_seed<seed>` run name.
- Final numbers are mean ± std over seeds 42 / 43 / 44.
- Config, seed, library versions, and hardware are saved per run.

---

## A note on scope

HAM10000 and ISIC don't cover every population, device, or clinic. The trust score is a research tool, not a clinical decision rule — real use would need prospective validation and clinical review. Heatmaps are hints about model behavior, not proof of cause, which is the whole reason EDAS and the counterfactual tests exist.
