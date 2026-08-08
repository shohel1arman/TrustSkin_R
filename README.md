# TrustSkin

**Beyond Accuracy — A Multidimensional Framework for Trustworthy Skin Lesion Classification under Uncertainty and Distribution Shift**

TrustSkin evaluates whether an image-based skin-lesion classifier is genuinely *trustworthy*, not merely accurate. A model can be highly confident and still wrong; a clinically meaningful system should predict accurately, recognize its own uncertainty, focus on the lesion rather than acquisition artifacts, stay stable under realistic image changes, behave consistently with its own explanations, and degrade gracefully on unfamiliar data. TrustSkin measures all of these jointly and combines them into a single per-image Trustworthiness Score.

Target venue: **IET Computer Vision** (Wiley, open access).

---

## Why this project

Most skin-lesion papers report accuracy, F1, and AUROC and stop there. TrustSkin is deliberately not another classifier. The contribution is an evaluation *framework* and two proposed instruments — the Explanation–Decision Alignment Score (EDAS) and an image-level Trustworthiness Score (TS) — validated behaviorally. The central experimental question throughout is: **do these reliability dimensions reveal unsafe behavior that conventional accuracy and confidence metrics miss?**

The four backbones (EfficientNet-B3, ConvNeXt-Tiny, ViT-B/16, Swin-T) are representative substrates, not the novelty. A recurring question is whether the *most accurate* architecture is also the *most trustworthy*.

---

## Trustworthiness dimensions

| Dimension | What it asks | Core methods |
|---|---|---|
| Correctness | Is the prediction right? | Accuracy, balanced accuracy, macro-F1, MCC, AUROC, AUPRC |
| Calibration | Does confidence match correctness? | ECE, MCE, Brier, NLL, reliability diagrams, temperature scaling |
| Uncertainty | Does the model know when it may be wrong? | MC Dropout, deep ensembles, risk–coverage, AURC, E-AURC |
| Explanation quality | Does the explanation point at the lesion? | Grad-CAM/++, Integrated Gradients, attention rollout; IoU, Dice, pointing game |
| Robustness | Do predictions survive realistic perturbations? | 7 perturbation families × 5 severities; consistency, confidence degradation, JS divergence |
| Explanation stability | Do explanations survive the same perturbations? | SSIM, cosine, IoU/Dice, Spearman between clean and perturbed maps |
| Alignment (EDAS) | Do explanation-important regions actually drive the decision? | Paired counterfactual masking of important vs control regions |
| Distribution shift | Do trust signals degrade informatively off-domain? | External ISIC evaluation on a locked set |

---

## Project phases

Each phase is designed and sandbox-verified before running, then run on the GPU node, then reviewed against results before moving on. Phases 2–3 reuse the saved probability files from Phase 1, so they need no re-inference.

| Phase | Content | Status |
|---|---|---|
| 1 | Leakage-controlled splits, baseline training (4 backbones), full classification metrics | ready |
| 2 | Calibration: ECE, MCE, Brier, NLL, reliability diagrams, temperature scaling | next |
| 3 | Uncertainty: MC Dropout, deep ensembles, risk–coverage, AURC, error detection | planned |
| 4 | XAI + localization: Grad-CAM/++, IG, attention rollout; IoU/Dice/pointing game vs lesion masks; lesion/background/artifact attribution | planned |
| 5 | Robustness + explanation stability: perturbation families × severities | planned |
| 6 | EDAS: counterfactual important-vs-control region interventions | planned |
| 7 | Trustworthiness Score, external ISIC shift, ablations (A0–A6), statistical tests | planned |

---

## Repository layout

```
TrustSkin_R/
├── configs/                 # one YAML per backbone + smoke config
│   ├── efficientnet.yaml
│   ├── convnext.yaml
│   ├── vit.yaml
│   ├── swin.yaml
│   └── smoke.yaml           # tiny resnet18 pipeline test, NOT for experiments
├── src/
│   ├── data/
│   │   ├── prepare_splits.py   # lesion-grouped, leakage-checked train/val/test
│   │   └── dataset.py          # HAM10000 dataset + train/eval transforms
│   ├── models/
│   │   └── factory.py          # timm backbones + MC-Dropout hook (Phase 3)
│   ├── training/
│   │   └── train.py            # AdamW, cosine, AMP, class-weighted loss, early stop
│   └── evaluation/
│       └── evaluate.py         # all Section-9 metrics + saved logits/probs
├── data/
│   ├── ham10000/            # dataset (gitignored)
│   └── splits/              # locked CSVs (committed)
├── results/
│   ├── csv/                 # metrics / per-class / confusion
│   ├── logits/              # saved probabilities, reused by Phases 2–3 (gitignored)
│   └── checkpoints/         # best.pt, history.csv, config, env (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

Requires Python 3.10+ and [`uv`](https://github.com/astral-sh/uv). If conda is active, run `conda deactivate` first to avoid PATH conflicts.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -c "import torch, timm, sklearn, pandas, PIL, yaml; print('imports OK')"
```

---

## Dataset

**HAM10000** — 10,015 dermoscopic images, 7 diagnostic classes. Primary development and internal evaluation set.

| Code | Category | Type |
|---|---|---|
| MEL | Melanoma | Malignant |
| NV | Melanocytic nevi | Benign |
| BCC | Basal cell carcinoma | Malignant |
| AKIEC | Actinic keratoses / intraepithelial carcinoma | Pre-/malignant |
| BKL | Benign keratosis | Benign |
| DF | Dermatofibroma | Benign |
| VASC | Vascular lesion | Benign |

Source: Kaggle `kmader/skin-cancer-mnist-ham10000`, or Harvard Dataverse DOI `10.7910/DVN/DBW86T`.

```bash
mkdir -p data/ham10000 && cd data/ham10000
kaggle datasets download kmader/skin-cancer-mnist-ham10000
unzip -q skin-cancer-mnist-ham10000.zip
cd ../..
```

Expected after unzip: `HAM10000_metadata.csv`, plus images in `HAM10000_images_part_1/` and `HAM10000_images_part_2/`. If your unzip produces a single combined image folder, pass that one folder to `--image-dirs` instead of two.

**External validation (Phase 7):** ISIC 2018/2019, label- and preprocessing-mapped, kept isolated from all development. It is never seen during training or model selection.

The class distribution is heavily imbalanced (NV dominates), so the project emphasizes macro-averaged metrics, balanced accuracy, MCC, and per-class analysis over raw accuracy.

---

## Phase 1 workflow

**1. Build leakage-controlled splits.** HAM10000 has multiple images per lesion (`lesion_id`); if one lesion appears in two splits, metrics are inflated. The script groups by `lesion_id`, stratifies approximately by class, and hard-fails if any lesion leaks. Roughly 70/10/20 by lesion.

```bash
uv run python -m src.data.prepare_splits \
    --metadata data/ham10000/HAM10000_metadata.csv \
    --image-dirs data/ham10000/HAM10000_images_part_1 data/ham10000/HAM10000_images_part_2 \
    --out-dir data/splits --seed 42
```

Commit the resulting `data/splits/*.csv` — the exact split is part of reproducibility.

**2. Train each backbone.** Best checkpoint is selected on validation macro-F1. Add `--seed 43` / `--seed 44` later for the multi-seed final table and the Phase-3 ensemble.

```bash
uv run python -m src.training.train --config configs/efficientnet.yaml
uv run python -m src.training.train --config configs/convnext.yaml
uv run python -m src.training.train --config configs/vit.yaml
uv run python -m src.training.train --config configs/swin.yaml
```

**3. Evaluate on val and test.** Repeat per checkpoint and split.

```bash
uv run python -m src.evaluation.evaluate \
    --checkpoint results/checkpoints/efficientnet_b3_seed42/best.pt \
    --split-csv data/splits/test.csv --split-name test
```

Or all four at once:

```bash
for m in efficientnet_b3 convnext_tiny vit_b16 swin_tiny; do
  uv run python -m src.evaluation.evaluate --checkpoint results/checkpoints/${m}_seed42/best.pt --split-csv data/splits/val.csv  --split-name val
  uv run python -m src.evaluation.evaluate --checkpoint results/checkpoints/${m}_seed42/best.pt --split-csv data/splits/test.csv --split-name test
done
```

**Outputs:** metrics / per-class / confusion CSVs in `results/csv/`; raw logits and softmax probabilities in `results/logits/<run>_<split>.npz` (reused by Phases 2–3 with no re-inference); best checkpoint, `history.csv`, resolved config, and environment record in `results/checkpoints/<run>/`.

**Pipeline smoke test** (validates the code path before spending GPU time):

```bash
uv run python -m src.training.train --config configs/smoke.yaml
```

---

## Running on the GPU node

Train inside `tmux` so an SSH drop does not kill the run. Rebuild splits on the node with the same seed so the `path` column points at node-local files (lesion assignment is identical; only paths differ).

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

## Sanity expectations

On a lesion-grouped HAM10000 test split, test macro-F1 around **0.70–0.80** is healthy. Test macro-F1 above ~0.90 is a red flag for leakage, not a success — investigate the split before trusting it. One backbone far below the others usually means a learning-rate mismatch in its config.

---

## Reproducibility rules (do not break)

- The test split is **locked** after `prepare_splits`; never tune or model-select on it.
- External ISIC data (Phase 7) is never seen during development.
- Every result CSV and checkpoint is tied to a `<model>_seed<seed>` run name.
- Final reported numbers are mean ± std over seeds 42 / 43 / 44.
- Config, seed, library versions, and hardware are recorded per run (`config_used.yaml`, `env.json`).

---

## Ethical scope

HAM10000 and ISIC do not represent all populations, devices, or clinical settings. The Trustworthiness Score is a research instrument, not a validated clinical decision rule — any clinical use would require prospective validation, regulatory review, and clinician-in-the-loop studies. XAI heatmaps are hypotheses about model behavior, not causal proof, which is exactly why EDAS and counterfactual testing are included.

---

## Citation

A `CITATION.cff` and BibTeX entry will be added on submission.
