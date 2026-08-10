"""Evaluate a trained checkpoint on a split and save all Section-9 metrics.

Also saves raw logits/probabilities to results/logits/<run_name>_<split>.npz,
which Phase 2 (calibration) and Phase 3 (uncertainty) reuse without re-running
inference.

Usage:
    python -m src.evaluation.evaluate \
        --checkpoint results/checkpoints/efficientnet_b3_seed42/best.pt \
        --split-csv data/splits/test.csv --split-name test
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score,
    average_precision_score,
)
from torch.utils.data import DataLoader

from src.data.dataset import HAM10000Dataset
from src.data.prepare_splits import CLASSES
from src.models.factory import create_model


@torch.no_grad()
def collect_logits(model, loader, device):
    model.eval()
    logits_all, targets_all, ids_all = [], [], []
    n_batches = len(loader)
    for i, (x, y, ids) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        logits_all.append(model(x).cpu())
        targets_all.append(y)
        ids_all.extend(list(ids))
        if (i + 1) % 20 == 0 or (i + 1) == n_batches:
            print(f"  batch {i+1}/{n_batches}", flush=True)
    return torch.cat(logits_all).numpy(), torch.cat(targets_all).numpy(), ids_all


def specificity_per_class(cm: np.ndarray) -> np.ndarray:
    spec = np.zeros(cm.shape[0])
    total = cm.sum()
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        spec[c] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    run_name = f"{ckpt['model']}_seed{ckpt['seed']}"

    model = create_model(ckpt["model"], num_classes=cfg.get("num_classes", 7),
                         pretrained=False, drop_rate=cfg.get("drop_rate", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    ds = HAM10000Dataset(args.split_csv, cfg.get("image_size", 224), train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.get("num_workers", 4), pin_memory=True)

    logits, targets, image_ids = collect_logits(model, loader, device)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    preds = probs.argmax(1)

    out = Path(args.out_dir)
    (out / "logits").mkdir(parents=True, exist_ok=True)
    (out / "csv").mkdir(parents=True, exist_ok=True)
    np.savez(out / "logits" / f"{run_name}_{args.split_name}.npz",
             logits=logits, probs=probs, targets=targets,
             image_ids=np.array(image_ids), classes=np.array(CLASSES))

    cm = confusion_matrix(targets, preds, labels=list(range(len(CLASSES))))
    present = np.unique(targets)
    y_onehot = np.eye(len(CLASSES))[targets]

    metrics = {
        "run": run_name,
        "split": args.split_name,
        "n": int(len(targets)),
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
        "specificity_macro": float(specificity_per_class(cm)[present].mean()),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(targets, preds, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(targets, preds),
        "cohen_kappa": cohen_kappa_score(targets, preds),
    }
    try:
        metrics["auroc_macro_ovr"] = roc_auc_score(targets, probs, multi_class="ovr", average="macro")
        metrics["auprc_macro"] = average_precision_score(y_onehot[:, present], probs[:, present], average="macro")
    except ValueError as e:  # e.g. a class absent from this split
        metrics["auroc_macro_ovr"] = None
        metrics["auprc_macro"] = None
        print(f"AUROC/AUPRC skipped: {e}")

    per_class = pd.DataFrame({
        "class": CLASSES,
        "support": cm.sum(1),
        "precision": precision_score(targets, preds, average=None, labels=range(len(CLASSES)), zero_division=0),
        "recall_sensitivity": recall_score(targets, preds, average=None, labels=range(len(CLASSES)), zero_division=0),
        "specificity": specificity_per_class(cm),
        "f1": f1_score(targets, preds, average=None, labels=range(len(CLASSES)), zero_division=0),
    })

    pd.DataFrame([metrics]).to_csv(out / "csv" / f"{run_name}_{args.split_name}_metrics.csv", index=False)
    per_class.to_csv(out / "csv" / f"{run_name}_{args.split_name}_per_class.csv", index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(out / "csv" / f"{run_name}_{args.split_name}_confusion.csv")

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}, indent=2))
    print(per_class.to_string(index=False))
    print(f"\nSaved: metrics/per-class/confusion CSVs in {out}/csv/, raw probabilities in {out}/logits/")


if __name__ == "__main__":
    main()
