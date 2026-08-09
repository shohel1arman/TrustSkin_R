"""Phase 3 - Uncertainty via MC Dropout."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader

from src.data.dataset import HAM10000Dataset
from src.models.factory import create_model, enable_mc_dropout

EPS = 1e-12


def entropy(p, axis=-1):
    return -(p * np.log(p + EPS)).sum(axis=axis)


@torch.no_grad()
def mc_forward(model, loader, device, T):
    enable_mc_dropout(model)
    all_pass = []
    targets, ids = [], []
    first = True
    for t in range(T):
        pass_probs = []
        tg_tmp, id_tmp = [], []
        for x, y, image_ids in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            pass_probs.append(probs)
            if first:
                tg_tmp.append(y.numpy())
                id_tmp.extend(list(image_ids))
        all_pass.append(np.concatenate(pass_probs, axis=0))
        if first:
            targets = np.concatenate(tg_tmp, axis=0)
            ids = np.array(id_tmp)
            first = False
    return np.stack(all_pass, axis=0), targets, ids


def uncertainty_scores(probs_TNK):
    T, N, K = probs_TNK.shape
    mean_prob = probs_TNK.mean(axis=0)
    pe = entropy(mean_prob, axis=1)
    ee = entropy(probs_TNK, axis=2).mean(axis=0)
    mi = pe - ee
    var = probs_TNK.var(axis=0).mean(axis=1)
    conf = mean_prob.max(axis=1)
    preds = mean_prob.argmax(axis=1)
    return {"mean_prob": mean_prob, "pred": preds,
            "predictive_entropy": pe, "expected_entropy": ee,
            "mutual_information": mi, "variance": var, "confidence": conf}


def risk_coverage(correct, score_uncertain):
    order = np.argsort(score_uncertain)
    correct_sorted = correct[order]
    n = len(correct)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - cum_correct / np.arange(1, n + 1)
    _trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(_trap(risk, coverage))
    return coverage, risk, aurc


def error_detection(correct, score_uncertain):
    err = (1 - correct).astype(int)
    if err.sum() == 0 or err.sum() == len(err):
        return None, None
    return (float(roc_auc_score(err, score_uncertain)),
            float(average_precision_score(err, score_uncertain)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--out-dir", default="results/uncertainty")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    run = f"{ckpt['model']}_seed{ckpt['seed']}"
    model = create_model(ckpt["model"], num_classes=cfg.get("num_classes", 7),
                         pretrained=False, drop_rate=cfg.get("drop_rate", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    ds = HAM10000Dataset(args.split_csv, cfg.get("image_size", 224), train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=cfg.get("num_workers", 4))

    probs_TNK, targets, ids = mc_forward(model, loader, device, args.T)
    s = uncertainty_scores(probs_TNK)
    correct = (s["pred"] == targets).astype(int)

    out = Path(args.out_dir)
    (out / "per_image").mkdir(parents=True, exist_ok=True)
    (out / "risk_coverage").mkdir(parents=True, exist_ok=True)

    per_img = pd.DataFrame({
        "image_id": ids, "target": targets, "pred": s["pred"], "correct": correct,
        "confidence": s["confidence"], "predictive_entropy": s["predictive_entropy"],
        "expected_entropy": s["expected_entropy"], "mutual_information": s["mutual_information"],
        "variance": s["variance"]})
    per_img.to_csv(out / "per_image" / f"{run}_{args.split_name}.csv", index=False)
    np.savez(out / "per_image" / f"{run}_{args.split_name}_meanprob.npz",
             mean_prob=s["mean_prob"], targets=targets, image_ids=ids)

    scores = {"neg_confidence": -s["confidence"],
              "predictive_entropy": s["predictive_entropy"],
              "expected_entropy": s["expected_entropy"],
              "mutual_information": s["mutual_information"],
              "variance": s["variance"]}
    rows = []
    for name, sc in scores.items():
        auroc, auprc = error_detection(correct, sc)
        _, _, aurc = risk_coverage(correct, sc)
        rows.append({"run": run, "split": args.split_name, "score": name,
                     "err_auroc": None if auroc is None else round(auroc, 4),
                     "err_auprc": None if auprc is None else round(auprc, 4),
                     "aurc": round(aurc, 4)})
        cov, risk, _ = risk_coverage(correct, sc)
        idx = np.linspace(0, len(cov) - 1, 100).astype(int)
        pd.DataFrame({"coverage": cov[idx], "risk": risk[idx]}).to_csv(
            out / "risk_coverage" / f"{run}_{args.split_name}_{name}.csv", index=False)

    summary = pd.DataFrame(rows)
    summary.to_csv(out / f"{run}_{args.split_name}_uncertainty.csv", index=False)
    meta = {"run": run, "split": args.split_name, "T": args.T, "n": int(len(targets)),
            "accuracy": float(correct.mean()), "device": str(device)}
    (out / f"{run}_{args.split_name}_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"{run} [{args.split_name}] T={args.T} n={len(targets)} acc={correct.mean():.4f}")
    print(summary.to_string(index=False))
    print(f"\nSaved under {out}/")


if __name__ == "__main__":
    main()
