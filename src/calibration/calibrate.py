"""Phase 2 - Calibration analysis.

Reads the saved probability files from Phase 1
(results/logits/<run>_<split>.npz) and computes calibration metrics for each
model, fits temperature scaling on validation, applies it to test, and writes
reliability-diagram data and a summary CSV. No GPU, no re-inference.

Metrics:
  ECE   - Expected Calibration Error (equal-width bins)
  MCE   - Maximum Calibration Error
  Brier - multiclass Brier score (mean squared error vs one-hot)
  NLL   - negative log-likelihood

Temperature scaling divides logits by a single scalar T>0 found by minimizing
validation NLL. T>1 softens overconfident predictions; T<1 sharpens.

Usage:
    python -m src.calibration.calibrate --logits-dir results/logits \
        --runs efficientnet_b3_seed42 convnext_tiny_seed42 vit_b16_seed42 swin_tiny_seed42 \
        --out-dir results/calibration
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------- core metrics (operate on probabilities + integer targets) ----------

def _confidences_and_correct(probs: np.ndarray, targets: np.ndarray):
    preds = probs.argmax(1)
    conf = probs.max(1)
    correct = (preds == targets).astype(float)
    return conf, correct


def expected_calibration_error(probs, targets, n_bins=15):
    conf, correct = _confidences_and_correct(probs, targets)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, mce = 0.0, 0.0
    n = len(conf)
    per_bin = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        # last bin is closed on the right so conf==1.0 is included
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            per_bin.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                            "avg_conf": None, "accuracy": None})
            continue
        avg_conf = float(conf[mask].mean())
        acc = float(correct[mask].mean())
        gap = abs(acc - avg_conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        per_bin.append({"bin_lo": lo, "bin_hi": hi, "count": count,
                        "avg_conf": avg_conf, "accuracy": acc})
    return ece, mce, per_bin


def brier_score(probs, targets):
    n, k = probs.shape
    onehot = np.eye(k)[targets]
    return float(((probs - onehot) ** 2).sum(1).mean())


def nll(probs, targets, eps=1e-12):
    p = np.clip(probs[np.arange(len(targets)), targets], eps, 1.0)
    return float(-np.log(p).mean())


# ---------- temperature scaling ----------

def _softmax(logits, T=1.0):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(val_logits, val_targets, grid=None):
    """Find T>0 minimizing validation NLL via a fine grid search (no torch needed)."""
    if grid is None:
        grid = np.concatenate([np.linspace(0.5, 1.0, 26)[:-1],
                               np.linspace(1.0, 5.0, 81)])
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        p = _softmax(val_logits, T)
        cur = nll(p, val_targets)
        if cur < best_nll:
            best_nll, best_T = cur, float(T)
    return best_T, best_nll


def all_metrics(probs, targets, n_bins=15):
    ece, mce, per_bin = expected_calibration_error(probs, targets, n_bins)
    return {
        "ece": ece, "mce": mce,
        "brier": brier_score(probs, targets),
        "nll": nll(probs, targets),
        "accuracy": float((probs.argmax(1) == targets).mean()),
    }, per_bin


def load_npz(logits_dir: Path, run: str, split: str):
    f = logits_dir / f"{run}_{split}.npz"
    if not f.exists():
        raise FileNotFoundError(f"missing {f} - run Phase 1 evaluate for this run/split first")
    d = np.load(f, allow_pickle=True)
    return d["logits"], d["probs"], d["targets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", default="results/logits")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out-dir", default="results/calibration")
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()

    logits_dir = Path(args.logits_dir)
    out = Path(args.out_dir)
    (out / "reliability").mkdir(parents=True, exist_ok=True)

    rows = []
    for run in args.runs:
        val_logits, val_probs, val_targets = load_npz(logits_dir, run, "val")
        test_logits, test_probs, test_targets = load_npz(logits_dir, run, "test")

        # uncalibrated test metrics
        pre, pre_bins = all_metrics(test_probs, test_targets, args.n_bins)

        # fit T on val, apply to test
        T, val_nll = fit_temperature(val_logits, val_targets)
        test_probs_ts = _softmax(test_logits, T)
        post, post_bins = all_metrics(test_probs_ts, test_targets, args.n_bins)

        rows.append({
            "run": run, "temperature": round(T, 4),
            "ece": round(pre["ece"], 4), "ece_ts": round(post["ece"], 4),
            "mce": round(pre["mce"], 4), "mce_ts": round(post["mce"], 4),
            "brier": round(pre["brier"], 4), "brier_ts": round(post["brier"], 4),
            "nll": round(pre["nll"], 4), "nll_ts": round(post["nll"], 4),
            "accuracy": round(pre["accuracy"], 4),  # unchanged by T (argmax preserved)
        })

        # save reliability data for plotting (pre and post)
        pd.DataFrame(pre_bins).to_csv(out / "reliability" / f"{run}_test_pre.csv", index=False)
        pd.DataFrame(post_bins).to_csv(out / "reliability" / f"{run}_test_post.csv", index=False)

        print(f"{run}: T={T:.3f} | ECE {pre['ece']:.4f} -> {post['ece']:.4f} "
              f"| Brier {pre['brier']:.4f} -> {post['brier']:.4f} "
              f"| NLL {pre['nll']:.4f} -> {post['nll']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration_summary.csv", index=False)
    (out / "calibration_summary.json").write_text(json.dumps(rows, indent=2))
    print("\nSaved:", out / "calibration_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
