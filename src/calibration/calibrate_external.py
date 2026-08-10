"""Phase 7B - calibration on the external (ISIC 2019) set: APPLY, do not refit.

Honest OOD calibration test: we do NOT fit a new temperature on external data.
We load the temperature that Phase 2 fit on the HAM validation split
(results/calibration/calibration_summary.csv) and apply it to the external
logits. This asks whether HAM-derived calibration generalizes under
distribution shift. Metric functions are imported from calibrate.py so ECE/
MCE/Brier/NLL are defined identically to the in-distribution numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.calibrate import all_metrics, _softmax, load_npz


def load_ham_temperatures(summary_csv: Path) -> dict:
    df = pd.read_csv(summary_csv)
    return {str(r["run"]): float(r["temperature"]) for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logits-dir", default="results/logits")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split-name", default="external")
    ap.add_argument("--ham-summary", default="results/calibration/calibration_summary.csv")
    ap.add_argument("--out-dir", default="results/calibration_external")
    ap.add_argument("--n-bins", type=int, default=15)
    args = ap.parse_args()

    logits_dir = Path(args.logits_dir)
    out = Path(args.out_dir)
    (out / "reliability").mkdir(parents=True, exist_ok=True)

    temps = load_ham_temperatures(Path(args.ham_summary))

    rows = []
    for run in args.runs:
        if run not in temps:
            raise KeyError(f"no HAM temperature for {run} in {args.ham_summary}")
        T = temps[run]

        logits, probs, targets = load_npz(logits_dir, run, args.split_name)

        pre, pre_bins = all_metrics(probs, targets, args.n_bins)

        probs_ts = _softmax(logits, T)
        post, post_bins = all_metrics(probs_ts, targets, args.n_bins)

        rows.append({
            "run": run, "temperature_ham": round(T, 4),
            "ece": round(pre["ece"], 4), "ece_ts": round(post["ece"], 4),
            "mce": round(pre["mce"], 4), "mce_ts": round(post["mce"], 4),
            "brier": round(pre["brier"], 4), "brier_ts": round(post["brier"], 4),
            "nll": round(pre["nll"], 4), "nll_ts": round(post["nll"], 4),
            "accuracy": round(pre["accuracy"], 4),
        })

        pd.DataFrame(pre_bins).to_csv(out / "reliability" / f"{run}_external_pre.csv", index=False)
        pd.DataFrame(post_bins).to_csv(out / "reliability" / f"{run}_external_post.csv", index=False)

        print(f"{run}: T_ham={T:.3f} | ECE {pre['ece']:.4f} -> {post['ece']:.4f} "
              f"| Brier {pre['brier']:.4f} -> {post['brier']:.4f} "
              f"| NLL {pre['nll']:.4f} -> {post['nll']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration_external_summary.csv", index=False)
    (out / "calibration_external_summary.json").write_text(json.dumps(rows, indent=2))
    print("\nSaved:", out / "calibration_external_summary.csv")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
