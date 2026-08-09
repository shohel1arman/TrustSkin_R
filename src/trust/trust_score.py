"""Phase 7 - Trustworthiness Score (TS) + statistical validation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _minmax(s):
    s = np.asarray(s, dtype=float)
    lo, hi = np.nanmin(s), np.nanmax(s)
    if hi - lo < 1e-12:
        return np.zeros_like(s)
    return (s - lo) / (hi - lo)


def load_components(run, split, unc_dir, xai_dir, edas_dir, method="gradcam++"):
    unc = pd.read_csv(unc_dir / "per_image" / f"{run}_{split}.csv")
    base = unc[["image_id", "correct", "confidence", "predictive_entropy"]].copy()
    base["certainty"] = 1.0 - _minmax(base["predictive_entropy"])
    base["conf_n"] = _minmax(base["confidence"])
    loc_f = xai_dir / f"{run}_{split}_{method}_localization.csv"
    edas_f = edas_dir / f"{run}_{split}_edas.csv"
    loc = pd.read_csv(loc_f)[["image_id", "pointing", "focus_ratio"]] if loc_f.exists() else None
    edas = pd.read_csv(edas_f)[["image_id", "edas"]] if edas_f.exists() else None
    return base, loc, edas


def error_auroc(correct, score_trust):
    err = (1 - correct).astype(int)
    if err.sum() == 0 or err.sum() == len(err):
        return np.nan
    return roc_auc_score(err, -score_trust)


def bootstrap_auroc_diff(correct, ts, conf, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(correct)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c = correct[idx]
        if (1 - c).sum() == 0 or (1 - c).sum() == n:
            continue
        a_ts = error_auroc(c, ts[idx])
        a_cf = error_auroc(c, conf[idx])
        diffs.append(a_ts - a_cf)
    diffs = np.array(diffs)
    return (float(diffs.mean()), float(np.percentile(diffs, 2.5)),
            float(np.percentile(diffs, 97.5)), float((diffs > 0).mean()))


def evaluate(name, df, trust_col, out):
    correct = df["correct"].to_numpy()
    ts = df[trust_col].to_numpy()
    conf = df["conf_n"].to_numpy()
    ent_trust = df["certainty"].to_numpy()
    res = {
        "variant": name, "n": int(len(df)),
        "auroc_trust": round(float(error_auroc(correct, ts)), 4),
        "auroc_confidence": round(float(error_auroc(correct, conf)), 4),
        "auroc_certainty": round(float(error_auroc(correct, ent_trust)), 4),
    }
    md, lo, hi, pwin = bootstrap_auroc_diff(correct, ts, conf)
    res.update({"auroc_diff_TS_minus_conf": round(md, 4),
                "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
                "prob_TS_beats_conf": round(pwin, 4)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--unc-dir", default="results/uncertainty")
    ap.add_argument("--xai-dir", default="results/xai")
    ap.add_argument("--edas-dir", default="results/edas")
    ap.add_argument("--method", default="gradcam++")
    ap.add_argument("--out-dir", default="results/trust")
    args = ap.parse_args()

    base, loc, edas = load_components(
        args.run, args.split, Path(args.unc_dir), Path(args.xai_dir),
        Path(args.edas_dir), args.method)

    base["TS_light"] = base[["conf_n", "certainty"]].mean(axis=1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    results.append(evaluate("TS_light_full", base, "TS_light", out))

    full = base.copy()
    if loc is not None:
        loc = loc.copy()
        loc["focus_n"] = _minmax(np.clip(loc["focus_ratio"], 0, 10))
        loc["point_n"] = _minmax(loc["pointing"])
        loc["loc_score"] = loc[["focus_n", "point_n"]].mean(axis=1)
        full = full.merge(loc[["image_id", "loc_score"]], on="image_id", how="inner")
    if edas is not None:
        edas = edas.copy()
        edas["edas_n"] = _minmax(np.clip(edas["edas"], -0.5, 0.5))
        full = full.merge(edas[["image_id", "edas_n"]], on="image_id", how="inner")

    have_full = ("loc_score" in full.columns) and ("edas_n" in full.columns)
    if have_full and len(full) > 20:
        full["TS_full"] = full[["conf_n", "certainty", "loc_score", "edas_n"]].mean(axis=1)
        full.to_csv(out / f"{args.run}_{args.split}_trust_per_image.csv", index=False)
        results.append(evaluate("TS_full_subset", full, "TS_full", out))

    summary = pd.DataFrame(results)
    summary.to_csv(out / f"{args.run}_{args.split}_trust_summary.csv", index=False)
    (out / f"{args.run}_{args.split}_trust_summary.json").write_text(
        json.dumps(results, indent=2))
    print(f"\n=== Trust Score validation: {args.run} ===")
    print(summary.to_string(index=False))
    print(f"\nSaved under {out}/")


if __name__ == "__main__":
    main()
