"""Phase 7b - Learned Trust Score (nested CV) + trust-profile analysis."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.trust.trust_score import load_components, _minmax, error_auroc


def build_full_frame(run, split, unc_dir, xai_dir, edas_dir, method):
    base, loc, edas = load_components(run, split, unc_dir, xai_dir, edas_dir, method)
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
    return full


def learned_combiner_oof(df, components, seed=42, n_splits=5):
    X = df[components].to_numpy()
    y = df["correct"].to_numpy()
    oof = np.zeros(len(df))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    weights = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        weights.append(clf.coef_.ravel())
    err = (1 - y).astype(int)
    auroc = float(roc_auc_score(err, -oof)) if 0 < err.sum() < len(err) else float("nan")
    return auroc, np.mean(weights, axis=0)


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

    full = build_full_frame(args.run, args.split, Path(args.unc_dir),
                            Path(args.xai_dir), Path(args.edas_dir), args.method)
    have = ("loc_score" in full.columns) and ("edas_n" in full.columns)
    components = ["conf_n", "certainty"] + (["loc_score", "edas_n"] if have else [])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    auroc_learned, w = learned_combiner_oof(full, components)
    auroc_conf = float(error_auroc(full["correct"].to_numpy(), full["conf_n"].to_numpy()))

    corr_with_correct = {c: round(float(np.corrcoef(full[c], full["correct"])[0, 1]), 4)
                         for c in components}
    inter = full[components].corr().round(3)

    result = {
        "run": args.run, "n": int(len(full)), "components": components,
        "auroc_confidence": round(auroc_conf, 4),
        "auroc_learned_combiner_oof": round(auroc_learned, 4),
        "learned_beats_confidence": bool(auroc_learned > auroc_conf),
        "learned_weights": {c: round(float(wi), 4) for c, wi in zip(components, w)},
        "component_corr_with_correct": corr_with_correct,
    }
    (out / f"{args.run}_{args.split}_learned_trust.json").write_text(json.dumps(result, indent=2))
    inter.to_csv(out / f"{args.run}_{args.split}_component_correlation.csv")

    print(f"\n=== Learned Trust Score (OOF CV): {args.run} ===")
    print(f"n={len(full)}  components={components}")
    print(f"AUROC confidence        : {auroc_conf:.4f}")
    print(f"AUROC learned (OOF)     : {auroc_learned:.4f}  "
          f"({'BEATS' if auroc_learned > auroc_conf else 'does NOT beat'} confidence)")
    print(f"learned weights         : "
          + ", ".join(f"{c}={wi:.3f}" for c, wi in zip(components, w)))
    print(f"component corr w/correct: {corr_with_correct}")
    print("\ninter-component correlation:")
    print(inter.to_string())
    print(f"\nSaved under {out}/")


if __name__ == "__main__":
    main()
