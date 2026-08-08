"""Create leakage-controlled stratified splits for HAM10000.

HAM10000 contains multiple images of the same lesion (lesion_id). If images of
one lesion appear in both train and test, metrics are inflated by leakage.
This script groups by lesion_id and stratifies (approximately) by class using
StratifiedGroupKFold, producing train / val / test CSVs.

Usage:
    python -m src.data.prepare_splits \
        --metadata data/ham10000/HAM10000_metadata.csv \
        --image-dirs data/ham10000/HAM10000_images_part_1 data/ham10000/HAM10000_images_part_2 \
        --out-dir data/splits --seed 42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]


def find_image_path(image_id: str, image_dirs: list[Path]) -> str | None:
    for d in image_dirs:
        p = d / f"{image_id}.jpg"
        if p.exists():
            return str(p)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="Path to HAM10000_metadata.csv")
    ap.add_argument("--image-dirs", nargs="+", required=True, help="Directories containing the .jpg images")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-folds", type=int, default=5, help="1/N of lesions go to test (5 -> 20%)")
    ap.add_argument("--val-folds", type=int, default=8, help="1/N of remaining lesions go to val (8 -> 10% overall)")
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    required = {"image_id", "lesion_id", "dx"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Metadata file is missing columns: {missing}")

    image_dirs = [Path(d) for d in args.image_dirs]
    df["path"] = df["image_id"].map(lambda i: find_image_path(i, image_dirs))
    n_missing = int(df["path"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} images listed in metadata were not found on disk; they are dropped.")
        df = df.dropna(subset=["path"]).reset_index(drop=True)

    df["label"] = df["dx"].map({c: i for i, c in enumerate(CLASSES)})
    if df["label"].isna().any():
        bad = sorted(df.loc[df["label"].isna(), "dx"].unique())
        raise SystemExit(f"Unknown dx values in metadata: {bad}")
    df["label"] = df["label"].astype(int)

    # ---- stage 1: split off the test set, grouped by lesion_id ----
    sgkf = StratifiedGroupKFold(n_splits=args.test_folds, shuffle=True, random_state=args.seed)
    trainval_idx, test_idx = next(sgkf.split(df, df["label"], groups=df["lesion_id"]))
    test = df.iloc[test_idx].reset_index(drop=True)
    trainval = df.iloc[trainval_idx].reset_index(drop=True)

    # ---- stage 2: split train / val from the remainder, grouped by lesion_id ----
    sgkf2 = StratifiedGroupKFold(n_splits=args.val_folds, shuffle=True, random_state=args.seed)
    train_idx, val_idx = next(sgkf2.split(trainval, trainval["label"], groups=trainval["lesion_id"]))
    train = trainval.iloc[train_idx].reset_index(drop=True)
    val = trainval.iloc[val_idx].reset_index(drop=True)

    # ---- leakage check: no lesion_id may appear in two splits ----
    s_train, s_val, s_test = set(train.lesion_id), set(val.lesion_id), set(test.lesion_id)
    assert not (s_train & s_val), "lesion leakage between train and val"
    assert not (s_train & s_test), "lesion leakage between train and test"
    assert not (s_val & s_test), "lesion leakage between val and test"

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "lesion_id", "dx", "label", "path"]
    train[cols].to_csv(out / "train.csv", index=False)
    val[cols].to_csv(out / "val.csv", index=False)
    test[cols].to_csv(out / "test.csv", index=False)

    summary = {
        "seed": args.seed,
        "classes": CLASSES,
        "n_images": {"train": len(train), "val": len(val), "test": len(test)},
        "n_lesions": {"train": len(s_train), "val": len(s_val), "test": len(s_test)},
        "class_counts": {
            split: {CLASSES[k]: int(v) for k, v in d["dx"].map({c: i for i, c in enumerate(CLASSES)}).value_counts().sort_index().items()}
            for split, d in [("train", train), ("val", val), ("test", test)]
        },
    }
    (out / "split_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSplits written to {out}/  (train.csv, val.csv, test.csv, split_summary.json)")


if __name__ == "__main__":
    main()
