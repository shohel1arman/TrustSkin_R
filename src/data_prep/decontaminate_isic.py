"""Phase 7B - de-contaminate ISIC 2019 into a clean external test set.

Removes HAM10000-origin images (models trained on HAM10000), drops SCC + UNK,
maps the remaining 7 classes to the training label scheme, and writes a split
CSV compatible with the existing inference / calibration / uncertainty code.
"""
import argparse
from pathlib import Path

import pandas as pd

from src.data.prepare_splits import CLASSES

# ISIC 2019 one-hot column name -> our dx name (AK -> akiec is the key rename)
ISIC_TO_DX = {
    "MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec",
    "BKL": "bkl", "DF": "df", "VASC": "vasc",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--isic-dir", default="data/isic2019")
    ap.add_argument("--ham-meta", default="data/ham10000/HAM10000_metadata.csv")
    ap.add_argument("--out", default="data/splits/external_isic.csv")
    args = ap.parse_args()

    isic_dir = Path(args.isic_dir)
    gt = pd.read_csv(isic_dir / "ISIC_2019_Training_GroundTruth.csv")
    onehot = [c for c in gt.columns if c != "image"]
    gt["isic_class"] = gt[onehot].idxmax(axis=1)  # single label from one-hot
    n_total = len(gt)

    # de-contamination: drop any image whose ID is a HAM10000 image
    ham = pd.read_csv(args.ham_meta)
    id_col = "image_id" if "image_id" in ham.columns else "image"
    ham_ids = set(ham[id_col].astype(str))
    in_ham = gt["image"].astype(str).isin(ham_ids)
    n_ham = int(in_ham.sum())
    clean = gt[~in_ham].copy()

    # drop SCC + UNK, keep only the 7 shared classes
    n_before_class = len(clean)
    clean = clean[clean["isic_class"].isin(ISIC_TO_DX)].copy()
    n_dropped_class = n_before_class - len(clean)

    # map to the exact training label indices
    clean["dx"] = clean["isic_class"].map(ISIC_TO_DX)
    label_of = {name: i for i, name in enumerate(CLASSES)}
    clean["label"] = clean["dx"].map(label_of)

    # construct + verify paths (images live in class-named folders)
    clean["path"] = clean.apply(
        lambda r: str(isic_dir / r["isic_class"] / f"{r['image']}.jpg"), axis=1)
    exists = clean["path"].apply(lambda p: Path(p).exists())
    n_missing = int((~exists).sum())
    clean = clean[exists].copy()

    out = clean.rename(columns={"image": "image_id"})[
        ["image_id", "dx", "label", "path"]]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print("=== ISIC 2019 -> clean external set ===")
    print(f"total ISIC 2019 images      : {n_total}")
    print(f"removed (HAM10000-origin)   : {n_ham}")
    print(f"removed (SCC/UNK class)     : {n_dropped_class}")
    print(f"removed (image not on disk) : {n_missing}")
    print(f"FINAL external set size     : {len(out)}")
    print("\nper-class counts:")
    print(out["dx"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
