"""Phase 5a - Prediction robustness under perturbations."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.dataset import build_transforms
from src.data.prepare_splits import CLASSES
from src.models.factory import create_model
from src.robustness.perturbations import FAMILIES, apply

EPS = 1e-12


def js_divergence(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        return np.sum(a * (np.log(a + EPS) - np.log(b + EPS)), axis=-1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def stratified_subset(df, n_per_class, seed=42):
    parts = []
    for c in range(len(CLASSES)):
        sub = df[df["label"] == c]
        if len(sub):
            parts.append(sub.sample(n=min(n_per_class, len(sub)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


@torch.no_grad()
def probs_for(model, imgs, tfm, device, batch=64):
    out = []
    for i in range(0, len(imgs), batch):
        xb = torch.stack([tfm(im) for im in imgs[i:i + batch]]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--n-per-class", type=int, default=70)
    ap.add_argument("--out-dir", default="results/robustness")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    run = f"{ckpt['model']}_seed{ckpt['seed']}"
    model = create_model(ckpt["model"], num_classes=cfg.get("num_classes", 7),
                         pretrained=False, drop_rate=cfg.get("drop_rate", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    size = cfg.get("image_size", 224)
    tfm = build_transforms(size, train=False)

    df = pd.read_csv(args.split_csv)
    sub = stratified_subset(df, args.n_per_class, args.seed)
    targets = sub["label"].to_numpy()
    print(f"{run}: robustness on {len(sub)} images x {len(FAMILIES)} families x 5 severities")

    clean_imgs = [Image.open(p).convert("RGB") for p in sub["path"]]
    clean_probs = probs_for(model, clean_imgs, tfm, device)
    clean_pred = clean_probs.argmax(1)
    clean_acc = float((clean_pred == targets).mean())

    rows = [{"run": run, "family": "clean", "severity": 0, "accuracy": round(clean_acc, 4),
             "consistency": 1.0, "mean_confidence": round(float(clean_probs.max(1).mean()), 4),
             "js_divergence": 0.0}]

    for fam in FAMILIES:
        for sev in range(1, 6):
            pert = [apply(im, fam, sev, seed=args.seed + i) for i, im in enumerate(clean_imgs)]
            pprobs = probs_for(model, pert, tfm, device)
            ppred = pprobs.argmax(1)
            acc = float((ppred == targets).mean())
            consistency = float((ppred == clean_pred).mean())
            conf = float(pprobs.max(1).mean())
            js = float(js_divergence(clean_probs, pprobs).mean())
            rows.append({"run": run, "family": fam, "severity": sev,
                         "accuracy": round(acc, 4), "consistency": round(consistency, 4),
                         "mean_confidence": round(conf, 4), "js_divergence": round(js, 4)})
        print(f"  done {fam}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out / f"{run}_{args.split_name}_prediction_robustness.csv", index=False)

    summ = (res[res.severity > 0].groupby("family")
            .agg(mean_acc=("accuracy", "mean"),
                 mean_consistency=("consistency", "mean"),
                 acc_at_sev5=("accuracy", lambda s: s.iloc[-1]))
            .reset_index())
    summ["clean_acc"] = clean_acc
    summ["acc_drop_sev5"] = (clean_acc - summ["acc_at_sev5"]).round(4)
    summ.to_csv(out / f"{run}_{args.split_name}_robustness_summary.csv", index=False)

    meta = {"run": run, "n": len(sub), "clean_accuracy": round(clean_acc, 4),
            "families": FAMILIES, "device": str(device)}
    (out / f"{run}_{args.split_name}_robustness_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nclean acc {clean_acc:.4f}")
    print(summ.to_string(index=False))
    print(f"Saved under {out}/")


if __name__ == "__main__":
    main()
