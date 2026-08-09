"""Phase 5b - Explanation stability (Grad-CAM clean vs perturbed)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim

from src.data.dataset import build_transforms
from src.data.prepare_splits import CLASSES
from src.models.factory import create_model
from src.robustness.perturbations import apply
from src.xai.gradcam_localize import target_layer_and_reshape

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

EPS = 1e-12


def stratified_subset(df, n_per_class, seed=42):
    parts = []
    for c in range(len(CLASSES)):
        s = df[df["label"] == c]
        if len(s):
            parts.append(s.sample(n=min(n_per_class, len(s)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def map_similarity(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    s = float(ssim(a, b, data_range=1.0))
    af, bf = a.ravel(), b.ravel()
    cos = float((af @ bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + EPS))
    # guard against (near-)constant maps, which make rank correlation undefined
    if af.std() < 1e-6 or bf.std() < 1e-6:
        rho = float("nan")
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho = spearmanr(af, bf).correlation
        rho = float("nan") if rho is None or np.isnan(rho) else float(rho)
    ka = np.quantile(a, 0.8)
    kb = np.quantile(b, 0.8)
    sa, sb = (a >= ka).astype(np.uint8), (b >= kb).astype(np.uint8)
    inter = (sa & sb).sum()
    union = (sa | sb).sum()
    iou = float(inter / union) if union > 0 else 0.0
    return {"ssim": s, "cosine": cos, "spearman": rho, "iou_p20": iou}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--families", nargs="+",
                    default=["gaussian_noise", "gaussian_blur", "brightness"])
    ap.add_argument("--severities", nargs="+", type=int, default=[2, 4])
    ap.add_argument("--out-dir", default="results/robustness")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    arch = ckpt["model"]
    run = f"{arch}_seed{ckpt['seed']}"
    model = create_model(arch, num_classes=cfg.get("num_classes", 7),
                         pretrained=False, drop_rate=cfg.get("drop_rate", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    size = cfg.get("image_size", 224)
    tfm = build_transforms(size, train=False)
    layers, reshape = target_layer_and_reshape(model, arch)
    cam = GradCAMPlusPlus(model=model, target_layers=layers, reshape_transform=reshape)

    df = pd.read_csv(args.split_csv)
    sub = stratified_subset(df, args.n_per_class, args.seed)
    print(f"{run}: explanation stability on {len(sub)} imgs, "
          f"{len(args.families)} families, severities {args.severities}")

    def cam_of(pil):
        x = tfm(pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0], pred

    rows = []
    for i, r in sub.iterrows():
        clean = Image.open(r["path"]).convert("RGB")
        clean_cam, clean_pred = cam_of(clean)
        for fam in args.families:
            for sev in args.severities:
                pert = apply(clean, fam, sev, seed=args.seed + i)
                pcam, ppred = cam_of(pert)
                sim = map_similarity(clean_cam, pcam)
                sim.update({"run": run, "image_id": r["image_id"], "family": fam,
                            "severity": sev, "pred_unchanged": int(ppred == clean_pred)})
                rows.append(sim)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(sub)}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / f"{run}_{args.split_name}_explanation_stability.csv", index=False)

    summ = (per.groupby("family")
            .agg(ssim=("ssim", "mean"), cosine=("cosine", "mean"),
                 spearman=("spearman", "mean"), iou_p20=("iou_p20", "mean"),
                 pred_unchanged=("pred_unchanged", "mean"))
            .round(4).reset_index())
    summ.to_csv(out / f"{run}_{args.split_name}_explanation_stability_summary.csv", index=False)

    overall = {"run": run, "n_images": len(sub),
               "mean_ssim": round(float(per["ssim"].mean()), 4),
               "mean_cosine": round(float(per["cosine"].mean()), 4),  # cosine defined even for constant maps
               "mean_spearman": round(float(per["spearman"].dropna().mean()), 4),
               "mean_iou_p20": round(float(per["iou_p20"].mean()), 4)}
    (out / f"{run}_{args.split_name}_explanation_stability_overall.json").write_text(
        json.dumps(overall, indent=2))

    print(json.dumps(overall, indent=2))
    print(summ.to_string(index=False))
    print(f"Saved under {out}/")


if __name__ == "__main__":
    main()
