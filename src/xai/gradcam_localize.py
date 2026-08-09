"""Phase 4 - Explainability & localization (Grad-CAM vs lesion masks)."""
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

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def target_layer_and_reshape(model, arch):
    if arch == "efficientnet_b3":
        return [model.conv_head], None
    if arch == "convnext_tiny":
        return [model.stages[-1].blocks[-1].norm], None
    if arch == "vit_b16":
        def reshape(tensor, h=14, w=14):
            r = tensor[:, 1:, :].reshape(tensor.size(0), h, w, tensor.size(2))
            return r.permute(0, 3, 1, 2)
        return [model.blocks[-1].norm1], reshape
    if arch == "swin_tiny":
        # swin norm1 output is already spatial: (B, H, W, C) -> (B, C, H, W)
        def reshape(tensor):
            if tensor.dim() == 4:
                return tensor.permute(0, 3, 1, 2)
            n = tensor.size(1); s = int(round(n ** 0.5))
            return tensor.reshape(tensor.size(0), s, s, tensor.size(2)).permute(0, 3, 1, 2)
        return [model.layers[-1].blocks[-1].norm1], reshape
    raise ValueError(f"no gradcam target defined for {arch}")


def stratified_subset(df, n_per_class, seed=42):
    parts = []
    for c in range(len(CLASSES)):
        sub = df[df["label"] == c]
        if len(sub) == 0:
            continue
        take = min(n_per_class, len(sub))
        parts.append(sub.sample(n=take, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def load_mask(mask_dir, image_id, size):
    p = mask_dir / f"{image_id}_segmentation.png"
    if not p.exists():
        return None
    m = Image.open(p).convert("L").resize((size, size), Image.NEAREST)
    return (np.array(m) > 127).astype(np.uint8)


def localization_metrics(cam, mask, thr=0.5):
    # absolute-threshold IoU/Dice (kept for reference)
    sal = (cam >= thr).astype(np.uint8)
    inter = (sal & mask).sum(); union = (sal | mask).sum()
    iou = inter / union if union > 0 else 0.0
    dice = 2 * inter / (sal.sum() + mask.sum()) if (sal.sum() + mask.sum()) > 0 else 0.0
    # percentile-threshold IoU: top-k% of CAM energy, k matched to lesion size band.
    # Use top-20% most-activated pixels (standard for peaked CAMs like Grad-CAM++).
    k = np.quantile(cam, 0.80)
    salp = (cam >= k).astype(np.uint8)
    interp = (salp & mask).sum(); unionp = (salp | mask).sum()
    iou_p = interp / unionp if unionp > 0 else 0.0
    dice_p = 2 * interp / (salp.sum() + mask.sum()) if (salp.sum() + mask.sum()) > 0 else 0.0
    # threshold-free signals
    peak = np.unravel_index(np.argmax(cam), cam.shape)
    pointing = int(mask[peak] == 1)
    total = cam.sum() + 1e-12
    lesion_frac = float((cam * mask).sum() / total)
    # concentration ratio: mean CAM inside lesion / mean CAM outside (>1 = lesion-focused)
    in_mean = (cam * mask).sum() / (mask.sum() + 1e-12)
    out_mean = (cam * (1 - mask)).sum() / ((1 - mask).sum() + 1e-12)
    focus_ratio = float(min(in_mean / (out_mean + 1e-6), 50.0))  # cap: near-zero bg attention blows up the raw ratio
    return {"iou": float(iou), "dice": float(dice),
            "iou_p20": float(iou_p), "dice_p20": float(dice_p),
            "pointing": pointing, "lesion_frac": lesion_frac,
            "bg_frac": float(1 - lesion_frac), "focus_ratio": focus_ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--mask-dir", required=True)
    ap.add_argument("--method", choices=["gradcam", "gradcam++"], default="gradcam++")
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out-dir", default="results/xai")
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
    mask_dir = Path(args.mask_dir)

    df = pd.read_csv(args.split_csv)
    if args.limit >= 0:
        sub = df if args.limit == 0 else df.iloc[:args.limit].reset_index(drop=True)
    else:
        sub = stratified_subset(df, args.n_per_class, args.seed)
    print(f"{run}: localization on {len(sub)} images ({args.method})")

    layers, reshape = target_layer_and_reshape(model, arch)
    CAM = GradCAMPlusPlus if args.method == "gradcam++" else GradCAM
    cam_algo = CAM(model=model, target_layers=layers, reshape_transform=reshape)

    rows = []
    missing_mask = 0
    for i, r in sub.iterrows():
        img = Image.open(r["path"]).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        mask = load_mask(mask_dir, r["image_id"], size)
        if mask is None:
            missing_mask += 1
            continue
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        grayscale_cam = cam_algo(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        m = localization_metrics(grayscale_cam, mask, args.thr)
        m.update({"image_id": r["image_id"], "label": int(r["label"]),
                  "dx": CLASSES[int(r["label"])], "pred": pred,
                  "correct": int(pred == int(r["label"]))})
        rows.append(m)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sub)}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / f"{run}_{args.split_name}_{args.method}_localization.csv", index=False)

    agg = {
        "run": run, "method": args.method, "n": len(per), "missing_mask": missing_mask,
        "mean_iou": round(per["iou"].mean(), 4),
        "mean_iou_p20": round(per["iou_p20"].mean(), 4),
        "mean_dice": round(per["dice"].mean(), 4),
        "mean_focus_ratio": round(per["focus_ratio"].mean(), 4),
        "pointing_game": round(per["pointing"].mean(), 4),
        "mean_lesion_frac": round(per["lesion_frac"].mean(), 4),
        "mean_bg_frac": round(per["bg_frac"].mean(), 4),
        "iou_correct": round(per[per["correct"] == 1]["iou"].mean(), 4),
        "iou_wrong": round(per[per["correct"] == 0]["iou"].mean(), 4) if (per["correct"] == 0).any() else None,
    }
    (out / f"{run}_{args.split_name}_{args.method}_summary.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))
    print(f"\nSaved under {out}/")


if __name__ == "__main__":
    main()
