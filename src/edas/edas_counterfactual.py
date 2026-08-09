"""Phase 6 - Explanation-Decision Alignment Score (EDAS)."""
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
from src.xai.gradcam_localize import target_layer_and_reshape, stratified_subset

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

EPS = 1e-8


def mask_region(x, region_mask, fill_value=0.0):
    out = x.clone()
    rm = torch.from_numpy(region_mask.astype(bool))
    out[:, :, rm] = fill_value
    return out


def _bbox_side(n_pixels, H, W):
    side = int(round(np.sqrt(max(1, n_pixels))))
    return int(np.clip(side, 1, min(H, W)))


def topk_block_mask(cam, frac):
    """Contiguous square block centered on the attribution peak, area ~ frac."""
    H, W = cam.shape
    n_pixels = int(round(frac * H * W))
    side = _bbox_side(n_pixels, H, W)
    py, px = np.unravel_index(np.argmax(cam), cam.shape)
    y0 = int(np.clip(py - side // 2, 0, H - side))
    x0 = int(np.clip(px - side // 2, 0, W - side))
    m = np.zeros((H, W), dtype=bool)
    m[y0:y0 + side, x0:x0 + side] = True
    return m, (y0, x0, side)


def control_block_mask(shape, box, rng, min_dist_frac=0.3):
    """Same-size square block placed elsewhere; only location differs."""
    H, W = shape
    y0, x0, side = box
    best = None
    for _ in range(50):
        ny = int(rng.integers(0, max(1, H - side)))
        nx = int(rng.integers(0, max(1, W - side)))
        dist = np.hypot(ny - y0, nx - x0)
        if dist >= min_dist_frac * min(H, W):
            best = (ny, nx); break
        best = best or (ny, nx)
    ny, nx = best
    m = np.zeros((H, W), dtype=bool)
    m[ny:ny + side, nx:nx + side] = True
    return m


@torch.no_grad()
def conf_of(model, x, cls):
    return float(torch.softmax(model(x), dim=1)[0, cls].item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-csv", required=True)
    ap.add_argument("--split-name", default="test")
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--topk", type=float, default=0.15)
    ap.add_argument("--out-dir", default="results/edas")
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
    rng = np.random.default_rng(args.seed)
    print(f"{run}: EDAS on {len(sub)} images, topk={args.topk}")

    rows = []
    for i, r in sub.iterrows():
        img = Image.open(r["path"]).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = int(model(x).argmax(1).item())
        p0 = conf_of(model, x, pred)

        gcam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred)])[0]
        high, box = topk_block_mask(gcam, args.topk)
        rand = control_block_mask(gcam.shape, box, rng)

        p_high = conf_of(model, mask_region(x, high).to(device), pred)
        p_rand = conf_of(model, mask_region(x, rand).to(device), pred)

        drop_high = p0 - p_high
        drop_rand = p0 - p_rand
        edas = drop_high - drop_rand
        rows.append({
            "image_id": r["image_id"], "label": int(r["label"]),
            "dx": CLASSES[int(r["label"])], "pred": pred,
            "correct": int(pred == int(r["label"])),
            "p0": round(p0, 4), "p_high": round(p_high, 4), "p_rand": round(p_rand, 4),
            "drop_high": round(drop_high, 4), "drop_rand": round(drop_rand, 4),
            "edas": round(edas, 4), "edas_norm": round(edas / (p0 + EPS), 4),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(sub)}", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows)
    per.to_csv(out / f"{run}_{args.split_name}_edas.csv", index=False)

    agg = {
        "run": run, "n": len(per), "topk": args.topk,
        "mean_drop_high": round(float(per["drop_high"].mean()), 4),
        "mean_drop_rand": round(float(per["drop_rand"].mean()), 4),
        "mean_edas": round(float(per["edas"].mean()), 4),
        "mean_edas_norm": round(float(per["edas_norm"].mean()), 4),
        "faithful_fraction": round(float((per["edas"] > 0).mean()), 4),
        "edas_correct": round(float(per[per.correct == 1]["edas"].mean()), 4),
        "edas_wrong": round(float(per[per.correct == 0]["edas"].mean()), 4) if (per.correct == 0).any() else None,
    }
    (out / f"{run}_{args.split_name}_edas_summary.json").write_text(json.dumps(agg, indent=2))
    print(json.dumps(agg, indent=2))
    print(f"\nSaved under {out}/")


if __name__ == "__main__":
    main()
