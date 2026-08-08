"""Train a TrustSkin baseline model from a YAML config.

Usage:
    python -m src.training.train --config configs/efficientnet.yaml
    python -m src.training.train --config configs/efficientnet.yaml --seed 43

Model selection: best validation macro-F1. The best checkpoint, the config,
and a per-epoch history CSV are saved under results/checkpoints/<run_name>/.
"""

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataset import HAM10000Dataset, class_counts
from src.models.factory import create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, scaler=None, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y, _ in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train:
                optimizer.zero_grad(set_to_none=True)
            use_amp = scaler is not None and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * x.size(0)
            all_preds.append(logits.argmax(1).detach().cpu())
            all_targets.append(y.detach().cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    macro_f1 = f1_score(targets, preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="Overrides seed in config")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    # YAML 1.1 parses '1e-3' (no dot) as a string; cast numeric fields defensively
    for k in ("lr", "weight_decay", "label_smoothing", "drop_rate"):
        if k in cfg:
            cfg[k] = float(cfg[k])
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{cfg['model']}_seed{seed}"
    out_dir = Path(cfg.get("out_dir", "results/checkpoints")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_size = cfg.get("image_size", 224)
    train_ds = HAM10000Dataset(cfg["train_csv"], image_size, train=True)
    val_ds = HAM10000Dataset(cfg["val_csv"], image_size, train=False)
    nw = cfg.get("num_workers", 4)
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=nw, pin_memory=True, drop_last=True)
    val_ld = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=nw, pin_memory=True)

    model = create_model(cfg["model"], num_classes=cfg.get("num_classes", 7),
                         pretrained=cfg.get("pretrained", True),
                         drop_rate=cfg.get("drop_rate", 0.3)).to(device)

    if cfg.get("class_weighted_loss", True):
        counts = class_counts(cfg["train_csv"], cfg.get("num_classes", 7))
        weights = (counts.sum() / (len(counts) * counts.clamp(min=1))).to(device)
    else:
        weights = None
    criterion = nn.CrossEntropyLoss(weight=weights,
                                    label_smoothing=cfg.get("label_smoothing", 0.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg.get("weight_decay", 0.05))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    best_f1, best_state, patience_left = -1.0, None, cfg.get("early_stop_patience", 7)
    history_path = out_dir / "history.csv"
    with history_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_macro_f1", "val_loss", "val_macro_f1", "lr", "seconds"])
        for epoch in range(1, cfg["epochs"] + 1):
            t0 = time.time()
            tr_loss, tr_f1 = run_epoch(model, train_ld, criterion, optimizer, device, scaler, train=True)
            va_loss, va_f1 = run_epoch(model, val_ld, criterion, optimizer, device, scaler, train=False)
            lr_now = scheduler.get_last_lr()[0]
            scheduler.step()
            dt = time.time() - t0
            writer.writerow([epoch, f"{tr_loss:.4f}", f"{tr_f1:.4f}", f"{va_loss:.4f}", f"{va_f1:.4f}", f"{lr_now:.2e}", f"{dt:.1f}"])
            f.flush()
            print(f"epoch {epoch:03d} | train loss {tr_loss:.4f} f1 {tr_f1:.4f} | val loss {va_loss:.4f} f1 {va_f1:.4f} | {dt:.1f}s")
            if va_f1 > best_f1:
                best_f1, best_state, patience_left = va_f1, copy.deepcopy(model.state_dict()), cfg.get("early_stop_patience", 7)
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"Early stopping at epoch {epoch} (best val macro-F1 {best_f1:.4f})")
                    break

    torch.save({"model": cfg["model"], "seed": seed, "state_dict": best_state,
                "val_macro_f1": best_f1, "config": cfg}, out_dir / "best.pt")
    (out_dir / "config_used.yaml").write_text(yaml.safe_dump({**cfg, "seed": seed}))
    (out_dir / "env.json").write_text(json.dumps({
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }, indent=2))
    print(f"Saved best checkpoint (val macro-F1 {best_f1:.4f}) to {out_dir}/best.pt")


if __name__ == "__main__":
    main()
