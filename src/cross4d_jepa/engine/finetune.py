"""End-to-end supervised training -- the fine-tuning protocol (label efficiency and
full-label fine-tuning), as opposed to the frozen-encoder linear probe.

  * encoder_ckpt=None  -> FROM-SCRATCH supervised (does our backbone+recipe reach
    the field's level at all?).
  * encoder_ckpt=<ssl> -> fine-tune our pretrained encoder (does SSL pretraining
    help the standard fine-tune number?).
  * label_fraction<1.0 -> semi-supervised (Uni4D 50%-labels protocol).

Competitive recipe: per-clip point-cloud augmentation (scale + gravity-axis
rotation + jitter), warmup+cosine LR, label smoothing. Recognition only
(global clip feature -> linear head -> cross-entropy).
"""
import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..config import Config
from ..models import build_encoder
from ..utils import get_logger
from .common import resolve_device
from .linear_probe import build_labeled_splits

log = get_logger()


def _num_classes(ds) -> int:
    return max(int(ds[i]["label"]) for i in range(len(ds))) + 1


def _augment(pts: torch.Tensor) -> torch.Tensor:
    """Per-clip point-cloud augmentation on [B,T,N,3] (consistent across frames so
    motion is preserved): random uniform scale, gravity (z) rotation, point jitter."""
    B, dev = pts.shape[0], pts.device
    s = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
    ang = torch.empty(B, device=dev).uniform_(-0.31, 0.31)            # ~+/-18 deg about z
    cos, sin = torch.cos(ang), torch.sin(ang)
    R = torch.zeros(B, 3, 3, device=dev)
    R[:, 2, 2] = 1.0
    R[:, 0, 0] = cos; R[:, 0, 1] = -sin; R[:, 1, 0] = sin; R[:, 1, 1] = cos
    pts = torch.einsum("btnc,bdc->btnd", pts * s, R)                  # scale then rotate
    return pts + torch.randn_like(pts) * 0.01                          # jitter


def finetune(
    cfg: Config,
    encoder_ckpt: Optional[str] = None,
    label_fraction: float = 1.0,
    epochs: int = 150,
    lr: float = 1e-3,
    weight_decay: float = 5e-2,
    augment: bool = True,
) -> float:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    encoder = build_encoder(cfg.encoder).to(device)
    if encoder_ckpt:
        encoder.load_state_dict(torch.load(encoder_ckpt, map_location="cpu")["encoder"])

    train_ds, val_ds = build_labeled_splits(cfg)
    num_classes = max(_num_classes(train_ds), _num_classes(val_ds))
    if label_fraction < 1.0:
        n = len(train_ds)
        g = torch.Generator().manual_seed(cfg.seed)
        idx = torch.randperm(n, generator=g)[: max(1, int(round(n * label_fraction)))].tolist()
        train_ds = Subset(train_ds, idx)

    tl = DataLoader(train_ds, batch_size=cfg.optim.batch_size, shuffle=True,
                    num_workers=cfg.optim.num_workers, drop_last=True)
    vl = DataLoader(val_ds, batch_size=cfg.optim.batch_size, num_workers=cfg.optim.num_workers)

    head = nn.Linear(encoder.embed_dim, num_classes).to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=lr, weight_decay=weight_decay)
    warm = max(1, int(0.05 * epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / warm if ep < warm
        else 0.5 * (1 + math.cos(math.pi * (ep - warm) / max(1, epochs - warm))))
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for ep in range(epochs):
        encoder.train(); head.train()
        for batch in tl:
            pts = batch["points"].to(device, non_blocking=True)
            if augment:
                pts = _augment(pts)
            y = batch["label"].to(device)
            opt.zero_grad(set_to_none=True)
            loss = lossf(head(encoder(pts)["global"]), y)
            if torch.isfinite(loss):
                loss.backward(); opt.step()
        sched.step()
        if ep % 10 == 0 or ep == epochs - 1:
            encoder.eval(); head.eval()
            correct = total = 0
            with torch.no_grad():
                for batch in vl:
                    logits = head(encoder(batch["points"].to(device))["global"])
                    correct += (logits.argmax(1).cpu() == batch["label"]).sum().item()
                    total += batch["label"].numel()
            acc = correct / max(total, 1)
            best = max(best, acc)
            log.info(f"  ep {ep:03d} val {acc:.4f} (best {best:.4f})")

    kind = "from-scratch" if not encoder_ckpt else "ssl-finetune"
    log.info(f"{kind} (label_frac={label_fraction}, aug={augment}): best={best:.4f} "
             f"(classes={num_classes}, epochs={epochs})")
    return best


def main():  # console entry: c4d-finetune
    import argparse

    ap = argparse.ArgumentParser(description="End-to-end supervised (from-scratch / finetune)")
    ap.add_argument("config")
    ap.add_argument("--encoder", default=None, help="SSL checkpoint to fine-tune; omit for from-scratch")
    ap.add_argument("--label-fraction", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    cfg.seed = int(os.environ.get("C4D_SEED", cfg.seed))   # multi-seed: C4D_SEED overrides
    acc = finetune(cfg, encoder_ckpt=args.encoder,
                   label_fraction=args.label_fraction, epochs=args.epochs,
                   augment=not args.no_augment)
    print(f"RESULT[{args.tag}] task=finetune best={acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
