#!/usr/bin/env python
"""Test-time robustness (eval only, no retraining).

Train ONE linear head on CLEAN MSR train features (exactly the linear_probe recipe:
frozen encoder -> global feature, full-batch Adam, 300 steps, no standardization),
then evaluate that SAME head on PERTURBED TEST clips:
  (a) point dropout    -- keep {75,50,25}% of points per frame
  (b) Gaussian jitter  -- add N(0, sigma*cloud_scale) noise, sigma in {0.01,0.02,0.05}
  (c) temporal subsample -- keep {18,12,6} of the 24 frames (uniformly), feed fewer frames
  (d) frame order -- shuffled / reversed (all frames kept, only temporal order destroyed)

Run for the DENSE encoder (runs/dense_msr_point4d/encoder.pt) AND a RANDOM-INIT
encoder of the same arch. Prints a markdown table: clean + each perturbation level,
dense vs random. Story: does dense pretraining degrade more gracefully.

The encoder global feature is mean-pooled over tokens, so it is robust to changing
N (dropout) and T (temporal subsample) without retraining the head.

cloud_scale = per-clip half-extent (max-min).max()/2 -- the same scale the Point4D
encoder normalizes by, so jitter sigma is a fraction of the cloud's size.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn

from cross4d_jepa.config import Config
from cross4d_jepa.engine.common import resolve_device
from cross4d_jepa.engine.linear_probe import build_labeled_splits
from cross4d_jepa.models import build_encoder


# ---- perturbations (operate on one clip [T,N,3]) ----------------------------
def perturb_dropout(clip, keep, rng):
    """Keep a random `keep` fraction of the N points (same kept indices across frames)."""
    T, N, _ = clip.shape
    k = max(64, int(round(N * keep)))            # >=64 so FPS(256)->KNN stays well-posed
    idx = torch.as_tensor(rng.choice(N, k, replace=False))
    return clip[:, idx, :]


def perturb_jitter(clip, sigma, rng):
    """Add Gaussian noise std = sigma * cloud_scale (per-clip half-extent)."""
    flat = clip.reshape(-1, 3)
    scale = (flat.max(0).values - flat.min(0).values).max().item() / 2 + 1e-6
    noise = torch.as_tensor(rng.standard_normal(clip.shape) * (sigma * scale), dtype=clip.dtype)
    return clip + noise


def perturb_shuffle(clip, rng):
    """Randomly permute the FRAME ORDER, keeping every frame and every point.

    Each frame's geometry is untouched, so any per-frame (static) cue survives
    intact and only temporal coherence is destroyed. A representation built purely
    by pooling static per-frame features is invariant to this; a drop therefore
    measures how much genuinely TEMPORAL structure the encoder learned from a
    per-frame teacher."""
    T = clip.shape[0]
    return clip[torch.as_tensor(rng.permutation(T))]


def perturb_reverse(clip, rng):
    """Reverse the frame order -- a structured, deterministic temporal corruption."""
    return clip.flip(0)


def perturb_temporal(clip, n_frames, rng):
    """Keep n_frames uniformly-spaced frames (fewer frames fed to the encoder)."""
    T = clip.shape[0]
    sel = np.linspace(0, T - 1, n_frames).round().astype(int)
    return clip[torch.as_tensor(sel)]


# ---- feature extraction with an optional per-clip transform -----------------
@torch.no_grad()
def extract(encoder, ds, device, transform=None, rng=None):
    feats, labels = [], []
    for i in range(len(ds)):
        it = ds[i]
        clip = it["points"]                                   # [T,N,3]
        if transform is not None:
            clip = transform(clip, rng)
        g = encoder(clip[None].to(device))["global"]          # [1,D]
        feats.append(g.cpu())
        labels.append(it["label"].view(1))
    return torch.cat(feats), torch.cat(labels)


def train_head(fx_tr, y_tr, num_classes, seed, device, epochs=300, lr=1e-2):
    """Same recipe as engine.linear_probe.linear_probe (no standardization)."""
    torch.manual_seed(seed)
    head = nn.Linear(fx_tr.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    fx_tr, y_tr = fx_tr.to(device), y_tr.to(device)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        lossf(head(fx_tr), y_tr).backward()
        opt.step()
    return head.eval()


@torch.no_grad()
def evaluate(head, fx, y, device):
    return (head(fx.to(device)).argmax(1) == y.to(device)).float().mean().item()


def run_encoder(tag, encoder, cfg, train_ds, val_ds, device, seed):
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # clean train features -> head (trained once)
    fx_tr, y_tr = extract(encoder, train_ds, device)
    fx_va_clean, y_va = extract(encoder, val_ds, device)
    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    head = train_head(fx_tr, y_tr, num_classes, seed, device)

    rows = {}
    rows["clean"] = evaluate(head, fx_va_clean, y_va, device)

    # perturbed test features (same head). Fixed seed per level for reproducibility.
    specs = (
        [(f"dropout {int(p*100)}%", lambda c, r, p=p: perturb_dropout(c, p, r)) for p in (0.75, 0.50, 0.25)]
        + [(f"jitter s={s}", lambda c, r, s=s: perturb_jitter(c, s, r)) for s in (0.01, 0.02, 0.05)]
        + [(f"temporal {nf}f", lambda c, r, nf=nf: perturb_temporal(c, nf, r)) for nf in (18, 12, 6)]
        + [("frames shuffled", perturb_shuffle), ("frames reversed", perturb_reverse)]
    )
    for name, fn in specs:
        rng = np.random.default_rng(seed)
        fx_va, _ = extract(encoder, val_ds, device, transform=fn, rng=rng)
        rows[name] = evaluate(head, fx_va, y_va, device)

    print(f"[{tag}] num_classes={num_classes} chance={1.0/num_classes:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Test-time robustness (dense vs random)")
    ap.add_argument("config", nargs="?", default="configs/msr_point4d.yaml")
    ap.add_argument("--encoder", default="runs/dense_msr_point4d/encoder.pt",
                    help="dense encoder.pt to probe")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = resolve_device(cfg.device)
    train_ds, val_ds = build_labeled_splits(cfg)

    # dense encoder (from ckpt)
    dense = build_encoder(cfg.encoder)
    state = torch.load(args.encoder, map_location="cpu")
    dense.load_state_dict(state["encoder"])
    dense_rows = run_encoder("dense", dense, cfg, train_ds, val_ds, device, cfg.seed)

    # random-init encoder (same arch)
    torch.manual_seed(cfg.seed)
    rand = build_encoder(cfg.encoder)
    rand_rows = run_encoder("random", rand, cfg, train_ds, val_ds, device, cfg.seed)

    # table
    levels = list(dense_rows.keys())
    print("\n| perturbation | dense acc | random acc |")
    print("|---|---|---|")
    for lv in levels:
        print(f"| {lv} | {dense_rows[lv]:.4f} | {rand_rows[lv]:.4f} |")


if __name__ == "__main__":
    main()
