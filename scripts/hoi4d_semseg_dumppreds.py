#!/usr/bin/env python
"""Dump per-point ARGMAX predictions for the qualitative HOI4D semseg figure.

For each of dense / global / random frozen encoders: train the SAME linear per-point
head on the train split (split==0) using the exact eval protocol (reuse
hoi4d_semseg_eval helpers), then run per-point inference on a handful of VAL clips
and record the argmax predicted class. Save ONE npz with, for the chosen viz clips:
  coords      [K,N,3]   one representative (mid) frame per clip
  gt          [K,N]     ground-truth per-point label at that frame
  pred_dense  [K,N]     dense-encoder per-point argmax at that frame
  pred_global [K,N]     global-encoder per-point argmax
  pred_random [K,N]     random-encoder per-point argmax
  clip_ids    [K]       clip id strings
  viz_frame   [K]       which frame index was kept per clip

Viz only: single head-training seed (0). Not the metric.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from cross4d_jepa.config import Config
from cross4d_jepa.models import build_encoder
from cross4d_jepa.data.datasets.hoi4d_semseg import HOI4DSemSegClips

# reuse the EXACT eval protocol pieces
from hoi4d_semseg_eval import (
    NUM_CLASSES, IGNORE, log, threenn_interp,
    extract_split, train_eval_head, metrics_from_conf,
)


@torch.no_grad()
def per_point_feats_full(encoder, ds, rec, device, T):
    """All-point all-frame per-point features for ONE clip via 3-NN interp.
    Returns feats [T,N,C] (float, on device), pts [T,N,3], lab [T,N]."""
    pts = torch.from_numpy(ds.load_full(rec)[:T]).float().unsqueeze(0).to(device)  # [1,T,N,3]
    out = encoder(pts)
    N = pts.shape[2]
    lab = ds.semantic(rec)[:T].astype(np.int64)
    tok = out["tokens"][0]
    txyz = out["token_xyz"][0]
    ttok = out["token_t"][0]
    feats = torch.empty(T, N, tok.shape[-1], device=device)
    for f in range(T):
        sel = ttok == f
        feats[f] = threenn_interp(txyz[sel], tok[sel], pts[0, f])
    return feats, pts[0], torch.from_numpy(lab)


def build_head(Xtr, ytr, device, num_classes, seed, epochs, lr, bs):
    """Train the linear head with the eval protocol and RETURN the trained head
    (eval train_eval_head returns only a confusion matrix; here we keep the head)."""
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)
    keep = (ytr != IGNORE)
    idx_all = torch.nonzero(keep, as_tuple=False).squeeze(1)
    P = idx_all.numel()
    steps = max(1, (P + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)
    for ep in range(epochs):
        perm = idx_all[torch.randperm(P)]
        for s in range(steps):
            b = perm[s * bs:(s + 1) * bs]
            xb = Xtr[b].to(device).float()
            yb = ytr[b].to(device)
            opt.zero_grad()
            lossf(head(xb), yb).backward()
            opt.step()
            sched.step()
    head.eval()
    return head


def load_encoder(cfg, device, ckpt=None, random_init=False):
    enc = build_encoder(cfg.encoder)
    if not random_init:
        state = torch.load(ckpt, map_location="cpu")
        enc.load_state_dict(state["encoder"])
        log(f"loaded encoder {ckpt}")
    else:
        log("random-init encoder (no checkpoint)")
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dense", required=True)
    ap.add_argument("--global-enc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=100_000)
    ap.add_argument("--k-train", type=int, default=256)
    ap.add_argument("--n-viz", type=int, default=4)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = cfg.data.num_frames
    log(f"device={device} T={T}")

    tr = HOI4DSemSegClips(cfg, "train")
    te = HOI4DSemSegClips(cfg, "test")
    log(f"train clips={len(tr)} test clips={len(te)}")

    # --- pick viz clips: val clips with the most distinct (non-ignore) classes ---
    counts = []
    for ri, rec in enumerate(te.records):
        sem = te.semantic(rec)[:T]
        u = np.unique(sem[sem != IGNORE])
        counts.append((len(u), ri, rec["clip_id"], u.tolist()))
    counts.sort(key=lambda x: -x[0])
    viz = counts[:args.n_viz]
    viz_recs = [te.records[c[1]] for c in viz]
    log("viz clips (n_classes, clip_id, classes):")
    for c in viz:
        log(f"  {c[0]:>2d}  {c[2]}  {c[3]}")

    encs = {
        "dense":  dict(ckpt=args.dense, random_init=False),
        "global": dict(ckpt=args.global_enc, random_init=False),
        "random": dict(ckpt=None, random_init=True),
    }

    K_N = cfg.data.num_points
    coords = np.zeros((args.n_viz, K_N, 3), np.float32)
    gt = np.zeros((args.n_viz, K_N), np.int64)
    preds = {m: np.zeros((args.n_viz, K_N), np.int64) for m in encs}
    viz_frame = np.zeros(args.n_viz, np.int64)
    coords_done = False

    # for sanity: per-point accuracy over ALL viz-clip points (all frames, non-ignore)
    sanity = {m: [0, 0] for m in encs}  # correct, total

    for m, spec in encs.items():
        log(f"==== method={m} ====")
        enc = load_encoder(cfg, device, spec["ckpt"], spec["random_init"])

        # train head on full train split (same protocol as eval)
        log("  extracting TRAIN feats (split==0)...")
        Xtr, ytr = extract_split(enc, tr, device, T, global_broadcast=False, k_sub=args.k_train)
        log(f"    train pts={Xtr.shape[0]:,} C={Xtr.shape[1]}")
        head = build_head(Xtr, ytr, device, NUM_CLASSES, args.seed, args.epochs, args.lr, args.bs)
        del Xtr, ytr

        mid = T // 2
        for vi, rec in enumerate(viz_recs):
            feats, pts, lab = per_point_feats_full(enc, te, rec, device, T)  # [T,N,C],[T,N,3],[T,N]
            logits = head(feats.reshape(-1, feats.shape[-1]).float())        # [T*N,K]
            pred = logits.argmax(1).reshape(T, K_N).cpu().numpy()            # [T,N]
            labn = lab.numpy()
            # sanity over all frames, non-ignore points
            mask = labn != IGNORE
            sanity[m][0] += int((pred[mask] == labn[mask]).sum())
            sanity[m][1] += int(mask.sum())
            # keep mid frame for viz
            preds[m][vi] = pred[mid]
            if not coords_done:
                coords[vi] = pts[mid].cpu().numpy()
                gt[vi] = labn[mid]
                viz_frame[vi] = mid
        coords_done = True
        del enc, head
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        coords=coords, gt=gt,
        pred_dense=preds["dense"], pred_global=preds["global"], pred_random=preds["random"],
        clip_ids=np.array([r["clip_id"] for r in viz_recs]),
        viz_frame=viz_frame,
    )
    log(f"WROTE {args.out}")

    # report sanity per-point accuracy (over all viz-clip frames, non-ignore)
    for m in encs:
        c, t = sanity[m]
        log(f"SANITY method={m} per-point-acc(all-viz-frames,non-ignore)={100.0*c/max(t,1):.4f}% ({c}/{t})")
    # also mid-frame-only accuracy (what the figure shows)
    for m in encs:
        cor = tot = 0
        for vi in range(args.n_viz):
            mask = gt[vi] != IGNORE
            cor += int((preds[m][vi][mask] == gt[vi][mask]).sum())
            tot += int(mask.sum())
        log(f"SANITY method={m} per-point-acc(mid-frame-only,non-ignore)={100.0*cor/max(tot,1):.4f}% ({cor}/{tot})")

    # colormap note: distinct non-ignore classes across the viz mid frames
    allc = set()
    for vi in range(args.n_viz):
        allc.update(np.unique(gt[vi][gt[vi] != IGNORE]).tolist())
    log(f"COLORMAP distinct non-ignore GT classes across viz mid frames: {len(allc)} -> {sorted(allc)}")


if __name__ == "__main__":
    main()
