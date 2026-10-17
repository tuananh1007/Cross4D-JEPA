"""MaST-Pre faithful reimplementation on the SAME Point4D backbone (matched baseline).

Paper: Shen et al., "Masked Spatio-Temporal Structure Prediction for Self-supervised
Learning on Point Cloud Videos", ICCV 2023 (arXiv 2308.09245).

Pretext (two heads, equally weighted -> L = L_app + L_motion, paper Eq.5):
  1. APPEARANCE / masked point-tube reconstruction (paper Eq.2): mask a high fraction
     (75%) of spatio-temporal point tubes; from the visible tubes' encoder features,
     a lightweight transformer decoder reconstructs the LOCAL point coordinates inside
     each masked tube. Loss = symmetric L2 Chamfer distance between the predicted and
     ground-truth tube point sets, averaged over masked tubes/frames.
  2. MOTION / temporal cardinality difference (paper Eq.3-4): around each key point the
     spherical neighbourhood is split into 8 octants; each neighbour is SOFT-assigned to
     its octant by angular distance (deviation d from the octant centre line / 90 deg ->
     fraction kept, remainder to the adjacent octant). Per-frame soft cardinality
     C in R^8; the temporal cardinality difference CD = C_t - C_{t-1} in R^8 over the l
     frames of a tube. A linear head on the decoded masked-tube tokens predicts CD;
     loss = smooth-L1 (Huber).

FAITHFUL ADAPTATION to OUR backbone (documented assumptions, flagged in report):
  * Point4D produces, per frame, n2=64 sa2 tokens, each anchored at an FPS centre with a
    feature max-pooled over its kNN neighbourhood. A MaST-Pre "point tube" is exactly an
    FPS-centre-anchored spatio-temporal point neighbourhood, so we identify ONE sa2 token
    == ONE point tube (its key point = the sa2 FPS centre). This is the natural, faithful
    mapping; it reuses Point4D's own tokenizer byte-identically (no new geometry path).
  * The tube's ground-truth point set used for Chamfer + cardinality is the kNN
    neighbourhood of the token's FPS centre in the ORIGINAL (un-normalised) frame, in
    LOCAL coords (neighbour - centre), matching the paper's f(p - p_hat_i) target.
  * Masking is at the TUBE (token) level, 75% random, exactly the i-JEPA template's
    masking machinery (encode visible tokens only; decode with mask tokens at masked
    positions). Encoder = our Point4D transformer (the matched backbone), decoder = a
    small 4-layer transformer (paper's "lightweight vanilla Transformer", 4 layers).

The saved encoder.pt holds a real Point4DEncoder state_dict so downstream linear_probe
loads it byte-identically to the dense / i-JEPA encoders.
"""
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ..config import Config
from ..models import build_encoder
from ..models.encoders.point4d import (
    _sinusoidal_time, farthest_point_sample, index_points, knn)
from ..utils import get_logger
from .common import resolve_device
from .linear_probe import build_labeled_splits

log = get_logger()

K_TUBE = 16          # neighbours per tube (matches Point4D sa2 k=16 -> tube point set)


def _tokenize_with_tubes(enc, points, k_tube=K_TUBE):
    """Run Point4D's front end and ALSO return, per sa2 token, its local tube point set
    (k_tube nearest input points to the token's FPS centre, in LOCAL normalised coords)
    and that centre's xyz. Mirrors Point4DEncoder.forward up to the transformer.

    NB: this is NOT no_grad -- gradients MUST flow through sa1/sa2/pos_mlp so the whole
    Point4D backbone trains (same as the i-JEPA template's _tokenize). The geometric
    targets (tube_xyz / centre_xyz) are .detach()'d by the caller since they are
    reconstruction ground truth, not part of the predictive forward graph.

    Returns:
      tok        [B,M,C]   set-abstraction tokens (pre-transformer), M = T*n2
      tube_xyz   [B,M,k,3] local neighbour coords (neighbour - centre), normalised frame
      centre_xyz [B,M,3]   token FPS-centre coords (normalised frame)
    """
    B, T, N, _ = points.shape
    C = enc._embed_dim
    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    flat = points.reshape(B, T * N, 3)
    mn, mx = flat.min(1).values, flat.max(1).values
    center = ((mn + mx) / 2).view(B, 1, 1, 3)
    scale = ((mx - mn).max(1).values / 2 + 1e-6).view(B, 1, 1, 1)
    points = (points - center) / scale
    xyz = points.reshape(B * T, N, 3)
    x1, f1 = enc.sa1(xyz, None)
    x2, f2 = enc.sa2(x1, f1)
    n2 = f2.shape[1]
    # tube point sets: kNN of each sa2 centre x2 among the ORIGINAL frame points xyz
    nidx = knn(xyz, x2, k_tube)                       # [BT, n2, k]
    grouped = index_points(xyz, nidx) - x2.unsqueeze(2)   # local coords [BT,n2,k,3]
    tube_xyz = grouped.reshape(B, T * n2, k_tube, 3).detach()   # geometric GT (stop-grad)
    centre_xyz = x2.reshape(B, T * n2, 3).detach()
    x2r = x2.reshape(B, T, n2, 3)
    tok = f2.reshape(B, T, n2, C) + enc.pos_mlp(x2r)
    tok = tok + _sinusoidal_time(T, C, points.device)[None, :, None, :]
    tok = tok.reshape(B, T * n2, C)
    return tok, tube_xyz, centre_xyz


def _encode_visible(enc, tok, vis_mask):
    """Encoder transformer+norm over VISIBLE tokens only (padded, key-padding-masked).
    Returns ctx [B,Lmax,C] and per-batch visible counts. (Same as i-JEPA template.)"""
    B, M, C = tok.shape
    counts = vis_mask.sum(1)
    Lmax = int(counts.max().item())
    ctx_in = tok.new_zeros(B, Lmax, C)
    pad = torch.ones(B, Lmax, dtype=torch.bool, device=tok.device)
    for b in range(B):
        v = tok[b][vis_mask[b]]
        ctx_in[b, : v.shape[0]] = v
        pad[b, : v.shape[0]] = False
    ctx = enc.transformer(ctx_in, src_key_padding_mask=pad)
    ctx = enc.norm(ctx)
    return ctx, counts


class Decoder(nn.Module):
    """Lightweight vanilla transformer decoder (paper: 4 layers). Places encoded visible
    tokens at their positions and a shared learnable mask token at masked positions
    (both + token-position embedding), runs self-attention over the full token set, and
    emits per-token features feeding the reconstruction + cardinality heads."""

    def __init__(self, dim, M, k_tube, width=192, depth=4, heads=4, n_oct=8):
        super().__init__()
        self.proj_in = nn.Linear(dim, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, M, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            batch_first=True, activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.recon_head = nn.Linear(width, k_tube * 3)    # reconstruct k_tube local pts
        self.motion_head = nn.Linear(width, n_oct)        # per-token CD in R^8
        self.k_tube = k_tube
        self.n_oct = n_oct

    def forward(self, ctx, counts, vis_mask):
        B, M = vis_mask.shape
        W = self.mask_token.shape[-1]
        ctx = self.proj_in(ctx)
        seq = self.mask_token.expand(B, M, W) + self.pos
        for b in range(B):
            vidx = vis_mask[b].nonzero(as_tuple=False).squeeze(1)
            seq[b, vidx] = ctx[b, : counts[b]] + self.pos[0, vidx]
        h = self.norm(self.blocks(seq))                   # [B,M,W]
        recon = self.recon_head(h).reshape(B, M, self.k_tube, 3)
        motion = self.motion_head(h)                      # [B,M,n_oct]
        return recon, motion


def _soft_cardinality(local_xyz, n_oct=8, eps=1e-6):
    """Soft octant cardinality histogram (paper temporal-cardinality module).

    local_xyz [...,k,3] neighbour coords relative to the tube centre. Octant = sign
    pattern of (x,y,z) -> 8 bins. Each point is soft-assigned: weight in its own octant
    = 1 - d/90deg where d is the angular deviation from the octant CENTRE LINE
    (the (+-1,+-1,+-1)/sqrt 3 diagonal), remainder 1 - w spread to the rest; a point on
    the diagonal gives w=1, a point on an axis gives w~=0.5. Returns C [...,8] soft count.
    """
    *lead, k, _ = local_xyz.shape
    v = local_xyz
    norm = v.norm(dim=-1, keepdim=True).clamp_min(eps)
    u = v / norm                                          # unit dirs
    oct_idx = ((v[..., 0] >= 0).long() * 4
               + (v[..., 1] >= 0).long() * 2
               + (v[..., 2] >= 0).long())                 # [...,k] in 0..7
    sign = torch.sign(v)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    centre = sign / math.sqrt(3.0)                        # octant centre-line unit vec
    cos = (u * centre).sum(-1).clamp(-1.0, 1.0)
    ang = torch.arccos(cos)                               # deviation from centre line
    w = (1.0 - ang / (math.pi / 2.0)).clamp(0.0, 1.0)     # d/90deg fraction kept in own octant
    rem = 1.0 - w                                         # remainder per point [...,k]
    # own-octant soft count
    own = local_xyz.new_zeros(*lead, n_oct)
    own.scatter_add_(-1, oct_idx, w)
    # remainder spread uniformly over the OTHER (n_oct-1) octants (paper: remaining
    # fraction goes to neighbouring octant(s); uniform spread is the simplest defensible
    # choice). Each octant receives rem/(n_oct-1) from every point, minus the point's own.
    spread = local_xyz.new_zeros(*lead, n_oct)
    spread.scatter_add_(-1, oct_idx, rem)                 # total remainder per own-octant
    rem_total = rem.sum(-1, keepdim=True)                 # [...,1] total remainder
    C = own + rem_total / (n_oct - 1) - spread / (n_oct - 1)
    return C


def _tube_cardinality_diff(tube_xyz, T, n2, n_oct=8):
    """Per-token temporal cardinality difference target.

    tube_xyz [B,M,k,3] local neighbour coords, M=T*n2 (frame-major). Compute soft
    cardinality C [B,M,8], reshape to [B,T,n2,8], CD_t = C_t - C_{t-1}. The token at
    frame t carries CD between its frame and the previous frame's co-located token
    (same spatial slot n2). Frame 0 has no predecessor -> target 0 (and masked from loss
    via the returned validity mask).
    Returns cd [B,M,8] and valid [B,M] (False for frame-0 tokens)."""
    B, M, k, _ = tube_xyz.shape
    C = _soft_cardinality(tube_xyz, n_oct).reshape(B, T, n2, n_oct)
    cd = torch.zeros_like(C)
    cd[:, 1:] = C[:, 1:] - C[:, :-1]
    cd = cd.reshape(B, M, n_oct)
    valid = torch.ones(B, T, n2, dtype=torch.bool, device=tube_xyz.device)
    valid[:, 0] = False
    return cd, valid.reshape(B, M)


def _chamfer(pred, gt):
    """Symmetric L2 Chamfer distance between two point sets. pred,gt [P,k,3] (P tubes).
    Returns scalar mean over P (paper Eq.2: mean of both directions)."""
    d = torch.cdist(pred, gt)                             # [P,k,k]
    fwd = d.min(2).values.pow(2).mean(1)                  # pred->gt
    bwd = d.min(1).values.pow(2).mean(1)                  # gt->pred
    return (fwd + bwd).mean()


def pretrain_mastpre(cfg: Config, mask_ratio: float = 0.75, max_clips: Optional[int] = None,
                     dec_width: int = 192, dec_depth: int = 4, k_tube: int = K_TUBE) -> dict:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    nf = cfg.data.num_frames

    train_ds, _ = build_labeled_splits(cfg)
    base = train_ds.base
    recs = base.records[:max_clips] if max_clips else base.records
    log.info(f"MaST-Pre pretrain: loading {len(recs)} clips (mask_ratio={mask_ratio}, "
             f"k_tube={k_tube}, dec={dec_width}x{dec_depth})")
    clips = []
    for rec in recs:
        pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)
        clips.append(pts)

    enc = build_encoder(cfg.encoder).to(device)
    with torch.no_grad():
        tok0, _, _ = _tokenize_with_tubes(enc, clips[0][None].to(device), k_tube)
        M = tok0.shape[1]
    n2 = M // nf
    dec = Decoder(cfg.encoder.embed_dim, M, k_tube, width=dec_width, depth=dec_depth,
                  heads=cfg.encoder.num_heads).to(device)
    log.info(f"  tokens/clip M={M} (n2={n2} per frame); decoder {dec_width}x{dec_depth}")

    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()),
                            lr=cfg.optim.base_lr, weight_decay=1e-4)
    steps = cfg.optim.epochs * len(clips)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))

    n_keep = max(1, int(round((1 - mask_ratio) * M)))
    enc.train(); dec.train()
    for ep in range(cfg.optim.epochs):
        l_app_e = l_mot_e = 0.0
        for i in np.random.permutation(len(clips)):
            pts = clips[i][None].to(device)
            tok, tube_xyz, _ = _tokenize_with_tubes(enc, pts, k_tube)   # tok: grad; tube_xyz: detached GT
            cd_gt, cd_valid = _tube_cardinality_diff(tube_xyz, nf, n2)
            vis = torch.zeros(1, M, dtype=torch.bool, device=device)
            perm = torch.randperm(M, device=device)
            vis[0, perm[:n_keep]] = True
            ctx, counts = _encode_visible(enc, tok, vis)
            recon, motion = dec(ctx, counts, vis)         # [1,M,k,3], [1,M,8]
            mvis = (~vis)[0]                               # masked positions
            # appearance: Chamfer on masked tubes only
            l_app = _chamfer(recon[0][mvis], tube_xyz[0][mvis])
            # motion: smooth-L1 on masked tubes that have a valid temporal diff
            mm = mvis & cd_valid[0]
            l_mot = (F.smooth_l1_loss(motion[0][mm], cd_gt[0][mm])
                     if mm.any() else motion.sum() * 0.0)
            loss = l_app + l_mot                          # paper Eq.5, equal weight
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); opt.step()
            sched.step()
            l_app_e += float(l_app); l_mot_e += float(l_mot)
        if ep % 10 == 0 or ep == cfg.optim.epochs - 1:
            n = len(clips)
            log.info(f"  ep {ep:03d} L_app {l_app_e/n:.4f} L_motion {l_mot_e/n:.4f}")

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "encoder.pt"
    torch.save({"encoder": enc.state_dict(), "cfg": cfg.to_dict(), "embed_dim": enc.embed_dim}, ckpt)
    log.info(f"MaST-Pre pretrain done -> {ckpt}")
    return {"encoder_ckpt": str(ckpt)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MaST-Pre pretraining on Point4D backbone")
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--dec-width", type=int, default=192)
    ap.add_argument("--dec-depth", type=int, default=4)
    ap.add_argument("--k-tube", type=int, default=K_TUBE)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.out:
        cfg.out_dir = args.out
    if args.epochs:
        cfg.optim.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    pretrain_mastpre(cfg, mask_ratio=args.mask_ratio, max_clips=args.max_clips,
                     dec_width=args.dec_width, dec_depth=args.dec_depth, k_tube=args.k_tube)


if __name__ == "__main__":
    main()
