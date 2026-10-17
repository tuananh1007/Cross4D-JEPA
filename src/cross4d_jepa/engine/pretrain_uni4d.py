"""Uni4D faithful reimplementation on the SAME Point4D backbone (matched baseline).

Paper: Zuo et al., "Uni4D: A Unified Self-Supervised Learning Framework for Point Cloud
Videos" (arXiv 2504.04837).

Pretext = a SELF-DISENTANGLED masked autoencoder with FOUR losses, uniformly summed
(paper Eq.8: L = L_geo + L_lat + L_global + L_motion):

  1. L_geo  (Eq.4): mask a high fraction (75%) of point tubes; the decoder, prompted by
     learnable GEOMETRY tokens T_geo, reconstructs masked tubes' point coords -> L2
     Chamfer distance. Low-level appearance/geometry.
  2. L_lat  (Eq.5): the SAME decoder, prompted by learnable LATENT tokens T_lat, is
     distilled to a MOMENTUM (EMA) encoder that sees the FULL unmasked tokens:
        L_lat = 1 - cos(MLP(Z_lat), stopgrad(Z_target)).
     Disentangling T_geo (low-level) from T_lat (high-level) keeps semantics from being
     diminished by raw-coordinate reconstruction.
  3. L_motion (Eq.6): BIDIRECTIONAL frame-level contrastive alignment. Per-frame pooled
     features h_i; an MLP-predicted h_hat_i is pulled to its temporal neighbour (forward
     j=i-1, backward j=i+1) against the other frames as negatives (InfoNCE, tau=0.1).
  4. L_global (Eq.7): video-level contrastive alignment. The clip query q (online) is
     pulled to the momentum clip key q+ against a memory QUEUE of past keys (InfoNCE,
     tau=0.1, MoCo-style).

FAITHFUL ADAPTATION to OUR backbone (documented assumptions, flagged in report):
  * "Point tube" == ONE Point4D sa2 token (FPS-centre-anchored neighbourhood), same as
    our MaST-Pre port; tube point set for Chamfer = kNN of the centre in LOCAL coords.
  * Masking + visible-only encoding + mask-token decoding reuse the i-JEPA template
    machinery byte-identically (encoder = our Point4D transformer, the matched backbone;
    decoder = small 4-layer transformer per the paper's "lightweight 4-layer decoder").
  * Momentum encoder = EMA copy of the full Point4D encoder, fed ALL tokens (paper:
    "momentum encoder receives unmasked full embeddings").
  * Frame feature h_i = mean of that frame's tokens; clip feature q = mean of all tokens
    (online ctx vs momentum target), matching frame-/video-level granularity.
  * Per-term weights uniform (paper states uniform combination, no coefficients given) -
    FLAGGED as an assumption. tau=0.1, queue size 4096 (MoCo default; FLAGGED).

The saved encoder.pt holds a real Point4DEncoder state_dict so downstream linear_probe
loads it byte-identically to the dense / i-JEPA encoders.
"""
import math
import os
from pathlib import Path
from typing import Optional

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from ..config import Config
from ..models import build_encoder
from ..models.encoders.point4d import _sinusoidal_time, index_points, knn
from ..utils import get_logger
from .common import resolve_device
from .linear_probe import build_labeled_splits

log = get_logger()

K_TUBE = 16


def _tokenize_with_tubes(enc, points, k_tube=K_TUBE):
    """Point4D front end -> (tok [B,M,C], tube_xyz [B,M,k,3] local coords). Mirrors
    Point4DEncoder.forward up to the transformer (identical embeddings).

    NOT no_grad: gradients flow through sa1/sa2/pos_mlp so the full backbone trains
    (same as the i-JEPA template). tube_xyz is detached -- it is Chamfer ground truth."""
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
    nidx = knn(xyz, x2, k_tube)
    grouped = index_points(xyz, nidx) - x2.unsqueeze(2)
    tube_xyz = grouped.reshape(B, T * n2, k_tube, 3).detach()   # Chamfer GT (stop-grad)
    x2r = x2.reshape(B, T, n2, 3)
    tok = f2.reshape(B, T, n2, C) + enc.pos_mlp(x2r)
    tok = tok + _sinusoidal_time(T, C, points.device)[None, :, None, :]
    tok = tok.reshape(B, T * n2, C)
    return tok, tube_xyz


def _encode_visible(enc, tok, vis_mask):
    """Encoder transformer+norm over VISIBLE tokens only. Returns ctx [B,Lmax,C], counts."""
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


class DisentangledDecoder(nn.Module):
    """Single lightweight 4-layer transformer decoder, prompted by TWO learnable token
    types. For each masked position we emit a geometry feature (from a T_geo prompt) and
    a latent feature (from a T_lat prompt). We run the decoder twice over the full token
    sequence: once with masked slots = T_geo, once = T_lat (visible slots = ctx in both),
    sharing all decoder weights -> the disentanglement is in the two prompt tokens, the
    backbone is shared (paper: 'a single lightweight decoder processes both')."""

    def __init__(self, dim, M, k_tube, width=192, depth=4, heads=4):
        super().__init__()
        self.proj_in = nn.Linear(dim, width)
        self.t_geo = nn.Parameter(torch.zeros(1, 1, width))
        self.t_lat = nn.Parameter(torch.zeros(1, 1, width))
        nn.init.trunc_normal_(self.t_geo, std=0.02)
        nn.init.trunc_normal_(self.t_lat, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, M, width))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 4,
            batch_first=True, activation="gelu", norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.recon_head = nn.Linear(width, k_tube * 3)    # geometry: 1D-conv-style head
        self.lat_head = nn.Linear(width, dim)             # latent: back to encoder dim
        self.k_tube = k_tube

    def _run(self, ctx, counts, vis_mask, prompt):
        B, M = vis_mask.shape
        W = prompt.shape[-1]
        seq = prompt.expand(B, M, W) + self.pos
        for b in range(B):
            vidx = vis_mask[b].nonzero(as_tuple=False).squeeze(1)
            seq[b, vidx] = ctx[b, : counts[b]] + self.pos[0, vidx]
        return self.norm(self.blocks(seq))                # [B,M,W]

    def forward(self, ctx, counts, vis_mask):
        ctx = self.proj_in(ctx)
        B, M = vis_mask.shape
        h_geo = self._run(ctx, counts, vis_mask, self.t_geo)
        h_lat = self._run(ctx, counts, vis_mask, self.t_lat)
        recon = self.recon_head(h_geo).reshape(B, M, self.k_tube, 3)
        lat = self.lat_head(h_lat)                        # [B,M,dim]
        return recon, lat


@torch.no_grad()
def _ema_update(online, target, m):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1 - m)
    for bo, bt in zip(online.buffers(), target.buffers()):
        bt.data.copy_(bo.data)


@torch.no_grad()
def _target_tokens(tgt_enc, tok):
    """Momentum-encoder features over ALL tokens (full unmasked), stop-grad."""
    z = tgt_enc.transformer(tok)
    return tgt_enc.norm(z)                                # [B,M,C]


def _chamfer(pred, gt):
    d = torch.cdist(pred, gt)
    fwd = d.min(2).values.pow(2).mean(1)
    bwd = d.min(1).values.pow(2).mean(1)
    return (fwd + bwd).mean()


def _info_nce(pred, pos, negs, tau):
    """pred [N,D] (after MLP), pos [N,D] positive, negs [K,D] shared negatives.
    InfoNCE: -log exp(pred.pos/tau) / (exp(pred.pos/tau) + sum exp(pred.negs/tau))."""
    pred = F.normalize(pred, dim=-1)
    pos = F.normalize(pos, dim=-1)
    negs = F.normalize(negs, dim=-1)
    l_pos = (pred * pos).sum(-1, keepdim=True) / tau      # [N,1]
    l_neg = pred @ negs.t() / tau                         # [N,K]
    logits = torch.cat([l_pos, l_neg], dim=1)             # [N,1+K]
    labels = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)
    return F.cross_entropy(logits, labels)


def pretrain_uni4d(cfg: Config, mask_ratio: float = 0.75, max_clips: Optional[int] = None,
                   ema: float = 0.996, tau: float = 0.1, queue_size: int = 4096,
                   dec_width: int = 192, dec_depth: int = 4, k_tube: int = K_TUBE) -> dict:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    nf = cfg.data.num_frames
    dim = cfg.encoder.embed_dim

    train_ds, _ = build_labeled_splits(cfg)
    base = train_ds.base
    recs = base.records[:max_clips] if max_clips else base.records
    log.info(f"Uni4D pretrain: loading {len(recs)} clips (mask_ratio={mask_ratio}, "
             f"ema={ema}, tau={tau}, queue={queue_size}, k_tube={k_tube})")
    clips = []
    for rec in recs:
        pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)
        clips.append(pts)

    enc = build_encoder(cfg.encoder).to(device)               # online encoder
    tgt_enc = copy.deepcopy(enc).to(device)                   # momentum (EMA) encoder
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        tok0, _ = _tokenize_with_tubes(enc, clips[0][None].to(device), k_tube)
        M = tok0.shape[1]
    n2 = M // nf
    dec = DisentangledDecoder(dim, M, k_tube, width=dec_width, depth=dec_depth,
                              heads=cfg.encoder.num_heads).to(device)
    # projection/prediction MLP heads (online side) for the three alignment losses
    mlp_lat = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)).to(device)
    mlp_motion = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)).to(device)
    mlp_global = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim)).to(device)
    log.info(f"  tokens/clip M={M} (n2={n2}/frame); decoder {dec_width}x{dec_depth}")

    queue = F.normalize(torch.randn(queue_size, dim, device=device), dim=-1)  # global negatives
    qptr = 0

    params = (list(enc.parameters()) + list(dec.parameters()) + list(mlp_lat.parameters())
              + list(mlp_motion.parameters()) + list(mlp_global.parameters()))
    opt = torch.optim.AdamW(params, lr=cfg.optim.base_lr, weight_decay=1e-4)
    steps = cfg.optim.epochs * len(clips)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))

    n_keep = max(1, int(round((1 - mask_ratio) * M)))
    enc.train(); dec.train()
    for ep in range(cfg.optim.epochs):
        ema_t = 1 - (1 - ema) * 0.5 * (1 + math.cos(math.pi * ep / cfg.optim.epochs))
        lg = ll = lm = lglob = 0.0
        for i in np.random.permutation(len(clips)):
            pts = clips[i][None].to(device)
            tok, tube_xyz = _tokenize_with_tubes(enc, pts, k_tube)
            vis = torch.zeros(1, M, dtype=torch.bool, device=device)
            perm = torch.randperm(M, device=device)
            vis[0, perm[:n_keep]] = True
            ctx, counts = _encode_visible(enc, tok, vis)
            recon, lat = dec(ctx, counts, vis)            # [1,M,k,3], [1,M,dim]
            with torch.no_grad():
                tgt = _target_tokens(tgt_enc, tok)        # [1,M,dim] full, stop-grad
            mvis = (~vis)[0]

            # 1) L_geo: Chamfer on masked tubes
            l_geo = _chamfer(recon[0][mvis], tube_xyz[0][mvis])

            # 2) L_lat: cosine distillation of decoded latent tokens to momentum target
            #    at masked positions (1 - cos), MLP on online side.
            p_lat = mlp_lat(lat[0][mvis])                 # [Nmask,dim]
            t_lat = tgt[0][mvis]
            l_lat = (1.0 - F.cosine_similarity(p_lat, t_lat, dim=-1)).mean()

            # 3) L_motion: bidirectional frame-level InfoNCE. Online per-frame feature =
            #    mean over that frame's full decoded latent tokens; target per-frame =
            #    momentum mean. Predict frame i's feature -> pull to neighbour frames.
            h_on = lat[0].reshape(nf, n2, dim).mean(1)    # [T,dim] online
            h_tg = tgt[0].reshape(nf, n2, dim).mean(1)    # [T,dim] momentum target
            pred_h = mlp_motion(h_on)                     # [T,dim]
            # forward neighbour j=i-1, backward neighbour j=i+1; negatives = all other frames
            l_fwd = _info_nce(pred_h[1:], h_tg[:-1], h_tg, tau)   # i>=1 -> i-1
            l_bwd = _info_nce(pred_h[:-1], h_tg[1:], h_tg, tau)   # i<=T-2 -> i+1
            l_mot = 0.5 * (l_fwd + l_bwd)

            # 4) L_global: video-level InfoNCE vs momentum key + memory queue.
            q_on = lat[0].mean(0, keepdim=True)           # [1,dim] online clip feature
            q_tg = tgt[0].mean(0, keepdim=True)           # [1,dim] momentum key
            pred_q = mlp_global(q_on)
            l_glob = _info_nce(pred_q, q_tg, queue, tau)

            loss = l_geo + l_lat + l_mot + l_glob         # paper Eq.8, uniform
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); opt.step()
            _ema_update(enc, tgt_enc, ema_t)
            # enqueue momentum key (dequeue oldest)
            with torch.no_grad():
                queue[qptr] = F.normalize(q_tg[0], dim=-1)
                qptr = (qptr + 1) % queue_size
            sched.step()
            lg += float(l_geo); ll += float(l_lat); lm += float(l_mot); lglob += float(l_glob)
        if ep % 10 == 0 or ep == cfg.optim.epochs - 1:
            n = len(clips)
            log.info(f"  ep {ep:03d} L_geo {lg/n:.4f} L_lat {ll/n:.4f} "
                     f"L_motion {lm/n:.4f} L_global {lglob/n:.4f} ema {ema_t:.4f}")

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "encoder.pt"
    torch.save({"encoder": enc.state_dict(), "cfg": cfg.to_dict(), "embed_dim": enc.embed_dim}, ckpt)
    log.info(f"Uni4D pretrain done -> {ckpt}")
    return {"encoder_ckpt": str(ckpt)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Uni4D pretraining on Point4D backbone")
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--queue-size", type=int, default=4096)
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
    pretrain_uni4d(cfg, mask_ratio=args.mask_ratio, max_clips=args.max_clips,
                   ema=args.ema, tau=args.tau, queue_size=args.queue_size,
                   dec_width=args.dec_width, dec_depth=args.dec_depth, k_tube=args.k_tube)


if __name__ == "__main__":
    main()
