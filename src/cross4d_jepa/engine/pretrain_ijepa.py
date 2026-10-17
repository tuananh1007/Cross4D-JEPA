"""INTRA-MODAL masked latent self-prediction pretraining (i-JEPA / Point-MAE style).

The matched intra-modal baseline for the dense cross-modal distillation method: SAME
Point4D backbone, SAME MSR data, SAME ~100-epoch / AdamW-5e-4 budget, SAME mean-pooled
no-standardization 300-epoch-seeded linear-probe protocol. The ONLY thing that changes
vs the dense run is the supervision signal: instead of distilling a frozen DINOv2
teacher per-point, we predict the encoder's OWN (EMA, stop-grad) token features at
masked spatio-temporal positions -- a standard self-distillation / i-JEPA pretext with
NO external teacher.

Why this is the fair test: i-JEPA / V-JEPA (the family our method belongs to) mask a
fraction of tokens, encode the visible ones with a context encoder, and predict the
EMA target encoder's representations of the masked tokens in latent space. We do exactly
that on Point4D's spatio-temporal tokens (the T*n2 set-abstraction tokens the dense run
also uses), with a small predictor and a VICReg variance backstop copied from the dense
engine. The saved encoder.pt holds a real Point4DEncoder state_dict so downstream
linear_probe loads it byte-identically to the dense encoder.

Design (i-JEPA standard, adapted to irregular 4D point tokens):
  tokens = sa1 -> sa2 -> +pos_mlp(xyz) +sinusoidal_time   (identical to Point4D.forward)
  mask ~60% of the T*n2 tokens at random (per clip).
  context: the SAME nn.TransformerEncoder runs over VISIBLE tokens only -> ctx feats.
  predictor: small TransformerEncoder over [ctx feats ; learnable mask tokens + their
             token-position embedding] -> predicts masked targets.
  target: EMA copy of the FULL encoder (sa1/sa2/pos/transformer/norm) on ALL tokens,
          stop-grad, LayerNorm'd (i-JEPA normalizes targets) -> masked-position feats.
  loss = smoothL1(pred_masked, target_masked) + vicreg * relu(1 - std(global)).mean()
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
from ..models.encoders.point4d import _sinusoidal_time
from ..utils import get_logger
from .common import resolve_device
from .linear_probe import build_labeled_splits

log = get_logger()


def _tokenize(enc, points):
    """Run Point4D's front end -> (tokens [B,M,C], M). Mirrors Point4DEncoder.forward
    up to (but not including) the transformer, so masking can be applied to the token
    set the transformer would normally see. Uses the encoder's OWN submodules so the
    geometry/pos/time embeddings are identical to the dense run."""
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
    x2r = x2.reshape(B, T, n2, 3)
    tok = f2.reshape(B, T, n2, C) + enc.pos_mlp(x2r)
    tok = tok + _sinusoidal_time(T, C, points.device)[None, :, None, :]
    tok = tok.reshape(B, T * n2, C)
    return tok


def _encode_visible(enc, tok, vis_mask):
    """Run the encoder's transformer+norm over the VISIBLE tokens only.

    tok [B,M,C], vis_mask [B,M] bool (True = visible). Tokens are padded to the max
    visible count per batch; an attention key-padding mask hides the pad slots so the
    self-attention only mixes real visible tokens (no leakage from masked positions).
    Returns ctx [B, Lmax, C] and the per-batch visible counts."""
    B, M, C = tok.shape
    counts = vis_mask.sum(1)                                  # [B]
    Lmax = int(counts.max().item())
    ctx_in = tok.new_zeros(B, Lmax, C)
    pad = torch.ones(B, Lmax, dtype=torch.bool, device=tok.device)   # True = pad (ignore)
    for b in range(B):
        v = tok[b][vis_mask[b]]                               # [Lb,C]
        ctx_in[b, : v.shape[0]] = v
        pad[b, : v.shape[0]] = False
    ctx = enc.transformer(ctx_in, src_key_padding_mask=pad)
    ctx = enc.norm(ctx)
    return ctx, counts


class Predictor(nn.Module):
    """Narrow transformer predictor (i-JEPA style). Sees context tokens + learnable
    mask tokens placed at masked positions (with their token-index pos embedding), and
    regresses the target feature at each masked position."""

    def __init__(self, dim, M, width=192, depth=4, heads=4):
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
        self.proj_out = nn.Linear(width, dim)

    def forward(self, ctx, counts, vis_mask):
        """ctx [B,Lmax,W_in=dim], counts [B], vis_mask [B,M] -> pred [B,M,dim]
        (only masked positions are meaningful; caller selects them)."""
        B, M = vis_mask.shape
        W = self.mask_token.shape[-1]
        ctx = self.proj_in(ctx)                               # [B,Lmax,W]
        Lmax = ctx.shape[1]
        # build full [B,M,W] sequence: visible positions get their ctx token + pos,
        # masked positions get the shared mask token + pos. Then run the predictor over
        # the full M-length sequence (pad-free: every position is a real token slot).
        seq = self.mask_token.expand(B, M, W) + self.pos      # masked default
        for b in range(B):
            vidx = vis_mask[b].nonzero(as_tuple=False).squeeze(1)   # [Lb]
            seq[b, vidx] = ctx[b, : counts[b]] + self.pos[0, vidx]
        out = self.norm(self.blocks(seq))
        return self.proj_out(out)                             # [B,M,dim]


@torch.no_grad()
def _ema_update(online, target, m):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data.mul_(m).add_(po.data, alpha=1 - m)
    for bo, bt in zip(online.buffers(), target.buffers()):
        bt.data.copy_(bo.data)


@torch.no_grad()
def _target_tokens(tgt_enc, tok):
    """Target features for ALL tokens from the EMA encoder's transformer+norm, then
    LayerNorm (i-JEPA normalizes targets). tok is shared (same set-abstraction tokens);
    we only need the EMA transformer's read of them. Stop-grad throughout."""
    z = tgt_enc.transformer(tok)
    z = tgt_enc.norm(z)
    z = F.layer_norm(z, (z.shape[-1],))
    return z


def pretrain_ijepa(cfg: Config, mask_ratio: float = 0.6, max_clips: Optional[int] = None,
                   ema: float = 0.996, vicreg: float = 1.0,
                   pred_width: int = 192, pred_depth: int = 4,
                   mask_mode: str = "random") -> dict:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    nf = cfg.data.num_frames

    train_ds, _ = build_labeled_splits(cfg)
    base = train_ds.base
    recs = base.records[:max_clips] if max_clips else base.records
    log.info(f"i-JEPA intra-modal pretrain: loading {len(recs)} clips (mask_ratio={mask_ratio}, "
             f"mask_mode={mask_mode}, ema={ema}, vicreg={vicreg})")
    clips = []
    for rec in recs:
        pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)   # [F,N,3]
        clips.append(pts)

    enc = build_encoder(cfg.encoder).to(device)               # online/context encoder
    tgt_enc = copy.deepcopy(enc).to(device)                   # EMA target encoder
    for p in tgt_enc.parameters():
        p.requires_grad_(False)

    # determine token count M with one tokenize pass
    with torch.no_grad():
        M = _tokenize(enc, clips[0][None].to(device)).shape[1]
    pred = Predictor(cfg.encoder.embed_dim, M, width=pred_width, depth=pred_depth,
                     heads=cfg.encoder.num_heads).to(device)
    log.info(f"  tokens per clip M={M}; predictor width={pred_width} depth={pred_depth}")

    opt = torch.optim.AdamW(list(enc.parameters()) + list(pred.parameters()),
                            lr=cfg.optim.base_lr, weight_decay=1e-4)
    steps = cfg.optim.epochs * len(clips)
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))

    n_keep = max(1, int(round((1 - mask_ratio) * M)))
    n2 = M // nf                                              # spatial tokens per frame
    enc.train(); pred.train()
    t = 0
    for ep in range(cfg.optim.epochs):
        ema_t = 1 - (1 - ema) * 0.5 * (1 + math.cos(math.pi * ep / cfg.optim.epochs))  # ema -> 1.0
        for i in np.random.permutation(len(clips)):
            pts = clips[i][None].to(device)                  # [1,F,N,3]
            tok = _tokenize(enc, pts)                         # [1,M,C]
            vis = torch.zeros(1, M, dtype=torch.bool, device=device)
            if mask_mode == "block":
                # temporal-block mask (V-JEPA multiblock style): hide a contiguous span
                # of whole frames -> the model must predict masked frames' tokens from
                # the surviving frames, forcing motion/geometry reasoning rather than
                # local spatial interpolation (the harder, standard 4D-SSL pretext).
                n_mask_frames = max(1, int(round(mask_ratio * nf)))
                start = int(torch.randint(0, nf - n_mask_frames + 1, (1,)).item())
                vis[0] = True
                vis[0, start * n2:(start + n_mask_frames) * n2] = False
            else:
                # random scatter mask: keep n_keep visible, predict the rest
                perm = torch.randperm(M, device=device)
                vis[0, perm[:n_keep]] = True
            ctx, counts = _encode_visible(enc, tok, vis)
            pred_all = pred(ctx, counts, vis)                # [1,M,C]
            with torch.no_grad():
                tgt = _target_tokens(tgt_enc, tok)           # [1,M,C] stop-grad
            mvis = ~vis
            p_m = pred_all[mvis]                             # [Nmask,C]
            t_m = tgt[mvis]
            recon = F.smooth_l1_loss(p_m, t_m)
            # VICReg variance backstop on the ONLINE global feature across the masked-
            # token batch (anti-collapse; B=1 clip so use the token dim as the batch).
            gfeat = ctx[0, : counts[0]]                       # visible ctx tokens [L,C]
            var = F.relu(1.0 - gfeat.std(0)).mean()
            loss = recon + vicreg * var
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); opt.step()
            _ema_update(enc, tgt_enc, ema_t)
            sched.step(); t += 1
        if ep % 10 == 0 or ep == cfg.optim.epochs - 1:
            with torch.no_grad():
                fstd = ctx[0, : counts[0]].std().item()
            log.info(f"  ep {ep:03d} recon {recon.item():.4f} var {var.item():.4f} "
                     f"feat_std {fstd:.4f} ema {ema_t:.4f}")

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "encoder.pt"
    torch.save({"encoder": enc.state_dict(), "cfg": cfg.to_dict(), "embed_dim": enc.embed_dim}, ckpt)
    log.info(f"i-JEPA intra-modal pretrain done -> {ckpt}")
    return {"encoder_ckpt": str(ckpt)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Intra-modal i-JEPA pretraining (no external teacher)")
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--mask-mode", choices=["random", "block"], default="random")
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--vicreg", type=float, default=1.0)
    ap.add_argument("--pred-width", type=int, default=192)
    ap.add_argument("--pred-depth", type=int, default=4)
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.out:
        cfg.out_dir = args.out
    if args.epochs:
        cfg.optim.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    pretrain_ijepa(cfg, mask_ratio=args.mask_ratio, max_clips=args.max_clips,
                   ema=args.ema, vicreg=args.vicreg,
                   pred_width=args.pred_width, pred_depth=args.pred_depth,
                   mask_mode=args.mask_mode)


if __name__ == "__main__":
    main()
