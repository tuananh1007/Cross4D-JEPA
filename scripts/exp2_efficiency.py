#!/usr/bin/env python
"""Efficiency table: Point4D encoder params + FLOPs (one clip [1,T,2048,3]) and the
per-token dense head. Backs the paper's single-GPU / cheap claim.

Params via sum(numel); FLOPs via torch's built-in FlopCounterMode (no thop/fvcore
dependency). The dense distillation head is nn.Linear(embed_dim, dp) (dp=384, the
pretrain_dense default). Optionally also reports the official P4Transformer for
context. Pure measurement -- prints a markdown table.
"""
import argparse

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from cross4d_jepa.config import Config
from cross4d_jepa.models import build_encoder


def nparams(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def flops_fwd(m: nn.Module, *inputs) -> int:
    m.eval()
    fc = FlopCounterMode(display=False)
    with torch.no_grad(), fc:
        m(*inputs)
    return fc.get_total_flops()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="config whose encoder to measure")
    ap.add_argument("--frames", type=int, default=24, help="T in the [1,T,2048,3] probe clip")
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--dp", type=int, default=384, help="dense-target dim (per-token head out)")
    ap.add_argument("--p4t-config", default=None, help="optional P4Transformer config for context")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T, N = 1, args.frames, args.points

    # ---- Point4D encoder (the deployed backbone) ----
    enc = build_encoder(cfg.encoder).to(device)
    D = enc.embed_dim
    clip = torch.randn(B, T, N, 3, device=device)
    enc_p = nparams(enc)
    enc_f = flops_fwd(enc, clip)

    # number of tokens the encoder emits (drives the per-token head cost)
    with torch.no_grad():
        ntok = enc(clip)["tokens"].shape[1]

    # ---- per-token dense distillation head: Linear(D, dp) ----
    dense_head = nn.Linear(D, args.dp).to(device)
    dense_head_p = nparams(dense_head)
    dense_head_f = flops_fwd(dense_head, torch.randn(B, ntok, D, device=device))

    print("\n================ EXP 2: EFFICIENCY ================")
    print(f"clip = [{B},{T},{N},3]   encoder={cfg.encoder.name} embed_dim={D}")
    print(f"encoder tokens emitted = {ntok}")
    print("\n| Component | Params (M) | FLOPs (G) |")
    print("|---|---:|---:|")
    print(f"| Point4D encoder | {enc_p/1e6:.3f} | {enc_f/1e9:.3f} |")
    print(f"| Dense per-token head  Linear({D},{args.dp}) | {dense_head_p/1e6:.4f} | {dense_head_f/1e9:.4f} |")

    if args.p4t_config:
        pcfg = Config.from_yaml(args.p4t_config)
        p4t = build_encoder(pcfg.encoder).to(device)
        p4t_p = nparams(p4t)
        try:
            p4t_f = flops_fwd(p4t, torch.randn(B, T, N, 3, device=device))
            fstr = f"{p4t_f/1e9:.3f}"
        except Exception as e:
            fstr = f"n/a ({type(e).__name__})"
        print(f"| (context) {pcfg.encoder.name} embed_dim={p4t.embed_dim} | {p4t_p/1e6:.3f} | {fstr} |")
    print("==================================================\n")


if __name__ == "__main__":
    main()
