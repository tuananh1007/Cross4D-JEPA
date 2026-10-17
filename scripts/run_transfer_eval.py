#!/usr/bin/env python
"""Baseline + cross-dataset transfer eval (no new pretraining required).

Loads an encoder (a pretrained checkpoint, or random init) and evaluates it on a
TARGET dataset's downstream task. Covers three table rows at once:

  * random-init  : --encoder random  (lower bound: is SSL doing anything?)
  * in-domain    : --encoder <ckpt trained on the target dataset>  (our reference)
  * cross-dataset: --encoder <ckpt trained on a DIFFERENT dataset>  (generalization)

The encoder ARCHITECTURE is taken from --arch-config (the config that trained the
checkpoint); the DATA/task come from --target-config. Both Point4D configs share
dims, so an MSR-trained encoder loads cleanly for a HOI4D target and vice versa.
"""
import argparse

import torch

from cross4d_jepa.config import Config
from cross4d_jepa.engine.linear_probe import linear_probe
from cross4d_jepa.models import build_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="random", help="checkpoint path, or 'random' for random init")
    ap.add_argument("--arch-config", required=True, help="config providing the encoder architecture")
    ap.add_argument("--target-config", required=True, help="config providing the target dataset/task")
    ap.add_argument("--standardize", action="store_true",
                    help="probe diagnostic: enable feature standardization (OFF by default, matching the paper)")
    ap.add_argument("--tag", default="", help="label for the printed RESULT line")
    ap.add_argument("--seed", type=int, default=None, help="override config seed for multi-seed random-init baseline")
    args = ap.parse_args()

    arch = Config.from_yaml(args.arch_config)
    if args.encoder == "random":
        seed = args.seed if args.seed is not None else arch.seed
        torch.manual_seed(seed)   # reproducible random-init baseline (multi-seed for mean/std)
    enc = build_encoder(arch.encoder)
    if args.encoder != "random":
        enc.load_state_dict(torch.load(args.encoder, map_location="cpu")["encoder"])

    tgt = Config.from_yaml(args.target_config)
    acc = linear_probe(tgt, encoder=enc, standardize=args.standardize)
    print(f"RESULT[{args.tag}] task=probe acc={acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
