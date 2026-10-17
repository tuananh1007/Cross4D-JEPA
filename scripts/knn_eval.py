#!/usr/bin/env python
"""k-NN / few-shot frozen-feature classification (a 2nd frozen-feature probe beyond
the linear probe). Extracts the encoder's mean-pooled global clip feature (frozen,
no training), then classifies the TEST split by majority vote over the k nearest
TRAIN clips in feature space (cosine distance). A dense SSL encoder should put
same-action clips near each other, so kNN accuracy >> random-init confirms the
linear-probe result is real structure, not just a linearly-separable head.

Mirrors run_transfer_eval.py: arch comes from --arch-config, data/task from
--target-config, --encoder is a checkpoint path or 'random'. Reuses
linear_probe.build_labeled_splits + the same frozen-feature extraction path.
"""
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cross4d_jepa.config import Config
from cross4d_jepa.engine.common import resolve_device
from cross4d_jepa.engine.linear_probe import build_labeled_splits
from cross4d_jepa.models import build_encoder


@torch.no_grad()
def _extract(encoder, ds, cfg, device):
    """Mean-pooled global feature for every clip -> (feats [M,D], labels [M])."""
    loader = DataLoader(ds, batch_size=cfg.optim.batch_size, num_workers=cfg.optim.num_workers)
    feats, labels = [], []
    for batch in loader:
        g = encoder(batch["points"].to(device))["global"]   # already tok.mean(1)
        feats.append(g.cpu())
        labels.append(batch["label"])
    return torch.cat(feats), torch.cat(labels)


def knn_accuracy(fx_tr, y_tr, fx_va, y_va, k: int) -> float:
    """Cosine-distance k-NN majority vote. Ties broken by closest-neighbour order."""
    a = F.normalize(fx_va, dim=1)
    b = F.normalize(fx_tr, dim=1)
    sim = a @ b.t()                                 # [Mva, Mtr], higher = closer
    nn = sim.topk(min(k, b.shape[0]), dim=1).indices
    votes = y_tr[nn]                                # [Mva, k]
    num_classes = int(max(y_tr.max(), y_va.max())) + 1
    onehot = F.one_hot(votes, num_classes).sum(1)  # [Mva, C]
    pred = onehot.argmax(1)
    return (pred == y_va).float().mean().item()


def main():
    ap = argparse.ArgumentParser(description="k-NN frozen-feature classification")
    ap.add_argument("--encoder", default="random", help="checkpoint path, or 'random'")
    ap.add_argument("--arch-config", required=True, help="config providing encoder arch")
    ap.add_argument("--target-config", required=True, help="config providing dataset/task")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    arch = Config.from_yaml(args.arch_config)
    if args.encoder == "random":
        torch.manual_seed(arch.seed)               # reproducible random-init baseline
    enc = build_encoder(arch.encoder)
    if args.encoder != "random":
        enc.load_state_dict(torch.load(args.encoder, map_location="cpu")["encoder"])

    tgt = Config.from_yaml(args.target_config)
    device = resolve_device(tgt.device)
    enc = enc.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    train_ds, val_ds = build_labeled_splits(tgt)
    fx_tr, y_tr = _extract(enc, train_ds, tgt, device)
    fx_va, y_va = _extract(enc, val_ds, tgt, device)
    num_classes = int(max(y_tr.max(), y_va.max())) + 1

    accs = {k: knn_accuracy(fx_tr, y_tr, fx_va, y_va, k) for k in args.ks}
    line = " ".join(f"k{k}={accs[k]:.4f}" for k in args.ks)
    print(f"RESULT[{args.tag}] task=knn n_train={len(y_tr)} n_test={len(y_va)} "
          f"chance={1.0/num_classes:.4f} {line}", flush=True)


if __name__ == "__main__":
    main()
