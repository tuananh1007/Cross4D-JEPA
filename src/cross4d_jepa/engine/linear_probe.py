"""Linear probing — the primary evaluation protocol.

Freeze the pretrained encoder, extract its global clip feature, train only a
linear head.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..config import Config
from ..models import build_encoder
from ..utils import get_logger
from .common import resolve_device

log = get_logger()


def _load_encoder(cfg: Config, encoder_ckpt: str, device):
    encoder = build_encoder(cfg.encoder)
    state = torch.load(encoder_ckpt, map_location="cpu")
    encoder.load_state_dict(state["encoder"])
    return encoder.to(device).eval()


@torch.no_grad()
def _extract(encoder, loader, device):
    feats, labels = [], []
    for batch in loader:
        g = encoder(batch["points"].to(device))["global"]
        feats.append(g.cpu())
        labels.append(batch["label"])
    return torch.cat(feats), torch.cat(labels)


class _PointsLabels(torch.utils.data.Dataset):
    """Strip a clip provider's items down to {points, label} for probe/anticipation
    (avoids collating string clip_ids / view lists)."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        it = self.base[i]
        return {"points": it["points"], "label": it["label"]}


def build_labeled_splits(cfg: Config):
    """Return (train_ds, val_ds) of {points, label} for downstream eval."""
    if cfg.data.name == "msr_action3d":
        from ..data.datasets.msr_action3d import MSRAction3DClips
        return (_PointsLabels(MSRAction3DClips(cfg, "train")),
                _PointsLabels(MSRAction3DClips(cfg, "test")))
    if cfg.data.name == "dt4d_animals":
        from ..data.datasets.dt4d_animals import DT4DAnimalsClips
        return (_PointsLabels(DT4DAnimalsClips(cfg, "train")),
                _PointsLabels(DT4DAnimalsClips(cfg, "test")))
    if cfg.data.name == "ntu_rgbd60":
        from ..data.datasets.ntu_rgbd import NTURGBD60Clips
        return (_PointsLabels(NTURGBD60Clips(cfg, "train")),
                _PointsLabels(NTURGBD60Clips(cfg, "test")))
    if cfg.data.name == "hoi4d_semseg":
        from ..data.datasets.hoi4d_semseg import HOI4DSemSegClips
        return (_PointsLabels(HOI4DSemSegClips(cfg, "train")),
                _PointsLabels(HOI4DSemSegClips(cfg, "test")))
    raise NotImplementedError(
        f"Labeled splits for '{cfg.data.name}' not wired. Add it in build_labeled_splits."
    )


def _balanced_subset(ds, k: Optional[int], seed: int):
    """Seeded class-balanced subset of `ds` (a _PointsLabels) to ~k items, for huge
    eval splits (NTU). Reads labels from the npz filename (Aaaa -> label aaa-1) when
    the base exposes `records` with clip_ids, avoiding loading every point array;
    otherwise returns ds unchanged. No-op when k is None or k >= len(ds)."""
    if k is None or k >= len(ds):
        return ds
    import re as _re
    base = getattr(ds, "base", None)
    recs = getattr(base, "records", None)
    if recs is None or not recs or "clip_id" not in recs[0]:
        rng = np.random.RandomState(seed)
        return Subset(ds, sorted(rng.choice(len(ds), k, replace=False).tolist()))
    A = _re.compile(r"A(\d+)")
    by_cls: dict = {}
    for i, r in enumerate(recs):
        m = A.search(r["clip_id"])
        c = int(m.group(1)) - 1 if m else -1
        by_cls.setdefault(c, []).append(i)
    rng = np.random.RandomState(seed)
    for c in by_cls:
        rng.shuffle(by_cls[c])
    # round-robin across classes until we reach k -> balanced subset
    order = sorted(by_cls)
    picked, pos = [], {c: 0 for c in order}
    while len(picked) < k:
        progressed = False
        for c in order:
            if pos[c] < len(by_cls[c]):
                picked.append(by_cls[c][pos[c]]); pos[c] += 1; progressed = True
                if len(picked) >= k:
                    break
        if not progressed:
            break
    return Subset(ds, sorted(picked))


def linear_probe(
    cfg: Config,
    encoder_ckpt: Optional[str] = None,
    encoder=None,
    epochs: int = 300,
    lr: float = 1e-2,
    standardize: bool = False,
    max_train: Optional[int] = None,
    max_test: Optional[int] = None,
    splits=None,
) -> float:
    device = resolve_device(cfg.device)
    if encoder is None:
        assert encoder_ckpt, "provide encoder_ckpt or encoder"
        encoder = _load_encoder(cfg, encoder_ckpt, device)
    else:
        encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    train_ds, val_ds = build_labeled_splits(cfg) if splits is None else splits
    train_ds = _balanced_subset(train_ds, max_train, cfg.seed)
    val_ds = _balanced_subset(val_ds, max_test, cfg.seed + 1)
    tl = DataLoader(train_ds, batch_size=cfg.optim.batch_size, num_workers=cfg.optim.num_workers)
    vl = DataLoader(val_ds, batch_size=cfg.optim.batch_size, num_workers=cfg.optim.num_workers)

    fx_tr, y_tr = _extract(encoder, tl, device)
    fx_va, y_va = _extract(encoder, vl, device)
    num_classes = int(max(y_tr.max(), y_va.max())) + 1

    # Seed the head init so the probe is deterministic; train to convergence (300
    # full-batch steps). Feature standardization is OFF by default: it z-scores each
    # dim, which disproportionately *inflates the random-init baseline* (random
    # projections gain a lot from equalized scales, SSL features do not) and masks
    # the real SSL gain (with standardization SSL ~ random; without, SSL >> random).
    # Keep it as an opt-in diagnostic only.
    if standardize:
        mu, sd = fx_tr.mean(0, keepdim=True), fx_tr.std(0, keepdim=True).clamp_min(1e-6)
        fx_tr = (fx_tr - mu) / sd
        fx_va = (fx_va - mu) / sd
    torch.manual_seed(cfg.seed)

    head = nn.Linear(fx_tr.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    fx_tr, y_tr = fx_tr.to(device), y_tr.to(device)
    fx_va, y_va = fx_va.to(device), y_va.to(device)
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        lossf(head(fx_tr), y_tr).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        acc = (head(fx_va).argmax(1) == y_va).float().mean().item()
    log.info(f"linear probe: acc={acc:.4f} (chance={1.0/num_classes:.4f}, classes={num_classes})")
    return acc


def main():  # console entry: c4d-probe
    import argparse

    ap = argparse.ArgumentParser(description="Linear probe a pretrained encoder")
    ap.add_argument("config")
    ap.add_argument("--encoder", default=None, help="encoder.pt from pretraining")
    ap.add_argument("--random-init", action="store_true",
                    help="probe a freshly built (untrained) encoder of the same arch")
    ap.add_argument("--max-train", type=int, default=None, help="cap LP train clips (balanced subset)")
    ap.add_argument("--max-test", type=int, default=None, help="cap LP eval clips (balanced subset)")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    enc = build_encoder(cfg.encoder) if args.random_init else None
    assert enc is not None or args.encoder, "provide --encoder or --random-init"
    linear_probe(cfg, encoder_ckpt=args.encoder, encoder=enc,
                 max_train=args.max_train, max_test=args.max_test)


if __name__ == "__main__":
    main()
