"""HOI4D semantic-segmentation subset provider (per-point semseg challenge data).

Single .npz holds the whole subsampled dataset at the encoder's resolution:
  pcd      float32 [Ncl, 24, 2048, 3]  metric-scale XYZ, 24-frame clips
  semantic uint8   [Ncl, 24, 2048]     per-point per-frame class (0 = ignore/bg)
  split    uint8   [Ncl]               0 = train, 1 = test

Exposes the same clip-provider contract as MSRAction3DClips (records / load_full /
__getitem__ / __len__) so the existing dense-distillation engine
(cross4d_jepa.engine.pretrain_dense via build_labeled_splits) can pretrain on the
split==0 clips with no engine changes. `label` is a dummy 0 (pretrain ignores it;
per-point semantic labels live in the npz and are read directly by the seg eval).
The npz is memory-mapped and shared across train/test instances so it loads once.
"""
import os
from typing import Optional

import numpy as np
import torch

from ...config import Config

_CACHE: dict = {}


def _load_npz(path: str):
    if path not in _CACHE:
        _CACHE[path] = np.load(path, mmap_mode="r")
    return _CACHE[path]


class HOI4DSemSegClips:
    """Clip provider over the npz. split='train' -> split==0, 'test' -> split==1."""

    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.T = cfg.data.num_frames
        self.N = cfg.data.num_points
        path = cfg.data.root
        if not os.path.exists(path):
            raise FileNotFoundError(f"HOI4D semseg npz not found at {path} (set data.root).")
        self.npz = _load_npz(path)
        want = 0 if split == "train" else 1
        sp = np.asarray(self.npz["split"])
        self.indices = np.nonzero(sp == want)[0]
        if self.indices.size == 0:
            raise RuntimeError(f"No {split} clips (split=={want}) in {path}")
        self.records = [{"idx": int(i), "clip_id": f"clip{int(i):05d}"} for i in self.indices]

    def __len__(self) -> int:
        return len(self.records)

    def load_full(self, rec) -> np.ndarray:
        """Clip [T, N, 3] float32 (metric scale; the render path centers per clip)."""
        return np.asarray(self.npz["pcd"][rec["idx"], : self.T], dtype=np.float32)

    def semantic(self, rec) -> np.ndarray:
        """Per-point per-frame labels [T, N] uint8 (0 = ignore/background)."""
        return np.asarray(self.npz["semantic"][rec["idx"], : self.T])

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        full = self.load_full(rec)                       # [T, N, 3]
        return {
            "points": torch.from_numpy(full).float(),
            "clip_id": rec["clip_id"],
            "label": torch.tensor(0, dtype=torch.long),  # dummy; pretrain ignores it
        }
