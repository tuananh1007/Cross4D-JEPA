"""HOI4D action-sequence point-cloud-clip loader.

Reads the compact .npz produced by hoi4d_preprocess.py: per sequence
  points [F,N,3] (from depth), rgb [F,3,256,256], labels [F]
  (per-frame action class, -1 = unlabeled).

In this release it is used only by scripts/target_coverage.py (HOI4D target
coverage); the HOI4D results use the semantic-segmentation subset (hoi4d_semseg.py).
"""
import glob
import os

import numpy as np
import torch

from ...config import Config
from .hoi4d_preprocess import ACTIONS


class HOI4DClips:
    """split: deterministic 80/20 over sequences (preliminary; the official HOI4D
    action-seg split can be dropped in later)."""

    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.T = cfg.data.num_frames
        self.N = cfg.data.num_points
        self.horizons = list(cfg.data.horizons)
        self.num_actions = len(ACTIONS)

        files = sorted(glob.glob(os.path.join(cfg.data.root, "*.npz")))
        if not files:
            raise FileNotFoundError(
                f"No preprocessed HOI4D .npz under {cfg.data.root}. Run "
                "data/datasets/hoi4d_preprocess.py (tar + action labels -> npz) first."
            )
        # deterministic 80/20 split by hash of sequence name
        self.records = []
        for f in files:
            seq = os.path.basename(f)[:-4]
            is_test = (abs(hash(seq)) % 5 == 0)
            if (split == "test") == is_test:
                self.records.append({"path": f, "clip_id": seq})
        if not self.records:
            raise RuntimeError(f"No HOI4D {split} records under {cfg.data.root}")

    def __len__(self) -> int:
        return len(self.records)

    def load_full(self, rec) -> np.ndarray:
        return np.load(rec["path"])["points"].astype(np.float32)      # [F, N, 3]

    def frame_labels(self, rec) -> np.ndarray:
        return np.load(rec["path"])["labels"][: self.T].astype(np.int64)  # [T]

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        full = self.load_full(rec)
        out = {
            "points": torch.from_numpy(full[: self.T]).float(),
            "clip_id": rec["clip_id"],
            "t0": self.T - 1,
        }
        return out
