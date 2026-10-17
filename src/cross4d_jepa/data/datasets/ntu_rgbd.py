"""NTU-RGBD 60 point-cloud-clip loader (cross-subject).

Reads the compact .npz produced by ntu_preprocess.py: points [F,N,3] (from masked
depth) and label (action-1, 60 classes). Teacher views are rendered from the points
by engine.pretrain_dense, so the optional rgb array is not used.

Standard NTU-60 cross-subject (X-Sub) split.
"""
import glob
import os
import re

import numpy as np
import torch

from ...config import Config

# NTU-60 cross-subject training performer IDs (Shahroudy et al., CVPR 2016).
_TRAIN_SUBJECTS = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
_PID = re.compile(r"P(\d+)")


class NTURGBD60Clips:
    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.T = cfg.data.num_frames
        self.N = cfg.data.num_points

        files = sorted(glob.glob(os.path.join(cfg.data.root, "*.npz")))
        if not files:
            raise FileNotFoundError(
                f"No preprocessed NTU .npz under {cfg.data.root}. Run "
                "data/datasets/ntu_preprocess.py (masked depth [+RGB] -> npz) first."
            )
        self.records = []
        for f in files:
            name = os.path.basename(f)[:-4]
            subj = int(_PID.search(name).group(1))
            if (split == "train") == (subj in _TRAIN_SUBJECTS):
                self.records.append({"path": f, "clip_id": name})
        if not self.records:
            raise RuntimeError(f"No NTU {split} records under {cfg.data.root}")

    def __len__(self) -> int:
        return len(self.records)

    def load_full(self, rec) -> np.ndarray:
        return np.load(rec["path"])["points"].astype(np.float32)       # [F, N, 3]

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        z = np.load(rec["path"])
        return {
            "points": torch.from_numpy(z["points"][: self.T].astype(np.float32)).float(),
            "label": int(z["label"]),
            "clip_id": rec["clip_id"],
            "t0": self.T - 1,
        }
