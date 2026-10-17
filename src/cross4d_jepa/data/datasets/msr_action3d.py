"""MSR-Action3D point-cloud-clip loader (depth-only -> RENDERED teacher views).

567 videos, 20 actions, ~24 frames; standard PSTNet/P4Transformer preprocessing
-> one .npz per video named `aXX_sYY_eZZ_sdepth.npz` holding a point-cloud
sequence (array [F, M, 3] or an object array of per-frame [Mi,3]). Label = action
XX-1; cross-subject split = train subjects {1,3,5,7,9}, test {2,4,6,8,10}.

Temporal layout: resample each video to F_canon = T + max(horizons) frames; the
encoder (and the rendered teacher views) see the first T frames.
"""
import glob
import os
import re
from typing import List

import numpy as np
import torch

from ...config import Config

_TRAIN_SUBJECTS = {1, 3, 5, 7, 9}
_NAME = re.compile(r"a(\d+)_s(\d+)_e(\d+)")


def _load_sequence(path: str) -> List[np.ndarray]:
    """Load an .npz point-cloud sequence -> list of per-frame [Mi,3] float arrays."""
    blob = np.load(path, allow_pickle=True)
    arr = blob[blob.files[0]] if hasattr(blob, "files") else blob
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        return [np.asarray(f, np.float32)[:, :3] for f in arr]
    arr = np.asarray(arr, np.float32)
    if arr.ndim == 3:                      # [F, M, 3]
        return [arr[i][:, :3] for i in range(arr.shape[0])]
    raise ValueError(f"Unexpected array shape {arr.shape} in {path}")


def _sample_points(frame: np.ndarray, n: int, rng) -> np.ndarray:
    m = frame.shape[0]
    if m == 0:
        return np.zeros((n, 3), np.float32)
    idx = rng.choice(m, n, replace=(m < n))
    return frame[idx]


class MSRAction3DClips:
    """Clip provider. split='train' for pretraining + cache; 'test' for eval."""

    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.T = cfg.data.num_frames
        self.N = cfg.data.num_points
        self.horizons = list(cfg.data.horizons)
        self.F_canon = self.T + max(self.horizons)

        files = sorted(glob.glob(os.path.join(cfg.data.root, "*.npz")))
        if not files:
            raise FileNotFoundError(
                f"No .npz under {cfg.data.root}. Expected PSTNet-style MSR-Action3D "
                "(aXX_sYY_eZZ_sdepth.npz). See the Data section of the README."
            )
        self.records = []
        for f in files:
            m = _NAME.search(os.path.basename(f))
            if not m:
                continue
            action, subject = int(m.group(1)), int(m.group(2))
            is_train = subject in _TRAIN_SUBJECTS
            if (split == "train") == is_train:
                self.records.append({"path": f, "clip_id": os.path.basename(f)[:-4],
                                     "label": action - 1, "subject": subject})
        if not self.records:
            raise RuntimeError(f"No {split} records found under {cfg.data.root}")

    def __len__(self) -> int:
        return len(self.records)

    def load_full(self, rec) -> np.ndarray:
        """Full resampled video [F_canon, N, 3] (for rendering at cache time)."""
        frames = _load_sequence(rec["path"])
        rng = np.random.default_rng(abs(hash(rec["clip_id"])) % (2**32))
        idx = np.linspace(0, len(frames) - 1, self.F_canon).round().astype(int)
        return np.stack([_sample_points(frames[i], self.N, rng) for i in idx])

    # ---- pretrain clip provider -------------------------------------------
    def __getitem__(self, idx: int):
        rec = self.records[idx]
        full = self.load_full(rec)                       # [F_canon, N, 3]
        return {
            "points": torch.from_numpy(full[: self.T]).float(),   # student sees first T
            "clip_id": rec["clip_id"],
            "t0": self.T - 1,
            "label": torch.tensor(rec["label"], dtype=torch.long),
        }
