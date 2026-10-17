"""DeformingThings4D-Animals point-cloud-clip loader.

DeformingThings4D ships per-animation `.anime` files (binary: header nf,nv,nt; rest
verts; faces; per-frame vertex offsets). The animals subset (non-driving, deformable
animal motion).

Student input = points sampled from each frame's mesh verts; teacher views are
RENDERED (textured/shaded). Downstream label = animal CATEGORY (linear probe).
Point root at .../DeformingThings4D/animals (or the dir containing the .anime dirs).
"""
import glob
import os
import re
from typing import List, Optional

import numpy as np
import torch

from ...config import Config


def anime_read(filename: str):
    """Return nf, nv, nt, vert_data[nv,3], face_data[nt,3], offset_data[nf-1,nv,3]."""
    with open(filename, "rb") as f:
        nf = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        nv = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        nt = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        vert = np.fromfile(f, dtype=np.float32, count=nv * 3).reshape(nv, 3)
        _faces = np.fromfile(f, dtype=np.int32, count=nt * 3)
        offset = np.fromfile(f, dtype=np.float32, count=-1)
    offset = offset.reshape(nf - 1, nv, 3)
    return nf, nv, vert, offset


def _category(seq_name: str) -> str:
    # DT4D animal names are "<animal><ID>_<motion>" with the ID starting at the
    # first uppercase letter or digit: "bear84Q_..."->"bear", "bearPDD_..."->"bear",
    # "cattleAFK_..."->"cattle". Category = leading LOWERCASE run.
    m = re.match(r"[a-z]+", seq_name)
    return m.group(0) if m else seq_name.lower()


class DT4DAnimalsClips:
    def __init__(self, cfg: Config, split: str = "train"):
        self.cfg = cfg
        self.split = split
        self.T = cfg.data.num_frames
        self.N = cfg.data.num_points
        self.horizons = list(cfg.data.horizons)
        self.F_canon = self.T + max(self.horizons)

        root = cfg.data.root
        files = sorted(glob.glob(os.path.join(root, "animals", "**", "*.anime"), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(root, "**", "*.anime"), recursive=True))
        if not files:
            raise FileNotFoundError(
                f"No .anime files under {root}. Extract DeformingThings4D.zip and point "
                "root at the dir containing the animal animations."
            )
        # build category label space, then a deterministic cross-IDENTITY split
        recs = []
        for fp in files:
            name = os.path.splitext(os.path.basename(fp))[0]
            recs.append({"path": fp, "clip_id": name, "identity": name.split("_")[0],
                         "category": _category(name)})
        self.categories = sorted({r["category"] for r in recs})
        self.cat_idx = {c: i for i, c in enumerate(self.categories)}
        ids = sorted({r["identity"] for r in recs})
        test_ids = set(ids[::5])  # ~20% of identities held out (cross-identity)
        keep_test = (split == "test")
        self.records = [
            {**r, "label": self.cat_idx[r["category"]]}
            for r in recs if (r["identity"] in test_ids) == keep_test
        ]
        if not self.records:
            raise RuntimeError(f"No DT4D-Animals {split} records under {root}")

    def __len__(self) -> int:
        return len(self.records)

    def load_full(self, rec) -> np.ndarray:
        nf, nv, vert, offset = anime_read(rec["path"])
        rng = np.random.default_rng(abs(hash(rec["clip_id"])) % (2**32))
        idx = np.linspace(0, nf - 1, self.F_canon).round().astype(int)
        out = []
        for i in idx:
            verts = vert if i == 0 else vert + offset[i - 1]               # [nv,3]
            sel = rng.choice(nv, self.N, replace=(nv < self.N))
            out.append(verts[sel].astype(np.float32))
        return np.stack(out)                                                # [F_canon,N,3]

    def load_corr(self, rec, max_verts: int = 2048, num_frames: Optional[int] = None):
        """Correspondence-preserving verts: [F,nv,3] with FIXED vertex identity across
        frames (NO per-frame resampling). Vertex i is the SAME semantic point in every
        frame (fixed topology) => free GT correspondence. If nv > max_verts, take a
        FIXED random subset of indices (same for all frames). Returns (verts, sel) where
        sel are the original vertex ids kept (for reference). F = min(num_frames, nf)."""
        nf, nv, vert, offset = anime_read(rec["path"])
        F = nf if num_frames is None else min(num_frames, nf)
        idx = np.linspace(0, nf - 1, F).round().astype(int)
        rng = np.random.default_rng(abs(hash(rec["clip_id"])) % (2**32))
        sel = (np.arange(nv) if nv <= max_verts
               else np.sort(rng.choice(nv, max_verts, replace=False)))
        out = [(vert if i == 0 else vert + offset[i - 1])[sel].astype(np.float32) for i in idx]
        return np.stack(out), sel                                           # [F,len(sel),3], [len(sel)]

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        full = self.load_full(rec)
        return {
            "points": torch.from_numpy(full[: self.T]).float(),
            "clip_id": rec["clip_id"],
            "t0": self.T - 1,
            "label": torch.tensor(rec["label"], dtype=torch.long),
        }
