"""Point4D: a real hierarchical 4D point encoder (pure PyTorch, no custom CUDA).

PointNet++-style set abstraction (farthest-point sampling + kNN grouping + local
MLP + max-pool) applied per frame to capture LOCAL geometry, then a spatio-temporal
transformer over the per-frame tokens. This is the order-invariant, geometry-aware
backbone the leaderboard methods (P4Transformer/PSTNet) use in spirit; we keep it
dependency-free (FPS/kNN in torch) so it builds anywhere on the cluster.

Same {"tokens","global"} contract as the other encoders.
"""
import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import PointVideoEncoder


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """xyz [B,N,3] -> indices [B,npoint] of FPS centers (order-invariant set)."""
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    # start from the point farthest from the centroid (deterministic & order-invariant,
    # unlike starting at index 0 which depends on point order)
    far = ((xyz - xyz.mean(1, keepdim=True)) ** 2).sum(-1).argmax(1)
    ar = torch.arange(B, device=device)
    for i in range(npoint):
        idx[:, i] = far
        cen = xyz[ar, far].unsqueeze(1)              # [B,1,3]
        d = ((xyz - cen) ** 2).sum(-1)               # [B,N]
        dist = torch.minimum(dist, d)
        far = dist.argmax(1)
    return idx


def index_points(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """x [B,N,C], idx [B,...] -> [B,...,C]."""
    B = x.shape[0]
    ar = torch.arange(B, device=x.device).view(B, *([1] * (idx.dim() - 1)))
    return x[ar, idx]


def knn(xyz: torch.Tensor, new_xyz: torch.Tensor, k: int) -> torch.Tensor:
    """neighbor indices [B,M,k] of each new_xyz point among xyz."""
    d = torch.cdist(new_xyz, xyz)                    # [B,M,N]
    return d.topk(k, dim=-1, largest=False).indices


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, k: int, in_c: int, out_c: int):
        super().__init__()
        self.npoint, self.k = npoint, k
        self.mlp = nn.Sequential(
            nn.Linear(in_c + 3, out_c), nn.GELU(),
            nn.Linear(out_c, out_c), nn.GELU(),
        )

    def forward(self, xyz, feat):
        """xyz [B,N,3], feat [B,N,Cin] or None -> new_xyz [B,npoint,3], new_feat [B,npoint,Cout]."""
        fps = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps)                         # [B,npoint,3]
        nidx = knn(xyz, new_xyz, self.k)                         # [B,npoint,k]
        grouped_xyz = index_points(xyz, nidx) - new_xyz.unsqueeze(2)   # local coords
        if feat is not None:
            grouped = torch.cat([grouped_xyz, index_points(feat, nidx)], dim=-1)
        else:
            grouped = grouped_xyz
        out = self.mlp(grouped).amax(dim=2)                      # local MLP + max-pool
        return new_xyz, out


def _sinusoidal_time(t, dim, device):
    pos = torch.arange(t, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(t, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class Point4DEncoder(PointVideoEncoder):
    def __init__(self, embed_dim: int = 256, depth: int = 4, num_heads: int = 4,
                 in_channels: int = 3, npoints=(256, 64), ks=(32, 16)):
        super().__init__()
        self._embed_dim = embed_dim
        self.in_channels = in_channels
        c1 = embed_dim // 2
        extra = in_channels - 3
        self.sa1 = SetAbstraction(npoints[0], ks[0], extra, c1)
        self.sa2 = SetAbstraction(npoints[1], ks[1], c1, embed_dim)
        self.pos_mlp = nn.Linear(3, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, N, _ = points.shape
        C = self._embed_dim
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)  # sanitize bad verts (DT4D)
        # per-clip normalization to a unit cube (uniform scale preserves aspect +
        # inter-frame motion) — Simple4D had this and Point4D didn't, which is the
        # likely cause of Point4D's underperformance on metric-coord depth clouds.
        flat = points.reshape(B, T * N, 3)
        mn, mx = flat.min(1).values, flat.max(1).values            # [B,3]
        center = ((mn + mx) / 2).view(B, 1, 1, 3)
        scale = ((mx - mn).max(1).values / 2 + 1e-6).view(B, 1, 1, 1)
        points = (points - center) / scale
        xyz = points.reshape(B * T, N, 3)
        feat = features.reshape(B * T, N, -1) if features is not None else None
        x1, f1 = self.sa1(xyz, feat)                 # [BT, n1, c1]
        x2, f2 = self.sa2(x1, f1)                    # [BT, n2, C]
        n2 = f2.shape[1]
        x2r = x2.reshape(B, T, n2, 3)
        tok = f2.reshape(B, T, n2, C) + self.pos_mlp(x2r)
        tok = tok + _sinusoidal_time(T, C, points.device)[None, :, None, :]
        tok = tok.reshape(B, T * n2, C)
        tok = self.norm(self.transformer(tok))
        # token-center coords in ORIGINAL frame (un-normalize) -> lets dense distillation
        # match each token to its nearest input point's teacher target.
        token_xyz = (x2r * scale + center).reshape(B, T * n2, 3)
        # per-token input-frame index (tokens are frame-major) for dense matching
        token_t = torch.arange(T, device=points.device).repeat_interleave(n2)[None].expand(B, -1)
        return {"tokens": tok, "global": tok.mean(dim=1), "token_xyz": token_xyz, "token_t": token_t}
