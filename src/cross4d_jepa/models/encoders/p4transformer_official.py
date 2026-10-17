"""Official P4Transformer (Fan et al., CVPR 2021) wired to our encoder contract.

Uses the VENDORED official ops (encoders/p4t_official/): the point 4D convolution
front-end (P4DConv, requires the `pointnet2._ext` CUDA ops) + the official
transformer, with the exact model config from the official ``train-msr.py``
(models/msr.py ``P4Transformer``). We expose our {"tokens","global","token_xyz"}
contract instead of the classification head.

Design notes (faithful-choice decisions):
  * P4DConv config mirrors models/msr.py: in_planes=0 (xyz-only, like the official
    MSR model -> the optional `features` arg is ignored), mlp_planes=[dim],
    temporal_padding=[1, 0], temporal_kernel_size=3, temporal_stride=2,
    spatial_stride=32, radius=0.7, nsamples=32, operator='+',
    spatial_pooling='max', temporal_pooling='max'.
  * Embedding head mirrors msr.py: append a per-frame time channel (t+1) to the
    token centers -> Conv1d(4->dim) positional embedding, add to the per-token
    P4DConv features, then the official Transformer (token count + order preserved).
  * P4DConv returns new_xyzs [B,L,n,3] and new_features [B,L,C,n] (channels-first);
    we permute features to [B,L,n,C] and flatten to [B, L*n, C] (as in msr.py).
  * token_xyz must be in the ORIGINAL input frame. The official ball_query uses an
    absolute radius (0.7) tuned to MSRAction3D scale, so we normalize the input to
    a unit cube (mirroring Point4D) before P4DConv, then un-normalize the token
    centers back to the original frame for token_xyz.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .base import PointVideoEncoder
from .p4t_official.point_4d_convolution import P4DConv
from .p4t_official.transformer import Transformer


class P4TransformerOfficialEncoder(PointVideoEncoder):
    def __init__(self, embed_dim: int = 1024, depth: int = 5, num_heads: int = 8,
                 dim_head: int = 128, mlp_dim: int = 2048, in_channels: int = 3,
                 radius: float = 0.7, nsamples: int = 32, spatial_stride: int = 32,
                 temporal_kernel_size: int = 3, temporal_stride: int = 2,
                 emb_relu: bool = False, **kwargs):
        super().__init__()
        self._embed_dim = embed_dim
        # xyz-only front-end, exactly as the official MSR model (in_planes=0).
        self.tube_embedding = P4DConv(
            in_planes=0, mlp_planes=[embed_dim], mlp_batch_norm=[False],
            mlp_activation=[False], spatial_kernel_size=[radius, nsamples],
            spatial_stride=spatial_stride, temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride, temporal_padding=[1, 0],
            operator='+', spatial_pooling='max', temporal_pooling='max')
        self.pos_embedding = nn.Conv1d(in_channels=4, out_channels=embed_dim,
                                       kernel_size=1, stride=1, padding=0, bias=True)
        self.emb_relu = nn.ReLU() if emb_relu else False
        self.transformer = Transformer(embed_dim, depth, num_heads, dim_head, mlp_dim)
        # final LayerNorm (mirrors Point4D): the official P4T head pools+classifies so
        # it never normalized token features; under a scale-invariant cosine distillation
        # loss that lets feature magnitude explode (feat_std -> 80+), which hurts the
        # linear probe. Normalizing the token output keeps magnitudes bounded.
        self.norm = nn.LayerNorm(embed_dim)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def forward(self, points: torch.Tensor, features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, T, N, _ = points.shape
        device = points.device
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
        # per-clip unit-cube normalization so the official absolute ball-query radius
        # (tuned to MSRAction3D scale) is meaningful on our metric depth clouds.
        flat = points.reshape(B, T * N, 3)
        mn, mx = flat.min(1).values, flat.max(1).values
        center = ((mn + mx) / 2).view(B, 1, 1, 3)
        scale = ((mx - mn).max(1).values / 2 + 1e-6).view(B, 1, 1, 1)
        xyzs_norm = (points - center) / scale                                   # [B,T,N,3]

        # P4DConv front-end (operates on the normalized coords).
        new_xyzs, new_features = self.tube_embedding(xyzs_norm)                  # [B,L,n,3], [B,L,C,n]
        L, n = new_xyzs.shape[1], new_xyzs.shape[2]

        # per-token positional embedding: (xyz, t+1) -> Conv1d(4->C), as in msr.py.
        xyzts = []
        for t, xyz in enumerate(torch.unbind(new_xyzs, dim=1)):                 # each [B,n,3]
            tt = torch.ones((B, n, 1), dtype=torch.float32, device=device) * (t + 1)
            xyzts.append(torch.cat([xyz, tt], dim=2))                           # [B,n,4]
        xyzts = torch.stack(xyzts, dim=1).reshape(B, L * n, 4)                  # [B, L*n, 4]
        pos = self.pos_embedding(xyzts.permute(0, 2, 1)).permute(0, 2, 1)       # [B, L*n, C]

        feats = new_features.permute(0, 1, 3, 2).reshape(B, L * n, self._embed_dim)  # [B, L*n, C]
        embedding = pos + feats
        if self.emb_relu:
            embedding = self.emb_relu(embedding)
        tokens = self.norm(self.transformer(embedding))                         # [B, L*n, C]

        # token centers back in the ORIGINAL input frame for dense distillation.
        token_xyz = new_xyzs.reshape(B, L * n, 3) * scale.view(B, 1, 1) + center.view(B, 1, 3)
        # per-token input-frame index: P4DConv replicate-pads temporal_padding[0]
        # frames, so anchor l sits at original frame (kernel//2 - pad0) + l*stride.
        te = self.tube_embedding
        anchor_f = (torch.arange(L, device=device) * te.temporal_stride
                    + (te.temporal_kernel_size // 2 - te.temporal_padding[0])).clamp(0, T - 1)
        token_t = anchor_f.repeat_interleave(n)[None].expand(B, -1)
        return {"tokens": tokens, "global": tokens.mean(dim=1), "token_xyz": token_xyz, "token_t": token_t}
