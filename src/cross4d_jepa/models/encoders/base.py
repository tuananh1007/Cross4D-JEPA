"""Student encoder interface.

Every 4D point encoder (Point4D, P4Transformer) conforms to this contract so the
distillation head and the eval heads are backbone-agnostic, and the encoder is kept
identical across pretraining and downstream evaluation.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn


class PointVideoEncoder(nn.Module, ABC):
    """Maps a point-cloud clip to spatio-temporal features.

    forward(points, features) -> dict with:
      - "tokens":  [B, M, C]  spatio-temporal tokens (predictor context / keys)
      - "global":  [B, C]     pooled clip feature (linear-probe / fine-tune head)

    `points`:   [B, T, N, 3]
    `features`: [B, T, N, F] optional extra per-point features (None for xyz-only)
    """

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        ...

    @abstractmethod
    def forward(
        self, points: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        ...
