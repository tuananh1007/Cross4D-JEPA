from ...config import EncoderConfig
from .base import PointVideoEncoder


def build_encoder(cfg: EncoderConfig) -> PointVideoEncoder:
    name = cfg.name.lower()
    if name == "point4d":
        from .point4d import Point4DEncoder

        return Point4DEncoder(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            in_channels=cfg.in_channels,
            npoints=tuple(cfg.npoints),
            ks=tuple(cfg.ks),
        )
    if name == "p4transformer_official":
        from .p4transformer_official import P4TransformerOfficialEncoder

        return P4TransformerOfficialEncoder(
            embed_dim=cfg.embed_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            in_channels=cfg.in_channels,
            spatial_stride=cfg.spatial_stride,
            temporal_stride=cfg.temporal_stride,
            radius=cfg.radius,
        )
    raise ValueError(f"Unknown encoder '{cfg.name}'")


__all__ = ["build_encoder", "PointVideoEncoder"]
