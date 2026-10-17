"""Lightweight config: nested dataclasses with YAML load/merge.

We avoid heavyweight config frameworks so the package stays dependency-light and
the configs in `configs/*.yaml` map 1:1 onto these fields.
"""
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class DataConfig:
    name: str = "msr_action3d"         # msr_action3d | ntu_rgbd60 | dt4d_animals | hoi4d_semseg | hoi4d
    root: Optional[str] = None         # dataset root on disk
    num_frames: int = 24               # T
    num_points: int = 2048             # N
    # Legacy name with a live effect: the MSR-Action3D and DeformingThings4D loaders
    # resample each video to num_frames + max(horizons) frames and use the FIRST
    # num_frames, so max(horizons) sets how much of the clip's end is dropped. Every
    # config pins the value its paper runs used.
    horizons: List[int] = field(default_factory=lambda: [4])


@dataclass
class EncoderConfig:
    name: str = "point4d"              # point4d | p4transformer_official
    embed_dim: int = 256
    depth: int = 4
    num_heads: int = 4
    in_channels: int = 3               # xyz (+ extra features if any)
    # Point4D set-abstraction granularity: centroids and kNN group size per level.
    # npoints[1] is the TOKEN granularity the dense objective sees; shrinking it
    # (with ks[1] grown to hold coverage constant) turns Point4D from a per-point
    # encoder into a pooling one — the granularity axis.
    npoints: List[int] = field(default_factory=lambda: [256, 64])
    ks: List[int] = field(default_factory=lambda: [32, 16])
    # P4Transformer-official pooling (defaults = official MSR config). Lower the
    # strides for a finer, less-pooled variant (more tokens) better suited to
    # dense per-point distillation.
    spatial_stride: int = 32
    temporal_stride: int = 2
    radius: float = 0.7


@dataclass
class TeacherConfig:
    # Default frozen teacher for engine.pretrain_dense (overridable with --teacher):
    # dinov2 | dinov3 | vjepa2 | videomaev2 | vit-in1k | vit-rand | rawpix
    name: str = "dinov2"


@dataclass
class LossConfig:
    vicreg: float = 0.0                # VICReg variance coefficient (inactive at batch size 1; see README)


@dataclass
class OptimConfig:
    epochs: int = 200
    batch_size: int = 32               # downstream probe / fine-tune loaders (pretraining is one clip per step)
    base_lr: float = 5e-4
    num_workers: int = 8


@dataclass
class Config:
    seed: int = 0
    out_dir: str = "runs/default"
    device: str = "cuda"
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    # ---- (de)serialization ----------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Config":
        return _build(cls, raw)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)


def _build(dc_type, raw: Dict[str, Any]):
    """Recursively build a (possibly nested) dataclass from a dict, ignoring
    unknown keys with a clear error so config typos surface early."""
    kwargs = {}
    valid = {f.name: f for f in fields(dc_type)}
    unknown = set(raw) - set(valid)
    if unknown:
        raise ValueError(f"Unknown config keys for {dc_type.__name__}: {sorted(unknown)}")
    for name, f in valid.items():
        if name not in raw:
            continue
        val = raw[name]
        if is_dataclass(f.type) and isinstance(val, dict):
            kwargs[name] = _build(f.type, val)
        else:
            kwargs[name] = val
    return dc_type(**kwargs)
