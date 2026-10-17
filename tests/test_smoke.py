"""CPU smoke tests: no datasets, no teacher download.

Covers config parsing, the dataset loaders (on tiny synthetic files), the Point4D
encoder contract, the render correspondence that builds the dense targets (with the
network-free `rawpix` teacher), and the checkpoint format the released encoders use.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from cross4d_jepa.config import Config
from cross4d_jepa.engine.linear_probe import _load_encoder, build_labeled_splits
from cross4d_jepa.engine.pretrain_dense import RAWPIX_PATCH, _clip_dense_targets_rawpix
from cross4d_jepa.models import build_encoder

CONFIGS = sorted((Path(__file__).parents[1] / "configs").rglob("*.yaml"))


def _clip(T=4, N=256, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, N, 3, generator=g)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_parses(path):
    cfg = Config.from_yaml(path)
    assert cfg.encoder.name in {"point4d", "p4transformer_official"}
    if cfg.encoder.name == "point4d":
        build_encoder(cfg.encoder)


def test_point4d_contract():
    cfg = Config()
    enc = build_encoder(cfg.encoder).eval()
    T, N = 4, 512
    with torch.no_grad():
        out = enc(_clip(T, N)[None])
    n_tok = T * cfg.encoder.npoints[1]
    assert out["tokens"].shape == (1, n_tok, cfg.encoder.embed_dim)
    assert out["global"].shape == (1, cfg.encoder.embed_dim)
    assert out["token_xyz"].shape == (1, n_tok, 3)
    assert out["token_t"].shape == (1, n_tok)


def test_dense_targets_rawpix():
    dp, img = 32, 126
    proj = torch.randn(RAWPIX_PATCH * RAWPIX_PATCH * 3, dp)
    pts = _clip()
    tgt = _clip_dense_targets_rawpix(pts, proj, "cpu", img, dp)
    assert tgt.shape == (pts.shape[0], pts.shape[1], dp)
    covered = (tgt.abs().sum(-1) > 0).float().mean().item()
    assert 0.0 < covered < 1.0        # visible points get a target, occluded ones stay masked


def _fake_datasets(root):
    """Tiny on-disk stand-ins in each loader's format (MSR, NTU, HOI4D semseg)."""
    rng = np.random.default_rng(0)
    (root / "msr").mkdir()
    for a, s in [(1, 1), (2, 2), (1, 3), (2, 4)]:          # subjects 1,3 train / 2,4 test
        np.savez(root / "msr" / f"a{a:02d}_s{s:02d}_e01_sdepth.npz", rng.normal(size=(30, 300, 3)))
    (root / "ntu").mkdir()
    for p in (1, 3):                                       # performer 1 train / 3 test
        np.savez(root / "ntu" / f"S001C001P{p:03d}R001A001.npz",
                 points=rng.normal(size=(24, 2048, 3)).astype(np.float16), label=0)
    np.savez(root / "semseg.npz", pcd=rng.normal(size=(2, 24, 2048, 3)).astype(np.float32),
             semantic=np.ones((2, 24, 2048), np.uint8), split=np.array([0, 1], np.uint8))
    return {"msr_action3d": root / "msr", "ntu_rgbd60": root / "ntu", "hoi4d_semseg": root / "semseg.npz"}


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_loaders_build(path, tmp_path):
    cfg = Config.from_yaml(path)
    roots = _fake_datasets(tmp_path)
    if cfg.data.name not in roots:
        pytest.skip(f"no synthetic stand-in for {cfg.data.name}")
    cfg.data.root = str(roots[cfg.data.name])
    train, test = build_labeled_splits(cfg)
    item = train[0]
    assert item["points"].shape == (cfg.data.num_frames, cfg.data.num_points, 3)
    if cfg.data.name == "msr_action3d":                    # clip resampled to T + max(horizons)
        assert train.base.load_full(train.base.records[0]).shape[0] == cfg.data.num_frames + max(cfg.data.horizons)


def test_checkpoint_roundtrip(tmp_path):
    cfg = Config()
    enc = build_encoder(cfg.encoder)
    ckpt = tmp_path / "encoder.pt"
    torch.save({"encoder": enc.state_dict(), "cfg": cfg.to_dict(), "embed_dim": enc.embed_dim}, ckpt)
    loaded = _load_encoder(cfg, str(ckpt), "cpu")
    for k, v in enc.state_dict().items():
        assert torch.equal(v, loaded.state_dict()[k])
