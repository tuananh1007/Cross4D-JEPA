"""Preprocess raw MSR-Action3D depth sequences -> point-cloud .npz.

The official MSR-Action3D release ships per-video depth `.bin` files (login-gated
OneDrive). This converts them to the PSTNet/P4Transformer-style point clouds the
loader (msr_action3d.py) consumes: one `aXX_sYY_eZZ_sdepth.npz` per video holding
`point_clouds` of shape [F, N, 3] (3DV-PointNet++ style back-projection + sampling).

.bin format (standard MSRAction3D depth): little-endian int32 header
[nframes, ncols, nrows] followed by nframes * nrows * ncols int32 depth values (mm;
background = 0). If a real file disagrees, run with --selftest to confirm the parser
and adjust _read_depth_bin (header order / endianness) to match.

Runs on CPU (no GPU needed):
    python -m cross4d_jepa.data.datasets.msr_preprocess datasets/MSRAction3D_raw datasets/MSRAction3D
"""
import argparse
import glob
import os
from pathlib import Path

import numpy as np

# Kinect v1 intrinsics scaled to the MSR 320x240 depth maps (fx=fy~525 @ 640x480 -> /2).
_FX = _FY = 262.5
_DEPTH_SCALE = 1000.0   # mm -> m


def _read_depth_bin(path: str) -> np.ndarray:
    """Return depth maps [nframes, nrows, ncols] (int32, mm)."""
    with open(path, "rb") as f:
        head = np.frombuffer(f.read(12), dtype="<u4")   # numFrames, dims(1)=ncols, dims(2)=nrows
        nframes, ncols, nrows = int(head[0]), int(head[1]), int(head[2])
        data = np.frombuffer(f.read(), dtype="<u4")      # uint32 depths (matches loadDepthMap.m)
    expected = nframes * nrows * ncols
    if data.size != expected:
        raise ValueError(
            f"{os.path.basename(path)}: depth payload {data.size} != nframes*nrows*ncols "
            f"({nframes}*{nrows}*{ncols}={expected}). Header may differ — inspect/adjust "
            "_read_depth_bin (endianness or header order)."
        )
    return data.reshape(nframes, nrows, ncols)


def depth_to_points(depth: np.ndarray, n: int, rng) -> np.ndarray:
    """One depth map [H,W] (mm) -> sampled point cloud [n,3] (metric)."""
    H, W = depth.shape
    rows, cols = np.nonzero(depth > 0)
    if rows.size == 0:
        return np.zeros((n, 3), np.float32)
    z = depth[rows, cols].astype(np.float32) / _DEPTH_SCALE
    x = (cols - W / 2.0) * z / _FX
    y = (rows - H / 2.0) * z / _FY
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    idx = rng.choice(pts.shape[0], n, replace=(pts.shape[0] < n))
    return pts[idx]


def preprocess_file(in_path: str, out_path: str, n_points: int, rng) -> int:
    depth = _read_depth_bin(in_path)
    seq = np.stack([depth_to_points(depth[f], n_points, rng) for f in range(depth.shape[0])])
    np.savez_compressed(out_path, point_clouds=seq)
    return depth.shape[0]


def preprocess_dir(in_dir: str, out_dir: str, n_points: int = 2048) -> int:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(in_dir, "*.bin")))
    if not files:
        raise FileNotFoundError(f"No .bin depth files under {in_dir} (extract the archive first).")
    rng = np.random.default_rng(0)
    done = 0
    for i, f in enumerate(files):
        stem = Path(f).stem
        if not stem.endswith("sdepth"):
            stem = stem + "_sdepth" if "_s" in stem else stem  # keep aXX_sYY_eZZ naming
        out = os.path.join(out_dir, stem + ".npz")
        nf = preprocess_file(f, out, n_points, rng)
        done += 1
        if i % 50 == 0:
            print(f"[{i+1}/{len(files)}] {stem}: {nf} frames", flush=True)
    print(f"done: {done} videos -> {out_dir}", flush=True)
    return done


def _selftest():
    """Round-trip a synthetic .bin to validate the parser + back-projection."""
    import tempfile

    d = tempfile.mkdtemp()
    nframes, ncols, nrows = 5, 320, 240
    depth = np.zeros((nframes, nrows, ncols), np.int32)
    depth[:, 100:140, 150:190] = 1500  # a foreground blob (1.5 m)
    p = os.path.join(d, "a01_s01_e01_sdepth.bin")
    with open(p, "wb") as f:
        f.write(np.array([nframes, ncols, nrows], dtype="<i4").tobytes())
        f.write(depth.astype("<i4").tobytes())
    assert np.array_equal(_read_depth_bin(p), depth), "parse round-trip failed"
    pts = depth_to_points(depth[0], 2048, np.random.default_rng(0))
    assert pts.shape == (2048, 3) and np.isfinite(pts).all() and pts[:, 2].min() > 0
    n = preprocess_dir(d, os.path.join(d, "out"), n_points=2048)
    blob = np.load(os.path.join(d, "out", "a01_s01_e01_sdepth.npz"))
    assert blob["point_clouds"].shape == (nframes, 2048, 3)
    print(f"selftest OK: parsed {n} video, point clouds {blob['point_clouds'].shape}, "
          f"z-range [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}] m")


def main():
    ap = argparse.ArgumentParser(description="MSR-Action3D raw depth -> point-cloud .npz")
    ap.add_argument("in_dir", nargs="?", help="dir of raw .bin depth sequences")
    ap.add_argument("out_dir", nargs="?", help="output dir for .npz")
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--selftest", action="store_true", help="validate parser on synthetic data")
    args = ap.parse_args()
    if args.selftest:
        _selftest()
        return
    if not args.in_dir or not args.out_dir:
        ap.error("in_dir and out_dir required (or use --selftest)")
    preprocess_dir(args.in_dir, args.out_dir, args.points)


if __name__ == "__main__":
    main()
