"""NTU-RGBD 60 preprocessing: masked depth -> point clouds (+ optional real RGB) -> npz.

Each sample is named SsssCcccPpppRrrrAaaa (setup/camera/performer/replication/action).
Masked depth (nturgb+d_depth_masked/<sample>/MDepth-*.png, 16-bit mm, Kinect v2
512x424, background already removed -> body pixels only) is back-projected to 3D and
sampled to N points over F frames. Optional RGB (nturgb+d_rgb/<sample>_rgb.avi) is
sampled to F frames @256 for the V-JEPA2 teacher (SSL pretraining). Clip-level action
label = A-1 (60 classes). Cross-subject split is applied in the loader.

points-only works without RGB (enough for from-scratch / fine-tune); RGB is needed
only for SSL pretraining (cache build). RGB decode needs `av` (uv pip install av).
"""
import argparse
import glob
import os
import re

import numpy as np

# Kinect v2 depth intrinsics commonly used for NTU (mm depth).
_FX = _FY = 365.456
_CX, _CY = 254.878, 205.395
_DEPTH_SCALE = 1000.0
_NAME = re.compile(r"S(\d+)C(\d+)P(\d+)R(\d+)A(\d+)")


def _depth_to_points(depth, N, rng):
    ys, xs = np.nonzero(depth)                       # masked -> body pixels only
    if len(xs) == 0:
        return np.zeros((N, 3), np.float32)
    z = depth[ys, xs].astype(np.float32) / _DEPTH_SCALE
    x = (xs.astype(np.float32) - _CX) * z / _FX
    y = (ys.astype(np.float32) - _CY) * z / _FY
    pts = np.stack([x, y, z], 1)
    idx = rng.choice(len(pts), N, replace=len(pts) < N)
    return pts[idx].astype(np.float32)


def _read_depth_folder(folder, F, N, rng):
    from PIL import Image
    pngs = sorted(glob.glob(os.path.join(folder, "*.png")))
    if not pngs:
        raise ValueError(f"no depth pngs in {folder}")
    idx = np.linspace(0, len(pngs) - 1, F).round().astype(int)
    return np.stack([_depth_to_points(np.array(Image.open(pngs[i])), N, rng) for i in idx])


def _read_rgb(avi, F):
    import imageio.v3 as iio
    from PIL import Image
    vid = iio.imread(avi, plugin="pyav")             # [T,H,W,3] uint8
    idx = np.linspace(0, len(vid) - 1, F).round().astype(int)
    out = [np.asarray(Image.fromarray(vid[i]).resize((256, 256))).transpose(2, 0, 1) for i in idx]
    return np.stack(out).astype(np.uint8)            # [F,3,256,256]


def preprocess_sample(folder, rgb_dir, out, F, N):
    name = os.path.basename(folder.rstrip("/"))
    m = _NAME.search(name)
    if not m:
        raise ValueError(f"bad sample name {name}")
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    saved = {"points": _read_depth_folder(folder, F, N, rng).astype(np.float16),
             "label": np.int8(int(m.group(5)) - 1), "seq": name}
    if rgb_dir:
        avi = os.path.join(rgb_dir, name + "_rgb.avi")
        if os.path.exists(avi):
            saved["rgb"] = _read_rgb(avi, F)
    np.savez_compressed(out, **saved)
    return "rgb" in saved


def _process_one(task):
    folder, rgb_dir, out, F, N = task
    try:
        has_rgb = preprocess_sample(folder, rgb_dir, out, F, N)
        return (os.path.basename(folder), has_rgb, None)
    except Exception as e:  # noqa: BLE001
        return (os.path.basename(folder), False, str(e))


def main():
    ap = argparse.ArgumentParser(description="NTU-RGBD masked depth (+RGB) -> npz")
    ap.add_argument("depth_dir", help="nturgb+d_depth_masked (folders SsssCcccPpppRrrrAaaa)")
    ap.add_argument("out_dir")
    ap.add_argument("--rgb-dir", default="", help="nturgb+d_rgb (<sample>_rgb.avi); omit for points-only")
    ap.add_argument("--F", type=int, default=24)
    ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    folders = sorted(d for d in glob.glob(os.path.join(args.depth_dir, "S*")) if os.path.isdir(d))
    todo = [d for d in folders
            if not os.path.exists(os.path.join(args.out_dir, os.path.basename(d) + ".npz"))]
    skip = len(folders) - len(todo)
    print(f"{len(folders)} samples ({skip} done, {len(todo)} to do, workers={args.workers}, "
          f"rgb={'yes' if args.rgb_dir else 'no'}) -> {args.out_dir}", flush=True)
    tasks = [(d, args.rgb_dir, os.path.join(args.out_dir, os.path.basename(d) + ".npz"),
              args.F, args.N) for d in todo]
    done = errs = withrgb = 0
    if args.workers > 1:
        from multiprocessing import Pool
        it = Pool(args.workers).imap_unordered(_process_one, tasks)
    else:
        it = (_process_one(t) for t in tasks)
    for i, (name, has_rgb, err) in enumerate(it):
        if err:
            errs += 1; print(f"  skip {name}: {err}", flush=True)
        else:
            done += 1; withrgb += int(has_rgb)
        if i % 200 == 0:
            print(f"[{i+1}/{len(tasks)}] {name} rgb={has_rgb}", flush=True)
    print(f"done: {done} processed ({withrgb} with rgb), {errs} errors, {skip} skipped", flush=True)


if __name__ == "__main__":
    main()
