"""Preprocess HOI4D per-sequence tars (RGB-D) + action annotations -> compact .npz.

The HF mirror (Livioni/hoi4d) ships one tar per sequence with images/ (RGB),
raw_depth/ (uint16 mm, 1920x1080), camera/, text/ (captions). The official
HOI4D action-segmentation labels come separately as <seq>/action/color.json
(events with event/startTime/endTime in seconds). This converts each sequence to:

    points : [F, N, 3] float16   (depth back-projected to points, sampled)
    rgb    : [F, 3, 256, 256] uint8 (resized RGB -> the REAL-RGB teacher input)
    labels : [F] int8            (per-frame action class; -1 = unlabeled/background)
    fps    : float

Camera intrinsics aren't in this mirror; we use a documented HOI4D-style pinhole
(per-clip normalization in the encoder makes the exact value non-critical for the
SSL signal; the teacher uses real RGB which is exact).
"""
import argparse
import glob
import io
import json
import os
import tarfile

import numpy as np
from PIL import Image

# fixed 20-action vocabulary (normalized lowercase) -> class index
ACTIONS = ["binding", "carry", "carrywithbothhands", "close", "cut", "dump", "go",
           "grasp", "lookaround", "open", "paper-cut", "pickup", "press", "pull",
           "push", "putdown", "reachout", "rest", "stop", "turn&on&the&switch"]
ACT_IDX = {a: i for i, a in enumerate(ACTIONS)}

# HOI4D depth is 1920x1080 uint16 mm; approximate pinhole (see module docstring).
_FX = _FY = 1000.0
_CX, _CY = 960.0, 540.0
_DEPTH_SCALE = 1000.0


def _seq_to_anno_path(seq: str) -> str:
    # "ZY20210800003_H3_C1_N02_S86_s01_T2" -> "ZY20210800003/H3/C1/N02/S86/s01/T2"
    return seq.replace("_", "/", 6)


def _per_frame_labels(color_json: dict, nframes: int) -> np.ndarray:
    dur = float(color_json.get("info", {}).get("duration", 0)) or nframes / 15.0
    fps = nframes / dur if dur > 0 else 15.0
    lab = np.full(nframes, -1, np.int8)
    for e in color_json.get("events", []):
        a = ACT_IDX.get((e.get("event") or "").strip().lower())
        if a is None:
            continue
        f0 = int(round(e.get("startTime", 0) * fps))
        f1 = int(round(e.get("endTime", 0) * fps))
        lab[max(0, f0):min(nframes, f1 + 1)] = a
    return lab, fps


def _depth_to_points(depth: np.ndarray, n: int, rng) -> np.ndarray:
    rows, cols = np.nonzero(depth > 0)
    if rows.size == 0:
        return np.zeros((n, 3), np.float32)
    z = depth[rows, cols].astype(np.float32) / _DEPTH_SCALE
    x = (cols - _CX) * z / _FX
    y = (rows - _CY) * z / _FY
    pts = np.stack([x, y, z], 1).astype(np.float32)
    return pts[rng.choice(pts.shape[0], n, replace=(pts.shape[0] < n))]


def preprocess_seq(tar_path: str, anno_root: str, out_path: str,
                   F: int = 32, N: int = 2048, img: int = 256) -> int:
    seq = os.path.basename(tar_path)[:-7]  # strip .tar.gz
    cj_path = os.path.join(anno_root, _seq_to_anno_path(seq), "action", "color.json")
    if not os.path.exists(cj_path):
        raise FileNotFoundError(f"no action/color.json for {seq} at {cj_path}")
    color = json.load(open(cj_path))

    with tarfile.open(tar_path, "r:gz") as t:
        members = {m.name: m for m in t.getmembers()}
        depth_names = sorted(n for n in members if "/raw_depth/" in n and n.endswith(".png"))
        rgb_names = sorted(n for n in members if "/images/" in n and n.endswith((".png", ".jpg")))
        nframes = len(depth_names)
        idx = np.linspace(0, nframes - 1, F).round().astype(int)
        rng = np.random.default_rng(abs(hash(seq)) % (2**32))
        pts, rgbs = [], []
        for i in idx:
            d = np.array(Image.open(io.BytesIO(t.extractfile(depth_names[i]).read())))
            pts.append(_depth_to_points(d, N, rng).astype(np.float16))
            r = Image.open(io.BytesIO(t.extractfile(rgb_names[i]).read())).convert("RGB").resize((img, img))
            rgbs.append(np.asarray(r, np.uint8).transpose(2, 0, 1))
    labels_full, fps = _per_frame_labels(color, nframes)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, points=np.stack(pts), rgb=np.stack(rgbs),
                        labels=labels_full[idx].astype(np.int8), fps=np.float32(fps), seq=seq)
    return nframes


def _process_one(task):
    """Picklable worker for the multiprocessing pool: one tar -> one npz."""
    tp, anno_root, out, F, N = task
    try:
        nf = preprocess_seq(tp, anno_root, out, F, N)
        return (os.path.basename(tp)[:-7], nf, None)
    except Exception as e:  # noqa: BLE001 — record + continue, don't kill the pool
        return (os.path.basename(tp)[:-7], None, str(e))


def main():
    ap = argparse.ArgumentParser(description="HOI4D tar + action labels -> compact npz")
    ap.add_argument("tar_dir"); ap.add_argument("anno_root"); ap.add_argument("out_dir")
    ap.add_argument("--F", type=int, default=32); ap.add_argument("--N", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel processes (each tar is independent; ~104s/seq single-threaded)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tars = sorted(glob.glob(os.path.join(args.tar_dir, "*.tar.gz")))
    todo = [tp for tp in tars
            if not os.path.exists(os.path.join(args.out_dir, os.path.basename(tp)[:-7] + ".npz"))]
    skip = len(tars) - len(todo)
    print(f"{len(tars)} sequences ({skip} already done, {len(todo)} to do, "
          f"workers={args.workers}) -> {args.out_dir}", flush=True)
    tasks = [(tp, args.anno_root, os.path.join(args.out_dir, os.path.basename(tp)[:-7] + ".npz"),
              args.F, args.N) for tp in todo]

    done = errs = 0
    if args.workers > 1:
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            for i, (name, nf, err) in enumerate(pool.imap_unordered(_process_one, tasks)):
                if err: errs += 1; print(f"  skip {name}: {err}", flush=True)
                else:   done += 1
                if i % 25 == 0:
                    print(f"[{i+1}/{len(tasks)}] {name} ({nf} frames)", flush=True)
    else:
        for i, task in enumerate(tasks):
            name, nf, err = _process_one(task)
            if err: errs += 1; print(f"  skip {name}: {err}", flush=True)
            else:   done += 1
            if i % 25 == 0:
                print(f"[{i+1}/{len(tasks)}] {name} ({nf} frames)", flush=True)
    print(f"done: {done} processed, {skip} already-done, {errs} errors -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
