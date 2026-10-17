#!/usr/bin/env python
"""Build the data behind the project page (docs/) from the released checkpoints.

Writes into --out (default docs/data). Each array file is JSON {"encoding": "base64",
"bytes", "data"} holding the raw little-endian arrays listed below, back to back:
  demo.json            index of everything below (clip ids, actions, shapes, accuracies)
  msr_<clip>.json      MSR-Action3D clip, all frames: int16 xyz [T,N,3] (normalized to
                       [-1,1] * 32767), then uint8 PCA-RGB [T,N,3] for the dense, global
                       and random-init encoders (in that order)
  hoi4d_<clip>.json    HOI4D semseg clip, all frames: int16 xyz [T,N,3], then uint8 class
                       ids [T,N] for ground truth, dense and global in-domain encoders
  pipeline.json        one MSR frame through the render correspondence: int16 xyz [N,3],
                       uint8 per-point target RGB [N,3], uint8 patch index [N] (255 = not
                       visible, so no target)
                       (render image and patch grid go into demo.json)

Per-point features follow the paper's figures: MSR uses each point's nearest encoder
token with one PCA basis per encoder shared across all clips and frames; HOI4D trains the
per-point linear head of scripts/hoi4d_semseg_eval.py (seed 0) and takes its argmax.
Run from the repository root on a GPU machine, with the MSR-Action3D and HOI4D-semseg
data in datasets/ and the main + hoi4d_semseg checkpoint archives in checkpoints/.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hoi4d_semseg_dumppreds import build_head, load_encoder, per_point_feats_full  # noqa: E402
from hoi4d_semseg_eval import IGNORE, NUM_CLASSES, extract_split  # noqa: E402

from cross4d_jepa.config import Config  # noqa: E402
from cross4d_jepa.data.datasets.hoi4d_semseg import HOI4DSemSegClips  # noqa: E402
from cross4d_jepa.data.render.renderer import (RenderConfig, make_cameras,  # noqa: E402
                                               render_frame, render_frame_index)
from cross4d_jepa.engine.linear_probe import build_labeled_splits  # noqa: E402
from cross4d_jepa.models.encoders.point4d import knn  # noqa: E402

MSR_ACTIONS = ["high arm wave", "horizontal arm wave", "hammer", "hand catch", "forward punch",
               "high throw", "draw x", "draw tick", "draw circle", "hand clap", "two hand wave",
               "side-boxing", "bend", "forward kick", "side kick", "jogging", "tennis swing",
               "tennis serve", "golf swing", "pick up & throw"]


def write_blob(out, name, *arrays):
    raw = b"".join(np.ascontiguousarray(a).tobytes() for a in arrays)
    with open(os.path.join(out, name), "w") as fh:
        json.dump({"encoding": "base64", "bytes": len(raw), "data": base64.b64encode(raw).decode()}, fh)


def quantize_xyz(pts):
    """[T,N,3] -> int16 in [-32767, 32767] after centering on the clip mean and scaling by
    its largest absolute offset (one transform per clip, so motion is preserved)."""
    c = pts.reshape(-1, 3).mean(0)
    s = np.abs(pts.reshape(-1, 3) - c).max() + 1e-8
    return np.round((pts - c) / s * 32767).astype(np.int16)


@torch.no_grad()
def nearest_token_feats(enc, pts, device):
    """pts [T,N,3] -> per-point features [T,N,C] from each point's nearest token."""
    x = pts[None].to(device)
    out = enc(x)
    tok, txyz, ttok = out["tokens"][0], out["token_xyz"][0], out["token_t"][0]
    feats = torch.zeros(pts.shape[0], pts.shape[1], tok.shape[1], device=device)
    for f in range(pts.shape[0]):
        sel = ttok == f
        nnt = knn(txyz[sel][None], x[0, f][None], 1)[0, :, 0]
        feats[f] = tok[sel][nnt]
    return feats.cpu().numpy()


def pca_fit(X, rng, max_rows=60000):
    """3-component PCA + 2/98-percentile range on (a subsample of) features X [M,C]."""
    if X.shape[0] > max_rows:
        X = X[rng.choice(X.shape[0], max_rows, replace=False)]
    X = X.astype(np.float64)
    mean = X.mean(0, keepdims=True)
    V3 = np.linalg.svd(X - mean, full_matrices=False)[2][:3]
    proj = (X - mean) @ V3.T
    return mean, V3, np.percentile(proj, 2, 0), np.percentile(proj, 98, 0)


def pca_rgb(F, params):
    mean, V3, lo, hi = params
    proj = (F.reshape(-1, F.shape[-1]).astype(np.float64) - mean) @ V3.T
    rgb = np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1).reshape(*F.shape[:-1], 3)
    return np.round(rgb * 255).astype(np.uint8)


def msr_demo(args, device, out, meta):
    cfg = Config.from_yaml(args.msr_config)
    _, test = build_labeled_splits(cfg)
    base = test.base
    picks = []
    for lab in args.msr_actions:
        i = next(i for i, r in enumerate(base.records) if r["label"] == lab)
        picks.append(base.records[i])
    T = cfg.data.num_frames
    encs = {}
    for name, ckpt in [("dense", args.dense), ("global", args.global_ckpt), ("random", None)]:
        torch.manual_seed(0)
        encs[name] = load_encoder(cfg, device, ckpt, random_init=ckpt is None)
    clips = []
    for rec in picks:
        pts = base.load_full(rec)[:T].astype(np.float32)
        feats = {k: nearest_token_feats(e, torch.as_tensor(pts), device) for k, e in encs.items()}
        clips.append((rec, pts, feats))
    rng = np.random.default_rng(0)
    bases = {k: pca_fit(np.concatenate([f[k].reshape(-1, f[k].shape[-1]) for _, _, f in clips]), rng)
             for k in encs}
    meta["msr"] = []
    for rec, pts, feats in clips:
        fn = f"msr_{rec['clip_id']}.json"
        write_blob(out, fn, quantize_xyz(pts), *(pca_rgb(feats[k], bases[k]) for k in ("dense", "global", "random")))
        meta["msr"].append({"clip": rec["clip_id"], "label": rec["label"], "action": MSR_ACTIONS[rec["label"]],
                            "frames": int(pts.shape[0]), "points": int(pts.shape[1]), "file": fn})
        print("wrote", fn, flush=True)
    return picks[0], cfg


def pipeline_demo(rec, cfg, device, out, meta, frame, img=126):
    """One frame through render -> DINOv2 patch grid -> back-projected per-point targets."""
    from transformers import AutoModel
    dino = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()
    _, test = build_labeled_splits(cfg)
    pts = torch.as_tensor(test.base.load_full(rec)[:cfg.data.num_frames], dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K = torch.as_tensor(cams[0][0], dtype=torch.float32, device=device)
    Rt = torch.as_tensor(cams[0][1], dtype=torch.float32, device=device)
    p = pts[frame].to(device)
    with torch.no_grad():
        im = render_frame(p, K, Rt, img, img, 1)                                  # [3,H,W]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        tok = dino(pixel_values=((im - mean) / std)[None]).last_hidden_state[0, 1:]  # [P,D]
        g = img // 14
        grid = tok.reshape(g, g, -1)
        idx = render_frame_index(p, K, Rt, img, img, 1)
    rng = np.random.default_rng(0)
    params = pca_fit(tok.cpu().numpy(), rng)
    patch_rgb = pca_rgb(grid.cpu().numpy(), params)                               # [g,g,3]
    # Same pixel -> patch mapping as dense_point_targets (nearest upsampling of the grid);
    # numpy fancy assignment keeps the last pixel per point, consistently for both arrays.
    idx = idx.cpu().numpy()
    yi = np.minimum(np.arange(img) * g // img, g - 1)
    pix_patch = (yi[:, None] * g + yi[None, :]).reshape(-1)
    flat = idx.reshape(-1); seen = flat >= 0
    N = p.shape[0]
    rgb = np.zeros((N, 3), np.uint8); patch = np.full(N, 255, np.uint8)
    rgb[flat[seen]] = patch_rgb.reshape(-1, 3)[pix_patch[seen]]
    patch[flat[seen]] = pix_patch[seen]
    vis = patch != 255
    write_blob(out, "pipeline.json", quantize_xyz(pts[frame:frame + 1].numpy())[0], rgb, patch)
    meta["pipeline"] = {"clip": rec["clip_id"], "action": MSR_ACTIONS[rec["label"]], "frame": frame,
                        "image_size": img, "patch": 14, "grid": g, "points": int(N),
                        "visible_points": int(vis.sum()), "file": "pipeline.json",
                        "render": np.round(im[0].cpu().numpy() * 255).astype(np.uint8).tolist(),
                        "patch_rgb": patch_rgb.tolist()}
    print("wrote pipeline.json", int(vis.sum()), "visible of", N, flush=True)


def hoi4d_demo(args, device, out, meta):
    cfg = Config.from_yaml(args.hoi4d_config)
    T = cfg.data.num_frames
    tr, te = HOI4DSemSegClips(cfg, "train"), HOI4DSemSegClips(cfg, "test")
    counts = []
    for ri, rec in enumerate(te.records):                   # same pick rule as the paper figure
        sem = te.semantic(rec)[:T]
        counts.append((len(np.unique(sem[sem != IGNORE])), ri))
    counts.sort(key=lambda x: -x[0])
    viz = [te.records[ri] for _, ri in counts[:args.n_hoi4d]]
    preds, coords, gts = {}, {}, {}
    for name, ckpt in [("dense", args.hoi4d_dense), ("global", args.hoi4d_global)]:
        enc = load_encoder(cfg, device, ckpt)
        Xtr, ytr = extract_split(enc, tr, device, T, global_broadcast=False, k_sub=256)
        head = build_head(Xtr, ytr, device, NUM_CLASSES, 0, 100, 1e-3, 100_000)
        del Xtr, ytr
        for rec in viz:
            with torch.no_grad():
                feats, pts, lab = per_point_feats_full(enc, te, rec, device, T)
                preds[(name, rec["clip_id"])] = head(feats.reshape(-1, feats.shape[-1]).float()).argmax(1) \
                    .reshape(T, -1).cpu().numpy().astype(np.uint8)
            coords[rec["clip_id"]] = pts.cpu().numpy(); gts[rec["clip_id"]] = lab.numpy().astype(np.uint8)
        del enc, head
        torch.cuda.empty_cache()
    meta["hoi4d"] = []
    for rec in viz:
        cid = rec["clip_id"]; g = gts[cid]; m = g != IGNORE
        acc = {k: round(float((preds[(k, cid)][m] == g[m]).mean()), 4) for k in ("dense", "global")}
        fn = f"hoi4d_{cid}.json"
        write_blob(out, fn, quantize_xyz(coords[cid]), g, preds[("dense", cid)], preds[("global", cid)])
        meta["hoi4d"].append({"clip": cid, "frames": T, "points": int(g.shape[1]), "file": fn,
                              "classes": sorted(int(c) for c in np.unique(g[m])), "point_accuracy": acc})
        print("wrote", fn, acc, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="docs/data")
    ap.add_argument("--msr-config", default="configs/msr_point4d.yaml")
    ap.add_argument("--dense", default="checkpoints/main/msr_dense_dinov2_s0.pt")
    ap.add_argument("--global-ckpt", default="checkpoints/main/msr_global_dinov2_s0.pt")
    ap.add_argument("--msr-actions", type=int, nargs="+", default=[0, 10, 12, 13, 18],
                    help="MSR action labels to show (first test clip of each)")
    ap.add_argument("--pipeline-frame", type=int, default=12)
    ap.add_argument("--hoi4d-config", default="configs/hoi4d_semseg_point4d.yaml")
    ap.add_argument("--hoi4d-dense", default="checkpoints/hoi4d_semseg/hoi4d_dense_dinov2_indomain.pt")
    ap.add_argument("--hoi4d-global", default="checkpoints/hoi4d_semseg/hoi4d_global_dinov2_indomain.pt")
    ap.add_argument("--n-hoi4d", type=int, default=4)
    ap.add_argument("--skip", nargs="*", default=[], choices=["msr", "pipeline", "hoi4d"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    meta_path = os.path.join(args.out, "demo.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    if "msr" not in args.skip or "pipeline" not in args.skip:
        rec, cfg = msr_demo(args, device, args.out, meta) if "msr" not in args.skip else (None, None)
        if "pipeline" not in args.skip:
            if rec is None:
                cfg = Config.from_yaml(args.msr_config)
                base = build_labeled_splits(cfg)[1].base
                rec = next(r for r in base.records if r["label"] == args.msr_actions[0])
            pipeline_demo(rec, cfg, device, args.out, meta, args.pipeline_frame)
    if "hoi4d" not in args.skip:
        hoi4d_demo(args, device, args.out, meta)
    json.dump(meta, open(meta_path, "w"))
    print("wrote", meta_path, flush=True)


if __name__ == "__main__":
    main()
