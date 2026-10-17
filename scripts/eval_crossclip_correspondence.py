#!/usr/bin/env python
"""Zero-shot CROSS-CLIP 3D dense-correspondence eval on DeformingThings4D-Animals.

HARDER counterpart to scripts/eval_dense_correspondence.py. The within-clip eval is
dominated by XYZ-NN because DT4D deforms slowly, so the spatially-nearest vertex in
frame t+Delta IS the same vertex (geometry is near-oracle). This eval removes that
crutch: it matches vertices ACROSS DIFFERENT ANIMATION CLIPS of the SAME animal
identity, which share mesh topology (same nv, same vertex ordering -> vertex i is the
same semantic point => FREE GT) but are in DIFFERENT POSES (root motion + articulation),
so raw XYZ position is NOT a reliable match. This is the regime where a semantic feature
should beat geometry; if it does, the dense capability claim holds.

Pairing: per test identity with >=2 clips, sample clip pairs (A,B). For each pair pick a
query frame in A and a target frame in B (default: each clip's middle frame = a
representative deformed pose, maximizing pose divergence). Query vertex i in A is matched
by feature cosine to the most-similar vertex in B; GT match = vertex i.

Per-vertex features come from the SAME nearest-token IDW propagation as the within-clip
eval (vertex_feats). Each clip is encoded independently (separate forward pass) -- that is
the realistic cross-clip use. Cosine similarity is scale-invariant, and the encoder's
per-clip normalization is internal, so features remain comparable across clips.

Methods: DENSE (distilled encoder.pt), RANDOM (random-init, seeded), XYZ-RAW (vertex xyz),
XYZ-CANON (vertex xyz after per-pose center + unit-scale -- the FAIREST geometry baseline,
removes the trivial global-translation gap from root-motion clips). Reports recall@1/5/10,
median rank, mean Euclidean error of the top-1 match vs the GT vertex position.
"""
import argparse
import collections
import os

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from cross4d_jepa.config import Config
from cross4d_jepa.data.datasets.dt4d_animals import DT4DAnimalsClips
from cross4d_jepa.models import build_encoder


@torch.no_grad()
def vertex_feats(enc, verts, device, k=3):
    """verts [F,nv,3] -> per-vertex features [F,nv,C] from the k NEAREST tokens in the
    SAME frame (IDW), identical to scripts/eval_dense_correspondence.py:vertex_feats."""
    F_, nv, _ = verts.shape
    pts = torch.as_tensor(verts, dtype=torch.float32, device=device)[None]   # [1,F,nv,3]
    out = enc(pts)
    tok = out["tokens"][0]                                                   # [M,C]
    txyz = out["token_xyz"][0]                                              # [M,3]
    ttok = out["token_t"][0]                                                # [M]
    C = tok.shape[1]
    feats = torch.zeros(F_, nv, C, device=device)
    for f in range(F_):
        sel = ttok == f
        if not sel.any():
            continue
        tx, tf = txyz[sel], tok[sel]
        kk = min(k, tx.shape[0])
        d = torch.cdist(pts[0, f], tx)
        dk, idk = d.topk(kk, dim=-1, largest=False)
        w = 1.0 / (dk + 1e-8); w = w / w.sum(-1, keepdim=True)
        feats[f] = (w[..., None] * tf[idk]).sum(1)
    return feats


def anime_read(filename):
    """nf, nv, vert[nv,3], offset[nf-1,nv,3] (mirrors dt4d_animals.anime_read)."""
    with open(filename, "rb") as f:
        nf = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        nv = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        nt = int(np.fromfile(f, dtype=np.int32, count=1)[0])
        vert = np.fromfile(f, dtype=np.float32, count=nv * 3).reshape(nv, 3)
        _ = np.fromfile(f, dtype=np.int32, count=nt * 3)
        offset = np.fromfile(f, dtype=np.float32, count=-1).reshape(nf - 1, nv, 3)
    return nf, nv, vert, offset


def load_clip_verts(path, num_frames, max_verts, sel):
    """Correspondence-preserving verts [F,len(sel),3] for ONE clip with a GIVEN fixed
    vertex subset `sel` (so the same vertex indices are used in both clips of a pair).
    F = min(num_frames, nf), frames evenly sampled. Returns (verts[F,nv,3], mid_frame)."""
    nf, nv, vert, offset = anime_read(path)
    F = min(num_frames, nf)
    idx = np.linspace(0, nf - 1, F).round().astype(int)
    out = [(vert if i == 0 else vert + offset[i - 1])[sel].astype(np.float32) for i in idx]
    return np.stack(out), F // 2                                            # [F,nv,3], mid


def build_pairs(ds, max_verts, num_frames, pairs_per_id, max_ids, seed):
    """Group test records by identity; keep identities with >=2 clips (shared topology).
    For each, sample `pairs_per_id` distinct clip pairs. Returns a list of dicts with the
    two clips' verts [F,nv,3], their mid-frame indices, the shared vertex subset, and tags.
    Subset `sel` is chosen ONCE per identity from its nv (shared across that identity)."""
    by_id = collections.defaultdict(list)
    for rec in ds.records:
        by_id[rec["identity"]].append(rec)
    multi = {k: v for k, v in by_id.items() if len(v) >= 2}
    rng = np.random.default_rng(seed)
    ids = sorted(multi)
    if max_ids:
        ids = ids[:max_ids]
    pairs = []
    for ident in ids:
        recs = multi[ident]
        # shared topology within an identity -> one nv, one subset reused for every clip
        nf0, nv, _, _ = anime_read(recs[0]["path"])
        sel = (np.arange(nv) if nv <= max_verts
               else np.sort(rng.choice(nv, max_verts, replace=False)))
        npair = min(pairs_per_id, len(recs) * (len(recs) - 1) // 2)
        seen = set()
        tries = 0
        while len(seen) < npair and tries < 50 * npair:
            tries += 1
            i, j = rng.choice(len(recs), 2, replace=False)
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            va, ma = load_clip_verts(recs[i]["path"], num_frames, max_verts, sel)
            vb, mb = load_clip_verts(recs[j]["path"], num_frames, max_verts, sel)
            pairs.append({"ident": ident, "vA": va, "vB": vb, "mA": ma, "mB": mb,
                          "tagA": recs[i]["clip_id"], "tagB": recs[j]["clip_id"]})
    return pairs


@torch.no_grad()
def pair_metrics_feat(featA_q, vB_t, featB_t, device):
    """featA_q [nv,C] query-frame feats from clip A; featB_t [nv,C] target-frame feats from
    clip B; vB_t [nv,3] target positions. Match each query vertex i to top-cosine vertex in
    B; GT = i. Returns rank, hit@{1,5,10}, euclid err of top-1 vs GT pos (per vertex)."""
    nv = featA_q.shape[0]
    a = F.normalize(featA_q.float(), dim=-1)
    b = F.normalize(featB_t.float(), dim=-1)
    s = a @ b.T                                                             # [nv,nv] cosine
    gt = torch.arange(nv, device=device)
    gt_sim = s[gt, gt][:, None]
    rank = (s > gt_sim).sum(1) + 1
    top1 = s.argmax(1)
    v = torch.as_tensor(vB_t, dtype=torch.float32, device=device)
    eerr = (v[top1] - v[gt]).norm(dim=-1)
    return rank, (rank <= 1).float(), (rank <= 5).float(), (rank <= 10).float(), eerr


@torch.no_grad()
def pair_metrics_xyz(vA_q, vB_t, device, canon=False):
    """XYZ-NN baseline across clips: nearest vertex in B (by Euclid) to query vertex in A;
    GT = same index. canon=True applies per-pose center + unit-scale before matching."""
    a = torch.as_tensor(vA_q, dtype=torch.float32, device=device)
    b = torch.as_tensor(vB_t, dtype=torch.float32, device=device)
    if canon:
        a = a - a.mean(0, keepdim=True); a = a / (a.norm(dim=-1).max() + 1e-8)
        bc = b - b.mean(0, keepdim=True); bc = bc / (bc.norm(dim=-1).max() + 1e-8)
    else:
        bc = b
    d = torch.cdist(a, bc)                                                  # [nv,nv]
    nv = a.shape[0]
    gt = torch.arange(nv, device=device)
    gt_d = d[gt, gt][:, None]
    rank = (d < gt_d).sum(1) + 1                                            # smaller dist = better
    top1 = d.argmin(1)
    eerr = (b[top1] - b[gt]).norm(dim=-1)                                   # err in ORIGINAL B coords
    return rank, (rank <= 1).float(), (rank <= 5).float(), (rank <= 10).float(), eerr


def aggregate(name, per_pair):
    """per_pair: list of (rank,h1,h5,h10,eerr) tensors. Print + return row."""
    rank = torch.cat([p[0] for p in per_pair]).float()
    row = {
        "recall@1": torch.cat([p[1] for p in per_pair]).mean().item(),
        "recall@5": torch.cat([p[2] for p in per_pair]).mean().item(),
        "recall@10": torch.cat([p[3] for p in per_pair]).mean().item(),
        "median_rank": rank.median().item(),
        "euclid_err": torch.cat([p[4] for p in per_pair]).mean().item(),
    }
    print(f"[{name}] R@1 {row['recall@1']:.3f}  R@5 {row['recall@5']:.3f}  "
          f"R@10 {row['recall@10']:.3f}  medRank {row['median_rank']:.1f}  "
          f"Eerr {row['euclid_err']:.4f}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser(description="Zero-shot CROSS-CLIP 3D correspondence (DT4D-Animals)")
    ap.add_argument("config")
    ap.add_argument("--encoder", required=True, help="DENSE encoder.pt to evaluate")
    ap.add_argument("--max-ids", type=int, default=None, help="cap identities (smoke)")
    ap.add_argument("--pairs-per-id", type=int, default=3, help="clip pairs sampled per identity")
    ap.add_argument("--num-frames", type=int, default=16, help="frames per clip fed to encoder (F)")
    ap.add_argument("--max-verts", type=int, default=2048, help="fixed vertex-subset cap")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.data.num_frames = args.num_frames

    ds = DT4DAnimalsClips(cfg, "test")
    print(f"building cross-clip pairs (max_ids={args.max_ids}, pairs/id={args.pairs_per_id}, "
          f"F={args.num_frames}, max_verts={args.max_verts})...", flush=True)
    pairs = build_pairs(ds, args.max_verts, args.num_frames, args.pairs_per_id,
                        args.max_ids, args.seed)
    n_ids = len({p["ident"] for p in pairs})
    print(f"{len(pairs)} cross-clip pairs over {n_ids} identities.", flush=True)
    if not pairs:
        raise RuntimeError("No multi-clip identities in the test split -- cannot build cross-clip eval.")

    # sanity: report pose divergence between paired query/target frames (why XYZ should fail)
    div = []
    for p in pairs:
        a = torch.tensor(p["vA"][p["mA"]]); b = torch.tensor(p["vB"][p["mB"]])
        div.append(float((a - b).norm(dim=-1).mean()))
    print(f"mean per-vertex pose divergence between paired frames: {np.mean(div):.3f} "
          f"(min {np.min(div):.3f}, max {np.max(div):.3f})", flush=True)

    dense = build_encoder(cfg.encoder)
    state = torch.load(args.encoder, map_location="cpu")
    dense.load_state_dict(state["encoder"])
    dense = dense.to(device).eval()

    torch.manual_seed(cfg.seed)
    rand = build_encoder(cfg.encoder).to(device).eval()

    acc = {m: [] for m in ("dense", "random", "xyz_raw", "xyz_canon")}
    for p in pairs:
        vA, vB, mA, mB = p["vA"], p["vB"], p["mA"], p["mB"]
        # feature methods: encode each clip once, take query/target frame feats
        for name, enc in (("dense", dense), ("random", rand)):
            fA = vertex_feats(enc, vA, device)[mA]                          # [nv,C]
            fB = vertex_feats(enc, vB, device)[mB]
            acc[name].append(pair_metrics_feat(fA, vB[mB], fB, device))
        acc["xyz_raw"].append(pair_metrics_xyz(vA[mA], vB[mB], device, canon=False))
        acc["xyz_canon"].append(pair_metrics_xyz(vA[mA], vB[mB], device, canon=True))

    print("\n=== per-method aggregate ===", flush=True)
    rows = {m: aggregate(m, acc[m]) for m in ("dense", "random", "xyz_raw", "xyz_canon")}

    print("\n========== RESULTS (zero-shot CROSS-CLIP 3D correspondence, DT4D-Animals) ==========",
          flush=True)
    print(f"{'method':<12} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} "
          f"{'medRank':>8} {'EuclidErr':>10}", flush=True)
    for m in ("dense", "random", "xyz_raw", "xyz_canon"):
        r = rows[m]
        print(f"{m:<12} {r['recall@1']:>9.3f} {r['recall@5']:>9.3f} {r['recall@10']:>10.3f} "
              f"{r['median_rank']:>8.1f} {r['euclid_err']:>10.4f}", flush=True)
    print("\nVerdict regime: query/target are DIFFERENT poses of the same mesh -> XYZ-NN should "
          "collapse (raw and canonicalized). If dense >> XYZ here, the dense-feature "
          "correspondence capability holds in the geometry-unreliable regime.", flush=True)


if __name__ == "__main__":
    main()
