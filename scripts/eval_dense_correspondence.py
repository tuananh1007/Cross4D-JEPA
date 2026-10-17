#!/usr/bin/env python
"""Zero-shot 3D dense-correspondence eval on DeformingThings4D-Animals.

Showcases that DENSE-distilled per-point features enable cross-frame correspondence
with NO correspondence supervision -- a capability GLOBAL distillation cannot offer
(global has a single per-clip vector, no per-point features, so it can't be evaluated
here; that's the point).

Per test clip: load correspondence-preserving verts [F,nv,3] (FIXED vertex identity
across frames => free GT correspondence; vertex i is the SAME semantic point in every
frame). Run the encoder -> tokens + token_xyz + token_t. Build PER-VERTEX features by
assigning each vertex the feature of its NEAREST token within the SAME frame (knn on
token_xyz grouped by token_t) -> feats [F,nv,C].

Metric: for frames t and t+D, for each vertex i in frame t rank vertices in frame t+D
by feature cosine similarity; GT match = vertex i. Report recall@1/@5/@10, median rank,
and mean Euclidean error between the top-1-matched vertex and the GT vertex position.
Aggregated over clips and over all valid (t, t+D) frame pairs.

Methods (same pipeline): DENSE (distilled encoder.pt), RANDOM (random-init encoder),
XYZ (feature = vertex xyz; geometry alone is ambiguous under deformation).
"""
import argparse
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
    SAME frame (knn on token_xyz, grouped by token_t), inverse-distance weighted.

    Point4D pools each frame to only 64 tokens, so a pure nearest-token (k=1) assignment
    makes ~32 vertices share an identical feature -> cosine ties that swamp the metric for
    EVERY token-based method. Standard PointNet++ feature propagation (IDW over the k
    nearest tokens) gives each vertex a position-dependent feature and removes the ties;
    k=1 reduces to the literal nearest-token rule."""
    F_, nv, _ = verts.shape
    pts = torch.as_tensor(verts, dtype=torch.float32, device=device)[None]   # [1,F,nv,3]
    out = enc(pts)
    tok = out["tokens"][0]                                                   # [M,C]
    txyz = out["token_xyz"][0]                                               # [M,3]
    ttok = out["token_t"][0]                                                 # [M]
    C = tok.shape[1]
    feats = torch.zeros(F_, nv, C, device=device)
    for f in range(F_):
        sel = ttok == f
        if not sel.any():
            continue
        tx, tf = txyz[sel], tok[sel]                                         # [m,3],[m,C]
        kk = min(k, tx.shape[0])
        d = torch.cdist(pts[0, f], tx)                                       # [nv,m]
        dk, idk = d.topk(kk, dim=-1, largest=False)                         # [nv,kk]
        w = 1.0 / (dk + 1e-8); w = w / w.sum(-1, keepdim=True)              # IDW weights
        feats[f] = (w[..., None] * tf[idk]).sum(1)                          # [nv,C]
    return feats                                                            # [F,nv,C]


@torch.no_grad()
def corr_metrics(feats, verts, delta, device):
    """feats [F,nv,C], verts [F,nv,3]. For every valid frame pair (t,t+delta), match each
    vertex i in t to its top-cosine vertex in t+delta; GT = i. Returns per-pair-per-vertex
    arrays (concatenated): rank, hit@k flags, euclid err of the top-1 match vs GT pos."""
    F_, nv, _ = feats.shape
    fn = F.normalize(feats.float(), dim=-1)                                 # cosine via dot
    v = torch.as_tensor(verts, dtype=torch.float32, device=device)
    ranks, hit1, hit5, hit10, eerr = [], [], [], [], []
    gt = torch.arange(nv, device=device)
    for t in range(F_ - delta):
        s = fn[t] @ fn[t + delta].T                                         # [nv,nv] cosine sim
        # rank of the GT match = #vertices strictly more similar than the GT, +1
        gt_sim = s[gt, gt][:, None]                                         # [nv,1]
        rank = (s > gt_sim).sum(1) + 1                                      # [nv]
        top1 = s.argmax(1)                                                  # [nv]
        ranks.append(rank)
        hit1.append((rank <= 1).float())
        hit5.append((rank <= 5).float())
        hit10.append((rank <= 10).float())
        eerr.append((v[t + delta][top1] - v[t + delta][gt]).norm(dim=-1))   # dist(matched, GT)
    if not ranks:
        return None
    return {"rank": torch.cat(ranks), "hit1": torch.cat(hit1), "hit5": torch.cat(hit5),
            "hit10": torch.cat(hit10), "eerr": torch.cat(eerr)}


def run_method(name, feat_fn, clips, deltas, device):
    """feat_fn(verts)->[F,nv,C]. Aggregate metrics over clips for each delta."""
    acc = {d: {k: [] for k in ("rank", "hit1", "hit5", "hit10", "eerr")} for d in deltas}
    for verts in clips:
        feats = feat_fn(verts)
        for d in deltas:
            m = corr_metrics(feats, verts, d, device)
            if m is None:
                continue
            for k in acc[d]:
                acc[d][k].append(m[k])
    rows = {}
    for d in deltas:
        rank = torch.cat(acc[d]["rank"]).float()
        rows[d] = {
            "recall@1": torch.cat(acc[d]["hit1"]).mean().item(),
            "recall@5": torch.cat(acc[d]["hit5"]).mean().item(),
            "recall@10": torch.cat(acc[d]["hit10"]).mean().item(),
            "median_rank": rank.median().item(),
            "euclid_err": torch.cat(acc[d]["eerr"]).mean().item(),
        }
        print(f"[{name}] D={d}: R@1 {rows[d]['recall@1']:.3f}  R@5 {rows[d]['recall@5']:.3f}  "
              f"R@10 {rows[d]['recall@10']:.3f}  medRank {rows[d]['median_rank']:.1f}  "
              f"Eerr {rows[d]['euclid_err']:.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Zero-shot 3D dense-correspondence eval on DT4D-Animals")
    ap.add_argument("config")
    ap.add_argument("--encoder", required=True, help="DENSE encoder.pt to evaluate")
    ap.add_argument("--max-clips", type=int, default=None, help="cap test clips (smoke)")
    ap.add_argument("--num-frames", type=int, default=16, help="frames per clip (F)")
    ap.add_argument("--max-verts", type=int, default=2048, help="fixed vertex-subset cap")
    ap.add_argument("--deltas", type=int, nargs="+", default=[1, 4, 8])
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.data.num_frames = args.num_frames                                   # encoder sees F frames

    ds = DT4DAnimalsClips(cfg, "test")
    recs = ds.records[: args.max_clips] if args.max_clips else ds.records
    print(f"loading {len(recs)} test clips (F={args.num_frames}, max_verts={args.max_verts})...",
          flush=True)
    clips = []
    for rec in recs:
        verts, _ = ds.load_corr(rec, max_verts=args.max_verts, num_frames=args.num_frames)
        if verts.shape[0] > max(args.deltas):                               # need >=1 valid pair
            clips.append(verts)
    print(f"{len(clips)} clips usable.", flush=True)

    # DENSE: distilled encoder
    dense = build_encoder(cfg.encoder)
    state = torch.load(args.encoder, map_location="cpu")
    dense.load_state_dict(state["encoder"])
    dense = dense.to(device).eval()

    # RANDOM-init: same arch, untrained (seeded for reproducibility)
    torch.manual_seed(cfg.seed)
    rand = build_encoder(cfg.encoder).to(device).eval()

    deltas = args.deltas
    print("\n=== DENSE (distilled) ===", flush=True)
    r_dense = run_method("dense", lambda v: vertex_feats(dense, v, device), clips, deltas, device)
    print("\n=== RANDOM-init ===", flush=True)
    r_rand = run_method("random", lambda v: vertex_feats(rand, v, device), clips, deltas, device)
    print("\n=== XYZ-NN (raw geometry) ===", flush=True)
    r_xyz = run_method("xyz",
                       lambda v: torch.as_tensor(v, dtype=torch.float32, device=device),
                       clips, deltas, device)

    # results table
    print("\n================ RESULTS (zero-shot 3D correspondence, DT4D-Animals) ================",
          flush=True)
    for d in deltas:
        print(f"\n-- Delta = {d} --", flush=True)
        print(f"{'method':<10} {'recall@1':>9} {'recall@5':>9} {'recall@10':>10} "
              f"{'medRank':>8} {'EuclidErr':>10}", flush=True)
        for name, r in (("dense", r_dense), ("random", r_rand), ("xyz", r_xyz)):
            row = r[d]
            print(f"{name:<10} {row['recall@1']:>9.3f} {row['recall@5']:>9.3f} "
                  f"{row['recall@10']:>10.3f} {row['median_rank']:>8.1f} "
                  f"{row['euclid_err']:>10.4f}", flush=True)
    print("\nNote: global distillation has NO per-point features, so it cannot be evaluated "
          "on dense correspondence at all -- that is the differentiated capability.", flush=True)


if __name__ == "__main__":
    main()
