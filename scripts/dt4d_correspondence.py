#!/usr/bin/env python
"""TRUE per-point QUANTITATIVE temporal correspondence eval on DeformingThings4D-Animals.

The paper's thesis is per-point density. This is a per-point quantitative test that needs
NO semantic labels: DT4D animals are animated meshes whose vertices CORRESPOND across
frames (vertex k is the SAME material point in every frame, fixed topology) => free
ground-truth correspondence (GT match = identity, k<->k). DT4DAnimalsClips.load_corr
returns correspondence-preserving verts [F,N,3] with FIXED vertex identity across frames
(NO per-frame resampling), so the GT is exact.

Protocol (frozen encoder, parameter-free / linear-free, no feature standardization):
  1. Load clip verts [F,N,3] (FIXED vertex ids; N = min(orig_nv, max_verts)).
  2. Run the FROZEN encoder once -> M tokens with token_xyz (orig frame) + token_t.
  3. Interpolate the M tokens to ALL N points per frame via 3-NN inverse-distance
     weighting within the SAME frame (the PointNet++ seg-readout propagation pattern,
     applied to every point not just the tokens) -> per-point feats [F,N,C].
  4. For frame pair (t, t+D) with D giving real deformation (D=T/2 and D=T-1):
     for each point i in frame t, find its nearest-FEATURE neighbor j among the N
     points of frame t+D (cosine; no standardization). A match is CORRECT if
        (a) j == i  (exact GT vertex index), OR
        (b) ||pos_{t+D}[j] - pos_{t+D}[i]|| <= tau   (within tolerance of the GT point's
            true position at t+D; handles near-coincident / symmetric verts).
     tau is set per clip as a fraction of the clip's bbox diagonal (scale-invariant).
  Report: correspondence accuracy (% correct under tau), exact-index accuracy (% j==i),
  and top-5 (GT index in the 5 nearest-feature neighbors). Aggregated over clips, over
  all valid (t,t+D) pairs, and over the supplied encoders (seeds) as mean +/- std.

Methods, IDENTICAL pipeline: RANDOM (random-init encoder), GLOBAL (global-distilled
encoders), DENSE (dense-distilled encoders), and XYZ (feature = raw xyz; geometry-only
reference). The GLOBAL encoder has the SAME architecture and DOES emit per-point tokens;
its global distillation loss only supervised the pooled clip vector, so its per-point
tokens are expected to carry little correspondence signal -- that contrast is the point.
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
def point_feats(enc, verts, device, k=3):
    """verts [F,N,3] -> per-point features [F,N,C] from the k NEAREST tokens in the SAME
    frame (knn on token_xyz grouped by token_t), inverse-distance weighted. Point4D pools
    each frame to ~64 tokens; IDW over the k nearest tokens propagates to ALL N points and
    gives each point a position-dependent feature (k=1 reduces to nearest-token)."""
    F_, N, _ = verts.shape
    pts = torch.as_tensor(verts, dtype=torch.float32, device=device)[None]   # [1,F,N,3]
    out = enc(pts)
    tok = out["tokens"][0]                                                    # [M,C]
    txyz = out["token_xyz"][0]                                                # [M,3]
    ttok = out["token_t"][0]                                                  # [M]
    C = tok.shape[1]
    feats = torch.zeros(F_, N, C, device=device)
    for f in range(F_):
        sel = ttok == f
        if not sel.any():
            continue
        tx, tf = txyz[sel], tok[sel]                                          # [m,3],[m,C]
        kk = min(k, tx.shape[0])
        d = torch.cdist(pts[0, f], tx)                                        # [N,m]
        dk, idk = d.topk(kk, dim=-1, largest=False)                          # [N,kk]
        w = 1.0 / (dk + 1e-8); w = w / w.sum(-1, keepdim=True)               # IDW weights
        feats[f] = (w[..., None] * tf[idk]).sum(1)                           # [N,C]
    return feats                                                             # [F,N,C]


@torch.no_grad()
def corr_metrics(feats, verts, delta, tau, device):
    """feats [F,N,C], verts [F,N,3]. For every valid pair (t,t+delta), match each point i
    in t to its nearest-FEATURE point j in t+delta (cosine). GT match index = i.
    Returns concatenated per-pair-per-point arrays:
      acc_tau : 1 if (j==i) OR ||pos_{t+d}[j]-pos_{t+d}[i]|| <= tau else 0
      acc_idx : 1 if j==i (exact GT vertex index)
      top5    : 1 if i is among the 5 nearest-feature neighbors of i."""
    F_, N, _ = feats.shape
    fn = F.normalize(feats.float(), dim=-1)                                   # cosine via dot
    v = torch.as_tensor(verts, dtype=torch.float32, device=device)
    gt = torch.arange(N, device=device)
    acc_tau, acc_idx, top5 = [], [], []
    for t in range(F_ - delta):
        s = fn[t] @ fn[t + delta].T                                          # [N,N] cosine
        top1 = s.argmax(1)                                                    # [N] matched index
        idx_hit = (top1 == gt)                                                # exact GT index
        pos_err = (v[t + delta][top1] - v[t + delta][gt]).norm(dim=-1)        # dist(matched, GT pos)
        tau_hit = idx_hit | (pos_err <= tau)
        k5 = min(5, N)
        top5_idx = s.topk(k5, dim=1).indices                                  # [N,5]
        t5_hit = (top5_idx == gt[:, None]).any(1)
        acc_tau.append(tau_hit.float())
        acc_idx.append(idx_hit.float())
        top5.append(t5_hit.float())
    if not acc_tau:
        return None
    return {"acc_tau": torch.cat(acc_tau), "acc_idx": torch.cat(acc_idx),
            "top5": torch.cat(top5)}


def run_method(name, feat_fn, clips, deltas, tau_frac, device, verbose=True):
    """feat_fn(verts)->[F,N,C]. tau per clip = tau_frac * bbox-diagonal of the clip.
    Aggregate over clips for each delta. Returns {delta: {acc_tau, acc_idx, top5}} means."""
    acc = {d: {k: [] for k in ("acc_tau", "acc_idx", "top5")} for d in deltas}
    for verts in clips:
        v = verts.reshape(-1, 3)
        diag = float(np.linalg.norm(v.max(0) - v.min(0)))
        tau = tau_frac * diag
        feats = feat_fn(verts)
        for d in deltas:
            m = corr_metrics(feats, verts, d, tau, device)
            if m is None:
                continue
            for kk in acc[d]:
                acc[d][kk].append(m[kk])
    rows = {}
    for d in deltas:
        rows[d] = {kk: torch.cat(acc[d][kk]).mean().item() for kk in ("acc_tau", "acc_idx", "top5")}
        if verbose:
            print(f"[{name}] D={d}: acc(tau) {rows[d]['acc_tau']:.4f}  "
                  f"acc(exact-idx) {rows[d]['acc_idx']:.4f}  top5 {rows[d]['top5']:.4f}",
                  flush=True)
    return rows


def load_clips(cfg, max_clips, num_frames, max_verts, deltas):
    ds = DT4DAnimalsClips(cfg, "test")
    clips = []
    skipped = 0
    for rec in ds.records:
        if max_clips and len(clips) >= max_clips:
            break
        try:
            verts, _ = ds.load_corr(rec, max_verts=max_verts, num_frames=num_frames)
        except Exception as e:
            skipped += 1
            continue
        if verts.shape[0] > max(deltas):                                     # need >=1 valid pair
            clips.append(verts)
    print(f"loaded {len(clips)} usable test clips (skipped {skipped} unreadable .anime), "
          f"F={num_frames}, N(verts)={max_verts}", flush=True)
    return clips


def agg_seeds(per_seed_rows, deltas):
    """per_seed_rows: list of {delta:{metric:val}}. Return {delta:{metric:(mean,std)}}."""
    out = {}
    for d in deltas:
        out[d] = {}
        for kk in ("acc_tau", "acc_idx", "top5"):
            vals = np.array([r[d][kk] for r in per_seed_rows])
            out[d][kk] = (float(vals.mean()), float(vals.std()))
    return out


def main():
    ap = argparse.ArgumentParser(description="Per-point temporal correspondence on DT4D-Animals")
    ap.add_argument("config")
    ap.add_argument("--dense", nargs="*", default=[], help="dense-distilled encoder.pt (seeds)")
    ap.add_argument("--global-enc", nargs="*", default=[], dest="global_enc",
                    help="global-distilled encoder.pt (seeds)")
    ap.add_argument("--max-clips", type=int, default=50, help="cap test clips")
    ap.add_argument("--num-frames", type=int, default=16, help="frames per clip (T=F)")
    ap.add_argument("--max-verts", type=int, default=2048, help="fixed vertex count N")
    ap.add_argument("--deltas", type=int, nargs="+", default=None,
                    help="frame gaps; default = [T//2, T-1]")
    ap.add_argument("--tau-frac", type=float, default=0.02,
                    help="tolerance as fraction of per-clip bbox diagonal")
    ap.add_argument("--rand-seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="seeds for random-init encoders")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.data.num_frames = args.num_frames
    T = args.num_frames
    deltas = args.deltas if args.deltas is not None else [T // 2, T - 1]

    print(f"=== DT4D-Animals temporal correspondence | T={T} deltas={deltas} "
          f"tau_frac={args.tau_frac} (of bbox diag) | GT=vertex identity (k<->k) ===", flush=True)
    clips = load_clips(cfg, args.max_clips, args.num_frames, args.max_verts, deltas)
    if not clips:
        print("NO usable clips -- abort.", flush=True)
        return

    results = {}  # method -> agg_seeds output

    # ---- RANDOM (random-init, multiple seeds) ----
    print("\n=== RANDOM-init ===", flush=True)
    rand_rows = []
    for s in args.rand_seeds:
        torch.manual_seed(s)
        rand = build_encoder(cfg.encoder).to(device).eval()
        rand_rows.append(run_method(f"random_s{s}", lambda v: point_feats(rand, v, device),
                                    clips, deltas, args.tau_frac, device))
    results["random"] = agg_seeds(rand_rows, deltas)

    # ---- GLOBAL (global-distilled, identical pipeline) ----
    if args.global_enc:
        print("\n=== GLOBAL (global-distilled) ===", flush=True)
        g_rows = []
        for ck in args.global_enc:
            enc = build_encoder(cfg.encoder)
            enc.load_state_dict(torch.load(ck, map_location="cpu")["encoder"])
            enc = enc.to(device).eval()
            print(f"-- {ck}", flush=True)
            g_rows.append(run_method(os.path.basename(os.path.dirname(ck)),
                                     lambda v: point_feats(enc, v, device),
                                     clips, deltas, args.tau_frac, device))
        results["global"] = agg_seeds(g_rows, deltas)

    # ---- DENSE (dense-distilled, identical pipeline) ----
    if args.dense:
        print("\n=== DENSE (dense-distilled) ===", flush=True)
        d_rows = []
        for ck in args.dense:
            enc = build_encoder(cfg.encoder)
            enc.load_state_dict(torch.load(ck, map_location="cpu")["encoder"])
            enc = enc.to(device).eval()
            print(f"-- {ck}", flush=True)
            d_rows.append(run_method(os.path.basename(os.path.dirname(ck)),
                                     lambda v: point_feats(enc, v, device),
                                     clips, deltas, args.tau_frac, device))
        results["dense"] = agg_seeds(d_rows, deltas)

    # ---- XYZ geometry-only reference (deterministic; std=0) ----
    print("\n=== XYZ-NN (raw geometry reference) ===", flush=True)
    xyz_rows = [run_method("xyz",
                           lambda v: torch.as_tensor(v, dtype=torch.float32, device=device),
                           clips, deltas, args.tau_frac, device)]
    results["xyz"] = agg_seeds(xyz_rows, deltas)

    # ---- final table ----
    print("\n================ RESULTS (per-point temporal correspondence, DT4D-Animals) ================",
          flush=True)
    print(f"#clips={len(clips)}  N={args.max_verts}  T={T}  tau_frac={args.tau_frac}*bboxdiag  "
          f"GT=vertex-identity  seeds(rand)={args.rand_seeds}", flush=True)
    order = [m for m in ("random", "global", "dense", "xyz") if m in results]
    for d in deltas:
        print(f"\n-- Delta = {d} --", flush=True)
        print(f"{'method':<10} {'acc@tau (mean+/-std)':>24} {'exact-idx':>20} {'top5':>20}",
              flush=True)
        for m in order:
            at = results[m][d]["acc_tau"]; ai = results[m][d]["acc_idx"]; t5 = results[m][d]["top5"]
            print(f"{m:<10} {at[0]:>12.4f} +/- {at[1]:.4f}   {ai[0]:>9.4f} +/- {ai[1]:.4f}   "
                  f"{t5[0]:>9.4f} +/- {t5[1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
