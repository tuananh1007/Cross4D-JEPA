"""Point/token -> teacher-patch assignment statistics (Eqs. 3-4 of the paper).

Two distinct many-to-one effects share the same cause -- a finite teacher patch grid:

  (a) POINT -> PATCH  (Eq. 3): several visible points project into one patch cell and
      therefore receive the SAME teacher feature.
  (b) TOKEN -> TARGET (Eq. 4): student tokens are FPS centroids matched to their
      nearest input point; two tokens whose points fall in one cell get one target.

Only (b) enters the loss, because the objective is per TOKEN, not per raw point. This
script measures both at the shipped setting and across teacher grid sizes, so the
accuracy sweep over render resolution can be read as a collision sweep.

Token centroids come from FPS on the input points and are therefore WEIGHT-INDEPENDENT,
so a freshly built encoder gives the same assignment as a trained one.

  python -m scripts.token_collision configs/msr_point4d.yaml --imgs 70 98 126 168 224
"""
import argparse
from collections import Counter

import numpy as np
import torch

from cross4d_jepa.config import Config
from cross4d_jepa.engine.linear_probe import build_labeled_splits
from cross4d_jepa.models import build_encoder
from cross4d_jepa.data.render.renderer import RenderConfig, make_cameras, render_frame_index
from cross4d_jepa.utils.logging import get_logger

log = get_logger()


def _patch_of_points(pts_f, K, Rt, img, patch):
    """-> (point ids visible in this frame, the patch-cell id each one lands in)."""
    idx = render_frame_index(pts_f, K, Rt, img, img, 1)          # [H,W] nearest-point ids
    H = W = img
    hp = wp = img // patch
    ys, xs = torch.meshgrid(torch.arange(H, device=idx.device),
                            torch.arange(W, device=idx.device), indexing="ij")
    cell = (ys * hp // H).clamp(max=hp - 1) * wp + (xs * wp // W).clamp(max=wp - 1)
    vis = idx >= 0
    return idx[vis].reshape(-1), cell[vis].reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--imgs", type=int, nargs="+", default=[126])
    ap.add_argument("--patch", type=int, default=14)
    ap.add_argument("--max-clips", type=int, default=60)
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds, _ = build_labeled_splits(cfg)
    base = train_ds.base
    recs = base.records[:args.max_clips]
    nf = cfg.data.num_frames

    enc = build_encoder(cfg.encoder).to(device).eval()
    n2 = int(cfg.encoder.npoints[-1])

    print(f"\nclips={len(recs)}  frames/clip={nf}  points/frame={cfg.data.num_points}  "
          f"student tokens/frame n2={n2}\n")
    hdr = (f"{'img':>5} {'grid':>7} {'cells':>6} {'vis pts':>8} {'pts/occ':>8} "
           f"{'occ cells':>10} {'tok/cell':>9} {'distinct tgt':>13} {'eq4 dup':>8}")
    print(hdr); print("-" * len(hdr))

    for img in args.imgs:
        hp = img // args.patch
        vis_pts, pts_per_occ, occ_cells, tok_per_cell, distinct, eq4dup, nfr = [], [], [], [], [], [], 0
        for rec in recs:
            pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)
            c = pts.reshape(-1, 3).mean(0)
            s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
            cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1),
                                   centroid=c.tolist(), scale=s)
            K, Rt = cams[0]
            K = torch.as_tensor(K, dtype=torch.float32, device=device)
            Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
            with torch.no_grad():
                out = enc(pts[None].to(device))
            if "token_xyz" not in out or "token_t" not in out:
                raise SystemExit("encoder must return token_xyz + token_t for Eq. 4 stats")
            tok_xyz, tok_t = out["token_xyz"][0], out["token_t"][0]
            for f in range(pts.shape[0]):
                pf = pts[f].to(device)
                ids, cells = _patch_of_points(pf, K, Rt, img, args.patch)
                if ids.numel() == 0:
                    continue
                nfr += 1
                vis_pts.append(ids.unique().numel())
                cc = Counter(cells.tolist())
                occ_cells.append(len(cc))
                pts_per_occ.append(np.mean([v for v in cc.values()]))
                # Eq. 4: tokens of THIS frame -> nearest input point -> its patch cell
                tf = tok_xyz[tok_t == f]                     # this frame's token centroids
                if tf.numel() == 0:
                    continue
                d = torch.cdist(tf, pf)                       # [M_f, N]
                nn = d.argmin(1)
                eq4dup.append(1.0 - nn.unique().numel() / max(1, nn.numel()))
                pid2cell = {int(i): int(cl) for i, cl in zip(ids.tolist(), cells.tolist())}
                tcells = [pid2cell[int(p)] for p in nn.tolist() if int(p) in pid2cell]
                if tcells:
                    tok_per_cell.append(len(tcells) / len(set(tcells)))
                    distinct.append(len(set(tcells)) / len(tcells))
        m = lambda a: float(np.mean(a)) if a else float("nan")
        print(f"{img:>5} {f'{hp}x{hp}':>7} {hp*hp:>6} {m(vis_pts):>8.0f} {m(pts_per_occ):>8.1f} "
              f"{m(occ_cells):>10.1f} {m(tok_per_cell):>9.2f} {m(distinct)*100:>12.1f}% "
              f"{m(eq4dup)*100:>7.1f}%")

    print("\npts/occ    = visible points sharing one occupied patch cell  (effect (a))")
    print("tok/cell   = supervised student tokens per occupied cell      (effect (b))")
    print("distinct tgt = share of supervised tokens with a UNIQUE target")
    print("eq4 dup    = share of tokens whose nearest input point is not unique\n")


if __name__ == "__main__":
    main()
