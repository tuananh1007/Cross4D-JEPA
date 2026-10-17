"""Per-point TARGET COVERAGE under the headline single-camera render.

Mirrors engine.pretrain_dense._clip_dense_targets EXACTLY: same RenderConfig
(image_size=img, num_cameras=1), same centroid/scale, same render_frame_index
(radius=1). A point n at frame f is "covered" iff its id appears in the per-pixel
nearest-point index map for that frame -- i.e. it is visible/unoccluded and wins a
pixel under the splat. That is precisely the set _clip_dense_targets writes a
non-zero teacher target to (tgt[f, ids] = ...).

Coverage (per clip) = (# covered (f,n) pairs) / (F * N), averaged over sampled clips.
Reports mean coverage % per dataset. Uses img=126 (headline default).
"""
import argparse
import numpy as np
import torch

from cross4d_jepa.config import Config
from cross4d_jepa.data.render.renderer import RenderConfig, make_cameras, render_frame_index


def clip_coverage(points, device, img):
    """points [F,N,3] -> (covered_count, total=F*N) under the single-camera render."""
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0)
    s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=device)
    Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
    Fr, N, _ = pts.shape
    covered = 0
    for f in range(Fr):
        idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)  # [H,W], -1 empty
        ids = idx.reshape(-1)
        ids = ids[ids >= 0]
        covered += int(torch.unique(ids).numel())  # distinct points that win >=1 pixel this frame
    return covered, Fr * N


def get_dataset(name, device):
    """Return (records, load_full, nf) for the train split, using the dataset's own
    config so num_frames/num_points match the headline cache for that dataset."""
    if name == "msr":
        from cross4d_jepa.data.datasets.msr_action3d import MSRAction3DClips
        cfg = Config.from_yaml("configs/msr_point4d.yaml")
        ds = MSRAction3DClips(cfg, "train")
        return ds.records, ds.load_full, cfg.data.num_frames
    if name == "dt4d":
        from cross4d_jepa.data.datasets.dt4d_animals import DT4DAnimalsClips
        cfg = Config.from_yaml("configs/dt4d_point4d.yaml")
        ds = DT4DAnimalsClips(cfg, "train")
        return ds.records, ds.load_full, cfg.data.num_frames
    if name == "hoi4d":
        from cross4d_jepa.data.datasets.hoi4d import HOI4DClips
        cfg = Config.from_yaml("configs/hoi4d_action_point4d.yaml")
        ds = HOI4DClips(cfg, "train")
        return ds.records, ds.load_full, cfg.data.num_frames
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["msr", "dt4d", "hoi4d"])
    ap.add_argument("--n-clips", type=int, default=50)
    ap.add_argument("--img", type=int, default=126)  # headline default
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} img={args.img} n_clips={args.n_clips} seed={args.seed}", flush=True)

    for name in args.datasets:
        records, load_full, nf = get_dataset(name, device)
        rng = np.random.RandomState(args.seed)
        k = min(args.n_clips, len(records))
        sel = sorted(rng.choice(len(records), k, replace=False).tolist())
        per_clip = []
        tot_cov, tot_all = 0, 0
        for j, i in enumerate(sel):
            pts = np.asarray(load_full(records[i]))[:nf]  # [F,N,3], first nf frames (as pretrain does)
            cov, tot = clip_coverage(pts, device, args.img)
            per_clip.append(cov / tot)
            tot_cov += cov
            tot_all += tot
        per_clip = np.array(per_clip)
        print(f"[{name}] clips={k} nf={nf} N={pts.shape[1]} "
              f"mean_coverage={100*per_clip.mean():.2f}% "
              f"(std={100*per_clip.std():.2f}%, micro={100*tot_cov/tot_all:.2f}%, "
              f"min={100*per_clip.min():.2f}%, max={100*per_clip.max():.2f}%)", flush=True)


if __name__ == "__main__":
    main()
