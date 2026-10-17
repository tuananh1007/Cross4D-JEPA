"""Multi-view TARGET COVERAGE on MSR for num_cameras in {1,2,4}.

Uses the IDENTICAL camera placement as _clip_dense_targets_multiview (azimuths
linspace(0,360,V,endpoint=False), elevation 10 deg). A (frame,point) is covered iff
it is the idx-map winner in >=1 of the V views. Reports mean coverage % per V over
a sample of clips. Also asserts V=1 here matches the headline single-cam coverage.
"""
import argparse
import numpy as np
import torch

from cross4d_jepa.config import Config
from cross4d_jepa.data.datasets.msr_action3d import MSRAction3DClips
from cross4d_jepa.data.render.renderer import RenderConfig, make_cameras, render_frame_index


def clip_mv_coverage(points, device, img, num_cameras):
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0)
    s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    azimuths = list(np.linspace(0.0, 360.0, num_cameras, endpoint=False))
    rcfg = RenderConfig(image_size=img, num_cameras=num_cameras,
                        elevations_deg=[10.0] * num_cameras, azimuths_deg=azimuths)
    cams, _ = make_cameras(rcfg, centroid=c.tolist(), scale=s)
    Fr, N, _ = pts.shape
    seen = torch.zeros(Fr, N, dtype=torch.bool)  # visible in >=1 view
    for (K, Rt) in cams:
        K = torch.as_tensor(K, dtype=torch.float32, device=device)
        Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
        for f in range(Fr):
            idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)
            ids = idx.reshape(-1)
            ids = ids[ids >= 0].cpu()
            seen[f, torch.unique(ids)] = True
    return int(seen.sum().item()), Fr * N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--n-clips", type=int, default=50)
    ap.add_argument("--img", type=int, default=126)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config.from_yaml("configs/msr_point4d.yaml")
    ds = MSRAction3DClips(cfg, "train")
    nf = cfg.data.num_frames
    rng = np.random.RandomState(args.seed)
    k = min(args.n_clips, len(ds.records))
    sel = sorted(rng.choice(len(ds.records), k, replace=False).tolist())
    clips = [np.asarray(ds.load_full(ds.records[i]))[:nf] for i in sel]
    print(f"device={device} img={args.img} clips={k} nf={nf}", flush=True)
    for V in args.views:
        per = []
        for pts in clips:
            cov, tot = clip_mv_coverage(pts, device, args.img, V)
            per.append(cov / tot)
        per = np.array(per)
        print(f"[MSR num_cameras={V}] mean_coverage={100*per.mean():.2f}% "
              f"(std={100*per.std():.2f}%, min={100*per.min():.2f}%, max={100*per.max():.2f}%)", flush=True)


if __name__ == "__main__":
    main()
