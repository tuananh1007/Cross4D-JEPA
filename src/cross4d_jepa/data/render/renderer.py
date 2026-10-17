"""Render the teacher's 2D views from geometry.

For datasets without usable RGB (MSR-Action3D is depth-only), render a short
multi-view video per clip to feed the frozen teacher, and RECORD pi (intrinsics +
extrinsics) per view for predictor conditioning.

Implementation: a dependency-free, pure-PyTorch z-buffered point splatter (no
PyTorch3D / Blender — those are painful to install on the cluster). Points are
projected with a pinhole camera, splatted to a small disk, z-buffered (nearest
wins), and shaded by depth (closer = brighter). Runs on CPU or GPU. `pi` is the
flattened K(9) + [R|t](12) = 21-vector matching PredictorConfig.pi_dim.
"""
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch


@dataclass
class RenderConfig:
    image_size: int = 256                      # match V-JEPA 2 crop (256)
    num_cameras: int = 2                       # start with 1-2; ablate view count
    shading: str = "depth"                     # depth | normal (depth = MSR default)
    point_radius: int = 2                      # splat half-width in px (denser silhouette @256)
    fov_deg: float = 50.0
    cam_radius: float = 2.5                    # camera distance (in normalized units)
    elevations_deg: List[float] = field(default_factory=lambda: [10.0, 10.0])
    azimuths_deg: List[float] = field(default_factory=lambda: [0.0, 60.0])
    teacher_window: int = 64                   # frames fed to the teacher (V-JEPA 2 fpc64)
    window_src: int = 12                        # source frames in the teacher's window ENDING at tau
                                               # (fixed-length, sliding -> horizons differ -> forecasting signal;
                                               # a growing-from-0 window made horizons ~identical)


# ---- camera construction ---------------------------------------------------
def _look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """World->camera [R|t] (3x4) for a camera at `eye` looking at `target`."""
    eye, target, up = map(lambda a: np.asarray(a, np.float32), (eye, target, up))
    f = target - eye
    f /= np.linalg.norm(f) + 1e-8
    s = np.cross(f, up)
    s /= np.linalg.norm(s) + 1e-8
    u = np.cross(s, f)
    R = np.stack([s, u, f], axis=0)            # rows: right, up, forward (z into scene)
    t = -R @ eye
    return np.concatenate([R, t[:, None]], axis=1).astype(np.float32)  # [3,4]


def make_cameras(cfg: RenderConfig, centroid=(0, 0, 0), scale=1.0) -> Tuple[List[np.ndarray], np.ndarray]:
    """Return per-camera (K[3,3], Rt[3,4]) and the flattened pi [V,21]."""
    H = W = cfg.image_size
    fpx = 0.5 * H / np.tan(np.radians(cfg.fov_deg) / 2.0)
    K = np.array([[fpx, 0, W / 2.0], [0, fpx, H / 2.0], [0, 0, 1]], np.float32)
    cams, pis = [], []
    centroid = np.asarray(centroid, np.float32)
    for i in range(cfg.num_cameras):
        el = np.radians(cfg.elevations_deg[i % len(cfg.elevations_deg)])
        az = np.radians(cfg.azimuths_deg[i % len(cfg.azimuths_deg)])
        r = cfg.cam_radius * scale
        eye = centroid + r * np.array([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)], np.float32)
        Rt = _look_at(eye, centroid)
        cams.append((K, Rt))
        pis.append(np.concatenate([K.reshape(-1), Rt.reshape(-1)]))   # 9 + 12 = 21
    return cams, np.stack(pis).astype(np.float32)


# ---- rasterization ---------------------------------------------------------
def _project(points: torch.Tensor, K: torch.Tensor, Rt: torch.Tensor):
    """points [N,3] world -> pixel (u,v) [N,2] and camera-space depth z [N]."""
    N = points.shape[0]
    ph = torch.cat([points, torch.ones(N, 1, device=points.device, dtype=points.dtype)], 1)
    cam = ph @ Rt.T                                 # [N,3] camera coords
    z = cam[:, 2].clamp(min=1e-4)
    uv = (cam @ K.T)                                # [N,3]
    uv = uv[:, :2] / z[:, None]
    return uv, z


def render_frame(points: torch.Tensor, K: torch.Tensor, Rt: torch.Tensor,
                 H: int, W: int, radius: int = 1) -> torch.Tensor:
    """Render one point frame -> depth-shaded image [3,H,W] in [0,1]."""
    uv, z = _project(points, K, Rt)
    u = uv[:, 0].round().long()
    v = uv[:, 1].round().long()
    zbuf = torch.full((H * W,), float("inf"), device=points.device, dtype=points.dtype)
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & torch.isfinite(z)
            if ok.any():
                idx = (vv[ok] * W + uu[ok])
                zbuf.scatter_reduce_(0, idx, z[ok], reduce="amin", include_self=True)
    zbuf = zbuf.view(H, W)
    valid = torch.isfinite(zbuf)
    if valid.any():
        zmin = zbuf[valid].min()
        zmax = zbuf[valid].max()
        shaded = torch.zeros_like(zbuf)
        shaded[valid] = 1.0 - (zbuf[valid] - zmin) / (zmax - zmin + 1e-6)  # closer = brighter
    else:
        shaded = torch.zeros_like(zbuf)
    return shaded[None].repeat(3, 1, 1)              # [3,H,W]


def render_frame_index(points: torch.Tensor, K: torch.Tensor, Rt: torch.Tensor,
                       H: int, W: int, radius: int = 1) -> torch.Tensor:
    """Per-pixel NEAREST-point index map [H,W] (-1 = empty) -- the 2D<->3D
    correspondence for DENSE cross-modal distillation: lets us pull the teacher's
    2D patch feature at each pixel back onto the 3D point it depicts."""
    uv, z = _project(points, K, Rt)
    u, v = uv[:, 0].round().long(), uv[:, 1].round().long()
    pid = torch.arange(points.shape[0], device=points.device)
    pix, pids, zs = [], [], []
    for du in range(-radius, radius + 1):
        for dv in range(-radius, radius + 1):
            uu, vv = u + du, v + dv
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & torch.isfinite(z)
            pix.append(vv[ok] * W + uu[ok]); pids.append(pid[ok]); zs.append(z[ok])
    pix, pids, zs = torch.cat(pix), torch.cat(pids), torch.cat(zs)
    # deterministic nearest-wins: scatter-reduce the per-pixel MIN depth, then keep
    # the point(s) that achieve it. (Plain idxbuf[pix]=pid is unsafe for duplicate
    # pixels -- PyTorch doesn't define which duplicate write wins.)
    zbuf = torch.full((H * W,), float("inf"), device=points.device, dtype=zs.dtype)
    zbuf.scatter_reduce_(0, pix, zs, reduce="amin", include_self=True)
    win = zs == zbuf[pix]                                  # winners (ties share equal z)
    idxbuf = torch.full((H * W,), -1, dtype=torch.long, device=points.device)
    idxbuf[pix[win]] = pids[win]
    return idxbuf.view(H, W)


def dense_point_targets(idx_hw: torch.Tensor, patch_feats: torch.Tensor) -> torch.Tensor:
    """Scatter teacher patch features onto points via the correspondence.
    idx_hw [H,W] nearest-point indices (-1 empty); patch_feats [Hp,Wp,D] teacher
    grid (upsampled to H,W by nearest). Returns (sum[N,D], count[N]) accumulators so
    callers can average over frames/views. N inferred from idx_hw.max()+1 -> pass N."""
    H, W = idx_hw.shape
    Hp, Wp, D = patch_feats.shape
    # nearest-upsample patch grid to pixel grid
    yi = (torch.arange(H, device=idx_hw.device) * Hp // H).clamp(max=Hp - 1)
    xi = (torch.arange(W, device=idx_hw.device) * Wp // W).clamp(max=Wp - 1)
    feat_hw = patch_feats[yi][:, xi]                       # [H,W,D]
    flat_idx = idx_hw.reshape(-1)
    vis = flat_idx >= 0
    return flat_idx[vis], feat_hw.reshape(-1, D)[vis]      # (point ids, their teacher feats)


def render_video(points: torch.Tensor, K, Rt, cfg: RenderConfig) -> torch.Tensor:
    """points [F,N,3] -> rendered frames [F,3,H,W]."""
    H = W = cfg.image_size
    K = torch.as_tensor(K, dtype=points.dtype, device=points.device)
    Rt = torch.as_tensor(Rt, dtype=points.dtype, device=points.device)
    return torch.stack([render_frame(points[f], K, Rt, H, W, cfg.point_radius)
                        for f in range(points.shape[0])])


def render_clip_views(points: torch.Tensor, cfg: RenderConfig) -> Tuple[torch.Tensor, np.ndarray]:
    """points [F,N,3] (a whole video) -> (views [V,F,3,H,W], pi [V,21]).

    Cameras are placed on an arc around the sequence centroid, scaled to the
    sequence extent so the subject fills the frame regardless of dataset units.
    """
    pts = torch.as_tensor(points, dtype=torch.float32)
    centroid = pts.reshape(-1, 3).mean(0)
    scale = (pts.reshape(-1, 3) - centroid).abs().max().item() + 1e-6
    cams, pi = make_cameras(cfg, centroid=centroid.tolist(), scale=scale)
    views = torch.stack([render_video(pts, K, Rt, cfg) for (K, Rt) in cams])  # [V,F,3,H,W]
    return views, pi
