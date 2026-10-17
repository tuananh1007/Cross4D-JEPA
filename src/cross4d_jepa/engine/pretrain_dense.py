"""DENSE cross-modal distillation pretraining -- the Cross4D-JEPA method
(`--global` gives the per-clip global-distillation baseline).

Per clip: render -> per-pixel nearest-point index -> teacher (DINOv2) patch features
-> PER-FRAME per-point targets [F,N,Dp] (random-projected for memory), computed ONCE
and held on CPU. Student encoder -> tokens + token_xyz; a per-token head predicts the
teacher feature of each token's nearest input point; per-token cosine loss + VICReg
anti-collapse. Saves encoder.pt for the standard downstream (LP / fine-tune).
"""
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Teacher weights download from Hugging Face on the first cache build; export
# HF_HUB_OFFLINE=1 in the environment to force offline use of a pre-populated cache.

from ..config import Config
from ..models import build_encoder
from ..models.encoders.point4d import knn
from ..utils import get_logger
from .common import resolve_device
from .linear_probe import build_labeled_splits  # only to reuse the clip provider path

log = get_logger()


DINOV3_WEIGHTS = "weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
VIT_IN1K_ID = "google/vit-base-patch16-224"   # ImageNet-21k+1k SUPERVISED ViT-B/16
RAWPIX_PATCH = 14                             # rawpix grid = 126/14 = 9x9 (= DINOv2's)


def _teacher(device, teacher: str = "dinov2"):
    if teacher == "videomaev2":
        return _teacher_videomaev2(device)
    if teacher == "dinov3":
        # Official Meta DINOv3 ViT-L/16 .pth loaded into transformers DINOv3ViTModel.
        from ..models.teacher.dinov3 import load_dinov3_vitl16
        return load_dinov3_vitl16(DINOV3_WEIGHTS, device)
    if teacher == "rawpix":
        # NON-FOUNDATION control with NO network: a patch's "teacher feature" is its
        # raw rendered pixels, so the target carries render geometry and nothing
        # learned. Isolates 2D-foundation semantics from the dense correspondence task.
        return None, RAWPIX_PATCH * RAWPIX_PATCH * 3
    if teacher in {"vit-in1k", "vit-rand"}:
        # NON-FOUNDATION controls at DINOv2's 9x9 grid (patch16 @144px): an
        # ImageNet-supervised ViT-B/16, and the SAME architecture left UNTRAINED.
        from transformers import AutoConfig, AutoModel
        vc = AutoConfig.from_pretrained(VIT_IN1K_ID)
        m = (AutoModel.from_pretrained(VIT_IN1K_ID) if teacher == "vit-in1k"
             else AutoModel.from_config(vc))
        m = m.to(device).eval()
        for q in m.parameters():
            q.requires_grad_(False)
        return m, int(vc.hidden_size)
    from transformers import AutoModel
    mid = {"dinov2": "facebook/dinov2-large",
           "vjepa2": "facebook/vjepa2-vitl-fpc64-256"}[teacher]
    m = AutoModel.from_pretrained(mid).to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    d = int(getattr(m.config, "hidden_size", 1024))
    return m, d


def _teacher_videomaev2(device):
    """OpenGVLab/VideoMAEv2-Base -- a custom_code video ViT. Its remote modeling
    file targets transformers 4.38 and breaks the 5.x from_pretrained finalize path,
    so we load the class via the dynamic-module loader and load weights straight from
    the safetensors (verified: 0 missing / 0 unexpected keys). Returns the inner
    VisionTransformer (the part that emits patch tokens) and its embed_dim."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    repo = "OpenGVLab/VideoMAEv2-Base"
    ConfigCls = get_class_from_dynamic_module("modeling_config.VideoMAEv2Config", repo)
    ModelCls = get_class_from_dynamic_module("modeling_videomaev2.VideoMAEv2", repo)
    cfg = ConfigCls.from_pretrained(repo)
    m = ModelCls(cfg)
    m.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")), strict=True)
    vt = m.model.to(device).eval()                       # inner VisionTransformer
    for p in vt.parameters():
        p.requires_grad_(False)
    return vt, int(vt.embed_dim)


@torch.no_grad()
def _videomaev2_tokens(vt, video):
    """video [1,fpc,3,H,W] -> pre-pool patch tokens [1, gt*gh*gw, D] in (t,h,w) order.

    Mirrors VisionTransformer.forward_features up to (but NOT including) the mean-pool:
    Conv3d tubelet embed -> +sinusoidal pos_embed -> 12 blocks. fc_norm/head are the
    mean-pool + classifier we skip to keep per-token spatial features."""
    x = video.permute(0, 2, 1, 3, 4)                     # [1,fpc,3,H,W] -> [1,3,fpc,H,W] (Conv3d wants C,T)
    z = vt.patch_embed(x)                                # [1, gt*gh*gw, D]
    if vt.pos_embed is not None:
        z = z + vt.pos_embed.type_as(z).to(z.device)
    for blk in vt.blocks:
        z = blk(z)
    return z


def _clip_dense_targets(points, dino, proj, device, img, dp, patch=14, drop=1, interp=False):
    """points [F,N,3] -> per-frame per-point teacher targets [F,N,dp] (CPU, fp16).

    `patch`/`drop`/`interp` keep the DINOv2 default (14, 1, False) and additionally
    serve the patch-16 ViT controls, whose learned absolute position embeddings must
    be interpolated to the 9x9 grid."""
    from ..data.render.renderer import (RenderConfig, make_cameras, render_frame,
                                        render_frame_index, dense_point_targets)
    imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=device); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
    Fr, N, _ = pts.shape
    tgt = torch.zeros(Fr, N, dp, dtype=torch.float16)
    with torch.no_grad():
        imgs = torch.stack([render_frame(pts[f].to(device), K, Rt, img, img, 1) for f in range(Fr)])
        kw = {"interpolate_pos_encoding": True} if interp else {}
        feats = dino(pixel_values=(imgs - imean) / istd, **kw).last_hidden_state    # [F,drop+P,D]
        hp = wp = img // patch
        grid = (feats[:, drop:, :].reshape(Fr, hp, wp, -1) @ proj)                  # [F,hp,wp,dp]
        for f in range(Fr):
            idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)
            ids, fe = dense_point_targets(idx, grid[f])
            tgt[f, ids] = fe.half().cpu()
    return tgt


def _clip_dense_targets_rawpix(points, proj, device, img, dp, patch=RAWPIX_PATCH):
    """RAW-PIXEL "teacher" (no network): a patch's target is its flattened rendered
    pixels (patch*patch*3), random-projected exactly like a real teacher's tokens.
    Same render, same z-buffer correspondence, same cosine objective -- only the
    learned semantics are removed."""
    from ..data.render.renderer import (RenderConfig, make_cameras, render_frame,
                                        render_frame_index, dense_point_targets)
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=device); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
    Fr, N, _ = pts.shape
    tgt = torch.zeros(Fr, N, dp, dtype=torch.float16)
    hp = wp = img // patch
    with torch.no_grad():
        imgs = torch.stack([render_frame(pts[f].to(device), K, Rt, img, img, 1) for f in range(Fr)])
        pk = imgs.unfold(2, patch, patch).unfold(3, patch, patch)        # [F,3,hp,wp,p,p]
        pk = pk.permute(0, 2, 3, 1, 4, 5).reshape(Fr, hp, wp, -1)       # [F,hp,wp,p*p*3]
        grid = pk @ proj                                                 # [F,hp,wp,dp]
        for f in range(Fr):
            idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)
            ids, fe = dense_point_targets(idx, grid[f])
            tgt[f, ids] = fe.half().cpu()
    return tgt


def _clip_dense_targets_dinov3(points, dino, proj, device, img, dp):
    """DINOv3 DENSE targets via the SAME render correspondence as DINOv2.

    Identical to _clip_dense_targets except: DINOv3 is patch-16 (grid = img//16, so
    img=144 -> 9x9, matching DINOv2's 126/14=9), and it prepends CLS + 4 register
    tokens -> drop the first 5 tokens (NOT 1) to recover the patch grid."""
    from ..data.render.renderer import (RenderConfig, make_cameras, render_frame,
                                        render_frame_index, dense_point_targets)
    imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=device); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
    Fr, N, _ = pts.shape
    tgt = torch.zeros(Fr, N, dp, dtype=torch.float16)
    with torch.no_grad():
        imgs = torch.stack([render_frame(pts[f].to(device), K, Rt, img, img, 1) for f in range(Fr)])
        feats = dino(pixel_values=(imgs - imean) / istd).last_hidden_state         # [F,1+4+P,D]
        hp = wp = img // 16
        grid = (feats[:, 5:, :].reshape(Fr, hp, wp, -1) @ proj)                     # drop CLS+4 reg
        for f in range(Fr):
            idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)
            ids, fe = dense_point_targets(idx, grid[f])
            tgt[f, ids] = fe.half().cpu()
    return tgt


def _clip_dense_targets_multiview(points, dino, proj, device, img, dp, num_cameras):
    """Multi-view variant of _clip_dense_targets (#cameras ablation).

    Renders `num_cameras` DINOv2 views on an evenly-spaced azimuth arc around the
    clip centroid (azimuths linspace(0,360,num_cameras,endpoint=False); elevation 10
    deg, same cam_radius as the headline). For each input point and frame we AVERAGE
    the projected, random-projected teacher feature over exactly the views that SEE
    that point (idx-map winner under the render correspondence). A point visible in 0
    views at a frame keeps a zero target (stays masked, identical to the single-cam
    rule tg.abs().sum(-1)>0 in the training loop). num_cameras=1 reduces EXACTLY to
    _clip_dense_targets (azimuth 0, elevation 10 -> same single camera)."""
    from ..data.render.renderer import (RenderConfig, make_cameras, render_frame,
                                        render_frame_index, dense_point_targets)
    imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    azimuths = list(np.linspace(0.0, 360.0, num_cameras, endpoint=False))
    rcfg = RenderConfig(image_size=img, num_cameras=num_cameras,
                        elevations_deg=[10.0] * num_cameras, azimuths_deg=azimuths)
    cams, _ = make_cameras(rcfg, centroid=c.tolist(), scale=s)
    Fr, N, _ = pts.shape
    acc = torch.zeros(Fr, N, dp, dtype=torch.float32)   # sum of feats over visible views
    cnt = torch.zeros(Fr, N, 1, dtype=torch.float32)    # # views that saw each (f,n)
    with torch.no_grad():
        for (K, Rt) in cams:
            K = torch.as_tensor(K, dtype=torch.float32, device=device)
            Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
            imgs = torch.stack([render_frame(pts[f].to(device), K, Rt, img, img, 1) for f in range(Fr)])
            feats = dino(pixel_values=(imgs - imean) / istd).last_hidden_state      # [F,1+P,D]
            hp = wp = img // 14
            grid = (feats[:, 1:, :].reshape(Fr, hp, wp, -1) @ proj)                 # [F,hp,wp,dp]
            for f in range(Fr):
                idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)
                ids, fe = dense_point_targets(idx, grid[f])
                ids = ids.cpu()                                 # acc/cnt live on CPU
                acc[f, ids] += fe.float().cpu()
                cnt[f, ids] += 1.0
    tgt = torch.where(cnt > 0, acc / cnt.clamp(min=1.0), torch.zeros_like(acc))     # mean over visible views
    return tgt.half()


def _clip_dense_targets_vjepa2(points, vjepa, proj, device, img, dp):
    """DENSE V-JEPA2 targets via the SAME render correspondence as DINOv2.

    V-JEPA2 (facebook/vjepa2-vitl-fpc64-256) is a video ViT with spatio-temporal
    tubelet tokens: a Conv3d(kernel=(2,16,16), stride=(2,16,16)) over a 64x256x256
    clip then .flatten(2).transpose(1,2) -> [1, 8192, D] tokens flattened in (t,h,w)
    row-major order, i.e. tubelet-major then a 16x16 spatial grid. So
    last_hidden_state.reshape(32, 16, 16, D) is [T_tubelet, h, w, D] and tubelet i
    covers input frames {2i, 2i+1}.

    To get PER-FRAME spatial patch features we run V-JEPA2 ONCE per clip on a 64-frame
    256x256 render of the SAME camera, then for each of the nf rendered frames take the
    16x16 spatial slice of the tubelet that frame maps to (nearest-replicate tiling of
    nf->64, so input frame f sits at temporal position round(f/(nf-1)*63), tubelet =
    that//2). Each frame's 16x16 patch grid is then mapped onto points by the identical
    render_frame_index correspondence used for DINOv2 -- isolating granularity vs modality.
    """
    from ..data.render.renderer import (RenderConfig, make_cameras, render_frame,
                                        render_frame_index, dense_point_targets)
    imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=device); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
    Fr, N, _ = pts.shape
    fpc = int(getattr(vjepa.config, "frames_per_clip", 64))      # 64
    tub = int(getattr(vjepa.config, "tubelet_size", 2))          # 2
    ps = int(getattr(vjepa.config, "patch_size", 16))            # 16
    gh = gw = img // ps                                          # spatial grid (16 @ 256)
    gt = fpc // tub                                              # tubelet grid (32)
    tgt = torch.zeros(Fr, N, dp, dtype=torch.float16)
    with torch.no_grad():
        # render the nf frames once, tile nf->fpc by nearest-replicate (temporal nearest)
        rendered = torch.stack([render_frame(pts[f].to(device), K, Rt, img, img, 1) for f in range(Fr)])  # [Fr,3,H,W]
        ti = (torch.arange(fpc, device=device).float() / max(1, fpc - 1) * (Fr - 1)).round().long()       # [fpc] -> src frame
        clip = rendered[ti][None]                                # [1,fpc,3,H,W]
        out = vjepa(pixel_values_videos=(clip - imean) / istd).last_hidden_state  # [1,gt*gh*gw,D]
        gridT = out.reshape(gt, gh, gw, -1) @ proj               # [gt,gh,gw,dp]
        for f in range(Fr):
            tpos = round(f / max(1, Fr - 1) * (fpc - 1))         # this frame's temporal slot in the 64-clip
            tub_i = min(tpos // tub, gt - 1)
            idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)
            ids, fe = dense_point_targets(idx, gridT[tub_i])
            tgt[f, ids] = fe.half().cpu()
    return tgt


def _clip_dense_targets_videomaev2(points, vt, proj, device, img, dp):
    """DENSE VideoMAEv2 targets via the SAME render correspondence as DINOv2/V-JEPA2.

    VideoMAEv2-Base is a video ViT with a Conv3d(kernel=(2,16,16),stride=same) tubelet
    embed over a 16x224x224 clip -> 8x14x14 = 1568 tokens flattened (t,h,w) row-major
    (tubelet i covers input frames {2i,2i+1}). Identical structure to V-JEPA2, just
    fpc=16, img=224, 14x14 grid, 8 tubelets. We run the encoder ONCE per clip on a
    16-frame 224x224 render (nf->16 nearest-replicate tiling), reshape pre-pool tokens
    to [8,14,14,D], and for each rendered frame take the 14x14 spatial slice of the
    tubelet it maps to, then back-project onto points with render_frame_index -- the
    same correspondence used for the other teachers (isolating granularity vs modality)."""
    from ..data.render.renderer import (RenderConfig, make_cameras, render_frame,
                                        render_frame_index, dense_point_targets)
    imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
    istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=device); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
    Fr, N, _ = pts.shape
    fpc = int(vt.patch_embed.num_patches // ((img // vt.patch_embed.patch_size[0]) ** 2) * vt.tubelet_size)  # 16
    tub = int(vt.tubelet_size)                                   # 2
    ps = int(vt.patch_embed.patch_size[0])                       # 16
    gh = gw = img // ps                                          # spatial grid (14 @ 224)
    gt = fpc // tub                                              # tubelet grid (8)
    tgt = torch.zeros(Fr, N, dp, dtype=torch.float16)
    with torch.no_grad():
        rendered = torch.stack([render_frame(pts[f].to(device), K, Rt, img, img, 1) for f in range(Fr)])  # [Fr,3,H,W]
        ti = (torch.arange(fpc, device=device).float() / max(1, fpc - 1) * (Fr - 1)).round().long()       # [fpc] -> src frame
        clip = rendered[ti][None]                                # [1,fpc,3,H,W]
        out = _videomaev2_tokens(vt, (clip - imean) / istd)      # [1,gt*gh*gw,D]
        gridT = out.reshape(gt, gh, gw, -1) @ proj               # [gt,gh,gw,dp]
        for f in range(Fr):
            tpos = round(f / max(1, Fr - 1) * (fpc - 1))         # this frame's temporal slot in the 16-clip
            tub_i = min(tpos // tub, gt - 1)
            idx = render_frame_index(pts[f].to(device), K, Rt, img, img, 1)
            ids, fe = dense_point_targets(idx, gridT[tub_i])
            tgt[f, ids] = fe.half().cpu()
    return tgt


def _clip_global_target(points, teacher, proj, device, img, dp, is_video, is_vmae=False, is_dino3=False):
    """GLOBAL distillation target: the clip's MEAN teacher patch embedding (one
    [dp] vector per clip, random-projected + L2-normalized). One target per clip --
    the encoder's pooled 'global' feature is trained to predict it (cosine)."""
    from ..data.render.renderer import RenderConfig, make_cameras, render_frame
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=device); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
    Fr = pts.shape[0]
    with torch.no_grad():
        rendered = torch.stack([render_frame(pts[f].to(device), K, Rt, img, img, 1) for f in range(Fr)])  # [Fr,3,H,W]
        if is_vmae:
            imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
            istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
            fpc = int(teacher.patch_embed.num_patches // ((img // teacher.patch_embed.patch_size[0]) ** 2) * teacher.tubelet_size)
            ti = (torch.arange(fpc, device=device).float() / max(1, fpc - 1) * (Fr - 1)).round().long()
            clip = rendered[ti][None]
            feats = _videomaev2_tokens(teacher, (clip - imean) / istd)  # [1,L,D]
            mean = feats[0].mean(0)                              # mean over ALL spatio-temporal patches
        elif is_video:
            imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
            istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)
            fpc = int(getattr(teacher.config, "frames_per_clip", 64))
            ti = (torch.arange(fpc, device=device).float() / max(1, fpc - 1) * (Fr - 1)).round().long()
            clip = rendered[ti][None]
            feats = teacher(pixel_values_videos=(clip - imean) / istd).last_hidden_state  # [1,L,D]
            mean = feats[0].mean(0)                              # mean over ALL spatio-temporal patches
        else:
            imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            feats = teacher(pixel_values=(rendered - imean) / istd).last_hidden_state    # [Fr,1(+4)+P,D]
            drop = 5 if is_dino3 else 1                          # DINOv3: CLS + 4 register tokens
            mean = feats[:, drop:, :].reshape(-1, feats.shape[-1]).mean(0)  # mean over all patches, all frames
        g = mean @ proj                                         # [dp]
        g = F.normalize(g, dim=-1)
    return g.half().cpu()                                       # [dp]


def _compute_clip_target(pts, dino, proj, device, img, dp, *, global_mode, is_video,
                         is_vmae, is_dino3, num_cameras,
                         is_rawpix=False, patch=14, interp=False):
    """Dispatch to the right teacher routine for one clip's target. Single source of
    truth shared by the in-memory and disk-streaming precompute paths (identical to
    the original inline if/elif chain)."""
    if global_mode:
        return _clip_global_target(pts, dino, proj, device, img, dp, is_video, is_vmae, is_dino3)  # [dp]
    if is_vmae:
        return _clip_dense_targets_videomaev2(pts, dino, proj, device, img, dp)     # [F,N,dp]
    if is_dino3:
        return _clip_dense_targets_dinov3(pts, dino, proj, device, img, dp)         # [F,N,dp]
    if is_video:
        return _clip_dense_targets_vjepa2(pts, dino, proj, device, img, dp)         # [F,N,dp]
    if is_rawpix:
        return _clip_dense_targets_rawpix(pts, proj, device, img, dp)               # [F,N,dp]
    if num_cameras > 1:
        return _clip_dense_targets_multiview(pts, dino, proj, device, img, dp, num_cameras)  # [F,N,dp]
    return _clip_dense_targets(pts, dino, proj, device, img, dp, patch, 1, interp)   # [F,N,dp]


class _DiskClipDataset(torch.utils.data.Dataset):
    """Streaming dataset for the disk-cache path: each item reads ONE per-clip
    .npz (written by the precompute pass) and returns (points [F,N,3], target).
    Holds nothing in RAM beyond the clip currently being read."""

    def __init__(self, cache_dir, clip_ids):
        self.cache_dir = Path(cache_dir)
        self.clip_ids = clip_ids

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        z = np.load(self.cache_dir / f"{self.clip_ids[i]}.npz")
        pts = torch.from_numpy(z["points"].astype(np.float32))      # [F,N,3]
        tgt = torch.from_numpy(z["target"].astype(np.float32))      # [F,N,dp] or [dp]
        return pts, tgt


def _collate_clip(batch):
    # batch_size is fixed at 1; just unwrap the single (pts, tgt) so the training
    # body can index it exactly like the in-memory clips[i] tuple.
    return batch[0]


def _precompute_to_disk(recs, base, cache_dir, dino, proj, device, img, dp, nf, *,
                        global_mode, is_video, is_vmae, is_dino3, num_cameras,
                        is_rawpix=False, patch=14, interp=False):
    """Compute each clip's teacher target and write cache_dir/<clip_id>.npz (points +
    target, both fp16). Skips clips already cached (resumable). Frees per-clip tensors
    each iteration so memory stays flat -- nothing accumulates across clips."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(recs)
    done = 0
    for i, rec in enumerate(recs):
        fp = cache_dir / f"{rec['clip_id']}.npz"
        if fp.exists():
            done += 1
            if i % 200 == 0:
                log.info(f"  cache {i+1}/{n} (skip existing)")
            continue
        pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)        # [F,N,3]
        tgt = _compute_clip_target(pts, dino, proj, device, img, dp,
                                   global_mode=global_mode, is_video=is_video,
                                   is_vmae=is_vmae, is_dino3=is_dino3, num_cameras=num_cameras, is_rawpix=is_rawpix, patch=patch, interp=interp)
        # atomic write (crash-safe). Name ends in .npz so np.savez does NOT append a
        # second .npz; os.replace(tmp, fp) then publishes it under the final name.
        tmp = cache_dir / f".{rec['clip_id']}.tmp.npz"
        np.savez(tmp, points=pts.numpy().astype(np.float32), target=tgt.cpu().numpy())
        os.replace(tmp, fp)
        del pts, tgt
        done += 1
        if i % 200 == 0:
            log.info(f"  cache {i+1}/{n}")
            torch.cuda.empty_cache()
    log.info(f"precompute-to-disk done: {done}/{n} clips cached in {cache_dir}")


# ---------------------------------------------------------------------------
# TOKEN-GRANULARITY cache (the loss is per-token: each student token's target is
# the teacher feature of its nearest INPUT point). The per-point cache stores all
# N=2048 point targets/frame but the loss only ever reads the ~n2=64 tokens/frame.
# This mode precomputes, per clip, EXACTLY those per-token targets + the token
# positions, so the cache is ~Np/n2 (=2048/64=32x) smaller. At train time we match
# the LIVE student tokens to the cached tokens by NEAREST token_xyz (robust to any
# FPS-order drift between cache-build and train), then apply the same cosine loss.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encoder_token_xyz(enc, pts, device):
    """Run the student encoder forward on one clip's points to get its per-frame
    token positions. token_xyz/token_t are produced by the set-abstraction FPS +
    per-clip normalization (see Point4DEncoder.forward); both are WEIGHT-INDEPENDENT
    (pure coordinate ops), so an untrained encoder yields the same positions the
    trained one will. Returns token_xyz [n_tok,3] (orig coords) and token_t [n_tok]."""
    out = enc(pts[None].to(device))
    return out["token_xyz"][0].cpu(), out["token_t"][0].cpu()


def _clip_token_target(pts, tgt, token_xyz, token_t, nf):
    """Gather the per-token teacher target the loss needs from a full per-point
    target. For each token, target = its nearest INPUT point's teacher feature --
    EXACTLY what the per-point training loop reads (knn(pts[f], txyz[sel],1) -> tgt[f]).
    pts [F,N,3], tgt [F,N,dp], token_xyz [n_tok,3], token_t [n_tok] (input-frame idx).
    Returns tok_tgt [F,n2,dp] in frame-major (t,j) order matching token_t, plus the
    matching token_xyz reshaped to [F,n2,3]. Zero (no-target) point features are kept
    verbatim so the train-time `tg.abs().sum(-1)>0` mask drops them identically."""
    F_, N, dp = tgt.shape
    frames = sorted(int(f) for f in torch.unique(token_t).tolist())
    n2 = (token_t == frames[0]).sum().item()
    tok_tgt = torch.zeros(F_, n2, dp, dtype=torch.float16)
    tok_xyz = torch.zeros(F_, n2, 3, dtype=torch.float32)
    for f in frames:
        sel = token_t == f
        txyz = token_xyz[sel]                                         # [n2,3]
        nnf = knn(pts[f][None], txyz[None], 1)[0, :, 0]               # token -> nearest input point
        tok_tgt[f] = tgt[f][nnf].half()
        tok_xyz[f] = txyz
    return tok_tgt, tok_xyz


class _DiskTokenDataset(torch.utils.data.Dataset):
    """Streaming dataset for the TOKEN-granularity cache: each item reads ONE per-clip
    .npz holding points [F,N,3] + per-token teacher targets token_target [F,n2,dp] +
    their positions token_xyz [F,n2,3]. Returns (points, token_target, token_xyz)."""

    def __init__(self, cache_dir, clip_ids):
        self.cache_dir = Path(cache_dir)
        self.clip_ids = clip_ids

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, i):
        z = np.load(self.cache_dir / f"{self.clip_ids[i]}.npz")
        pts = torch.from_numpy(z["points"].astype(np.float32))           # [F,N,3]
        tok_tgt = torch.from_numpy(z["token_target"].astype(np.float32)) # [F,n2,dp]
        tok_xyz = torch.from_numpy(z["token_xyz"].astype(np.float32))    # [F,n2,3]
        return pts, tok_tgt, tok_xyz


def _collate_token_clip(batch):
    return batch[0]                                                      # batch_size 1: unwrap the triple


def _precompute_tokens_to_disk(recs, base, cache_dir, dino, proj, enc, device, img, dp, nf, *,
                               is_video, is_vmae, is_dino3, num_cameras,
                               is_rawpix=False, patch=14, interp=False):
    """TOKEN-granularity precompute: per clip compute the FULL per-point teacher target
    (same _compute_clip_target as the per-point path), run the encoder to get the token
    positions, gather the per-token targets the loss reads, and write
    cache_dir/<clip_id>.npz = points[F,N,3] fp16 + token_target[F,n2,dp] fp16 +
    token_xyz[F,n2,3] fp32 (~1.5MB/clip vs ~38MB). Resumable; flat memory."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(recs)
    done = 0
    for i, rec in enumerate(recs):
        fp = cache_dir / f"{rec['clip_id']}.npz"
        if fp.exists():
            done += 1
            if i % 200 == 0:
                log.info(f"  token-cache {i+1}/{n} (skip existing)")
            continue
        pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)        # [F,N,3]
        tgt = _compute_clip_target(pts, dino, proj, device, img, dp,
                                   global_mode=False, is_video=is_video,
                                   is_vmae=is_vmae, is_dino3=is_dino3, num_cameras=num_cameras, is_rawpix=is_rawpix, patch=patch, interp=interp)
        token_xyz, token_t = _encoder_token_xyz(enc, pts, device)
        tok_tgt, tok_xyz = _clip_token_target(pts, tgt, token_xyz, token_t, nf)
        tmp = cache_dir / f".{rec['clip_id']}.tmp.npz"
        np.savez(tmp, points=pts.numpy().astype(np.float16),
                 token_target=tok_tgt.numpy(), token_xyz=tok_xyz.numpy())
        os.replace(tmp, fp)
        del pts, tgt, tok_tgt, tok_xyz
        done += 1
        if i % 200 == 0:
            log.info(f"  token-cache {i+1}/{n}")
            torch.cuda.empty_cache()
    log.info(f"precompute-tokens-to-disk done: {done}/{n} clips cached in {cache_dir}")


def _fit_pca_proj(recs, base, dino, dino_d, device, img, dp, nf, n_clips=32):
    """Top-dp PCA projection on DINOv2 patch features from a clip sample, a drop-in
    for the random projection (random-vs-PCA ablation)."""
    from ..data.render.renderer import RenderConfig, make_cameras, render_frame
    imean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    istd = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    feats_all = []
    with torch.no_grad():
        for rec in recs[:n_clips]:
            pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)
            c = pts.reshape(-1, 3).mean(0); sc = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
            cams, _ = make_cameras(RenderConfig(image_size=img, num_cameras=1), centroid=c.tolist(), scale=sc)
            K, Rt = cams[0]
            K = torch.as_tensor(K, dtype=torch.float32, device=device); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=device)
            imgs = torch.stack([render_frame(pts[fi].to(device), K, Rt, img, img, 1) for fi in range(pts.shape[0])])
            fe = dino(pixel_values=(imgs - imean) / istd).last_hidden_state[:, 1:, :].reshape(-1, dino_d)
            feats_all.append(fe.float().cpu())
    X = torch.cat(feats_all, 0); X = X - X.mean(0, keepdim=True)
    evals, evecs = torch.linalg.eigh(X.T @ X)                 # ascending
    proj = evecs[:, -dp:].flip(1).contiguous().to(device)     # top-dp PCs [D, dp]
    log.info(f"PCA proj on {X.shape[0]} tokens / {min(n_clips, len(recs))} clips -> {tuple(proj.shape)}")
    return proj


def pretrain_dense(cfg: Config, img: int = 126, dp: int = 384, max_clips: Optional[int] = None,
                   teacher: str = "dinov2", global_mode: bool = False, num_cameras: int = 1,
                   target_cache_dir: Optional[str] = None, resume: bool = False,
                   token_cache: bool = False, proj_mode: str = "random") -> dict:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    import random as _random; _random.seed(cfg.seed)
    nf = cfg.data.num_frames
    is_vmae = (teacher == "videomaev2")
    is_video = (teacher == "vjepa2") or is_vmae
    is_dino3 = (teacher == "dinov3")
    is_rawpix = (teacher == "rawpix")
    is_vit = teacher in {"vit-in1k", "vit-rand"}
    if teacher == "vjepa2":
        img = 256                                  # V-JEPA2 crop (fpc64-256); patch16 -> 16x16 grid
    elif is_vmae:
        img = 224                                  # VideoMAEv2-Base crop; patch16 -> 14x14 grid
    elif is_dino3 or is_vit:
        img = 144                                  # patch16 -> 9x9 grid (matches DINOv2 126/14=9)
    patch, interp = (16, True) if is_vit else (14, False)
    if is_rawpix and global_mode:
        raise ValueError("--teacher rawpix is a DENSE-only control (it has no pooled embedding)")
    dino, dino_d = _teacher(device, teacher)
    proj = torch.randn(dino_d, dp, device=device) / dp ** 0.5

    train_ds, _ = build_labeled_splits(cfg)        # _PointsLabels wrapping the clip provider
    base = train_ds.base
    recs = base.records[:max_clips] if max_clips else base.records
    if proj_mode == "pca":
        proj = _fit_pca_proj(recs, base, dino, dino_d, device, img, dp, nf)
    mode = "GLOBAL" if global_mode else "DENSE"
    streaming = target_cache_dir is not None
    if token_cache and (not streaming or global_mode):
        raise ValueError("--token-cache requires --target-cache-dir and is incompatible with --global")
    kind = "token-stream" if token_cache else ("disk-stream" if streaming else "in-memory")
    log.info(f"{mode} pretrain ({teacher}): {kind} "
             f"targets for {len(recs)} clips (img={img}, dp={dp}, num_cameras={num_cameras})...")

    enc = None
    if token_cache:
        # TOKEN-GRANULARITY: cache only the ~n2 per-token targets the loss reads +
        # their positions (~32x smaller than the per-point cache). Train from a
        # streaming DataLoader that matches live tokens to cached tokens by nearest xyz.
        # The encoder is built HERE (before precompute) because it supplies the
        # weight-independent token_xyz the cache is keyed on; reused for training below.
        enc = build_encoder(cfg.encoder).to(device).eval()
        _precompute_tokens_to_disk(recs, base, target_cache_dir, dino, proj, enc, device,
                                   img, dp, nf, is_video=is_video, is_vmae=is_vmae,
                                   is_dino3=is_dino3, num_cameras=num_cameras, is_rawpix=is_rawpix, patch=patch, interp=interp)
        del dino; torch.cuda.empty_cache()
        clip_ids = [r["clip_id"] for r in recs]
        stream_ds = _DiskTokenDataset(target_cache_dir, clip_ids)
        n_clips = len(stream_ds)
    elif streaming:
        # DISK-STREAMING: precompute each clip's target to cache_dir/<clip_id>.npz
        # (resumable, flat memory), then train from a streaming DataLoader.
        _precompute_to_disk(recs, base, target_cache_dir, dino, proj, device, img, dp, nf,
                            global_mode=global_mode, is_video=is_video, is_vmae=is_vmae,
                            is_dino3=is_dino3, num_cameras=num_cameras, is_rawpix=is_rawpix, patch=patch, interp=interp)
        del dino; torch.cuda.empty_cache()
        clip_ids = [r["clip_id"] for r in recs]
        stream_ds = _DiskClipDataset(target_cache_dir, clip_ids)
        n_clips = len(stream_ds)
    else:
        clips = []
        for i, rec in enumerate(recs):
            pts = torch.as_tensor(base.load_full(rec)[:nf], dtype=torch.float32)    # [F,N,3]
            tgt = _compute_clip_target(pts, dino, proj, device, img, dp,
                                       global_mode=global_mode, is_video=is_video,
                                       is_vmae=is_vmae, is_dino3=is_dino3, num_cameras=num_cameras, is_rawpix=is_rawpix, patch=patch, interp=interp)
            clips.append((pts, tgt))
            if i % 50 == 0:
                log.info(f"  targets {i+1}/{len(recs)}")
        del dino; torch.cuda.empty_cache()
        n_clips = len(clips)

    if enc is None:                                    # non-token paths build the encoder here (original order)
        enc = build_encoder(cfg.encoder).to(device)
    head = nn.Linear(cfg.encoder.embed_dim, dp).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()),
                            lr=cfg.optim.base_lr, weight_decay=1e-4)
    steps = cfg.optim.epochs * n_clips
    warm = max(1, int(0.05 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: (t + 1) / warm if t < warm else 0.5 * (1 + math.cos(math.pi * (t - warm) / max(1, steps - warm))))

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    last = out_dir / "last.pt"
    start_ep = 0
    if resume and last.exists():
        st = torch.load(last, map_location=device)
        enc.load_state_dict(st["encoder"]); head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        start_ep = st["epoch"] + 1
        log.info(f"resumed from {last} at epoch {start_ep}")

    # Streaming: a DataLoader yields one clip per step with shuffle + workers
    # prefetching from disk. In-memory: iterate a permutation of the held clips.
    if streaming:
        loader = torch.utils.data.DataLoader(
            stream_ds, batch_size=1, shuffle=True, num_workers=cfg.optim.num_workers,
            collate_fn=_collate_token_clip if token_cache else _collate_clip,
            pin_memory=True, persistent_workers=cfg.optim.num_workers > 0)

    enc.train()
    t = start_ep * n_clips
    for ep in range(start_ep, cfg.optim.epochs):
        clip_iter = loader if streaming else (clips[i] for i in np.random.permutation(n_clips))
        for item in clip_iter:
            if token_cache:
                pts, tok_tgt, tok_xyz = item
                tok_tgt = tok_tgt.to(device).float(); tok_xyz = tok_xyz.to(device)  # [F,n2,dp]; [F,n2,3]
            else:
                pts, tgt = item
                tgt = tgt.to(device).float()                                       # [F,N,dp] or [dp]
            pts = pts[None].to(device)                                             # [1,F,N,3]
            out = enc(pts)
            if token_cache:
                # TOKEN-GRANULARITY loss: per frame, match each LIVE student token to its
                # nearest CACHED token (by token_xyz) and read that token's precomputed
                # target. Equivalent to the per-point path's token->nearest-point gather,
                # but the gather was done once at cache time. token_t groups live tokens.
                tok = out["tokens"][0]; txyz = out["token_xyz"][0]; ttok = out["token_t"][0]
                pred = head(tok)
                losses = []
                for f in torch.unique(ttok).tolist():
                    sel = ttok == f
                    m = knn(tok_xyz[f][None], txyz[sel][None], 1)[0, :, 0]          # live token -> cached token
                    tg = tok_tgt[f][m]
                    ok = tg.abs().sum(-1) > 0                                       # skip tokens whose cached target was empty
                    if ok.any():
                        losses.append((1 - F.cosine_similarity(pred[sel][ok], tg[ok], dim=-1)).mean())
                if not losses:
                    sched.step(); t += 1; continue
                dist = torch.stack(losses).mean()
                g = out["global"]
                vic = F.relu(1.0 - g.std(0)).mean() if g.shape[0] > 1 else g.pow(2).mean() * 0
                loss = dist + cfg.loss.vicreg * vic
                opt.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward(); opt.step()
                sched.step(); t += 1
                continue
            if global_mode:
                # GLOBAL distillation: pooled encoder feature -> clip's mean teacher embedding (cosine)
                g = out["global"]                                                  # [1,embed_dim]
                pred_g = head(g)                                                   # [1,dp]
                dist = (1 - F.cosine_similarity(pred_g, tgt[None], dim=-1)).mean()
                vic = g.pow(2).mean() * 0                                          # B=1: no batch std term
                loss = dist + cfg.loss.vicreg * vic
                opt.zero_grad(set_to_none=True)
                if torch.isfinite(loss):
                    loss.backward(); opt.step()
                sched.step(); t += 1
                continue
            tok = out["tokens"][0]; txyz = out["token_xyz"][0]
            pred = head(tok)
            # group tokens by their input-frame index; encoders that pool temporally
            # (e.g. P4Transformer) emit fewer frame-groups than nf. Fall back to
            # frame-major slicing for encoders that don't expose token_t.
            if "token_t" in out:
                ttok = out["token_t"][0]
                groups = [(f, ttok == f) for f in torch.unique(ttok).tolist()]
            else:
                n2 = tok.shape[0] // nf
                groups = [(f, slice(f * n2, (f + 1) * n2)) for f in range(nf)]
            losses = []
            for f, sel in groups:
                nnf = knn(pts[0, f][None], txyz[sel][None], 1)[0, :, 0]
                tg = tgt[f][nnf]
                ok = tg.abs().sum(-1) > 0                                          # skip points with no target
                if ok.any():
                    losses.append((1 - F.cosine_similarity(pred[sel][ok], tg[ok], dim=-1)).mean())
            if not losses:                                                    # clip with no visible targets (rare degenerate DT4D mesh)
                sched.step(); t += 1; continue
            dist = torch.stack(losses).mean()
            g = out["global"]
            vic = F.relu(1.0 - g.std(0)).mean() if g.shape[0] > 1 else g.pow(2).mean() * 0  # std anti-collapse
            loss = dist + cfg.loss.vicreg * vic
            opt.zero_grad(set_to_none=True)
            if torch.isfinite(loss):
                loss.backward(); opt.step()
            sched.step(); t += 1
        if ep % 10 == 0 or ep == cfg.optim.epochs - 1:
            log.info(f"  ep {ep:03d} dist {dist.item():.4f} feat_std {g.std().item():.4f}")
        if resume:                                     # checkpoint for resume (only when requested)
            torch.save({"encoder": enc.state_dict(), "head": head.state_dict(),
                        "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}, last)

    ckpt = out_dir / "encoder.pt"
    torch.save({"encoder": enc.state_dict(), "cfg": cfg.to_dict(), "embed_dim": enc.embed_dim,
                "run_args": {"teacher": teacher, "global_mode": global_mode, "proj_mode": proj_mode,
                             "num_cameras": num_cameras, "dp": dp, "img": img}}, ckpt)
    log.info(f"dense pretrain done -> {ckpt}")
    return {"encoder_ckpt": str(ckpt)}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Dense cross-modal distillation pretraining")
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--img", type=int, default=126)
    ap.add_argument("--dp", type=int, default=384)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--teacher", choices=["dinov2", "dinov3", "vjepa2", "videomaev2",
                                         "vit-in1k", "vit-rand", "rawpix"], default=None,
                    help="dense teacher (defaults to the config's teacher.name): dinov2 (per-frame image patches), "
                         "dinov3 (per-frame image patches, patch16 @144px -> 9x9, drop CLS+4 reg), "
                         "vjepa2 or videomaev2 (per-frame spatial slices of the video tubelet tokens), "
                         "vit-in1k / vit-rand / rawpix (NON-FOUNDATION controls at the same 9x9 grid: "
                         "ImageNet-supervised ViT-B/16, the same ViT untrained, and raw rendered pixels)")
    ap.add_argument("--global", dest="global_mode", action="store_true",
                    help="GLOBAL distillation: pooled encoder feature predicts the clip's "
                         "MEAN teacher embedding (one target/clip) instead of per-point targets")
    ap.add_argument("--num-cameras", type=int, default=1,
                    help="#teacher cameras (dinov2 dense only). 1 = headline single view. "
                         ">1 averages the random-projected teacher feature over the visible "
                         "views per (frame,point); points seen in 0 views stay masked.")
    ap.add_argument("--target-cache-dir", default=None,
                    help="DISK-STREAMING mode: precompute each clip's teacher target to "
                         "DIR/<clip_id>.npz (resumable, flat memory) then train from a "
                         "streaming DataLoader. Omit for the default in-memory path.")
    ap.add_argument("--resume", action="store_true",
                    help="resume encoder/opt/sched from <out_dir>/last.pt and checkpoint "
                         "last.pt every epoch (for long full-dataset runs).")
    ap.add_argument("--token-cache", action="store_true",
                    help="TOKEN-GRANULARITY cache (requires --target-cache-dir): cache only "
                         "the ~n2 per-token teacher targets the per-token loss reads + their "
                         "positions (~32x smaller than the per-point cache). Train matches live "
                         "tokens to cached tokens by nearest token_xyz. Equivalent training "
                         "signal to the per-point path.")
    ap.add_argument("--proj-mode", choices=["random", "pca"], default="random",
                    help="teacher-feature projection: random (default) or pca.")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    if args.out:
        cfg.out_dir = args.out
    if args.epochs:
        cfg.optim.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    teacher = args.teacher or getattr(cfg.teacher, "name", "dinov2")
    pretrain_dense(cfg, img=args.img, dp=args.dp, max_clips=args.max_clips,
                   teacher=teacher, global_mode=args.global_mode, num_cameras=args.num_cameras,
                   target_cache_dir=args.target_cache_dir, resume=args.resume,
                   token_cache=args.token_cache, proj_mode=args.proj_mode)


if __name__ == "__main__":
    main()
