#!/usr/bin/env python
"""SMOKE TEST for DENSE per-point cross-modal distillation, end-to-end on a few MSR
clips (a quick check that the install, data, and teacher download all work).

Per clip: render each frame -> per-pixel nearest-point index -> DINOv2 patch features
-> PER-FRAME per-point teacher targets [F,N,Dp] (random-projected to keep memory small;
per-frame because MSR resamples points each frame). Student Point4D -> tokens +
token_xyz; each token is matched to its nearest input point in its frame and a head
predicts that point's teacher feature; per-token cosine loss.

Checks: (a) dense loss DROPS (learns, no collapse: feat std stays healthy);
(b) linear-probe of mean-pooled features BEATS a random-init encoder.
Small/noisy by design -- a directional go/no-go for the full build.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel  # noqa: E402

from cross4d_jepa.config import Config  # noqa: E402
from cross4d_jepa.data.datasets.msr_action3d import MSRAction3DClips  # noqa: E402
from cross4d_jepa.data.render.renderer import (  # noqa: E402
    RenderConfig, make_cameras, render_frame, render_frame_index, dense_point_targets)
from cross4d_jepa.models import build_encoder  # noqa: E402
from cross4d_jepa.models.encoders.point4d import knn  # noqa: E402

DEV = "cuda"
K_TR = K_TE = 32           # clips
NF = 6                     # frames per clip
IMG = 126                  # 9x9 DINOv2 patches (patch=14)
DP = 384                   # random-projected teacher dim (memory)
STEPS = 300
torch.manual_seed(0)

cfg = Config.from_yaml("configs/msr_point4d.yaml")
cfg.device = DEV
dino = AutoModel.from_pretrained("facebook/dinov2-large").to(DEV).eval()
for p in dino.parameters():
    p.requires_grad_(False)
DINO_D = dino.config.hidden_size
proj = torch.randn(DINO_D, DP, device=DEV) / DP ** 0.5     # fixed random projection
imean = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
istd = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


@torch.no_grad()
def dino_feats(imgs):                                       # [B,3,H,W] in [0,1] -> [B,Hp,Wp,DP]
    out = dino(pixel_values=(imgs - imean) / istd).last_hidden_state  # [B,1+P,D]
    B = imgs.shape[0]; Hp = Wp = IMG // 14
    return (out[:, 1:, :].reshape(B, Hp, Wp, DINO_D) @ proj)


@torch.no_grad()
def clip_targets(points):                                  # [F,N,3] -> targets [F,N,DP], valid [F,N]
    pts = torch.as_tensor(points, dtype=torch.float32)
    c = pts.reshape(-1, 3).mean(0); s = (pts.reshape(-1, 3) - c).abs().max().item() + 1e-6
    cams, _ = make_cameras(RenderConfig(image_size=IMG, num_cameras=1), centroid=c.tolist(), scale=s)
    K, Rt = cams[0]
    K = torch.as_tensor(K, dtype=torch.float32, device=DEV); Rt = torch.as_tensor(Rt, dtype=torch.float32, device=DEV)
    Fr, N, _ = pts.shape
    imgs = torch.stack([render_frame(pts[f].to(DEV), K, Rt, IMG, IMG, 1) for f in range(Fr)])  # [F,3,H,W]
    feats = dino_feats(imgs)                                # [F,Hp,Wp,DP]
    tgt = torch.zeros(Fr, N, DP, device=DEV); valid = torch.zeros(Fr, N, dtype=torch.bool, device=DEV)
    for f in range(Fr):
        idx = render_frame_index(pts[f].to(DEV), K, Rt, IMG, IMG, 1)
        ids, fe = dense_point_targets(idx, feats[f])
        tgt[f, ids] = fe; valid[f, ids] = True
    return tgt, valid


def load(split, k):
    ds = MSRAction3DClips(cfg, split)
    out = []
    for rec in ds.records[:k]:
        full = ds.load_full(rec)[:NF]                      # [F,N,3]
        out.append((torch.as_tensor(full, dtype=torch.float32), int(rec["label"])))
    return out


print("loading clips + precomputing dense teacher targets...", flush=True)
tr = load("train", K_TR); te = load("test", K_TE)
tr_tgt = [clip_targets(p) for p, _ in tr]                  # list of (tgt[F,N,DP], valid[F,N])
ncls = max(max(y for _, y in tr), max(y for _, y in te)) + 1


def lp(encoder):
    encoder.eval()
    with torch.no_grad():
        Xtr = torch.stack([encoder(p[None].to(DEV))["global"][0] for p, _ in tr])
        Xte = torch.stack([encoder(p[None].to(DEV))["global"][0] for p, _ in te])
    ytr = torch.tensor([y for _, y in tr], device=DEV); yte = torch.tensor([y for _, y in te], device=DEV)
    head = nn.Linear(Xtr.shape[1], ncls).to(DEV)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(300):
        opt.zero_grad(); F.cross_entropy(head(Xtr), ytr).backward(); opt.step()
    return (head(Xte).argmax(1) == yte).float().mean().item()


enc = build_encoder(cfg.encoder).to(DEV)
rand_lp = lp(enc)                                          # random-init baseline (same arch, untrained)
print(f"random-init LP: {rand_lp:.3f} (chance {1/ncls:.3f})", flush=True)

head = nn.Linear(cfg.encoder.embed_dim, DP).to(DEV)
opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-3, weight_decay=1e-4)
enc.train()
loss0 = None
for step in range(STEPS):
    i = np.random.randint(len(tr))
    pts = tr[i][0][None].to(DEV)                           # [1,F,N,3]
    tgt, valid = tr_tgt[i]                                 # [F,N,DP],[F,N]
    out = enc(pts)
    tok = out["tokens"][0]; txyz = out["token_xyz"][0]     # [T*n2,C],[T*n2,3]
    n2 = tok.shape[0] // NF
    pred = head(tok)                                       # [T*n2,DP]
    # match each token (frame t, center) to nearest input point in frame t
    losses = []
    for t in range(NF):
        sl = slice(t * n2, (t + 1) * n2)
        nn_idx = knn(pts[0, t][None], txyz[sl][None], 1)[0, :, 0]   # [n2] nearest input-point idx
        v = valid[t][nn_idx]
        if v.any():
            losses.append((1 - F.cosine_similarity(pred[sl][v], tgt[t][nn_idx][v], dim=-1)).mean())
    loss = torch.stack(losses).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 0:
        loss0 = loss.item()
    if step % 50 == 0 or step == STEPS - 1:
        with torch.no_grad():
            fstd = enc(tr[0][0][None].to(DEV))["global"].std().item()
        print(f"  step {step:03d} dense_loss {loss.item():.4f} feat_std {fstd:.4f}", flush=True)

dense_lp = lp(enc)
print(f"\n=== SMOKE RESULT ===", flush=True)
print(f"dense_loss: {loss0:.4f} -> {loss.item():.4f}  (drops = learning)", flush=True)
print(f"LP  random-init {rand_lp:.3f}  vs  dense-distilled {dense_lp:.3f}  "
      f"(gain {dense_lp - rand_lp:+.3f})", flush=True)
print("VERDICT:", "PROMISING (dense beats random, loss drops)" if dense_lp > rand_lp + 0.03 and loss.item() < loss0
      else "WEAK/NULL (no clear gain) -- reconsider before full build", flush=True)
