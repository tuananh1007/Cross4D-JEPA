"""Per-point semantic-segmentation linear probe of a FROZEN 4D encoder (HOI4D semseg).

Protocol (matches the action LP rule: frozen encoder + single linear layer, NO
feature standardization, ignore class 0):

  * For each clip [T,N,3], run the frozen encoder -> spatio-temporal tokens with
    centers token_xyz [T*n2,3] and per-token frame index token_t. For each frame f,
    interpolate the n2 token features at that frame to all N=2048 points via 3-NN
    inverse-distance weighting from the token centers (PointNet++ FeaturePropagation
    readout) -> per-point per-frame features [T,N,C].
  * Linear head Linear(C -> num_classes), no BN/ReLU. Per-point cross-entropy
    ignoring class 0. Train on split==0, evaluate on split==1. To scale, the head
    TRAIN set keeps K=256 random points PER FRAME (~Nclips*T*K points) and the head
    is trained with minibatch Adam + cosine (never holds all ~87M features at once).
  * Metrics over ALL (point,frame) test predictions (TEST uses every point, no
    subsample): mIoU (mean over classes PRESENT in test of TP/(TP+FP+FN)), mAcc (mean
    per-class recall), OA (overall point acc), computed from a [K,K] confusion matrix
    accumulated incrementally over all test points. Reported mean +/- std over
    `--seeds` head-training seeds (encoder features are extracted ONCE; only the head
    init/optimization is re-seeded).

num_classes covers labels 0..48 (head has 49 logits); index 0 is the ignore class
(absent from this subset but masked anyway). mIoU/mAcc are averaged only over the
semantic classes that actually appear in the test split.
"""
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cross4d_jepa.config import Config
from cross4d_jepa.models import build_encoder
from cross4d_jepa.data.datasets.hoi4d_semseg import HOI4DSemSegClips

NUM_CLASSES = 49          # labels 0..48; 0 = ignore/background
IGNORE = 0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def threenn_interp(tok_xyz, tok_feat, query_xyz):
    """3-NN inverse-distance interpolation (PointNet++ FeaturePropagation).
    tok_xyz [M,3], tok_feat [M,C], query_xyz [N,3] -> [N,C]."""
    d = torch.cdist(query_xyz, tok_xyz)                      # [N,M]
    k = min(3, tok_xyz.shape[0])
    dist, idx = d.topk(k, dim=-1, largest=False)             # [N,k]
    w = 1.0 / (dist + 1e-8)
    w = w / w.sum(dim=-1, keepdim=True)                      # [N,k]
    nf = tok_feat[idx]                                       # [N,k,C]
    return (nf * w.unsqueeze(-1)).sum(dim=1)                 # [N,C]


@torch.no_grad()
def per_point_feats(encoder, ds, device, T, global_broadcast=False, k_sub=None):
    """Yield (feat [P,C] fp16 cpu, lab [P] int64) per clip via 3-NN interp.
    global_broadcast=True instead broadcasts the encoder's pooled clip 'global'
    feature to every point (the P4T-global pooling-backbone row: one feature per
    clip -> expected ~chance mIoU).
    k_sub: if set, keep K random points PER FRAME (so the assembled train set is
    ~Nclips*T*K points instead of every point) -> never holds all 87M features."""
    for rec in ds.records:
        pts = torch.from_numpy(ds.load_full(rec)[:T]).float().unsqueeze(0).to(device)  # [1,T,N,3]
        out = encoder(pts)
        N = pts.shape[2]
        lab = ds.semantic(rec)[:T].astype(np.int64)         # [T,N]
        if global_broadcast:
            g = out["global"][0]                            # [C] pooled clip token
            feats = g.view(1, 1, -1).expand(T, N, -1)
        else:
            tok = out["tokens"][0]                          # [T*n2, C]
            txyz = out["token_xyz"][0]                      # [T*n2, 3]
            ttok = out["token_t"][0]                        # [T*n2]
            feats = torch.empty(T, N, tok.shape[-1], device=device)
            for f in range(T):
                sel = ttok == f
                feats[f] = threenn_interp(txyz[sel], tok[sel], pts[0, f])
        if k_sub is not None and k_sub < N:
            # K random points per frame (independent per frame); subsample BEFORE
            # moving to CPU so we only ever materialize ~T*K features per clip.
            sub = torch.stack([torch.randperm(N, device=device)[:k_sub] for _ in range(T)])  # [T,K]
            feats = torch.gather(feats, 1, sub.unsqueeze(-1).expand(-1, -1, feats.shape[-1]))  # [T,K,C]
            lab = np.take_along_axis(lab, sub.cpu().numpy(), axis=1)                            # [T,K]
        C = feats.shape[-1]
        yield feats.reshape(-1, C).half().cpu(), torch.from_numpy(lab.reshape(-1))


def extract_split(encoder, ds, device, T, global_broadcast=False, k_sub=None):
    feats, labs = [], []
    t0 = time.time()
    for i, (fe, la) in enumerate(per_point_feats(encoder, ds, device, T, global_broadcast, k_sub)):
        feats.append(fe)
        labs.append(la)
        if i % 200 == 0:
            log(f"    extracted {i+1}/{len(ds.records)} clips ({time.time()-t0:.0f}s)")
    return torch.cat(feats), torch.cat(labs)                # [P,C] fp16, [P] int64


def metrics_from_conf(conf, present):
    """conf [K,K] (row=true,col=pred), present = bool mask of classes present in test.
    Returns (mIoU, mAcc, OA) over present classes (ignore index already excluded)."""
    conf = conf.double()
    tp = torch.diag(conf)
    fp = conf.sum(0) - tp
    fn = conf.sum(1) - tp
    denom = tp + fp + fn
    iou = tp / denom.clamp(min=1)
    recall = tp / (tp + fn).clamp(min=1)
    miou = iou[present].mean().item()
    macc = recall[present].mean().item()
    oa = (tp.sum() / conf.sum().clamp(min=1)).item()
    return miou, macc, oa


def train_eval_head(Xtr, ytr, Xte, yte, device, num_classes, seed, epochs, lr, bs):
    """Linear head, minibatch Adam + cosine, CE ignoring class 0. Returns confusion
    matrix [K,K] over test points (ignore-0 points excluded)."""
    torch.manual_seed(seed)
    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)

    # train only on non-ignore points (CE ignore_index already skips them, but we
    # also drop them from sampling so minibatches aren't dominated by ignore=0).
    keep = (ytr != IGNORE)
    idx_all = torch.nonzero(keep, as_tuple=False).squeeze(1)
    P = idx_all.numel()
    steps = max(1, (P + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps)

    for ep in range(epochs):
        perm = idx_all[torch.randperm(P)]
        for s in range(steps):
            b = perm[s * bs:(s + 1) * bs]
            xb = Xtr[b].to(device).float()
            yb = ytr[b].to(device)
            opt.zero_grad()
            lossf(head(xb), yb).backward()
            opt.step()
            sched.step()

    # eval -> confusion matrix over non-ignore test points
    head.eval()
    K = num_classes
    conf = torch.zeros(K, K, dtype=torch.long, device=device)
    with torch.no_grad():
        ev = 200_000
        for s in range(0, Xte.shape[0], ev):
            xb = Xte[s:s + ev].to(device).float()
            yb = yte[s:s + ev].to(device)
            pred = head(xb).argmax(1)
            m = yb != IGNORE
            yb, pred = yb[m], pred[m]
            lin = yb * K + pred
            conf += torch.bincount(lin, minlength=K * K).reshape(K, K)
    return conf.cpu()


def main():
    ap = argparse.ArgumentParser(description="HOI4D per-point semseg linear probe of a frozen encoder")
    ap.add_argument("config")
    ap.add_argument("--encoder", default=None, help="encoder.pt (omit with --random-init)")
    ap.add_argument("--random-init", action="store_true", help="freshly built untrained encoder")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=100_000, help="points per head-train minibatch")
    ap.add_argument("--k-train", type=int, default=256,
                    help="random points kept PER FRAME when assembling the head TRAIN set "
                         "(~Nclips*T*K points). TEST always uses ALL points for the metric.")
    ap.add_argument("--tag", default="semseg")
    ap.add_argument("--max-clips", type=int, default=None, help="cap clips per split (smoke only)")
    ap.add_argument("--global-broadcast", action="store_true",
                    help="broadcast the pooled clip 'global' token to all points "
                         "(P4T-global pooling-backbone row; expected ~chance mIoU)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = cfg.data.num_frames

    encoder = build_encoder(cfg.encoder)
    if not args.random_init:
        assert args.encoder, "provide --encoder or --random-init"
        state = torch.load(args.encoder, map_location="cpu")
        encoder.load_state_dict(state["encoder"])
        log(f"loaded encoder {args.encoder}")
    else:
        log("random-init encoder (no checkpoint)")
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    log(f"extracting TRAIN features (split==0, K={args.k_train} pts/frame)...")
    tr = HOI4DSemSegClips(cfg, "train")
    te = HOI4DSemSegClips(cfg, "test")
    if args.max_clips:
        tr.records = tr.records[:args.max_clips]
        te.records = te.records[:args.max_clips]
    Xtr, ytr = extract_split(encoder, tr, device, T, args.global_broadcast, k_sub=args.k_train)
    log(f"  train points: {Xtr.shape[0]:,}  C={Xtr.shape[1]}  fp16 {Xtr.numel()*2/1e9:.1f}GB")

    log("extracting TEST features (split==1, ALL points)...")
    Xte, yte = extract_split(encoder, te, device, T, args.global_broadcast)  # no subsample: metric over ALL test points
    log(f"  test points: {Xte.shape[0]:,}  fp16 {Xte.numel()*2/1e9:.1f}GB")

    present = torch.zeros(NUM_CLASSES, dtype=torch.bool)
    pres_cls = torch.unique(yte[yte != IGNORE])
    present[pres_cls] = True
    log(f"  classes present in test (excl ignore): {int(present.sum())} -> {pres_cls.tolist()}")

    results = []
    for seed in args.seeds:
        t0 = time.time()
        conf = train_eval_head(Xtr, ytr, Xte, yte, device, NUM_CLASSES,
                               seed, args.epochs, args.lr, args.bs)
        miou, macc, oa = metrics_from_conf(conf, present)
        results.append((miou, macc, oa))
        log(f"RESULT tag={args.tag} seed={seed} mIoU={miou*100:.4f} "
            f"mAcc={macc*100:.4f} OA={oa*100:.4f}  ({time.time()-t0:.0f}s)")

    arr = np.array(results) * 100.0
    mean, std = arr.mean(0), arr.std(0)
    log(f"SUMMARY tag={args.tag} seeds={args.seeds} n_classes_present={int(present.sum())} "
        f"mIoU={mean[0]:.4f}+/-{std[0]:.4f} mAcc={mean[1]:.4f}+/-{std[1]:.4f} "
        f"OA={mean[2]:.4f}+/-{std[2]:.4f}")


if __name__ == "__main__":
    main()
