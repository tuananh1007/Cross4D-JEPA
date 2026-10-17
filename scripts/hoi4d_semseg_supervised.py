"""SUPERVISED upper-bound for HOI4D per-point 4D semseg (Point4D + per-point head
trained END-TO-END), on the EXACT SAME data / split / readout / metric as the
frozen linear probe in scripts/hoi4d_semseg_eval.py.

The ONLY change vs the frozen probe is that the encoder is NOT frozen: Point4D and
the per-point linear head are trained jointly by per-point cross-entropy (ignore
index 0) on the train split (split==0), then evaluated on the val split (split==1)
with the identical 3-NN token->point readout and the identical streaming-confusion
mIoU/mAcc/OA (over classes PRESENT in the val split, ignore-0 excluded). The number
is therefore directly comparable to:
  RESULT tag=indomain_dense ... mIoU=23.59   (frozen dense-pretrained probe)
  RESULT tag=zeroshot_random ... mIoU=4.62   (frozen random encoder probe)

Two variants (--init):
  scratch : encoder random-init, trained end-to-end  -> the supervised ceiling.
  ftdense : encoder loaded from a dense-pretrained encoder.pt, trained end-to-end
            -> whether dense pretraining still helps under full supervision.

Reuses VERBATIM from scripts/hoi4d_semseg_eval.py: threenn_interp (now grad-enabled
in train), metrics_from_conf, NUM_CLASSES=49, IGNORE=0, the HOI4DSemSegClips
provider, and the val-time confusion-matrix logic (all val points, no subsample).
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from cross4d_jepa.config import Config
from cross4d_jepa.models import build_encoder
from cross4d_jepa.data.datasets.hoi4d_semseg import HOI4DSemSegClips

NUM_CLASSES = 49          # labels 0..48; 0 = ignore/background  (same as the probe)
IGNORE = 0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def threenn_interp(tok_xyz, tok_feat, query_xyz):
    """3-NN inverse-distance interpolation (PointNet++ FeaturePropagation), VERBATIM
    from scripts/hoi4d_semseg_eval.py. tok_xyz [M,3], tok_feat [M,C], query_xyz
    [N,3] -> [N,C]. Differentiable wrt tok_feat (used inside the training graph)."""
    d = torch.cdist(query_xyz, tok_xyz)                      # [N,M]
    k = min(3, tok_xyz.shape[0])
    dist, idx = d.topk(k, dim=-1, largest=False)             # [N,k]
    w = 1.0 / (dist + 1e-8)
    w = w / w.sum(dim=-1, keepdim=True)                      # [N,k]
    nf = tok_feat[idx]                                       # [N,k,C]
    return (nf * w.unsqueeze(-1)).sum(dim=1)                 # [N,C]


def metrics_from_conf(conf, present):
    """conf [K,K] (row=true,col=pred), present = bool mask of classes present in val.
    Returns (mIoU, mAcc, OA) over present classes (ignore index already excluded).
    VERBATIM from scripts/hoi4d_semseg_eval.py."""
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


def clip_pointwise_logits(encoder, head, pts, T):
    """pts [1,T,N,3] -> per-point logits [T,N,NUM_CLASSES] via the SAME readout as the
    probe: encoder tokens -> per-frame 3-NN interp to all N points -> linear head."""
    out = encoder(pts)                                       # tokens etc.
    N = pts.shape[2]
    tok = out["tokens"][0]                                   # [T*n2, C]
    txyz = out["token_xyz"][0]                               # [T*n2, 3]
    ttok = out["token_t"][0]                                 # [T*n2]
    feats = []
    for f in range(T):
        sel = ttok == f
        feats.append(threenn_interp(txyz[sel], tok[sel], pts[0, f]))  # [N,C]
    feats = torch.stack(feats, 0)                            # [T,N,C]
    return head(feats)                                       # [T,N,NUM_CLASSES]


@torch.no_grad()
def evaluate(encoder, head, ds, device, T, present):
    """Streaming confusion matrix over ALL val points (no subsample), exactly like the
    probe's eval. Returns (mIoU, mAcc, OA, conf)."""
    encoder.eval()
    head.eval()
    K = NUM_CLASSES
    conf = torch.zeros(K, K, dtype=torch.long, device=device)
    for rec in ds.records:
        pts = torch.from_numpy(ds.load_full(rec)[:T]).float().unsqueeze(0).to(device)  # [1,T,N,3]
        logits = clip_pointwise_logits(encoder, head, pts, T)                          # [T,N,K]
        pred = logits.argmax(-1).reshape(-1)                                           # [T*N]
        lab = torch.from_numpy(ds.semantic(rec)[:T].astype(np.int64).reshape(-1)).to(device)
        m = lab != IGNORE
        lab, pred = lab[m], pred[m]
        lin = lab * K + pred
        conf += torch.bincount(lin, minlength=K * K).reshape(K, K)
    miou, macc, oa = metrics_from_conf(conf.cpu(), present)
    return miou, macc, oa, conf.cpu()


def main():
    ap = argparse.ArgumentParser(description="HOI4D per-point semseg SUPERVISED end-to-end (Point4D + per-point head)")
    ap.add_argument("config")
    ap.add_argument("--init", choices=["scratch", "ftdense"], required=True,
                    help="scratch=random encoder; ftdense=load --encoder dense-pretrained ckpt")
    ap.add_argument("--encoder", default=None, help="encoder.pt for --init ftdense")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--amp", action="store_true", help="autocast bf16 forward/backward")
    ap.add_argument("--eval-every", type=int, default=10, help="eval val every N epochs (always eval last)")
    ap.add_argument("--tag", default="semseg_sup")
    ap.add_argument("--out-dir", default=None, help="dir for ckpt + resume (optional)")
    ap.add_argument("--max-clips", type=int, default=None, help="cap clips per split (smoke only)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = cfg.data.num_frames

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    encoder = build_encoder(cfg.encoder)
    if args.init == "ftdense":
        assert args.encoder, "--init ftdense requires --encoder"
        state = torch.load(args.encoder, map_location="cpu")
        encoder.load_state_dict(state["encoder"])
        log(f"init=ftdense: loaded encoder {args.encoder}")
    else:
        log("init=scratch: random-init encoder")
    encoder = encoder.to(device).train()
    # encoder NOT frozen -- the ONLY difference vs the frozen probe.
    head = nn.Linear(encoder.embed_dim, NUM_CLASSES).to(device)

    log("building datasets...")
    tr = HOI4DSemSegClips(cfg, "train")
    te = HOI4DSemSegClips(cfg, "test")
    if args.max_clips:
        tr.records = tr.records[:args.max_clips]
        te.records = te.records[:args.max_clips]
    log(f"  train clips: {len(tr.records)}  val clips: {len(te.records)}")

    # classes PRESENT in the val split (ignore-0 excluded) -- same metric scope as probe.
    present = torch.zeros(NUM_CLASSES, dtype=torch.bool)
    pres = set()
    for rec in te.records:
        pres.update(np.unique(te.semantic(rec)[:T]).tolist())
    pres.discard(IGNORE)
    for c in pres:
        present[c] = True
    log(f"  classes present in val (excl ignore): {int(present.sum())} -> {sorted(pres)}")

    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(tr.records)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch

    def lr_at(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))

    lossf = nn.CrossEntropyLoss(ignore_index=IGNORE)
    amp_dtype = torch.bfloat16
    step = 0
    start_epoch = 0

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt") if args.out_dir else None
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu")
        encoder.load_state_dict(ck["encoder"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"]
        step = ck["step"]
        log(f"RESUMED from {ckpt_path} at epoch {start_epoch} (step {step})")
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    log(f"training END-TO-END: init={args.init} seed={args.seed} epochs={args.epochs} "
        f"lr={args.lr} amp={args.amp} | {steps_per_epoch} clips/epoch")
    for ep in range(start_epoch, args.epochs):
        encoder.train()
        head.train()
        order = np.random.permutation(len(tr.records))
        t0 = time.time()
        run_loss, nb = 0.0, 0
        for j in order:
            rec = tr.records[j]
            pts = torch.from_numpy(tr.load_full(rec)[:T]).float().unsqueeze(0).to(device)  # [1,T,N,3]
            lab = torch.from_numpy(tr.semantic(rec)[:T].astype(np.int64)).reshape(-1).to(device)  # [T*N]
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            if args.amp:
                with torch.autocast("cuda", dtype=amp_dtype):
                    logits = clip_pointwise_logits(encoder, head, pts, T).reshape(-1, NUM_CLASSES)
                    loss = lossf(logits.float(), lab)
            else:
                logits = clip_pointwise_logits(encoder, head, pts, T).reshape(-1, NUM_CLASSES)
                loss = lossf(logits, lab)
            loss.backward()
            if args.clip_grad:
                torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
            opt.step()
            step += 1
            run_loss += loss.item()
            nb += 1
        log(f"  epoch {ep+1}/{args.epochs}  loss={run_loss/max(1,nb):.4f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  ({time.time()-t0:.0f}s)")

        if ckpt_path:
            torch.save({"encoder": encoder.state_dict(), "head": head.state_dict(),
                        "opt": opt.state_dict(), "epoch": ep + 1, "step": step}, ckpt_path)

        last = ep == args.epochs - 1
        if last or (args.eval_every and (ep + 1) % args.eval_every == 0):
            t0 = time.time()
            miou, macc, oa, _ = evaluate(encoder, head, te, device, T, present)
            tag = f"{args.tag}_s{args.seed}"
            kind = "RESULT" if last else "EVAL"
            log(f"{kind}[{tag}] epoch={ep+1} n_classes_present={int(present.sum())} "
                f"mIoU={miou*100:.4f} mAcc={macc*100:.4f} OA={oa*100:.4f}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
