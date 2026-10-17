#!/usr/bin/env bash
# Contribution of the teacher's learned semantics (paper Table 3): dense distillation
# on MSR-Action3D from non-foundation teachers at DINOv2's 9x9 grid (3 seeds), then
# the MSR linear probe and the HOI4D zero-shot per-point segmentation probe.
# The DINOv2 row reuses runs/msr_dense_dinov2_s{0,1,2} from scripts/reproduce/main.sh.
# Run from the repository root.
set -euo pipefail
export PYTHONHASHSEED=0
for t in vit-in1k vit-rand rawpix; do
  for s in 0 1 2; do
    python -m cross4d_jepa.engine.pretrain_dense configs/msr_point4d.yaml --teacher $t --seed $s --out runs/msr_dense_${t}_s$s
  done
done
for run in msr_dense_vit-in1k msr_dense_vit-rand msr_dense_rawpix msr_dense_dinov2; do
  for s in 0 1 2; do
    python -m cross4d_jepa.engine.linear_probe configs/msr_point4d.yaml --encoder runs/${run}_s$s/encoder.pt
    python scripts/hoi4d_semseg_eval.py configs/hoi4d_semseg_point4d.yaml --encoder runs/${run}_s$s/encoder.pt
  done
done
