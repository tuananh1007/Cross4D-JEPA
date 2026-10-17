#!/usr/bin/env bash
# 4D semantic segmentation on HOI4D (paper Table 5): per-point linear probe of a
# frozen encoder (mIoU / mAcc / OA, 3 probe-head seeds each), for a random-init
# encoder, MSR-pretrained encoders (zero-shot) and HOI4D-pretrained ones (in-domain).
# Run from the repository root.
set -euo pipefail
export PYTHONHASHSEED=0
EVAL="python scripts/hoi4d_semseg_eval.py configs/hoi4d_semseg_point4d.yaml"
$EVAL --random-init
for mode in dense global; do
  flag=$([ $mode = global ] && echo --global || true)
  python -m cross4d_jepa.engine.pretrain_dense configs/msr_point4d_resample32.yaml $flag --seed 0 \
         --out runs/msr_${mode}_dinov2_zeroshot
  $EVAL --encoder runs/msr_${mode}_dinov2_zeroshot/encoder.pt
  python -m cross4d_jepa.engine.pretrain_dense configs/hoi4d_semseg_point4d.yaml $flag \
         --max-clips 500 --epochs 60 --seed 0 --out runs/hoi4d_${mode}_dinov2_indomain
  $EVAL --encoder runs/hoi4d_${mode}_dinov2_indomain/encoder.pt
done
