#!/usr/bin/env bash
# Matched comparison with 4D self-supervised pretexts (paper Table 6), MSR-Action3D,
# same Point4D student: Uni4D and MaST-Pre at 1200 epochs, i-JEPA at 300 epochs (its
# best budget), 3 seeds each, then the linear probe. Also fine-tuning (label fractions
# 0.1-1.0, 3 seeds) from scratch and from the dense encoder of
# scripts/reproduce/hoi4d_semseg.sh (runs/msr_dense_dinov2_zeroshot). All of these
# use the 32-frame MSR resampling (configs/msr_point4d_resample32.yaml).
# Run from the repository root.
set -euo pipefail
export PYTHONHASHSEED=0
LP="python -m cross4d_jepa.engine.linear_probe configs/msr_point4d_resample32.yaml"
for s in 0 1 2; do
  python -m cross4d_jepa.engine.pretrain_uni4d configs/msr_point4d_resample32.yaml --epochs 1200 --mask-ratio 0.75 \
         --seed $s --out runs/msr_uni4d_e1200_s$s
  $LP --encoder runs/msr_uni4d_e1200_s$s/encoder.pt
  python -m cross4d_jepa.engine.pretrain_mastpre configs/msr_point4d_resample32.yaml --epochs 1200 --mask-ratio 0.75 \
         --seed $s --out runs/msr_mastpre_e1200_s$s
  $LP --encoder runs/msr_mastpre_e1200_s$s/encoder.pt
  python -m cross4d_jepa.engine.pretrain_ijepa configs/msr_point4d_resample32.yaml --epochs 300 --mask-ratio 0.6 \
         --seed $s --out runs/msr_ijepa_e300_s$s
  $LP --encoder runs/msr_ijepa_e300_s$s/encoder.pt
done
for f in 0.1 0.25 0.5 1.0; do
  for s in 0 1 2; do
    C4D_SEED=$s python -m cross4d_jepa.engine.finetune configs/msr_point4d_resample32.yaml --epochs 150 --no-augment \
           --label-fraction $f --tag scratch_f${f}_s$s
    C4D_SEED=$s python -m cross4d_jepa.engine.finetune configs/msr_point4d_resample32.yaml --epochs 150 --no-augment \
           --label-fraction $f --encoder runs/msr_dense_dinov2_zeroshot/encoder.pt --tag dense_f${f}_s$s
  done
done
