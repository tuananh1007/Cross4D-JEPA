#!/usr/bin/env bash
# Granularity vs. teacher modality (paper Table 2): MSR-Action3D, Point4D student,
# dense and global distillation from DINOv3, V-JEPA 2 and VideoMAE V2 (5 seeds).
# The DINOv2 row is produced by scripts/reproduce/main.sh. DINOv3 needs the local
# weight file (see README). Run from the repository root.
set -euo pipefail
export PYTHONHASHSEED=0
for t in dinov3 vjepa2 videomaev2; do
  for s in 0 1 2 3 4; do
    for mode in dense global; do
      flag=$([ $mode = global ] && echo --global || true)
      python -m cross4d_jepa.engine.pretrain_dense configs/msr_point4d.yaml --teacher $t $flag \
             --seed $s --out runs/msr_${mode}_${t}_s$s
      python -m cross4d_jepa.engine.linear_probe configs/msr_point4d.yaml --encoder runs/msr_${mode}_${t}_s$s/encoder.pt
    done
  done
done
