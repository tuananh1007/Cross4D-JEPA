#!/usr/bin/env bash
# Granularity (paper Table 1): random-init / global / dense linear probes on
# MSR-Action3D, DeformingThings4D and NTU RGB+D 60, Point4D student.
# Run from the repository root. Each pretrain+probe pair is independent.
set -euo pipefail
export PYTHONHASHSEED=0
PRE="python -m cross4d_jepa.engine.pretrain_dense"
LP="python -m cross4d_jepa.engine.linear_probe"

# MSR-Action3D, DINOv2 teacher: dense (5 seeds) and global (8 seeds)
for s in 0 1 2 3 4; do
  $PRE configs/msr_point4d.yaml --seed $s --out runs/msr_dense_dinov2_s$s
  $LP configs/msr_point4d.yaml --encoder runs/msr_dense_dinov2_s$s/encoder.pt
done
for s in 0 1 2 3 4 5 6 7; do
  $PRE configs/msr_point4d.yaml --global --seed $s --out runs/msr_global_dinov2_s$s
  $LP configs/msr_point4d.yaml --encoder runs/msr_global_dinov2_s$s/encoder.pt
done

# DeformingThings4D and NTU (1,000-clip subset), V-JEPA 2 teacher (set in the configs)
for s in 0 1 2 3 4; do
  for mode in dense global; do
    flag=$([ $mode = global ] && echo --global || true)
    $PRE configs/dt4d_point4d.yaml $flag --seed $s --out runs/dt4d_${mode}_vjepa2_s$s
    $LP configs/dt4d_point4d.yaml --encoder runs/dt4d_${mode}_vjepa2_s$s/encoder.pt
    $PRE configs/ntu_point4d.yaml $flag --max-clips 1000 --epochs 100 --seed $s --out runs/ntu_${mode}_vjepa2_s$s
    $LP configs/ntu_point4d.yaml --encoder runs/ntu_${mode}_vjepa2_s$s/encoder.pt
  done
done

# random-init baselines (3 seeds each; the MSR probe used the 32-frame resampling)
for cfg in configs/msr_point4d_resample32.yaml configs/dt4d_point4d.yaml configs/ntu_point4d.yaml; do
  for s in 0 1 2; do
    python scripts/run_transfer_eval.py --encoder random --arch-config $cfg --target-config $cfg --seed $s
  done
done
