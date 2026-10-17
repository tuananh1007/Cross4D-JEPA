#!/usr/bin/env bash
# Granularity across backbones and token granularities (paper Table 4), MSR-Action3D,
# DINOv2 teacher. Point4D token granularity n2 in {16,4,2,1} (n2=64 is
# scripts/reproduce/main.sh) and the official P4Transformer at four ball-query radii.
# The P4Transformer runs need the pointnet2 CUDA extension (see README).
# Run from the repository root.
set -euo pipefail
export PYTHONHASHSEED=0
PRE="python -m cross4d_jepa.engine.pretrain_dense"
LP="python -m cross4d_jepa.engine.linear_probe"
for n in 16 4 2 1; do
  for s in 0 1 2; do
    for mode in dense global; do
      flag=$([ $mode = global ] && echo --global || true)
      $PRE configs/ablations/gran_n$n.yaml $flag --seed $s --out runs/msr_${mode}_n${n}_s$s
      $LP configs/ablations/gran_n$n.yaml --encoder runs/msr_${mode}_n${n}_s$s/encoder.pt
    done
  done
done
# official radius 0.7: dense and global, 5 seeds
for s in 0 1 2 3 4; do
  for mode in dense global; do
    flag=$([ $mode = global ] && echo --global || true)
    $PRE configs/ablations/p4t_r070.yaml $flag --seed $s --out runs/p4t_r070_${mode}_s$s
    $LP configs/ablations/p4t_r070.yaml --encoder runs/p4t_r070_${mode}_s$s/encoder.pt
  done
done
# radius sweep: dense, 3 seeds
for r in 040 020 010; do
  for s in 0 1 2; do
    $PRE configs/ablations/p4t_r$r.yaml --seed $s --out runs/p4t_r${r}_dense_s$s
    $LP configs/ablations/p4t_r$r.yaml --encoder runs/p4t_r${r}_dense_s$s/encoder.pt
  done
done
