#!/bin/bash
set -eu
mode=$1
size=$2
stage=$3
seed=$((42+${SLURM_ARRAY_TASK_ID:-0}))
name=gemma_cv_nov02_${mode}_o${size}_s${seed}
source_root=/orcd/pool/003/kirilb/Steganography/releases/gemma_cv_nov02_v1
scratch=/orcd/scratch/orcd/003/kirilb/Steganography
cd "$source_root"
export EXPERIMENT_CONFIG="$source_root/configs/$name.json"
validation="$scratch/diagnostics/gemma_cv_nov02_${mode}_o${size}_s42/validation_001"
if [[ $stage == validate ]]; then run="$validation"; else run="$scratch/runs/$name"; fi
if [[ $stage == score ]]; then
  exec /orcd/scratch/orcd/003/kirilb/Steganography/envs/qwen3_pair_torch240_tf4513/bin/python cross_score.py "$size" "$seed"
fi
exec /bin/bash orcd/qwen_pair.slurm "$source_root" "$run" "$validation/VALIDATION.json" "$stage"
