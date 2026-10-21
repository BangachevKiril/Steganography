#!/bin/bash
#SBATCH --job-name=uni-qwen-guard
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=00:15:00
#SBATCH --no-requeue
set -euo pipefail
SOURCE_ROOT=${1:?frozen source required}
OUTPUT_ROOT=${2:?isolated output directory required}
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SOURCE_ROOT"
timeout --signal=TERM --kill-after=15 870 /orcd/pool/003/kirilb/Steganography/envs/qwen35_observer/bin/python -u "$SOURCE_ROOT/diagnostics/zero_guard_validation_001/validate.py" --source-root "$SOURCE_ROOT" --output-dir "$OUTPUT_ROOT" --budget-seconds 840
