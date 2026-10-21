#!/bin/bash
#SBATCH --job-name=uni-h100-probe
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
set -euo pipefail
PROBE_ROOT=${1:?probe source required}
TRAIN_SOURCE=${2:?training source required}
TRAIN_RUN=${3:?training run required}
CHECKPOINT=${4:?checkpoint required}
OUTPUT_ROOT=${5:?separate output required}
LEGACY_PY=/orcd/pool/003/kirilb/Steganography/envs/uni_author_torch240/bin/python
export PYTHONPATH="$PROBE_ROOT/deps"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROBE_ROOT"
nvidia-smi
exec "$LEGACY_PY" "$PROBE_ROOT/university_h100_probe.py" \
  --training-source "$TRAIN_SOURCE" --training-run "$TRAIN_RUN" \
  --checkpoint "$CHECKPOINT" --output "$OUTPUT_ROOT"
