#!/bin/bash
#SBATCH --job-name=sender-eos-1076
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
set -euo pipefail
PROBE_ROOT=${1:?probe source required}
TRAIN_RUN=${2:?training run required}
OUTPUT_ROOT=${3:?separate output required}
LEGACY_PY=/orcd/pool/003/kirilb/Steganography/envs/uni_author_torch240/bin/python
unset PYTHONPATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$PROBE_ROOT"
nvidia-smi
exec "$LEGACY_PY" "$PROBE_ROOT/extract_embeddings.py" \
  --training-run "$TRAIN_RUN" --split-dir "$PROBE_ROOT/split" --output "$OUTPUT_ROOT"
