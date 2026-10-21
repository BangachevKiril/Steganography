#!/bin/bash
#SBATCH --job-name=uni-corrected-l40s
#SBATCH --partition=mit_normal_gpu,mit_preemptable
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
SOURCE_ROOT=${1:?corrected source directory required}
RUN_ROOT=${2:?corrected run directory required}
unset PYTHONPATH
LEGACY_PY=/orcd/pool/003/kirilb/Steganography/envs/uni_author_torch240/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SOURCE_ROOT"
nvidia-smi
PREEMPTED=0
CHILD_PID=
trap 'PREEMPTED=1; if [[ -n "$CHILD_PID" ]]; then kill -USR1 "$CHILD_PID" 2>/dev/null || true; fi' USR1 TERM
run_stage() {
  "$@" &
  CHILD_PID=$!
  set +e
  wait "$CHILD_PID"
  STATUS=$?
  if kill -0 "$CHILD_PID" 2>/dev/null; then wait "$CHILD_PID"; STATUS=$?; fi
  set -e
  CHILD_PID=
  if [[ "$STATUS" == 75 || "$PREEMPTED" == 1 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if [[ "$STATUS" != 0 ]]; then exit "$STATUS"; fi
}
run_stage "$LEGACY_PY" "$SOURCE_ROOT/orcd/university_corrected_l40s.py" --training-source "$SOURCE_ROOT" --config "$SOURCE_ROOT/configs/university_training.json" --run-dir "$RUN_ROOT"
