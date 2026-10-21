#!/bin/bash
#SBATCH --job-name=piqa-hybrid
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
SOURCE_ROOT=${1:?source directory required}
RUN_ROOT=${2:?run directory required}
PYTHON=/orcd/pool/003/kirilb/SherlockHolmes/ralif_with_explicit_reward/smollm_grpo_20260918/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SOURCE_ROOT"
nvidia-smi
run_child() {
  "$@" &
  TRAIN_PID=$!
  PREEMPTED=0
  trap 'PREEMPTED=1; kill -USR1 "$TRAIN_PID" 2>/dev/null || true' USR1 TERM
  set +e
  wait "$TRAIN_PID"
  STATUS=$?
  if kill -0 "$TRAIN_PID" 2>/dev/null; then wait "$TRAIN_PID"; STATUS=$?; fi
  set -e
  if [[ "$STATUS" == 75 || "$PREEMPTED" == 1 ]]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
  fi
  if [[ "$STATUS" != 0 ]]; then exit "$STATUS"; fi
}
if [[ ! -f "$RUN_ROOT/smoke_passed" ]]; then
  run_child "$PYTHON" -m stego.smoke --config "$SOURCE_ROOT/configs/piqa_hybrid_pilot.json" --output "$RUN_ROOT/smoke.json"
  touch "$RUN_ROOT/smoke_passed"
fi
run_child "$PYTHON" -m stego.train --config "$SOURCE_ROOT/configs/piqa_hybrid_pilot.json" --run-dir "$RUN_ROOT"
