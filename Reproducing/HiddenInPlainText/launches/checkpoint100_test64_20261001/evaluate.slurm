#!/bin/bash
#SBATCH --job-name=piqa-test64-c100
#SBATCH --partition=mit_preemptable
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
EVAL_SOURCE=${1:?evaluation source directory required}
PYTHON=/orcd/pool/003/kirilb/SherlockHolmes/ralif_with_explicit_reward/smollm_grpo_20260918/venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$EVAL_SOURCE"
"$PYTHON" evaluate.py --config config.json &
CHILD_PID=$!
PREEMPTED=0
trap 'PREEMPTED=1; kill -USR1 "$CHILD_PID" 2>/dev/null || true' USR1 TERM
set +e
wait "$CHILD_PID"
STATUS=$?
if kill -0 "$CHILD_PID" 2>/dev/null; then wait "$CHILD_PID"; STATUS=$?; fi
set -e
if [[ "$STATUS" == 75 || "$PREEMPTED" == 1 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
exit "$STATUS"
