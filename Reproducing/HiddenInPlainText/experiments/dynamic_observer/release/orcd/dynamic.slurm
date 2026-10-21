#!/bin/bash
#SBATCH --job-name=dynamic-observer
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
SOURCE_ROOT=${1:?}; CONFIG=${2:?}; RUN=${3:?}; VALIDATION=${4:?}; MODE=${5:-train}
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SOURCE_ROOT"
CHILD_PID=
trap 'if [[ -n "$CHILD_PID" ]]; then kill -USR1 "$CHILD_PID" 2>/dev/null || true; fi' USR1 TERM
ARGS=(--config "$CONFIG" --run-dir "$RUN" --validation "$VALIDATION")
if [[ "$MODE" == validate ]]; then ARGS+=(--validate); fi
/orcd/pool/003/kirilb/Steganography/envs/uni_author_torch240/bin/python orcd/dynamic_run.py "${ARGS[@]}" &
CHILD_PID=$!
set +e
wait "$CHILD_PID"
STATUS=$?
if kill -0 "$CHILD_PID" 2>/dev/null; then wait "$CHILD_PID"; STATUS=$?; fi
set -e
if [[ "$STATUS" == 75 ]]; then scontrol requeue "$SLURM_JOB_ID"; exit 0; fi
exit "$STATUS"
