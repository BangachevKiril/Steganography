#!/bin/bash
#SBATCH --job-name=uni-qwen3-lora9b
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --requeue
#SBATCH --open-mode=append
set -euo pipefail
SOURCE_ROOT=${1:?source}
RUN_ROOT=${2:?run}
VALIDATION=${3:?validation JSON}
MODE=${4:?validate or train}
PYTHON=/orcd/scratch/orcd/003/kirilb/Steganography/envs/qwen3_pair_torch240_tf4513/bin/python
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$SOURCE_ROOT"
CHILD_PID=
trap 'if [[ -n "$CHILD_PID" ]]; then kill -USR1 "$CHILD_PID" 2>/dev/null || true; fi' USR1 TERM
ARGS=()
if [[ "$MODE" == validate ]]; then
    ARGS=(--validate)
    # These checks gate the real full-batch GPU validation and all training jobs.
    PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -m unittest discover -s tests -p 'test_qwen_pair.py' -v
    PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -m unittest discover -s tests -p 'test_sparse*.py' -v
    PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -m unittest discover -s tests -p 'test_probability*.py' -v
    PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -m unittest discover -s tests -p 'test_qwen_observer_ipc.py' -v
    PYTHONPATH="$SOURCE_ROOT" "$PYTHON" -m unittest discover -s tests -p 'test_ablation*.py' -v
    export ABLATION_CPU_VALIDATION_PASSED=passed
    export PROBABILITY_CPU_VALIDATION_PASSED=passed
    export SPARSE_CPU_VALIDATION_PASSED=23
fi
"$PYTHON" "$SOURCE_ROOT/orcd/qwen_pair_run.py" --config "${EXPERIMENT_CONFIG:?}" --run-dir "$RUN_ROOT" --validation "$VALIDATION" "${ARGS[@]}" &
CHILD_PID=$!
set +e
wait "$CHILD_PID"
STATUS=$?
if kill -0 "$CHILD_PID" 2>/dev/null; then wait "$CHILD_PID"; STATUS=$?; fi
set -e
if [[ "$STATUS" == 75 && "$MODE" != validate ]]; then scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; exit 0; fi
exit "$STATUS"
