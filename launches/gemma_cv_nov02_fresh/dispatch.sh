#!/bin/bash
set -euo pipefail
frozen=/orcd/pool/003/kirilb/Steganography/releases/gemma_cv_nov02_v1
here=/orcd/pool/003/kirilb/Steganography/releases/gemma_cv_nov02_fresh
attempt=/orcd/scratch/orcd/003/kirilb/Steganography/runs/gemma_cv_nov02_r2
python=/orcd/scratch/orcd/003/kirilb/Steganography/envs/qwen3_pair_torch240_tf4513/bin/python
unset PYTHONPATH PYTHONHOME
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Forward the batch-shell walltime signal to the child and preserve the Slurm
# job ID and dependency on graceful checkpoint/retry.
run_with_requeue() {
    local child='' rc=0 signalled=0 wait_interrupted=0
    trap 'signalled=1; wait_interrupted=1; if [[ -n "$child" ]]; then kill -USR1 "$child" 2>/dev/null || true; fi' USR1 TERM
    "$@" &
    child=$!
    if (( signalled )); then kill -USR1 "$child" 2>/dev/null || true; fi
    while true; do
        wait_interrupted=0
        if wait "$child"; then rc=0; break; else rc=$?; fi
        # Bash can return the signal while the child is still checkpointing.
        if (( rc <= 128 || !wait_interrupted )); then break; fi
    done
    trap - USR1 TERM
    # A signal received before Python installs its handlers terminates it with
    # 128+SIGUSR1. A successful child can also finish concurrently with a signal.
    if (( rc == 75 || (signalled && (rc == 0 || rc == 128 + $(kill -l USR1))) )); then
        scontrol requeue "${SLURM_JOB_ID}"; exit 0
    fi
    return "$rc"
}
if [[ ${BASH_SOURCE[0]} != "$0" ]]; then return; fi
cd "$frozen"
case "$1" in
curves)
    exec "$python" "$here/results.py" "$@" ;;
score)
    run_with_requeue "$python" "$here/results.py" "$@"; exit 0 ;;
validate)
    mode=$2; size=$3; seed=42 ;;
train)
    mode=$2; size=$3; seed=$4 ;;
*) exit 2 ;;
esac
name=gemma_cv_nov02_${mode}_o${size}_s${seed}
export EXPERIMENT_CONFIG="$frozen/configs/$name.json"
validation="$attempt/diagnostics/gemma_cv_nov02_${mode}_o${size}_s42/validation_001"
if [[ $1 == validate ]]; then
    exec /bin/bash "$frozen/orcd/qwen_pair.slurm" "$frozen" "$validation" "$validation/VALIDATION.json" validate
fi
run="$attempt/runs/$name"
if [[ ! -f "$run/COMPLETE.json" ]]; then
    run_with_requeue "$python" "$frozen/orcd/qwen_pair_run.py" --config "$EXPERIMENT_CONFIG" --run-dir "$run" --validation "$validation/VALIDATION.json"
fi
run_with_requeue "$python" "$here/finalize.py" "$run" "$EXPERIMENT_CONFIG"
