"""Submit the isolated r2 sweep. Run with the experiment Python on the ORCD login node."""
import argparse
import fcntl
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ATTEMPT = Path('/orcd/scratch/orcd/003/kirilb/Steganography/runs/gemma_cv_nov02_r2')
SIZES = ('0.8', '2', '4', '9', '27')
PREFIX = 'gcv2r2_'


def job_name(suffix):
    # The seven original validation gates also gate the rolling-retention jobs.
    return PREFIX + (suffix if suffix.startswith('v_') else 'keep1_' + suffix)


def resources(gpu):
    return ['--partition=mit_normal_gpu,mit_preemptable', '--gres=gpu:' + gpu + ':1',
            '--cpus-per-task=8', '--mem=256G', '--time=06:00:00']


def after(*ids):
    return ['--dependency=afterok:' + ':'.join(ids), '--kill-on-invalid-dep=yes']


def schedule(submit, gates_only):
    l40 = resources('l40s')
    h200 = resources('h200')
    gates = []
    # Only one validation runs per GPU class at a time, including its restart round trip.
    for mode, size in [('full', s) for s in SIZES[:-1]] + [('r16', '9'), ('r32', '9')]:
        deps = after(gates[-1]) if gates else []
        gates.append(submit(f'v_{mode}_{size}', l40 + ['--no-requeue'] + deps,
                            ['validate', mode, size]))
    big_gate = submit('v_full_27', h200 + ['--no-requeue'], ['validate', 'full', '27'])
    if gates_only:
        return

    train = ['--requeue', '--signal=B:USR1@900']
    # Persist lane order even through preemption: only successful final export
    # frees a slot. A Slurm array throttle alone can open another seed while a
    # preempted one still occupies tens of GiB of scratch.
    lanes = {'full_l40s': gates[-1], 'full_h200': big_gate,
             'lora_0': gates[-1], 'lora_1': gates[-1]}
    for mode, size in [('full', s) for s in SIZES] + [('r16', '9'), ('r32', '9')]:
        for seed in range(42, 47):
            if mode == 'full':
                lane = 'full_h200' if size == '27' else 'full_l40s'
            else:
                lane = 'lora_' + str((seed - 42) % 2)
            lanes[lane] = submit(f't_{mode}_{size}_s{seed}', (h200 if size == '27' else l40)
                                 + train + after(lanes[lane]), ['train', mode, size, str(seed)])
    # One task per scoring observer; each visits five seeds and five training
    # observers. All submitted tasks fit the account's per-user job limit.
    for size in SIZES:
        submit('score_' + size, (h200 if size == '27' else l40) + train
               + after(lanes['full_l40s'], lanes['full_h200']), ['score', size])
    submit('curves', ['--partition=mit_normal', '--cpus-per-task=1', '--mem=4G',
                      '--time=00:30:00'] + after(*lanes.values()), ['curves'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gates-only', action='store_true')
    args = parser.parse_args()
    ATTEMPT.mkdir(parents=True, exist_ok=True)
    with (ATTEMPT / 'SUBMIT_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = ATTEMPT / 'SUBMISSION.json'
        jobs = json.loads(receipt.read_text()) if receipt.exists() else {}
        if any(not name.startswith((PREFIX + 'v_', PREFIX + 'keep1_')) for name in jobs):
            raise RuntimeError('Reconcile the earlier submission receipt before queuing replacement jobs')
        logs = ATTEMPT / 'logs'
        logs.mkdir(exist_ok=True)

        def submit(suffix, options, script_args):
            name = job_name(suffix)
            if name in jobs:
                return jobs[name]
            queued = subprocess.check_output(['squeue', '-u', 'kirilb', '-n', name,
                                              '-h', '-o', '%i'], text=True).strip()
            prior = subprocess.check_output(['sacct', '-XnP', '-u', 'kirilb', '--name', name,
                                             '-S', '2026-11-02', '-o', 'JobID'], text=True).strip()
            if queued or prior:
                raise RuntimeError(f'Unrecorded job named {name}; reconcile before submitting')
            cmd = ['sbatch', '--parsable', '--account=mit_general', '--job-name=' + name,
                   '--chdir=' + str(HERE), '--open-mode=append',
                   '--output=' + str(logs / '%x-%A_%a.out'),
                   '--error=' + str(logs / '%x-%A_%a.err'), *options,
                   str(HERE / 'dispatch.sh'), *script_args]
            job = subprocess.check_output(cmd, text=True).strip().split(';')[0]
            if not job.isdigit():
                raise RuntimeError('Unexpected sbatch response')
            jobs[name] = job
            tmp = receipt.with_suffix('.tmp')
            tmp.write_text(json.dumps(jobs, indent=2) + '\n')
            tmp.replace(receipt)
            print(name, job, flush=True)
            return job

        schedule(submit, args.gates_only)


if __name__ == '__main__':
    main()
