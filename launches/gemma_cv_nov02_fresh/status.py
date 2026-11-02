"""Summarize Slurm and public Gemma r2 evidence without copying example or model data."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from submit import ATTEMPT, PREFIX

FROZEN = Path('/orcd/pool/003/kirilb/Steganography/releases/gemma_cv_nov02_v1')
METRICS = ('receiver_accuracy', 'observer_accuracy', 'receiver_pass', 'observer_pass',
           'observer_auc', 'literal_counts')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def evaluations(run, config_digest):
    points = {}
    for path in sorted((run / 'evaluations').glob('step_*/COMPLETE.json')):
        marker = read(path)
        raw = (path.parent / 'SUMMARY.json').read_bytes()
        summary = json.loads(raw)
        identity = run / 'evaluation_states' / path.parent.name / 'COMPLETE.json'
        if (marker.get('config_digest') != config_digest or summary['config_digest'] != config_digest
                or hashlib.sha256(raw).hexdigest() != marker.get('summary_sha256')
                or hashlib.sha256(identity.read_bytes()).hexdigest()
                != marker.get('source_checkpoint', {}).get('complete_sha256')
                or summary['university']['n'] != 2000):
            raise ValueError(f'Invalid public evaluation evidence at {path}')
        points[str(summary['completed_update'])] = {k: summary['university'][k] for k in METRICS}
    return points


def rows(command):
    output = subprocess.check_output(command, text=True)
    return [line.split('|') for line in output.splitlines() if line]


def snapshot():
    jobs = read(ATTEMPT / 'SUBMISSION.json')
    if not jobs:
        raise ValueError(f'No submission receipt at {ATTEMPT}')
    queue = {job: dict(name=name, state=state, partition=partition, reason=reason,
                       start_estimate=start, dependency=dependency)
             for job, name, state, partition, reason, start, dependency in
             rows(['squeue', '-r', '-h', '-u', 'kirilb', '-o', '%i|%j|%T|%P|%R|%S|%E'])
             if name.startswith(PREFIX)}
    accounting = {job: dict(state=state, exit_code=code, partition=partition, restarts=restarts)
                  for job, state, code, partition, restarts in
                  rows(['sacct', '-nPX', '--array', '-j', ','.join(jobs.values()),
                        '-o', 'JobID,State,ExitCode,Partition,Restarts'])}
    runs = {}
    cross_scores = 0
    gates = {}
    for config_path in sorted((FROZEN / 'configs').glob('*.json')):
        config = read(config_path)
        digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        run = ATTEMPT / 'runs' / config['name']
        ablation = config['ablation']
        condition = f"{ablation['receiver_training']}_{ablation['observer_size']}"
        checkpoints = [int(p.parent.name[5:]) for p in (run / 'checkpoints').glob('step_*/COMPLETE.json')
                       if (p.parent / 'state.pt').is_file() and read(p).get('config_digest') == digest]
        progress = read(run / 'STATUS.json')
        saved = read(run / 'final_weights' / 'COMPLETE.json')
        scores = sorted(p.stem for p in (run / 'final_cross_scoring').glob('observer_*.json'))
        cross_scores += len(scores)
        runs[config['name']] = dict(job=jobs.get(f"{PREFIX}t_{condition}_s{config['seed']}"),
                                    stage=progress.get('stage'), observed_update=progress.get('completed_update'),
                                    restart_marker_update=max(checkpoints, default=None),
                                    evaluations=evaluations(run, digest),
                                    completion_marker=read(run / 'COMPLETE.json').get('completed_update'),
                                    final_weights_manifest=saved.get('weights_sha256') if saved.get('config_digest') == digest else None,
                                    cross_score_files=scores,
                                    failure_category=read(run / 'FAILURE.json').get('failure_category'))
        if config['seed'] != 42:
            continue
        directory = ATTEMPT / 'diagnostics' / config['name'] / 'validation_001'
        validation = read(directory / 'VALIDATION.json')
        job = jobs.get(PREFIX + 'v_' + condition)
        full = ablation['receiver_training'] == 'full'
        scheduler = accounting.get(job, {})
        passed = (scheduler.get('state') == 'COMPLETED' and scheduler.get('exit_code') == '0:0'
                  and validation.get('status') == 'passed' and validation.get('config_digest') == digest
                  and validation.get('source_sha256') == config['source_sha256']
                  and validation.get('receiver', {}).get('receiver_training', {}).get('optimizer_sample_count') == 256
                  and validation.get('sender', {}).get('n') == 192 and validation.get('sender_trainable_updated')
                  and validation.get('evaluation_weight_identity') and validation.get('cpu_tests_passed') == 23
                  and validation.get('slurm_job_id') == job and validation.get('receiver_parameter_training') == config['receiver_parameter_training']
                  and validation.get('sender', {}).get('objective') == config['objective']
                  and validation.get('ablation_tests_passed') and validation.get('probability_tests_passed')
                  and (validation.get('receiver_full_updated') and validation.get('fp32_master_restart_verified')
                       if full else validation.get('receiver_lora_updated') and validation.get('receiver_base_unchanged')))
        gates[condition] = dict(job=job, verified=bool(passed), stage=read(directory / 'STATUS.json').get('stage'))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(), attempt=str(ATTEMPT), jobs=jobs,
                queue=queue, accounting=accounting, validation=gates, runs=runs,
                cross_score_files=cross_scores,
                curve_files=sorted(p.name for p in (ATTEMPT / 'gemma_cv_nov02_results').glob('*.svg')))


if __name__ == '__main__':
    print(json.dumps(snapshot(), indent=2))
