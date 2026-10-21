#!/usr/bin/env python3
"""Validate a downloaded RUNNING continuation snapshot; never contacts the cluster.

Only a local monitor diagnostic is written. Evaluation row audits and launch-record
milestones remain the caller's responsibility. Slurm exclusivity is limited to
the visible matching-name jobs in the supplied queue listing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import sys

WORKSPACE = Path(__file__).resolve().parents[2]
CONFIG = '3fe0ebcb50bb71fc727f8255161f45531c3dfe073eab644d2423cc41ce13fa92'
ORIGIN = 530
ORIGIN_ELAPSED = 27930.404972237535
TARGET = ORIGIN_ELAPSED + 27900
VERSIONS = {'torch': '2.4.0+cu124', 'transformers': '4.43.2', 'trl': '0.9.6',
            'peft': '0.12.0', 'accelerate': '0.33.0', 'bitsandbytes': '0.43.3',
            'numpy': '1.26.4', 'datasets': '2.20.0', 'tokenizers': '0.19.1'}
MANIFESTS = ('CORRECTED_L40S_RUNTIME.json', 'L40S_MEMORY_OFFLOAD.json',
             'L40S_EVALUATION_MEMORY.json')
TERMINALS = ('FAILURE.json', 'BUDGET_COMPLETE.json', 'COMPLETE.json',
             'CONTINUATION_COMPLETE.json')


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True))


def finite(value, where):
    if isinstance(value, float):
        require(math.isfinite(value), 'Nonfinite number: ' + where)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, where + '.' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite(item, f'{where}[{index}]')


def phase(update):
    return 'receiver' if (update - 1) % 7 < 2 else 'sender'


def count(value, low, high, where):
    require(type(value) is int and low <= value <= high, 'Invalid count: ' + where)


def slurm_identity(text, launch, source_files):
    lines = text.splitlines()
    records = [line for line in lines if line.startswith('JobId=')]
    require(len(records) == 1, 'Expected one detailed Slurm job record')
    line = records[0]
    # Values such as SubmitLine contain spaces; ordinary split() loses boundaries.
    fields = dict(re.findall(r'(?:^| )([A-Za-z][A-Za-z0-9/:]*)=(.*?)(?= [A-Za-z][A-Za-z0-9/:]*=|$)', line))
    job = str(launch['job_id'])
    expected = {'JobId': job, 'JobState': 'RUNNING', 'JobName': 'uni-corr-ext',
                'Comment': launch['extension_id'], 'TimeLimit': '06:00:00',
                'NumNodes': '1', 'NumCPUs': '8', 'Requeue': '1',
                'Command': launch['remote_extension'] + '/university_corrected_extension.slurm',
                'StdOut': launch['run_dir'] + f'/slurm-{job}.out',
                'StdErr': launch['run_dir'] + f'/slurm-{job}.err'}
    for key, value in expected.items():
        require(fields.get(key) == value, f'Slurm {key} mismatch: {fields.get(key)!r}')
    require(shlex.split(fields['SubmitLine']) == launch['command'], 'Slurm submission command changed')
    require(fields.get('Partition') in ('mit_normal_gpu', 'mit_preemptable'), 'Unexpected GPU partition')
    for key in ('ReqTRES', 'AllocTRES'):
        tres = dict(item.split('=', 1) for item in fields[key].split(','))
        require(all(tres.get(k) == v for k, v in {'mem': '128G', 'gres/gpu': '1',
                                                'gres/gpu:l40s': '1', 'node': '1'}.items()),
                'Expected one L40S and 128G host RAM: ' + key)
    queue = [row.split() for row in text[:text.index('JobId=')].splitlines()
             if re.match(r'^\s*\d+\s', row)]
    matches = [row for row in queue if len(row) >= 3 and row[1] == fields['JobName']]
    require(len(matches) == 1 and matches[0][0] == job and matches[0][2] == 'RUNNING',
            'Queue does not contain exactly one RUNNING matching extension job')
    require(sum(row[0] == job for row in queue) == 1, 'Duplicate/missing queue job')
    require(not any(row[0] == str(launch['parent_job_id']) for row in queue),
            'Original job remains active in queue')
    identity = [json.loads(line) for line in lines if line.startswith('{')]
    require(len(identity) == 1 and identity[0].get('all_passed') is True,
            'Missing successful remote identity check result')
    checks = identity[0]['checks']
    required = set(source_files) | {'extension_wrapper', 'extension_manifest',
                                  'archive_manifest', 'pinned_origin_marker'}
    require(len(source_files) == 33 and required <= checks.keys()
            and all(value is True for value in checks.values()), 'Incomplete/failed source identity checks')
    return fields, identity[0]


def check_metrics(text, update):
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    require([row['update'] for row in rows] == list(range(1, update + 1)),
            'Metrics must be sequential 1..verified_update, with no later rows')
    for row in rows:
        u = row['update']
        finite(row, f'update{u}')
        require(row['config_digest'] == CONFIG and row['kind'] == phase(u)
                and row['cycle'] == (u - 1) // 7 + 1, f'Update{u} schedule/config mismatch')
        n = row['n']
        require(type(n) is int and n > 0 and row['duration_seconds'] > 0, f'Update{u} invalid population/duration')
        for key in ('receiver_correct', 'observer_flagged', 'receiver_pass', 'receiver_fail'):
            count(row[key], 0, n, f'update{u}.{key}')
        require(row['receiver_pass'] + row['receiver_fail'] == n, f'Update{u} prediction counts mismatch')
        if row['kind'] == 'sender':
            require(n == 192, f'Update{u} sender population changed')
            for key in ('reward_mean', 'reward_std'):
                require(type(row[key]) in (int, float), f'Update{u} missing {key}')
            require(row['reward_std'] >= 0, f'Update{u} negative reward variation')
            for key in ('ppo/loss/total', 'ppo/loss/policy', 'ppo/loss/value', 'objective/kl', 'objective/kl_coef'):
                require(type(row['ppo'].get(key)) in (int, float), f'Update{u} invalid PPO {key}')
        else:
            optimization = row['receiver_training']
            expected = {'n': 256, 'optimizer_sample_count': 256, 'microbatches': 4,
                        'optimizer_steps': 1, 'gradient_reduction': 'mean_cross_entropy_over_all_examples'}
            require(all(optimization.get(k) == v for k, v in expected.items()),
                    f'Update{u} receiver optimization changed')
            count(optimization['correct'], 0, 256, f'update{u}.optimization.correct')
            require(type(optimization['loss']) in (int, float), f'Update{u} missing receiver loss')
            batches = optimization['minibatch_label_counts']
            require(len(batches) == 4, f'Update{u} receiver microbatch count mismatch')
            for batch in batches:
                require(batch['n'] == 64 and batch['PASS'] + batch['FAIL'] == 64,
                        f'Update{u} receiver microbatch size mismatch')
                for label in ('PASS', 'FAIL'):
                    count(batch[label], 0, 64, f'update{u}.microbatch.{label}')
            require(all(sum(batch[label] for batch in batches) == optimization['label_counts'][label]
                        for label in ('PASS', 'FAIL')), f'Update{u} receiver optimization label totals mismatch')
    return rows


def parity(value):
    flags = ('exact_initial_parameters', 'exact_outputs', 'exact_loss', 'exact_gradients',
             'exact_adamw_state', 'exact_updated_parameters', 'exact_post_test_rng',
             'exact_baseline_repeat', 'caller_rng_restored', 'cuda_transfer_verified',
             'all_packed_cuda_tensors_on_pinned_cpu')
    require(value.get('status') == 'passed' and value.get('device') == 'NVIDIA L40S'
            and all(value.get(k) is True for k in flags), 'Exact CUDA toy parity/RNG restoration failed')
    require(value['actual_training_model_updates'] == 0, 'Toy preflight changed training models')


def audit(args):
    launch = read(args.launch)
    snapshot = read(args.snapshot)
    update, files = snapshot.get('verified_update'), snapshot['files']
    require(type(update) is int and update >= ORIGIN and snapshot.get('state_sha256_verified') is True,
            'Routine audit requires a remotely hash-verified continuation checkpoint')
    results = Path(launch['local_results'])
    require(results.resolve().is_relative_to(WORKSPACE / 'results'), 'Diagnostic destination is outside workspace results')
    get = lambda name: json.loads(files[name])
    config = get('config.json')
    require(digest(config) == CONFIG == launch['config_digest'], 'Scientific config digest changed')
    require(config['max_active_seconds'] == 27900 and config['target_gpu'] == 'L40S', 'Original config budget/GPU changed')
    require(config['source_sha256'] == launch['source_sha256'], 'Frozen source manifest changed')
    identity_text = args.identity.read_text()
    slurm, identity = slurm_identity(identity_text, launch, config['source_sha256'])
    require(not any(name in files for name in TERMINALS), 'Failure or terminal marker present; use exception/completion workflow')
    status = get('STATUS.json')
    require(status['status'] == 'running' and status['config_digest'] == CONFIG
            and str(status['slurm_job_id']) == str(launch['job_id']), 'STATUS does not describe current running job')
    run = launch['run_dir']
    checkpoint_name = f'checkpoints/step_{update:06d}/COMPLETE.json'
    checkpoint = get(checkpoint_name)
    require(snapshot['checkpoint'] == run + f'/checkpoints/step_{update:06d}'
            and checkpoint['config_digest'] == CONFIG and checkpoint['completed_update'] == update
            and checkpoint['phase'] == phase(update), 'Verified checkpoint identity mismatch')
    finite(checkpoint, 'checkpoint')
    require(checkpoint['elapsed'] >= ORIGIN_ELAPSED, 'Checkpoint elapsed time reset')
    manifest_text = Path(launch['extension_manifest']).read_text()
    manifest = json.loads(manifest_text)
    require(sha(manifest_text) == launch['extension_manifest_sha256']
            and get('CONTINUATION.json') == manifest, 'Sealed continuation manifest changed')
    for key, expected in {'config_digest': CONFIG, 'origin_update': ORIGIN,
                          'origin_elapsed_seconds': ORIGIN_ELAPSED, 'additional_active_seconds': 27900,
                          'target_active_seconds': TARGET, 'run_dir': run,
                          'wrapper_sha256': launch['wrapper_sha256'],
                          'archive_manifest_sha256': launch['archive_manifest_sha256'],
                          'training_source': launch['remote_source']}.items():
        require(manifest.get(key) == expected, 'Continuation identity/budget mismatch: ' + key)
    require(launch['target_active_seconds'] == TARGET, 'Launch target budget changed')
    origin_name = f'checkpoints/step_{ORIGIN:06d}/COMPLETE.json'
    require(sha(files[origin_name]) == manifest['parent_checkpoint_complete_sha256']
            and get(origin_name)['state_sha256'] == manifest['parent_state_sha256'], 'Pinned origin marker changed')
    stage1 = WORKSPACE / 'results' / config['name']
    for name in (*MANIFESTS, 'INITIALIZATION.json'):
        require(files[name] == (stage1 / name).read_text(), 'Inherited runtime manifest changed: ' + name)
    runtime = get(MANIFESTS[0])
    require(runtime['source_sha256'] == config['source_sha256'], 'Runtime source manifest differs')
    rows = check_metrics(files['metrics.jsonl'], update)
    timing = get('TIMING.json')
    finite(timing, 'timing')
    require(timing['config_digest'] == CONFIG, 'TIMING config mismatch')
    if timing['completed_update'] <= update:
        prefix = rows[:timing['completed_update']]
        require(timing['counts'] == {kind: sum(row['kind'] == kind for row in prefix)
                                    for kind in ('sender', 'receiver')}, 'TIMING update counts mismatch')
    events = {name: json.loads(text) for name, text in files.items() if name.startswith('runtime_events/')}
    current = {name: event for name, event in events.items()
               if str(event.get('slurm_job_id')) == str(launch['job_id'])}

    def latest(kind, time_key):
        matches = [(name, event) for name, event in current.items() if event.get('kind') == kind]
        require(bool(matches), 'Missing current-job event: ' + kind)
        return max(matches, key=lambda item: item[1][time_key])

    def event_pins(event, extension=False):
        require(event.get('config_digest') == CONFIG, 'Runtime event config mismatch')
        require(event.get('wrapper_sha256') == (manifest['wrapper_sha256'] if extension else runtime['wrapper_sha256']),
                'Runtime event wrapper mismatch')
        for key in ('memory_helper_sha256', 'evaluation_helper_sha256'):
            if key in event:
                require(event[key] == runtime[key], 'Runtime helper mismatch: ' + key)

    preflight_name, preflight = latest('corrected_L40S_initialization', 'started_at')
    event_pins(preflight)
    restart = lambda event: int(event.get('slurm_restart_count') or 0)
    require(restart(preflight) == int(slurm['Restarts']), 'Latest startup does not match Slurm restart count')
    require(preflight['status'] == 'initialization_verified', 'Current startup not verified: ' + preflight['status'])
    require(preflight['before_model_loading'] is True
            and preflight['parity_completed_at'] <= preflight['initialized_at'], 'Preflight/model-loading order changed')
    require(preflight['gpu'] == 'NVIDIA L40S' and preflight['gpu_memory_bytes'] >= 44 * 2**30
            and preflight['versions'] == VERSIONS and preflight['cuda'] == '12.4'
            and preflight['executable'] == '/orcd/pool/003/kirilb/Steganography/envs/uni_author_torch240/bin/python',
            'L40S/legacy runtime identity changed')
    parity(preflight['cuda_toy_parity'])
    restored = preflight['checkpoint']
    restored_update, restored_elapsed = restored['marker']['completed_update'], restored['marker']['elapsed']
    require(preflight['source_checkpoint'] == restored and preflight['fresh_baseline_initialization'] is False
            and restored['marker']['config_digest'] == CONFIG and ORIGIN <= restored_update <= update
            and preflight['completed_update'] == restored_update
            and preflight['restored_elapsed_seconds'] == restored_elapsed
            and ORIGIN_ELAPSED <= restored_elapsed <= checkpoint['elapsed']
            and restored['path'] == run + f'/checkpoints/step_{restored_update:06d}'
            and preflight['configured_active_budget_seconds'] == 27900, 'Restoration/elapsed provenance mismatch')
    # A restored non-evaluation checkpoint may subsequently be pruned. The hash-bound
    # runtime event embeds its marker; verify raw bytes too whenever retained.
    restored_name = f'checkpoints/step_{restored_update:06d}/COMPLETE.json'
    restored_text = files.get(restored_name)
    if restored_text is None and (results / restored_name).exists():
        restored_text = (results / restored_name).read_text()
    if restored_text is not None:
        require(sha(restored_text) == restored['complete_sha256']
                and json.loads(restored_text) == restored['marker'], 'Restored checkpoint marker changed')
    smoke = get('smoke.json')
    require(smoke['config_digest'] == CONFIG and smoke['status'] == 'resumed'
            and smoke['completed_update'] == restored_update and smoke['initialization_verified'] is True
            and smoke['gpu'] == preflight['gpu'] and smoke['torch'] == VERSIONS['torch']
            and smoke['cuda'] == preflight['cuda'], 'Smoke/restoration identity mismatch')
    extension_name, extension = latest('authorized_budget_extension_restored', 'activated_at')
    event_pins(extension, extension=True)
    require(restart(extension) == restart(preflight) and extension['extension_id'] == manifest['extension_id']
            and extension['completed_update'] == restored_update
            and extension['restored_elapsed_seconds'] == restored_elapsed
            and extension['checkpoint'] == restored['path'] and extension['target_active_seconds'] == TARGET
            and extension['remaining_active_seconds_at_restored_checkpoint'] == max(0, TARGET - restored_elapsed),
            'Extension activation reset progress/budget or differs from restored checkpoint')
    memory_name, memory = latest('L40S_PPO_saved_activation_offload', 'enabled_at')
    event_pins(memory)
    require(memory['status'] == 'memory_offload_enabled' and restart(memory) == restart(preflight)
            and memory['preflight_event'] == preflight_name
            and memory['preflight_event_sha256'] == sha(files[preflight_name])
            and memory['memory_manifest_sha256'] == sha(files['L40S_MEMORY_OFFLOAD.json'])
            and memory['completed_update_at_enable'] == restored_update
            and memory['restored_elapsed_seconds'] == restored_elapsed
            and memory['checkpoint'] == restored['path'], 'Offload event binding mismatch')
    parity(memory['cuda_toy_parity'])
    observation = memory.get('last_ppo_memory')
    committed_memory, memory_source = None, None
    if observation:
        finite(observation, 'memory')
        require(observation['ppo_call_status'] == 'completed' and observation['error'] is None,
                'Latest PPO attempt failed')
        if observation['attempted_update'] <= update:
            require(phase(observation['attempted_update']) == 'sender', 'PPO event is not a sender update')
            committed_memory, memory_source = observation, memory_name
    if committed_memory is None:
        # Historical completed samples are observations, not an invented current
        # peak; retain their original update and diagnostic provenance.
        for path in sorted((results / 'diagnostics').glob('monitor_update_*.json'), reverse=True):
            prior = read(path)
            sample = prior.get('last_committed_ppo_memory')
            if (prior.get('job_id') == str(launch['job_id']) and sample
                    and sample['attempted_update'] <= update and sample.get('ppo_call_status') == 'completed'
                    and sample.get('error') is None and prior.get('state_sha256_verified_remotely') is True):
                committed_memory, memory_source = sample, str(path)
                break

    completed_evaluations = []
    for name in files:
        match = re.fullmatch(r'evaluations/step_(\d+)/COMPLETE.json', name)
        if not match or int(match[1]) > update:
            continue
        step = int(match[1]); directory = name.rsplit('/', 1)[0]
        completion = get(name); summary_name = directory + '/SUMMARY.json'
        checkpoint_file = f'checkpoints/step_{step:06d}/COMPLETE.json'
        marker_text = files.get(checkpoint_file)
        if marker_text is None:
            marker_text = (results / checkpoint_file).read_text()
        identity_expected = {'path': run + f'/checkpoints/step_{step:06d}',
                             'completed_update': step, 'complete_sha256': sha(marker_text)}
        require(completion['status'] == 'complete' and completion['config_digest'] == CONFIG
                and completion['completed_update'] == step and completion['source_checkpoint'] == identity_expected
                and completion['summary_sha256'] == sha(files[summary_name])
                and completion['evaluation_config_digest'] == digest(get(directory + '/config.json')),
                f'Evaluation{step} completion/hash binding mismatch')
        summary = get(summary_name)
        require(summary['config_digest'] == CONFIG and summary['completed_update'] == step
                and summary['source_checkpoint'] == identity_expected, f'Evaluation{step} summary identity mismatch')
        completed_evaluations.append(step)
    # Only the latest attempt for each evaluation/current job is authoritative.
    cleanups = {}
    for name, event in current.items():
        if event.get('kind') == 'L40S_evaluation_parent_cache_cleanup':
            step = event['checkpoint']['completed_update']
            if step not in cleanups or event['started_at'] > cleanups[step][1]['started_at']:
                cleanups[step] = (name, event)
    cleanup_evidence, pending = [], []
    for step, (name, event) in sorted(cleanups.items()):
        if step > update:
            pending.append({'update': step, 'event': name, 'outside_verified_boundary': True})
            continue
        event_pins(event)
        require(event['manifest_sha256'] == sha(files['L40S_EVALUATION_MEMORY.json']), 'Cleanup manifest binding mismatch')
        state = event['status']
        if state in ('failed', 'preempted') and event['started_at'] < preflight['started_at']:
            continue  # Retained old-attempt history; do not erase or treat as current.
        require(state in ('original_judge_completed', 'cleaning_parent_cache', 'launching_original_judge'),
                'Current evaluation cleanup failed: ' + state)
        verification = event.get('cleanup_verification')
        if verification:
            require(verification['live_tensor_metadata_unchanged'] is True and verification['rng_unchanged'] is True
                    and verification['before']['cuda_allocated_bytes'] == verification['after']['cuda_allocated_bytes'],
                    f'Evaluation{step} cleanup changed live tensors/RNG')
        if state != 'original_judge_completed':
            pending.append({'update': step, 'event': name, 'status': state})
            continue
        require(verification is not None, 'Completed judge lacks cleanup invariants')
        require(event['started_at'] <= event['cache_cleaned_at'] <= event['finished_at'], 'Cleanup chronology invalid')
        if step in completed_evaluations:
            completion = get(f'evaluations/step_{step:06d}/COMPLETE.json')
            require(event['checkpoint'] == completion['source_checkpoint']
                    and event['evaluation_config_digest'] == completion['evaluation_config_digest']
                    and event['finished_at'] <= completion['completed_at'], 'Judge/evaluation completion binding mismatch')
        else:
            pending.append({'update': step, 'event': name, 'status': 'judge_finished_completion_pending'})
        cleanup_evidence.append({'update': step, 'event': name, 'sha256': sha(files[name]),
                                 'before': verification['before'], 'after': verification['after']})
    reported = set(launch.get('reported_evaluation_updates', []))
    newly_complete = sorted(set(completed_evaluations) - reported)
    previous_boundary = ORIGIN
    for path in (results / 'diagnostics').glob('monitor_update_*.json'):
        prior = read(path)
        boundary = prior.get('verified_update', 0)
        if prior.get('job_id') == str(launch['job_id']) and previous_boundary < boundary < update:
            previous_boundary = boundary
    new_rows = rows[previous_boundary:]
    report = read(results / 'REPORT.json') if (results / 'REPORT.json').exists() else {}
    audit_result = {
        'checked_at': datetime.now(timezone.utc).isoformat(), 'routine_audit_passed': True,
        'verified_update': update, 'job_id': str(launch['job_id']), 'node': slurm['NodeList'],
        'slurm_state': slurm['JobState'], 'slurm_restart_count': int(slurm['Restarts']),
        'slurm_runtime': slurm['RunTime'], 'slurm_time_limit': slurm['TimeLimit'],
        'exclusive_matching_slurm_job_in_supplied_queue': True,
        'exclusivity_scope': 'Only visible matching-name jobs; no claim about differently named jobs.',
        'state_sha256_verified_remotely': True, 'checkpoint': checkpoint,
        'target_active_seconds': TARGET, 'added_active_seconds_at_checkpoint': checkpoint['elapsed'] - ORIGIN_ELAPSED,
        'remaining_active_seconds_at_checkpoint': max(0, TARGET - checkpoint['elapsed']),
        'failure_json_present': False, 'terminal_markers_present': [], 'identity_checks': identity,
        'inherited_runtime_manifests_unchanged': True,
        'numeric_metrics_finite': True, 'metrics_sequential_schedule_verified': True,
        'new_metrics_window': [previous_boundary + 1, update] if new_rows else [],
        'timing_counts_verified_through_update': timing['completed_update'] if timing['completed_update'] <= update else None,
        'timing_observation_after_verified_boundary': timing['completed_update'] > update,
        'receiver_256_four_microbatches_one_step_verified': True, 'sender_population_192_verified': True,
        'sender_reward_variation_verified': all(row['reward_std'] > 0 for row in new_rows if row['kind'] == 'sender'),
        'single_label_prediction_updates_since_previous_check': [row['update'] for row in new_rows
                                                               if row['receiver_pass'] in (0, row['n'])],
        'restored_update': restored_update, 'restored_elapsed_seconds': restored_elapsed,
        'restored_checkpoint_raw_marker_available': restored_text is not None,
        'runtime_preflight_event': preflight_name, 'runtime_preflight_event_sha256': sha(files[preflight_name]),
        'extension_event': extension_name, 'extension_event_sha256': sha(files[extension_name]),
        'legacy_l40s_preflight_parity_and_rng_verified': True,
        'memory_event_attempted_update': observation['attempted_update'] if observation else None,
        'memory_event_after_verified_boundary': bool(observation and observation['attempted_update'] > update),
        'last_committed_ppo_memory': committed_memory, 'last_committed_ppo_memory_source': memory_source,
        'completed_evaluation_bindings_verified': sorted(completed_evaluations),
        'new_evaluations': newly_complete, 'evaluation_public_rows_require_separate_audit': newly_complete,
        'evaluation_cleanup_invariants_passed': True, 'evaluation_cleanup_events': cleanup_evidence,
        'pending_evaluation_cleanup': pending,
        'status_observed_completed_update': status['completed_update'], 'current_stage': status['stage'],
        'report_completed_update': report.get('completed_update'),
        'report_matches_verified_update': report.get('completed_update') == update,
        'report_sha256': sha((results / 'report.md').read_text()) if (results / 'report.md').exists() else None,
        'inputs': {'snapshot': str(args.snapshot), 'identity': str(args.identity),
                   'identity_sha256': sha(identity_text), 'launch': str(args.launch)},
    }
    return results / 'diagnostics' / f'monitor_update_{update:06d}.json', audit_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, default=Path('/tmp/uni_corrected_extension_snapshot.json'))
    parser.add_argument('--identity', type=Path, default=Path('/tmp/uni_corrected_extension_identity.txt'))
    parser.add_argument('--launch', type=Path, default=WORKSPACE / 'launches/university_corrected_extension_latest.json')
    args = parser.parse_args()
    try:
        path, result = audit(args)
        if path.exists():
            previous = read(path)
            if previous.get('verified_update') == result['verified_update']:
                # Caller-owned independent row validation is deliberately separate.
                if 'independent_public_rollout_audit' in previous:
                    result['independent_public_rollout_audit'] = previous['independent_public_rollout_audit']
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        temporary.replace(path)
        sample = result['last_committed_ppo_memory']
        print(json.dumps({'ok': True, 'verified_update': result['verified_update'],
                          'job_id': result['job_id'], 'stage': result['current_stage'],
                          'remaining_active_seconds': result['remaining_active_seconds_at_checkpoint'],
                          'committed_memory_update': sample['attempted_update'] if sample else None,
                          'memory_observation_ahead': result['memory_event_after_verified_boundary'],
                          'new_evaluations': result['new_evaluations'],
                          'pending_evaluation_cleanup': result['pending_evaluation_cleanup'],
                          'report_matches_verified_update': result['report_matches_verified_update'],
                          'diagnostic': str(path)}, sort_keys=True))
        return 0
    except (AuditError, KeyError, TypeError, OSError, ValueError) as exc:
        print(json.dumps({'ok': False, 'error': str(exc),
                          'action': 'Use the main agent exception/requeue/completion workflow; no milestones changed.'}))
        return 1


if __name__ == '__main__':
    sys.exit(main())
