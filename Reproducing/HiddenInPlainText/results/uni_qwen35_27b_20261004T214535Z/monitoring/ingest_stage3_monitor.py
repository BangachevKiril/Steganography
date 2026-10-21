"""Local-only ingestion of ordinary segment3 progress after the audited catch-up.
Stop before writes for new evaluations, allocations, failures or completion.

Usage: python3 SCRIPT SNAPSHOT_JSON SLURM_TEXT
Does not connect remotely, touch frozen source, or mark new milestones reported.
"""
import hashlib
import json
import math
import re
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

latest = Path('/Users/kirilbangachev/Documents/ChatGPT/Steganography/launches/university_qwen_latest.json')
original = latest.read_bytes()
d = json.loads(original)
src, out = Path(d['local_source']), Path(d['local_results'])
snapshot_path, slurm_path = map(Path, sys.argv[1:])
s = json.loads(snapshot_path.read_text())
f, hashes, update = s['files'], s['files_sha256'], s['verified_update']
sha = lambda b: hashlib.sha256(b).hexdigest()
assert s['state_sha256_verified'] and update >= d['verified_update']
assert set(f) == set(hashes)
for name, body in f.items():
    p = Path(name)
    assert not p.is_absolute() and '..' not in p.parts and p.suffix in ('.json', '.jsonl', '.md')
    assert sha(body.encode()) == hashes[name], name
config = json.loads((src / 'configs/university_training.json').read_text())
assert config == json.loads(f['config.json'])
assert sha(json.dumps(config, sort_keys=True).encode()) == d['config_digest']
assert len(config['source_sha256']) == 41
for name, digest in config['source_sha256'].items():
    assert sha((src / name).read_bytes()) == digest, name
assert json.loads(f['QWEN_H100_RUNTIME.json'])['source_sha256'] == config['source_sha256']
verifier = runpy.run_path(str(src / 'orcd/monitoring/snapshot_qwen_university.py'))
assert verifier['verify_completed_evaluations'](f, Path(d['run_dir']), d['config_digest'], update) == s['verified_evaluations']
assert s['verified_evaluations'] == d['verified_evaluations'], 'New evaluation requires audit'
assert hashes['FAILURE.json'] == d['last_failure_artifact_sha256'], 'Changed failure requires investigation'
assert hashes['FULL_CYCLE_SMOKE.json'] == d['full_cycle_evidence']['sha256']
assert sorted(n for n in f if re.fullmatch(r'SEGMENT_\d+_COMPLETE.json', n)) == ['SEGMENT_001_COMPLETE.json', 'SEGMENT_002_COMPLETE.json'], 'New or missing segment completion requires audit'
assert not any(n in f for n in ('COMPLETE.json', 'BUDGET_COMPLETE.json', 'SEGMENT_003_BUDGET_COMPLETE.json')), 'Final budget/completion requires milestone audit'
observer = json.loads(f['QWEN_OBSERVER.json'])
auth = observer['runtime_authorization']
assert auth['recovery_manifest_sha256'] == d['recovery_manifest_sha256'] == sha(Path(d['recovery_manifest']).read_bytes())
assert auth['authorization_sha256'] == sha((src / 'recovery/zero_block_guard_003/AUTHORIZATION.json').read_bytes())
assert auth['runtime_after_sha256'] == '5ea1130b150506e7a7e94534ee4e9aedd6e57ef576bfc50ced6e00927d926612'
runtime_expected = dict(json.loads((src / 'OBSERVER_RUNTIME.json').read_text())['source_sha256'])
runtime_expected[auth['runtime_path']] = auth['runtime_after_sha256']
assert observer['runtime_audit']['runtime_source_sha256_verified'] == runtime_expected
assert auth['cumulative_thresholds'] == d['cumulative_thresholds'] == [27900, 55800, 83700]
audit_path = out / 'monitoring/STAGE3_CATCHUP_RUNTIME_VERIFICATION.json'
audit = json.loads(audit_path.read_text())
assert audit['status'] == 'passed' and audit['config_digest'] == d['config_digest']
expected_allocation = audit['current_allocation']
assert expected_allocation['job_id'] == '24936933' and expected_allocation['restart_count'] == 2 and expected_allocation['node'] == 'node3009'
assert observer['started_at'] == expected_allocation['observer_started_at'] and observer['slurm_job_id'] == '24936933', 'New observer allocation requires audit'
assert d['current_segment'] == 3 and d['current_job_id'] == '24936933', 'Parent must ingest audited catch-up before ordinary monitoring'
assert observer['model_artifact_sha256'] == {x['path']: x['sha256'] for x in json.loads((src / 'MODEL_MANIFEST.json').read_text())['files']}
clock = json.loads(f['SEGMENT_ACTIVE_CLOCK.json'])
assert clock['segment'] == 3 and clock['active_threshold_seconds'] == 83700 and clock['total_active_budget_seconds'] == 83700
checkpoint_marker = json.loads(f[f'checkpoints/step_{update:06d}/COMPLETE.json'])
assert checkpoint_marker['completed_update'] == update and checkpoint_marker['state_sha256'] == s['state_sha256']
assert checkpoint_marker['config_digest'] == d['config_digest']
assert clock['checkpoint']['completed_update'] >= update
if clock['checkpoint']['completed_update'] == update:
    assert clock['checkpoint']['state_sha256'] == s['state_sha256']
    assert clock['checkpoint']['complete_sha256'] == hashes[f'checkpoints/step_{update:06d}/COMPLETE.json']
    verified_active_seconds = clock['active_seconds']
else:
    # Live files can advance while state.pt is hashed. Cap timing at the verified marker.
    verified_active_seconds = checkpoint_marker['elapsed']
status = json.loads(f['STATUS.json'])
assert status['status'] == 'running' and status['slurm_job_id'] == '24936933'
timing = json.loads(f['TIMING.json'])
observed_timing_ahead = timing['completed_update'] > update
metrics = [json.loads(x) for x in f['metrics.jsonl'].splitlines() if x.strip()]
metrics = [m for m in metrics if m['update'] <= update]
assert [m['update'] for m in metrics] == list(range(1, update + 1))

def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value), 'Nonfinite public numeric value'
    elif isinstance(value, dict):
        for x in value.values():
            finite(x)
    elif isinstance(value, list):
        for x in value:
            finite(x)

for m in metrics:
    finite(m)
    assert m['config_digest'] == d['config_digest']
    assert m['kind'] == ('receiver' if (m['update'] - 1) % 7 < 2 else 'sender')
    if m['kind'] == 'receiver':
        t = m['receiver_training']
        assert (t['n'], t['optimizer_sample_count'], t['microbatches'], t['optimizer_steps']) == (256, 256, 4, 1)
        assert len(t['minibatch_label_counts']) == 4
        assert all(b['n'] == 64 and b['PASS'] + b['FAIL'] == 64 for b in t['minibatch_label_counts'])
    else:
        assert m['n'] == 192 and m['reward_std'] > 0
    if m['update'] > d['verified_update']:
        rows = json.loads(f[f"rollouts/update_{m['update']:06d}.json"])['rows']
        finite(rows)
        assert len(rows) == m['n']
        assert sum(x['receiver_prediction'] == 1 for x in rows) == m['receiver_pass']
        assert sum(x['receiver_prediction'] == 0 for x in rows) == m['receiver_fail']
        assert sum(x['receiver_prediction'] == x['true_label'] for x in rows) == m['receiver_correct']
        assert sum(x['observer_flag'] for x in rows) == m['observer_flagged']
        assert all(x['observer_flag'] == (x['observer_score'] < 0) for x in rows)

if observed_timing_ahead:
    timing = dict(config_digest=d['config_digest'], completed_update=update, active_seconds=verified_active_seconds,
        counts={kind: sum(m['kind'] == kind for m in metrics) for kind in ('receiver', 'sender')},
        mean_seconds={kind: sum(m['duration_seconds'] for m in metrics if m['kind'] == kind) / sum(m['kind'] == kind for m in metrics) for kind in ('receiver', 'sender')},
        source='Active time from hash-verified checkpoint marker; counts and duration means from committed metrics. Live TIMING.json advanced during snapshot.')

slurm = slurm_path.read_text()
now = datetime.now(timezone.utc).isoformat()
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
completed_jobs = {
    '24936931': ('01:24:30', '2026-10-05T18:09:37', '2026-10-05T19:34:07', 'node4109', '2026-10-05T23:34:07Z'),
    '24936932': ('01:56:23', '2026-10-06T01:34:42', '2026-10-06T03:31:05', 'node2900', '2026-10-06T07:31:05Z'),
}
assert [j['job_id'] for j in d['jobs']] == ['24936931', '24936932', '24936933']
for job in d['jobs']:
    if job['segment'] < 3:
        elapsed, start, end, node, end_utc = completed_jobs[job['job_id']]
        accounting = '|'.join((job['job_id'], 'COMPLETED', '0:0', elapsed, start, end, node))
        assert accounting in slurm, 'Completed segment accounting changed or unavailable'
        job.update(state='COMPLETED', reason='None', exit_code='0:0', runtime=elapsed,
                   actual_ended_at=end_utc, last_checked=now)
        continue
    block = slurm.split('JobId=' + job['job_id'] + ' ')[1].split('\n\n')[0]
    def get(key):
        return re.search(r'(?m)(?:^|\s)' + re.escape(key) + r'=([^\s]*)', block).group(1)
    assert 'QWEN_RUNTIME_RECOVERY_SHA256=' + d['recovery_manifest_sha256'] in block
    assert (get('Restarts'), get('NodeList')) == ('2', 'node3009'), 'New allocation requires audit'
    assert get('StartTime') == expected_allocation['slurm_start_time'], 'New allocation start requires audit'
    assert get('Requeue') == '1' and get('TimeLimit') == '06:00:00'
    resources = 'cpu=8,mem=192G,node=1,billing=8,gres/gpu=1,gres/gpu:h100=1'
    assert get('ReqTRES') == get('AllocTRES') == resources, 'Hardware changed'
    assert get('JobState') == 'RUNNING' and get('ExitCode') == '0:0' and get('Dependency') == '(null)'
    assert get('Reason') == 'None' and get('Partition') in ('mit_normal_gpu', 'mit_preemptable')
    job.update(state=get('JobState'), reason=get('Reason'), exit_code=get('ExitCode'), runtime=get('RunTime'),
               partition=get('Partition'), node=get('NodeList'), restarts=int(get('Restarts')), last_checked=now,
               requested_resources=get('ReqTRES'), allocated_resources=get('AllocTRES'))

memory = max([json.loads(v) for n, v in f.items() if n.startswith('runtime_events/h100_memory_') and 'preflight' not in n], key=lambda m: m.get('enabled_at', 0))
ppo = memory['last_ppo_memory']
assert ppo['error'] is None and ppo['ppo_call_status'] == 'completed', 'New PPO error or incomplete attempt requires review'
finite(ppo)
last = metrics[-1]
meta = {k: v for k, v in s.items() if k != 'files'}
meta.update(checked_at=now, snapshot_sha256=sha(snapshot_path.read_bytes()), public_artifacts_hash_verified=len(f),
            source_files_hash_verified=41, rollout_scalar_audit_updates=[d['verified_update'] + 1, update], rollout_counts_match_metrics=True,
            live_timing_ahead_of_verified_checkpoint=observed_timing_ahead, verified_active_seconds=verified_active_seconds)
snapshot_relative = Path('monitoring') / (stamp + '_snapshot.json')
queue_relative = Path('queue_checks') / (stamp + '_heartbeat.json')
d.update(last_checked=now, last_successful_remote_check=now, access_state='available', access_failure_reported=False,
         last_snapshot=str(snapshot_relative), last_queue_check=str(queue_relative), verified_update=update,
         verified_checkpoint=s['checkpoint'], verified_state_sha256=s['state_sha256'], current_status=status,
         current_stage=status['stage'], verified_evaluations=s['verified_evaluations'], source_hashes_verified=41,
         last_failure_is_historical=True, verified_complete_cycles=update // 7, last_verified_timing=timing)
d['receiver_bias_latest_observation'] = dict(update=last['update'], kind=last['kind'], n=last['n'], PASS=last['receiver_pass'],
    FAIL=last['receiver_fail'], correct=last['receiver_correct'], cohort='receiver collection before filtering/update' if last['kind'] == 'receiver' else 'sender rollout192')
d['last_observed_ppo_memory'] = ppo
d['last_observed_ppo_memory_update_committed'] = ppo['attempted_update'] <= update
if d['last_observed_ppo_memory_update_committed']:
    d['last_ppo_memory'] = ppo
d['current_allocation_memory'] = dict(restart_count=int(memory.get('slurm_restart_count') or 0), event_started_at=memory['enabled_at'],
    resumed_update=memory['completed_update_at_enable'], last_ppo_memory=d['last_ppo_memory'])

for segment, threshold in ((1, 27900), (2, 55800)):
    sn = f'SEGMENT_{segment:03d}_COMPLETE.json'
    bn = f'SEGMENT_{segment:03d}_BUDGET_COMPLETE.json'
    seg, budget = json.loads(f[sn]), json.loads(f[bn])
    expected = audit['segments'][str(segment)]
    assert hashes[sn] == expected['complete_sha256'] and hashes[bn] == expected['budget_sha256'], 'Completed segment marker changed'
    assert seg['segment'] == budget['segment'] == segment
    assert seg['active_threshold_seconds'] == budget['active_threshold_seconds'] == threshold
    assert seg['total_active_budget_seconds'] == budget['total_active_budget_seconds'] == 83700
    assert seg['config_digest'] == budget['config_digest'] == d['config_digest']
    assert seg['completed_update'] == budget['completed_update'] == expected['completed_update']
    assert seg['checkpoint'] == budget['checkpoint']
    assert seg['evaluation_complete_sha256'] == next(e for e in s['verified_evaluations'] if e['completed_update'] == seg['completed_update'])['complete_sha256']
    marker_path = f"checkpoints/step_{seg['completed_update']:06d}/COMPLETE.json"
    assert seg['checkpoint']['complete_sha256'] == hashes[marker_path]
    assert seg['checkpoint']['state_sha256'] == json.loads(f[marker_path])['state_sha256']
    assert seg['active_seconds'] >= budget['active_seconds'] >= threshold
assert memory['slurm_job_id'] == '24936933' and int(memory.get('slurm_restart_count') or 0) == 2
assert memory['enabled_at'] == expected_allocation['offload_enabled_at'], 'New offload startup requires audit'
assert memory['completed_update_at_enable'] == expected_allocation['resumed_update']
pre = json.loads(f[expected_allocation['preflight_event']])
assert hashes[expected_allocation['preflight_event']] == expected_allocation['preflight_sha256']
assert memory['preflight_event_sha256'] == expected_allocation['preflight_sha256']
assert pre['status'] == 'initialization_verified' and pre['completed_update'] == expected_allocation['resumed_update'] and pre['fresh_baseline_initialization'] is False
assert pre['checkpoint'] == expected_allocation['checkpoint']
assert pre['gpu'] == 'NVIDIA H100 80GB HBM3' and pre['gpu_memory_bytes'] >= 75 * 1024**3 and pre['cuda_toy_parity']['status'] == 'passed'
d['current_job_id'] = '24936933'; d['current_segment'] = 3; d['current_allocation_restart_count'] = 2
d['jobs'][2]['actual_started_at'] = expected_allocation['allocation_started_at']
d['jobs'][2]['scheduler_start_estimate'] = expected_allocation['allocation_started_at']

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_bytes(data)
    temp.replace(path)

def encoded(value):
    return (json.dumps(value, indent=2) + '\n').encode()

assert latest.read_bytes() == original, 'Launch record changed concurrently'
for name, body in f.items():
    write(out / name, body.encode())
write(out / snapshot_relative, encoded(meta))
write(src / queue_relative, encoded(dict(checked_at=now, ssh_existing_control_only=True, slurm_output=slurm, jobs=d['jobs'])))
for path in (latest, src / 'LAUNCH.json'):
    write(path, encoded(d))
recent = [{k: m[k] for k in ('update', 'kind', 'n', 'receiver_correct', 'receiver_pass', 'receiver_fail', 'observer_flagged')} for m in metrics[-10:]]
print(json.dumps(dict(verified_update=update, state_sha256=s['state_sha256'], public_hashes_verified=len(f), source_hashes_verified=41,
    active_seconds_at_checkpoint=verified_active_seconds, current_stage=status['stage'], verified_evaluation_updates=[x['completed_update'] for x in s['verified_evaluations']],
    new_failure=False, allocation_unchanged=True, segment1_complete=True, segment2_complete=True, segment3_running=True, last_committed_ppo_memory=d['last_ppo_memory'], recent_metrics=recent), indent=2))
