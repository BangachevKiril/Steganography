"""Read-only public snapshot; hash the newest checkpoint in place, never export it.

Run on the host containing the run directory. stdout is the snapshot JSON; no
files are written and no subprocesses/network calls are made. Completion-marker
copies include checkpoints whose state.pt was deliberately pruned.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT_PUBLIC = (
    'config.json', 'STATUS.json', 'FAILURE.json', 'smoke.json', 'TRAINING_SUMMARY.json',
    'INITIALIZATION.json', 'TIMING.json', 'COMPLETE.json', 'LATEST.json',
    'PROMPTS.json', 'PROTOCOL.json', 'QUERY_TOKENIZATION.json', 'TRAIN_SPLIT.json',
    'PPO_CONFIG.json', 'FIRST_RECEIVER_UPDATE.json', 'FIRST_SENDER_UPDATE.json',
    'BUDGET_COMPLETE.json', 'PREEMPTED.json', 'QWEN_H100_RUNTIME.json',
    'H100_MEMORY_OFFLOAD.json', 'H100_EVALUATION_MEMORY.json', 'QWEN_OBSERVER.json',
    'SEGMENT_ACTIVE_CLOCK.json', 'FULL_CYCLE_SMOKE.json', 'RUNTIME.json', 'PPO_MEMORY.json', 'RECEIVER_MEMORY.json', 'RECEIVER_UPDATE_AUDIT.json', 'CYCLE_ACTIVE_CLOCK.json', 'CHECKPOINT_POLICY.json', 'LAST_ROLLBACK.json', 'INCOMPLETE_CHECKPOINT_CLEANUP.json',
) + tuple(f'SEGMENT_{n:03d}_{kind}.json' for n in (1, 2, 3)
          for kind in ('BUDGET_COMPLETE', 'COMPLETE'))
EVALUATION_PUBLIC = frozenset((
    'config.json', 'SELECTION.json', 'STATUS.json', 'UNIVERSITY_GENERATIONS.json',
    'UNIVERSITY_RECEIVER.json', 'UNIVERSITY_RESULTS.json', 'PIQA_GENERATIONS.json',
    'PIQA_JUDGED.json', 'PIQA_JUDGE_STATUS.json', 'SUMMARY.json', 'COMPLETE.json',
    'report.md',
))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(value):
    return sha256_bytes(json.dumps(value, sort_keys=True).encode())


def state_sha256(path):
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def verify_completed_evaluations(files, run, expected, verified_update):
    """Verify available completion/summary/judge hashes from captured public bytes.

    The evaluator does not store a hash of every university row in COMPLETE;
    this function checks its actual hash bindings and exact cohort identities.
    Snapshot-wide files_sha256 additionally binds every exported public byte.
    """
    verified = []

    def payload(name):
        if name not in files:
            raise ValueError('Completed evaluation is missing public artifact: ' + name)
        return json.loads(files[name])

    def file_sha(name):
        return sha256_bytes(files[name].encode())

    for name in sorted(files):
        match = re.fullmatch(r'evaluations/step_(\d+)/COMPLETE\.json', name)
        if not match:
            continue
        update = int(match.group(1))
        prefix = name.rsplit('/', 1)[0]
        marker = payload(name)
        checkpoint_name = f'evaluation_states/step_{update:06d}/COMPLETE.json'
        checkpoint = payload(checkpoint_name)
        identity = dict(path=str(run / 'evaluation_states' / f'step_{update:06d}'),
                        completed_update=update, complete_sha256=file_sha(checkpoint_name))
        config = payload(prefix + '/config.json')
        evaluation_digest = digest(config)
        summary_name = prefix + '/SUMMARY.json'
        summary = payload(summary_name)
        if (checkpoint.get('kind') != 'in_memory_evaluation_weights'
                or checkpoint.get('resumable') is not False
                or any(not re.fullmatch(r'[0-9a-f]{64}', checkpoint.get('model_identity',{}).get(k,'')) for k in ('sender_sha256','receiver_sha256'))
                or checkpoint.get('config_digest') != expected
                or checkpoint.get('completed_update') != update
                or config.get('training_config_digest') != expected
                or config.get('completed_update') != update
                or config.get('source_checkpoint') != identity
                or marker.get('status') != 'complete'
                or marker.get('completed_update') != update
                or marker.get('parameter_updates') != update
                or marker.get('config_digest') != expected
                or marker.get('evaluation_config_digest') != evaluation_digest
                or marker.get('source_checkpoint') != identity
                or marker.get('summary_sha256') != file_sha(summary_name)
                or summary.get('config_digest') != expected
                or summary.get('completed_update') != update
                or summary.get('parameter_updates') != update
                or summary.get('evaluation_config_digest') != evaluation_digest
                or summary.get('source_checkpoint') != identity
                or summary.get('university', {}).get('n') != 2000
                or summary.get('piqa_enabled') is not False):
            raise ValueError(f'Evaluation {update} completion/summary identity mismatch')
        populations = {}
        for filename, count, id_key in (
                ('UNIVERSITY_GENERATIONS.json', 2000, 'example_id'),
                ('UNIVERSITY_RECEIVER.json', 2000, 'example_id'),
                ('UNIVERSITY_RESULTS.json', 2000, 'example_id')):
            value = payload(prefix + '/' + filename)
            rows = value.get('rows', [])
            ids = [row[id_key] for row in rows]
            if (value.get('config_digest') != evaluation_digest or len(rows) != count
                    or len(set(ids)) != count
                    or (id_key == 'example_id' and ids != list(range(2000)))):
                raise ValueError(f'Evaluation {update} invalid cohort: {filename}')
            populations[filename] = ids
        verified.append(dict(completed_update=update,
                             recoverable=verified_update is not None and update<=verified_update,
                             identity_kind='in_memory_evaluation_weights',
                             complete_sha256=file_sha(name),
                             summary_sha256=file_sha(summary_name),
                             source_checkpoint=identity))
    return verified


def snapshot(run, expected):
    run = Path(run).resolve()
    files = {}

    def capture(path):
        # Only caller-selected public paths are accepted, never redirected links.
        if path.resolve() != path.absolute():
            raise ValueError('Public artifact must not be a symbolic link: ' + str(path))
        relative = str(path.relative_to(run))
        value = path.read_bytes().decode('utf-8')
        files[relative] = value
        return value

    config = json.loads(capture(run / 'config.json'))
    if digest(config) != expected:
        raise ValueError('Run config mismatch')
    checkpoints = []
    for path in sorted((run / 'checkpoints').glob('step_*/COMPLETE.json')):
        match = re.fullmatch(r'step_(\d+)', path.parent.name)
        if not match:
            continue
        marker = json.loads(capture(path))
        update = int(match.group(1))
        if (marker.get('config_digest') != expected
                or type(marker.get('completed_update')) is not int
                or marker['completed_update'] != update
                or not re.fullmatch(r'[0-9a-f]{64}', marker.get('state_sha256', ''))):
            raise ValueError('Checkpoint config/identity mismatch: ' + str(path))
        state = path.parent / 'state.pt'
        if state.is_file():
            if state.is_symlink():
                raise ValueError('Checkpoint state must not be a symbolic link')
            checkpoints.append((update, path, marker))
    step = checkpoint = state_hash = None
    if checkpoints:
        step, path, marker = max(checkpoints, key=lambda item: item[0])
        state_hash = state_sha256(path.parent / 'state.pt')
        if state_hash != marker['state_sha256']:
            raise ValueError('Committed state hash mismatch')
        checkpoint = str(path.parent)
    for path in sorted((run/'evaluation_states').glob('step_*/COMPLETE.json')):
        capture(path)
    for name in ROOT_PUBLIC:
        path = run / name
        if name != 'config.json' and path.is_file():
            capture(path)
    for pattern in ('h100_*.json', 'qwen_*.json'):
        for path in sorted((run / 'runtime_events').glob(pattern)):
            if path.is_file():
                capture(path)
    if step is not None:
        path = run / 'metrics.jsonl'
        rows = [json.loads(line) for line in capture(path).splitlines() if line.strip()] if path.exists() else []
        files['observed_metrics.jsonl'] = ''.join(json.dumps(row,allow_nan=False)+'\n' for row in rows)
        rows = [row for row in rows if row['update'] <= step]
        if [row['update'] for row in rows] != list(range(1, step + 1)):
            raise ValueError('Metrics incomplete')
        if any(row.get('config_digest') != expected for row in rows):
            raise ValueError('Metrics config mismatch')
        files['metrics.jsonl'] = ''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows)
        for path in sorted((run / 'rollouts').glob('update_*.json')):
            match = re.fullmatch(r'update_(\d+)\.json', path.name)
            if match and int(match.group(1)) <= step:
                capture(path)
        for directory in sorted((run / 'evaluations').glob('step_*')):
            match = re.fullmatch(r'step_(\d+)', directory.name)
            if not match:
                continue
            for name in sorted(EVALUATION_PUBLIC):
                path = directory / name
                if path.is_file():
                    capture(path)
    verified_evaluations = verify_completed_evaluations(files, run, expected, step)
    return dict(verified_update=step, checkpoint=checkpoint,
                state_sha256_verified=step is not None, state_sha256=state_hash,
                files=files,
                files_sha256={name: sha256_bytes(value.encode()) for name, value in files.items()},
                verified_evaluations=verified_evaluations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('config_digest')
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run, args.config_digest), allow_nan=False))


if __name__ == '__main__':
    main()
