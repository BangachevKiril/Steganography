"""Preserve trained model parameters and all evaluations; retire completed restart payloads."""
import fcntl
import hashlib
import json
import os
import signal
import sys
from pathlib import Path

from submit import ATTEMPT


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_evaluations(run, cfg_hash):
    final = read(run / 'COMPLETE.json')
    require(final == read(run / 'BUDGET_COMPLETE.json'), 'Completion budgets differ')
    for k, expected in [('config_digest', cfg_hash), ('completed_update', 1400),
                        ('completed_cycles', 200), ('receiver_updates', 400), ('sender_updates', 1000)]:
        require(final[k] == expected, 'Invalid completion: ' + k)
    checkpoint = run / 'checkpoints/step_001400'
    cp = read(checkpoint / 'COMPLETE.json')
    require(final['checkpoint']['path'] == str(checkpoint)
            and final['checkpoint']['complete_sha256'] == sha(checkpoint / 'COMPLETE.json')
            and cp['config_digest'] == cfg_hash and cp['completed_update'] == 1400
            and final['checkpoint']['state_sha256'] == cp['state_sha256'], 'Final checkpoint identity')
    results = {}
    for step in range(0, 1401, 50):
        name = f'step_{step:06d}'
        folder = run / 'evaluations' / name
        mark = read(folder / 'COMPLETE.json')
        ident = run / 'evaluation_states' / name
        evaluated = read(ident / 'COMPLETE.json')
        source = mark['source_checkpoint']
        summary = read(folder / 'SUMMARY.json')
        rows_file = folder / 'UNIVERSITY_RESULTS.json'
        observed = read(rows_file)
        require(mark['config_digest'] == evaluated['config_digest'] == summary['config_digest'] == cfg_hash
                and mark['completed_update'] == evaluated['completed_update'] == summary['completed_update'] == step
                and mark['summary_sha256'] == sha(folder / 'SUMMARY.json')
                and source == summary['source_checkpoint'] and source['completed_update'] == step
                and source['path'] == str(ident) and source['complete_sha256'] == sha(ident / 'COMPLETE.json')
                and observed['config_digest'] == mark['evaluation_config_digest']
                and summary['university']['n'] == 2000, 'Evaluation provenance: ' + name)
        rows = observed['rows']
        require([r['example_id'] for r in rows] == list(range(2000))
                and all(isinstance(r['input'], str) and isinstance(r['answer'], str)
                        and r['true_label'] in (0, 1) and r['receiver_prediction'] in (0, 1)
                        and r['observer_prediction'] in (0, 1)
                        and 0 <= r['receiver_probabilities'][1] <= 1
                        and 0 <= r['observer_good_probability'] <= 1 for r in rows),
                'Missing evaluation transmissions: ' + name)
        results[name] = sha(rows_file)
        if step == 1400:
            require(sha(folder / 'COMPLETE.json') == final['evaluation_complete_sha256']
                    and evaluated['model_identity'] == cp['model_identity'], 'Final evaluated weights')
    return cp, results


def verify_export(payload, copied):
    import torch
    for key in ('sender_trainable', 'receiver_trainable'):
        require(bool(payload[key]) and set(payload[key]) == set(copied[key]), 'Incomplete export: ' + key)
        for name, tensor in payload[key].items():
            check = copied[key][name]
            require(tensor.dtype == check.dtype and torch.equal(tensor, check),
                    'Exported tensor differs: ' + name)
    require(any('v_head' in name for name in copied['sender_trainable']), 'Missing PPO value head')


def finalize(run, config_path):
    import torch
    config = read(config_path)
    require(run == ATTEMPT / 'runs' / config['name'] and run.resolve() == run and read(run / 'config.json') == config,
            'Refusing to finalize outside the matching fresh run')
    cfg_hash = digest(config)
    with (run / 'LOCK').open('r+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cp, transmissions = validate_evaluations(run, cfg_hash)
        export = run / 'final_weights'
        export.mkdir(exist_ok=True)
        weights = export / 'weights.pt'
        manifest = export / 'COMPLETE.json'
        metadata = dict(config_digest=cfg_hash, completed_update=1400, seed=config['seed'],
                        source_state_sha256=cp['state_sha256'], model_identity=cp['model_identity'],
                        sender_base=config['assets']['sender'], receiver_base=config['assets']['receiver'],
                        resumable=False)
        if not manifest.exists():
            state = run / 'checkpoints/step_001400/state.pt'
            require(sha(state) == cp['state_sha256'], 'Final restart hash mismatch')
            payload = torch.load(str(state), map_location='cpu', mmap=True, weights_only=False)
            require(payload['completed_update'] == 1400 and payload['config_digest'] == cfg_hash,
                    'Final restart metadata mismatch')
            tmp = weights.with_suffix('.tmp')
            torch.save({**metadata, 'sender_trainable': payload['sender_trainable'],
                        'receiver_trainable': payload['receiver_trainable']}, tmp)
            with tmp.open('rb') as stream:
                os.fsync(stream.fileno())
            restored = torch.load(str(tmp), map_location='cpu', mmap=True, weights_only=False)
            verify_export(payload, restored)
            del restored, payload
            tmp.replace(weights)
            tmp_manifest = manifest.with_suffix('.tmp')
            tmp_manifest.write_text(json.dumps({**metadata, 'weights_sha256': sha(weights),
                                                'evaluation_results_sha256': transmissions}, indent=2) + '\n')
            with tmp_manifest.open('rb') as stream:
                os.fsync(stream.fileno())
            tmp_manifest.replace(manifest)
        saved = read(manifest)
        require(all(saved[k] == v for k, v in metadata.items())
                and saved['weights_sha256'] == sha(weights)
                and saved['evaluation_results_sha256'] == transmissions,
                'Refusing retirement without verified weights and results')
        # Final restart last: a killed or retried compact operation remains recoverable.
        for state in sorted((run / 'checkpoints').glob('step_*/state.pt')):
            marker = read(state.parent / 'COMPLETE.json')
            step = marker['completed_update']
            require(state.resolve() == state and state.stat().st_nlink == 1
                    and state.parent.name == f'step_{step:06d}' and 0 <= step <= 1400
                    and marker['config_digest'] == cfg_hash, 'Unexpected restart file')
            state.unlink()


def interrupted(signum, frame):
    # Retry at the next allocation. Temporary exports are replaceable, and
    # retirement starts only after the durable round-tripped export exists.
    raise SystemExit(75)


if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    finalize(Path(sys.argv[1]), Path(sys.argv[2]))
