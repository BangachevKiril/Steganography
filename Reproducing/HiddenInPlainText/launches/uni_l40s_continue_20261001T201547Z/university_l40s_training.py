#!/usr/bin/env python3
"""Resume the frozen university run on an explicitly authorized L40S GPU.

Only the two hardware guards in Trainer.__init__ are replaced in memory. The
original source files, configuration digest, checkpoint format and training
methods remain unchanged. Original main() owns the run lock and signal handling.
This launcher refuses step-zero initialization or a missing resume checkpoint.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time


ORIGINAL_GUARD = """    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
        raise RuntimeError('This historical runtime requires the configured A100 80GB allocation')
    if 'A100' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Historical bitsandbytes observer requires A100')
"""

L40S_GUARD = """    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 44 * 1024**3:
        raise RuntimeError('Authorized L40S continuation requires at least 44 GiB GPU memory')
    if 'L40S' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Authorized continuation requires a genuine L40S GPU')
"""


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def transform_init(source):
    """Fail closed unless precisely the audited, adjacent guards are present."""
    original = textwrap.dedent(source)
    tree = ast.parse(original)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != '__init__' or original.count(ORIGINAL_GUARD) != 1):
        raise ValueError('Frozen Trainer.__init__ does not contain the exact audited hardware guard')
    patched = original.replace(ORIGINAL_GUARD, L40S_GUARD, 1)
    # Replacement must be reversible and may not add/remove any statements.
    if patched.replace(L40S_GUARD, ORIGINAL_GUARD, 1) != original:
        raise ValueError('Unexpected changes outside the hardware guard')
    if len(ast.parse(patched).body[0].body) != len(tree.body[0].body):
        raise ValueError('Hardware replacement changed initialization structure')
    return original, patched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-source', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    source, config_path, run = (getattr(args, name).resolve()
                                for name in ('training_source', 'config', 'run_dir'))
    if not run.is_dir() or not (run / 'config.json').is_file():
        raise ValueError('Continuation requires an existing configured training run')
    config = read(config_path)
    if read(run / 'config.json') != config:
        raise ValueError('Continuation configuration differs from the existing run')
    expected_digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_hashes = config.get('source_sha256')
    if not source_hashes or 'stego/university_training.py' not in source_hashes:
        raise ValueError('Missing frozen training-source hash manifest')
    verified_hashes = {}
    for relative, expected in source_hashes.items():
        path = source / relative
        actual = sha256(path)
        if actual != expected:
            raise ValueError('Frozen source hash mismatch: ' + relative)
        verified_hashes[relative] = actual

    sys.dont_write_bytecode = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(source))
    os.chdir(source)
    from stego import university_training as training
    if Path(training.__file__).resolve() != source / 'stego' / 'university_training.py':
        raise ValueError('Imported training module is not from the verified source')
    if training.digest(config) != expected_digest:
        raise ValueError('Original configuration digest implementation differs')

    import torch
    import bitsandbytes
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('L40S continuation requires exactly one visible CUDA GPU')
    props = torch.cuda.get_device_properties(0)
    if 'L40S' not in props.name or props.total_memory < 44 * 1024**3:
        raise RuntimeError('L40S continuation requires a genuine L40S with at least44 GiB')
    versions = {name: importlib.metadata.version(name) for name in
               ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'bitsandbytes', 'numpy', 'datasets', 'tokenizers')}
    for name, expected in {'torch': '2.4.0', 'transformers': '4.43.2', 'trl': '0.9.6', 'bitsandbytes': '0.43.3'}.items():
        if versions[name].split('+')[0] != expected:
            raise RuntimeError('Original runtime version required: ' + name + '==' + expected)

    original_init, patched_init = transform_init(inspect.getsource(training.Trainer.__init__))
    compiled = {}
    exec(compile(patched_init, str(Path(__file__).resolve()) + ':authorized_hardware_guard', 'exec'),
         training.__dict__, compiled)
    hardware_init = compiled['__init__']
    original_latest = training.latest_checkpoint
    observed_resume = {}

    def required_latest(candidate_run, config_digest, verify_hash=True):
        if Path(candidate_run).resolve() != run or config_digest != expected_digest:
            raise ValueError('Resume request targets another run or configuration')
        # Always verify the state hash, even if a caller asks for less checking.
        path = original_latest(candidate_run, config_digest, verify_hash=True)
        if path is None:
            raise RuntimeError('No valid committed checkpoint; refusing fresh initialization')
        marker = read(path / 'COMPLETE.json')
        if marker['completed_update'] < 1:
            raise RuntimeError('Only step zero exists; refusing fresh initialization')
        observed_resume.clear()
        observed_resume.update(path=str(path), marker=marker)
        return path

    # This also closes the original init's else:step-zero branch if a checkpoint
    # unexpectedly disappears between the locked preflight and actual restore.
    training.latest_checkpoint = required_latest
    wrapper_sha = sha256(__file__)
    migration = {
        'schema_version': 1, 'kind': 'authorized_L40S_hardware_migration',
        'training_source': str(source), 'training_run': str(run),
        'config_digest': expected_digest, 'source_sha256': verified_hashes,
        'wrapper_sha256': wrapper_sha,
        'original_init_sha256': hashlib.sha256(original_init.encode()).hexdigest(),
        'patched_init_sha256': hashlib.sha256(patched_init.encode()).hexdigest(),
        'original_hardware_guard': ORIGINAL_GUARD, 'replacement_hardware_guard': L40S_GUARD,
        'hardware_waiver': 'User accepted the measured L40S observer-score differences and authorized continuation on L40S.',
        'scientific_configuration_changed': False,
        'observer_numerics': 'Original bitsandbytes0.43.3, unchanged64-row right-padding/last-column scorer; L40S hardware may change scores.',
        'resume_policy': 'Highest hash-verified committed positive-update checkpoint of this same run; retain saved optimizer/RNG/stream state and elapsed budget.',
    }

    def resumed_init(self, init_config, init_run):
        # Called only by original main after it acquires the original run LOCK.
        if init_config != config or Path(init_run).resolve() != run:
            raise ValueError('Original main supplied a different run configuration')
        chosen = required_latest(init_run, expected_digest)
        preflight = dict(observed_resume)
        manifest_path = run / 'L40S_MIGRATION.json'
        if manifest_path.exists() and read(manifest_path) != migration:
            raise ValueError('Saved L40S migration manifest differs from this launcher')
        training.atomic_json(manifest_path, migration)
        events = run / 'runtime_events'
        events.mkdir(exist_ok=True)
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        safe_job_id = ''.join(char for char in job_id if char.isalnum() or char in '-_')
        event_path = events / f'l40s_{safe_job_id}_{time.time_ns()}.json'
        event = {
            'kind': 'L40S_continuation', 'status': 'restoring', 'started_at': time.time(),
            'config_digest': expected_digest, 'migration_manifest_sha256': sha256(manifest_path),
            'wrapper_sha256': wrapper_sha, 'source_checkpoint': str(chosen),
            'checkpoint_complete': preflight['marker'],
            'slurm_job_id': job_id, 'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
            'gpu': props.name, 'gpu_memory_bytes': props.total_memory,
            'gpu_capability': list(torch.cuda.get_device_capability(0)),
            'visible_gpu_count': torch.cuda.device_count(), 'cuda': torch.version.cuda,
            'python': sys.version, 'executable': sys.executable, 'versions': versions,
            'bitsandbytes_module': bitsandbytes.__file__,
            'original_init_sha256': migration['original_init_sha256'],
            'patched_init_sha256': migration['patched_init_sha256'],
        }
        training.atomic_json(event_path, event)
        try:
            hardware_init(self, init_config, init_run)
            # Check restoration, including the budget, before original main may
            # call train(). No training settings or state are changed here.
            actual = observed_resume['marker']
            if (str(self.current_checkpoint) != observed_resume['path']
                    or self.update != actual['completed_update'] or self.elapsed != actual['elapsed']
                    or self.config_digest != expected_digest):
                raise RuntimeError('Restored checkpoint/update/elapsed budget verification failed')
            if observed_resume['path'] != preflight['path']:
                raise RuntimeError('Checkpoint changed while the original run lock was held')
            event.update(status='restored', restored_at=time.time(),
                completed_update=self.update, restored_elapsed_seconds=self.elapsed,
                configured_active_budget_seconds=self.config['max_active_seconds'],
                latest_state_sha256=actual['state_sha256'])
            training.atomic_json(event_path, event)
        except BaseException as exc:
            event.update(status='failed', failed_at=time.time(), error_type=type(exc).__name__, error=str(exc))
            training.atomic_json(event_path, event)
            raise

    training.Trainer.__init__ = resumed_init
    # Keep original main, relative source-hash checks, LOCK, SIGUSR1/SIGTERM,
    # PREEMPTED/FAILURE handling and checkpoint/evaluation scheduling intact.
    sys.argv = [str(source / 'stego' / 'university_training.py'),
                '--config', str(config_path), '--run-dir', str(run)]
    training.main()


if __name__ == '__main__':
    main()
