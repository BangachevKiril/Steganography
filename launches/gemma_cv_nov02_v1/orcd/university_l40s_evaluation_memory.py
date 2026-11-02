#!/usr/bin/env python3
"""Release only unused parent CUDA cache before the unchanged PIQA judge child.

The pinned memory/hardware launchers still own model initialization, restoration,
locking, training, evaluation scheduling and preemption. No live tensor moves.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time


MEMORY_LAUNCHER_SHA256 = '9ce57ee0ff9a043273f80a75424a80bdf8a97189e0ccaedb00dce2f0f9e9b2c4'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def host_rss_bytes():
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def live_state_metadata(trainer):
    """Describe reachable model/optimizer tensors without copying their values."""
    import torch

    def describe(value):
        if isinstance(value, torch.Tensor):
            try:
                version = value._version
            except RuntimeError:
                version = 'untracked_inference_tensor'
            return {'tensor_id': id(value), 'version': version,
                    'shape': list(value.shape), 'dtype': str(value.dtype),
                    'device': str(value.device), 'layout': str(value.layout),
                    'requires_grad': value.requires_grad}
        if isinstance(value, dict):
            return {('tensor_key_' + str(id(key)) if isinstance(key, torch.Tensor)
                     else str(key)): describe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [describe(item) for item in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        return {'object_id': id(value), 'type': type(value).__qualname__}

    models = {'sender': trainer.sender, 'receiver': trainer.receiver,
              'observer': trainer.observer.model, 'ppo_model': trainer.ppo.model,
              'ppo_reference': trainer.ppo.ref_model}
    state = {'models': {}, 'optimizers': {}}
    for role, model in models.items():
        if model is None:
            state['models'][role] = None
            continue
        state['models'][role] = {
            'model_id': id(model),
            'parameters': {name: describe(param) for name, param in model.named_parameters()},
            'gradients': {name: describe(param.grad) for name, param in model.named_parameters()},
            'buffers': {name: describe(buffer) for name, buffer in model.named_buffers()},
            'modes': {name: module.training for name, module in model.named_modules()},
        }
    for role, optimizer in {'sender': trainer.sender_optimizer,
                            'receiver': trainer.receiver_optimizer,
                            'ppo': trainer.ppo.optimizer}.items():
        state['optimizers'][role] = {'optimizer_id': id(optimizer),
                                    'state': describe(optimizer.state),
                                    'param_groups': describe(optimizer.param_groups)}
    return state


def memory_snapshot(torch_module):
    free, total = torch_module.cuda.mem_get_info(0)
    return {'cuda_allocated_bytes': torch_module.cuda.memory_allocated(0),
            'cuda_reserved_bytes': torch_module.cuda.memory_reserved(0),
            'device_free_bytes': free, 'device_total_bytes': total,
            'host_rss_bytes': host_rss_bytes()}


def release_unused_cuda_cache(trainer, training, assert_equal, *,
                              torch_module=None, collect=gc.collect):
    """Observe invariants across synchronization, GC and allocator-cache release."""
    if torch_module is None:
        import torch as torch_module
    torch_module.cuda.synchronize()
    before_state = live_state_metadata(trainer)
    before_rng = training.capture_rng()
    before_memory = memory_snapshot(torch_module)
    collected = collect()
    torch_module.cuda.empty_cache()
    torch_module.cuda.synchronize()
    after_state = live_state_metadata(trainer)
    after_rng = training.capture_rng()
    if before_state != after_state:
        raise RuntimeError('Live model/optimizer tensor metadata changed during cache cleanup')
    assert_equal(after_rng, before_rng, 'evaluation_cache_cleanup_rng')
    return {'before': before_memory, 'after': memory_snapshot(torch_module),
            'gc_collected_objects': collected, 'live_tensor_metadata_unchanged': True,
            'rng_unchanged': True,
            'live_tensor_metadata_sha256': hashlib.sha256(
                json.dumps(before_state, sort_keys=True).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('memory-launcher', 'hardware-launcher', 'training-source', 'config', 'run-dir'):
        parser.add_argument('--' + argument, required=True, type=Path)
    args = parser.parse_args()
    memory_path, hardware_path, source, config_path, run = (getattr(args, name).resolve()
        for name in ('memory_launcher', 'hardware_launcher', 'training_source', 'config', 'run_dir'))
    if sha256(memory_path) != MEMORY_LAUNCHER_SHA256:
        raise ValueError('The pinned L40S memory launcher changed')
    config = read(config_path)
    if not run.is_dir() or read(run / 'config.json') != config:
        raise ValueError('Evaluation cleanup requires the existing unchanged run configuration')
    for relative, expected in config['source_sha256'].items():
        if sha256(source / relative) != expected:
            raise ValueError('Frozen source changed: ' + relative)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    os.chdir(source)
    from stego import university_training as training
    from stego import university_evaluation as evaluation
    for module in (training, evaluation):
        if Path(module.__file__).resolve().parent != source / 'stego':
            raise ValueError('Frozen module imported from another source')
    specification = importlib.util.spec_from_file_location('_verified_l40s_memory_launcher', memory_path)
    memory = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(memory)
    if sha256(hardware_path) != memory.HARDWARE_LAUNCHER_SHA256:
        raise ValueError('The pinned L40S hardware launcher changed')
    original_judge = evaluation._run_judge
    wrapper_sha = sha256(__file__)
    config_digest = training.digest(config)
    manifest = {
        'kind': 'L40S_evaluation_parent_cache_cleanup', 'config_digest': config_digest,
        'training_source': str(source), 'source_sha256': config['source_sha256'],
        'wrapper_sha256': wrapper_sha, 'memory_launcher_sha256': MEMORY_LAUNCHER_SHA256,
        'hardware_launcher_sha256': memory.HARDWARE_LAUNCHER_SHA256,
        'original_run_judge_source_sha256': hashlib.sha256(inspect.getsource(original_judge).encode()).hexdigest(),
        'scope': 'Before the unchanged frozen evaluator._run_judge child launch',
        'cleanup': ['torch.cuda.synchronize', 'gc.collect', 'torch.cuda.empty_cache', 'torch.cuda.synchronize'],
        'models_moved': False, 'live_tensors_mutated': False, 'scientific_configuration_changed': False,
        'judge_command_and_configuration_changed': False,
        'invariants': 'Model parameters/buffers/gradients/modes and optimizer tensor identities/versions/shapes/dtypes/devices/state metadata; Python/NumPy/Torch/CUDA RNG.',
    }

    def run_judge_after_cleanup(trainer, judge_config, evaluation_run, evaluation_digest):
        # Called by original evaluation while original main owns the run LOCK.
        if (Path(trainer.run).resolve() != run or trainer.config_digest != config_digest
                or trainer.config != config):
            raise RuntimeError('Cache cleanup targeted another run or configuration')
        checkpoint = evaluation._checkpoint(trainer)
        if Path(evaluation_run).resolve() != run / 'evaluations' / f'step_{trainer.update:06d}':
            raise RuntimeError('Cache cleanup targeted another evaluation directory')
        if (judge_config.get('training_config_digest') != config_digest
                or training.digest(judge_config) != evaluation_digest):
            raise RuntimeError('Cache cleanup received a changed judge configuration')
        # Preserve the original preemption boundary before doing cleanup work.
        evaluation._boundary(trainer, evaluation_run, evaluation_digest)
        evaluation._write_once(run / 'L40S_EVALUATION_MEMORY.json', manifest)
        events = run / 'runtime_events'
        events.mkdir(exist_ok=True)
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        safe_job_id = ''.join(c for c in job_id if c.isalnum() or c in '-_')
        event_path = events / f'l40s_eval_memory_{safe_job_id}_{time.time_ns()}.json'
        event = {'kind': manifest['kind'], 'status': 'cleaning_parent_cache',
                 'started_at': time.time(), 'config_digest': config_digest,
                 'evaluation_config_digest': evaluation_digest,
                 'evaluation_dir': str(evaluation_run), 'checkpoint': checkpoint,
                 'checkpoint_complete': read(Path(trainer.current_checkpoint) / 'COMPLETE.json'),
                 'slurm_job_id': job_id, 'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
                 'wrapper_sha256': wrapper_sha, 'memory_launcher_sha256': MEMORY_LAUNCHER_SHA256,
                 'hardware_launcher_sha256': memory.HARDWARE_LAUNCHER_SHA256,
                 'manifest_sha256': sha256(run / 'L40S_EVALUATION_MEMORY.json')}
        training.atomic_json(event_path, event)
        try:
            event['cleanup_verification'] = release_unused_cuda_cache(trainer, training, memory.exact_equal)
            event.update(status='launching_original_judge', cache_cleaned_at=time.time())
            training.atomic_json(event_path, event)
            result = original_judge(trainer, judge_config, evaluation_run, evaluation_digest)
        except BaseException as exc:
            event.update(status='preempted' if isinstance(exc, SystemExit) and exc.code == 75 else 'failed',
                         finished_at=time.time(), error_type=type(exc).__name__, error=str(exc))
            try:
                training.atomic_json(event_path, event)
            except Exception:
                pass
            raise
        event.update(status='original_judge_completed', finished_at=time.time())
        try:
            training.atomic_json(event_path, event)
        except Exception:
            pass
        return result

    evaluation._run_judge = run_judge_after_cleanup
    sys.argv = [str(memory_path), '--hardware-launcher', str(hardware_path),
                '--training-source', str(source), '--config', str(config_path), '--run-dir', str(run)]
    memory.main()


if __name__ == '__main__':
    main()
