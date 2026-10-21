#!/usr/bin/env python3
"""Resume L40S university PPO with saved activations held in pinned CPU RAM.

The existing hardware launcher, original run configuration, and frozen training
source remain unchanged. The sole training execution wrapper places PPO.step in
layout-preserving saved-tensor CPU hooks, without recomputation, batch
changes, dtype changes or new model evaluations. A CUDA-only toy parity check
must pass before model loading; the caller RNG is restored around that check.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time


HARDWARE_LAUNCHER_SHA256 = "17dbf6d322d3bc5cac719c0a1ebef701e88e4a25d6ec4e9a302268a414f91565"


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def source_sha256(function):
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


class ParityError(AssertionError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = details


def exact_equal(actual, expected, label='state'):
    """Assert exact nested tensor/RNG equality without publishing their values."""
    import numpy as np
    import torch
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()
                or not torch.equal(actual.detach().cpu(), expected.detach().cpu())):
            details = {'field': label, 'expected_shape': list(expected.shape),
                       'expected_dtype': str(expected.dtype)}
            if isinstance(actual, torch.Tensor) and actual.shape == expected.shape:
                a, b = actual.detach().cpu(), expected.detach().cpu()
                details['different_elements'] = int((a != b).sum())
                if torch.isfinite(a).all() and torch.isfinite(b).all():
                    details['max_absolute_difference'] = float((a.double() - b.double()).abs().max()) if a.numel() else 0.
            raise ParityError('Tensor parity failed: ' + label, details)
    elif isinstance(expected, np.ndarray):
        if (not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype
                or not np.array_equal(actual, expected)):
            raise AssertionError('Array parity failed: ' + label)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError('Mapping parity failed: ' + label)
        for key in expected:
            exact_equal(actual[key], expected[key], label + '.' + str(key))
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError('Sequence parity failed: ' + label)
        for index, value in enumerate(expected):
            exact_equal(actual[index], value, label + '.' + str(index))
    elif actual != expected:
        raise AssertionError('Value parity failed: ' + label)


def cpu_snapshot(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(item) for item in value)
    return value


def host_rss_bytes():
    """Linux process RSS, including pinned CPU activations, when available."""
    try:
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def flat_storage_view(tensor):
    """View the complete backing storage without changing dtype or layout."""
    import torch
    if tensor.layout != torch.strided or tensor.is_conj() or tensor.is_neg():
        raise RuntimeError('CPU offload only supports ordinary dense strided tensors')
    nbytes = tensor.untyped_storage().nbytes()
    if nbytes % tensor.element_size():
        raise RuntimeError('Saved tensor storage is not an integral number of elements')
    return tensor.detach().as_strided((nbytes // tensor.element_size(),), (1,), storage_offset=0)


def layout_preserving_cpu_offload(stats=None):
    """Copy complete CUDA storage; restore its original size/stride/offset.

    CPU tensors retain their backing storage. Separate saved tensors receive
    separate snapshots; cross-tensor storage alias identity is not promised.
    Full-storage copies can consume more host RAM than logical tensor contents.
    """
    import torch
    if stats is None:
        stats = {}

    def add(key, value=1):
        stats[key] = stats.get(key, 0) + value

    def pack(tensor):
        flat = flat_storage_view(tensor)
        if tensor.device.type == 'cpu':
            return ('cpu', tensor.detach())
        if tensor.device.type != 'cuda':
            raise RuntimeError('Saved-tensor offload supports only CPU and CUDA')
        snapshot = torch.empty(flat.shape, dtype=flat.dtype, device='cpu', pin_memory=True)
        snapshot.copy_(flat, non_blocking=False)
        if snapshot.numel() and not snapshot.is_pinned():
            raise RuntimeError('CUDA backing storage was not copied into pinned CPU RAM')
        add('packed_cuda_tensors')
        add('packed_storage_bytes', flat.numel() * flat.element_size())
        add('noncontiguous_cuda_tensors', int(not tensor.is_contiguous()))
        add('nonzero_offset_cuda_tensors', int(tensor.storage_offset() != 0))
        add('expanded_cuda_tensors', int(any(n > 1 and stride == 0 for n, stride in zip(tensor.size(), tensor.stride()))))
        return ('cuda', snapshot, tensor.device, tuple(tensor.size()), tuple(tensor.stride()), tensor.storage_offset())

    def unpack(saved):
        if saved[0] == 'cpu':
            return saved[1]
        _, snapshot, device, size, stride, offset = saved
        restored = snapshot.to(device, non_blocking=True).as_strided(size, stride, storage_offset=offset)
        if restored.device != device or tuple(restored.stride()) != stride or restored.storage_offset() != offset:
            raise RuntimeError('CPU offload failed to restore the exact saved layout')
        add('unpacked_cuda_tensors')
        return restored

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)


def cuda_toy_parity(training):
    """Check real CUDA offload transfers; never touches full training models."""
    import torch
    from torch import nn
    from torch.nn import functional as F
    if not torch.cuda.is_available():
        raise RuntimeError('CPU-only execution cannot verify CUDA activation offload')
    if torch.__version__.split('+')[0] != '2.4.0':
        raise RuntimeError('The offload parity check requires the original torch2.4.0')
    actual_rng = training.capture_rng()
    transfers = {'packed_cuda_tensors': 0, 'unpacked_cuda_tensors': 0,
                 'all_packed_cuda_tensors_on_pinned_cpu': True}

    class ToyLoRA(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = nn.Linear(16, 12, bias=False, device='cuda:0', dtype=torch.float16)
            self.base.requires_grad_(False)
            self.a = nn.Parameter(torch.randn(4, 16, device='cuda:0') * .03)
            self.b = nn.Parameter(torch.randn(12, 4, device='cuda:0') * .03)
            self.value = nn.Linear(12, 1, device='cuda:0', dtype=torch.float32)
            self.frozen_out = nn.Linear(12, 12, bias=False, device='cuda:0', dtype=torch.float32)
            self.frozen_out.requires_grad_(False)
            self.register_buffer('scale', torch.randn(1, 12, device='cuda:0'))
            self.dropout = nn.Dropout(.05)

        def forward(self, inputs):
            adapted = F.linear(F.linear(self.dropout(inputs.float()), self.a), self.b)
            logits = torch.tanh(self.base(inputs).float() + adapted)
            # Explicit non-dense, nonzero-offset and expanded saved views, plus
            # a frozen linear whose input gradient requires its transposed weight.
            shifted = torch.cat((logits[:, :1], logits), dim=1)[:, 1:]
            logits = self.frozen_out(shifted.square() * self.scale.expand_as(shifted))
            return logits, self.value(logits)

    def one_run(offload):
        training.seed_all(20261001)
        model = ToyLoRA().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.41e-5, weight_decay=.01)
        inputs = torch.randn(8, 16, device='cuda:0', dtype=torch.float16)
        labels = torch.arange(8, device='cuda:0') % 12
        initial = cpu_snapshot(model.state_dict())
        context = layout_preserving_cpu_offload(transfers) if offload else nullcontext()
        with context:
            logits, values = model(inputs)
            loss = F.cross_entropy(logits, labels) + .1 * values.square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite toy loss')
            loss.backward()
            gradients = {name: cpu_snapshot(parameter.grad) for name, parameter in model.named_parameters()
                         if parameter.requires_grad}
            if any(value is None or not torch.isfinite(value).all() for value in gradients.values()):
                raise FloatingPointError('Missing/nonfinite toy gradients')
            if not all(bool(torch.any(value != 0)) for value in gradients.values()):
                raise AssertionError('Toy parity test has an inactive trainable parameter')
            optimizer.step()
        torch.cuda.synchronize()
        return {'initial': initial, 'outputs': cpu_snapshot((logits, values)),
                'loss': cpu_snapshot(loss), 'gradients': gradients,
                'updated_parameters': cpu_snapshot(model.state_dict()),
                'optimizer_state': cpu_snapshot(optimizer.state_dict()),
                'post_rng': training.capture_rng()}

    try:
        baseline = one_run(False)
        repeat = one_run(False)
        exact_equal(repeat, baseline, 'cuda_toy_baseline_repeat')
        offloaded = one_run(True)
        exact_equal(offloaded, baseline, 'cuda_toy')
        if not transfers['packed_cuda_tensors'] or not transfers['unpacked_cuda_tensors']:
            raise AssertionError('Toy did not exercise CUDA activation offload/reload')
        if not all(transfers.get(key, 0) for key in ('noncontiguous_cuda_tensors', 'nonzero_offset_cuda_tensors', 'expanded_cuda_tensors')):
            raise AssertionError('Toy did not exercise all required saved-view layouts')
        result = {'status': 'passed', 'device': torch.cuda.get_device_name(0),
            'test': 'Same-seeded CUDA FP16 frozen linear + FP32 low-rank adapters, dropout, value head, post-adapter frozen linear and saved strided/offset/expanded views',
            'exact_initial_parameters': True, 'exact_outputs': True, 'exact_loss': True,
            'exact_gradients': True, 'exact_adamw_state': True, 'exact_updated_parameters': True,
            'exact_post_test_rng': True, 'loss': float(baseline['loss']),
            'exact_baseline_repeat': True, 'toy_optimizer_steps': 3, 'actual_training_model_updates': 0,
            'cuda_transfer_verified': True, **transfers,
            'limitation': 'A tiny graph parity check does not establish full-model numerical equivalence or memory capacity.'}
    finally:
        training.restore_rng(actual_rng)
        exact_equal(training.capture_rng(), actual_rng, 'caller_rng_restored')
    result['caller_rng_restored'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('hardware-launcher', 'training-source', 'config', 'run-dir'):
        parser.add_argument('--' + argument, required=True, type=Path)
    args = parser.parse_args()
    hardware_path, source, config_path, run = (getattr(args, name).resolve()
        for name in ('hardware_launcher', 'training_source', 'config', 'run_dir'))
    if sha256(hardware_path) != HARDWARE_LAUNCHER_SHA256:
        raise ValueError('The original L40S hardware launcher hash changed')
    config = json.loads(config_path.read_text())
    if not run.is_dir() or json.loads((run / 'config.json').read_text()) != config:
        raise ValueError('Memory offload requires the existing unchanged run configuration')
    for relative, expected in config['source_sha256'].items():
        if sha256(source / relative) != expected:
            raise ValueError('Frozen training source changed: ' + relative)
    sys.dont_write_bytecode = True
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.path.insert(0, str(source))
    os.chdir(source)
    from stego import university_training as training
    if Path(training.__file__).resolve() != source / 'stego' / 'university_training.py':
        raise ValueError('Training module imported from another source')
    original_train = training.Trainer.train
    original_train_sha = source_sha256(original_train)
    wrapper_sha = sha256(__file__)
    config_digest = training.digest(config)
    # Fail fast before any full model is constructed. Only a uniquely named
    # public event is written here; original manifests/FAILURE remain untouched.
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or 'L40S' not in torch.cuda.get_device_name(0)
            or torch.cuda.get_device_properties(0).total_memory < 44 * 1024**3):
        raise RuntimeError('Layout-preserving offload preflight requires one L40S with at least44 GiB')
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    safe_job_id = ''.join(character for character in job_id if character.isalnum() or character in '-_')
    events = run / 'runtime_events'
    events.mkdir(exist_ok=True)
    preflight_path = events / f'l40s_memory_preflight_{safe_job_id}_{time.time_ns()}.json'
    preflight = {'kind': 'L40S_layout_preserving_offload_preflight', 'status': 'checking_cuda_parity',
        'before_model_loading': True, 'started_at': time.time(), 'config_digest': config_digest,
        'wrapper_sha256': wrapper_sha, 'hardware_launcher_sha256': HARDWARE_LAUNCHER_SHA256,
        'slurm_job_id': job_id, 'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__}
    training.atomic_json(preflight_path, preflight)
    try:
        parity = cuda_toy_parity(training)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        preflight.update(status='cuda_preflight_passed', completed_at=time.time(), cuda_toy_parity=parity)
        training.atomic_json(preflight_path, preflight)
    except BaseException as exc:
        preflight.update(status='cuda_preflight_failed', failed_at=time.time(),
            error_type=type(exc).__name__, error=str(exc), parity_failure=getattr(exc, 'details', None))
        training.atomic_json(preflight_path, preflight)
        raise

    def offload_train(self):
        # Original main's LOCK is held; the hardware launcher has restored and
        # verified update/elapsed budget before it calls this method.
        import torch
        if (Path(self.run).resolve() != run or self.config_digest != config_digest
                or self.update < 1 or not getattr(self, 'current_checkpoint', None)):
            raise RuntimeError('Activation offload requires the restored original training run')
        if torch.__version__.split('+')[0] != '2.4.0':
            raise RuntimeError('Activation offload requires original torch2.4.0')
        original_step = self.ppo.step
        step_sha = source_sha256(original_step)
        manifest = {
            'kind': 'L40S_PPO_saved_activation_offload', 'config_digest': config_digest,
            'training_source': str(source), 'wrapper_sha256': wrapper_sha,
            'hardware_launcher_sha256': HARDWARE_LAUNCHER_SHA256,
            'original_train_source_sha256': original_train_sha, 'original_ppo_step_source_sha256': step_sha,
            'scope': 'Only the existing instance PPO.step call',
            'implementation': 'torch.autograd.graph.saved_tensors_hooks with full-storage pinned-CPU snapshots and exact size/stride/storage-offset restoration',
            'offload_layout_policy': 'Dense strided tensors only; CPU detached pass-through; reject conjugate/negative flags and non-CPU/CUDA devices; no cross-tensor alias identity guarantee.',
            'host_memory_note': 'Each saved CUDA tensor snapshots its full underlying storage; host memory may exceed the logical activation size.',
            'scientific_configuration_changed': False, 'batch_sizes_changed': False,
            'dtypes_changed': False, 'activation_recomputation': False,
            'receiver_offload': False, 'frozen_source_files_changed': False,
            'original_hardware_migration_manifest_preserved': True,
            'parity_policy': 'Exact CUDA toy outputs/loss/gradients/AdamW state/updated parameters/post-RNG; fail closed on any difference.',
        }
        manifest_path = run / 'L40S_MEMORY_OFFLOAD.json'
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Saved memory-offload manifest differs from this versioned launcher')
        training.atomic_json(manifest_path, manifest)
        job_id = os.environ.get('SLURM_JOB_ID', 'local')
        safe_job_id = ''.join(character for character in job_id if character.isalnum() or character in '-_')
        events = run / 'runtime_events'
        events.mkdir(exist_ok=True)
        event_path = events / f'l40s_memory_{safe_job_id}_{time.time_ns()}.json'
        event = {'kind': 'L40S_PPO_saved_activation_offload', 'status': 'memory_offload_enabled',
            'started_at': time.time(), 'slurm_job_id': job_id,
            'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
            'config_digest': config_digest, 'completed_update_at_enable': self.update,
            'checkpoint': str(self.current_checkpoint), 'restored_elapsed_seconds': self.elapsed,
            'memory_manifest_sha256': sha256(manifest_path), 'wrapper_sha256': wrapper_sha,
            'hardware_launcher_sha256': HARDWARE_LAUNCHER_SHA256,
            'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
            'bitsandbytes': importlib.metadata.version('bitsandbytes'),
            'enabled_at': time.time(), 'cuda_toy_parity': parity,
            'preflight_event': str(preflight_path.relative_to(run)),
            'preflight_event_sha256': sha256(preflight_path)}
        training.atomic_json(event_path, event)

        def offloaded_step(*step_args, **step_kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            started = time.monotonic()
            upcoming_update = self.update + 1
            outcome, error = 'completed', None
            transfers = {}
            try:
                with layout_preserving_cpu_offload(transfers):
                    return original_step(*step_args, **step_kwargs)
            except BaseException as exc:
                outcome, error = 'failed', {'type': type(exc).__name__, 'message': str(exc)}
                raise
            finally:
                # Timing/memory observation must not suppress the original PPO
                # error or change its numerical result/optimization state.
                try:
                    torch.cuda.synchronize()
                    free, total = torch.cuda.mem_get_info(0)
                    event['last_ppo_memory'] = {'attempted_update': upcoming_update,
                        'ppo_call_status': outcome, 'error': error,
                        'duration_seconds': time.monotonic() - started,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                        'end_device_free_bytes': free, 'device_total_bytes': total,
                        'end_host_rss_bytes': host_rss_bytes(),
                        'saved_tensor_transfers': transfers,
                        'note': 'PPO-call completion alone is not a committed training checkpoint.'}
                    training.atomic_json(event_path, event)
                except Exception:
                    pass

        self.ppo.step = offloaded_step
        try:
            return original_train(self)
        finally:
            self.ppo.step = original_step

    training.Trainer.train = offload_train
    specification = importlib.util.spec_from_file_location('_verified_university_l40s_hardware_launcher', hardware_path)
    hardware = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(hardware)
    sys.argv = [str(hardware_path), '--training-source', str(source),
                '--config', str(config_path), '--run-dir', str(run)]
    hardware.main()


if __name__ == '__main__':
    main()
