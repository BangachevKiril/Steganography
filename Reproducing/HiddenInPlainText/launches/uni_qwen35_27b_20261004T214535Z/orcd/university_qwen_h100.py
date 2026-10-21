#!/usr/bin/env python3
"""Run three fixed-budget segments with frozen Qwen3.5-27B-FP8 on one H100.

Only helper functions are imported from the two hash-pinned historical runtime
modules. Their continuation-only main functions are never called. The corrected
training source owns initialization, scientific behavior, locking and recovery.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import time


EXPERIMENT_NAME = 'uni_qwen35_27b_20261004T214535Z'
MEMORY_HELPER_SHA256 = '9ce57ee0ff9a043273f80a75424a80bdf8a97189e0ccaedb00dce2f0f9e9b2c4'
EVALUATION_HELPER_SHA256 = 'd5f9352f0c6984c29f0078ec035be77822cb81c2fe2010f38248c634326e90f2'
BASELINE_STATE_SHA256 = 'bafa15aa893985940993b5e2baeaa96e5c2d2b88803d0848c62649e97712f43b'
BASELINE_CONFIG_DIGEST = '4caf38d2397dec5d62b0c99f673e1e0049fd0cc4be3d973f77b0a3f4b2098ee6'
BASELINE_RECEIVER_SHA256 = '2679f33e7ec8b6e17c07f83941b9fa3d491cbf5b407bcecb029b51c7c80b512b'
BASELINE_RECEIVER_HEAD_SHA256 = '7da225b99b6cab5e8678d807c4a27cf36f2008563c2fe98fe25687221f211003'


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def load_helper(path, expected, module_name):
    if sha256(path) != expected:
        raise ValueError('Audited runtime helper changed: ' + str(path))
    specification = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def write_once(training, path, value):
    if path.exists() and read(path) != value:
        raise ValueError('Immutable runtime manifest changed: ' + str(path))
    training.atomic_json(path, value)


def checked_checkpoint(training, run, config_digest):
    checkpoint = training.latest_checkpoint(run, config_digest, verify_hash=True)
    if checkpoint is None:
        return None
    marker = read(checkpoint / 'COMPLETE.json')
    if (marker['completed_update'] < 0 or marker['config_digest'] != config_digest
            or checkpoint.resolve().parent != (run / 'checkpoints').resolve()
            or checkpoint.name != 'step_%06d' % marker['completed_update']):
        raise ValueError('Checkpoint does not belong to this corrected run')
    return {'path': str(checkpoint.resolve()), 'marker': marker,
            'complete_sha256': sha256(checkpoint / 'COMPLETE.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('training-source', 'config', 'run-dir'):
        parser.add_argument('--' + argument, required=True, type=Path)
    parser.add_argument('--segment', required=True, type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    source, config_path, run = (getattr(args, name).resolve()
        for name in ('training_source', 'config', 'run_dir'))
    config = read(config_path)
    if (source.name != EXPERIMENT_NAME or run.name != EXPERIMENT_NAME
            or config.get('name') != EXPERIMENT_NAME
            or config_path != source / 'configs' / 'university_training.json'):
        raise ValueError('This launcher is restricted to its new corrected experiment')
    if (config.get('baseline_state_sha256') != BASELINE_STATE_SHA256
            or config.get('baseline_config_digest') != BASELINE_CONFIG_DIGEST
            or config.get('expected_baseline_receiver_sha256') != BASELINE_RECEIVER_SHA256
            or config.get('expected_baseline_receiver_head_sha256') != BASELINE_RECEIVER_HEAD_SHA256
            or config.get('target_gpu') != 'H100'
            or config.get('corrected_protocol') != 'university_bugfix_v1'
            or config.get('evaluate_initial_checkpoint') is not True
            or config.get('max_active_seconds') != 83700):
        raise ValueError('Verified initialization or authorized active budget changed')
    for relative, expected in config['source_sha256'].items():
        if sha256(source / relative) != expected:
            raise ValueError('Corrected source hash mismatch: ' + relative)
    helper_root = Path(__file__).resolve().parent
    if helper_root != source / 'orcd':
        raise ValueError('Launcher must belong to the pinned corrected source release')
    memory = load_helper(helper_root / 'university_l40s_memory_training.py',
                         MEMORY_HELPER_SHA256, '_corrected_memory_helpers')
    cleanup = load_helper(helper_root / 'university_l40s_evaluation_memory.py',
                          EVALUATION_HELPER_SHA256, '_corrected_evaluation_helpers')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    os.chdir(source)
    from stego import university_training as training
    from stego import university_evaluation as evaluation
    for module in (training, evaluation):
        if Path(module.__file__).resolve().parent != source / 'stego':
            raise ValueError('Training/evaluation imported from another source')
    from orcd.qwen_observer_client import QwenObserverClient
    from orcd.qwen_segment_training import install_segment_training
    observers = []
    def make_observer(observer_config):
        worker = QwenObserverClient(observer_config, run=run)
        observers.append(worker)
        audit = dict(worker.ready, config_digest=training.digest(config),
                     slurm_job_id=os.environ.get('SLURM_JOB_ID'), started_at=time.time())
        training.atomic_json(event_path('qwen_observer'), audit)
        training.atomic_json(run / 'QWEN_OBSERVER.json', audit)
        return worker
    training.Observer = make_observer
    install_segment_training(training, args.segment)
    config_digest = training.digest(config)
    wrapper_sha = sha256(__file__)
    original_init = training.Trainer.__init__
    original_judge = evaluation._run_judge
    original_checkpoint = training.Trainer.checkpoint
    def checkpoint_with_cycle_audit(self):
        original_checkpoint(self)
        if self.update == 7:
            kinds = [r['kind'] for r in self.metrics]
            if kinds != ['receiver', 'receiver'] + ['sender'] * 5:
                raise RuntimeError('First completed training cycle changed its phase schedule')
            baseline_evaluation = run / 'evaluations/step_000000/COMPLETE.json'
            training.atomic_json(run / 'FULL_CYCLE_SMOKE.json', dict(
                status='passed', config_digest=self.config_digest, completed_update=7,
                checkpoint_complete_sha256=sha256(Path(self.current_checkpoint)/'COMPLETE.json'),
                initial_evaluation_complete_sha256=sha256(baseline_evaluation),
                receiver_updates=2, sender_updates=5,
                observer_revision=config['qwen_observer']['revision'],
                gpu='H100', verified_at=time.time()))
    training.Trainer.checkpoint = checkpoint_with_cycle_audit
    runtime_manifest = {
        'schema_version': 1, 'kind': 'qwen_university_H100_runtime',
        'experiment': EXPERIMENT_NAME, 'config_digest': config_digest,
        'training_source': str(source), 'source_sha256': config['source_sha256'],
        'wrapper_sha256': wrapper_sha,
        'memory_helper_sha256': MEMORY_HELPER_SHA256,
        'evaluation_helper_sha256': EVALUATION_HELPER_SHA256,
        'helper_main_functions_called': False,
        'fresh_initialization': 'Verified original baseline checkpoint0; zero optimizers; matching receiver digest.',
        'resume_policy': 'Only this corrected run highest complete hash-verified checkpoint, including update0; preserve elapsed budget.',
        'scientific_corrections': config.get('corrections',
            'See corrected experiment protocol and source manifest.'),
        'memory_scope': 'PPO.step saved tensors only; original full storage, size, stride and storage offset.',
        'judge_cleanup_scope': 'Synchronize, collect unreachable Python objects and release unused CUDA allocator cache.',
        'max_active_seconds': 83700,
    }

    def event_path(prefix):
        job = os.environ.get('SLURM_JOB_ID', 'local')
        safe = ''.join(c for c in job if c.isalnum() or c in '-_')
        events = run / 'runtime_events'
        events.mkdir(exist_ok=True)
        return events / f'{prefix}_{safe}_{time.time_ns()}.json'

    def initialized(self, init_config, init_run):
        # training.main has already acquired the exclusive run LOCK and checked
        # the source/config identity before invoking this initialization wrapper.
        if init_config != config or Path(init_run).resolve() != run:
            raise ValueError('Initialization targets another experiment')
        import torch
        import bitsandbytes
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or 'H100' not in torch.cuda.get_device_name(0)
                or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3):
            raise RuntimeError('Corrected experiment requires one genuine H100 with at least75 GiB')
        versions = {name: importlib.metadata.version(name) for name in
            ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'bitsandbytes', 'numpy', 'datasets', 'tokenizers')}
        for name, expected in {'torch': '2.4.0', 'transformers': '4.43.2',
                               'trl': '0.9.6', 'bitsandbytes': '0.43.3'}.items():
            if versions[name].split('+')[0] != expected:
                raise RuntimeError('Original runtime required: ' + name + '==' + expected)
        selected = checked_checkpoint(training, run, config_digest)
        preflight_path = event_path('h100_memory_preflight')
        event = {'kind': 'qwen_H100_initialization', 'status': 'checking_cuda_parity',
            'started_at': time.time(), 'config_digest': config_digest,
            'wrapper_sha256': wrapper_sha, 'before_model_loading': True,
            'memory_helper_sha256': MEMORY_HELPER_SHA256,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
            'gpu': torch.cuda.get_device_name(0), 'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'gpu_capability': list(torch.cuda.get_device_capability(0)),
            'versions': versions, 'cuda': torch.version.cuda, 'executable': sys.executable,
            'bitsandbytes_module': bitsandbytes.__file__, 'source_checkpoint': selected,
            'fresh_baseline_initialization': selected is None}
        training.atomic_json(preflight_path, event)
        try:
            parity = memory.cuda_toy_parity(training)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            event.update(status='cuda_preflight_passed', cuda_toy_parity=parity,
                         parity_completed_at=time.time())
            training.atomic_json(preflight_path, event)
            if training.STOP:
                raise SystemExit(75)
            original_init(self, init_config, init_run)
            if self.observer.ready.get('transformers_version') != '5.2.0':
                raise RuntimeError('Frozen observer runtime requires Transformers 5.2.0')
            if self.observer.ready.get('torch_version', '').split('+')[0] != '2.10.0':
                raise RuntimeError('Frozen observer runtime requires Torch 2.10.0')
            actual = checked_checkpoint(training, run, config_digest)
            if actual is None or str(Path(self.current_checkpoint).resolve()) != actual['path']:
                raise RuntimeError('Initialized model lacks its verified committed checkpoint')
            if selected is not None:
                if (actual != selected or self.update != selected['marker']['completed_update']
                        or self.elapsed != selected['marker']['elapsed']):
                    raise RuntimeError('Checkpoint restoration or elapsed budget mismatch')
            else:
                initialization = read(run / 'INITIALIZATION.json')
                if (self.update != 0 or actual['marker']['completed_update'] != 0
                        or self.elapsed != 0 or len(self.sender_optimizer.state)
                        or len(self.receiver_optimizer.state)
                        or initialization['config_digest'] != config_digest
                        or initialization['parameter_updates'] != 0
                        or initialization['baseline_state_sha256'] != BASELINE_STATE_SHA256
                        or initialization['receiver_state_sha256'] != BASELINE_RECEIVER_SHA256
                        or initialization['receiver_head_sha256'] != BASELINE_RECEIVER_HEAD_SHA256
                        or initialization['optimizer_state_sizes'] != {'sender': 0, 'receiver': 0}
                        or training.tensor_digest(self.receiver) != BASELINE_RECEIVER_SHA256):
                    raise RuntimeError('Fresh baseline model/optimizer identity verification failed')
            write_once(training, run / 'QWEN_H100_RUNTIME.json', runtime_manifest)
            event.update(status='initialization_verified', initialized_at=time.time(),
                completed_update=self.update, restored_elapsed_seconds=self.elapsed,
                checkpoint=actual, configured_active_budget_seconds=config['max_active_seconds'])
            training.atomic_json(preflight_path, event)
        except BaseException as exc:
            event.update(status='preempted' if isinstance(exc, SystemExit) and exc.code == 75 else 'failed',
                failed_at=time.time(), error_type=type(exc).__name__, error=str(exc),
                parity_failure=getattr(exc, 'details', None))
            training.atomic_json(preflight_path, event)
            raise

        original_step = self.ppo.step
        memory_manifest = {
            'kind': 'H100_PPO_saved_activation_offload', 'config_digest': config_digest,
            'training_source': str(source), 'wrapper_sha256': wrapper_sha,
            'memory_helper_sha256': MEMORY_HELPER_SHA256,
            'original_ppo_step_source_sha256': hashlib.sha256(inspect.getsource(original_step).encode()).hexdigest(),
            'implementation': 'Hash-pinned layout_preserving_cpu_offload helper; full-storage pinned-CPU snapshots and exact original size/stride/storage-offset restoration.',
            'scope': 'Only this initialized instance PPO.step',
            'receiver_offload': False, 'activation_recomputation': False,
            'memory_wrapper_changes_batch_sizes_or_dtypes': False,
            'scientific_corrections_are_in_separate_frozen_training_source': True,
            'parity_policy': 'Exact CUDA toy parity before full model initialization; caller RNG restored.',
            'limitations': 'Toy parity does not establish full-model numerical equivalence. Full-storage copies use substantial host memory; cross-tensor storage alias identity is not promised.'}
        write_once(training, run / 'H100_MEMORY_OFFLOAD.json', memory_manifest)
        memory_path = event_path('h100_memory')
        memory_event = {'kind': memory_manifest['kind'], 'status': 'memory_offload_enabled',
            'config_digest': config_digest, 'wrapper_sha256': wrapper_sha,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
            'enabled_at': time.time(), 'completed_update_at_enable': self.update,
            'checkpoint': str(self.current_checkpoint), 'restored_elapsed_seconds': self.elapsed,
            'memory_manifest_sha256': sha256(run / 'H100_MEMORY_OFFLOAD.json'),
            'cuda_toy_parity': parity, 'preflight_event': str(preflight_path.relative_to(run)),
            'preflight_event_sha256': sha256(preflight_path)}
        training.atomic_json(memory_path, memory_event)

        def offloaded_step(*step_args, **step_kwargs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(0)
            began = time.monotonic()
            upcoming = self.update + 1
            outcome, error, transfers = 'completed', None, {}
            try:
                with memory.layout_preserving_cpu_offload(transfers):
                    return original_step(*step_args, **step_kwargs)
            except BaseException as exc:
                outcome, error = 'failed', {'type': type(exc).__name__, 'message': str(exc)}
                raise
            finally:
                try:
                    torch.cuda.synchronize()
                    free, total = torch.cuda.mem_get_info(0)
                    memory_event['last_ppo_memory'] = {
                        'attempted_update': upcoming, 'ppo_call_status': outcome, 'error': error,
                        'duration_seconds': time.monotonic() - began,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
                        'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
                        'end_device_free_bytes': free, 'device_total_bytes': total,
                        'end_host_rss_bytes': memory.host_rss_bytes(),
                        'saved_tensor_transfers': transfers,
                        'note': 'PPO-call completion alone does not establish a committed checkpoint.'}
                    training.atomic_json(memory_path, memory_event)
                except Exception:
                    pass

        self.ppo.step = offloaded_step

    evaluation_manifest = {
        'kind': 'H100_evaluation_parent_cache_cleanup', 'config_digest': config_digest,
        'training_source': str(source), 'wrapper_sha256': wrapper_sha,
        'evaluation_helper_sha256': EVALUATION_HELPER_SHA256,
        'original_run_judge_source_sha256': hashlib.sha256(inspect.getsource(original_judge).encode()).hexdigest(),
        'scope': 'Before the original corrected evaluator._run_judge boundary',
        'cleanup': ['torch.cuda.synchronize', 'gc.collect', 'torch.cuda.empty_cache', 'torch.cuda.synchronize'],
        'models_moved': False, 'live_tensors_mutated': False,
        'judge_command_and_configuration_changed_by_cleanup': False,
        'invariants': 'Live model/optimizer tensor identities, versions, shapes, dtypes, devices, modes and RNG unchanged.'}

    def run_judge_after_cleanup(trainer, judge_config, evaluation_run, evaluation_digest):
        if (Path(trainer.run).resolve() != run or trainer.config_digest != config_digest
                or trainer.config != config
                or Path(evaluation_run).resolve() != run / 'evaluations' / f'step_{trainer.update:06d}'
                or judge_config.get('training_config_digest') != config_digest
                or training.digest(judge_config) != evaluation_digest):
            raise ValueError('Judge cleanup targeted a different run/configuration')
        evaluation._boundary(trainer, evaluation_run, evaluation_digest)
        checkpoint = evaluation._checkpoint(trainer)
        write_once(training, run / 'H100_EVALUATION_MEMORY.json', evaluation_manifest)
        path = event_path('h100_eval_memory')
        event = {'kind': evaluation_manifest['kind'], 'status': 'cleaning_parent_cache',
            'started_at': time.time(), 'config_digest': config_digest,
            'evaluation_config_digest': evaluation_digest, 'evaluation_dir': str(evaluation_run),
            'checkpoint': checkpoint, 'checkpoint_complete': read(Path(trainer.current_checkpoint) / 'COMPLETE.json'),
            'wrapper_sha256': wrapper_sha, 'evaluation_helper_sha256': EVALUATION_HELPER_SHA256,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
            'manifest_sha256': sha256(run / 'H100_EVALUATION_MEMORY.json')}
        training.atomic_json(path, event)
        try:
            event['observer_cleanup_verification'] = trainer.observer.cleanup()
            event['cleanup_verification'] = cleanup.release_unused_cuda_cache(trainer, training, memory.exact_equal)
            event.update(status='launching_original_judge', cache_cleaned_at=time.time())
            training.atomic_json(path, event)
            result = original_judge(trainer, judge_config, evaluation_run, evaluation_digest)
        except BaseException as exc:
            event.update(status='preempted' if isinstance(exc, SystemExit) and exc.code == 75 else 'failed',
                finished_at=time.time(), error_type=type(exc).__name__, error=str(exc))
            training.atomic_json(path, event)
            raise
        event.update(status='original_judge_completed', finished_at=time.time())
        training.atomic_json(path, event)
        return result

    training.Trainer.__init__ = initialized
    evaluation._run_judge = run_judge_after_cleanup
    sys.argv = [str(source / 'stego' / 'university_training.py'),
                '--config', str(config_path), '--run-dir', str(run)]
    try:
        training.main()
    finally:
        for observer in observers:
            observer.close()


if __name__ == '__main__':
    main()
