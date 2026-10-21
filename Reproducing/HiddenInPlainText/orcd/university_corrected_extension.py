#!/usr/bin/env python3
"""Administrative budget extension; the corrected scientific release is immutable."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time

CONFIG_DIGEST = '3fe0ebcb50bb71fc727f8255161f45531c3dfe073eab644d2423cc41ce13fa92'
TRAINING_SHA256 = '3447f1a8f616acbca50e5f3ae13362ce5cd396da05fd7e347a09e6182d5f78b9'
RUNTIME_SHA256 = '453aff7ed707c3d835149e0c8ac4ec0563b4a6526a9309b91d9b31f329cb1ea2'
ORIGIN_STATE = '14c2e9342d13b12f8992feb18ff30ecdbd9c57d27ee62a3436047771e52bf6b9'
ORIGIN_MANIFEST = '8645f0ce9a7bbb5ff21b7ea53190ee2cc4ae7eff37a87195f8712c5801df148d'
ORIGIN_ELAPSED = 27930.404972237535
BASE = Path('/orcd/pool/003/kirilb/Steganography')
EXPERIMENT = 'uni_corrected_20261002T151448Z'


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def validate_manifest(manifest):
    expected = {
        'schema_version': 1, 'kind': 'authorized_corrected_university_extension',
        'config_digest': CONFIG_DIGEST, 'origin_update': 530,
        'origin_elapsed_seconds': ORIGIN_ELAPSED, 'additional_active_seconds': 27900,
        'target_active_seconds': ORIGIN_ELAPSED + 27900,
        'parent_state_sha256': ORIGIN_STATE,
        'parent_checkpoint_complete_sha256': ORIGIN_MANIFEST,
        'runtime_launcher_sha256': RUNTIME_SHA256,
        'run_dir': str(BASE / 'runs' / EXPERIMENT),
        'training_source': str(BASE / 'releases' / EXPERIMENT),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError('Continuation identity/budget mismatch: ' + key)
    if not manifest.get('authorization') or not manifest.get('extension_id'):
        raise ValueError('Missing explicit continuation authorization')
    archive = Path(manifest['archive_dir']).resolve()
    if archive.parent != BASE / 'archives' or archive.name != EXPERIMENT + '_stage1_update530':
        raise ValueError('Unexpected completed-stage archive')


def _patch_function(training, original, before, after):
    source = textwrap.dedent(inspect.getsource(original))
    if source.count(before) != 1:
        raise ValueError('Frozen function no longer has exactly one administrative patch point')
    namespace = {}
    # Use the live module globals so signal handlers can still change STOP.
    exec(compile(source.replace(before, after), '<authorized_university_extension>', 'exec'),
         training.__dict__, namespace)
    return namespace[original.__name__]


def patch_train_function(training, manifest):
    return _patch_function(training, training.Trainer.train,
        "self.config['max_active_seconds']", repr(manifest['target_active_seconds']))


def patch_checkpoint_function(training, manifest):
    return _patch_function(training, training.Trainer.checkpoint,
        "step>0 and step<self.update-keep+1 and step%self.config['evaluation_interval']!=0",
        "step>0 and step<self.update-keep+1 and step%self.config['evaluation_interval']!=0"
        + ' and step != ' + str(manifest['origin_update']))


def activate(trainer, manifest):
    """Called under training.main's exclusive run lock, after audited restoration."""
    from stego.university_training import atomic_json
    run, archive = Path(manifest['run_dir']), Path(manifest['archive_dir'])
    if Path(trainer.run).resolve() != run or trainer.config_digest != manifest['config_digest']:
        raise ValueError('Continuation targeted a different run/configuration')
    if trainer.update < manifest['origin_update'] or trainer.elapsed < manifest['origin_elapsed_seconds']:
        raise ValueError('Continuation cannot discard previously committed progress')
    pin = archive / 'checkpoints' / 'step_000530'
    checks = {
        archive / 'COMPLETE.json': manifest['parent_complete_sha256'],
        archive / 'BUDGET_COMPLETE.json': manifest['parent_budget_complete_sha256'],
        archive / 'evaluations/step_000530/COMPLETE.json': manifest['parent_evaluation_complete_sha256'],
        pin / 'COMPLETE.json': manifest['parent_checkpoint_complete_sha256'],
        pin / 'state.pt': manifest['parent_state_sha256'],
    }
    for path, expected in checks.items():
        if sha256(path) != expected:
            raise ValueError('Preserved completed-stage identity changed: ' + str(path))
    saved = run / 'CONTINUATION.json'
    if saved.exists():
        if read(saved) != manifest:
            raise ValueError('Continuation authorization changed after activation')
    else:
        if trainer.update != manifest['origin_update'] or trainer.elapsed != manifest['origin_elapsed_seconds']:
            raise ValueError('First continuation must restore exact checkpoint530')
        current = Path(trainer.current_checkpoint).resolve()
        if (current != run / 'checkpoints/step_000530'
                or sha256(current / 'COMPLETE.json') != manifest['parent_checkpoint_complete_sha256']):
            raise ValueError('Restored checkpoint does not match the authorized origin')
        for name, key in [('COMPLETE.json', 'parent_complete_sha256'),
                          ('BUDGET_COMPLETE.json', 'parent_budget_complete_sha256')]:
            if sha256(run / name) != manifest[key]:
                raise ValueError('Original terminal marker mismatch: ' + name)
        # Seal before removing markers; interrupted activation is safely repeatable.
        atomic_json(saved, manifest)
    for name, key in [('COMPLETE.json', 'parent_complete_sha256'),
                      ('BUDGET_COMPLETE.json', 'parent_budget_complete_sha256')]:
        path = run / name
        if path.exists():
            if sha256(path) == manifest[key]:
                path.unlink()
            elif read(path).get('completed_update', -1) <= manifest['origin_update']:
                raise ValueError('Unexpected terminal marker: ' + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    args = parser.parse_args()
    manifest = read(args.manifest)
    validate_manifest(manifest)
    if sha256(__file__) != manifest['wrapper_sha256']:
        raise ValueError('Continuation launcher hash mismatch')
    source, run = Path(manifest['training_source']), Path(manifest['run_dir'])
    config_path = source / 'configs/university_training.json'
    config = read(config_path)
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != CONFIG_DIGEST:
        raise ValueError('Frozen scientific configuration changed')
    for relative, expected in config['source_sha256'].items():
        if sha256(source / relative) != expected:
            raise ValueError('Frozen scientific source changed: ' + relative)
    if sha256(source / 'stego/university_training.py') != TRAINING_SHA256:
        raise ValueError('Unsupported training source')
    runtime_path = source / 'orcd/university_corrected_l40s.py'
    if sha256(runtime_path) != RUNTIME_SHA256:
        raise ValueError('Audited L40S launcher changed')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    os.chdir(source)
    from stego import university_training as training
    if Path(training.__file__).resolve() != source / 'stego/university_training.py':
        raise ValueError('Imported training from another release')
    extended_train = patch_train_function(training, manifest)
    training.Trainer.checkpoint = patch_checkpoint_function(training, manifest)

    def administrative_train(self):
        activate(self, manifest)
        events = run / 'runtime_events'
        events.mkdir(exist_ok=True)
        training.atomic_json(events / ('l40s_extension_%s_%s.json' %
            (os.environ.get('SLURM_JOB_ID', 'local'), time.time_ns())), {
                'kind': 'authorized_budget_extension_restored',
                'extension_id': manifest['extension_id'], 'config_digest': self.config_digest,
                'wrapper_sha256': manifest['wrapper_sha256'],
                'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'slurm_restart_count': os.environ.get('SLURM_RESTART_COUNT'),
                'completed_update': self.update, 'restored_elapsed_seconds': self.elapsed,
                'target_active_seconds': manifest['target_active_seconds'],
                'remaining_active_seconds_at_restored_checkpoint': max(0, manifest['target_active_seconds'] - self.elapsed),
                'checkpoint': str(self.current_checkpoint), 'activated_at': time.time()})
        extended_train(self)
        if (run / 'COMPLETE.json').exists():
            completed = read(run / 'COMPLETE.json')
            training.atomic_json(run / 'CONTINUATION_COMPLETE.json', {
                'extension_id': manifest['extension_id'], 'config_digest': self.config_digest,
                'completed_update': self.update, 'origin_update': manifest['origin_update'],
                'additional_active_seconds': manifest['additional_active_seconds'],
                'target_active_seconds': manifest['target_active_seconds'],
                'actual_added_active_seconds_before_final_evaluation':
                    read(run / 'BUDGET_COMPLETE.json')['active_seconds'] - manifest['origin_elapsed_seconds'],
                'actual_added_active_seconds_including_final_evaluation':
                    completed['active_seconds'] - manifest['origin_elapsed_seconds'],
                'complete_sha256': sha256(run / 'COMPLETE.json'),
                'budget_complete_sha256': sha256(run / 'BUDGET_COMPLETE.json')})

    training.Trainer.train = administrative_train
    spec = importlib.util.spec_from_file_location('_audited_corrected_runtime', runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    sys.argv = [str(runtime_path), '--training-source', str(source),
                '--config', str(config_path), '--run-dir', str(run)]
    runtime.main()


if __name__ == '__main__':
    main()
