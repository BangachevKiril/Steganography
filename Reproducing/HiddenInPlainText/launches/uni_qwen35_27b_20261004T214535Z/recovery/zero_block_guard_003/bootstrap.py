#!/usr/bin/env python3
"""Authorized FP8 zero-block repair overlay for the immutable university run.

The frozen training configuration and source files remain unchanged. This
bootstrap changes exactly one checksum expectation in the private configuration
copy sent to the frozen observer worker. The worker still verifies the actual
runtime file, and its ready audit is supplemented with the authorization chain.
No numerical function, training loop, checkpoint, clock, or optimizer is wrapped.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys


EXPERIMENT = 'uni_qwen35_27b_20261004T214535Z'
SOURCE = Path('/orcd/pool/003/kirilb/Steganography/releases') / EXPERIMENT
RUN = Path('/orcd/pool/003/kirilb/Steganography/runs') / EXPERIMENT
CONFIG_DIGEST = '42580b347ec09edac9f497c73c30f97f0c457a7955000873b81a8d0b5494e12d'
RUNTIME = '/orcd/pool/003/kirilb/Steganography/envs/qwen35_observer/lib/python3.12/site-packages/transformers/integrations/finegrained_fp8.py'
BEFORE = 'a486628c4f03abfe299653200ebd2cc577a9ff514fa37cc773031440702107a7'
AFTER = '5ea1130b150506e7a7e94534ee4e9aedd6e57ef576bfc50ced6e00927d926612'
WRAPPER_SHA256 = '882d72b93c0adf8ab22ca11851d321b37bf782f5bd810f0bd0472b13108c63c3'
THRESHOLDS = [27900, 55800, 83700]
RELATIVE = Path('recovery/zero_block_guard_003')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def checked_path(value):
    path = Path(value)
    require(path.is_absolute() and '..' not in path.parts,
            'Recovery paths must be absolute and without traversal')
    return path


def verify_frozen(source):
    config = read(source / 'configs/university_training.json')
    require(digest(config) == CONFIG_DIGEST, 'Original configuration digest changed')
    require(len(config['source_sha256']) == 41, 'Original source count changed')
    for relative, expected in config['source_sha256'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts,
                'Original source path is invalid')
        require(sha256(source / path) == expected,
                'Original source checksum changed: ' + relative)
    require(config['qwen_observer']['runtime_source_sha256'][RUNTIME] == BEFORE,
            'Original FP8 checksum expectation changed')
    require(config['max_active_seconds'] == THRESHOLDS[-1], 'Original budget changed')
    require(sha256(source / 'orcd/university_qwen_h100.py') == WRAPPER_SHA256,
            'Original H100 wrapper changed')
    return config


def effective_config(config, frozen):
    """Return the only permitted observer configuration override, without mutation."""
    require(config == frozen and digest(config) == CONFIG_DIGEST,
            'Observer received a configuration other than the frozen configuration')
    result = copy.deepcopy(config)
    hashes = result['qwen_observer']['runtime_source_sha256']
    require(hashes[RUNTIME] == BEFORE, 'Unexpected original runtime checksum')
    hashes[RUNTIME] = AFTER
    reverted = copy.deepcopy(result)
    reverted['qwen_observer']['runtime_source_sha256'][RUNTIME] = BEFORE
    require(reverted == frozen and config == frozen,
            'Recovery changed more than the authorized runtime checksum')
    return result


def verify_recovery(source, run, config, manifest_path):
    require(source == SOURCE and run == RUN, 'Recovery is restricted to its approved run')
    require(manifest_path == SOURCE / RELATIVE / 'RECOVERY_MANIFEST.json',
            'Recovery manifest is outside the approved source release')
    expected_manifest = os.environ.get('QWEN_RUNTIME_RECOVERY_SHA256', '')
    require(re.fullmatch(r'[0-9a-f]{64}', expected_manifest) is not None,
            'Submission must pin QWEN_RUNTIME_RECOVERY_SHA256')
    require(sha256(manifest_path) == expected_manifest, 'Recovery manifest checksum changed')
    manifest = read(manifest_path)
    require(manifest.get('schema_version') == 1
            and manifest.get('kind') == 'qwen_fp8_zero_block_guard_003',
            'Unrecognized recovery manifest')
    require(manifest.get('source_root') == str(source)
            and manifest.get('run_dir') == str(run)
            and manifest.get('config_digest') == CONFIG_DIGEST
            and manifest.get('source_sha256') == config['source_sha256']
            and manifest.get('wrapper_sha256') == WRAPPER_SHA256,
            'Recovery manifest does not bind the frozen run and source')
    require(manifest.get('runtime') == {
        'path': RUNTIME, 'before_sha256': BEFORE, 'after_sha256': AFTER},
        'Recovery runtime transition differs from the approved guard')
    own_dir = source / RELATIVE
    require(Path(__file__).resolve() == own_dir / 'bootstrap.py',
            'Bootstrap must execute from its recorded recovery directory')
    files = manifest.get('recovery_files_sha256', {})
    require(set(files) == {'bootstrap.py', 'training.slurm'},
            'Recovery manifest must bind exactly its bootstrap and Slurm wrapper')
    for name, expected in files.items():
        require(sha256(own_dir / name) == expected, 'Recovery file changed: ' + name)
    authorization = manifest['authorization']
    authorization_path = checked_path(authorization['path'])
    require(authorization_path == own_dir / 'AUTHORIZATION.json'
            and sha256(authorization_path) == authorization['sha256'],
            'Human authorization checksum/path mismatch')
    approval = read(authorization_path)
    require(approval.get('authorization_source') == 'Direct human message in this chat: approve'
            and isinstance(approval.get('approved_at'), str)
            and bool(approval['approved_at'])
            and approval.get('runtime_before_sha256') == BEFORE
            and approval.get('runtime_after_sha256') == AFTER
            and approval.get('config_digest') == CONFIG_DIGEST
            and approval.get('cumulative_thresholds') == THRESHOLDS
            and approval.get('original_source_files_preserved') == 41
            and approval.get('production_retry_requires_validation_pass') is True,
            'Human authorization does not approve this exact runtime recovery')
    validation = manifest['validation']
    validation_path = checked_path(validation['path'])
    diagnostic_root = Path('/orcd/pool/003/kirilb/Steganography/diagnostics') / EXPERIMENT
    require(validation_path.name == 'DIAGNOSTIC.json'
            and diagnostic_root in validation_path.parents,
            'Recovery validation must be this run public isolated diagnostic')
    require(sha256(validation_path) == validation['sha256'],
            'Recovery GPU validation checksum changed')
    evidence = read(validation_path)
    require(evidence.get('status') == 'passed'
            and evidence.get('runtime_after_sha256') == AFTER,
            'Recovery GPU validation did not pass for the approved runtime')
    for key, expected in validation.get('required', {}).items():
        require(key in evidence and evidence[key] == expected,
                'Recovery GPU validation requirement failed: ' + key)
    require(sha256(RUNTIME) == AFTER, 'Installed runtime does not match validated guard')
    for name, expected in config['qwen_observer']['runtime_source_sha256'].items():
        require(sha256(name) == (AFTER if name == RUNTIME else expected),
                'A runtime source outside the approved transition changed: ' + name)
    return {
        'kind': 'qwen_fp8_zero_block_guard_003',
        'recovery_manifest_path': str(manifest_path),
        'recovery_manifest_sha256': expected_manifest,
        'authorization_path': str(authorization_path),
        'authorization_sha256': authorization['sha256'],
        'validation_path': str(validation_path),
        'validation_sha256': validation['sha256'],
        'config_digest': CONFIG_DIGEST,
        'original_source_files_verified': 41,
        'runtime_path': RUNTIME,
        'runtime_before_sha256': BEFORE,
        'runtime_after_sha256': AFTER,
        'override_scope': 'One observer runtime checksum expectation; no scientific configuration changes',
        'cumulative_thresholds': THRESHOLDS,
    }


def install_client_overlay(source, frozen, audit):
    sys.path.insert(0, str(source))
    from orcd import qwen_observer_client as client
    require(Path(client.__file__).resolve() == source / 'orcd/qwen_observer_client.py',
            'Observer client imported from another release')
    original_client = client.QwenObserverClient

    class AuthorizedQwenObserverClient(original_client):
        def __init__(self, config, run=None):
            require(run is not None and Path(run).resolve() == RUN,
                    'Observer override requested outside the authorized run')
            effective = effective_config(config, frozen)
            super().__init__(effective, run=run)
            try:
                runtime_audit = self.ready.get('runtime_audit', {})
                require(runtime_audit.get('runtime_source_sha256_verified')
                        == effective['qwen_observer']['runtime_source_sha256'],
                        'Observer did not attest the actual authorized runtime checksums')
                require(runtime_audit.get('expected_versions_verified')
                        == frozen['qwen_observer']['expected_versions'],
                        'Observer did not attest its original package versions')
                require(config == frozen and digest(config) == CONFIG_DIGEST,
                        'Observer startup mutated the frozen configuration')
                self.ready['runtime_authorization'] = copy.deepcopy(audit)
            except BaseException:
                self.close()
                raise

    client.QwenObserverClient = AuthorizedQwenObserverClient
    client.Observer = AuthorizedQwenObserverClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('training-source', 'config', 'run-dir'):
        parser.add_argument('--' + argument, required=True, type=Path)
    parser.add_argument('--segment', required=True, type=int, choices=(1, 2, 3))
    parser.add_argument('--recovery-manifest', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true',
                      help='Verify the complete production gate using CPU only; never launch models')
    mode.add_argument('--verify-local-frozen', action='store_true',
                      help='Verify local frozen source/config and pure mapping only; not a production gate')
    args = parser.parse_args()
    source, config_path, run = (getattr(args, name).resolve()
        for name in ('training_source', 'config', 'run_dir'))
    require(config_path == source / 'configs/university_training.json',
            'Configuration is outside the frozen release')
    config = verify_frozen(source)
    mapped = effective_config(config, config)
    require(mapped != config and digest(config) == CONFIG_DIGEST,
            'Effective observer configuration mapping failed')
    if args.verify_local_frozen:
        print(json.dumps({'status': 'passed', 'scope': 'local_frozen_source_and_mapping_only',
                          'source_files_verified': 41, 'config_digest': CONFIG_DIGEST,
                          'original_config_unchanged': True, 'gpu_validation': False}))
        return
    manifest = (args.recovery_manifest or source / RELATIVE / 'RECOVERY_MANIFEST.json').resolve()
    audit = verify_recovery(source, run, config, manifest)
    if args.verify_only:
        print(json.dumps({'status': 'passed', 'scope': 'production_recovery_gate', **audit}))
        return
    sys.dont_write_bytecode = True
    install_client_overlay(source, config, audit)
    wrapper_path = source / 'orcd/university_qwen_h100.py'
    spec = importlib.util.spec_from_file_location('_authorized_original_qwen_wrapper', wrapper_path)
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    sys.argv = [str(wrapper_path), '--training-source', str(source), '--config', str(config_path),
                '--run-dir', str(run), '--segment', str(args.segment)]
    wrapper.main()


if __name__ == '__main__':
    main()
