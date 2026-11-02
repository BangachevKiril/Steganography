"""Run the frozen Gemma trainer with one on-disk restart for this fresh attempt."""
import argparse
import os
import re
import stat
import sys
import time
from pathlib import Path

from finalize import digest, read, require, sha
from submit import ATTEMPT

FROZEN = Path('/orcd/pool/003/kirilb/Steganography/releases/gemma_cv_nov02_v1')
RUN_NAME = re.compile(r'gemma_cv_nov02_(?:full_o(?:0\.8|2|4|9|27)|r(?:16|32)_o9)_s4[2-6]')
TEMP = re.compile(r'\.step_([0-9]{6})\.[1-9][0-9]*\.tmp')


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def own_file(path):
    mode = path.lstat()
    require(stat.S_ISREG(mode.st_mode) and mode.st_nlink == 1,
            'Unexpected checkpoint file: ' + str(path))


def partials_to_retire(root, update, config_digest):
    partials = []
    for path in sorted(root.glob('.step_*.tmp')):
        match = TEMP.fullmatch(path.name)
        require(match and int(match[1]) <= 1400 and path.resolve() == path and path.is_dir(),
                'Unrecognized temporary checkpoint path: ' + str(path))
        for file in path.iterdir():
            require(file.name in ('state.pt', 'COMPLETE.json')
                    or re.fullmatch(r'COMPLETE\.json\.tmp-[0-9a-f]{32}', file.name),
                    'Unexpected content in temporary checkpoint: ' + str(file))
            own_file(file)
        # A crash between writing COMPLETE and publishing the directory could
        # leave a higher, recoverable state. Keep it intact for reconciliation.
        if (path / 'COMPLETE.json').exists():
            if (path / 'state.pt').exists():
                require(int(match[1]) < update, 'Unpublished, potentially complete checkpoint: ' + str(path))
            marker = read(path / 'COMPLETE.json')
            require(marker['config_digest'] == config_digest and marker['completed_update'] == int(match[1]),
                    'Temporary checkpoint marker identity mismatch')
        partials.append(path)
    return partials


def retain(trainer, event, atomic_json):
    """Called exclusively inside the frozen trainer's held run LOCK."""
    run = Path(trainer.run)
    cfg = trainer.config
    require(RUN_NAME.fullmatch(cfg['name']) and run == ATTEMPT / 'runs' / cfg['name']
            and run == run.resolve() and read(run / 'config.json') == cfg
            and trainer.config_digest == digest(cfg), 'Retention run identity mismatch')
    root = run / 'checkpoints'
    keep = Path(trainer.current_checkpoint)
    step = trainer.update
    require(type(step) is int and 0 <= step <= 1400 and keep == root / f'step_{step:06d}'
            and keep == keep.resolve(), 'Unrecognized retained checkpoint')
    aged = []
    for folder in sorted(root.glob('step_*')):
        require(folder == folder.resolve() and folder.is_dir() and re.fullmatch(r'step_[0-9]{6}', folder.name),
                'Unrecognized checkpoint directory')
        state = folder / 'state.pt'
        if not state.exists() and not state.is_symlink():
            continue  # Metadata for retired checkpoints is permanently retained.
        own_file(state)
        own_file(folder / 'COMPLETE.json')
        marker = read(folder / 'COMPLETE.json')
        require(marker['config_digest'] == trainer.config_digest
                and marker['completed_update'] == int(folder.name[5:]) <= step,
                'Unexpected checkpoint configuration or update')
        if folder != keep:
            aged.append(state)
    own_file(keep / 'state.pt')
    current = read(keep / 'COMPLETE.json')
    require(current['state_sha256'] == sha(keep / 'state.pt'),
            'Retained restart checksum mismatch; keeping all checkpoint evidence')
    # The frozen writer fsyncs both files before publication. Persist the new
    # directory entry before allowing the only previous restart to disappear.
    for path in (keep, root, run):
        sync_dir(path)
    partials = partials_to_retire(root, step, trainer.config_digest)
    require(event == 'restore' or not partials, 'Partial writes require verified trainer restore')
    for directory in (run / 'retention_events', run / 'rolled_back_attempts'):
        require(directory == directory.resolve(), 'Unexpected audit or archive path')
    latest = run / 'LATEST.json'
    expected = dict(config_digest=trainer.config_digest, completed_update=step, checkpoint=str(keep))
    if not latest.exists() or read(latest) != expected:
        # A killed writer may have atomically published step_U before LATEST.
        require(event == 'restore', 'Unexpected latest checkpoint receipt after save')
        if latest.exists():
            previous = read(latest)
            old = previous['completed_update']
            require(previous['config_digest'] == trainer.config_digest and type(old) is int
                    and 0 <= old < step and previous['checkpoint'] == str(root / f'step_{old:06d}'),
                    'Unexpected latest checkpoint receipt before restore')
        atomic_json(latest, expected)
        sync_dir(run)
    policy = cfg['sparse_checkpoint_policy']
    effective = dict(config_digest=trainer.config_digest, frozen_retained_restart_states=policy['retained_restart_states'],
                     effective_retained_restart_states=1, overlay_sha256=sha(Path(__file__)),
                     interval_updates=policy['interval_updates'], forced_saves=policy['forced_saves'])
    atomic_json(run / 'CHECKPOINT_RETENTION.json', effective)
    namespace = f'{time.time_ns()}_{event}_{step:06d}'
    archive = run / 'rolled_back_attempts' / (namespace + '_partial_metadata')
    archive_targets = {folder: archive / folder.name for folder in partials
                       if any(f.name != 'state.pt' for f in folder.iterdir())}
    payloads = aged + [folder / 'state.pt' for folder in partials if (folder / 'state.pt').exists()]
    record = dict(**effective, event=event, retained_update=step, retained_sha256=current['state_sha256'],
                  retired_payloads=[str(p.relative_to(run)) for p in payloads],
                  archived_partial_metadata=[str(p.relative_to(run)) for p in archive_targets.values()])
    receipt = run / 'retention_events' / (namespace + '.json')
    atomic_json(receipt, dict(record, status='planned'))
    sync_dir(receipt.parent)
    sync_dir(run)
    for file in payloads:
        file.unlink()
        sync_dir(file.parent)
    for folder in partials:
        if folder in archive_targets:
            archive.mkdir(parents=True, exist_ok=True)
            sync_dir(archive.parent)
            sync_dir(run)
            folder.rename(archive_targets[folder])
            sync_dir(archive)
        else:
            folder.rmdir()
    sync_dir(root)
    atomic_json(receipt, dict(record, status='applied'))


def install(cycles, sparse):
    # qwen_pair_run.main installs the cycle trainer last, immediately before
    # opening run/LOCK. Hook its final checkpoint method including smoke checks.
    original_install = cycles.install_cycle_training

    def install_hooks(training):
        result = original_install(training)
        original_save = training.Trainer.checkpoint
        original_archive = sparse.archive_after_rollback

        def checkpoint(self):
            original_save(self)
            retain(self, 'save', training.atomic_json)

        def archive_after_restore(self):
            # The sparse restore has loaded weights/optimizer/RNG and called
            # publish_metrics. Inspect partials BEFORE its blanket temp cleanup.
            retain(self, 'restore', training.atomic_json)
            original_archive(self)

        training.Trainer.checkpoint = checkpoint
        sparse.archive_after_rollback = archive_after_restore
        return result

    cycles.install_cycle_training = install_hooks


def main():
    parser = argparse.ArgumentParser()
    for flag in ('config', 'run-dir', 'validation'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    name = args.config.stem
    require(RUN_NAME.fullmatch(name) and args.config == FROZEN / 'configs' / (name + '.json')
            and args.run_dir == ATTEMPT / 'runs' / name and args.run_dir == args.run_dir.resolve()
            and args.validation == ATTEMPT / 'diagnostics' / (name.rsplit('_s', 1)[0] + '_s42')
            / 'validation_001/VALIDATION.json', 'Unexpected fresh trainer paths')
    sys.path.insert(0, str(FROZEN))
    from orcd import qwen_cycle_training, qwen_pair_run, sparse_runtime
    install(qwen_cycle_training, sparse_runtime)
    qwen_pair_run.main()


if __name__ == '__main__':
    main()
