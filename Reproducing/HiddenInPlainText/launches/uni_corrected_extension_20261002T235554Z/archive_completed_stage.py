"""Cluster-side preservation of completed stage; no model data leaves the cluster."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

base = Path('/orcd/pool/003/kirilb/Steganography')
name = 'uni_corrected_20261002T151448Z'
run, source = base / 'runs' / name, base / 'releases' / name
archive = base / 'archives' / (name + '_stage1_update530')
expected_config = '3fe0ebcb50bb71fc727f8255161f45531c3dfe073eab644d2423cc41ce13fa92'

def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

with (run / 'LOCK').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = json.loads((run / 'config.json').read_text())
    assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == expected_config
    for relative, expected in config['source_sha256'].items():
        assert sha(source / relative) == expected, relative
    assert sha(run / 'COMPLETE.json') == 'c3f8a17513ce0ba9812b73306166542dfc571776c5205cd9cad0be6a71fa94b8'
    assert sha(run / 'evaluations/step_000530/COMPLETE.json') == '54e00b3734d7d4d5fc733597370d0f4424ba52d709f334db4c94f329eb06aff7'
    highest = max(int(p.parent.name.split('_')[1]) for p in (run / 'checkpoints').glob('step_*/COMPLETE.json'))
    assert highest == 530
    assert sha(run / 'checkpoints/step_000530/state.pt') == '14c2e9342d13b12f8992feb18ff30ecdbd9c57d27ee62a3436047771e52bf6b9'
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        record = json.loads((archive / 'ARCHIVE_MANIFEST.json').read_text())
        for relative, expected in record['sha256'].items():
            assert sha(archive / relative) == expected, relative
    else:
        temporary = Path(tempfile.mkdtemp(prefix=archive.name + '.tmp-', dir=str(archive.parent)))
        files = {}
        top = ['config.json', 'PROMPTS.json', 'QUERY_TOKENIZATION.json', 'TRAIN_SPLIT.json',
               'PPO_CONFIG.json', 'STATUS.json', 'smoke.json', 'INITIALIZATION.json', 'TIMING.json',
               'COMPLETE.json', 'BUDGET_COMPLETE.json', 'LATEST.json', 'PREEMPTED.json',
               'FIRST_RECEIVER_UPDATE.json', 'FIRST_SENDER_UPDATE.json', 'metrics.jsonl',
               'report.md', 'REPORT.json', 'per_update.csv', 'CORRECTED_L40S_RUNTIME.json',
               'L40S_MEMORY_OFFLOAD.json', 'L40S_EVALUATION_MEMORY.json', 'SUBMISSION.json']
        public = [run / item for item in top if (run / item).is_file()]
        public += list((run / 'runtime_events').glob('l40s_*.json'))
        public += list((run / 'rollouts').glob('update_*.json'))
        evaluation_names = {'config.json', 'SELECTION.json', 'STATUS.json', 'UNIVERSITY_GENERATIONS.json',
            'UNIVERSITY_RECEIVER.json', 'UNIVERSITY_RESULTS.json', 'PIQA_GENERATIONS.json',
            'PIQA_JUDGED.json', 'PIQA_JUDGE_STATUS.json', 'SUMMARY.json', 'COMPLETE.json', 'report.md'}
        for directory in (run / 'evaluations').glob('step_*'):
            public += [p for p in directory.iterdir() if p.is_file() and p.name in evaluation_names]
        public += list((run / 'checkpoints').glob('step_*/COMPLETE.json'))
        for path in public:
            relative = str(path.relative_to(run))
            destination = temporary / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(path), str(destination))
            files[relative] = sha(path)
            assert sha(destination) == files[relative]
        # Immutable checkpoints are hardlinked only inside the remote filesystem.
        # All preserved states survive later retention without doubling disk usage.
        for path in (run / 'checkpoints').glob('step_*/state.pt'):
            relative = str(path.relative_to(run))
            marker = json.loads((path.parent / 'COMPLETE.json').read_text())
            assert marker['config_digest'] == expected_config
            assert sha(path) == marker['state_sha256']
            os.link(str(path), str(temporary / relative))
            files[relative] = marker['state_sha256']
        record = {'kind': 'completed_corrected_stage_archive', 'completed_update': 530,
            'original_run': str(run), 'config_digest': expected_config,
            'state_storage': 'Remote immutable hardlinks; no checkpoint deserialization or modification.',
            'sha256': files}
        (temporary / 'ARCHIVE_MANIFEST.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        os.rename(str(temporary), str(archive))
    print(json.dumps({'archive_dir': str(archive), 'files_preserved': len(record['sha256']),
        'archive_manifest_sha256': sha(archive / 'ARCHIVE_MANIFEST.json'), 'highest_verified_update': 530,
        'source_files_verified': len(config['source_sha256']),
        'parent_budget_complete_sha256': sha(archive / 'BUDGET_COMPLETE.json')}, indent=2))
