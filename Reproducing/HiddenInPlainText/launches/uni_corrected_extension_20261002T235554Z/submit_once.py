"""Single authorized continuation submission; never duplicate a possibly active job."""
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

base = Path('/orcd/pool/003/kirilb/Steganography')
extension_id = 'uni_corrected_extension_20261002T235554Z'
extension = base / 'releases' / extension_id
manifest = json.loads((extension / 'EXTENSION.json').read_text())
run, source = Path(manifest['run_dir']), Path(manifest['training_source'])
receipt = run / ('SUBMISSION_' + extension_id + '.json')

def write(value):
    temporary = receipt.with_suffix('.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(receipt))

with (run / 'SUBMISSION.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if not saved.get('job_id'):
            raise RuntimeError('Earlier dispatch has unknown outcome; reconcile Slurm before retry')
        print(json.dumps(saved, indent=2))
        raise SystemExit(0)
    jobs = subprocess.check_output(['squeue', '-u', getpass.getuser(), '-h', '-o', '%A'], universal_newlines=True).split()
    for job in sorted(set(jobs)):
        details = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], universal_newlines=True)
        if str(run) in details or str(source) in details or str(extension) in details or 'JobName=uni-corr-ext ' in details:
            raise RuntimeError('An active job may already own this run: ' + job)
    assert hashlib.sha256((extension / 'EXTENSION.json').read_bytes()).hexdigest() == '5503909537275e0f1027335e580f9842d7973cfc281ca4cec590dbb3c458b61e'
    subprocess.check_call(['sha256sum', '-c', 'SHA256SUMS'], cwd=str(extension), stdout=subprocess.DEVNULL)
    assert hashlib.sha256((run / 'COMPLETE.json').read_bytes()).hexdigest() == manifest['parent_complete_sha256']
    assert hashlib.sha256((Path(manifest['archive_dir']) / 'ARCHIVE_MANIFEST.json').read_bytes()).hexdigest() == manifest['archive_manifest_sha256']
    command = ['sbatch', '--parsable', '--output=' + str(run / 'slurm-%j.out'),
        '--error=' + str(run / 'slurm-%j.err'), '--comment=' + extension_id,
        str(extension / 'university_corrected_extension.slurm'), str(source), str(extension)]
    record = {'extension_id': extension_id, 'status': 'dispatching', 'submitted_at': time.time(),
        'run_dir': str(run), 'remote_extension': str(extension), 'command': command,
        'config_digest': manifest['config_digest'], 'origin_update': 530,
        'additional_active_seconds': 27900, 'target_active_seconds': manifest['target_active_seconds']}
    write(record)
    result = subprocess.check_output(command, universal_newlines=True).strip()
    job = result.split(';')[0]
    if not job.isdigit():
        raise RuntimeError('Unexpected sbatch result; reconcile before retry: ' + result)
    record.update(job_id=job, status='submitted', sbatch_result=result)
    write(record)
    print(json.dumps(record, indent=2))
