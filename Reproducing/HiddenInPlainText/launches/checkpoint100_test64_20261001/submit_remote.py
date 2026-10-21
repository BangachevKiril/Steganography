"""Python 3.6-compatible single-job submission guard for the authorized evaluation."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

source = Path('/orcd/pool/003/kirilb/Steganography/releases/checkpoint100_test64_20261001')
config = json.loads((source / 'config.json').read_text())
run = Path(config['run_dir'])
run.mkdir(parents=True, exist_ok=True)
with (run / 'SUBMIT.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = run / 'SUBMISSION.json'
    if record.exists():
        print(record.read_text())
    else:
        assert not (Path(config['training_run']) / 'USER_STOP.json').exists()
        assert not (run / 'USER_STOP.json').exists()
        manifest = json.loads((run / 'MANIFEST.json').read_text())
        expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
        assert manifest['config_digest'] == expected
        for name, checksum in config['driver_sha256'].items():
            assert hashlib.sha256((source / name).read_bytes()).hexdigest() == checksum
        queue = subprocess.check_output(['squeue', '-u', 'kirilb', '-h', '-o', '%i|%j|%T'], universal_newlines=True)
        assert not any('|piqa-test64-c100|' in row for row in queue.splitlines()), queue
        job = subprocess.check_output(['sbatch', '--parsable', '--output='+str(run / 'slurm-%j.out'),
            '--error='+str(run / 'slurm-%j.err'), str(source / 'evaluate.slurm'), str(source)],
            universal_newlines=True).strip().split(';')[0]
        assert job.isdigit(), job
        value = {'job_id': job, 'run_dir': str(run), 'source': str(source),
            'config_digest': expected, 'submitted_at': datetime.now(timezone.utc).isoformat(),
            'pre_submission_queue': queue, 'inference_only': True}
        with record.open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        print(json.dumps(value))
