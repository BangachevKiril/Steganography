"""Freeze, verify, and submit the user-authorized fresh option-A pilot once."""
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false', 'orcd-login']


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


REMOTE = r'''
import pathlib, json, subprocess, tarfile, hashlib, fcntl, datetime, os
source=pathlib.Path(EXPECTED['remote_source'])
run=pathlib.Path(EXPECTED['remote_run'])
run.mkdir(parents=True,exist_ok=True)
lock=(run/'SUBMIT.lock').open('a')
fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
record_path=run/'SUBMISSION.json'
if record_path.exists():
 record=json.loads(record_path.read_text())
 assert record['config_digest']==EXPECTED['config_digest']
 print(json.dumps(record))
 raise SystemExit(0)
if (run/'LATEST.json').exists() or (run/'COMPLETE.json').exists() or (run/'USER_STOP.json').exists():
 raise RuntimeError('Run already has artifacts; inspect before submitting')
archive=pathlib.Path(str(source)+'.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED['archive_sha256']
if not source.exists():
 source.mkdir()
 with tarfile.open(str(archive),'r:gz') as tf:
  for member in tf.getmembers():
   if not member.isfile() or member.name.startswith('/') or '..' in pathlib.PurePosixPath(member.name).parts:
    raise RuntimeError('Unexpected archive member')
  tf.extractall(str(source))
config=json.loads((source/'configs/piqa_hybrid_option_a.json').read_text())
assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==EXPECTED['config_digest']
assert config['initialization']=='fresh_base_and_adapters' and config['launch_ready'] is True
assert config['train_microbatch_size']==4 and config['decoder_rollouts_per_answer']==1
assert config['group_size']==8 and config['total_steps']==100
for name,digest in config['source_sha256'].items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest,name
for name,digest in config['data_sha256'].items():
 assert hashlib.sha256((pathlib.Path(config['data_dir'])/name).read_bytes()).hexdigest()==digest,name
for key in ['communicator_path','judge_path']:
 assert pathlib.Path(config[key]).is_dir()
runtime=pathlib.Path('/orcd/pool/003/kirilb/SherlockHolmes/ralif_with_explicit_reward/smollm_grpo_20260918/venv/bin/python')
assert runtime.is_file()
queue=subprocess.check_output(['squeue','-u','kirilb','-h','-o','%i|%j|%T'],universal_newlines=True)
if any('piqa-hybrid' in line for line in queue.splitlines()):
 raise RuntimeError('An active PIQA hybrid job already exists; inspect before submitting')
for name,relative in [('config.json','configs/piqa_hybrid_option_a.json'),('PROMPTS.json','configs/piqa_hybrid_option_a_prompts.json')]:
 content=(source/relative).read_bytes()
 target=run/name
 if target.exists():
  assert target.read_bytes()==content
 else:
  target.write_bytes(content)
for entry in source.rglob('*'):
 entry.chmod(0o555 if entry.is_dir() else 0o444)
source.chmod(0o555)
job=subprocess.check_output(['sbatch','--parsable','--output='+str(run/'slurm-%j.out'),
 '--error='+str(run/'slurm-%j.err'),str(source/'orcd/hybrid_option_a_run.slurm'),str(source),str(run)],
 universal_newlines=True).strip().split(';')[0]
if not job.isdigit(): raise RuntimeError('Unexpected sbatch response: '+job)
submission={'slurm_job_id':job,'remote_run':str(run),'config_digest':EXPECTED['config_digest'],
 'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preflight':'passed'}
with record_path.open('x') as stream:
 json.dump(submission,stream,indent=2)
 stream.write('\n')
print(json.dumps(submission))
'''


def main():
    subprocess.run(['ssh', '-O', 'check', 'orcd-login'], check=True, timeout=10)
    current = json.loads((ROOT / 'launches/latest.json').read_text())
    if current.get('experiment') == 'piqa_qwen3_4b_hybrid_option_a_fresh_100':
        raise RuntimeError('Option A already has a launch record; inspect it instead of submitting again')
    if current.get('state') != 'stopped_by_user' or not current.get('stop_verified'):
        raise RuntimeError('Expected the preceding hybrid pilot to be verified stopped')
    now = datetime.datetime.now(datetime.timezone.utc)
    stamp = now.strftime('%Y%m%dT%H%M%SZ')
    release_name = 'hybrid_option_a_' + stamp
    stage = ROOT / 'launches' / release_name
    stage.mkdir()
    files = [*ROOT.glob('stego/*.py'), *ROOT.glob('tests/test_*.py'),
             ROOT / 'README.md', ROOT / 'experiments/hybrid_option_a_rl.md',
             ROOT / 'experiments/single_decoder_minibatches.md', ROOT / 'experiments/hybrid_decoder_rl.md',
             ROOT / 'orcd/hybrid_option_a_run.slurm', ROOT / 'configs/piqa_hybrid_option_a_prompts.json']
    for path in files:
        target = stage / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    config_path = ROOT / 'configs/piqa_hybrid_option_a.json'
    config = json.loads(config_path.read_text())
    config['source_sha256'] = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(stage.rglob('*')) if p.is_file()}
    save(config_path, config)
    shutil.copy2(config_path, stage / 'configs/piqa_hybrid_option_a.json')
    tests = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                           cwd=stage, capture_output=True, text=True)
    test_log = ROOT / 'launches' / ('option_a_local_validation_' + stamp + '.log')
    test_log.write_text(tests.stdout + tests.stderr)
    if tests.returncode:
        raise RuntimeError('Frozen-source tests failed; see ' + str(test_log))
    count = int(re.search(r'Ran (\d+) tests', tests.stderr).group(1))
    archive = ROOT / 'launches' / (release_name + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as bundle:
        for path in sorted(stage.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                bundle.add(path, arcname=str(path.relative_to(stage)), recursive=False)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    remote_source = '/orcd/pool/003/kirilb/Steganography/releases/' + release_name
    remote_run = '/orcd/pool/003/kirilb/Steganography/runs/piqa_hybrid_option_a_fresh_' + stamp
    local_results = ROOT / 'results' / ('piqa_hybrid_option_a_fresh_' + stamp)
    local_results.mkdir()
    shutil.copy2(config_path, local_results / 'config.json')
    shutil.copy2(stage / 'configs/piqa_hybrid_option_a_prompts.json', local_results / 'PROMPTS.json')
    previous = ROOT / 'launches' / ('hybrid_decoder_' + current['release'] + '.json')
    if previous.exists():
        assert json.loads(previous.read_text()) == current
    else:
        save(previous, current)
    record = dict(release=stamp, experiment=config['experiment'], archive=str(archive),
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), remote_source=remote_source,
        remote_run=remote_run, local_results=str(local_results), config_digest=digest, model='Qwen3-4B',
        total_steps=100, state='prepared', initialization='fresh_base_and_adapters',
        previous_run_record=str(previous), train_microbatch_size=4, decoder_rollouts_per_answer=1,
        sender_group_size=8, receiver_prompt_variant='concise_plus_prior', interceptor_prompt_variant='original',
        receiver_system_prompt_sha256=config['receiver_system_prompt_sha256'],
        receiver_system_prompt_by_identity_sha256=config['receiver_system_prompt_by_identity_sha256'],
        interceptor_system_prompt_sha256=config['interceptor_system_prompt_sha256'],
        sender_system_prompt_sha256=config['sender_system_prompt_sha256'],
        decoder_objective=config['decoder_objective'], decoder_baseline=.5, decoder_ce_weight=1.0,
        reported_training_phase_ends=[], reported_validation_steps=[], first_training_step_reported=False,
        initial_timing_reported=False, full_timing_reported=False, completion_reported=False,
        monitor_access_status='connected', auto_resume_allowed=True, created_at=now.isoformat(),
        local_validation={'unittest_tests_passed':count,'log':str(test_log)}, slurm_job_id=None)
    launch_paths = [ROOT / 'launches' / name for name in
                    ('latest.json', 'hybrid_decoder_latest.json', 'hybrid_option_a_latest.json')]
    for path in launch_paths:
        save(path, record)
    print(json.dumps({'stage':'frozen_and_tested','tests_passed':count,'release':stamp,'config_digest':digest}), flush=True)
    upload_code = ('import sys,pathlib,hashlib; p=pathlib.Path(' + repr(remote_source + '.tar.gz') + '); '
                   'b=sys.stdin.buffer.read(); assert hashlib.sha256(b).hexdigest()==' + repr(record['archive_sha256']) + '; '
                   'assert not p.exists() or p.read_bytes()==b; '
                   'p.write_bytes(b) if not p.exists() else None')
    subprocess.run(SSH + ['python3 -c ' + shlex.quote(upload_code)], input=archive.read_bytes(), check=True, timeout=60)
    submission_program = 'EXPECTED = ' + repr(record) + '\n' + REMOTE
    submission_path = ROOT / 'launches' / ('submit_option_a_' + stamp + '.py')
    submission_path.write_text(submission_program)
    submitted = subprocess.run(SSH + ['python3 -'], input=submission_program, text=True,
                               capture_output=True, check=True, timeout=60)
    submission = json.loads(submitted.stdout)
    record.update(submission, state='submitted', submission_script=str(submission_path))
    for path in launch_paths:
        save(path, record)
    save(local_results / 'SUBMISSION.json', submission)
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
