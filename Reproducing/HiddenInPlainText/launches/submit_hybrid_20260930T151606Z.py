EXPECTED = {'release': '20260930T151606Z', 'experiment': 'piqa_qwen3_4b_hybrid_decoder_fresh_100', 'archive': '/Users/kirilbangachev/Documents/ChatGPT/Steganography/launches/hybrid_decoder_20260930T151606Z.tar.gz', 'archive_sha256': 'd44d7754856add1e3e5f3296e011a3dcd6ae6272188102d418f6c8ecf9a06523', 'remote_source': '/orcd/pool/003/kirilb/Steganography/releases/hybrid_decoder_20260930T151606Z', 'remote_run': '/orcd/pool/003/kirilb/Steganography/runs/piqa_hybrid_decoder_fresh_20260930T151606Z', 'local_results': '/Users/kirilbangachev/Documents/ChatGPT/Steganography/results/piqa_hybrid_decoder_fresh_20260930T151606Z', 'config_digest': '027cff0d41872b1094b5e1f70b02b00bdf03774692fa6f936c87140acfb15911', 'model': 'Qwen3-4B', 'total_steps': 100, 'state': 'prepared', 'initialization': 'fresh_base_and_adapters', 'previous_run_record': '/Users/kirilbangachev/Documents/ChatGPT/Steganography/launches/revised_pilot_20260929T173207Z.json', 'decoder_objective': 'fixed_baseline_pg_ce', 'decoder_baseline': 0.5, 'decoder_ce_weight': 1.0, 'sender_system_prompt_sha256': '7691c2fe05b5662569253a54765856470ce823eb43231f237857c33c09a1f36e', 'reported_training_phase_ends': [], 'reported_validation_steps': [], 'first_training_step_reported': False, 'initial_timing_reported': False, 'full_timing_reported': False, 'completion_reported': False, 'monitor_access_status': 'connected', 'created_at': '2026-09-30T15:16:06.305144+00:00', 'local_validation': {'unittest_tests_passed': 77, 'all_wrong_decoder_gradient_regression': 'passed'}, 'slurm_job_id': None}

import pathlib, json, subprocess, tarfile, hashlib, fcntl, datetime
source=pathlib.Path(EXPECTED['remote_source'])
run=pathlib.Path(EXPECTED['remote_run'])
run.mkdir(parents=True,exist_ok=True)
lock=(run/'SUBMIT.lock').open('a')
fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
record_path=run/'SUBMISSION.json'
if record_path.exists():
 print(record_path.read_text())
 raise SystemExit(0)
if (run/'LATEST.json').exists() or (run/'COMPLETE.json').exists():
 raise RuntimeError('Run already has training artifacts but no submission record; inspect before submitting')
archive=pathlib.Path(str(source)+'.tar.gz')
assert hashlib.sha256(archive.read_bytes()).hexdigest()==EXPECTED['archive_sha256']
if not source.exists():
 source.mkdir()
 with tarfile.open(str(archive),'r:gz') as tf:
  for member in tf.getmembers():
   if not member.isfile() or member.name.startswith('/') or '..' in pathlib.PurePosixPath(member.name).parts:
    raise RuntimeError('Unexpected archive member')
  tf.extractall(str(source))
config_path=source/'configs/piqa_hybrid_pilot.json'
config=json.loads(config_path.read_text())
assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==EXPECTED['config_digest']
for name,digest in config['source_sha256'].items():
 assert hashlib.sha256((source/name).read_bytes()).hexdigest()==digest, name
for name,digest in config['data_sha256'].items():
 assert hashlib.sha256((pathlib.Path(config['data_dir'])/name).read_bytes()).hexdigest()==digest, name
for key in ['communicator_path','judge_path']:
 assert pathlib.Path(config[key]).is_dir()
assert pathlib.Path('/orcd/pool/003/kirilb/SherlockHolmes/ralif_with_explicit_reward/smollm_grpo_20260918/venv/bin/python').is_file()
queue=subprocess.check_output(['squeue','-u','kirilb','-h','-o','%i|%j|%T'],universal_newlines=True)
if any('piqa-hybrid' in line for line in queue.splitlines()):
 raise RuntimeError('An active piqa-hybrid job already exists; inspect before submitting')
job=subprocess.check_output(['sbatch','--parsable','--output='+str(run/'slurm-%j.out'),'--error='+str(run/'slurm-%j.err'),str(source/'orcd/hybrid_run.slurm'),str(source),str(run)],universal_newlines=True).strip().split(';')[0]
if not job.isdigit(): raise RuntimeError('Unexpected sbatch response: '+job)
submission={'slurm_job_id':job,'remote_run':str(run),'config_digest':EXPECTED['config_digest'],'submitted_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
with record_path.open('x') as stream:
 json.dump(submission,stream,indent=2)
 stream.write('\n')
print(json.dumps(submission))
