EXPECTED = {'release': '20260930T194303Z', 'experiment': 'piqa_qwen3_4b_hybrid_option_a_fresh_100', 'archive': '/Users/kirilbangachev/Documents/ChatGPT/Steganography/launches/hybrid_option_a_20260930T194303Z.tar.gz', 'archive_sha256': '357c0822d3fe8fc10d25d4d742e134409b01cce5d5b35e9e8646c515fbd1b5fa', 'remote_source': '/orcd/pool/003/kirilb/Steganography/releases/hybrid_option_a_20260930T194303Z', 'remote_run': '/orcd/pool/003/kirilb/Steganography/runs/piqa_hybrid_option_a_fresh_20260930T194303Z', 'local_results': '/Users/kirilbangachev/Documents/ChatGPT/Steganography/results/piqa_hybrid_option_a_fresh_20260930T194303Z', 'config_digest': '9eee6eb77c005c7a37a052b425fe1faeb362758c433813115ade14e960659578', 'model': 'Qwen3-4B', 'total_steps': 100, 'state': 'prepared', 'initialization': 'fresh_base_and_adapters', 'previous_run_record': '/Users/kirilbangachev/Documents/ChatGPT/Steganography/launches/hybrid_decoder_20260930T151606Z.json', 'train_microbatch_size': 4, 'decoder_rollouts_per_answer': 1, 'sender_group_size': 8, 'receiver_prompt_variant': 'concise_plus_prior', 'interceptor_prompt_variant': 'original', 'receiver_system_prompt_sha256': '3e0531c128902b43024d33e196d272c7b3b82bddcbc96be6209a885619e3736b', 'receiver_system_prompt_by_identity_sha256': {'A': 'c996d6c6cf96954ff347ed3b29f2564ab90d59025c8bc36422835754749e4a7c', 'B': 'dc28685a160869b714bcb1bf43d5a029a0b7d76f5c21eb5ff9f903c1d57a638c'}, 'interceptor_system_prompt_sha256': '9b5c8065282a827ff1cdc1b2cc58431c5c3536f253dbda7b99696dacb4bd13b4', 'sender_system_prompt_sha256': '7691c2fe05b5662569253a54765856470ce823eb43231f237857c33c09a1f36e', 'decoder_objective': 'fixed_baseline_pg_ce', 'decoder_baseline': 0.5, 'decoder_ce_weight': 1.0, 'reported_training_phase_ends': [], 'reported_validation_steps': [], 'first_training_step_reported': False, 'initial_timing_reported': False, 'full_timing_reported': False, 'completion_reported': False, 'monitor_access_status': 'connected', 'auto_resume_allowed': True, 'created_at': '2026-09-30T19:43:03.213986+00:00', 'local_validation': {'unittest_tests_passed': 123, 'log': '/Users/kirilbangachev/Documents/ChatGPT/Steganography/launches/option_a_local_validation_20260930T194303Z.log'}, 'slurm_job_id': None}

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
