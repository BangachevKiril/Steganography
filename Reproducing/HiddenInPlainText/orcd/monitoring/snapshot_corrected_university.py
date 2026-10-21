"""Read-only public snapshot for the university run; never exports state.pt."""
import hashlib,json,pathlib,sys
run=pathlib.Path(sys.argv[1]);expected=sys.argv[2]
config=json.loads((run/'config.json').read_text())
if hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()!=expected:raise ValueError('Run config mismatch')
checkpoints=[]
for p in (run/'checkpoints').glob('step_*/COMPLETE.json'):
    marker=json.loads(p.read_text())
    if marker['config_digest']!=expected:raise ValueError('Checkpoint config mismatch')
    if (p.parent/'state.pt').is_file():checkpoints.append((marker['completed_update'],p,marker,p.read_text()))
if not checkpoints:
    files={}
    for name in ['config.json','STATUS.json','FAILURE.json','INITIALIZATION.json','CORRECTED_L40S_RUNTIME.json','L40S_MEMORY_OFFLOAD.json','L40S_EVALUATION_MEMORY.json']:
        p=run/name
        if p.is_file():files[name]=p.read_text()
    for p in (run/'runtime_events').glob('l40s_*.json'):
        if p.is_file():files[str(p.relative_to(run))]=p.read_text()
    print(json.dumps(dict(verified_update=None,checkpoint=None,state_sha256_verified=False,files=files)))
    sys.exit(0)
step,path,marker,marker_text=max(checkpoints)
h=hashlib.sha256()
with (path.parent/'state.pt').open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
if h.hexdigest()!=marker['state_sha256']:raise ValueError('Committed state hash mismatch')
files={str(path.relative_to(run)):marker_text}
for name in ['config.json','STATUS.json','FAILURE.json','smoke.json','INITIALIZATION.json','TIMING.json','COMPLETE.json','LATEST.json','PROMPTS.json','QUERY_TOKENIZATION.json','TRAIN_SPLIT.json','PPO_CONFIG.json','FIRST_RECEIVER_UPDATE.json','FIRST_SENDER_UPDATE.json','BUDGET_COMPLETE.json','PREEMPTED.json','CORRECTED_L40S_RUNTIME.json','L40S_MIGRATION.json','L40S_MEMORY_OFFLOAD.json','L40S_EVALUATION_MEMORY.json']:
    p=run/name
    if p.is_file():files[name]=p.read_text()
for p in (run/'runtime_events').glob('l40s_*.json'):
    if p.is_file():files[str(p.relative_to(run))]=p.read_text()
for _, checkpoint_path, _, checkpoint_text in checkpoints:
    files[str(checkpoint_path.relative_to(run))]=checkpoint_text
initial=run/'checkpoints/step_000000/COMPLETE.json'
if initial.is_file():files[str(initial.relative_to(run))]=initial.read_text()
rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines() if line.strip()] if (run/'metrics.jsonl').exists() else []
rows=[row for row in rows if row['update']<=step]
if [r['update'] for r in rows]!=list(range(1,step+1)):raise ValueError('Metrics incomplete')
files['metrics.jsonl']=''.join(json.dumps(r)+'\n' for r in rows)
for p in (run/'rollouts').glob('update_*.json'):
    if int(p.stem.split('_')[1])<=step:files[str(p.relative_to(run))]=p.read_text()
public_names={'config.json','SELECTION.json','STATUS.json','UNIVERSITY_GENERATIONS.json','UNIVERSITY_RECEIVER.json','UNIVERSITY_RESULTS.json','PIQA_GENERATIONS.json','PIQA_JUDGED.json','PIQA_JUDGE_STATUS.json','SUMMARY.json','COMPLETE.json','report.md'}
for directory in (run/'evaluations').glob('step_*'):
    if int(directory.name.split('_')[1])>step:continue
    for p in directory.iterdir():
        if p.is_file() and p.name in public_names:files[str(p.relative_to(run))]=p.read_text()
print(json.dumps(dict(verified_update=step,checkpoint=str(path.parent),state_sha256_verified=True,files=files)))
