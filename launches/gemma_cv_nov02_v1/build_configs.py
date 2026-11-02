"""Create one config per unique method, observer and matched seed."""
import copy,hashlib,json
from pathlib import Path
source=Path(__file__).resolve().parent
def read(path):return json.loads(path.read_text())
def sha(path):
    with path.open('rb') as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
models=read(source/'observer_assets.json')
revision='457f2e15bf550c227ce6ad86e2ec108d3e42c106'
gemma=dict(repo='unsloth/gemma-2-2b-it',revision=revision,path='/orcd/pool/003/kirilb/Steganography/assets/university/cache/models--unsloth--gemma-2-2b-it/snapshots/'+revision)
gemma_hashes={p.name:sha(p) for p in Path(gemma['path']).iterdir() if p.is_file()}
assert gemma_hashes['model.safetensors']=='bf06a1e6cfe1610beb98a2975e5602e7fc108d902b3ff9dd62282d749c7a2394'
base=read(source/'template.json')
base['source_sha256']={f:sha(source/f) for f in list(base['source_sha256'])+['evaluation_measures.py','tests/test_ablation_gemma.py','dispatch.sh','build_configs.py','cross_score.py','plot_results.py']}
base['assets'].update(sender=gemma,receiver=gemma)
base['pair_model_change'].update(sender=gemma['repo'],receiver=gemma['repo'],model_files_sha256=gemma_hashes,prior_run=None)
base['protocol']['university_generation']['backend']='Gemma 2 IT native template; pinned Transformers 4.51.3'
base['protocol']['experiment']='November 2 Gemma receiver training and observer size sweep'
base['corrections']=[v for v in base['corrections'] if v['id']!='observer_score_position']
conditions=[('full',s) for s in ['0.8','2','4','9','27']]+[('r16','9'),('r32','9')]
for mode,size in conditions:
    for seed in range(42,47):
        d=copy.deepcopy(base);name=f'gemma_cv_nov02_{mode}_o{size}_s{seed}';d['name']=name;d['seed']=seed
        full=mode=='full';rank=None if full else int(mode[1:]);kind='full' if full else 'lora'
        gpu='H200' if size=='27' else 'L40S'
        d.update(receiver_parameter_training=kind,receiver_rank=rank,target_gpu=gpu,ablation=dict(plan_commit='fa52e0d9d2279e7dddeea934beff5ad94e9811fc',seed=seed,receiver_training=mode,observer_size=size))
        d['protocol']['initialization']=dict(global_seed=seed,author_ppo_seed=seed,description='Independent sender and receiver initialized from the pinned Gemma 2 IT checkpoint.')
        d['protocol']['training'].update(author_ppo_seed=seed,receiver_training_mode=mode,receiver_full_parameter_training=full)
        d['pair_model_change']['receiver_dtype']='BF16 base; FP32 CPU masters for all trainable weights'
        d['pair_model_change']['receiver_optimizer']='FP32 CPU AdamW on full receiver' if full else 'FP32 CPU AdamW on receiver q/v LoRA'
        d['pair_model_change']['checkpoint_storage']='Trainable sender, trainable receiver, optimizer, RNG, shuffle streams, PPO and progress; frozen bases hash-pinned'
        m=models[size]
        d['assets']['observer']=dict(repo=m['repo_id'],revision=m['revision'],path=m['path'])
        d['qwen_observer'].update(target_gpu=gpu,worker_path=str(source/'orcd/qwen_observer_worker.py'),model_path=m['path'],revision=m['revision'],artifact_sha256=m['artifact_sha256'])
        d['protocol']['observer'].update(model=m['repo_id'],revision=m['revision'],scope='Frozen official observer on the allocated GPU')
        path=source/'configs'/f'{name}.json';path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(d,indent=2)+'\n')
