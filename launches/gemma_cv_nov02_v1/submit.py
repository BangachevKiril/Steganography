"""Submit the seven validated conditions and their final scoring tasks exactly once."""
import fcntl,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
log=Path('/orcd/pool/003/kirilb/Steganography/logs/gemma_cv_nov02_v1')
log.mkdir(parents=True,exist_ok=True)
receipt=root/'SUBMISSION.json'
lock=(root/'SUBMIT_LOCK').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
jobs=json.loads(receipt.read_text()) if receipt.exists() else {}
def submit(name,args,script):
    if name in jobs:return jobs[name]
    queued=subprocess.check_output(['squeue','-u','kirilb','-h','-n',name,'-o','%i'],text=True).strip()
    if queued:raise RuntimeError('Found unrecorded active job: '+name+' '+queued)
    cmd=['sbatch','--parsable','--account=mit_general','--job-name='+name,'--output='+str(log/'%x-%A_%a.out'),'--error='+str(log/'%x-%A_%a.err'),'--chdir='+str(root)]+args+script
    jid=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    if not jid.isdigit():raise ValueError(jid)
    jobs[name]=jid
    tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(receipt)
    return jid
conditions=[('full',s) for s in ('9','0.8','2','4','27')]+[('r16','9'),('r32','9')]
def resources(size):
    return ['--partition=mit_normal_gpu,mit_preemptable','--gres=gpu:'+('h200' if size=='27' else 'l40s')+':1','--cpus-per-task=8','--mem=256G','--time=06:00:00']
for mode,size in conditions:
    n='gcv2_'+mode+'_'+size
    gate=submit(n+'_v',resources(size)+['--no-requeue'],['dispatch.sh',mode,size,'validate'])
    submit(n+'_t',resources(size)+['--array=0-4','--requeue','--signal=B:USR1@900','--dependency=afterok:'+gate,'--kill-on-invalid-dep=yes'],['dispatch.sh',mode,size,'train'])
full=[jobs['gcv2_full_'+s+'_t'] for s in ('0.8','2','4','9','27')]
for size in ('0.8','2','4','9'):
    submit('gcv2_score_'+size,resources(size)+['--array=0-4','--dependency=aftercorr:'+':'.join(full),'--kill-on-invalid-dep=yes'],['dispatch.sh','full',size,'score'])
trains=[jobs['gcv2_'+m+'_'+s+'_t'] for m,s in conditions]
submit('gcv2_curves',['--partition=mit_normal','--cpus-per-task=1','--mem=4G','--time=00:30:00','--dependency=afterok:'+':'.join(trains),'--kill-on-invalid-dep=yes'],['--wrap','/orcd/pool/003/kirilb/Steganography/envs/uni_author_torch240/bin/python plot_results.py'])
# The user submit limit cannot accommodate the last five scoring tasks immediately.
# Release them from a small dependent CPU job after the full-SFT arrays complete.
submit('gcv2_release_score27',['--partition=mit_normal','--cpus-per-task=1','--mem=1G','--time=00:10:00','--dependency=afterok:'+':'.join(full),'--kill-on-invalid-dep=yes'],['--wrap','/orcd/scratch/orcd/003/kirilb/Steganography/envs/qwen3_pair_torch240_tf4513/bin/python submit.py --release-score27'])
import sys
if '--release-score27' in sys.argv:
    for size in ('0.8','2','4','9','27'):
        for seed in range(42,47):
            marker=Path(f'/orcd/scratch/orcd/003/kirilb/Steganography/runs/gemma_cv_nov02_full_o{size}_s{seed}/COMPLETE.json')
            assert json.loads(marker.read_text())['completed_update']==1400
    submit('gcv2_score_27',resources('27')+['--array=0-4'],['dispatch.sh','full','27','score'])
print(json.dumps(jobs))
