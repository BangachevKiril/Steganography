"""Read the existing option-A job and copy only explicitly allowed public files."""
from pathlib import Path
import datetime
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no', '-o', 'ProxyCommand=false', 'orcd-login']

REMOTE = r'''
import pathlib, json, hashlib, re, subprocess
root=pathlib.Path(RUN)
config=json.loads((root/'config.json').read_text())
assert hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==DIGEST
commits=[]
for path in sorted((root/'checkpoints').glob('step_*')):
 if not re.match(r'^step_[0-9]{6}$',path.name): continue
 marker=path/'COMPLETE.json'
 state=path/'state.pt'
 if not marker.is_file() or not state.is_file() or state.stat().st_size==0: continue
 value=json.loads(marker.read_text())
 if value.get('config_digest')!=DIGEST: continue
 assert value['global_step']==int(path.name.split('_')[1])
 commits.append({'step':value['global_step'],'manifest':str(marker.relative_to(root)),
                 'state_bytes':state.stat().st_size})
verified=max([c['step'] for c in commits]) if commits else None
names=['config.json','PROMPTS.json','INITIALIZATION.json','smoke.json','STATUS.json',
       'TIMING.json','COMPLETE.json','LATEST.json']+[c['manifest'] for c in commits]
if verified is not None:
 for folder in ('metrics','validation'):
  for path in sorted((root/folder).glob('step_*.json')):
   match=re.match(r'^step_([0-9]{6})[.]json$',path.name)
   if match and int(match.group(1))<=verified:
    value=json.loads(path.read_text())
    assert value['global_step']==int(match.group(1))
    names.append(str(path.relative_to(root)))
files={name:(root/name).read_text() for name in names if (root/name).is_file()}
queue_result=subprocess.run(['squeue','-j',JOB,'-h','-o','%i|%T|%M|%R'],
 stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
if queue_result.returncode and 'Invalid job id specified' not in queue_result.stderr:
 raise RuntimeError(queue_result.stderr)
queue=queue_result.stdout.strip()
accounting=subprocess.check_output(['sacct','-j',JOB,'--noheader','--parsable2',
 '-o','JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).strip()
print(json.dumps({'config_digest':DIGEST,'verified_step':verified,'commits':commits,
 'queue':queue,'queue_stderr':queue_result.stderr.strip(),
 'accounting':accounting,'files':files}))
'''


def main():
    subprocess.run(['ssh', '-O', 'check', 'orcd-login'], check=True, timeout=10)
    record = json.loads((ROOT/'launches/hybrid_option_a_latest.json').read_text())
    assert record['experiment'] == 'piqa_qwen3_4b_hybrid_option_a_fresh_100'
    program = ('RUN='+repr(record['remote_run'])+'\nDIGEST='+repr(record['config_digest'])+
               '\nJOB='+repr(record['slurm_job_id'])+'\n'+REMOTE)
    remote = subprocess.run(SSH+['python3 -'], input=program, capture_output=True,
                            text=True, check=True, timeout=60)
    snapshot = json.loads(remote.stdout)
    assert snapshot['config_digest'] == record['config_digest']
    destination = Path(record['local_results'])
    for name, content in snapshot.pop('files').items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert relative.parts[0] in ('metrics','validation','checkpoints','config.json','PROMPTS.json',
            'INITIALIZATION.json','smoke.json','STATUS.json','TIMING.json','COMPLETE.json','LATEST.json')
        if relative.parts[0] == 'checkpoints':
            assert relative.name == 'COMPLETE.json'
        json.loads(content)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.json.sync-tmp')
        temporary.write_text(content)
        temporary.replace(target)
    snapshot['checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (destination/'MONITOR_SNAPSHOT.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print(json.dumps(snapshot, indent=2))


if __name__ == '__main__':
    main()
