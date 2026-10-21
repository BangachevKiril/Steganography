"""Fetch public evaluation artifacts only and summarize the prespecified test."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RECORD = ROOT / 'launches/checkpoint100_test64_20261001.json'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def wilson(k, n):
    z = 1.959963984540054
    p = k/n
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [center-half, center+half]


def exact_coin_two_sided(k, n):
    if not n:
        return 1.0
    return min(1.0, 2*sum(math.comb(n, j) for j in range(min(k, n-k)+1))/2**n)


def main():
    check = subprocess.run(['ssh', '-O', 'check', 'orcd-login'], capture_output=True, text=True)
    if check.returncode:
        print(json.dumps({'access_blocked': True, 'detail': check.stderr.strip()}))
        return
    record = json.loads(RECORD.read_text())
    remote = record['run_dir']
    job = record['job_id']
    script = '''import json, subprocess
from pathlib import Path
run = Path(REMOTE)
names = ['evaluation_config.json','training_config.json','PROMPTS.json','selection.json',
         'MANIFEST.json','SUBMISSION.json','LOAD_VERIFICATION.json','STATUS.json','results.json','COMPLETE.json']
files = {n:(run/n).read_text() for n in names if (run/n).is_file()}
queue = subprocess.run(['squeue','-j',JOB,'-h','-o','%i|%T|%M|%R'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
account = subprocess.run(['sacct','-j',JOB,'--noheader','--parsable2','--format=JobID,State,ExitCode,Elapsed,NodeList'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
logs = {}
for suffix in ('out','err'):
    p = run/('slurm-'+JOB+'.'+suffix)
    if p.exists(): logs[suffix] = p.read_text()[-6000:]
print(json.dumps({'files': files,'queue':queue.stdout,'queue_error':queue.stderr,'accounting':account.stdout,'logs':logs}))
'''.replace('REMOTE', repr(remote)).replace('JOB', repr(str(job)))
    fetched = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', '-o', 'ControlMaster=no',
        '-o', 'ProxyCommand=false', 'orcd-login', 'python3 -'], input=script, text=True)
    data = json.loads(fetched)
    local = Path(record['local_results'])
    local.mkdir(parents=True, exist_ok=True)
    for name, content in data.pop('files').items():
        (local/name).write_text(content)
    data['checked_at'] = datetime.now(timezone.utc).isoformat()
    (local/'MONITOR_SNAPSHOT.json').write_text(json.dumps(data, indent=2)+'\n')
    record['last_checked_at'] = data['checked_at']
    if (local/'STATUS.json').exists():
        record['state'] = json.loads((local/'STATUS.json').read_text())['status']
    else:
        record['state'] = 'queued' if 'PENDING' in data['queue'] else 'submitted'
    if not (local/'COMPLETE.json').exists():
        RECORD.write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps({'complete':False, 'state':record['state'], **data}))
        return
    complete = json.loads((local/'COMPLETE.json').read_text())
    manifest = json.loads((local/'MANIFEST.json').read_text())
    selection = json.loads((local/'selection.json').read_text())
    config = json.loads((local/'evaluation_config.json').read_text())
    results = json.loads((local/'results.json').read_text())
    assert complete['results_sha256'] == hashlib.sha256((local/'results.json').read_bytes()).hexdigest()
    assert complete['manifest_digest'] == results['manifest_digest'] == digest(manifest)
    assert manifest['config_digest'] == digest(config) == record['config_digest']
    assert manifest['selection_digest'] == digest(selection)
    rows = results['rows']
    assert len(rows) == results['n'] == complete['n'] == 64
    assert [(r['question_id'],r['sender'],r['bit'],r['public_string']) for r in rows] == [
        (e['question_id'],e['sender'],e['bit'],e['public_string']) for e in selection['episodes']]
    assert all(r['c_r']==int(r['receiver_guess']==r['bit']) and r['c_i']==int(r['interceptor_guess']==r['bit'])
               and r['flipped_receiver_correct']==1-r['c_r'] for r in rows)
    k = sum(r['c_r'] for r in rows)
    discord = Counter((r['c_r'],r['c_i']) for r in rows)
    interval = wilson(k,64)
    summary = {key:value for key,value in results.items() if key != 'rows'}
    summary.update({'receiver_wilson_95': interval, 'flipped_receiver_wilson_95': [1-interval[1],1-interval[0]],
        'receiver_two_sided_exact_fair_coin_p': exact_coin_two_sided(k,64),
        'receiver_interceptor_paired_counts': {str(key):value for key,value in sorted(discord.items())},
        'receiver_interceptor_exact_mcnemar_p': exact_coin_two_sided(discord[(1,0)],discord[(1,0)]+discord[(0,1)]),
        'mean_sender_reward': sum(r['sender_reward'] for r in rows)/64,
        'mean_flipped_sender_reward': sum(r['flipped_sender_reward'] for r in rows)/64,
        'conditional_probabilities': {role: {
            'mean_p0':sum(r[role+'_conditional_p0'] for r in rows)/64,
            'mean_p_true':sum(r[role+'_conditional_p0'] if r['bit']==0 else 1-r[role+'_conditional_p0'] for r in rows)/64,
            'cross_entropy':-sum(math.log(max(1e-15,r[role+'_conditional_p0'] if r['bit']==0 else 1-r[role+'_conditional_p0'])) for r in rows)/64}
            for role in ('receiver','interceptor')},
        'statistical_note':'Fair-coin p-value assumes independent fair-coin successes. Original/flip are paired complements, not independent samples. Prior32 validation remains separate. Probabilities are conditional on each sampled private reasoning trajectory.'})
    (local/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    record['state']='completed'
    record['completed_at']=complete['completed_at']
    record['verified_results_sha256']=complete['results_sha256']
    RECORD.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'complete':True, 'analysis':summary, 'accounting':data['accounting']}))


if __name__ == '__main__':
    main()
