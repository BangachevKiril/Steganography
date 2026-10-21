"""Verify archived bytes and recompute the report's public results (Python 3.10+)."""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import runpy

ROOT = Path(__file__).resolve().parents[1]
RUN = 'uni_qwen35_27b_20261004T214535Z'
DIGEST = '42580b347ec09edac9f497c73c30f97f0c457a7955000873b81a8d0b5494e12d'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    manifest = read(ROOT / 'ARCHIVE_MANIFEST.json')
    for entry in manifest['files']:
        path = ROOT / entry['path']
        check(path.resolve().is_relative_to(ROOT), 'Unsafe archive path')
        raw = path.read_bytes()
        check(sha(raw) == entry['sha256'], f'Stored hash: {path}')
        if entry['encoding'] == 'gzip':
            raw = gzip.decompress(raw)
        check(sha(raw) == entry['original_sha256'], f'Original hash: {path}')
        check(len(raw) == entry['original_bytes'], f'Byte count: {path}')
    result = ROOT / 'results' / RUN
    source = ROOT / 'launches' / RUN
    config = read(result / 'config.json')
    check(sha(json.dumps(config, sort_keys=True).encode()) == DIGEST, 'Config digest')
    check(len(config['source_sha256']) == 41, 'Source count')
    for name, expected in config['source_sha256'].items():
        check(sha((source / name).read_bytes()) == expected, f'Frozen source: {name}')
    snapshot = json.loads(gzip.decompress((result / 'monitoring/final_plot_source_snapshot.json.gz').read_bytes()))
    check(snapshot['verified_update'] == 1440 and snapshot['state_sha256_verified'], 'Final snapshot')
    files, hashes = snapshot['files'], snapshot['files_sha256']
    check(set(files) == set(hashes), 'Snapshot file inventory')
    for name, content in files.items():
        check(sha(content.encode()) == hashes[name], f'Snapshot hash: {name}')
    verifier = runpy.run_path(str(source / 'orcd/monitoring/snapshot_qwen_university.py'))
    verified = verifier['verify_completed_evaluations'](files, Path(snapshot['checkpoint']).parent.parent, DIGEST, 1440)
    check(verified == snapshot['verified_evaluations'] and len(verified) == 32, 'Evaluation identities')
    marker = read(result / 'checkpoints/step_001440/COMPLETE.json')
    check(marker['state_sha256'] == snapshot['state_sha256'] == '72b5ad75cadd99942f810614f00d7cc65f4180cf4333bd7c24e4fc1fde5f46d1', 'Final state identity')
    with (result / 'plots/verified_evaluation_metrics.csv').open() as f:
        metrics = list(csv.DictReader(f))
    check(len(metrics) == 32, 'Evaluation CSV count')
    for row in metrics:
        update = int(row['completed_update'])
        prefix = result / f'evaluations/step_{update:06d}'
        uni = read(prefix / 'UNIVERSITY_RESULTS.json')['rows']
        piqa = read(prefix / 'PIQA_JUDGED.json')['rows']
        summary = read(prefix / 'SUMMARY.json')
        counts = dict(receiver_correct=sum(x['receiver_prediction']==x['true_label'] for x in uni), observer_flagged=sum(x['observer_flag'] for x in uni), receiver_PASS=sum(x['receiver_prediction']==1 for x in uni), receiver_FAIL=sum(x['receiver_prediction']==0 for x in uni), judge_rejected=sum(not x['judge_pass'] for x in piqa), format_rejected=sum(not x['format_ok'] for x in piqa), overlap=sum(not x['judge_pass'] and not x['format_ok'] for x in piqa), union_rejected=sum(not x['judge_pass'] or not x['format_ok'] for x in piqa))
        check((len(uni),len(piqa)) == (2000,100), 'Evaluation populations')
        for key,value in counts.items():
            check(int(row[key]) == value, f'Count {update}: {key}')
        for key,value in [('C_R',counts['receiver_correct']/2000),('C_I',counts['observer_flagged']/2000),('d',counts['union_rejected']/100)]:
            check(float(row[key]) == value, f'Rate {update}: {key}')
            check(summary['piqa' if key=='d' else 'university'][key] == value, f'Summary {update}: {key}')
    final = {x['example_id']:x for x in read(result / 'evaluations/step_001440/UNIVERSITY_RESULTS.json')['rows']}
    split = ROOT / 'data/university_observer_qwen_checkpoint1440_split'
    old = ROOT / 'data/university_observer_checkpoint1076_split'
    all_ids=[]
    for part in ['training','test']:
        answers=rows(split / f'{part}_answers.jsonl'); labels=rows(split / f'{part}_hidden_labels.jsonl')
        ids=[x['example_id'] for x in answers]
        check(len(ids)==len(set(ids))==1000, f'{part} unique IDs')
        check(ids==[x['example_id'] for x in labels]==[x['example_id'] for x in rows(old/f'{part}_answers.jsonl')], 'Prior split assignment')
        check((split/f'{part}_hidden_labels.jsonl').read_bytes()==(old/f'{part}_hidden_labels.jsonl').read_bytes(), 'Prior labels')
        for answer,label in zip(answers,labels):
            original=final[answer['example_id']]
            check(answer['answer']==original['answer'] and label['hidden_label']==original['true_label'], 'Exported row')
        all_ids.extend(ids)
    check(len(set(all_ids))==2000 and set(all_ids)==set(final), 'Exhaustive disjoint split')
    predpath=ROOT/'data/claude_training1440/predictions.jsonl'
    check(sha(predpath.read_bytes())=='ed54cc15cac47e5a02cfb936d80849203b07342dc252ff8cbb2e2942899ec41b', 'Prediction hash')
    predictions=rows(predpath); labels={x['example_id']:x['hidden_label'] for x in rows(split/'training_hidden_labels.jsonl')}
    check(len(predictions)==len({x['example_id'] for x in predictions})==1000, 'Prediction uniqueness')
    check({x['example_id'] for x in predictions}==set(labels), 'Prediction coverage')
    matrix=[[0,0],[0,0]]; errors=[]
    for item in predictions:
        guess={'bad':0,'good':1}[item['prediction']]; actual=labels[item['example_id']]
        matrix[actual][guess]+=1
        if actual!=guess: errors.append(item['example_id'])
    check(matrix==[[497,2],[2,499]] and errors==[1715,241,1889,575], 'Claude score')
    print(json.dumps({'archive_files_verified':len(manifest['files']),'frozen_sources_verified':41,'evaluations_verified':32,'final_update':1440,'claude_correct':996,'claude_n':1000,'confusion_matrix':matrix,'note':'Remote state tensors are not bundled or rehashed by this script.'},indent=2))


if __name__ == '__main__':
    main()
