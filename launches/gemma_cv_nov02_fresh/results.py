"""Score original frozen observers on the fresh final texts and render the seven curve sets."""
import csv
import fcntl
import io
import json
import os
import signal
import sys
from pathlib import Path
from statistics import mean

from finalize import read, require, sha
from submit import ATTEMPT, SIZES

FROZEN = Path('/orcd/pool/003/kirilb/Steganography/releases/gemma_cv_nov02_v1')
METRICS = ('receiver_accuracy', 'observer_accuracy', 'receiver_pass', 'observer_pass')
CROSS_METRICS = ('observer_accuracy', 'observer_pass', 'observer_auc')
stop = False


def request_stop(signum, frame):
    global stop
    stop = True


def name(mode, size, seed):
    return f'gemma_cv_nov02_{mode}_o{size}_s{seed}'


def atomic_text(path, content):
    # Called with the export directory locked. A retry replaces a partial temp.
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def publish_cross_scores(size, records):
    expected = {(trained, seed) for trained in SIZES for seed in range(42, 47)}
    require(size in SIZES and len(records) == len(expected)
            and {(row['training_observer'], row['seed']) for row in records} == expected
            and all(0 <= row[k] <= 1 for row in records for k in CROSS_METRICS),
            'Incomplete cross-score column')
    out = ATTEMPT / 'gemma_cv_nov02_results' / 'cross_scores'
    out.mkdir(parents=True, exist_ok=True)
    column = {'scoring_observer': size, 'cohort_size': 2000, 'per_seed': records}
    with (out / 'LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = out / f'observer_{size}.json'
        if path.exists():
            require(read(path) == column, 'Existing cross-score column differs')
        else:
            atomic_text(path, json.dumps(column, indent=2, allow_nan=False) + '\n')
        if not all((out / f'observer_{s}.json').is_file() for s in SIZES):
            return
        # One row per training/scoring pair. The JSON columns retain the five
        # original seed scores and identify each input set of final transmissions.
        columns = {s: read(out / f'observer_{s}.json')['per_seed'] for s in SIZES}
        table = io.StringIO()
        writer = csv.writer(table, lineterminator='\n')
        writer.writerow(['training_observer_B', 'scoring_observer_B', 'seeds',
                         *['mean_' + metric for metric in CROSS_METRICS]])
        for trained in SIZES:
            for scorer in SIZES:
                group = [r for r in columns[scorer] if r['training_observer'] == trained]
                require(sorted(r['seed'] for r in group) == list(range(42, 47)),
                        'Cross-score seed set differs')
                writer.writerow([trained, scorer, len(group), *[mean(r[k] for r in group)
                                                                for k in CROSS_METRICS]])
        atomic_text(out / 'condition_means.csv', table.getvalue())


def score(size):
    sys.path.insert(0, str(FROZEN))
    from evaluation_measures import measures
    from orcd.qwen_observer_client import QwenObserverClient
    from stego.train import atomic_json

    require(size in SIZES, 'Unknown scoring observer')
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    # Frozen model/prompt settings are identical across seeds.
    cfg = read(FROZEN / 'configs' / (name('full', size, 42) + '.json'))
    judge = QwenObserverClient(cfg)
    records = []
    try:
        for seed in range(42, 47):
            for trained in SIZES:
                run = ATTEMPT / 'runs' / name('full', trained, seed)
                manifest = read(run / 'final_weights/COMPLETE.json')
                marker = read(run / 'COMPLETE.json')
                folder = run / 'evaluations/step_001400'
                require(marker['evaluation_complete_sha256'] == sha(folder / 'COMPLETE.json'),
                        'Final evaluation marker changed')
                src = folder / 'UNIVERSITY_RESULTS.json'
                source_hash = sha(src)
                require(source_hash == manifest['evaluation_results_sha256']['step_001400'],
                        'Final evaluation text changed')
                source = read(src)
                require(source['config_digest'] == read(folder / 'COMPLETE.json')['evaluation_config_digest'],
                        'Final text config changed')
                rows = source['rows']
                key = dict(training_observer=trained, scoring_observer=cfg['assets']['observer'],
                           seed=seed, source_sha256=source_hash)
                out = run / 'final_cross_scoring' / f'observer_{size}.json'
                if out.exists():
                    done = read(out)
                    require(all(done[k] == v for k, v in key.items())
                            and [r['example_id'] for r in done['rows']] == list(range(2000)),
                            'Prior cross-score mismatched')
                    require(done['metrics'] == measures([{**r, **v} for r, v in zip(rows, done['rows'])]),
                            'Prior cross-score metrics differ')
                    metrics = done['metrics']
                else:
                    scored = []
                    for start in range(0, len(rows), 64):
                        if stop:
                            raise SystemExit(75)
                        batch = rows[start:start + 64]
                        verdicts = judge.score([r['answer'] for r in batch])
                        scored.extend({**r, **v} for r, v in zip(batch, verdicts))
                    metrics = measures(scored)
                    out.parent.mkdir(exist_ok=True)
                    fields = ('example_id', 'true_label', 'observer_prediction', 'observer_good_probability')
                    atomic_json(out, {**key, 'metrics': metrics,
                                      'rows': [{k: r[k] for k in fields} for r in scored]})
                records.append(dict(training_observer=trained, seed=seed, source_sha256=source_hash,
                                    **{k: metrics[k] for k in CROSS_METRICS}))
    finally:
        judge.close()
    publish_cross_scores(size, records)


def curves():
    # Use the same fixed steps and visible per-seed traces as the frozen v1 plot.
    out = ATTEMPT / 'gemma_cv_nov02_results'
    out.mkdir(exist_ok=True)
    steps = list(range(0, 1401, 50))
    colors = ['#316fa0', '#ba6131', '#277a68', '#a34f81', '#8163a3', '#151515']
    for mode, size in [('full', s) for s in SIZES] + [('r16', '9'), ('r32', '9')]:
        runs = []
        for seed in range(42, 47):
            run = ATTEMPT / 'runs' / name(mode, size, seed)
            require(read(run / 'final_weights/COMPLETE.json')['completed_update'] == 1400,
                    'Missing final learned parameters')
            runs.append([read(run / f'evaluations/step_{step:06d}/SUMMARY.json')['university']
                         for step in steps])
        svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="660" font-family="sans-serif">',
               '<rect width="100%" height="100%" fill="white"/>',
               f'<text x="500" y="32" text-anchor="middle" font-size="22">Gemma {mode} receiver · Qwen3.5-{size}B observer</text>']
        for index, metric in enumerate(METRICS):
            x, y, w, h = 65 + (index % 2) * 480, 92 + (index // 2) * 265, 385, 180
            svg.append(f'<text x="{x}" y="{y-15}">{metric.replace("_", " ").capitalize()}</text>')
            for tick in (0, .5, 1):
                yy = y + (1 - tick) * h
                svg.append(f'<path d="M{x} {yy}h{w}" stroke="#ddd"/><text x="{x-6}" y="{yy+4}" text-anchor="end">{tick:.0%}</text>')
            for tick in (0, 700, 1400):
                svg.append(f'<text x="{x+w*tick/1400}" y="{y+h+22}" text-anchor="middle">{tick}</text>')
            series = [[v[metric] for v in run] for run in runs]
            series.append([mean(v) for v in zip(*series)])
            for i, vals in enumerate(series):
                points = ' '.join(f'{x+w*s/1400:.2f},{y+h*(1-v):.2f}' for s, v in zip(steps, vals))
                svg.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="{3 if i==5 else 1.5}"/>')
        svg.append('<text x="500" y="602" text-anchor="middle">Total parameter updates</text>')
        for i, label in enumerate([f'Seed {s}' for s in range(42, 47)] + ['Mean']):
            svg.append(f'<text x="{155+i*126}" y="642" fill="{colors[i]}">{label}</text>')
        file = out / f'{mode}_o{size}.svg'
        tmp = file.with_suffix('.tmp')
        tmp.write_text('\n'.join(svg + ['</svg>']))
        tmp.replace(file)


if __name__ == '__main__':
    if sys.argv[1:] == ['curves']:
        curves()
    elif len(sys.argv) == 3 and sys.argv[1] == 'score':
        score(sys.argv[2])
    else:
        raise SystemExit('Expected curves or score OBSERVER_SIZE')
