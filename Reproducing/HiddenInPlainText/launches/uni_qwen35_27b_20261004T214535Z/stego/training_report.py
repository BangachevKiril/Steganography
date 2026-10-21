"""Report public RL metrics, bounded by an independently audited checkpoint.

The required --completed-step is supplied after checking the remote checkpoint's
commit marker. LATEST.json and files left by an interrupted step are not authority.
No model, private trajectory, or tensor checkpoint is loaded by this utility.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics


COUNTS = {'C_R': 'correct_receiver', 'C_I': 'correct_interceptor', 'd': 'rejected_answers'}


def committed_rows(directory: Path, completed_step: int) -> list[dict]:
    rows = []
    for path in sorted(directory.glob('step_*.json')):
        match = re.fullmatch(r'step_(\d+)\.json', path.name)
        if not match or int(match[1]) > completed_step:
            continue
        row = json.loads(path.read_text())
        step = int(match[1])
        if row.get('global_step') != step:
            raise ValueError(f'Step identity mismatch: {path}')
        rows.append(row)
    if len({row['global_step'] for row in rows}) != len(rows):
        raise ValueError(f'Duplicate step files in {directory}')
    return sorted(rows, key=lambda row: row['global_step'])


def fraction(row: dict, metric: str) -> tuple[int, int] | None:
    """Return verified integer count/denominator, or an unmeasured value."""
    n = row.get('n')
    count = row.get(COUNTS[metric])
    if count is None and row.get(metric) is not None and n is not None:
        product = row[metric] * n
        if not math.isfinite(product) or not math.isclose(product, round(product), abs_tol=1e-7):
            raise ValueError(f'Cannot recover an integer count for {metric}: {row}')
        count = round(product)
    if count is None or n is None:
        return None
    if (isinstance(n, bool) or isinstance(count, bool) or int(n) != n or int(count) != count
            or n < 0 or count < 0 or count > n):
        raise ValueError(f'Invalid count for {metric}: {count}/{n}')
    if n == 0:
        return None
    if row.get(metric) is not None and not math.isclose(row[metric], count / n, abs_tol=1e-7):
        raise ValueError(f'Count/rate mismatch for {metric}: {row}')
    return int(count), int(n)


def measures(row: dict) -> dict:
    # Without a matched receiver/judge probe, top-level I accuracy describes
    # optimization guesses. Keep it exclusively in the optimization column.
    unmonitored = (row.get('phase') == 'interceptor'
                   and 'C_R' not in row and 'correct_receiver' not in row)
    values = {key: None if unmonitored else fraction(row, key) for key in COUNTS}
    training = row.get('interceptor_training', row if unmonitored else {})
    values['interceptor_training_C_I'] = fraction(training, 'C_I')
    return values


def aggregate(rows: list[dict]) -> dict:
    totals = {}
    values = [measures(row) for row in rows]
    for key in (*COUNTS, 'interceptor_training_C_I'):
        measured = [value[key] for value in values if value[key] is not None]
        totals[key] = ((sum(x[0] for x in measured), sum(x[1] for x in measured))
                       if measured else None)
    return totals


def show(value: tuple[int, int] | None) -> str:
    return 'not measured' if value is None else f'{value[0]}/{value[1]} ({100 * value[0] / value[1]:.1f}%)'


def decoder_rows(rows):
    """Keep optimization guesses distinct from matched monitoring."""
    result = []
    for row in rows:
        guesses = dict(row.get('decoder_training', {}))
        if 'decoder_guesses' in row.get('interceptor_training', {}):
            guesses['interceptor'] = row['interceptor_training']['decoder_guesses']
        for role, item in guesses.items():
            updates = [value for agent, value in row.get('updates', {}).items()
                       if (agent == 'I') == (role == 'interceptor')]
            examples = sum(value.get('decoder_examples', 0) for value in updates)
            def weighted(key):
                return (sum(value[key] * value['decoder_examples'] for value in updates)
                        / examples if examples and all(key in value for value in updates) else None)
            result.append({'global_step': row['global_step'], 'phase': row['phase'], 'role': role,
                           'rollouts_per_answer': row.get('decoder_rollouts_per_answer', ''),
                           'answer_contexts': row.get('decoder_answer_contexts', ''),
                           'n': item['n'], 'zero': item['zero'], 'correct': item['correct'],
                           **{f'bit{bit}_{key}': item['by_true_bit'][str(bit)][key]
                              for bit in (0, 1) for key in ('n', 'zero', 'correct')},
                           'ce': weighted('decoder_ce'),
                           'probability_zero': weighted('decoder_probability_zero_mean'),
                           'reasoning_pg_active_fraction': weighted('reasoning_pg_active_fraction')})
    return result


def render_report(run_dir: Path, completed_step: int) -> dict:
    if completed_step < 0:
        raise ValueError('completed_step must be nonnegative')
    rows = committed_rows(run_dir / 'metrics', completed_step)
    validation = committed_rows(run_dir / 'validation', completed_step)
    metric_names = [*COUNTS, 'interceptor_training_C_I']
    fields = ['global_step', 'phase', 'seconds']
    for metric in metric_names:
        fields.extend([metric + '_count', metric + '_n', metric + '_rate'])
    with (run_dir / 'training_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, '') for key in ('global_step', 'phase', 'seconds')}
            for metric, value in measures(row).items():
                output[metric + '_count'] = '' if value is None else value[0]
                output[metric + '_n'] = '' if value is None else value[1]
                output[metric + '_rate'] = '' if value is None else value[0] / value[1]
            writer.writerow(output)

    lines = [
        '# Training metrics', '',
        f'Only steps at or below the independently verified committed checkpoint **{completed_step}** are included.', '',
        '`C_R` is receiver accuracy, `C_I` is interceptor accuracy, and `d` is the fraction of answers rejected by the judge or format check. Counts are pooled within each ten-step phase; rates are weighted by their measured denominators.', '',
        'Joint steps measure all sender answers (normally 128 per step). Interceptor-only monitoring measures the same 16 answers with receiver, interceptor, and judge; interceptor optimization accuracy is labeled separately. Optimization uses the configured decoder rollout count: one guess per answer gives 16 interceptor-only guesses, while eight gives 128. These measurements are taken before the step’s updates. Blank CSV cells and “not measured” mean the metric was unavailable, never zero.', '',
        '| Steps | Phase | Steps observed | C_R | C_I | d | Interceptor optimization C_I |',
        '|---|---|---:|---|---|---|---|',
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[((row['global_step'] - 1) // 10, row['phase'])].append(row)
    blocks = []
    for (block, phase), members in sorted(grouped.items()):
        start, end = block * 10 + 1, min((block + 1) * 10, completed_step)
        totals = aggregate(members)
        block_record = {'start': start, 'end': end, 'phase': phase, 'observed_steps': len(members), **totals}
        blocks.append(block_record)
        suffix = ' (partial)' if end < (block + 1) * 10 else ''
        lines.append(f'| {start}–{end}{suffix} | {phase} | {len(members)}/{end-start+1} | '
                     + ' | '.join(show(totals[key]) for key in metric_names) + ' |')
    if not rows:
        lines.extend(['', 'No committed training metrics are available yet.'])
    missing = sorted(set(range(1, completed_step + 1)) - {row['global_step'] for row in rows})
    if missing:
        lines.extend(['', '**Missing committed-step metric files:** ' + ', '.join(map(str, missing)) + '. Available counts above exclude them.'])
    lines.extend(['', '## Validation', '',
                  'Validation uses a single rollout per sampled question; these results are kept separate from training.', '',
                  '| Step | C_R | C_I | d |', '|---:|---|---|---|'])
    for row in validation:
        lines.append(f"| {row['global_step']} | " + ' | '.join(show(fraction(row, key)) for key in COUNTS) + ' |')
    if not validation:
        lines.extend(['', 'No committed-step validation results are available yet.'])

    decoders = decoder_rows(rows)
    if decoders:
        with (run_dir / 'decoder_metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(decoders[0]))
            writer.writeheader()
            writer.writerows(decoders)
        lines.extend(['', '## Decoder optimization diagnostics', '',
                      'These diagnostics describe the guesses used for optimization. With single decoder rollouts, joint updates reuse one guess for each sender answer and interceptor-only updates use one guess for each of 16 answers. Legacy grouped sampling uses eight guesses per selected answer context. Counts are pooled within a phase. CE and conditional final-bit probabilities are measured before the update; the probability conditions on each sampled reasoning prefix. Reasoning PG activity is a nonzero-advantage fraction, not evidence of a useful learned code.', '',
                      '| Steps | Phase | Role | Zero guesses | Correct guesses | Correct, true bit 0 | Correct, true bit 1 | Mean CE |',
                      '|---|---|---|---|---|---|---|---|'])
        decoder_blocks = defaultdict(list)
        for row in decoders:
            decoder_blocks[((row['global_step'] - 1) // 10, row['phase'], row['role'])].append(row)
        for (block, phase, role), members in sorted(decoder_blocks.items()):
            total = lambda key: sum(row[key] for row in members)
            measured = [row for row in members if row['ce'] is not None]
            mean_ce = (sum(row['ce'] * row['n'] for row in measured) / sum(row['n'] for row in measured)
                       if measured else None)
            cells = [show((total(key), total('n'))) for key in ('zero', 'correct')]
            cells += [show((total(f'bit{bit}_correct'), total(f'bit{bit}_n'))) for bit in (0, 1)]
            cells += [f'{mean_ce:.4f}' if mean_ce is not None else 'not measured']
            lines.append(f'| {block * 10 + 1}–{min((block + 1) * 10, completed_step)} | {phase} | {role} | '
                         + ' | '.join(cells) + ' |')
        lines.extend(['', '[Per-step decoder diagnostics](decoder_metrics.csv)', ''])

    lines.extend(['', '## Timing', ''])
    durations = defaultdict(list)
    for row in rows:
        seconds = row.get('seconds')
        if isinstance(seconds, (int, float)) and math.isfinite(seconds) and seconds >= 0:
            durations[row['phase']].append(seconds)
    for phase, values in sorted(durations.items()):
        lines.append(f'- {phase}: median {statistics.median(values):.1f} seconds per step, measured over {len(values)} committed steps.')
    if not durations:
        lines.append('No committed-step timings are available yet.')
    timing_path = run_dir / 'TIMING.json'
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        if timing.get('completed_step') == completed_step:
            eta = timing.get('eta_seconds_excluding_queue')
            if eta is not None:
                lines.extend(['', f'Recorded remaining-time estimate: {eta / 3600:.2f} hours. ' + timing.get('note', '')])
        else:
            lines.extend(['', 'TIMING.json does not match the audited checkpoint, so its remaining-time estimate is not shown.'])
    lines.extend(['', '[Per-step counts and rates](training_metrics.csv)', ''])
    (run_dir / 'report.md').write_text('\n'.join(lines))
    return {'completed_step': completed_step, 'observed_steps': len(rows), 'blocks': blocks,
            'missing_steps': missing, 'validation_steps': [row['global_step'] for row in validation]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--completed-step', type=int, required=True,
                        help='Highest committed step independently verified from the remote checkpoint')
    args = parser.parse_args()
    summary = render_report(args.run_dir, args.completed_step)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
