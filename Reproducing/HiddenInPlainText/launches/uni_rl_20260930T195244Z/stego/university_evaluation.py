"""Resumable public evaluations of committed university-training checkpoints.

The caller owns model initialization, checkpointing, and training RNG/mode
restoration. This module never constructs an optimizer or changes parameters.
The modern PIQA judge runs in a separate Python process on the same allocation.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from .train import atomic_json
from .university_baseline import (
    digest, generate_hf, load_rows, normalize_label, piqa_messages, save_rows,
    sha256, uni_prompt, university_caps, validate_prefix,
)
from .university_metrics import render_baseline_report, summarize_baseline


def _read(path):
    return json.loads(Path(path).read_text())


def _write_once(path, value):
    path = Path(path)
    if path.exists():
        if _read(path) != value:
            raise ValueError(f"Evaluation provenance changed: {path.name}")
    else:
        atomic_json(path, value)


def _atomic_text(path, text):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.tmp-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _prefix(run, name, evaluation_digest, selected, identity, batch_size):
    rows = load_rows(run, name, evaluation_digest)
    validate_prefix(rows, selected, identity)
    if len(rows) != len(selected) and len(rows) % batch_size:
        raise ValueError(f"Incomplete committed evaluation batch: {name}")
    for row, original in zip(rows, selected):
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError(f"Committed evaluation inputs changed: {name}")
    return rows


def _checkpoint(trainer):
    checkpoint = Path(trainer.current_checkpoint)
    marker_path = checkpoint / 'COMPLETE.json'
    marker = _read(marker_path)
    recorded_update = marker.get('completed_update', marker.get('update', marker.get('completed_step')))
    if recorded_update != trainer.update or marker.get('config_digest') != trainer.config_digest:
        raise ValueError('Evaluation requires the current committed training checkpoint')
    return {'path': str(checkpoint), 'completed_update': trainer.update,
            'complete_sha256': sha256(marker_path)}


def _evaluation_config(trainer, checkpoint):
    config = copy.deepcopy(trainer.config)
    evaluation_seed = config.get('eval_seed', config.get('evaluation_seed'))
    if type(evaluation_seed) is not int or evaluation_seed < 0:
        raise ValueError('Evaluation requires a nonnegative eval_seed/evaluation_seed')
    if not config.get('judge_python'):
        raise ValueError('Evaluation requires the modern judge_python executable')
    # These fixed settings are shared with the completed step-zero baseline.
    config.update(seed=42, evaluation_seed=evaluation_seed, university_n=2000,
                  sender_batch_size=64, piqa_batch_size=16, think_tokens=256,
                  final_answer_tokens=128, temperature=0.6, generation_batch_size=16,
                  stage='training_evaluation', completed_update=trainer.update,
                  parameter_updates=trainer.update, training_config_digest=trainer.config_digest,
                  source_checkpoint=checkpoint)
    return config


def _selection(config):
    import pyarrow.parquet as pq

    dataset_file = config['assets']['dataset']['files']['test']
    if dataset_file.get('sha256') and sha256(dataset_file['path']) != dataset_file['sha256']:
        raise ValueError('University test data changed')
    university = [
        {'example_id': index, 'input': row['input'], 'true_label': normalize_label(row['label'])}
        for index, row in enumerate(pq.read_table(dataset_file['path']).to_pylist())
    ]
    piqa_payload = _read(config['piqa_selection'])
    piqa = piqa_payload['questions']
    if len(university) != 2000 or len(piqa) != 100:
        raise ValueError('Training evaluation requires the fixed 2000 university/100 PIQA cohort')
    if len({row['question_id'] for row in piqa}) != 100:
        raise ValueError('PIQA evaluation contains repeated question IDs')
    return university, piqa, piqa_payload


def _status(trainer, run, evaluation_digest, stage, **extra):
    atomic_json(run / 'STATUS.json', {
        'status': 'running', 'stage': stage, 'completed_update': trainer.update,
        'config_digest': trainer.config_digest, 'evaluation_config_digest': evaluation_digest,
        'updated_at': time.time(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **extra,
    })
    trainer.log_status('evaluation_' + stage, evaluation_update=trainer.update,
                       evaluation_dir=str(run), **extra)


def _boundary(trainer, run, evaluation_digest):
    if trainer.should_stop():
        atomic_json(run / 'STATUS.json', {
            'status': 'preempted', 'stage': 'between_committed_batches',
            'completed_update': trainer.update, 'config_digest': trainer.config_digest,
            'evaluation_config_digest': evaluation_digest, 'updated_at': time.time(),
        })
        raise SystemExit(75)


def _run_judge(trainer, config, run, evaluation_digest):
    _boundary(trainer, run, evaluation_digest)
    _status(trainer, run, evaluation_digest, 'piqa_judging')
    command = [config['judge_python'], '-m', 'stego.university_judge',
               '--config', str(run / 'config.json'), '--run-dir', str(run)]
    source = Path(__file__).resolve().parent.parent
    signal_sent = False
    with (run / 'PIQA_JUDGE.log').open('a') as log:
        child = subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                if trainer.should_stop() and not signal_sent:
                    try:
                        child.send_signal(signal.SIGUSR1)
                    except ProcessLookupError:
                        pass
                    signal_sent = True
                try:
                    code = child.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise
    if code == 75:
        atomic_json(run / 'STATUS.json', {
            'status': 'preempted', 'stage': 'piqa_judging', 'completed_update': trainer.update,
            'config_digest': trainer.config_digest, 'evaluation_config_digest': evaluation_digest,
            'updated_at': time.time(),
        })
        raise SystemExit(75)
    if code:
        raise RuntimeError(f'PIQA judge exited {code}; inspect {run / "PIQA_JUDGE.log"}')
    _boundary(trainer, run, evaluation_digest)


def _baseline_d(config):
    value = config.get('baseline_d')
    if value is None and config.get('baseline_summary'):
        value = _read(config['baseline_summary'])['piqa']['d']
    if type(value) not in (int, float) or not 0 <= value <= 1:
        raise ValueError('Evaluation requires baseline_d or baseline_summary for the measured PIQA baseline')
    return float(value)


def _report(summary):
    rendered = render_baseline_report({**summary, 'stage': 'pretraining_baseline',
                                      'completed_step': 0, 'parameter_updates': 0})
    update = summary['completed_update']
    rendered = rendered.replace('# University experiment: baseline',
                                f'# University experiment: evaluation after update {update}', 1)
    rendered = rendered.replace('Step zero, before any parameter updates.',
                                f'Frozen evaluation of committed training update {update}.', 1)
    rendered = rendered.replace('| Metric | Baseline |', f'| Metric | Update {update} |', 1)
    rendered = rendered.replace(
        'This step-zero evaluation establishes the quality baseline; '
        'it does not yet measure degradation from training. ',
        'University and PIQA prompts, cohorts, sampling seeds, and judge settings match the saved baseline. ', 1)
    rendered += (f"\nPIQA baseline d: {100 * summary['baseline_d']:.1f}%. "
                 f"Change from baseline: {summary['delta_d_percentage_points']:+.1f} percentage points.\n")
    return rendered


def evaluate_training(trainer):
    """Evaluate the caller's committed frozen models; return the public summary."""
    import torch

    if type(trainer.update) is not int or trainer.update < 1:
        raise ValueError('Training evaluation requires a positive committed update')
    checkpoint = _checkpoint(trainer)
    config = _evaluation_config(trainer, checkpoint)
    evaluation_digest = digest(config)
    run = Path(trainer.run) / 'evaluations' / f'step_{trainer.update:06d}'
    run.mkdir(parents=True, exist_ok=True)
    _write_once(run / 'config.json', config)
    completion = run / 'COMPLETE.json'
    if completion.exists():
        marker = _read(completion)
        if (marker.get('config_digest') != trainer.config_digest
                or marker.get('evaluation_config_digest') != evaluation_digest
                or marker.get('completed_update') != trainer.update
                or marker.get('source_checkpoint') != checkpoint
                or marker.get('summary_sha256') != sha256(run / 'SUMMARY.json')):
            raise ValueError('Completed evaluation provenance mismatch')
        summary = _read(run / 'SUMMARY.json')
        if (summary.get('stage') != 'training_evaluation'
                or summary.get('completed_update') != trainer.update
                or summary.get('config_digest') != trainer.config_digest):
            raise ValueError('Completed evaluation summary mismatch')
        return summary

    try:
        _boundary(trainer, run, evaluation_digest)
        university, piqa, piqa_payload = _selection(config)
        _write_once(run / 'SELECTION.json', {'config_digest': evaluation_digest,
                    'university': university, 'piqa': piqa_payload})
        baseline_d = _baseline_d(config)
        trainer.sender.eval()
        trainer.receiver.eval()
        versions = {
            'sender': {name: param._version for name, param in trainer.sender.named_parameters()},
            'receiver': {name: param._version for name, param in trainer.receiver.named_parameters()},
        }
        caps = university_caps(len(university), config['evaluation_seed'])
        generations = _prefix(run, 'UNIVERSITY_GENERATIONS.json', evaluation_digest,
                              university, 'example_id', 64)
        for start in range(len(generations), len(university), 64):
            _boundary(trainer, run, evaluation_digest)
            batch = university[start:start + 64]
            _status(trainer, run, evaluation_digest, 'university_generation',
                    completed=len(generations), total=len(university))
            prompts = [uni_prompt(row['input']) for row in batch]
            answers = generate_hf(trainer.sender.pretrained_model, trainer.tokenizer, prompts,
                                  seed=config['evaluation_seed'] + 10000 + start,
                                  cap=caps[start // 192], temperature=1.0)
            if len(answers) != len(batch):
                raise RuntimeError('University generation count mismatch')
            generations.extend({**row, **answer, 'prompt': prompt}
                               for row, answer, prompt in zip(batch, answers, prompts))
            save_rows(run, 'UNIVERSITY_GENERATIONS.json', generations, evaluation_digest)

        scored = _prefix(run, 'UNIVERSITY_RECEIVER.json', evaluation_digest,
                         generations, 'example_id', 64)
        for start in range(len(scored), len(generations), 64):
            _boundary(trainer, run, evaluation_digest)
            batch = generations[start:start + 64]
            _status(trainer, run, evaluation_digest, 'receiver_scoring',
                    completed=len(scored), total=len(generations))
            tokens = trainer.receiver_tokenizer([row['answer'] for row in batch],
                return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda:0')
            with torch.inference_mode():
                logits = trainer.receiver(**tokens).logits
                predictions = logits.argmax(dim=-1).cpu().tolist()
                probabilities = logits.softmax(dim=-1).cpu().tolist()
            scored.extend({**row, 'receiver_prediction': int(prediction),
                           'receiver_probabilities': probability}
                          for row, prediction, probability in zip(batch, predictions, probabilities))
            save_rows(run, 'UNIVERSITY_RECEIVER.json', scored, evaluation_digest)

        observed = _prefix(run, 'UNIVERSITY_RESULTS.json', evaluation_digest,
                           scored, 'example_id', 64)
        for start in range(len(observed), len(scored), 64):
            _boundary(trainer, run, evaluation_digest)
            batch = scored[start:start + 64]
            _status(trainer, run, evaluation_digest, 'observer_scoring',
                    completed=len(observed), total=len(scored))
            score = trainer.observer.score if hasattr(trainer.observer, 'score') else trainer.observer
            verdicts = score([row['answer'] for row in batch])
            if len(verdicts) != len(batch):
                raise RuntimeError('Observer verdict count mismatch')
            for row, verdict in zip(batch, verdicts):
                if type(verdict.get('observer_flag')) is not bool:
                    raise ValueError('Observer must return a boolean observer_flag')
                observed.append({**row, 'observer_score': float(verdict['observer_score']),
                                 'observer_flag': verdict['observer_flag'],
                                 'observer_batch_start': start, 'observer_batch_size': len(batch)})
            save_rows(run, 'UNIVERSITY_RESULTS.json', observed, evaluation_digest)

        piqa_answers = _prefix(run, 'PIQA_GENERATIONS.json', evaluation_digest,
                              piqa, 'question_id', 16)
        for start in range(len(piqa_answers), len(piqa), 16):
            _boundary(trainer, run, evaluation_digest)
            batch = piqa[start:start + 16]
            _status(trainer, run, evaluation_digest, 'piqa_generation',
                    completed=len(piqa_answers), total=len(piqa))
            prompts = [trainer.tokenizer.apply_chat_template(piqa_messages(row['question']),
                       tokenize=False, add_generation_prompt=True) for row in batch]
            answers = generate_hf(trainer.sender.pretrained_model, trainer.tokenizer, prompts,
                                  seed=42 + 20000 + start, cap=128, temperature=0.6,
                                  add_special_tokens=False)
            if len(answers) != len(batch):
                raise RuntimeError('PIQA generation count mismatch')
            piqa_answers.extend({**row, **answer, 'prompt': prompt}
                                for row, answer, prompt in zip(batch, answers, prompts))
            save_rows(run, 'PIQA_GENERATIONS.json', piqa_answers, evaluation_digest)

        for role in ('sender', 'receiver'):
            model = getattr(trainer, role)
            if versions[role] != {name: param._version for name, param in model.named_parameters()}:
                raise RuntimeError(f'{role} parameters changed during evaluation')
        _run_judge(trainer, config, run, evaluation_digest)
        judge = _read(run / 'PIQA_JUDGED.json')
        if (judge.get('config_digest') != evaluation_digest or judge.get('complete') is not True
                or judge.get('n') != 100 or judge.get('target_questions') != 100
                or judge.get('source_generations_sha256') != sha256(run / 'PIQA_GENERATIONS.json')):
            raise ValueError('PIQA judge completion evidence mismatch')
        validate_prefix(judge['rows'], piqa, 'question_id')
        if len(judge['rows']) != 100:
            raise ValueError('PIQA judgment cohort is incomplete')
        _boundary(trainer, run, evaluation_digest)
        summary = summarize_baseline(observed, judge['rows'])
        summary.update(stage='training_evaluation', completed_update=trainer.update,
                       completed_step=trainer.update, parameter_updates=trainer.update,
                       config_digest=trainer.config_digest, evaluation_config_digest=evaluation_digest,
                       source_checkpoint=checkpoint, baseline_d=baseline_d,
                       delta_d_percentage_points=100 * (summary['piqa']['d'] - baseline_d))
        atomic_json(run / 'SUMMARY.json', summary)
        _atomic_text(run / 'report.md', _report(summary))
        atomic_json(completion, {
            'status': 'complete', 'completed_update': trainer.update,
            'parameter_updates': trainer.update, 'config_digest': trainer.config_digest,
            'evaluation_config_digest': evaluation_digest, 'source_checkpoint': checkpoint,
            'summary_sha256': sha256(run / 'SUMMARY.json'), 'completed_at': time.time(),
        })
        atomic_json(run / 'STATUS.json', {
            'status': 'complete', 'stage': 'training_evaluation', 'completed_update': trainer.update,
            'config_digest': trainer.config_digest, 'evaluation_config_digest': evaluation_digest,
            'updated_at': time.time(),
        })
        trainer.log_status('evaluation_complete', evaluation_update=trainer.update,
                           evaluation_dir=str(run))
        return summary
    except Exception as exc:
        atomic_json(run / 'STATUS.json', {
            'status': 'failed', 'stage': 'training_evaluation', 'completed_update': trainer.update,
            'config_digest': trainer.config_digest, 'evaluation_config_digest': evaluation_digest,
            'error_type': type(exc).__name__, 'error': str(exc), 'updated_at': time.time(),
        })
        raise
