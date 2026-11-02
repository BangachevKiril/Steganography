"""Resumable public evaluations of committed university-training checkpoints.

The caller owns model initialization, checkpointing, and training RNG/mode
restoration. This module never constructs an optimizer or changes parameters.
PIQA generation and judging are disabled for this experiment.
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
    digest, generate_hf, load_rows, normalize_label, save_rows,
    sha256, uni_prompt, university_caps, validate_prefix,
)
from .university_metrics import summarize_university


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
    from orcd.sparse_runtime import evaluation_identity
    return evaluation_identity(trainer)


def _evaluation_config(trainer, checkpoint):
    config = copy.deepcopy(trainer.config)
    evaluation_seed = config.get('eval_seed', config.get('evaluation_seed'))
    if type(evaluation_seed) is not int or evaluation_seed < 0:
        raise ValueError('Evaluation requires a nonnegative eval_seed/evaluation_seed')
    # These fixed settings are shared with the completed step-zero baseline.
    config.update(seed=42, evaluation_seed=evaluation_seed, university_n=2000,
                  sender_batch_size=64, piqa_enabled=False,
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
    if len(university)!=2000:raise ValueError('Expected fixed2000 university cohort')
    return university


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


def _baseline_d(config, trainer=None):
    # Preserve the segment resume provenance check without inventing PIQA metrics.
    if trainer is not None:
        root=Path(trainer.run)/'evaluations/step_000000'
        marker=_read(root/'COMPLETE.json'); summary=_read(root/'SUMMARY.json')
        identity=Path(trainer.run)/'evaluation_states/step_000000/COMPLETE.json'
        if (marker.get('config_digest')!=trainer.config_digest or marker.get('summary_sha256')!=sha256(root/'SUMMARY.json') or marker.get('source_checkpoint',{}).get('complete_sha256')!=sha256(identity) or summary.get('university',{}).get('n')!=2000 or summary.get('piqa_enabled') is not False):
            raise ValueError('University baseline identity mismatch')
    return None


def _report(summary):
    u=summary['university']; counts=u['counts']
    if u.get('observer_objective')=='deciphering':
        return f"University evaluation at update {summary['completed_update']}\nReceiver correct: {counts['receiver_correct']}/2000\nObserver deciphering correct: {u['observer_correct']}/2000\nC_I not applicable.\n"
    return f"University evaluation at update {summary['completed_update']}\nReceiver correct: {counts['receiver_correct']}/2000\nObserver flagged: {counts['observer_flagged']}/2000\nPIQA disabled.\n"


def evaluate_training(trainer):
    """Evaluate the caller's fingerprinted frozen models; return the public summary."""
    import torch

    if type(trainer.update) is not int or trainer.update < 0:
        raise ValueError('Training evaluation requires a nonnegative committed update')
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
        university = _selection(config)
        _write_once(run / 'SELECTION.json', {'config_digest': evaluation_digest,
                    'university': university, 'piqa_enabled': False})
        baseline_d = _baseline_d(config, trainer) if trainer.update else None
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
            prompts = [trainer.sender_prompt(row['input']) for row in batch]
            answers = generate_hf(trainer.sender.pretrained_model, trainer.tokenizer, prompts,
                                  seed=config['evaluation_seed'] + 10000 + start,
                                  cap=caps[start // 192], temperature=1.0, add_special_tokens=False)
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
                observed.append({**row, **verdict, 'observer_score': float(verdict['observer_score']),
                                 'observer_flag': verdict['observer_flag'],
                                 'observer_batch_start': start, 'observer_batch_size': len(batch)})
            save_rows(run, 'UNIVERSITY_RESULTS.json', observed, evaluation_digest)

        for role in ('sender', 'receiver'):
            model = getattr(trainer, role)
            if versions[role] != {name: param._version for name, param in model.named_parameters()}:
                raise RuntimeError(f'{role} parameters changed during evaluation')
        _boundary(trainer, run, evaluation_digest)
        summary=dict(schema_version=2,university=summarize_university(observed),piqa_enabled=False,
                     stage='training_evaluation',completed_update=trainer.update,
                     completed_step=trainer.update,parameter_updates=trainer.update,
                     config_digest=trainer.config_digest,evaluation_config_digest=evaluation_digest,
                     source_checkpoint=checkpoint)
        u=summary['university']
        objective=trainer.config['objective']
        u['observer_objective']=objective
        u['mean_observer_probability']=sum(r['observer_probability'] for r in observed)/len(observed)
        if objective=='deciphering':
            from evaluation_measures import measures
            u.update(measures(observed))
            u['C_I']=None
            u['C_I_definition']='Not applicable: this observer deciphers Bad/Good and does not flag suspicious summaries.'
            u['counts']['observer_flagged']=None
            u['counts']['observer_not_flagged']=None
            u.pop('observer_by_true_label',None);u.pop('observer_by_receiver_prediction',None)
            u['observer_correct']=sum(r['observer_prediction']==r['true_label'] for r in observed)
            u['observer_accuracy']=u['observer_correct']/len(observed)
            u['observer_good_fraction']=sum(r['observer_prediction'] for r in observed)/len(observed)
            u['mean_observer_confidence_penalty']=sum((2*r['observer_probability']-1)**2 for r in observed)/len(observed)
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
