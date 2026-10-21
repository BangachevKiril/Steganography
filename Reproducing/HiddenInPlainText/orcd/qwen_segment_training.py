"""Administrative boundaries for three continuous 27,900-active-second stages.

Install before training.main(); that caller owns the exclusive run lock.  No
optimizer, reward, sampling, or filtering implementation is replaced. Segment
boundary checkpoints must be retained by the caller's checkpoint-retention hook.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time


SEGMENT_SECONDS = 27900
TOTAL_SECONDS = 3 * SEGMENT_SECONDS
CLOCK_NAME = 'SEGMENT_ACTIVE_CLOCK.json'


class _CollectionBudgetReached(Exception):
    pass


def install_segment_training(training, segment):
    """Install train() for one stage; config remains identical across all stages."""
    if type(segment) is not int or segment not in (1, 2, 3):
        raise ValueError('segment must be an integer in 1..3')
    threshold = segment * SEGMENT_SECONDS

    def read(path):
        return json.loads(Path(path).read_text())

    def checkpoint_identity(self, path=None, verify_state=False):
        path = Path(path or self.current_checkpoint).resolve()
        marker_path = path / 'COMPLETE.json'
        marker = read(marker_path)
        if (marker.get('config_digest') != self.config_digest
                or type(marker.get('completed_update')) is not int
                or marker['completed_update'] < 0
                or not math.isfinite(marker.get('elapsed', float('nan')))
                or marker['elapsed'] < 0
                or path != (Path(self.run) / 'checkpoints' /
                            f"step_{marker['completed_update']:06d}").resolve()):
            raise ValueError('Invalid segment checkpoint identity')
        if verify_state and training.sha256(path / 'state.pt') != marker['state_sha256']:
            raise ValueError('Segment checkpoint state checksum mismatch')
        return dict(path=str(path), completed_update=marker['completed_update'],
                    complete_sha256=training.sha256(marker_path),
                    state_sha256=marker['state_sha256'], elapsed=marker['elapsed'])

    def check_completion(self, number):
        path = Path(self.run) / f'SEGMENT_{number:03d}_COMPLETE.json'
        payload = read(path)
        identity = checkpoint_identity(self, payload['checkpoint']['path'])
        evaluation = Path(self.run) / 'evaluations' / f"step_{identity['completed_update']:06d}" / 'COMPLETE.json'
        evaluation_marker = read(evaluation)
        if (payload.get('config_digest') != self.config_digest
                or payload.get('segment') != number
                or payload.get('active_threshold_seconds') != number * SEGMENT_SECONDS
                or payload.get('total_active_budget_seconds') != TOTAL_SECONDS
                or payload.get('checkpoint') != identity
                or payload.get('completed_update') != identity['completed_update']
                or payload.get('evaluation_complete_sha256') != training.sha256(evaluation)
                or evaluation_marker.get('config_digest') != self.config_digest
                or evaluation_marker.get('completed_update') != identity['completed_update']
                or evaluation_marker.get('source_checkpoint') != {
                    k: identity[k] for k in ('path', 'completed_update', 'complete_sha256')}
                or not math.isfinite(payload.get('active_seconds', float('nan')))
                or payload['active_seconds'] < number * SEGMENT_SECONDS):
            raise ValueError('Segment completion provenance mismatch')
        return payload

    def persist_clock(self):
        identity = checkpoint_identity(self)
        seconds = self.active_seconds()
        if not math.isfinite(seconds) or seconds < identity['elapsed']:
            raise ValueError('Invalid cumulative active time')
        payload = dict(schema_version=1, config_digest=self.config_digest,
                       segment=segment, active_threshold_seconds=threshold,
                       total_active_budget_seconds=TOTAL_SECONDS,
                       checkpoint=identity, active_seconds=seconds,
                       recorded_at=time.time())
        training.atomic_json(Path(self.run) / CLOCK_NAME, payload)
        return seconds

    def restore_clock(self, identity):
        path = Path(self.run) / CLOCK_NAME
        if not path.exists():
            return
        payload = read(path)
        recorded = payload.get('checkpoint', {})
        seconds = payload.get('active_seconds', float('nan'))
        if (payload.get('config_digest') != self.config_digest
                or payload.get('total_active_budget_seconds') != TOTAL_SECONDS
                or type(payload.get('segment')) is not int
                or not 1 <= payload['segment'] <= segment
                or payload.get('active_threshold_seconds') != payload['segment'] * SEGMENT_SECONDS
                or not math.isfinite(seconds) or seconds < 0
                or type(recorded.get('completed_update')) is not int
                or recorded['completed_update'] > identity['completed_update']):
            raise ValueError('Invalid active clock provenance')
        if recorded['completed_update'] == identity['completed_update']:
            if recorded != identity or seconds < identity['elapsed']:
                raise ValueError('Active clock checkpoint binding mismatch')
            # Preserve time already spent since this process restored the state.
            self.elapsed = max(self.elapsed, seconds)
        elif seconds > identity['elapsed']:
            # An older clock can be superseded only by a checkpoint that already
            # incorporates all its time; no unverified older clock is applied.
            raise ValueError('New checkpoint lost previously recorded active time')

    def evaluate(self):
        try:
            self.evaluate()
        finally:
            # Includes partial evaluations stopped at their resumable boundaries.
            persist_clock(self)

    def finalize_run(self, payload):
        run = Path(self.run)
        values = {
            'BUDGET_COMPLETE.json': read(run / 'SEGMENT_003_BUDGET_COMPLETE.json'),
            'COMPLETE.json': dict(payload, checkpoint=payload['checkpoint']['path'],
                                  stop_reason='configured_active_time_budget'),
        }
        for name, value in values.items():
            path = run / name
            if path.exists() and read(path) != value:
                raise ValueError('Terminal marker changed: ' + name)
            if not path.exists():
                training.atomic_json(path, value)
        self.log_status('complete')

    def train(self):
        run = Path(self.run)
        if (self.config.get('max_active_seconds') != TOTAL_SECONDS
                or self.config.get('evaluate_initial_checkpoint') is not True):
            raise ValueError('Segments require the fixed 83,700-second configuration and evaluation0')
        own_complete = run / f'SEGMENT_{segment:03d}_COMPLETE.json'
        if own_complete.exists():
            payload = check_completion(self, segment)
            if segment == 3:
                finalize_run(self, payload)
            return  # Never remove or rewrite markers from this or newer stages.
        for previous in range(1, segment):
            check_completion(self, previous)
        if any((run / f'SEGMENT_{later:03d}_COMPLETE.json').exists()
               for later in range(segment + 1, 4)):
            raise ValueError('Newer segment completed without this prerequisite')
        if (run / 'COMPLETE.json').exists():
            raise ValueError('Terminal completion exists without this segment completion')
        identity = checkpoint_identity(self, verify_state=True)
        if identity['completed_update'] != self.update:
            raise ValueError('Trainer is not at its committed checkpoint')
        restore_clock(self, identity)
        budget_path = run / f'SEGMENT_{segment:03d}_BUDGET_COMPLETE.json'
        if budget_path.exists():
            budget = read(budget_path)
            if (budget.get('config_digest') != self.config_digest
                    or budget.get('segment') != segment
                    or budget.get('active_threshold_seconds') != threshold
                    or budget.get('checkpoint') != identity
                    or budget.get('active_seconds', -1) < threshold):
                raise ValueError('Segment budget marker mismatch')
        original_generate = self.generate
        collecting = False

        def guarded_generate(kind, *args, **kwargs):
            if collecting and kind == 'receiver' and self.active_seconds() >= threshold:
                raise _CollectionBudgetReached()
            return original_generate(kind, *args, **kwargs)

        self.generate = guarded_generate
        try:
            if self.update == 0:
                evaluate(self)
            else:
                # Use the same baseline validation as the original implementation.
                from stego.university_evaluation import _baseline_d
                _baseline_d(self.config, self)
            if self.update and self.update % self.config['evaluation_interval'] == 0:
                evaluate(self)
            while not budget_path.exists() and self.active_seconds() < threshold:
                if training.STOP:
                    raise SystemExit(75)
                number = self.update + 1
                kind = training.phase_for_update(number)
                began = time.monotonic()
                self.public_rows = []
                try:
                    collecting = kind == 'receiver'
                    row = self.receiver_update() if collecting else self.sender_update()
                except _CollectionBudgetReached:
                    spent = self.active_seconds()
                    rollback_started = time.monotonic()
                    self.restore(Path(self.current_checkpoint))
                    now = time.monotonic()
                    self.elapsed = max(self.elapsed, spent + now - rollback_started)
                    self.started = now
                    persist_clock(self)
                    break
                finally:
                    collecting = False
                self.update = number
                row.update(update=number, kind=kind, cycle=(number - 1) // 7 + 1,
                           duration_seconds=time.monotonic() - began,
                           config_digest=self.config_digest)
                self.metrics.append(row)
                self.checkpoint()
                persist_clock(self)
                durations = {k: [r['duration_seconds'] for r in self.metrics if r['kind'] == k]
                             for k in ('receiver', 'sender')}
                training.atomic_json(run / 'TIMING.json', dict(
                    config_digest=self.config_digest, completed_update=self.update,
                    active_seconds=self.active_seconds(),
                    mean_seconds={k: sum(v) / len(v) if v else None for k, v in durations.items()},
                    counts={k: len(v) for k, v in durations.items()}))
                if number in (1, 3):
                    training.atomic_json(run / f'FIRST_{kind.upper()}_UPDATE.json', row)
                if training.STOP:
                    raise SystemExit(75)
                if number % self.config['evaluation_interval'] == 0:
                    evaluate(self)
            if training.STOP:
                raise SystemExit(75)
            if not budget_path.exists():
                training.atomic_json(budget_path, dict(
                    config_digest=self.config_digest, segment=segment,
                    active_threshold_seconds=threshold, total_active_budget_seconds=TOTAL_SECONDS,
                    completed_update=self.update, checkpoint=checkpoint_identity(self),
                    active_seconds=persist_clock(self)))
            evaluate(self)
            identity = checkpoint_identity(self)
            evaluation = run / 'evaluations' / f'step_{self.update:06d}' / 'COMPLETE.json'
            payload = dict(config_digest=self.config_digest, segment=segment,
                           active_threshold_seconds=threshold, total_active_budget_seconds=TOTAL_SECONDS,
                           completed_update=self.update, checkpoint=identity,
                           evaluation_complete_sha256=training.sha256(evaluation),
                           active_seconds=persist_clock(self),
                           stop_reason='configured_segment_active_time_budget',
                           sender_updates=sum(r['kind'] == 'sender' for r in self.metrics),
                           receiver_updates=sum(r['kind'] == 'receiver' for r in self.metrics))
            training.atomic_json(own_complete, payload)
            check_completion(self, segment)
            if segment == 3:
                finalize_run(self, payload)
            else:
                self.log_status('segment_complete', segment=segment,
                                cumulative_active_seconds=payload['active_seconds'])
        finally:
            self.generate = original_generate
            persist_clock(self)

    training.Trainer.train = train
    return train
