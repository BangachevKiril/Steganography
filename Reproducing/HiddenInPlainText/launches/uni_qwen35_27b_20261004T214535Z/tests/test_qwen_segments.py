"""Torch-free tests of stage boundaries, elapsed clocks, and checkpoint recovery."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from orcd import qwen_segment_training as segments


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True))


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name)
        self.clock = SimpleNamespace(now=0.)
        self.addCleanup(patch.stopall)
        patch.object(segments.time, 'monotonic', side_effect=lambda: self.clock.now).start()
        patch.dict(sys.modules, {'stego.university_evaluation': SimpleNamespace(
            _baseline_d=lambda config, trainer: .08)}).start()
        clock = self.clock

        class Trainer:
            def __init__(self, run, restore=True):
                self.run = Path(run)
                self.config = {'max_active_seconds': 83700, 'evaluate_initial_checkpoint': True,
                               'evaluation_interval': 50}
                self.config_digest = 'same-config-all-stages'
                self.update = 0
                self.elapsed = 0.
                self.started = clock.now
                self.metrics, self.public_rows = [], []
                self.stream_cursor = 0
                self.optimizer_steps = 0
                self.update_cost = 10000.
                self.collection_batches = 1
                self.evaluation_cost = 5.
                self.preempt_evaluation_update = None
                self.evaluation_calls = []
                self.statuses = []
                existing = sorted((self.run / 'checkpoints').glob('step_*'))
                if existing and restore:
                    self.restore(existing[-1])
                else:
                    self.checkpoint()

            def active_seconds(self):
                return self.elapsed + clock.now - self.started

            def generate(self, kind):
                self.stream_cursor += 1
                self.public_rows.append({'pending': self.stream_cursor})
                clock.now += self.update_cost

            def receiver_update(self):
                for _ in range(self.collection_batches):
                    self.generate('receiver')
                self.optimizer_steps += 1
                return {'n': 256}

            def sender_update(self):
                self.generate('sender')
                self.optimizer_steps += 1
                return {'n': 192}

            def checkpoint(self):
                path = self.run / 'checkpoints' / f'step_{self.update:06d}'
                path.mkdir(parents=True)
                state = dict(update=self.update, elapsed=self.active_seconds(),
                             metrics=self.metrics, public_rows=self.public_rows,
                             stream_cursor=self.stream_cursor, optimizer_steps=self.optimizer_steps)
                atomic_json(path / 'state.pt', state)
                atomic_json(path / 'COMPLETE.json', dict(
                    config_digest=self.config_digest, completed_update=self.update,
                    state_sha256=sha256(path / 'state.pt'), elapsed=state['elapsed']))
                self.current_checkpoint = path

            def restore(self, path):
                state = read(Path(path) / 'state.pt')
                for name in ('update', 'elapsed', 'metrics', 'public_rows',
                             'stream_cursor', 'optimizer_steps'):
                    setattr(self, name, state[name])
                self.started = clock.now
                self.current_checkpoint = Path(path)

            def evaluate(self):
                self.evaluation_calls.append(self.update)
                path = self.run / 'evaluations' / f'step_{self.update:06d}'
                path.mkdir(parents=True, exist_ok=True)
                if (path / 'COMPLETE.json').exists():
                    return
                clock.now += self.evaluation_cost
                if self.preempt_evaluation_update == self.update:
                    self.preempt_evaluation_update = None
                    raise SystemExit(75)
                atomic_json(path / 'COMPLETE.json', dict(
                    config_digest=self.config_digest, completed_update=self.update,
                    source_checkpoint=dict(path=str(self.current_checkpoint.resolve()),
                                           completed_update=self.update,
                                           complete_sha256=sha256(self.current_checkpoint / 'COMPLETE.json'))))

            def log_status(self, stage, **kwargs):
                self.statuses.append((stage, kwargs))

        self.training = SimpleNamespace(Trainer=Trainer, STOP=False, sha256=sha256,
                                        atomic_json=atomic_json,
                                        phase_for_update=lambda n: 'receiver' if (n - 1) % 7 < 2 else 'sender')

    def trainer(self, segment):
        segments.install_segment_training(self.training, segment)
        return self.training.Trainer(self.run)

    def test_three_continuous_segments_and_idempotence(self):
        last_update = 0
        for segment in (1, 2, 3):
            trainer = self.trainer(segment)
            trainer.train()
            complete = read(self.run / f'SEGMENT_{segment:03d}_COMPLETE.json')
            self.assertEqual(complete['active_threshold_seconds'], segment * 27900)
            self.assertGreaterEqual(complete['active_seconds'], segment * 27900)
            self.assertGreater(trainer.update, last_update)
            self.assertEqual(trainer.optimizer_steps, trainer.update)
            self.assertEqual(len(trainer.metrics), trainer.update)
            last_update = trainer.update
            if segment < 3:
                self.assertFalse((self.run / 'COMPLETE.json').exists())
                self.assertFalse((self.run / 'BUDGET_COMPLETE.json').exists())
        self.assertEqual([r['kind'] for r in trainer.metrics],
                         [self.training.phase_for_update(n) for n in range(1, trainer.update + 1)])
        terminal_before = (self.run / 'COMPLETE.json').read_bytes()
        clock_before = (self.run / segments.CLOCK_NAME).read_bytes()
        self.trainer(1).train()
        self.assertEqual((self.run / 'COMPLETE.json').read_bytes(), terminal_before)
        self.assertEqual((self.run / segments.CLOCK_NAME).read_bytes(), clock_before)

    def test_final_evaluation_preemption_preserves_time_and_update(self):
        trainer = self.trainer(1)
        trainer.preempt_evaluation_update = 3
        with self.assertRaises(SystemExit) as caught:
            trainer.train()
        self.assertEqual(caught.exception.code, 75)
        self.assertTrue((self.run / 'SEGMENT_001_BUDGET_COMPLETE.json').exists())
        self.assertFalse((self.run / 'SEGMENT_001_COMPLETE.json').exists())
        seconds = read(self.run / segments.CLOCK_NAME)['active_seconds']
        checkpoint_seconds = read(trainer.current_checkpoint / 'COMPLETE.json')['elapsed']
        self.assertGreater(seconds, checkpoint_seconds)
        resumed = self.trainer(1)
        resumed.train()
        complete = read(self.run / 'SEGMENT_001_COMPLETE.json')
        self.assertEqual(resumed.update, 3)
        self.assertGreaterEqual(complete['active_seconds'], seconds + resumed.evaluation_cost)

    def test_partial_receiver_collection_ends_stage_without_committing_it(self):
        trainer = self.trainer(1)
        trainer.update_cost = 15000.
        trainer.collection_batches = 10
        trainer.train()
        complete = read(self.run / 'SEGMENT_001_COMPLETE.json')
        self.assertEqual(trainer.update, 0)
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertEqual(trainer.stream_cursor, 0)
        self.assertEqual(trainer.public_rows, [])
        self.assertGreaterEqual(complete['active_seconds'], 30000.)

    def test_stop_mid_update_restores_checkpoint_clock_on_requeue(self):
        trainer = self.trainer(1)
        original = trainer.receiver_update

        def receive_then_stop():
            row = original()
            self.training.STOP = True
            return row

        trainer.receiver_update = receive_then_stop
        with self.assertRaises(SystemExit):
            trainer.train()
        self.assertEqual(trainer.update, 1)
        self.training.STOP = False
        resumed = self.trainer(1)
        resumed.train()
        self.assertEqual(resumed.update, 3)
        self.assertEqual(resumed.optimizer_steps, 3)

    def test_uncommitted_preemption_time_is_counted_but_state_is_not_committed(self):
        trainer = self.trainer(1)

        def interrupted_receiver():
            trainer.stream_cursor += 10
            trainer.optimizer_steps += 1  # Simulate partially modified in-memory state.
            self.clock.now += 7000.
            raise SystemExit(75)

        trainer.receiver_update = interrupted_receiver
        with self.assertRaises(SystemExit):
            trainer.train()
        self.assertEqual(trainer.update, 0)
        spent = read(self.run / segments.CLOCK_NAME)['active_seconds']
        self.assertEqual(spent, 7005.)
        resumed = self.trainer(1)
        self.assertEqual(resumed.stream_cursor, 0)
        self.assertEqual(resumed.optimizer_steps, 0)
        resumed.train()
        self.assertEqual(resumed.optimizer_steps, resumed.update)
        self.assertGreaterEqual(read(self.run / 'SEGMENT_001_COMPLETE.json')['active_seconds'],
                                spent + resumed.update * resumed.update_cost)

    def test_optimizer_failure_is_propagated_without_completion(self):
        trainer = self.trainer(1)

        def broken_receiver():
            self.clock.now += 4.
            raise FloatingPointError('nonfinite attempted update')

        trainer.receiver_update = broken_receiver
        with self.assertRaisesRegex(FloatingPointError, 'nonfinite'):
            trainer.train()
        self.assertFalse((self.run / 'SEGMENT_001_COMPLETE.json').exists())
        self.assertEqual(read(self.run / segments.CLOCK_NAME)['active_seconds'], 9.)

    def test_regular_evaluations_remain_at_configured_interval(self):
        trainer = self.trainer(1)
        trainer.config['evaluation_interval'] = 2
        trainer.train()
        self.assertEqual(trainer.evaluation_calls, [0, 2, 3])

    def test_invalid_stage_missing_prerequisite_and_clock_tampering_fail(self):
        for value in (0, 4, True, '1'):
            with self.assertRaises(ValueError):
                segments.install_segment_training(self.training, value)
        trainer = self.trainer(2)
        with self.assertRaises(FileNotFoundError):
            trainer.train()
        trainer = self.trainer(1)
        trainer.preempt_evaluation_update = 0
        with self.assertRaises(SystemExit):
            trainer.train()
        path = self.run / segments.CLOCK_NAME
        payload = read(path)
        payload['checkpoint']['state_sha256'] = 'wrong'
        atomic_json(path, payload)
        with self.assertRaisesRegex(ValueError, 'binding'):
            self.trainer(1).train()

    def test_stage_three_terminal_marker_crash_is_recoverable(self):
        for segment in (1, 2, 3):
            self.trainer(segment).train()
        (self.run / 'COMPLETE.json').unlink()
        (self.run / 'BUDGET_COMPLETE.json').unlink()
        self.trainer(3).train()
        self.assertTrue((self.run / 'COMPLETE.json').exists())
        self.assertTrue((self.run / 'BUDGET_COMPLETE.json').exists())


if __name__ == '__main__':
    unittest.main()
