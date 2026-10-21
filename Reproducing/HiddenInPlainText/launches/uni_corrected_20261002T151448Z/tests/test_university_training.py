import copy
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from stego.university_training import (
    ShuffleStream, Trainer, capture_rng, isolated_models, latest_checkpoint,
    phase_for_update, restore_rng, reward_value,
)


class UniversityTrainingTests(unittest.TestCase):
    def test_schedule_starts_with_two_receiver_then_five_sender_updates(self):
        self.assertEqual([phase_for_update(i) for i in range(1, 22)],
                         (["receiver"] * 2 + ["sender"] * 5) * 3)
        for invalid in (0, -1):
            with self.assertRaises(ValueError):
                phase_for_update(invalid)

    def test_reward_masks_only_receiver_term_and_uses_character_length(self):
        self.assertAlmostEqual(reward_value(1.5, -0.5, "x" * 20), 0.0)
        self.assertAlmostEqual(reward_value(-1.5, -0.5, "x" * 20), 0.0)
        self.assertAlmostEqual(reward_value(1.5, 0, "x" * 10), math.tanh(1.5) + 0.05)
        self.assertAlmostEqual(reward_value(1.5, 0.5, "x" * 200), math.tanh(1.5) + 0.2)
        self.assertAlmostEqual(reward_value(0, 0, "a b c"), 0.025)

    def test_stream_matches_pytorch_shuffled_loader_including_short_epoch(self):
        for drop_last in (False, True):
            with self.subTest(drop_last=drop_last):
                torch.manual_seed(182)
                loader = torch.utils.data.DataLoader(list(range(10)), batch_size=4,
                                                     shuffle=True, drop_last=drop_last)
                expected = []
                for _ in range(2):
                    expected.extend(batch.tolist() for batch in loader)
                expected_rng = torch.get_rng_state().clone()
                torch.manual_seed(182)
                stream = ShuffleStream(range(10), 4, drop_last)
                actual = [stream.next() for _ in expected]
                self.assertEqual(actual, expected)
                self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))

    def test_stream_resume_continues_saved_permutation_and_rng(self):
        torch.manual_seed(128)
        first = ShuffleStream(range(10), 4)
        first.next()
        state, rng = copy.deepcopy(first.state_dict()), capture_rng()
        expected = [first.next() for _ in range(6)]
        second = ShuffleStream(range(10), 4)
        second.load_state_dict(state)
        restore_rng(rng)
        self.assertEqual([second.next() for _ in range(6)], expected)
        for invalid in ({**state, "permutation": [100] + state["permutation"][1:]},
                        {**state, "cursor": 11}):
            with self.assertRaises(ValueError):
                second.load_state_dict(invalid)

    def test_evaluation_restores_rng_and_mixed_module_modes_even_on_error(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        model.train()
        model[1].eval()
        modes = [module.training for module in model.modules()]
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(3))
        restore_rng(state)
        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            with isolated_models(model):
                model.eval()
                random.random()
                np.random.rand(20)
                torch.rand(20)
                raise RuntimeError("evaluation failed")
        actual = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(actual[:2], expected[:2])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertEqual([module.training for module in model.modules()], modes)

    def test_latest_checkpoint_ignores_uncommitted_state_and_stale_pointer(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            for step in (0, 2):
                checkpoint = run / "checkpoints" / f"step_{step:06d}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "state.pt").write_bytes(str(step).encode())
                (checkpoint / "COMPLETE.json").write_text(json.dumps({
                    "config_digest": "config", "completed_update": step,
                    "state_sha256": hashlib.sha256(str(step).encode()).hexdigest()}))
            (run / "LATEST.json").write_text(json.dumps({"completed_update": 0}))
            partial = run / "checkpoints/step_000003"
            partial.mkdir()
            (partial / "state.pt").write_bytes(b"uncommitted")
            selected = latest_checkpoint(run, "config")
            self.assertEqual(selected.name, "step_000002")
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                latest_checkpoint(run, "different")
            (selected / "state.pt").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                latest_checkpoint(run, "config")

    @staticmethod
    def small_trainer(run):
        trainer = Trainer.__new__(Trainer)
        trainer.run = run
        trainer.config = {"checkpoint_keep_recent": 3, "evaluation_interval": 7}
        trainer.config_digest = "cpu-checkpoint-config"
        trainer.update = 3
        trainer.metrics = [{"update": 3, "kind": "sender", "sentinel": "committed"}]
        trainer.public_rows = [{"update": 3, "sender_text": "public answer"}]
        trainer.sender = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        trainer.receiver = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
        trainer.sender.train()
        trainer.sender[1].eval()
        trainer.receiver.eval()
        trainer.sender_optimizer = torch.optim.AdamW(trainer.sender.parameters(), lr=0.01)
        trainer.receiver_optimizer = torch.optim.AdamW(trainer.receiver.parameters(), lr=0.02)
        for model, optimizer in ((trainer.sender, trainer.sender_optimizer),
                                 (trainer.receiver, trainer.receiver_optimizer)):
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler = torch.optim.lr_scheduler.StepLR(trainer.sender_optimizer, step_size=2)
        trainer.ppo = SimpleNamespace(kl_ctl=SimpleNamespace(value=0.083), current_step=9,
                                      lr_scheduler=scheduler, accelerator=SimpleNamespace(scaler=None))
        trainer.streams = {"sender": ShuffleStream(range(20), 4, True),
                           "receiver": ShuffleStream(range(20, 40), 4)}
        trainer.streams["sender"].next()
        trainer.streams["receiver"].next()
        trainer.active_seconds = lambda: 123.0
        trainer.publish_metrics = Mock()
        return trainer

    def assert_nested_equal(self, left, right):
        self.assertEqual(type(left), type(right))
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for lhs, rhs in zip(left, right):
                self.assert_nested_equal(lhs, rhs)
        else:
            self.assertEqual(left, right)

    def test_checkpoint_round_trips_parameters_optimizers_modes_sampler_and_rng(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            trainer = self.small_trainer(run)
            expected_sender = copy.deepcopy(trainer.sender.state_dict())
            expected_receiver = copy.deepcopy(trainer.receiver.state_dict())
            expected_optimizer = copy.deepcopy(trainer.sender_optimizer.state_dict())
            expected_receiver_optimizer = copy.deepcopy(trainer.receiver_optimizer.state_dict())
            expected_modes = {role: [m.training for m in getattr(trainer, role).modules()]
                              for role in ("sender", "receiver")}
            trainer.checkpoint()
            checkpoint = latest_checkpoint(run, trainer.config_digest)
            self.assertIsNotNone(checkpoint)
            expected_random = (random.random(), np.random.rand(), torch.rand(3))
            expected_next_sender = trainer.streams["sender"].next()
            expected_next_receiver = trainer.streams["receiver"].next()
            with torch.no_grad():
                for model in (trainer.sender, trainer.receiver):
                    for param in model.parameters():
                        param.add_(42)
                    model.train()
            trainer.ppo.kl_ctl.value = 8
            trainer.ppo.current_step = 88
            trainer.update = 4
            trainer.metrics = []
            trainer.public_rows = []
            trainer.restore(checkpoint)
            self.assert_nested_equal(trainer.sender.state_dict(), expected_sender)
            self.assert_nested_equal(trainer.receiver.state_dict(), expected_receiver)
            self.assert_nested_equal(trainer.sender_optimizer.state_dict(), expected_optimizer)
            self.assert_nested_equal(trainer.receiver_optimizer.state_dict(), expected_receiver_optimizer)
            self.assertEqual(trainer.ppo.kl_ctl.value, 0.083)
            self.assertEqual(trainer.ppo.current_step, 9)
            self.assertEqual(trainer.update, 3)
            self.assertEqual(trainer.elapsed, 123.0)
            self.assertEqual(trainer.metrics[0]["sentinel"], "committed")
            self.assertEqual(trainer.public_rows, [{"update": 3, "sender_text": "public answer"}])
            for role, modes in expected_modes.items():
                self.assertEqual([m.training for m in getattr(trainer, role).modules()], modes)
            actual_random = (random.random(), np.random.rand(), torch.rand(3))
            self.assertEqual(actual_random[:2], expected_random[:2])
            self.assertTrue(torch.equal(actual_random[2], expected_random[2]))
            self.assertEqual(trainer.streams["sender"].next(), expected_next_sender)
            self.assertEqual(trainer.streams["receiver"].next(), expected_next_receiver)
            with self.assertRaisesRegex(ValueError, "overwrite committed"):
                trainer.checkpoint()

    def test_nonfinite_parameters_cannot_be_committed(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = self.small_trainer(Path(temporary))
            with torch.no_grad():
                next(trainer.receiver.parameters()).flatten()[0] = float("nan")
            with self.assertRaisesRegex(FloatingPointError, "Nonfinite parameter"):
                trainer.checkpoint()
            self.assertIsNone(latest_checkpoint(trainer.run, trainer.config_digest))


if __name__ == "__main__":
    unittest.main()
