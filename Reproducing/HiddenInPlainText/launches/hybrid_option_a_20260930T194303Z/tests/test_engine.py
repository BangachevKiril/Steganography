"""CPU checks for the sampling/action bookkeeping used by the GPU engine."""
import math
import unittest
from types import SimpleNamespace

import torch

from stego.engine import Engine, EngineConfig, allowed_continuations, centered_advantages


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return {"</think>": [3], "\n\n": [4], "0": [5], "1": [6],
                "PASS": [7], "FAIL": [8]}[text]

    def apply_chat_template(self, messages, **kwargs):
        return [1, 2] if kwargs["enable_thinking"] else [1, 4]

    def decode(self, ids, skip_special_tokens=True):
        return "".join({0: "", 1: "x", 2: "thought", 3: "", 4: "\n\n",
                       5: "0", 6: "1", 7: "PASS", 8: "FAIL", 9: ""}[i] for i in ids)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(10, 10))
        with torch.no_grad():
            # Strong EOS pressure must be excluded in the private stage.
            self.table[2, 9] = 50
            self.table[2, 3] = 10
            self.table[4, 5] = 1.0
            self.table[4, 6] = 0.2
        self.generation_config = SimpleNamespace(eos_token_id=9)

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, return_dict=True,
                logits_to_keep=0):
        logits = self.table[input_ids]
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        elif logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits, past_key_values=object() if use_cache else None)


def tiny_engine(think_tokens=3):
    engine = object.__new__(Engine)
    engine.torch = torch
    engine.config = EngineConfig(think_tokens=think_tokens, device="cpu")
    engine.device = torch.device("cpu")
    engine.tokenizer = TinyTokenizer()
    engine.model = TinyModel()
    engine._supports_logits_to_keep = True
    return engine


class EngineSamplingTests(unittest.TestCase):
    def test_training_microbatch_size_must_be_a_positive_integer(self):
        for invalid in (0, -1, True, 1.5):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                EngineConfig(train_microbatch_size=invalid)
        self.assertEqual(EngineConfig(train_microbatch_size=4).train_microbatch_size, 4)

    def test_advantages_are_centered_without_variance_scaling(self):
        self.assertEqual(centered_advantages([-2, 1, 1]), [-2.0, 1.0, 1.0])
        self.assertEqual(centered_advantages([0, 0]), [0, 0])
        with self.assertRaises(ValueError):
            centered_advantages([])

    def test_prefix_language(self):
        self.assertEqual(allowed_continuations([[2, 3], [2, 4]], []), [2])
        self.assertEqual(allowed_continuations([[2, 3], [2, 4]], [2]), [3, 4])
        self.assertEqual(allowed_continuations([[2, 3], [2, 4]], [2, 3]), [])

    def test_generation_logps_match_training_probabilities(self):
        engine = tiny_engine()
        with torch.inference_mode():
            trajectories = engine._generate_batch("A", engine.model, engine.tokenizer,
                [[{"role": "user", "content": "test"}]] * 8, "bit",
                torch.Generator().manual_seed(12))
        for trajectory in trajectories:
            expected = torch.tensor([p for p, m in zip(trajectory.old_logps, trajectory.action_mask) if m])
            actual = engine._trajectory_logps(trajectory)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
            self.assertEqual(trajectory.completion_ids[0], 3)
            self.assertFalse(trajectory.action_mask[1])  # injected newline
            self.assertIn(trajectory.final_text, ("0", "1"))
            self.assertEqual(trajectory.supports[-1], [5, 6])
            self.assertFalse(trajectory.forced_think_close)
            self.assertLessEqual(trajectory.thought_token_count, 3)
            final_logits = engine.model.table[4, [5, 6]] / engine.config.temperature
            correct = torch.log_softmax(final_logits, 0)[int(trajectory.final_text)]
            self.assertAlmostEqual(trajectory.old_logps[-1], correct.item(), places=6)

    def test_forced_close_does_not_receive_policy_loss(self):
        engine = tiny_engine(think_tokens=1)
        with torch.no_grad():
            engine.model.table[2, 2] = 60  # sample another thought token, not close
        with torch.inference_mode():
            trajectory = engine._generate_batch("B", engine.model, engine.tokenizer,
                [[{"role": "user", "content": "test"}]], "bit",
                torch.Generator().manual_seed(1))[0]
        self.assertTrue(trajectory.forced_think_close)
        self.assertEqual(trajectory.completion_ids[:3], [2, 3, 4])
        self.assertEqual(trajectory.action_mask[:3], [True, False, False])
        actual = engine._trajectory_logps(trajectory)
        self.assertEqual(actual.numel(), 2)  # one thought token, one final bit
        (-actual.sum()).backward()
        self.assertTrue(torch.isfinite(engine.model.table.grad).all())

    def test_no_thinking_uses_direct_final_output(self):
        engine = tiny_engine(think_tokens=0)
        with torch.inference_mode():
            trajectory = engine._generate_batch("I", engine.model, engine.tokenizer,
                [[{"role": "user", "content": "test"}]], "bit",
                torch.Generator().manual_seed(1))[0]
        self.assertEqual(trajectory.thought_token_count, 0)
        self.assertEqual(len(trajectory.completion_ids), 1)
        self.assertEqual(trajectory.action_mask, [True])


if __name__ == "__main__":
    unittest.main()
