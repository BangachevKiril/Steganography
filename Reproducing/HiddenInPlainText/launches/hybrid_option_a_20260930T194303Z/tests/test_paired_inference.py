"""CPU checks for matched, independent sampling streams and frozen inference."""
from dataclasses import asdict
import json
from types import SimpleNamespace
import unittest

import torch

from stego.engine import Engine, EngineConfig
from stego.paired_inference import SeededFrozenInferenceEngine
from stego.protocol import Episode, build_messages


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return {"</think>": [3], "\n\n": [4], "0": [5], "1": [6],
                "PASS": [7], "FAIL": [8]}[text]

    def apply_chat_template(self, messages, **kwargs):
        slow = "slow" in messages[-1]["content"]
        prefix = [1, 1, 1] if slow else [1]
        return prefix + ([10 if slow else 2] if kwargs["enable_thinking"] else [4])

    def decode(self, ids, skip_special_tokens=True):
        words = {0: "", 1: "prompt", 2: "PRIVATE_START", 3: "", 4: "\n\n",
                 5: "0", 6: "1", 7: "PASS", 8: "FAIL", 9: "",
                 10: "PRIVATE_START", 11: "PRIVATE_REASONING"}
        return "".join(words[token] for token in ids)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.full((12, 12), -float("inf")))
        with torch.no_grad():
            self.table[:, 9] = 50  # Forbidden in thoughts, likely in free text.
            self.table[0, 0] = 0  # Harmless support for masked stopped rows.
            self.table[2, 3] = 0  # The quick row closes after one draw.
            self.table[10, 11] = 0
            self.table[11, 11] = 0  # The slow row hits the thinking cap.
            self.table[4, :] = -float("inf")
            self.table[4, 5:9] = torch.tensor([0.2, 0.6, 0.5, -0.1])
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


def make_engine(batch_size=16):
    engine = object.__new__(SeededFrozenInferenceEngine)
    engine.torch = torch
    engine.config = EngineConfig(think_tokens=8, final_answer_tokens=5,
                                 generation_batch_size=batch_size, device="cpu")
    engine.device = torch.device("cpu")
    engine.model = Model().requires_grad_(False).eval()
    engine.tokenizer = Tokenizer()
    engine.judge, engine.judge_tokenizer = engine.model, engine.tokenizer
    engine.optimizers = {}
    engine._judge_versions = {name: p._version for name, p in engine.model.named_parameters()}
    engine._supports_logits_to_keep = True
    return engine


def messages(text):
    return [{"role": "user", "content": text}]


class PairedInferenceTests(unittest.TestCase):
    def test_early_stop_does_not_advance_that_rows_stream(self):
        engine = make_engine()
        alone = torch.Generator().manual_seed(918)
        paired = torch.Generator().manual_seed(918)
        other = torch.Generator().manual_seed(700)
        with torch.inference_mode():
            single = engine._sample_stage(engine.model, engine.tokenizer,
                [[1, 2]], "thought", 8, [3], generator=[alone], forbidden=[9])
            batch = engine._sample_stage(engine.model, engine.tokenizer,
                [[1, 2], [1, 1, 1, 10]], "thought", 8, [3],
                generator=[paired, other], forbidden=[9])
        self.assertEqual(single[0], batch[0])
        self.assertEqual(batch[0][0], [3])
        self.assertEqual(batch[1][0], [11] * 8)
        torch.testing.assert_close(alone.get_state(), paired.get_state(), rtol=0, atol=0)

    def test_duplicate_seeds_are_independent_and_batch_partition_invariant(self):
        inputs = [messages("quick"), messages("slow"), messages("quick")] * 6
        seeds = [100 + i // 3 for i in range(len(inputs))]
        batched = make_engine(16).generate_seeded("V", inputs, "sender", seeds)
        singles = make_engine(1).generate_seeded("V", inputs, "sender", seeds)
        repeated = make_engine(16).generate_seeded("V", inputs, "sender", seeds)
        self.assertEqual([asdict(t) for t in batched], [asdict(t) for t in singles])
        self.assertEqual([asdict(t) for t in batched], [asdict(t) for t in repeated])
        for start in range(0, len(inputs), 3):
            self.assertEqual(asdict(batched[start]), asdict(batched[start + 2]))
            self.assertFalse(batched[start].forced_think_close)
            self.assertTrue(batched[start + 1].forced_think_close)

    def test_constraints_logps_forced_masks_and_public_separation(self):
        engine = make_engine()
        trajectory = engine.generate_seeded("V", [messages("slow")], "bit", [44])[0]
        self.assertEqual(trajectory.thought_token_count, 8)
        self.assertTrue(trajectory.forced_think_close)
        self.assertEqual(trajectory.completion_ids[8:10], [3, 4])
        self.assertEqual(trajectory.action_mask[8:10], [False, False])
        self.assertEqual(trajectory.modes[8:10], ["forced", "forced"])
        self.assertIn(trajectory.final_text, ("0", "1"))
        self.assertEqual(trajectory.supports[-1], [5, 6])
        self.assertIn("PRIVATE_REASONING", trajectory.thought_text)
        self.assertNotIn("PRIVATE", trajectory.final_text)
        expected = torch.tensor([v for v, mask in zip(trajectory.old_logps, trajectory.action_mask) if mask])
        actual = Engine._trajectory_logps(engine, trajectory)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        episode = Episode("train:0", "How should I carry a bag?", "A", 1, "0101")
        forwarded = build_messages("judge", episode, trajectory.final_text)
        self.assertNotIn("PRIVATE", json.dumps(forwarded))
        self.assertEqual(set(json.loads(forwarded[1]["content"])), {"Q", "x"})
        verdict = engine.generate_seeded("J", [forwarded], "judge", [44])[0]
        self.assertIn(verdict.final_text, ("PASS", "FAIL"))
        self.assertEqual(verdict.supports[-1], [7, 8])
        self.assertFalse(engine.model.training)
        self.assertTrue(all(not p.requires_grad for p in engine.model.parameters()))

    def test_validation_and_forbidden_mutations(self):
        engine = make_engine()
        with self.assertRaises(ValueError):
            engine.generate_seeded("V", [messages("quick")], "sender", [])
        for seed in (True, -1, 2**64, 1.5):
            with self.assertRaises(ValueError):
                engine.generate_seeded("V", [messages("quick")], "sender", [seed])
        for agent, kind in (("A", "sender"), ("J", "sender"), ("V", "judge")):
            with self.assertRaises(ValueError):
                engine.generate_seeded(agent, [messages("quick")], kind, [1])
        for method in (engine.update, engine.save, engine.load):
            with self.assertRaises(RuntimeError):
                method()
        with torch.no_grad():
            engine.model.table[4, 5] = 10
        with self.assertRaisesRegex(RuntimeError, "mutation"):
            engine.generate_seeded("J", [messages("quick")], "judge", [1])

    def test_rejects_shared_generator_objects(self):
        engine = make_engine()
        generator = torch.Generator().manual_seed(4)
        with self.assertRaises(ValueError):
            engine._sample_stage(engine.model, engine.tokenizer, [[1, 2], [1, 2]],
                "thought", 8, [3], generator=[generator, generator], forbidden=[9])


if __name__ == "__main__":
    unittest.main()
