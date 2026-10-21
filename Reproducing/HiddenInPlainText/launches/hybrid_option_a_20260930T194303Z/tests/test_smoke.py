import unittest
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from stego.engine import EngineConfig
from stego.smoke import _tensor_bytes, production_minibatch_canary, sample_indices


class FakeProductionEngine:
    AGENTS = ("A", "B", "I")

    def __init__(self, fail_update=False):
        self.config = EngineConfig(device="cpu", train_microbatch_size=4,
            think_tokens=32, final_answer_tokens=48, generation_batch_size=4,
            decoder_objective="fixed_baseline_pg_ce")
        self.model, self.judge = object(), object()
        self.versions = {agent: 0 for agent in self.AGENTS}
        self.generations, self.updates = [], []
        self.fail_update = fail_update

    def generate(self, agent, messages, kind, seed):
        self.generations.append({"agent": agent, "messages": deepcopy(messages),
                                 "kind": kind, "config": self.config})
        final_tokens = self.config.final_answer_tokens if kind == "sender" else 1
        return [SimpleNamespace(final_text="A useful answer. Another useful sentence." if kind == "sender" else str(index % 2),
            prompt_ids=list(range(10 + len(message[1]["content"]))),
            thought_token_count=self.config.think_tokens, final_token_count=final_tokens,
            action_mask=[True] * (self.config.think_tokens + final_tokens))
            for index, message in enumerate(messages)]

    def update(self, agent, groups, rewards, group_weights=None, target_bits=None):
        if self.fail_update:
            raise RuntimeError("synthetic GPU failure")
        self.updates.append({"agent": agent, "groups": groups, "rewards": rewards,
                             "weights": group_weights, "targets": target_bits})
        self.versions[agent] += 1
        return {"grad_norm": 0.5, "max_microbatch_trajectories": min(
            self.config.train_microbatch_size, sum(map(len, groups)))}

    def assert_judge_frozen(self):
        pass


def production_canary_test_environment(engine):
    stack = ExitStack()
    stack.enter_context(patch("stego.smoke.adapter_digest", side_effect=lambda unused, agent: str(engine.versions[agent])))
    stack.enter_context(patch("stego.smoke.frozen_fingerprint", return_value={"fixed": True}))
    stack.enter_context(patch("torch.cuda.synchronize"))
    stack.enter_context(patch("torch.cuda.reset_peak_memory_stats"))
    stack.enter_context(patch("torch.cuda.max_memory_allocated", return_value=1234))
    stack.enter_context(patch("torch.cuda.max_memory_reserved", return_value=2345))
    stack.enter_context(redirect_stdout(io.StringIO()))
    return stack


class SmokeUtilityTests(unittest.TestCase):
    def test_large_tensor_indices_never_round_past_the_end(self):
        # Test dimensions, without allocating these potentially huge tensors.
        for length in (2**24 + 1, 2**28, 151936 * 2560, 2**31 + 17, 2**53 + 7):
            with self.subTest(length=length):
                indices = sample_indices(length)
                self.assertEqual(len(indices), 32)
                self.assertEqual(indices[0], 0)
                self.assertEqual(indices[-1], length - 1)
                self.assertTrue(all(0 <= i < length for i in indices))
                self.assertEqual(indices, sorted(set(indices)))
        # The original float32 endpoint calculation reproduces the failure.
        self.assertEqual(torch.linspace(0, 2**28 - 1, 32)[-1].long().item(), 2**28)

    def test_small_and_empty_tensors(self):
        self.assertEqual(sample_indices(0), [])
        self.assertEqual(sample_indices(1), [0])
        self.assertEqual(sample_indices(4), [0, 1, 2, 3])
        self.assertEqual(sample_indices(100, 1), [0])
        with self.assertRaises(ValueError):
            sample_indices(-1)

    def test_scalar_and_bfloat16_hash_bytes(self):
        self.assertEqual(len(_tensor_bytes(torch.tensor(1.0))), 4)
        self.assertEqual(len(_tensor_bytes(torch.tensor([1, 2], dtype=torch.bfloat16))), 4)

    def test_production_canary_uses_real_budgets_prompts_and_mixed_roles_then_restores_config(self):
        engine = FakeProductionEngine()
        original = engine.config
        config = {"think_tokens": 256, "final_answer_tokens": 128,
                  "generation_batch_size": 16, "sender_system_prompt": "Frozen sender prompt.",
                  "receiver_system_prompt": "Frozen receiver {receiver} prompt."}
        with production_canary_test_environment(engine):
            result = production_minibatch_canary(engine, config, 123)
        self.assertIs(engine.config, original)
        self.assertEqual(len(engine.updates), 2)
        self.assertEqual([len(group) for group in engine.updates[0]["groups"]], [2, 2])
        mixed = engine.updates[1]
        self.assertEqual([len(group) for group in mixed["groups"]], [2, 1, 1])
        self.assertEqual(mixed["weights"], [0.5, 0.25, 0.25])
        self.assertEqual(mixed["targets"], [None, 0, 1])
        self.assertEqual(mixed["rewards"], [[-1.0, 1.0], [1.0], [1.0]])
        self.assertEqual(engine.versions, {"A": 2, "B": 0, "I": 0})
        for call in engine.generations:
            self.assertEqual(call["config"].think_tokens, 256)
            self.assertEqual(call["config"].final_answer_tokens, 128)
            self.assertEqual(call["config"].train_microbatch_size, 4)
            for message in call["messages"]:
                if call["kind"] == "sender":
                    self.assertEqual(message[0]["content"], config["sender_system_prompt"])
                else:
                    self.assertEqual(message[0]["content"], "Frozen receiver A prompt.")
                    self.assertNotIn("hidden_bit", json.loads(message[1]["content"]))
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["peak_gpu_memory_allocated_bytes"], 1234)
        self.assertEqual(result["peak_gpu_memory_reserved_bytes"], 2345)
        self.assertEqual(result["sender_trajectories"][0]["sampled_tokens"], 384)
        self.assertEqual(result["mixed_receiver_trajectories"][0]["sampled_tokens"], 257)
        self.assertNotIn("A useful answer", json.dumps(result))

    def test_production_canary_restores_config_after_a_runtime_failure(self):
        engine = FakeProductionEngine(fail_update=True)
        original = engine.config
        with production_canary_test_environment(engine), self.assertRaisesRegex(RuntimeError, "synthetic GPU failure"):
            production_minibatch_canary(engine, {"think_tokens": 256, "final_answer_tokens": 128}, 123)
        self.assertIs(engine.config, original)


if __name__ == "__main__":
    unittest.main()
