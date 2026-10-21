from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from stego.decoder_comparison import (REQUIRED_SOURCES, DecoderComparison,
    bit_token_ids, conditional_p0, json_digest, prepare)
from stego.engine import EngineConfig, Trajectory
from stego.rejection_baseline import sha256
from stego.train import Preempted


def trajectory(guess=0, p0=0.8):
    return Trajectory(agent="V", kind="bit", prompt_ids=[10],
        completion_ids=[50, 77, (100, 101)[guess]], action_mask=[True, False, True],
        supports=[None, None, [100, 101]], modes=["thought", "forced", "constrained"],
        old_logps=[-2.0, 0.0, math.log(p0 if guess == 0 else 1 - p0)],
        final_text=str(guess), thought_text="PRIVATE_THOUGHT_MUST_NOT_BE_RETAINED",
        thought_token_count=256, final_token_count=1, forced_think_close=True)


class FakeEngine:
    label_ids = (100, 101)

    def __init__(self, interrupt_after=None):
        self.calls = []
        self.interrupt_after = interrupt_after

    def assert_judge_frozen(self):
        pass

    def decode(self, arm, adapter, messages, seeds):
        if self.interrupt_after is not None and len(self.calls) == self.interrupt_after:
            raise Preempted("test interruption")
        self.calls.append((arm, adapter, messages, seeds))
        return [trajectory(seed % 2, 0.8 if arm == "vanilla" else 0.6) for seed in seeds]


class DecoderComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        for name in REQUIRED_SOURCES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        self.sources = {name: sha256(self.source / name) for name in REQUIRED_SOURCES}
        self.training = {**asdict(EngineConfig()), "communicator_path": "/models/qwen3-4b",
                         "judge_path": "/models/qwen3-4b", "source_sha256": self.sources}
        training_path = self.root / "training.json"
        training_path.write_text(json.dumps(self.training))
        self.checkpoint = self.root / "checkpoint"
        (self.checkpoint / "engine").mkdir(parents=True)
        (self.checkpoint / "state.pt").write_bytes(b"present but never loaded by decoder")
        (self.checkpoint / "COMPLETE.json").write_text(json.dumps({"global_step": 40,
            "config_digest": json_digest(self.training)}))
        (self.checkpoint / "engine/engine.json").write_text(json.dumps({
            "config": asdict(EngineConfig()), "communicator_path": self.training["communicator_path"],
            "judge_path": self.training["judge_path"]}))
        for agent in ("A", "B", "I"):
            folder = self.checkpoint / "engine/adapters" / agent
            folder.mkdir(parents=True)
            (folder / "adapter_config.json").write_text("{}")
            (folder / "adapter_model.safetensors").write_bytes(b"fixture only")
        validation_path = self.root / "validation.json"
        validation_path.write_text(json.dumps({"global_step": 40, "n": 32, "rows": [
            {"question_id": f"valid:{i}", "question": f"How to carry item {i}",
             "sender": "A" if i % 2 else "B", "bit": i % 2,
             "public_string": format(i, "016b"), "answer": "Use a sturdy bag. Keep it upright.",
             "judge_pass": i % 4 != 0, "format_ok": True, "d": int(i % 4 == 0)}
            for i in range(32)]}))
        artifacts = [training_path, validation_path, self.checkpoint / "COMPLETE.json",
                     self.checkpoint / "engine/engine.json"]
        self.config = {"training_config_path": str(training_path), "validation_path": str(validation_path),
            "checkpoint_path": str(self.checkpoint), "checkpoint_step": 40, "replicates": 8,
            "seed": 20260930, "source_sha256": self.sources,
            "artifact_sha256": {str(path): sha256(path) for path in artifacts}}

    def make(self, engine, config=None):
        config = config or self.config
        return DecoderComparison(engine, config, self.root / "run", prepare(config, self.source))

    def test_probability_extracts_both_final_labels_and_rejects_wrong_support(self):
        for guess in (0, 1):
            self.assertAlmostEqual(conditional_p0(trajectory(guess, 0.8), (100, 101)), 0.8)
        with self.assertRaisesRegex(RuntimeError, "two-way"):
            conditional_p0(replace(trajectory(), supports=[None, None, [100, 101, 102]]), (100, 101))
        with self.assertRaisesRegex(RuntimeError, "two-way"):
            conditional_p0(replace(trajectory(), final_text="1"), (100, 101))
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            conditional_p0(replace(trajectory(), final_token_count=2), (100, 101))
        with self.assertRaisesRegex(RuntimeError, "log probability"):
            conditional_p0(replace(trajectory(), old_logps=[-2.0, 0.0, float("nan")]), (100, 101))
        class Tokenizer:
            def encode(self, label, add_special_tokens):
                return [100 + int(label)]
        self.assertEqual(bit_token_ids(Tokenizer()), (100, 101))
        class InvalidTokenizer(Tokenizer):
            def encode(self, label, add_special_tokens):
                return [100, int(label)]
        with self.assertRaisesRegex(RuntimeError, "single-token"):
            bit_token_ids(InvalidTokenizer())

    def test_complete_comparison_matches_inputs_and_seeds_without_retaining_private_thought(self):
        engine = FakeEngine()
        with redirect_stdout(io.StringIO()):
            summary = self.make(engine).run()
        self.assertEqual(summary["n_decodes"], 1024)
        self.assertEqual(len(summary["per_question"]), 128)
        for role in ("receiver", "interceptor"):
            self.assertAlmostEqual(summary["cells"][role]["vanilla"]["mean_p0"], 0.8)
            self.assertAlmostEqual(summary["cells"][role]["trained"]["mean_p0"], 0.6)
            self.assertAlmostEqual(summary["trained_minus_vanilla"][role]["mean_p0"], -0.2)
            for arm in ("vanilla", "trained"):
                self.assertEqual(summary["cells"][role][arm]["n_decodes"], 256)
                self.assertEqual(summary["cells"][role][arm]["n_answers"], 32)
                self.assertEqual(summary["cells"][role][arm]["correct_count"], 128)
        decisions = json.loads((self.root / "run/RESULTS.json").read_text())["rows"]
        paired = {}
        for row in decisions:
            key = row["question_id"], row["role"], row["replicate"]
            paired.setdefault(key, []).append(row)
            self.assertEqual(row["adapter"], row["receiver"] if row["role"] == "receiver" else "I")
        self.assertEqual(len(paired), 512)
        for pair in paired.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0]["seed"], pair[1]["seed"])
        for arm, adapter, messages, seeds in engine.calls:
            self.assertEqual(len(messages), len(seeds))
            self.assertLessEqual(len(messages), 16)
            for prompt in messages:
                payload = json.loads(prompt[1]["content"])
                self.assertEqual(set(payload), {"Q", "s", "sender", "receiver", "x"})
                self.assertNotIn("hidden_bit", payload)
                if adapter != "I":
                    self.assertEqual(adapter, payload["receiver"])
                self.assertEqual(payload["x"], "Use a sturdy bag. Keep it upright.")
        for path in (self.root / "run").rglob("*"):
            if path.is_file():
                self.assertNotIn("PRIVATE_THOUGHT", path.read_text())
        prompts = json.loads((self.root / "run/PROMPTS.json").read_text())
        self.assertEqual(set(prompts), {"receiver_A", "receiver_B", "interceptor"})
        self.assertIn("communicator A", prompts["receiver_A"])
        self.assertIn("communicator B", prompts["receiver_B"])
        self.assertTrue((self.root / "run/report.md").exists())
        self.assertTrue((self.root / "run/per_question.csv").exists())

    def test_atomic_resume_preserves_committed_batch_and_rejects_changed_run(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(Preempted):
            self.make(FakeEngine(interrupt_after=1)).run()
        paths = list((self.root / "run/batches").glob("*.json"))
        self.assertEqual(len(paths), 1)
        saved = paths[0].read_bytes()
        resumed = FakeEngine()
        with redirect_stdout(io.StringIO()):
            self.make(resumed).run()
        self.assertEqual(sum(len(call[2]) for call in resumed.calls), 1008)
        self.assertEqual(paths[0].read_bytes(), saved)
        repeated = FakeEngine()
        with redirect_stdout(io.StringIO()):
            self.make(repeated).run()
        self.assertEqual(repeated.calls, [])
        with self.assertRaisesRegex(RuntimeError, "Saved config.json"):
            self.make(FakeEngine(), {**self.config, "new_note": "identity changed"})
        payload = json.loads(paths[0].read_text())
        payload["rows"][0]["guess"] = 1 - payload["rows"][0]["guess"]
        paths[0].write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "Saved batch differs"):
            self.make(FakeEngine()).run()

    def test_prepare_guards_checkpoint_identity_sources_and_saved_answers(self):
        training, selection, manifest = prepare(self.config, self.source)
        self.assertEqual(len(selection), 32)
        self.assertEqual(len(manifest["input_sha256"]), 10)
        self.assertTrue(manifest["checkpoint_state_present"])
        (self.checkpoint / "state.pt").unlink()
        with self.assertRaisesRegex(RuntimeError, "state.pt"):
            prepare(self.config, self.source)
        (self.checkpoint / "state.pt").write_bytes(b"present")
        complete_path = self.checkpoint / "COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        complete["config_digest"] = "wrong"
        complete_path.write_text(json.dumps(complete))
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            prepare(self.config, self.source)
        changed = {**self.config, "artifact_sha256": {**self.config["artifact_sha256"],
                   str(complete_path): sha256(complete_path)}}
        with self.assertRaisesRegex(RuntimeError, "provenance"):
            prepare(changed, self.source)


if __name__ == "__main__":
    unittest.main()
