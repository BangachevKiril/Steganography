from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from stego.paired_sender_probe import PairedProbe, json_digest, prepare
from stego.train import Preempted


@dataclass
class FakeTrajectory:
    final_text: str
    thought_text: str = "PRIVATE_REASONING_MUST_NEVER_REACH_JUDGE"
    thought_token_count: int = 256
    forced_think_close: bool = True
    final_token_count: int = 12
    completion_ids: tuple = (1, 99)


class FakeEngine:
    model = tokenizer = None
    config = SimpleNamespace(final_answer_tokens=128)

    def __init__(self, fail_after=None):
        self.calls, self.fail_after = [], fail_after

    def generate(self, agent, messages, kind, seed):
        if len(self.calls) == self.fail_after:
            raise Preempted("canary interruption")
        self.calls.append((agent, messages, kind, seed))
        payload = json.loads(messages[0][1]["content"])
        if agent == "J":
            assert set(payload) == {"Q", "x"}
            assert "PRIVATE_REASONING" not in json.dumps(messages)
            return [FakeTrajectory("PASS")]
        bit = payload["hidden_bit"]
        return [FakeTrajectory("Carry the bag gently. Keep it upright." if bit == 0
                               else "Keep the bag upright. Carry it gently.")]

    def assert_judge_frozen(self):
        pass

    @staticmethod
    def _eos_ids(model, tokenizer):
        return [99]


class PairedProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.selected = [{"question_id": f"train:{i}", "question": f"How should I carry item {i}?",
                          "answer": "Baseline answer.", "judge_pass": i == 4,
                          "format_ok": False if i == 4 else True, "d": 1}
                         for i in range(5)]
        self.config = {"communicator": "A", "seed": 844166641,
                       "question_ids": [row["question_id"] for row in self.selected]}
        self.manifest = {"config_digest": "fixed", "checkpoint_global_step": 20}

    def test_paired_inputs_share_public_randomness_and_initial_sampling_seed(self):
        engine = FakeEngine()
        self.manifest["checkpoint_global_step"] = 60
        with redirect_stdout(io.StringIO()):
            result = PairedProbe(engine, self.config, self.selected, self.root / "run", self.manifest).run()
        self.assertEqual(result["n_answers"], 10)
        self.assertEqual(result["checkpoint_global_step"], 60)
        self.assertEqual(len(engine.calls), 20)
        for i, pair in enumerate(result["pairs"]):
            first, judge0, second, judge1 = engine.calls[i * 4:(i + 1) * 4]
            p0, p1 = (json.loads(call[1][0][1]["content"]) for call in (first, second))
            self.assertEqual(p0.pop("hidden_bit"), 0)
            self.assertEqual(p1.pop("hidden_bit"), 1)
            self.assertEqual(p0, p1)
            self.assertEqual(first[3], second[3])
            self.assertEqual(judge0[3], judge1[3])
            self.assertEqual(len(p0["s"]), 16)
            self.assertEqual(p0["sender"], "A")
            self.assertEqual(p0["receiver"], "B")
            self.assertEqual(pair["public_string"], p0["s"])
            self.assertEqual([row["bit"] for row in pair["outputs"]], [0, 1])
            self.assertTrue(all(row["answer_final_tokens"] == 12 for row in pair["outputs"]))
            self.assertTrue(all(not row["answer_final_truncated"] for row in pair["outputs"]))
        self.assertNotIn("PRIVATE_REASONING", json.dumps(result))

    def test_resume_preserves_completed_pairs(self):
        interrupted = FakeEngine(fail_after=8)
        with redirect_stdout(io.StringIO()), self.assertRaises(Preempted):
            PairedProbe(interrupted, self.config, self.selected, self.root / "run", self.manifest).run()
        self.assertEqual(len(list((self.root / "run/pairs").glob("*.json"))), 2)
        resumed = FakeEngine()
        with redirect_stdout(io.StringIO()):
            result = PairedProbe(resumed, self.config, self.selected, self.root / "run", self.manifest).run()
        self.assertEqual(len(resumed.calls), 12)
        self.assertEqual(result["n_answers"], 10)
        self.assertTrue((self.root / "run/COMPLETE.json").exists())
        completed = FakeEngine()
        with redirect_stdout(io.StringIO()):
            PairedProbe(completed, self.config, self.selected, self.root / "run", self.manifest).run()
        self.assertEqual(completed.calls, [])

    def test_preparation_rejects_wrong_checkpoint_or_nonrejected_selection(self):
        checkpoint = self.root / "checkpoint"
        (checkpoint / "engine/adapters/A").mkdir(parents=True)
        (checkpoint / "engine/engine.json").write_text("{}")
        (checkpoint / "engine/optimizers.pt").write_bytes(b"test-only")
        (checkpoint / "engine/adapters/A/adapter_model.safetensors").write_bytes(b"test-only")
        training = {"communicator_path": "/qwen3-4b", "judge_path": "/qwen3-4b"}
        training_path = self.root / "training.json"
        training_path.write_text(json.dumps(training))
        baseline_path = self.root / "baseline.json"
        baseline_path.write_text(json.dumps({"rows": self.selected}))
        complete_path = checkpoint / "COMPLETE.json"
        complete_path.write_text(json.dumps({"global_step": 20, "config_digest": json_digest(training)}))
        config = {**self.config, "training_config_path": str(training_path),
                  "checkpoint_path": str(checkpoint), "baseline_answers_path": str(baseline_path)}
        original, selected, manifest = prepare(config, self.root)
        self.assertEqual(original, training)
        self.assertEqual(selected, self.selected)
        self.assertEqual(manifest["communicator_joint_updates"], 10)
        config["checkpoint_step"] = 60
        with self.assertRaisesRegex(ValueError, "step 60"):
            prepare(config, self.root)
        complete_path.write_text(json.dumps({"global_step": 60, "config_digest": json_digest(training)}))
        _, _, manifest60 = prepare(config, self.root)
        self.assertEqual(manifest60["checkpoint_global_step"], 60)
        self.assertEqual(manifest60["communicator_joint_updates"], 30)
        config.pop("checkpoint_step")
        complete_path.write_text(json.dumps({"global_step": 19, "config_digest": json_digest(training)}))
        with self.assertRaisesRegex(ValueError, "step 20"):
            prepare(config, self.root)
        complete_path.write_text(json.dumps({"global_step": 20, "config_digest": json_digest(training)}))
        self.selected[0]["d"] = 0
        baseline_path.write_text(json.dumps({"rows": self.selected}))
        with self.assertRaisesRegex(ValueError, "rejected"):
            prepare(config, self.root)


if __name__ == "__main__":
    unittest.main()
