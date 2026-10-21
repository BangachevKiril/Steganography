from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
import unittest

from stego.rejection_baseline import (Baseline, diagnostics, sha256, summarize,
                                      vanilla_messages, verify_config)
from stego.train import Preempted


@dataclass
class FakeTrajectory:
    final_text: str
    thought_text: str = "PRIVATE_REASONING_MUST_NOT_LEAK"
    thought_token_count: int = 256
    final_token_count: int = 10
    completion_ids: tuple = (1, 99)
    forced_think_close: bool = True


class FakeEngine:
    model = tokenizer = None

    def __init__(self, preempt_judge=False):
        self.calls = []
        self.preempt_judge = preempt_judge

    @staticmethod
    def _eos_ids(model, tokenizer):
        return [99]

    def assert_judge_frozen(self):
        pass

    def generate(self, agent, messages, kind, seed):
        self.calls.append((agent, messages, kind, seed))
        if agent == "J" and self.preempt_judge:
            self.preempt_judge = False
            raise Preempted("test interruption after saved answers")
        result = []
        for item in messages:
            payload = json.loads(item[1]["content"])
            number = int(payload["Q"].split()[-1])
            if agent == "V":
                text = "Use a sturdy bag. Keep it upright." if number % 2 == 0 else "Use a sturdy bag."
            else:
                assert set(payload) == {"Q", "x"}
                assert "PRIVATE_REASONING" not in json.dumps(item)
                text = "PASS" if number % 4 < 2 else "FAIL"
            result.append(FakeTrajectory(text))
        return result


class RejectionBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "train.jsonl").write_text("".join(json.dumps({
            "goal": f"How should I carry item {i}", "sol1": "DO_NOT_EXPOSE_CHOICES",
            "sol2": "DO_NOT_EXPOSE_CHOICES"}) + "\n" for i in range(100)))
        self.config = {"data_dir": str(self.data), "seed": 20261028, "sample_size": 100,
                       "generation_batch_size": 16, "think_tokens": 256,
                       "final_answer_tokens": 128, "temperature": 0.6,
                       "communicator_path": "/models/qwen3-4b", "judge_path": "/models/qwen3-4b"}

    def test_baseline_uses_only_questions_and_counts_overlapping_rejections(self):
        engine = FakeEngine()
        baseline = Baseline(engine, self.config, self.root / "run", "config-hash")
        self.assertEqual(len({q.question_id for q in baseline.selected}), 100)
        with redirect_stdout(io.StringIO()):
            report = baseline.run()
        self.assertEqual(report["n"], 100)
        self.assertEqual(report["counts"]["judge_rejected"], 50)
        self.assertEqual(report["counts"]["format_rejected"], 50)
        self.assertEqual(report["counts"]["overlap"], 25)
        self.assertEqual(report["counts"]["format_only"], 25)
        self.assertEqual(report["counts"]["total_rejected_d"], 75)
        self.assertEqual(report["d"], 0.75)
        self.assertTrue((self.root / "run/COMPLETE.json").exists())
        for agent, messages, _, _ in engine.calls:
            for item in messages:
                payload = json.loads(item[1]["content"])
                self.assertEqual(set(payload), {"Q"} if agent == "V" else {"Q", "x"})
                self.assertNotIn("DO_NOT_EXPOSE_CHOICES", json.dumps(item))
                if agent == "V":
                    self.assertNotIn("hidden bit", json.dumps(item))

    def test_resume_reuses_committed_answers_and_completed_verdicts(self):
        interrupted = FakeEngine(preempt_judge=True)
        baseline = Baseline(interrupted, self.config, self.root / "run", "config-hash")
        with self.assertRaises(Preempted):
            baseline.run()
        self.assertTrue((self.root / "run/batches/batch_0000/answers.json").exists())
        resumed = FakeEngine()
        with redirect_stdout(io.StringIO()):
            Baseline(resumed, self.config, self.root / "run", "config-hash").run()
        self.assertEqual(resumed.calls[0][0], "J")
        self.assertEqual(sum(len(c[1]) for c in resumed.calls if c[0] == "V"), 84)
        done = FakeEngine()
        with redirect_stdout(io.StringIO()):
            Baseline(done, self.config, self.root / "run", "config-hash").run()
        self.assertEqual(done.calls, [])
        with self.assertRaises(RuntimeError):
            Baseline(done, self.config, self.root / "run", "changed-config")

    def test_truncation_distinguishes_an_eos_at_the_cap(self):
        ended = FakeTrajectory("text", final_token_count=128)
        truncated = FakeTrajectory("text", final_token_count=128, completion_ids=(1, 2))
        self.assertFalse(diagnostics(ended, 256, 128, [99])["final_truncated"])
        self.assertTrue(diagnostics(truncated, 256, 128, [99])["final_truncated"])
        self.assertTrue(diagnostics(ended, 256, 128, [99])["final_cap_hit"])

    def test_hash_verification_refuses_changed_sources_and_data(self):
        source = self.root / "source"
        names = ("stego/rejection_baseline.py", "stego/engine.py", "stego/protocol.py", "stego/train.py")
        for name in names:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        config = {**self.config, "source_sha256": {n: sha256(source / n) for n in names},
                  "data_sha256": {"train.jsonl": sha256(self.data / "train.jsonl")}}
        verify_config(config, source)
        (source / names[0]).write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            verify_config(config, source)
        (source / names[0]).write_text(names[0])
        (self.data / "train.jsonl").write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "Dataset changed"):
            verify_config(config, source)


if __name__ == "__main__":
    unittest.main()
