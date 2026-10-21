from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import unittest

from stego.train import Preempted
from stego.university_judge import JudgeStage, config_digest, public_generation_rows


@dataclass
class Trajectory:
    final_text: str
    thought_text: str = "PRIVATE_REASONING_SENTINEL"
    thought_token_count: int = 256
    final_token_count: int = 1
    forced_think_close: bool = True


class FakeEngine:
    def __init__(self, after_generate=None, malformed=False):
        self.optimizers = {}
        self.calls = []
        self.after_generate = after_generate
        self.malformed = malformed
        self.frozen_checks = 0

    def assert_judge_frozen(self):
        self.frozen_checks += 1

    def generate(self, agent, messages, kind, seed):
        self.calls.append((agent, messages, kind, seed))
        rows = []
        for message in messages:
            public = json.loads(message[1]["content"])
            assert set(public) == {"Q", "x"}
            index = int(public["Q"].split()[-1])
            rows.append(Trajectory("invalid" if self.malformed else "PASS" if index % 2 == 0 else "FAIL"))
        if self.after_generate:
            self.after_generate()
        return rows


class UniversityJudgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"judge_path": "/models/Qwen3-4B", "think_tokens": 256,
                       "final_answer_tokens": 128, "generation_batch_size": 16,
                       "temperature": 0.6, "seed": 42}
        self.rows = [{"question_id": f"q:{i}", "question": f"Question {i}",
                      "answer": "Carry it carefully. Use a sturdy bag." if i % 3 else "One sentence.",
                      "generation_seed": 1042, "private_reasoning": "PRIVATE_INPUT_SENTINEL",
                      "answer_diagnostics": {"final_tokens": 12, "thought_text": "PRIVATE_DIAG_SENTINEL"}}
                     for i in range(20)]
        self.write_generations()

    def write_generations(self):
        (self.root / "PIQA_GENERATIONS.json").write_text(json.dumps({
            "config_digest": config_digest(self.config), "rows": self.rows}))

    def test_exact_judge_interface_seeds_and_public_only_commits(self):
        engine = FakeEngine()
        stage = JudgeStage(self.config, self.root)
        seen_config = []

        def factory(config):
            seen_config.append(config)
            return engine

        result = stage.run(factory)
        self.assertTrue(result["complete"])
        self.assertEqual(result["n"], 20)
        self.assertEqual([call[3] for call in engine.calls], [500042, 500058])
        self.assertEqual(seen_config[0]["communicator_path"], self.config["judge_path"])
        for agent, messages, kind, seed in engine.calls:
            self.assertEqual((agent, kind), ("J", "judge"))
            self.assertNotIn("PRIVATE", json.dumps(messages))
        self.assertEqual(sum(row["judge_pass"] for row in result["rows"]), 10)
        self.assertEqual(sum(row["format_ok"] for row in result["rows"]), 13)
        for row in result["rows"]:
            self.assertEqual(row["d"], int(not (row["judge_pass"] and row["format_ok"])))
        self.assertNotIn("PRIVATE", (self.root / "PIQA_JUDGED.json").read_text())
        self.assertGreaterEqual(engine.frozen_checks, 3)
        # Repeated invocation must not load a model after all rows are committed.
        resumed = JudgeStage(self.config, self.root).run(lambda config: self.fail("Reloaded judge"))
        self.assertEqual(resumed, result)

    def test_signal_commits_current_batch_then_resumes_exact_prefix(self):
        stop = {"value": False}
        engine = FakeEngine(after_generate=lambda: stop.update(value=True))
        stage = JudgeStage(self.config, self.root, lambda: stop["value"])
        with self.assertRaises(Preempted):
            stage.run(lambda config: engine)
        saved = json.loads((self.root / "PIQA_JUDGED.json").read_text())
        self.assertEqual(saved["n"], 16)
        self.assertFalse(saved["complete"])
        resumed_engine = FakeEngine()
        final = JudgeStage(self.config, self.root).run(lambda config: resumed_engine)
        self.assertEqual([call[3] for call in resumed_engine.calls], [500058])
        self.assertEqual(final["rows"][:16], saved["rows"])
        self.assertEqual(final["n"], 20)

    def test_malformed_verdict_never_commits_partial_batch(self):
        with self.assertRaisesRegex(RuntimeError, "Malformed"):
            JudgeStage(self.config, self.root).run(lambda config: FakeEngine(malformed=True))
        self.assertFalse((self.root / "PIQA_JUDGED.json").exists())

    def test_input_change_and_duplicate_resume_rows_are_rejected(self):
        JudgeStage(self.config, self.root).run(lambda config: FakeEngine())
        path = self.root / "PIQA_JUDGED.json"
        saved = json.loads(path.read_text())
        saved["rows"][1] = saved["rows"][0]
        path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            JudgeStage(self.config, self.root)
        self.rows[0]["answer"] = "Changed input. Different answer."
        self.write_generations()
        with self.assertRaisesRegex(ValueError, "configuration and input"):
            JudgeStage(self.config, self.root)

    def test_optimizer_presence_is_rejected(self):
        engine = FakeEngine()
        engine.optimizers = {"unexpected": object()}
        with self.assertRaisesRegex(RuntimeError, "no optimizers"):
            JudgeStage(self.config, self.root).run(lambda config: engine)
        self.assertFalse(engine.calls)

    def test_public_input_validation_and_fixed_config(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            public_generation_rows({"rows": [self.rows[0], self.rows[0]]})
        with self.assertRaisesRegex(ValueError, "nonempty"):
            public_generation_rows({"rows": []})
        changed = {**self.config, "temperature": 0.7}
        with self.assertRaisesRegex(ValueError, "temperature"):
            JudgeStage(changed, self.root)


if __name__ == "__main__":
    unittest.main()
