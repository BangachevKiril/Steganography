from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

from stego.prompt_comparison import (ARMS, REQUIRED_SOURCES, REVISED_SENDER_SYSTEM,
    PairedComparison, json_digest, prepare, sender_messages)
from stego.protocol import Episode, build_messages, load_piqa
from stego.rejection_baseline import sha256
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

    def __init__(self, interrupt_judge=False):
        self.calls = []
        self.interrupt_judge = interrupt_judge

    @staticmethod
    def _eos_ids(model, tokenizer):
        return [99]

    def assert_judge_frozen(self):
        pass

    def generate_seeded(self, agent, messages_batch, kind, seeds):
        self.calls.append((agent, messages_batch, kind, list(seeds)))
        if agent == "J" and self.interrupt_judge:
            raise Preempted("after committed sender answers")
        trajectories = []
        for messages in messages_batch:
            payload = json.loads(messages[1]["content"])
            if agent == "V":
                revised = messages[0]["content"] == REVISED_SENDER_SYSTEM
                answer = ("Use a sturdy bag. Keep it upright." if revised else
                          "Use a sturdy bag. The hidden bit is " + str(payload["hidden_bit"]) + ".")
            else:
                assert set(payload) == {"Q", "x"}
                answer = "FAIL" if "hidden bit" in payload["x"] else "PASS"
            trajectories.append(FakeTrajectory(answer))
        return trajectories


class PromptComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        (self.data / "train.jsonl").write_text("".join(json.dumps({
            "goal": f"How should I carry item {i}", "sol1": "DO_NOT_EXPOSE_CHOICES",
            "sol2": "DO_NOT_EXPOSE_CHOICES", "label": "DO_NOT_EXPOSE_LABEL"}) + "\n"
            for i in range(120)))
        self.source = self.root / "source"
        for name in REQUIRED_SOURCES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        sources = {name: sha256(self.source / name) for name in REQUIRED_SOURCES}
        self.training = self.root / "training_config.json"
        training = {"source_sha256": sources, "communicator_path": "/models/qwen3-4b",
                    "judge_path": "/models/qwen3-4b", "think_tokens": 256,
                    "final_answer_tokens": 128, "temperature": 0.6, "generation_batch_size": 16}
        self.training.write_text(json.dumps(training))
        baseline = self.root / "baseline_answers.json"
        selected = random.Random(20261028).sample(load_piqa(self.data / "train.jsonl"), 100)
        baseline.write_text(json.dumps({"rows": [{"question_id": q.question_id, "question": q.question,
            "answer": "DO_NOT_EXPOSE_BASELINE_ANSWER", "judge_verdict": "DO_NOT_EXPOSE_BASELINE_VERDICT",
            "d": 1} for q in selected]}))
        self.config = {**training, "data_dir": str(self.data), "seed": 20260929,
            "baseline_seed": 20261028, "sample_size": 100, "public_bits": 16, "sender": "A",
            "data_sha256": {"train.jsonl": sha256(self.data / "train.jsonl")},
            "reference_training_config": str(self.training),
            "reference_training_config_sha256": sha256(self.training),
            "baseline_answers_path": str(baseline), "baseline_answers_sha256": sha256(baseline)}

    def make(self, engine, config=None, run_name="run"):
        config = config or self.config
        return PairedComparison(engine, config, self.root / run_name, json_digest(config),
                                prepare(config, self.source))

    def test_matched_four_cells_use_only_selected_questions_and_public_outputs(self):
        engine = FakeEngine()
        comparison = self.make(engine)
        with redirect_stdout(io.StringIO()):
            summary = comparison.run()
        self.assertEqual(summary["n_answers"], 400)
        self.assertEqual(summary["paired"]["n_question_bit_pairs"], 200)
        self.assertEqual(summary["arms"]["original"]["counts"]["total_rejected_d"], 200)
        self.assertEqual(summary["arms"]["revised"]["counts"]["total_rejected_d"], 0)
        self.assertEqual(summary["paired"]["metrics"]["total_rejected_d"]["original_only"], 200)
        self.assertEqual(summary["paired"]["metrics"]["total_rejected_d"]["revised_minus_original_rate"], -1)
        result_text = (self.root / "run/RESULTS.json").read_text()
        rows = json.loads(result_text)["rows"]
        public_text = result_text + (self.root / "run/selection.json").read_text()
        for secret in ("PRIVATE_REASONING", "DO_NOT_EXPOSE"):
            self.assertNotIn(secret, public_text)
        groups = {}
        for row in rows:
            groups.setdefault(row["question_id"], []).append(row)
        self.assertEqual(len(groups), 100)
        for group in groups.values():
            self.assertEqual({(r["arm"], r["bit"]) for r in group},
                             {(a, b) for a in ARMS for b in (0, 1)})
            for field in ("public_string", "generation_seed", "judge_seed", "sender", "receiver"):
                self.assertEqual(len({r[field] for r in group}), 1)
            self.assertEqual(len(group[0]["public_string"]), 16)
        for agent, batch, kind, seeds in engine.calls:
            self.assertEqual(len(batch), len(seeds))
            self.assertLessEqual(len(batch), 16)
            for messages in batch:
                payload = json.loads(messages[1]["content"])
                self.assertEqual(set(payload), {"Q", "s", "sender", "receiver", "hidden_bit"}
                                 if agent == "V" else {"Q", "x"})
                self.assertNotIn("PRIVATE_REASONING", json.dumps(messages))
                self.assertNotIn("DO_NOT_EXPOSE", json.dumps(messages))
        episode = Episode("train:1", "Test question", "A", 1, "01" * 8)
        original = sender_messages("original", episode)
        revised = sender_messages("revised", episode)
        self.assertEqual(original, build_messages("sender", episode))
        self.assertEqual(revised[1], original[1])
        self.assertEqual(revised[0]["content"], REVISED_SENDER_SYSTEM)

    def test_preemption_reuses_answers_then_completed_results_and_rejects_changed_identity(self):
        interrupted = FakeEngine(interrupt_judge=True)
        comparison = self.make(interrupted)
        with self.assertRaises(Preempted):
            comparison.run()
        path = self.root / "run/batches/batch_0000/original_m0/answers.json"
        self.assertTrue(path.exists())
        saved_answers = path.read_bytes()
        resumed = FakeEngine()
        with redirect_stdout(io.StringIO()):
            self.make(resumed).run()
        self.assertEqual(resumed.calls[0][0], "J")
        self.assertEqual(sum(len(c[1]) for c in resumed.calls if c[0] == "V"), 384)
        self.assertEqual(saved_answers, path.read_bytes())
        done = FakeEngine()
        with redirect_stdout(io.StringIO()):
            self.make(done).run()
        self.assertEqual(done.calls, [])
        with self.assertRaisesRegex(RuntimeError, "Saved config.json"):
            self.make(FakeEngine(), {**self.config, "run_note": "changed identity"})
        corrupt = json.loads(path.read_text())
        corrupt["rows"][0]["answer"] = "Changed answer. Another sentence."
        path.write_text(json.dumps(corrupt))
        with self.assertRaisesRegex(RuntimeError, "Saved stage does not match"):
            self.make(FakeEngine()).run()

    def test_prepare_rejects_changed_artifacts_selection_and_original_sampler(self):
        prepare(self.config, self.source)
        baseline = Path(self.config["baseline_answers_path"])
        saved = baseline.read_text()
        values = json.loads(saved)
        values["rows"].reverse()
        baseline.write_text(json.dumps(values))
        with self.assertRaisesRegex(RuntimeError, "Input artifact changed"):
            prepare(self.config, self.source)
        config = {**self.config, "baseline_answers_sha256": sha256(baseline)}
        with self.assertRaisesRegex(RuntimeError, "Baseline questions or order"):
            prepare(config, self.source)
        baseline.write_text(saved)
        with self.assertRaisesRegex(ValueError, "think_tokens"):
            prepare({**self.config, "think_tokens": 512}, self.source)
        original = self.source / "stego/engine.py"
        original.write_text("changed sampler")
        with self.assertRaisesRegex(RuntimeError, "Source changed"):
            prepare(self.config, self.source)
        changed_hashes = {**self.config["source_sha256"], "stego/engine.py": sha256(original)}
        with self.assertRaisesRegex(RuntimeError, "differs from the original experiment"):
            prepare({**self.config, "source_sha256": changed_hashes}, self.source)


if __name__ == "__main__":
    unittest.main()
