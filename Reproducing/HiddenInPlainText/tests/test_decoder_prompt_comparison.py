from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import unittest

from stego.decoder_comparison import DecoderComparison, prepare as prepare_decoder
from stego.decoder_prompt_comparison import (ARMS, ROLES, CONTROL_FILES, NEW_VARIANTS, PRIOR,
    UNCERTAINTY, VARIANTS, DecoderPromptComparison, messages, prepare)
from stego.rejection_baseline import sha256
from stego.train import Preempted
from tests import test_decoder_comparison as decoder_fixture
from tests.test_decoder_comparison import FakeEngine, trajectory


class FakePromptEngine(FakeEngine):
    def decode(self, arm, adapter, messages_batch, seeds):
        super().decode(arm, adapter, messages_batch, seeds)
        output = []
        for prompt, seed in zip(messages_batch, seeds):
            system = prompt[0]["content"]
            if UNCERTAINTY in system:
                p0, guess = .5, seed % 2
            elif PRIOR in system and "repeated communication experiment" not in system:
                p0, guess = .55, 0
            elif PRIOR in system:
                p0, guess = .7, 0
            else:
                p0, guess = .65, 0
            output.append(trajectory(guess, p0 - (.1 if arm == "trained" else 0)))
        return output


class DecoderPromptComparisonTests(unittest.TestCase):
    def setUp(self):
        fixture = decoder_fixture.DecoderComparisonTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root, self.source, self.base_config = fixture.root, fixture.source, fixture.config
        self.control = self.root / "control"
        with redirect_stdout(io.StringIO()):
            DecoderComparison(FakeEngine(), self.base_config, self.control,
                              prepare_decoder(self.base_config, self.source)).run()
        (self.control / "MODEL_VERIFICATION.json").write_text(json.dumps({
            "all_parameters_frozen": True, "training_mode": False, "optimizer_count": 0,
            "merged_adapters": False, "label_token_ids": [100, 101],
            "adapters": {agent: {"exact_match": True, "verified_tensors": 504}
                         for agent in ("A", "B", "I")}}))
        module = "stego/decoder_prompt_comparison.py"
        (self.source / module).write_text(module)
        self.config = {**self.base_config, "control_run_path": str(self.control),
            "prompt_variants": list(NEW_VARIANTS),
            "source_sha256": {**self.base_config["source_sha256"], module: sha256(self.source / module)},
            "artifact_sha256": {**self.base_config["artifact_sha256"],
                **{str(self.control / name): sha256(self.control / name) for name in CONTROL_FILES}}}

    def make(self, engine, config=None):
        config = config or self.config
        return DecoderPromptComparison(engine, config, self.root / "new_run", prepare(config, self.source))

    def test_prompt_variants_preserve_json_and_do_not_leak_hidden_bit(self):
        _, selected, _, _ = prepare(self.config, self.source)
        for saved in selected:
            for role in ROLES:
                original = messages("original", role, saved)
                for variant in NEW_VARIANTS:
                    prompt = messages(variant, role, saved)
                    self.assertEqual(prompt[1], original[1])
                    payload = json.loads(prompt[1]["content"])
                    self.assertEqual(set(payload), {"Q", "x", "s", "sender", "receiver"})
                    self.assertNotIn("hidden_bit", payload)
                    self.assertIn("256 private thinking tokens", prompt[0]["content"])
                    if variant == "original_plus_prior":
                        self.assertEqual(prompt[0]["content"], original[0]["content"].rstrip() + "\n\n" + PRIOR)
                    else:
                        self.assertNotIn("If you are SENDER", prompt[0]["content"])
                        self.assertNotIn("Answer Q helpfully", prompt[0]["content"])
                        if role == "receiver":
                            self.assertIn(f"communicator {saved['receiver']}", prompt[0]["content"])
                    self.assertEqual(PRIOR in prompt[0]["content"], variant != "concise")
                    self.assertEqual(UNCERTAINTY in prompt[0]["content"], variant == "concise_plus_prior_uncertainty")

    def test_four_new_conditions_reuse_control_and_match_all_seed_contexts(self):
        engine = FakePromptEngine()
        with redirect_stdout(io.StringIO()):
            summary = self.make(engine).run()
        self.assertEqual(summary["n_decodes"], 5120)
        self.assertEqual(summary["n_new_decodes"], 4096)
        self.assertEqual(summary["n_reused_decodes"], 1024)
        self.assertEqual(summary["n_distinct_answers"], 32)
        self.assertEqual(len(summary["per_question"]), 640)
        self.assertEqual(sum(len(call[2]) for call in engine.calls), 4096)
        control_rows = json.loads((self.control / "RESULTS.json").read_text())["rows"]
        rows = json.loads((self.root / "new_run/RESULTS.json").read_text())["rows"]
        reused = [{k: v for k, v in r.items() if k not in ("variant", "origin")}
                  for r in rows if r["origin"] == "reused_control"]
        self.assertEqual(reused, control_rows)
        grouped = {}
        for row in rows:
            grouped.setdefault((row["question_id"], row["role"], row["replicate"]), []).append(row)
        for group in grouped.values():
            self.assertEqual(len(group), 10)
            for field in ("question", "answer", "bit", "public_string", "seed", "sender", "receiver", "adapter"):
                self.assertEqual(len({r[field] for r in group}), 1)
        for variant in VARIANTS:
            for role in ROLES:
                for arm in ARMS:
                    cell = summary["variants"][variant][role][arm]
                    self.assertEqual(cell["n_decodes"], 256)
                    self.assertEqual(cell["complete_eight_replicate_groups"], 32)
                    varied = variant in ("original", "concise_plus_prior_uncertainty")
                    self.assertEqual(cell["groups_with_both_guesses"], 32 if varied else 0)
                    self.assertEqual(cell["fraction_groups_with_both_guesses"], float(varied))
        difference = summary["paired_delta_vs_original"]["concise_plus_prior_uncertainty"]["receiver"]["vanilla"]
        self.assertEqual(difference["n_matched_decodes"], 256)
        self.assertAlmostEqual(difference["mean_p0"], -.3)
        for path in (self.root / "new_run").rglob("*"):
            if path.is_file():
                self.assertNotIn("PRIVATE_THOUGHT", path.read_text())
        for arm, adapter, prompts, seeds in engine.calls:
            for prompt in prompts:
                system = prompt[0]["content"]
                self.assertTrue(PRIOR in system or "repeated communication experiment" not in system)
        self.assertTrue((self.root / "new_run/per_question.csv").exists())

    def test_control_hash_provenance_and_context_tampering_are_rejected(self):
        prepare(self.config, self.source)
        result_path = self.control / "RESULTS.json"
        original_results = result_path.read_text()
        result = json.loads(original_results)
        result["rows"][0]["seed"] += 1
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            prepare(self.config, self.source)
        updated = {**self.config, "artifact_sha256": {**self.config["artifact_sha256"],
                   str(result_path): sha256(result_path)}}
        with self.assertRaisesRegex(RuntimeError, "completion hash"):
            prepare(updated, self.source)
        complete_path = self.control / "COMPLETE.json"
        original_complete = complete_path.read_text()
        complete = json.loads(original_complete)
        complete["results_sha256"] = sha256(result_path)
        complete_path.write_text(json.dumps(complete))
        updated["artifact_sha256"][str(complete_path)] = sha256(complete_path)
        with self.assertRaisesRegex(RuntimeError, "contexts, conditions, or matched seeds"):
            prepare(updated, self.source)
        result_path.write_text(original_results)
        complete_path.write_text(original_complete)
        manifest_path = self.control / "MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_sha256"]["stego/engine.py"] = "changed"
        manifest_path.write_text(json.dumps(manifest))
        updated = {**self.config, "artifact_sha256": {**self.config["artifact_sha256"],
                   str(manifest_path): sha256(manifest_path)}}
        with self.assertRaisesRegex(RuntimeError, "provenance manifest"):
            prepare(updated, self.source)

    def test_resume_only_generates_uncommitted_new_batches(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(Preempted):
            self.make(FakePromptEngine(interrupt_after=1)).run()
        paths = list((self.root / "new_run/batches").glob("*.json"))
        self.assertEqual(len(paths), 1)
        original = paths[0].read_bytes()
        resumed = FakePromptEngine()
        with redirect_stdout(io.StringIO()):
            self.make(resumed).run()
        self.assertEqual(sum(len(call[2]) for call in resumed.calls), 4080)
        self.assertEqual(paths[0].read_bytes(), original)
        completed = FakePromptEngine()
        with redirect_stdout(io.StringIO()):
            self.make(completed).run()
        self.assertEqual(completed.calls, [])
        with self.assertRaisesRegex(RuntimeError, "Saved config.json"):
            self.make(FakePromptEngine(), {**self.config, "new_note": "different identity"})
        changed = json.loads(paths[0].read_text())
        changed["rows"][0]["conditional_p0"] = .1
        paths[0].write_text(json.dumps(changed))
        with self.assertRaisesRegex(RuntimeError, "Saved batch differs"):
            self.make(FakePromptEngine()).run()


if __name__ == "__main__":
    unittest.main()
