import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from stego.university_training_report import (
    committed_updates, evaluation_summary, pool_updates, render_report, validate_update,
)


def update(number, kind="receiver", n=4, correct=2, flagged=1, passed=3, cycle=1, **extra):
    return {"update": number, "kind": kind, "cycle": cycle, "n": n,
            "receiver_correct": correct, "observer_flagged": flagged,
            "receiver_pass": passed, "receiver_fail": n - passed, **extra}


def evaluation(step=0):
    return {"completed_update": step, "university": {"n": 20, "C_R": 0.7, "C_I": 0.1,
            "counts": {"receiver_correct": 14, "observer_flagged": 2},
            "receiver_prediction_counts": {"PASS": 8, "FAIL": 12}},
            "piqa": {"n": 100, "d": 0.11, "counts": {"judge_rejected": 8,
                     "format_rejected": 5, "overlap": 2, "total_rejected_d": 11}},
            "private_reasoning": "PRIVATE_SENTINEL"}


class UniversityTrainingReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write_metrics(self, rows, tail=""):
        (self.root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows) + tail)

    def write_eval(self, step, payload):
        path = self.root / "evaluations" / f"step_{step:06d}" / "SUMMARY.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))
        marker = {"completed_update": step, "status": "complete",
                  "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for key in ("config_digest", "evaluation_config_digest"):
            if key in payload:
                marker[key] = payload[key]
        (path.parent / "COMPLETE.json").write_text(json.dumps(marker))
        return path

    def test_counts_are_weighted_and_cohorts_stay_separate(self):
        rows = [update(1, n=4, correct=4, flagged=0, passed=4, mean_loss=1.0),
                update(2, n=12, correct=0, flagged=6, passed=3, mean_loss=3.0),
                update(3, kind="sender", n=3, correct=2, flagged=3, passed=1, reward_mean=0.2)]
        self.write_metrics(rows)
        summary = render_report(self.root, 3)
        receiver = summary["by_kind"]["receiver"]
        self.assertEqual((receiver["receiver_correct"], receiver["n"]), (4, 16))
        self.assertEqual(receiver["C_R"], 0.25)
        self.assertEqual(receiver["mean_loss"], 2.5)
        self.assertEqual(summary["by_kind"]["sender"]["C_I"], 1)
        self.assertEqual(len(summary["cycles"]), 2)
        with self.assertRaisesRegex(ValueError, "separate"):
            pool_updates([validate_update(rows[0]), validate_update(rows[2])])

    def test_only_verified_updates_and_evaluations_are_included(self):
        self.write_metrics([update(1), update(2, kind="sender"), {"update": 3, "private_reasoning": "FUTURE"}])
        self.write_eval(2, evaluation(2))
        self.write_eval(3, evaluation(3)).write_text("invalid uncommitted JSON")
        (self.root / "LATEST.json").write_text('{"update": 900}')
        summary = render_report(self.root, 2)
        self.assertEqual(summary["observed_updates"], 2)
        self.assertEqual(summary["evaluation_updates"], [2])
        self.assertAlmostEqual(summary["evaluations"][0]["piqa"]["delta_d_percentage_points"], 3)
        report = (self.root / "report.md").read_text()
        self.assertIn("14/20 (70.0%)", report)
        self.assertIn("11/100 (11.0%)", report)
        self.assertIn("+3.0", report)
        self.assertIn("d₀ = 8.0%", report)
        self.assertIn("not ground-truth accuracy", report)
        self.assertNotIn("PRIVATE_SENTINEL", (self.root / "REPORT.json").read_text() + report)
        with (self.root / "per_update.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual([row["update"] for row in csv_rows], ["1", "2"])
        self.assertEqual(csv_rows[0]["mean_loss"], "")
        self.assertNotIn("d", csv_rows[0])

    def test_truncated_final_record_is_ignored_but_other_corruption_is_not(self):
        self.write_metrics([update(1)], '{"update": 2, "kind":')
        summary = render_report(self.root, 1)
        self.assertEqual(summary["observed_updates"], 1)
        self.assertIn("truncated final", summary["warnings"][0])
        self.write_metrics([], '{"update":\n' + json.dumps(update(2)) + "\n")
        with self.assertRaisesRegex(ValueError, "Malformed"):
            committed_updates(self.root / "metrics.jsonl", 2)

    def test_committed_duplicates_and_invalid_counts_fail(self):
        self.write_metrics([update(1), update(1)])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            render_report(self.root, 1)
        for changes in ({"n": True}, {"receiver_correct": 9}, {"receiver_fail": 4},
                        {"mean_loss": float("nan")}, {"C_R": 0.9}, {"kind": "mixed"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_update({**update(1), **changes})

    def test_zero_observations_are_unmeasured_and_missing_updates_are_listed(self):
        self.write_metrics([update(2, n=0, correct=0, flagged=0, passed=0)])
        summary = render_report(self.root, 3)
        self.assertEqual(summary["missing_updates"], [1, 3])
        self.assertIsNone(summary["by_kind"]["receiver"]["C_R"])
        self.assertIsNone(summary["by_kind"]["sender"]["C_I"])
        self.assertIn("not measured", (self.root / "report.md").read_text())

    def test_evaluation_count_rate_overlap_and_identity_checks(self):
        mismatched = evaluation(3)
        with self.assertRaisesRegex(ValueError, "directory"):
            evaluation_summary(mismatched, 2, 0.08)
        mismatched = evaluation(2)
        mismatched["piqa"]["counts"]["overlap"] = 1
        with self.assertRaisesRegex(ValueError, "overlap once"):
            evaluation_summary(mismatched, 2, 0.08)
        mismatched = evaluation(2)
        mismatched["university"]["C_R"] = 0.8
        with self.assertRaisesRegex(ValueError, "count/rate"):
            evaluation_summary(mismatched, 2, 0.08)

    def test_baseline_only_and_no_raw_private_fields(self):
        self.write_eval(0, evaluation(0))
        summary = render_report(self.root, 0)
        self.assertEqual(summary["observed_updates"], 0)
        self.assertEqual(summary["evaluation_updates"], [0])
        self.assertEqual(summary["missing_updates"], [])
        with (self.root / "per_update.csv").open() as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        row = update(1, private_reasoning="PRIVATE_SENTINEL", answer="UNNEEDED_PUBLIC_TEXT")
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(validate_update(row)))
        self.assertNotIn("UNNEEDED_PUBLIC_TEXT", json.dumps(validate_update(row)))

    def test_receiver_collection_and_filtered_optimization_keep_their_denominators(self):
        row = update(1, n=512, correct=128, flagged=256, passed=400,
                     receiver_training={"n": 256, "correct": 192,
                                        "optimizer_steps": 1, "loss": 0.4})
        self.write_metrics([row])
        summary = render_report(self.root, 1)
        cohort = summary["by_kind"]["receiver"]
        self.assertEqual(cohort["C_R"], 128 / 512)
        self.assertEqual(cohort["C_I"], 256 / 512)
        self.assertEqual(cohort["receiver_training"]["C_R"], 192 / 256)
        self.assertEqual(cohort["receiver_training"]["optimizer_steps"], 1)
        report = (self.root / "report.md").read_text()
        self.assertIn("Receiver collection (before filtering/update)", report)
        self.assertIn("Sender rollouts (before PPO)", report)
        self.assertIn("128/512 (25.0%)", report)
        self.assertIn("192/256 (75.0%)", report)
        with (self.root / "per_update.csv").open() as stream:
            output = next(csv.DictReader(stream))
        self.assertEqual(output["n"], "512")
        self.assertEqual(output["receiver_training_n"], "256")
        self.assertEqual(output["receiver_training_C_R"], "0.75")
        with self.assertRaises(ValueError):
            validate_update({**row, "kind": "sender"})

    def test_uncommitted_summary_is_ignored_and_committed_hash_is_verified(self):
        path = self.write_eval(1, evaluation(1))
        marker = path.parent / "COMPLETE.json"
        saved_marker = marker.read_text()
        marker.unlink()
        path.write_text("unfinished summary is not parseable")
        self.assertEqual(render_report(self.root, 1)["evaluation_updates"], [])
        marker.write_text(saved_marker)
        with self.assertRaisesRegex(ValueError, "summary hash"):
            render_report(self.root, 1)

    def test_committed_evaluation_marker_identity_and_config_are_verified(self):
        payload = {**evaluation(2), "config_digest": "training-digest",
                   "evaluation_config_digest": "evaluation-digest"}
        path = self.write_eval(2, payload)
        marker_path = path.parent / "COMPLETE.json"
        original = json.loads(marker_path.read_text())
        for field, value in (("completed_update", 1), ("config_digest", "different"),
                             ("evaluation_config_digest", "different"), ("status", "running")):
            with self.subTest(field=field):
                marker_path.write_text(json.dumps({**original, field: value}))
                with self.assertRaises(ValueError):
                    render_report(self.root, 2)
        marker_path.write_text(json.dumps(original))
        self.assertEqual(render_report(self.root, 2)["evaluation_updates"], [2])
        (self.root / "config.json").write_text('{"name": "different run"}')
        with self.assertRaisesRegex(ValueError, "run configuration"):
            render_report(self.root, 2)


if __name__ == "__main__":
    unittest.main()
