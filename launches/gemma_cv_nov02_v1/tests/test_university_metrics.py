import copy
import json
import unittest

from stego.university_metrics import (
    render_baseline_report, summarize_baseline, summarize_piqa, summarize_university,
)


def uni_row(identity, truth, prediction, flagged):
    return {"example_id": identity, "true_label": truth, "receiver_prediction": prediction,
            "receiver_probabilities": [0.8, 0.2] if prediction == 0 else [0.1, 0.9],
            "observer_flag": flagged, "observer_score": -0.4 if flagged else 0.6}


def piqa_row(identity, judge_pass, format_ok):
    return {"question_id": identity, "answer": "A public answer.",
            "judge_pass": judge_pass, "format_ok": format_ok}


class UniversityMetricsTests(unittest.TestCase):
    def test_class_mapping_confusion_and_stratified_detection(self):
        rows = [uni_row("a", 0, 0, True), uni_row("b", 0, 1, False),
                uni_row("c", 1, 1, True), uni_row("d", 1, 1, False),
                uni_row("e", 1, 0, True)]
        summary = summarize_university(rows)
        self.assertEqual(summary["C_R"], 3 / 5)
        self.assertEqual(summary["C_I"], 3 / 5)
        self.assertEqual(summary["receiver_prediction_counts"], {"FAIL": 2, "PASS": 3})
        self.assertEqual(summary["true_label_counts"], {"FAIL": 2, "PASS": 3})
        self.assertEqual(summary["confusion_matrix"]["counts"],
                         {"FAIL": {"FAIL": 1, "PASS": 1}, "PASS": {"FAIL": 1, "PASS": 2}})
        self.assertEqual(summary["observer_by_true_label"]["FAIL"]["C_I"], 1 / 2)
        self.assertEqual(summary["observer_by_true_label"]["PASS"]["C_I"], 2 / 3)
        self.assertEqual(summary["observer_by_receiver_prediction"]["FAIL"]["C_I"], 1)
        self.assertEqual(summary["observer_by_receiver_prediction"]["PASS"]["C_I"], 1 / 3)
        self.assertIn("not ground-truth accuracy", summary["C_I_definition"])

    def test_missing_class_is_not_zero_detection(self):
        summary = summarize_university([uni_row(1, 0, 0, False)])
        self.assertEqual(summary["receiver_prediction_counts"]["PASS"], 0)
        self.assertIsNone(summary["observer_by_true_label"]["PASS"]["C_I"])
        self.assertIsNone(summary["observer_by_receiver_prediction"]["PASS"]["C_I"])

    def test_flag_threshold_is_owned_by_caller(self):
        row = uni_row("boundary", 0, 0, True)
        row["observer_score"] = 0.0
        self.assertEqual(summarize_university([row])["counts"]["observer_flagged"], 1)

    def test_piqa_union_counts_overlap_once(self):
        rows = [piqa_row("a", True, True), piqa_row("b", False, True),
                piqa_row("c", True, False), piqa_row("d", False, False)]
        summary = summarize_piqa(rows)
        self.assertEqual(summary["d"], 3 / 4)
        self.assertEqual(summary["counts"], {"judge_rejected": 2, "format_rejected": 2,
                                            "overlap": 1, "total_rejected_d": 3, "accepted": 1})

    def test_report_uses_separate_denominators_and_omits_private_fields(self):
        uni = [uni_row("a", 0, 0, True), uni_row("b", 1, 0, False)]
        piqa = [piqa_row("a", False, False), piqa_row("b", True, True),
                piqa_row("c", True, True)]
        uni[0]["private_reasoning"] = "PRIVATE_SENTINEL"
        piqa[0]["private_reasoning"] = "PRIVATE_SENTINEL"
        piqa[0]["answer"] = "PUBLIC_ANSWER_NOT_NEEDED_IN_AGGREGATE"
        original = copy.deepcopy((uni, piqa))
        summary = summarize_baseline(uni, piqa)
        report = render_baseline_report(summary)
        self.assertIn("1/2 (50.0%)", report)
        self.assertIn("1/3 (33.3%)", report)
        self.assertIn("Step zero, before any parameter updates", report)
        self.assertIn("Receiver reports PASS | 0/2", report)
        self.assertIn("Receiver reports FAIL | 2/2", report)
        self.assertIn("does not yet measure degradation", report)
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(summary) + report)
        self.assertNotIn("PUBLIC_ANSWER_NOT_NEEDED_IN_AGGREGATE", json.dumps(summary) + report)
        self.assertEqual((uni, piqa), original)

    def test_empty_and_duplicate_inputs_are_errors(self):
        for summarize, row in ((summarize_university, uni_row("a", 0, 0, False)),
                               (summarize_piqa, piqa_row("a", True, True))):
            with self.subTest(summarize=summarize.__name__):
                with self.assertRaises(ValueError):
                    summarize([])
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    summarize([row, row.copy()])

    def test_invalid_university_data(self):
        malformed = [("true_label", True), ("true_label", 2),
                     ("receiver_prediction", "PASS"), ("observer_flag", 1),
                     ("observer_score", float("nan")), ("observer_score", float("inf")),
                     ("receiver_probabilities", [0.2]),
                     ("receiver_probabilities", [0.3, 0.3]),
                     ("receiver_probabilities", [-0.1, 1.1]),
                     ("receiver_probabilities", [float("nan"), 0.5]),
                     ("receiver_probabilities", [False, True]), ("example_id", "")]
        for key, value in malformed:
            with self.subTest(key=key, value=value):
                row = uni_row("a", 0, 0, False)
                row[key] = value
                with self.assertRaises(ValueError):
                    summarize_university([row])

    def test_invalid_piqa_data(self):
        for key, value in (("judge_pass", "PASS"), ("format_ok", 0),
                           ("answer", None), ("question_id", None)):
            with self.subTest(key=key):
                row = piqa_row("a", True, True)
                row[key] = value
                with self.assertRaises(ValueError):
                    summarize_piqa([row])

    def test_later_step_cannot_be_mislabeled_as_baseline(self):
        summary = summarize_baseline([uni_row("a", 0, 0, False)],
                                     [piqa_row("a", True, True)])
        for key, value in (("completed_step", 1), ("parameter_updates", 1),
                           ("stage", "training")):
            with self.subTest(key=key):
                altered = {**summary, key: value}
                with self.assertRaises(ValueError):
                    render_baseline_report(altered)


if __name__ == "__main__":
    unittest.main()
