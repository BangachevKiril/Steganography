"""Report public university-training metrics below an audited update boundary.

The caller must independently verify the checkpoint commit marker before passing
--completed-update. This module never reads model tensors or follows LATEST.json.
Training observations and frozen-model evaluation observations stay separate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile

from .university_metrics import OBSERVER_DEFINITION


KINDS = ("receiver", "sender")
COUNT_FIELDS = ("receiver_correct", "observer_flagged", "receiver_pass", "receiver_fail")
CSV_FIELDS = ("update", "kind", "cycle", "n", *COUNT_FIELDS, "C_R", "C_I",
              "receiver_pass_rate", "receiver_fail_rate", "mean_loss", "reward_mean",
              "receiver_training_n", "receiver_training_correct", "receiver_training_C_R",
              "receiver_training_optimizer_steps", "receiver_training_loss")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    return value


def _count(value, n, name):
    value = _integer(value, name)
    if value > n:
        raise ValueError(f"{name} exceeds its denominator")
    return value


def _rate(count, n):
    return count / n if n else None


def validate_update(row):
    """Return a public whitelist, including counts with their true denominator."""
    update = _integer(row.get("update"), "update", 1)
    cycle = _integer(row.get("cycle"), "cycle")
    if row.get("kind") not in KINDS:
        raise ValueError("Update kind must be receiver or sender")
    n = _integer(row.get("n"), "n")
    counts = {key: _count(row.get(key), n, key) for key in COUNT_FIELDS}
    if counts["receiver_pass"] + counts["receiver_fail"] != n:
        raise ValueError("Receiver PASS and FAIL counts must sum to n")
    public = {"update": update, "kind": row["kind"], "cycle": cycle, "n": n, **counts,
              "C_R": _rate(counts["receiver_correct"], n),
              "C_I": _rate(counts["observer_flagged"], n),
              "receiver_pass_rate": _rate(counts["receiver_pass"], n),
              "receiver_fail_rate": _rate(counts["receiver_fail"], n)}
    for key in ("mean_loss", "reward_mean"):
        public[key] = None if row.get(key) is None else _number(row[key], key)
    optimization = row.get("receiver_training")
    public["receiver_training"] = None
    if optimization is not None:
        if row["kind"] != "receiver" or not isinstance(optimization, dict):
            raise ValueError("receiver_training requires a receiver update and a metric object")
        opt_n = _integer(optimization.get("n"), "receiver_training n")
        opt_correct = _count(optimization.get("correct"), opt_n, "receiver_training correct")
        steps = _integer(optimization.get("optimizer_steps"), "receiver_training optimizer_steps")
        loss = optimization.get("loss")
        loss = None if loss is None else _number(loss, "receiver_training loss")
        public["receiver_training"] = {"n": opt_n, "correct": opt_correct,
                                       "C_R": _rate(opt_correct, opt_n),
                                       "optimizer_steps": steps, "loss": loss}
    for key in ("C_R", "C_I"):
        if key in row and row[key] is not None:
            supplied = _number(row[key], key)
            if public[key] is None or not math.isclose(supplied, public[key], abs_tol=1e-7):
                raise ValueError(f"{key} count/rate mismatch")
    return public


def committed_updates(path, completed_update):
    """Ignore future rows and one unfinished final JSONL record; reject other corruption."""
    _integer(completed_update, "completed_update")
    path = Path(path)
    if not path.exists():
        return [], []
    data = path.read_text()
    lines = data.splitlines(keepends=True)
    rows, warnings, seen = [], [], set()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1 and not line.endswith(("\n", "\r")):
                warnings.append(f"Ignored truncated final JSONL record at line {index + 1}.")
                break
            raise ValueError(f"Malformed metrics JSONL record at line {index + 1}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"Metrics JSONL line {index + 1} is not an object")
        update = _integer(raw.get("update"), "update", 1)
        if update > completed_update:
            continue
        if update in seen:
            raise ValueError(f"Duplicate committed update {update}")
        seen.add(update)
        rows.append(validate_update(raw))
    return sorted(rows, key=lambda row: row["update"]), warnings


def pool_updates(rows):
    """Pool counts, not per-update percentages, for one homogeneous cohort."""
    rows = list(rows)
    kinds = {row["kind"] for row in rows}
    if len(kinds) > 1:
        raise ValueError("Receiver collection and sender rollout cohorts must remain separate")
    n = sum(row["n"] for row in rows)
    counts = {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}
    result = {"observed_updates": len(rows), "n": n, **counts,
              "C_R": _rate(counts["receiver_correct"], n),
              "C_I": _rate(counts["observer_flagged"], n),
              "receiver_pass_rate": _rate(counts["receiver_pass"], n),
              "receiver_fail_rate": _rate(counts["receiver_fail"], n)}
    for key in ("mean_loss", "reward_mean"):
        measured = [row for row in rows if row.get(key) is not None and row["n"]]
        denominator = sum(row["n"] for row in measured)
        result[key] = sum(row[key] * row["n"] for row in measured) / denominator if denominator else None
        result[key + "_n"] = denominator
    optimization = [row["receiver_training"] for row in rows if row.get("receiver_training") is not None]
    opt_n = sum(row["n"] for row in optimization)
    opt_correct = sum(row["correct"] for row in optimization)
    loss_rows = [row for row in optimization if row["loss"] is not None and row["n"]]
    loss_n = sum(row["n"] for row in loss_rows)
    result["receiver_training"] = {
        "observed_updates": len(optimization), "n": opt_n, "correct": opt_correct,
        "C_R": _rate(opt_correct, opt_n),
        "optimizer_steps": sum(row["optimizer_steps"] for row in optimization),
        "loss": sum(row["loss"] * row["n"] for row in loss_rows) / loss_n if loss_n else None,
        "loss_n": loss_n,
    }
    return result


def _verify_optional_rate(container, key, expected):
    if container.get(key) is not None:
        if not math.isclose(_number(container[key], key), expected, abs_tol=1e-7):
            raise ValueError(f"Evaluation {key} count/rate mismatch")


def evaluation_summary(raw, update, baseline_d):
    """Extract only public aggregate fields from a completed evaluation summary."""
    for key in ("completed_update", "update"):
        if key in raw and raw[key] != update:
            raise ValueError(f"Evaluation {key} disagrees with its step directory")
    university, piqa = raw["university"], raw["piqa"]
    n = _integer(university["n"], "evaluation university n", 1)
    university_counts = university["counts"]
    correct = _count(university_counts["receiver_correct"], n, "receiver_correct")
    flagged = _count(university_counts["observer_flagged"], n, "observer_flagged")
    predictions = university["receiver_prediction_counts"]
    passed = _count(predictions["PASS"], n, "receiver PASS")
    failed = _count(predictions["FAIL"], n, "receiver FAIL")
    if passed + failed != n:
        raise ValueError("Evaluation PASS and FAIL counts must sum to n")
    _verify_optional_rate(university, "C_R", correct / n)
    _verify_optional_rate(university, "C_I", flagged / n)
    quality_n = _integer(piqa["n"], "evaluation PIQA n", 1)
    quality_counts = {key: _count(piqa["counts"][key], quality_n, key)
                      for key in ("judge_rejected", "format_rejected", "overlap", "total_rejected_d")}
    if quality_counts["overlap"] > min(quality_counts["judge_rejected"], quality_counts["format_rejected"]):
        raise ValueError("PIQA overlap exceeds a constituent rejection count")
    if (quality_counts["total_rejected_d"] != quality_counts["judge_rejected"]
            + quality_counts["format_rejected"] - quality_counts["overlap"]):
        raise ValueError("PIQA d must count judge OR format rejection, with overlap once")
    d = quality_counts["total_rejected_d"] / quality_n
    _verify_optional_rate(piqa, "d", d)
    return {"update": update, "university": {"n": n, "receiver_correct": correct,
            "observer_flagged": flagged, "receiver_pass": passed, "receiver_fail": failed,
            "C_R": correct / n, "C_I": flagged / n},
            "piqa": {"n": quality_n, **quality_counts, "d": d,
                     "delta_d_percentage_points": 100 * (d - baseline_d)}}


def committed_evaluations(run_dir, completed_update, baseline_d):
    summaries, seen = [], set()
    for path in sorted((Path(run_dir) / "evaluations").glob("step_*/SUMMARY.json")):
        match = re.fullmatch(r"step_(\d+)", path.parent.name)
        if not match:
            continue
        update = int(match[1])
        if update > completed_update:
            continue
        if update in seen:
            raise ValueError(f"Duplicate evaluation for update {update}")
        seen.add(update)
        summaries.append(evaluation_summary(json.loads(path.read_text()), update, baseline_d))
    return sorted(summaries, key=lambda row: row["update"])


def _fraction(count, n):
    return f"{count}/{n} ({100 * count / n:.1f}%)" if n else "not measured"


def _display_cohort(row):
    return " | ".join(_fraction(row[key], row["n"]) for key in COUNT_FIELDS)


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def render_report(run_dir, completed_update, baseline_d=0.08):
    run_dir = Path(run_dir)
    _integer(completed_update, "completed_update")
    _number(baseline_d, "baseline_d")
    if not 0 <= baseline_d <= 1:
        raise ValueError("baseline_d must be in [0, 1]")
    rows, warnings = committed_updates(run_dir / "metrics.jsonl", completed_update)
    evaluations = committed_evaluations(run_dir, completed_update, baseline_d)
    missing = sorted(set(range(1, completed_update + 1)) - {row["update"] for row in rows})
    by_kind = {kind: pool_updates([row for row in rows if row["kind"] == kind]) for kind in KINDS}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["cycle"], row["kind"])].append(row)
    cycles = [{"cycle": cycle, "kind": kind, "start_update": min(row["update"] for row in group),
               "end_update": max(row["update"] for row in group), **pool_updates(group)}
              for (cycle, kind), group in sorted(grouped.items())]
    summary = {"schema_version": 1, "completed_update": completed_update,
               "commit_boundary_source": "Caller-supplied independently verified checkpoint update",
               "C_I_definition": OBSERVER_DEFINITION,
               "C_R_definition": "Receiver university-class prediction accuracy; argmax decision",
               "d_definition": "PIQA judge rejection OR format failure, evaluation only",
               "baseline_d": baseline_d, "observed_updates": len(rows), "missing_updates": missing,
               "warnings": warnings, "by_kind": by_kind, "cycles": cycles,
               "evaluation_updates": [row["update"] for row in evaluations], "evaluations": evaluations}
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        record = {key: row.get(key) for key in CSV_FIELDS}
        for key in ("n", "correct", "C_R", "optimizer_steps", "loss"):
            record["receiver_training_" + key] = (row["receiver_training"].get(key)
                                                  if row["receiver_training"] is not None else None)
        writer.writerow({key: "" if value is None else value for key, value in record.items()})
    lines = ["# University experiment: training report", "",
             f"Only observations at or below the independently verified committed update **{completed_update}** are included. "
             "A stale latest-checkpoint pointer or later metric file cannot raise this boundary.", "",
             "C_R is receiver university-class accuracy using argmax: 0 = FAIL (Bad University), "
             "1 = PASS (Good University). " + OBSERVER_DEFINITION, "",
             "Receiver collection observations cover all generated summaries before filtering and the receiver update. "
             "Sender rollout observations are taken before PPO. These observations, receiver optimization "
             "examples, and frozen-model evaluations are separate cohorts. Training totals pool raw counts within a cohort; they are not "
             "counts of unique independent examples. PIQA d is measured only in evaluation, and is "
             "never inferred from a training observer flag.", "",
             "## Training cohorts", "",
             "| Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |",
             "|---|---:|---|---|---|---|"]
    names = {"receiver": "Receiver collection (before filtering/update)",
             "sender": "Sender rollouts (before PPO)"}
    for kind in KINDS:
        cohort = by_kind[kind]
        lines.append(f"| {names[kind]} | {cohort['observed_updates']} | {_display_cohort(cohort)} |")
    if not rows:
        lines += ["", "No committed training observations are available yet."]
    if missing:
        lines += ["", "Missing committed-update observations: " + ", ".join(map(str, missing))
                  + ". These updates are excluded from pooled denominators."]
    optimization = by_kind["receiver"]["receiver_training"]
    if optimization["observed_updates"]:
        loss = "not measured" if optimization["loss"] is None else f"{optimization['loss']:.4f}"
        lines += ["", "## Receiver optimization cohort", "",
                  "This filtered population is used for supervised updates. Its accuracy and loss "
                  "are separate from the all-summary receiver collection measurements above.", "",
                  "| Receiver updates observed | Optimizer steps | Optimization C_R | Mean loss |",
                  "|---:|---:|---|---:|",
                  f"| {optimization['observed_updates']} | {optimization['optimizer_steps']} | "
                  f"{_fraction(optimization['correct'], optimization['n'])} | {loss} |"]
    lines += ["", "## Per-cycle training cohorts", "",
              "| Cycle | Observed update range | Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |",
              "|---:|---|---|---:|---|---|---|---|"]
    for cohort in cycles:
        lines.append(f"| {cohort['cycle']} | {cohort['start_update']}–{cohort['end_update']} | "
                     f"{names[cohort['kind']]} | {cohort['observed_updates']} | {_display_cohort(cohort)} |")
    lines += ["", "## Frozen-model evaluations", "",
              "These measurements use the evaluation cohort with no parameter updates during evaluation. "
              "University and PIQA denominators are separate. Evaluations are reported individually, "
              "without pooling repeated evaluation contexts across checkpoints.", "",
              f"The fixed PIQA baseline is d₀ = {100 * baseline_d:.1f}%. "
              "A positive Δd means more rejected common-sense answers than at baseline; "
              "this descriptive difference alone is not a statistical significance claim.", "",
              "| Update | C_R | C_I flagged | Receiver PASS | Receiver FAIL | PIQA d | Δd (pp) |",
              "|---:|---|---|---|---|---|---:|"]
    for evaluation in evaluations:
        university, piqa = evaluation["university"], evaluation["piqa"]
        lines.append(f"| {evaluation['update']} | {_display_cohort(university)} | "
                     f"{_fraction(piqa['total_rejected_d'], piqa['n'])} | {piqa['delta_d_percentage_points']:+.1f} |")
    if not evaluations:
        lines += ["", "No completed evaluations below the committed boundary are available yet."]
    lines += ["", "| Evaluation update | Judge rejected | Format failed | Both failed |",
              "|---:|---|---|---|"]
    for evaluation in evaluations:
        quality = evaluation["piqa"]
        lines.append(f"| {evaluation['update']} | " + " | ".join(
            _fraction(quality[key], quality["n"]) for key in ("judge_rejected", "format_rejected", "overlap")) + " |")
    if warnings:
        lines += ["", *warnings]
    lines += ["", "[Per-update counts and rates](per_update.csv) · [Machine-readable report](REPORT.json)", ""]
    _atomic_text(run_dir / "per_update.csv", output.getvalue())
    _atomic_text(run_dir / "report.md", "\n".join(lines))
    _atomic_text(run_dir / "REPORT.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--completed-update", type=int, required=True,
                        help="Highest committed update independently verified from a checkpoint")
    parser.add_argument("--baseline-d", type=float, default=0.08)
    args = parser.parse_args()
    print(json.dumps(render_report(args.run_dir, args.completed_update, args.baseline_d), indent=2))


if __name__ == "__main__":
    main()
