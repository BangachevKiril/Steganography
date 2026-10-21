"""Judge committed university-baseline PIQA answers with the unchanged PIQA judge.

Only public final verdicts and numeric generation diagnostics are retained.
SIGUSR1/SIGTERM request an exit at the next committed batch boundary (code 75).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal

from .protocol import PiqaQuestion, build_messages, check_answer_format, parse_verdict
from .rejection_baseline import FrozenInferenceEngine, diagnostics, sha256
from .train import Preempted, atomic_json


DIAGNOSTIC_KEYS = {
    "thought_tokens", "thought_cap_hit", "forced_think_close", "final_tokens",
    "final_cap_hit", "final_truncated", "input_tokens", "output_tokens",
    "answer_tokens", "answer_token_count", "prompt_tokens", "generated_tokens",
}


def config_digest(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    """Reject accidental changes to this separately authorized judge setup."""
    required = {"think_tokens": 256, "final_answer_tokens": 128, "temperature": 0.6,
                "generation_batch_size": 16, "seed": 42}
    for key, value in required.items():
        if config.get(key) != value or isinstance(config.get(key), bool):
            raise ValueError(f"Judge configuration requires {key}={value}")
    if not isinstance(config.get("judge_path"), str) or not config["judge_path"].strip():
        raise ValueError("judge_path must name the frozen Qwen3-4B checkpoint")


def public_generation_rows(payload):
    """Whitelist public fields rather than copying arbitrary generation records."""
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("PIQA_GENERATIONS.json requires nonempty completed rows")
    public, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Generation rows must be objects")
        for field in ("question_id", "question"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Generation row lacks nonempty {field}")
        if not isinstance(row.get("answer"), str):
            raise ValueError("Generation row lacks a public answer string")
        if row["question_id"] in seen:
            raise ValueError(f"Duplicate question_id: {row['question_id']}")
        seen.add(row["question_id"])
        item = {key: row[key] for key in ("question_id", "question", "answer")}
        if "generation_seed" in row:
            if type(row["generation_seed"]) is not int or row["generation_seed"] < 0:
                raise ValueError("generation_seed must be a nonnegative integer")
            item["generation_seed"] = row["generation_seed"]
        if "answer_diagnostics" in row:
            if not isinstance(row["answer_diagnostics"], dict):
                raise ValueError("answer_diagnostics must be an object")
            item["answer_diagnostics"] = {}
            for key, value in row["answer_diagnostics"].items():
                if key not in DIAGNOSTIC_KEYS:
                    continue
                if type(value) not in (int, float, bool) or not math.isfinite(value):
                    raise ValueError(f"Public diagnostic {key} must be finite numeric/boolean data")
                item["answer_diagnostics"][key] = value
        public.append(item)
    return public


def validate_committed(payload, rows, digest, input_digest, batch_size, seed):
    """A resume must be an exact, whole-batch prefix of this fixed input."""
    if (payload.get("config_digest") != digest
            or payload.get("source_generations_sha256") != input_digest):
        raise ValueError("Committed judgments do not match this configuration and input")
    saved = payload.get("rows")
    if not isinstance(saved, list) or len(saved) > len(rows):
        raise ValueError("Invalid committed judgment rows")
    if len(saved) != len(rows) and len(saved) % batch_size:
        raise ValueError("Committed judgments must end at a complete batch boundary")
    if (payload.get("n") != len(saved) or payload.get("target_questions") != len(rows)
            or payload.get("complete") is not (len(saved) == len(rows))):
        raise ValueError("Committed judgment metadata disagrees with its row count")
    seen = set()
    for index, (generation, judgment) in enumerate(zip(rows, saved)):
        if not isinstance(judgment, dict) or judgment.get("question_id") in seen:
            raise ValueError("Malformed or duplicate committed judgment")
        seen.add(judgment.get("question_id"))
        if any(judgment.get(key) != value for key, value in generation.items()):
            raise ValueError("Committed judgment inputs differ from saved generations")
        if type(judgment.get("judge_pass")) is not bool:
            raise ValueError("Committed judge_pass must be a boolean")
        passed = parse_verdict(judgment.get("judge_verdict", ""))
        if passed is None or passed != judgment["judge_pass"]:
            raise ValueError("Committed judge verdict and boolean disagree")
        expected_format = check_answer_format(generation["answer"])
        if type(judgment.get("format_ok")) is not bool or judgment["format_ok"] != expected_format:
            raise ValueError("Committed format result differs from the unchanged PIQA checker")
        if judgment.get("d") != int(not (passed and expected_format)):
            raise ValueError("Committed PIQA rejection does not equal judge OR format failure")
        if judgment.get("judge_seed") != 500000 + seed + (index // batch_size) * batch_size:
            raise ValueError("Committed judge sampling seed differs from its batch")
        expected_keys = set(generation) | {"judge_seed", "judge_verdict", "judge_pass",
                                          "format_ok", "d", "judge_diagnostics"}
        if set(judgment) != expected_keys:
            raise ValueError("Committed judgment contains unexpected or missing fields")
        diag = judgment.get("judge_diagnostics")
        if (not isinstance(diag, dict) or set(diag) != {
                "thought_tokens", "thought_cap_hit", "forced_think_close", "final_tokens"}
                or any(type(value) not in (int, float, bool) or not math.isfinite(value)
                       for value in diag.values())):
            raise ValueError("Committed judgment diagnostics must contain public counts only")
    return saved


class JudgeStage:
    def __init__(self, config, run_dir, stop_requested=lambda: False):
        validate_config(config)
        self.config, self.run_dir = config, Path(run_dir)
        self.stop_requested = stop_requested
        self.digest = config_digest(config)
        self.input_path = self.run_dir / "PIQA_GENERATIONS.json"
        self.output_path = self.run_dir / "PIQA_JUDGED.json"
        self.input_digest = sha256(self.input_path)
        payload = json.loads(self.input_path.read_text())
        if "config_digest" in payload and payload["config_digest"] != self.digest:
            raise ValueError("PIQA generations were produced under a different configuration")
        self.rows = public_generation_rows(payload)
        self.committed = []
        if self.output_path.exists():
            self.committed = validate_committed(
                json.loads(self.output_path.read_text()), self.rows, self.digest,
                self.input_digest, config["generation_batch_size"], config["seed"])

    def status(self, status, **extra):
        atomic_json(self.run_dir / "PIQA_JUDGE_STATUS.json", {
            "status": status, "config_digest": self.digest,
            "completed_questions": len(self.committed), "target_questions": len(self.rows),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "updated_at": datetime.now(timezone.utc).isoformat(), **extra})

    def check_stop(self):
        if self.stop_requested():
            raise Preempted("Termination requested; stopped between committed judge batches")

    def payload(self):
        return {"config_digest": self.digest, "source_generations_sha256": self.input_digest,
                "n": len(self.committed), "target_questions": len(self.rows),
                "complete": len(self.committed) == len(self.rows), "rows": self.committed}

    def run(self, engine_factory=FrozenInferenceEngine):
        self.check_stop()
        if len(self.committed) == len(self.rows):
            self.status("complete")
            return self.payload()
        self.status("loading_judge")
        engine_config = {**self.config, "communicator_path": self.config["judge_path"]}
        engine = engine_factory(engine_config)
        engine.assert_judge_frozen()
        if engine.optimizers:
            raise RuntimeError("The university PIQA judge stage must have no optimizers")
        self.check_stop()
        batch_size = self.config["generation_batch_size"]
        for start in range(len(self.committed), len(self.rows), batch_size):
            self.check_stop()
            self.status("judging", batch_start=start)
            batch = self.rows[start:start + batch_size]
            messages = [build_messages("judge", PiqaQuestion(row["question_id"], row["question"]),
                                       row["answer"]) for row in batch]
            seed = 500000 + self.config["seed"] + start
            trajectories = engine.generate("J", messages, kind="judge", seed=seed)
            if len(trajectories) != len(batch):
                raise RuntimeError("Judge generation count mismatch")
            judged = []
            for row, trajectory in zip(batch, trajectories):
                passed = parse_verdict(trajectory.final_text)
                if passed is None:
                    raise RuntimeError("Malformed constrained judge verdict")
                format_ok = check_answer_format(row["answer"])
                judged.append({**row, "judge_seed": seed, "judge_verdict": trajectory.final_text,
                               "judge_pass": passed, "format_ok": format_ok,
                               "d": int(not (passed and format_ok)),
                               "judge_diagnostics": diagnostics(trajectory, self.config["think_tokens"])})
            engine.assert_judge_frozen()
            if engine.optimizers:
                raise RuntimeError("An optimizer appeared during frozen judging")
            if sha256(self.input_path) != self.input_digest:
                raise RuntimeError("PIQA generations changed during judging")
            # Never interrupt between extending in-memory state and publishing it.
            # Signal handlers only set a flag; they cannot raise here or mid-decode.
            self.committed.extend(judged)
            atomic_json(self.output_path, self.payload())
            self.status("complete" if len(self.committed) == len(self.rows) else "batch_complete")
            self.check_stop()
        return self.payload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.run_dir.mkdir(parents=True, exist_ok=True)
    termination = {"requested": False}

    def interrupted(signum, frame):
        termination["requested"] = True

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with (args.run_dir / "PIQA_JUDGE.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another worker already owns the PIQA judge stage") from None
        stage = None
        try:
            stage = JudgeStage(config, args.run_dir, lambda: termination["requested"])
            result = stage.run()
            print(json.dumps({"status": "complete", "n": result["n"],
                              "config_digest": result["config_digest"]}), flush=True)
        except Preempted as exc:
            if stage is not None:
                stage.status("preempted", reason=str(exc))
            raise SystemExit(75) from None
        except Exception as exc:
            if stage is not None:
                stage.status("failed", error=repr(exc))
            raise


if __name__ == "__main__":
    main()
