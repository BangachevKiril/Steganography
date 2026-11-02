"""Inference-only, paired-bit sender probe on selected baseline failures."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal

from .engine import Engine
from .protocol import Episode, build_messages, check_answer_format, parse_verdict, phase_for_step
from .train import Preempted, atomic_json, write_trace


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ProbeEngine(Engine):
    """Load the exact training engine/checkpoint while disabling all learning."""

    def _activate(self, agent, train):
        if train:
            raise RuntimeError("The paired sender probe is inference only")
        super()._activate(agent, train=False)
        self.model.requires_grad_(False)

    def update(self, *args, **kwargs):
        raise RuntimeError("The paired sender probe cannot update weights")


def prepare(config, source_root=None):
    source_root = Path(source_root or Path(__file__).resolve().parent.parent)
    for relative, expected in config.get("source_sha256", {}).items():
        if file_hash(source_root / relative) != expected:
            raise RuntimeError(f"Source changed since preparation: {relative}")
    for path, expected in config.get("artifact_sha256", {}).items():
        if file_hash(path) != expected:
            raise RuntimeError(f"Input artifact changed since preparation: {path}")
    training_path = Path(config["training_config_path"])
    checkpoint = Path(config["checkpoint_path"])
    baseline_path = Path(config["baseline_answers_path"])
    training = json.loads(training_path.read_text())
    complete = json.loads((checkpoint / "COMPLETE.json").read_text())
    checkpoint_step = config.get("checkpoint_step", 20)
    if complete["global_step"] != checkpoint_step:
        raise ValueError(f"The requested sender checkpoint must be global step {checkpoint_step}")
    if complete["config_digest"] != json_digest(training):
        raise RuntimeError("Checkpoint provenance does not match the original training configuration")
    if config.get("communicator", "A") not in ("A", "B"):
        raise ValueError("The sender must be communicator A or B")
    requested = config["question_ids"]
    if len(requested) != 5 or len(set(requested)) != 5:
        raise ValueError("Select exactly five distinct baseline-rejected questions")
    baseline = json.loads(baseline_path.read_text())["rows"]
    index = {row["question_id"]: row for row in baseline}
    selected = [index[question_id] for question_id in requested]
    if any(row["d"] != 1 for row in selected):
        raise ValueError("Every selected question must have been rejected in the vanilla baseline")
    # These are the actual sampler and prompt implementation used by training.
    for relative in ("stego/engine.py", "stego/protocol.py"):
        expected = training.get("source_sha256", {}).get(relative)
        if expected is not None and file_hash(source_root / relative) != expected:
            raise RuntimeError(f"Probe differs from the original experiment: {relative}")
    inputs = [training_path, baseline_path, checkpoint / "COMPLETE.json",
              checkpoint / "engine/engine.json", checkpoint / "engine/optimizers.pt"]
    inputs.extend(sorted((checkpoint / "engine/adapters").glob("*/adapter_model.safetensors")))
    manifest = {"config_digest": json_digest(config), "training_config_digest": json_digest(training),
                "checkpoint_global_step": checkpoint_step,
                "communicator_joint_updates": sum(phase_for_step(step) == "joint"
                    for step in range(1, checkpoint_step + 1)),
                "communicator": config.get("communicator", "A"),
                "input_sha256": {str(path): file_hash(path) for path in inputs},
                "source_sha256": {str(path.relative_to(source_root)): file_hash(path)
                    for path in sorted((source_root / "stego").glob("*.py"))},
                "selection": "Five selected vanilla-baseline failures; qualitative paired probe",
                "same_public_string_within_pair": True,
                "same_initial_generation_seed_within_pair": True}
    return training, selected, manifest


class PairedProbe:
    def __init__(self, engine, config, selected, run_dir, manifest):
        self.engine, self.config, self.selected = engine, config, selected
        self.run_dir, self.manifest = Path(run_dir), manifest
        self.digest = json_digest(manifest)
        path = self.run_dir / "MANIFEST.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise RuntimeError("Probe provenance changed; refusing to reuse saved pairs")
        atomic_json(path, manifest)

    def status(self, stage, **details):
        atomic_json(self.run_dir / "STATUS.json", {
            "status": "running", "stage": stage, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "updated_at": datetime.now(timezone.utc).isoformat(), **details})

    def run(self):
        rng = random.Random(self.config["seed"])
        sender = self.config.get("communicator", "A")
        pairs = []
        for number, baseline in enumerate(self.selected):
            public_string = format(rng.getrandbits(16), "016b")
            generation_seed = self.config["seed"] + 1000 + number
            judge_seed = self.config["seed"] + 1000000 + number
            path = self.run_dir / "pairs" / f"pair_{number:04d}.json"
            if path.exists():
                pair = json.loads(path.read_text())
                if pair["manifest_digest"] != self.digest or pair["question_id"] != baseline["question_id"]:
                    raise RuntimeError("Saved pair does not belong to this probe")
                if file_hash(self.run_dir / pair["private_trace"]) != pair["private_trace_sha256"]:
                    raise RuntimeError("Saved pair's private trace changed")
            else:
                rows, private = [], []
                for bit in (0, 1):
                    episode = Episode(baseline["question_id"], baseline["question"], sender, bit, public_string)
                    self.status("answering", question_index=number, question_id=episode.question_id, bit=bit)
                    sender_messages = build_messages("sender", episode)
                    # Separate calls reset the RNG to the same starting state.
                    answer = self.engine.generate(sender, [sender_messages], kind="sender", seed=generation_seed)[0]
                    judge_messages = build_messages("judge", episode, answer.final_text)
                    self.status("judging", question_index=number, question_id=episode.question_id, bit=bit)
                    judgment = self.engine.generate("J", [judge_messages], kind="judge", seed=judge_seed)[0]
                    passed = parse_verdict(judgment.final_text)
                    if passed is None:
                        raise RuntimeError("Malformed constrained judge verdict")
                    format_ok = check_answer_format(answer.final_text)
                    eos = self.engine._eos_ids(self.engine.model, self.engine.tokenizer)
                    truncated = (answer.final_token_count >= self.engine.config.final_answer_tokens
                                 and answer.completion_ids[-1] not in eos)
                    rows.append({"bit": bit, "answer": answer.final_text,
                                 "judge_verdict": judgment.final_text, "judge_pass": passed,
                                 "format_ok": format_ok, "d": int(not (passed and format_ok)),
                                 "answer_thought_tokens": answer.thought_token_count,
                                 "answer_final_tokens": answer.final_token_count,
                                 "answer_final_truncated": bool(truncated),
                                 "answer_forced_think_close": answer.forced_think_close,
                                 "judge_thought_tokens": judgment.thought_token_count,
                                 "judge_forced_think_close": judgment.forced_think_close})
                    private.append({"bit": bit, "sender_messages": sender_messages,
                                    "judge_messages": judge_messages,
                                    "answer_trajectory": asdict(answer), "judge_trajectory": asdict(judgment)})
                trace = f"private/pair_{number:04d}.json.gz"
                write_trace(self.run_dir / trace, {"manifest_digest": self.digest, "traces": private})
                pair = {"manifest_digest": self.digest, "question_id": baseline["question_id"],
                        "question": baseline["question"], "sender": sender, "receiver": episode.receiver,
                        "public_string": public_string, "generation_seed": generation_seed,
                        "judge_seed": judge_seed, "baseline": baseline, "outputs": rows,
                        "private_trace": trace, "private_trace_sha256": file_hash(self.run_dir / trace)}
                atomic_json(path, pair)
            self.engine.assert_judge_frozen()
            pairs.append(pair)
            self.status("pair_complete", completed_pairs=len(pairs), total_pairs=len(self.selected))
            print(json.dumps({"completed_pairs": len(pairs), "question_id": pair["question_id"]}), flush=True)
        results = {"manifest_digest": self.digest,
                   "checkpoint_global_step": self.manifest["checkpoint_global_step"],
                   "communicator": sender, "n_questions": len(pairs), "n_answers": 2 * len(pairs),
                   "pairs": pairs,
                   "rows": [{"question_id": pair["question_id"], "question": pair["question"],
                             "sender": sender, "public_string": pair["public_string"], **row}
                            for pair in pairs for row in pair["outputs"]]}
        atomic_json(self.run_dir / "RESULTS.json", results)
        atomic_json(self.run_dir / "COMPLETE.json", {"manifest_digest": self.digest,
            "n_questions": len(pairs), "n_answers": 2 * len(pairs),
            "finished_at": datetime.now(timezone.utc).isoformat()})
        atomic_json(self.run_dir / "STATUS.json", {"status": "complete", "n_answers": 2 * len(pairs)})
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / "RUN.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another worker owns this probe directory") from None

        def interrupt(signum, frame):
            raise Preempted(f"Received signal {signum}")
        for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, interrupt)
        try:
            saved = args.run_dir / "config.json"
            if saved.exists() and json.loads(saved.read_text()) != config:
                raise RuntimeError("Refusing to change a saved probe configuration")
            training, selected, manifest = prepare(config)
            atomic_json(saved, config)
            atomic_json(args.run_dir / "STATUS.json", {"status": "loading_checkpoint"})
            engine = ProbeEngine(training["communicator_path"], training["judge_path"], training)
            engine.load(Path(config["checkpoint_path"]) / "engine")
            PairedProbe(engine, config, selected, args.run_dir, manifest).run()
        except Preempted as exc:
            atomic_json(args.run_dir / "STATUS.json", {"status": "preempted", "reason": str(exc)})
            raise SystemExit(75)
        except Exception as exc:
            atomic_json(args.run_dir / "STATUS.json", {"status": "failed", "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
