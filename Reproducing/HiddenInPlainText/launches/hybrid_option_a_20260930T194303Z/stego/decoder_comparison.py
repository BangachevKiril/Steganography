"""Frozen vanilla versus checkpoint-40 decoding of the same saved answers.

P(0) is conditional on each decode's independently sampled private thinking.
Its mean over eight replicates estimates P(0) marginalized over that thinking;
no private thinking text or token sequences are saved by this experiment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import time

from .engine import EngineConfig
from .paired_inference import SeededFrozenInferenceEngine
from .protocol import Episode, build_messages
from .rejection_baseline import sha256
from .train import Preempted, atomic_json


ARMS = ("vanilla", "trained")
ROLES = ("receiver", "interceptor")
REQUIRED_SOURCES = {"stego/decoder_comparison.py", "stego/paired_inference.py",
    "stego/engine.py", "stego/protocol.py", "stego/rejection_baseline.py", "stego/train.py"}
PROBABILITY_NOTE = ("conditional_p0 is the two-label probability of 0 after that arm's own "
    "sampled private thinking, at temperature 0.6. Mean P(0) over eight independently sampled "
    "thinking paths estimates the marginal decoder probability. The experiment contains 32 "
    "distinct saved answers, not 256 independent questions. No private thinking is retained.")


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bit_token_ids(tokenizer):
    labels = [tokenizer.encode(label, add_special_tokens=False) for label in ("0", "1")]
    if any(len(label) != 1 for label in labels) or labels[0] == labels[1]:
        raise RuntimeError("P(0) extraction requires distinct single-token labels 0 and 1")
    return tuple(label[0] for label in labels)


def conditional_p0(trajectory, label_ids):
    """Extract the normalized two-way final probability, excluding thinking tokens."""
    if len(label_ids) != 2 or len(set(label_ids)) != 2:
        raise ValueError("Exactly two distinct label token IDs are required")
    fields = (trajectory.completion_ids, trajectory.action_mask, trajectory.supports,
              trajectory.modes, trajectory.old_logps)
    if len({len(field) for field in fields}) != 1:
        raise RuntimeError("Malformed trajectory lengths")
    indices = [i for i, mode in enumerate(trajectory.modes) if mode == "constrained"]
    if trajectory.kind != "bit" or trajectory.final_token_count != 1 or len(indices) != 1:
        raise RuntimeError("Expected exactly one constrained final-bit action")
    index = indices[0]
    guess = trajectory.final_text.strip()
    support = trajectory.supports[index]
    if (index != len(trajectory.completion_ids) - 1 or not trajectory.action_mask[index]
            or support is None or len(support) != 2 or set(support) != set(label_ids)
            or guess not in ("0", "1")
            or trajectory.completion_ids[index] != label_ids[int(guess)]):
        raise RuntimeError("Final action is not a correctly labeled two-way bit decision")
    logp = trajectory.old_logps[index]
    if not math.isfinite(logp) or logp > 1e-6:
        raise RuntimeError("Invalid normalized final-bit log probability")
    logp = min(0.0, logp)
    return math.exp(logp) if guess == "0" else -math.expm1(logp)


def verify_loaded_adapters(model, checkpoint, torch):
    """Check the canonical tensors after all loads, including PEFT export extras."""
    from safetensors.torch import load_file
    verified = {}
    for agent in ("A", "B", "I"):
        path = Path(checkpoint) / "engine/adapters" / agent / "adapter_model.safetensors"
        saved = load_file(str(path), device="cpu")
        matched = []
        for name, parameter in model.named_parameters():
            canonical = None
            for family in ("lora_A", "lora_B"):
                marker = f".{family}.{agent}."
                if marker in name:
                    canonical = name.replace(marker, f".{family}.")
                    break
            if canonical is None:
                continue
            if canonical not in saved or parameter.shape != saved[canonical].shape:
                raise RuntimeError(f"Loaded {agent} adapter parameter lacks matching checkpoint tensor: {name}")
            expected = saved[canonical].to(device=parameter.device, dtype=parameter.dtype)
            if not torch.equal(parameter.detach(), expected):
                raise RuntimeError(f"Loaded {agent} adapter parameter differs from checkpoint: {name}")
            matched.append(canonical)
        if len(matched) != 504 or len(set(matched)) != 504:
            raise RuntimeError(f"Expected 504 verified Qwen3-4B LoRA tensors for {agent}, found {len(matched)}")
        verified[agent] = {"verified_tensors": len(matched), "checkpoint_tensors": len(saved),
                           "extra_export_tensors": len(set(saved) - set(matched)), "exact_match": True}
    return verified


class DecoderEngine(SeededFrozenInferenceEngine):
    """One frozen base plus A/B/I adapters; no optimizer and no judge inference."""

    def __init__(self, training, checkpoint):
        super().__init__(training)
        from peft import PeftModel
        adapters = Path(checkpoint) / "engine/adapters"
        self.model = PeftModel.from_pretrained(self.model, str(adapters / "A"),
                                              adapter_name="A", is_trainable=False)
        for agent in ("B", "I"):
            self.model.load_adapter(str(adapters / agent), adapter_name=agent, is_trainable=False)
        if any(bool(getattr(module, "merged", False)) for module in self.model.modules()):
            raise RuntimeError("Merged adapters would invalidate the vanilla comparison")
        self.model.requires_grad_(False)
        self.model.eval()
        # The inherited invariant checks guard every base and adapter parameter.
        # This alias is never used to perform judge inference.
        self.judge = self.model
        self._judge_versions = {name: p._version for name, p in self.model.named_parameters()}
        self.label_ids = bit_token_ids(self.tokenizer)
        self.assert_judge_frozen()
        self.verification = {"all_parameters_frozen": True, "training_mode": self.model.training,
            "optimizer_count": len(self.optimizers), "merged_adapters": False,
            "label_token_ids": list(self.label_ids),
            "adapters": verify_loaded_adapters(self.model, checkpoint, self.torch)}

    def _activate(self, agent, train):
        if train or agent not in ("V", "A", "B", "I"):
            raise ValueError("Only frozen vanilla or A/B/I decoding is supported")
        if agent != "V":
            self.model.set_adapter(agent)
            self.model.requires_grad_(False)
            self.model.eval()
        self.assert_judge_frozen()

    def generate(self, *args, **kwargs):
        raise RuntimeError("Use the paired decode interface; judge/sender generation is disabled")

    def generate_seeded(self, agent, messages_batch, kind, seeds):
        if agent == "J" or kind != "bit":
            raise ValueError("This experiment supports bit decoding only")
        return super().generate_seeded(agent, messages_batch, kind, seeds)

    def decode(self, arm, adapter, messages, seeds):
        if arm not in ARMS or adapter not in ("A", "B", "I"):
            raise ValueError("Unknown decoder condition")
        self.assert_judge_frozen()
        try:
            if arm == "vanilla":
                with self.model.disable_adapter():
                    return self.generate_seeded("V", messages, kind="bit", seeds=seeds)
            return self.generate_seeded(adapter, messages, kind="bit", seeds=seeds)
        finally:
            # PEFT context restoration/selection can change requires_grad flags.
            self.model.requires_grad_(False)
            self.model.eval()
            self.assert_judge_frozen()


def prepare(config, source_root=None):
    source_root = Path(source_root or Path(__file__).resolve().parent.parent)
    if not REQUIRED_SOURCES <= set(config.get("source_sha256", {})):
        raise ValueError("Decoder comparison requires hashes for all sampler and protocol sources")
    for name, expected in config["source_sha256"].items():
        if sha256(source_root / name) != expected:
            raise RuntimeError(f"Source changed since preparation: {name}")
    checkpoint = Path(config["checkpoint_path"])
    required_inputs = [Path(config["training_config_path"]), Path(config["validation_path"]),
                       checkpoint / "COMPLETE.json", checkpoint / "engine/engine.json"]
    if not {str(path) for path in required_inputs} <= set(config.get("artifact_sha256", {})):
        raise ValueError("Hashes must include training config, validation, checkpoint COMPLETE, and engine metadata")
    for path, expected in config["artifact_sha256"].items():
        if sha256(path) != expected:
            raise RuntimeError(f"Input artifact changed since preparation: {path}")
    if (config["checkpoint_step"], config["replicates"], config["seed"]) != (40, 8, 20260930):
        raise ValueError("Use authorized checkpoint 40, eight replicates, and seed 20260930")
    if not (checkpoint / "state.pt").is_file():
        raise RuntimeError("Checkpoint lacks its committed state.pt")
    training = json.loads(Path(config["training_config_path"]).read_text())
    complete = json.loads((checkpoint / "COMPLETE.json").read_text())
    metadata = json.loads((checkpoint / "engine/engine.json").read_text())
    if complete["global_step"] != 40 or complete["config_digest"] != json_digest(training):
        raise RuntimeError("Checkpoint provenance does not match the requested training run and step")
    settings = EngineConfig.from_mapping(training)
    if (settings.think_tokens, settings.temperature, settings.generation_batch_size) != (256, 0.6, 16):
        raise ValueError("Decoder inference must preserve T=256, temperature=.6, and batch size 16")
    if metadata["config"] != asdict(settings) or any(
            metadata[field] != training[field] for field in ("communicator_path", "judge_path")):
        raise RuntimeError("Checkpoint engine metadata differs from the training configuration")
    for name in ("stego/engine.py", "stego/protocol.py", "stego/paired_inference.py"):
        if training["source_sha256"].get(name) != config["source_sha256"][name]:
            raise RuntimeError(f"Decoder sampler or prompts differ from the training source: {name}")
    validation = json.loads(Path(config["validation_path"]).read_text())
    rows = validation["rows"]
    if validation["global_step"] != 40 or validation["n"] != 32 or len(rows) != 32:
        raise ValueError("Use all 32 saved step-40 validation answers")
    if len({r["question_id"] for r in rows}) != 32:
        raise ValueError("Expected 32 distinct saved validation questions")
    selection = []
    for index, row in enumerate(rows):
        episode = Episode(row["question_id"], row["question"], row["sender"], row["bit"], row["public_string"])
        if len(episode.public_string) != 16 or not isinstance(row["answer"], str):
            raise ValueError("Invalid saved answer or public string")
        if row["d"] != int(not (row["judge_pass"] and row["format_ok"])):
            raise ValueError("Saved answer-quality fields disagree")
        selection.append({"answer_index": index, "question_id": episode.question_id,
            "question": episode.question, "sender": episode.sender, "receiver": episode.receiver,
            "bit": episode.bit, "public_string": episode.public_string, "answer": row["answer"],
            "d": row["d"], "judge_pass": row["judge_pass"], "format_ok": row["format_ok"]})
    inputs = list(required_inputs)
    for adapter in ("A", "B", "I"):
        inputs.extend(checkpoint / "engine/adapters" / adapter / name
                      for name in ("adapter_config.json", "adapter_model.safetensors"))
    manifest = {"config_digest": json_digest(config), "training_config_digest": json_digest(training),
        "checkpoint_step": 40, "n_saved_answers": 32, "replicates": 8, "n_decodes": 1024,
        "input_sha256": {str(path): sha256(path) for path in inputs},
        "checkpoint_state_present": True, "source_sha256": config["source_sha256"],
        "selection_digest": json_digest(selection), "probability_note": PROBABILITY_NOTE,
        "generation_settings": asdict(settings), "no_updates": True, "no_judge_reruns": True,
        "private_reasoning_saved": False}
    return training, selection, manifest


def decoder_messages(role, saved):
    if role not in ROLES:
        raise ValueError("Only receiver and interceptor inputs are allowed")
    episode = Episode(saved["question_id"], saved["question"], saved["sender"],
                      saved["bit"], saved["public_string"])
    return build_messages(role, episode, saved["answer"])


def aggregate(rows):
    if not rows:
        return {"n_decodes": 0, "n_answers": 0}
    n = len(rows)
    correct = sum(r["guess"] == r["bit"] for r in rows)
    return {"n_decodes": n, "n_answers": len({r["question_id"] for r in rows}),
        "mean_p0": sum(r["conditional_p0"] for r in rows) / n,
        "sampled_zero_count": sum(r["guess"] == 0 for r in rows),
        "sampled_zero_rate": sum(r["guess"] == 0 for r in rows) / n,
        "correct_count": correct, "accuracy": correct / n,
        "mean_probability_correct": sum(r["conditional_p0"] if r["bit"] == 0
                                         else 1 - r["conditional_p0"] for r in rows) / n,
        "forced_think_close_count": sum(r["forced_think_close"] for r in rows),
        "mean_thought_tokens": sum(r["thought_tokens"] for r in rows) / n}


def summarize(rows):
    cells = {}
    per_question = []
    for role in ROLES:
        cells[role] = {}
        for arm in ARMS:
            subset = [r for r in rows if r["role"] == role and r["arm"] == arm]
            cells[role][arm] = {**aggregate(subset),
                "by_true_bit": {str(bit): aggregate([r for r in subset if r["bit"] == bit]) for bit in (0, 1)},
                "by_receiver_identity": {agent: aggregate([r for r in subset if r["receiver"] == agent])
                                         for agent in ("A", "B")},
                "by_saved_quality": {str(d): aggregate([r for r in subset if r["d"] == d]) for d in (0, 1)}}
            for index in sorted({r["answer_index"] for r in subset}):
                group = [r for r in subset if r["answer_index"] == index]
                first = group[0]
                per_question.append({key: first[key] for key in
                    ("answer_index", "question_id", "question", "answer", "bit", "sender", "receiver", "d")}
                    | {"role": role, "arm": arm, **aggregate(group)})
    changes = {}
    for role in ROLES:
        vanilla, trained = cells[role]["vanilla"], cells[role]["trained"]
        changes[role] = {key: trained[key] - vanilla[key] for key in
            ("mean_p0", "sampled_zero_rate", "accuracy", "mean_probability_correct")
            if key in vanilla and key in trained}
    return {"n_decodes": len(rows), "cells": cells, "trained_minus_vanilla": changes,
            "per_question": per_question, "probability_note": PROBABILITY_NOTE}


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_report(run_dir, summary):
    per_question = summary["per_question"]
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=list(per_question[0]))
    writer.writeheader()
    writer.writerows(per_question)
    atomic_text(Path(run_dir) / "per_question.csv", csv_buffer.getvalue())
    lines = ["# Frozen decoder comparison", "",
        "The same 32 checkpoint-40 validation answers were decoded by the vanilla base and trained "
        "checkpoint-40 adapters. Each answer has eight independent replicates for each role and arm. "
        "Seeds, questions, public strings, sender identities, and answers are matched across arms. "
        "Receiver decoding uses the opposite communicator's adapter; interceptor decoding uses I.", "",
        "No training updates were performed. Saved answer-quality verdicts were retained; the judge was not rerun.", "",
        PROBABILITY_NOTE, "", "| Role | Model | Mean P(0) | Sampled 0 | Correct | Thinking cutoff |",
        "|---|---|---:|---:|---:|---:|"]
    for role in ROLES:
        for arm in ARMS:
            row = summary["cells"][role][arm]
            n = row["n_decodes"]
            lines.append(f"| {role} | {arm} | {row['mean_p0']:.4f} | "
                f"{row['sampled_zero_count']}/{n} ({row['sampled_zero_rate']:.1%}) | "
                f"{row['correct_count']}/{n} ({row['accuracy']:.1%}) | {row['forced_think_close_count']}/{n} |")
    lines.extend(["", "Trained minus vanilla mean P(0): " + "; ".join(
        f"{role} {summary['trained_minus_vanilla'][role]['mean_p0']:+.4f}" for role in ROLES) + "."])
    lines.extend(["", "`SUMMARY.json` includes results by true bit, receiver identity, and saved answer quality. "
        "`per_question.csv` contains each question/role/arm average, and `RESULTS.json` contains all public "
        "replicate decisions. Multiple replicates reduce sampling noise on these same answers; they do not "
        "establish performance on additional questions.", ""])
    atomic_text(Path(run_dir) / "report.md", "\n".join(lines))


class DecoderComparison:
    def __init__(self, engine, config, run_dir, prepared=None):
        self.engine, self.config, self.run_dir = engine, config, Path(run_dir)
        self.training, self.selection, self.manifest = prepared or prepare(config)
        self.digest = json_digest(config)
        if self.manifest["config_digest"] != self.digest:
            raise RuntimeError("Prepared inputs belong to a different configuration")
        prompts = {f"receiver_{agent}": decoder_messages("receiver", {
            **self.selection[0], "sender": "B" if agent == "A" else "A"})[0]["content"]
            for agent in ("A", "B")}
        prompts["interceptor"] = decoder_messages("interceptor", self.selection[0])[0]["content"]
        for name, value in (("config.json", config), ("selection.json", self.selection),
                            ("MANIFEST.json", self.manifest), ("PROMPTS.json", prompts)):
            path = self.run_dir / name
            if path.exists() and json.loads(path.read_text()) != value:
                raise RuntimeError(f"Saved {name} differs from this run")
            atomic_json(path, value)

    def status(self, stage, **details):
        atomic_json(self.run_dir / "STATUS.json", {"status": "running", "stage": stage,
            "config_digest": self.digest, "updated_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), **details})

    def requests(self, role, arm):
        result = []
        for saved in self.selection:
            adapter = saved["receiver"] if role == "receiver" else "I"
            for replicate in range(self.config["replicates"]):
                seed = self.config["seed"] + 100000 * ROLES.index(role) + saved["answer_index"] * 8 + replicate
                result.append({**saved, "role": role, "arm": arm, "adapter": adapter,
                               "replicate": replicate, "seed": seed})
        return result

    def run(self):
        started, all_rows = time.monotonic(), []
        batch_size = EngineConfig.from_mapping(self.training).generation_batch_size
        self.engine.assert_judge_frozen()
        for role in ROLES:
            for arm in ARMS:
                requests = self.requests(role, arm)
                for adapter in (("A", "B") if role == "receiver" else ("I",)):
                    own = [r for r in requests if r["adapter"] == adapter]
                    for number, start in enumerate(range(0, len(own), batch_size)):
                        batch = own[start:start + batch_size]
                        cell = f"{role}_{arm}_{adapter}_{number:04d}"
                        path = self.run_dir / "batches" / (cell + ".json")
                        if path.exists():
                            committed = json.loads(path.read_text())
                            rows = committed["rows"]
                            if (committed["config_digest"] != self.digest or committed["request_digest"] != json_digest(batch)
                                    or committed["rows_digest"] != json_digest(rows) or len(rows) != len(batch)
                                    or any(any(row[key] != value for key, value in request.items())
                                           for row, request in zip(rows, batch))):
                                raise RuntimeError(f"Saved batch differs from the requested decode condition: {cell}")
                        else:
                            self.status("decoding", cell=cell, completed_decodes=len(all_rows))
                            messages = [decoder_messages(role, r) for r in batch]
                            trajectories = self.engine.decode(arm, adapter, messages, [r["seed"] for r in batch])
                            if len(trajectories) != len(batch):
                                raise RuntimeError("Decoder output count mismatch")
                            rows = [{**request, "guess": int(t.final_text.strip()),
                                     "conditional_p0": conditional_p0(t, self.engine.label_ids),
                                     "thought_tokens": t.thought_token_count,
                                     "forced_think_close": t.forced_think_close,
                                     "thought_cap_hit": t.thought_token_count >= self.training["think_tokens"]}
                                    for request, t in zip(batch, trajectories)]
                            del trajectories
                            atomic_json(path, {"config_digest": self.digest, "request_digest": json_digest(batch),
                                               "rows_digest": json_digest(rows), "rows": rows})
                        self.engine.assert_judge_frozen()
                        all_rows.extend(rows)
                        self.status("batch_complete", cell=cell, completed_decodes=len(all_rows))
                        print(json.dumps({"completed_decodes": len(all_rows), "cell": cell}), flush=True)
        all_rows.sort(key=lambda r: (r["answer_index"], ROLES.index(r["role"]), ARMS.index(r["arm"]), r["replicate"]))
        summary = {"config_digest": self.digest, **summarize(all_rows),
                   "seconds_this_invocation": time.monotonic() - started}
        atomic_json(self.run_dir / "RESULTS.json", {"config_digest": self.digest,
                                                  "probability_note": PROBABILITY_NOTE, "rows": all_rows})
        atomic_json(self.run_dir / "SUMMARY.json", summary)
        write_report(self.run_dir, summary)
        complete = {"config_digest": self.digest, "n_saved_answers": len(self.selection),
            "n_decodes": len(all_rows), "finished_at": datetime.now(timezone.utc).isoformat(),
            "results_sha256": sha256(self.run_dir / "RESULTS.json")}
        atomic_json(self.run_dir / "COMPLETE.json", complete)
        atomic_json(self.run_dir / "STATUS.json", {"status": "complete", **complete})
        return summary


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
            raise RuntimeError("Another worker owns this decoder comparison") from None

        def interrupted(signum, frame):
            raise Preempted(f"Received signal {signum}")
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, interrupted)
        try:
            saved = args.run_dir / "config.json"
            if saved.exists() and json.loads(saved.read_text()) != config:
                raise RuntimeError("Refusing to change the saved decoder configuration")
            prepared = prepare(config)
            atomic_json(saved, config)
            atomic_json(args.run_dir / "STATUS.json", {"status": "loading_model", "config_digest": json_digest(config)})
            engine = DecoderEngine(prepared[0], config["checkpoint_path"])
            atomic_json(args.run_dir / "MODEL_VERIFICATION.json", engine.verification)
            DecoderComparison(engine, config, args.run_dir, prepared).run()
        except Preempted as exc:
            atomic_json(args.run_dir / "STATUS.json", {"status": "preempted", "reason": str(exc),
                "resume": "Reuse atomically committed decode batches; regenerate only incomplete batches"})
            raise SystemExit(75)
        except Exception as exc:
            atomic_json(args.run_dir / "STATUS.json", {"status": "failed", "error": repr(exc)})
            raise


if __name__ == "__main__":
    main()
