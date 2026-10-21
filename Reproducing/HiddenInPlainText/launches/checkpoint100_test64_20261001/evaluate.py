"""One prespecified, frozen checkpoint-100 evaluation on 64 new PIQA test goals."""
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized(text):
    return " ".join(text.casefold().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    source = Path(cfg["frozen_source"])
    sys.path.insert(0, str(source))
    from stego.engine import EngineConfig
    from stego.protocol import load_piqa, sample_episodes
    from stego.train import Experiment, atomic_json, summarize, reporting_diagnostics, Preempted
    from stego.rejection_baseline import FrozenInferenceEngine
    from stego.decoder_comparison import verify_loaded_adapters, bit_token_ids, conditional_p0

    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / "EVAL.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    training_path = Path(cfg["training_run"]) / "config.json"
    training = json.loads(training_path.read_text())
    checkpoint = Path(cfg["checkpoint"])
    complete = json.loads((checkpoint / "COMPLETE.json").read_text())
    assert complete["global_step"] == 100
    assert digest(training) == complete["config_digest"] == cfg["training_config_digest"]
    assert (checkpoint / "state.pt").stat().st_size > 0
    for name, expected in training["source_sha256"].items():
        assert file_hash(source / name) == expected, name
    for name, expected in cfg["driver_sha256"].items():
        assert file_hash(Path(args.config).parent / name) == expected, name
    for name, expected in training["data_sha256"].items():
        assert file_hash(Path(training["data_dir"]) / name) == expected, name
    settings = EngineConfig.from_mapping(training)
    metadata = json.loads((checkpoint / "engine/engine.json").read_text())
    assert metadata["config"] == asdict(settings)
    assert all(metadata[k] == training[k] for k in ("communicator_path", "judge_path"))
    assert (settings.think_tokens, settings.final_answer_tokens,
            settings.temperature, settings.generation_batch_size) == (256, 128, .6, 16)
    assert cfg["n"] == 64 and cfg["seed"] == 2026100101

    data = Path(training["data_dir"])
    excluded = {normalized(q.question) for split in ("train.jsonl", "valid.jsonl")
                for q in load_piqa(data / split)}
    tests = load_piqa(data / "tests.jsonl")
    eligible, seen = [], set()
    for q in tests:
        key = normalized(q.question)
        if key not in excluded and key not in seen:
            eligible.append(q)
            seen.add(key)
    episodes = sample_episodes(eligible, cfg["n"], random.Random(cfg["seed"]),
                              public_bits=training["public_bits"],
                              balanced=True, unique_questions=True)
    assert len({normalized(e.question) for e in episodes}) == 64
    assert Counter((e.sender, e.bit) for e in episodes) == {
        ("A", 0): 16, ("A", 1): 16, ("B", 0): 16, ("B", 1): 16}
    selection = {"seed": cfg["seed"], "split": "tests.jsonl", "n": 64,
                 "test_records": len(tests), "eligible_unique_goals": len(eligible),
                 "exclusion": "casefolded whitespace-normalized train and valid goals; duplicate test goals",
                 "episodes": [asdict(e) for e in episodes]}
    inputs = [training_path, checkpoint / "COMPLETE.json", checkpoint / "engine/engine.json",
              Path(cfg["training_run"]) / "PROMPTS.json", data / "tests.jsonl"]
    inputs.extend(checkpoint / "engine/adapters" / agent / name
                  for agent in ("A", "B", "I")
                  for name in ("adapter_config.json", "adapter_model.safetensors"))
    manifest = {"config_digest": digest(cfg), "selection_digest": digest(selection),
                "training_config_digest": digest(training), "checkpoint_step": 100,
                "input_sha256": {str(p): file_hash(p) for p in inputs},
                "generation_settings": asdict(settings), "no_updates": True,
                "rollouts_per_question": {"sender": 1, "receiver": 1, "interceptor": 1, "judge": 1},
                "private_reasoning_saved": False,
                "prespecified_comparison": "Original versus bitwise flipped receiver guesses on identical outputs; keep prior validation separate."}
    for name, value in (("selection.json", selection), ("MANIFEST.json", manifest)):
        path = run / name
        if path.exists():
            assert json.loads(path.read_text()) == value, "Provenance changed: " + name
        else:
            atomic_json(path, value)
    atomic_json(run / "evaluation_config.json", cfg)
    atomic_json(run / "training_config.json", training)
    atomic_json(run / "PROMPTS.json", json.loads((Path(cfg["training_run"]) / "PROMPTS.json").read_text()))
    if args.prepare_only:
        print(json.dumps({"prepared": True, "selection_digest": digest(selection),
                          "test_records": len(tests), "eligible": len(eligible), "selected": len(episodes)}))
        return
    if (run / "COMPLETE.json").exists():
        done = json.loads((run / "COMPLETE.json").read_text())
        assert done["manifest_digest"] == digest(manifest)
        assert done["results_sha256"] == file_hash(run / "results.json")
        print("Evaluation already complete; no generation performed.")
        return

    def status(stage, **fields):
        atomic_json(run / "STATUS.json", {"status": "running", "stage": stage,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "updated_at": datetime.now(timezone.utc).isoformat(), **fields})

    class EvaluationEngine(FrozenInferenceEngine):
        def __init__(self):
            super().__init__(training)
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
            self.model = PeftModel.from_pretrained(self.model,
                str(checkpoint / "engine/adapters/A"), adapter_name="A", is_trainable=False)
            for agent in ("B", "I"):
                self.model.load_adapter(str(checkpoint / "engine/adapters" / agent),
                                        adapter_name=agent, is_trainable=False)
            self.model.requires_grad_(False)
            self.model.eval()
            assert not any(bool(getattr(m, "merged", False)) for m in self.model.modules())
            # PEFT mutates the communicator base: the judge must be a separate vanilla model.
            self.judge = AutoModelForCausalLM.from_pretrained(training["judge_path"],
                torch_dtype=self.torch.bfloat16, attn_implementation="sdpa",
                local_files_only=True).to(self.device)
            self.judge.requires_grad_(False)
            self.judge.eval()
            self._judge_versions = {n: p._version for n, p in self.judge.named_parameters()}
            self.label_ids = bit_token_ids(self.tokenizer)
            self.verification = verify_loaded_adapters(self.model, checkpoint, self.torch)
            self.versions = {n: p._version for n, p in self.model.named_parameters()}
            self.check_frozen()

        def _activate(self, agent, train):
            if train or agent not in ("A", "B", "I"):
                raise RuntimeError("Only frozen checkpoint A/B/I inference is authorized")
            self.model.set_adapter(agent)
            self.model.requires_grad_(False)
            self.model.eval()
            self.check_frozen()

        def check_frozen(self):
            assert not self.optimizers and not self.model.training
            assert not any(p.requires_grad for p in self.model.parameters())
            assert all(p._version == self.versions[n] for n, p in self.model.named_parameters())
            assert not ({id(p) for p in self.judge.parameters()} & {id(p) for p in self.model.parameters()})
            self.assert_judge_frozen()

    def interrupted(signum, frame):
        raise Preempted()
    signal.signal(signal.SIGUSR1, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    start = time.time()
    try:
        status("loading_checkpoint")
        engine = EvaluationEngine()
        atomic_json(run / "LOAD_VERIFICATION.json", {"checkpoint_step": 100,
            "adapters": engine.verification, "all_parameters_frozen": True,
            "optimizer_count": 0, "judge_separate_and_frozen": True})
        experiment = Experiment(engine, training, run)
        experiment.current_step = 100
        torch = engine.torch
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
            send = experiment.generate([(e.sender, experiment.messages("sender", e)) for e in episodes],
                                       "sender", "test_sender")
            answers = [t.final_text for t in send]
            receiver, interceptor, rows = experiment.score_answers(episodes, answers)
        for row, recv, inter in zip(rows, receiver, interceptor):
            row["flipped_receiver_guess"] = 1 - row["receiver_guess"]
            row["flipped_receiver_correct"] = 1 - row["c_r"]
            row["flipped_sender_reward"] = 1 - row["c_r"] - row["c_i"] - row["d"]
            row["receiver_conditional_p0"] = conditional_p0(recv, engine.label_ids)
            row["interceptor_conditional_p0"] = conditional_p0(inter, engine.label_ids)
        engine.check_frozen()
        result = {"manifest_digest": digest(manifest), "checkpoint_step": 100,
                  "split": "tests.jsonl", "seed": cfg["seed"], **summarize(rows),
                  **reporting_diagnostics(rows), "flipped_receiver_correct": sum(1-r["c_r"] for r in rows),
                  "receiver_flip_gain": sum(1-2*r["c_r"] for r in rows) / len(rows),
                  "elapsed_seconds": time.time() - start, "rows": rows}
        atomic_json(run / "results.json", result)
        atomic_json(run / "COMPLETE.json", {"manifest_digest": digest(manifest),
            "results_sha256": file_hash(run / "results.json"), "n": 64,
            "completed_at": datetime.now(timezone.utc).isoformat(), "all_weights_unchanged": True})
        status("complete", status="completed", n=64)
        print(json.dumps({k: result[k] for k in ("n", "C_R", "C_I", "d", "flipped_receiver_correct")}))
    except Preempted:
        status("preempted", status="preempted")
        sys.exit(75)


if __name__ == "__main__":
    main()
