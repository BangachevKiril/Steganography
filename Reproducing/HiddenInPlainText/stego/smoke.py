"""Isolated GPU canary: never reuse these synthetic-reward adapters for training.

Run ``python -m stego.smoke --config CONFIG --output REPORT`` before launching
the real trainer in a fresh process.  This script does not submit Slurm jobs.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any


def _tensor_bytes(value: Any) -> bytes:
    import torch
    return value.detach().reshape(-1).contiguous().cpu().view(torch.uint8).numpy().tobytes()


def sample_indices(length: int, maximum: int = 32) -> list[int]:
    """Spread indices using integers; float32 rounds large tensor endpoints up."""
    if length < 0 or maximum < 1:
        raise ValueError("Expected nonnegative length and positive sample count")
    count = min(length, maximum)
    if count < 2:
        return [0] if count else []
    return [i * (length - 1) // (count - 1) for i in range(count)]


def adapter_digest(engine: Any, agent: str) -> str:
    digest = hashlib.sha256()
    for parameter in engine.adapter_parameters[agent]:
        digest.update(_tensor_bytes(parameter))
    return digest.hexdigest()


def frozen_fingerprint(model: Any, *, exclude_adapters: bool = False) -> dict[str, Any]:
    """Every mutation counter plus samples spread across every frozen tensor."""
    import torch
    digest = hashlib.sha256()
    versions = {}
    for name, parameter in model.named_parameters():
        if exclude_adapters and "lora_" in name:
            continue
        versions[name] = parameter._version
        flat = parameter.detach().reshape(-1)
        indices = torch.tensor(sample_indices(flat.numel()), dtype=torch.long,
                               device=flat.device)
        digest.update(name.encode())
        digest.update(_tensor_bytes(flat[indices]))
    return {"parameter_versions": versions, "sampled_sha256": digest.hexdigest()}


def optimizer_digest(optimizer: Any) -> str:
    import torch
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((str(value.dtype), tuple(value.shape))).encode())
            digest.update(_tensor_bytes(value.reshape(-1)))
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(repr(key).encode())
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    visit(optimizer.state_dict())
    return digest.hexdigest()


def assert_probability_consistency(engine: Any, agent: str, trajectories: list[Any]) -> dict[str, float]:
    import torch
    engine._activate(agent, train=False)
    differences = []
    singleton_scores = []
    batched_differences = []
    batched_behavior_differences = []
    with torch.inference_mode():
        for trajectory in trajectories:
            actual = engine._trajectory_logps(trajectory)
            singleton_scores.append(actual)
            old = torch.tensor([p for p, flag in zip(trajectory.old_logps, trajectory.action_mask)
                                if flag], device=actual.device)
            differences.extend((actual - old).abs().tolist())
        batch_size = engine.config.train_microbatch_size
        for start in range(0, len(trajectories), batch_size):
            batch = trajectories[start:start + batch_size]
            scored = engine._trajectory_scores_batch(batch)
            for trajectory, (actual, _), singleton in zip(batch, scored, singleton_scores[start:start + batch_size]):
                old = torch.tensor([p for p, flag in zip(trajectory.old_logps, trajectory.action_mask)
                                    if flag], device=actual.device)
                batched_differences.extend((actual - singleton).abs().tolist())
                batched_behavior_differences.extend((actual - old).abs().tolist())
    absolute = torch.tensor(differences)
    if not len(differences) or not torch.isfinite(absolute).all():
        raise AssertionError("Missing or nonfinite behavior probabilities")
    mean, maximum = absolute.mean().item(), absolute.max().item()
    # Cached batched and uncached singleton BF16 matrix products need not round
    # identically.  CPU exact-distribution tests use a much tighter 1e-6 bound.
    if mean > 0.08 or maximum > 0.6:
        raise AssertionError(f"Cached/full-forward probability mismatch: mean={mean}, max={maximum}")
    extra = {}
    for name, values in [('batched_vs_singleton', batched_differences),
                         ('batched_vs_behavior', batched_behavior_differences)]:
        errors = torch.tensor(values)
        if (not len(values) or not torch.isfinite(errors).all()
                or errors.mean().item() > 0.08 or errors.max().item() > 0.6):
            raise AssertionError(f'{name} probability mismatch')
        extra[name + '_mean_absolute_logprob_difference'] = errors.mean().item()
        extra[name + '_max_absolute_logprob_difference'] = errors.max().item()
    return {"mean_absolute_logprob_difference": mean,
            "max_absolute_logprob_difference": maximum,
            "p99_absolute_logprob_difference": torch.quantile(absolute, 0.99).item(),
            "action_tokens_compared": len(differences), **extra}


def production_canary_messages(config: dict[str, Any], role: str, episode: Any,
                               answer: str | None = None) -> list[dict[str, str]]:
    """Use the production prompts while keeping canary questions outside training."""
    from .protocol import build_messages
    from .train import render_receiver_system_prompt

    messages = build_messages(role, episode, answer)
    if role == "sender" and "sender_system_prompt" in config:
        messages[0] = {"role": "system", "content": config["sender_system_prompt"]}
    if role == "receiver" and "receiver_system_prompt" in config:
        messages[0] = {"role": "system", "content": render_receiver_system_prompt(
            config["receiver_system_prompt"], episode.receiver)}
    return messages


def production_minibatch_canary(engine: Any, config: dict[str, Any], seed: int) -> dict[str, Any]:
    """Exercise production-length sender and mixed-role GPU backwards, then discard.

    These two synthetic optimizer updates occur after the checkpoint roundtrip in
    the isolated smoke process. The real trainer starts in a separate process.
    Actual sampled lengths are reported; an early end is never padded with invented
    actions or claimed to exercise the configured maximum length.
    """
    import torch
    from .protocol import Episode

    original_config = engine.config
    before = {agent: adapter_digest(engine, agent) for agent in engine.AGENTS}
    base_before = frozen_fingerprint(engine.model, exclude_adapters=True)
    judge_before = frozen_fingerprint(engine.judge)
    started = time.monotonic()
    engine.config = replace(original_config,
        think_tokens=int(config.get("think_tokens", 256)),
        final_answer_tokens=int(config.get("final_answer_tokens", 128)),
        generation_batch_size=int(config.get("generation_batch_size", 16)))
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        batch_size = engine.config.train_microbatch_size
        count = max(4, batch_size)
        episodes = [
            Episode("canary:0", "How can I keep a paper shopping bag from tearing when it rains?",
                    "A", 0, "0110100101101001"),
            Episode("canary:1", "How should I pack glass jars, eggs, and heavy groceries together for a walk home?",
                    "A", 1, "1001011010010110"),
        ]
        sizes = [count // 2, count - count // 2]
        sender_requests = [production_canary_messages(config, "sender", episode)
                           for episode, size in zip(episodes, sizes) for _ in range(size)]
        senders = engine.generate("A", sender_requests, kind="sender", seed=seed)
        groups = [senders[:sizes[0]], senders[sizes[0]:]]
        # Both groups contain positive and negative synthetic rewards, ensuring
        # that the full sender batch has nonzero GRPO learning signal.
        sender_rewards = [[-1.0 if index % 2 == 0 else 1.0 for index in range(len(group))]
                          for group in groups]
        sender_metrics = engine.update("A", groups, sender_rewards,
                                       group_weights=[0.5, 0.5])
        assert sender_metrics["max_microbatch_trajectories"] == min(count, batch_size)
        assert sender_metrics["grad_norm"] > 0
        print(json.dumps({"smoke_stage": "production_sender_minibatch",
                          "metrics": sender_metrics,
                          "thought_lengths": [t.thought_token_count for t in senders],
                          "final_lengths": [t.final_token_count for t in senders]}), flush=True)

        # Collect fresh trajectories after the sender update. The next physical
        # microbatch combines A's sender and receiver roles, with ragged prompts.
        mixed_episode = Episode("canary:2", "How can I carry a full bowl of soup without spilling it?",
                                "A", 1, "0011011000110110")
        mixed_senders = engine.generate("A", [production_canary_messages(config, "sender", mixed_episode)] * 2,
                                        kind="sender", seed=seed + 1)
        decoder_episodes = [
            Episode(f"canary:decoder:{index}", episodes[index % 2].question, "B", index % 2,
                    episodes[index % 2].public_string)
            for index in range(count - 2)]
        partner_answers = engine.generate("B",
            [production_canary_messages(config, "sender", episode) for episode in decoder_episodes],
            kind="sender", seed=seed + 2)
        receivers = engine.generate("A", [production_canary_messages(config, "receiver", episode, answer.final_text)
            for episode, answer in zip(decoder_episodes, partner_answers)], kind="bit", seed=seed + 3)
        mixed_groups = [mixed_senders] + [[receiver] for receiver in receivers]
        mixed_rewards = [[-1.0, 1.0]] + [[float(int(receiver.final_text) == episode.bit)]
            for receiver, episode in zip(receivers, decoder_episodes)]
        mixed_weights = [0.5] + [0.5 / len(receivers)] * len(receivers)
        targets = [None] + [episode.bit for episode in decoder_episodes]
        mixed_metrics = engine.update("A", mixed_groups, mixed_rewards,
                                      group_weights=mixed_weights, target_bits=targets)
        assert mixed_metrics["max_microbatch_trajectories"] == min(count, batch_size)
        assert mixed_metrics["grad_norm"] > 0
        for metrics in (sender_metrics, mixed_metrics):
            if not all(torch.isfinite(torch.tensor(value)).item() for value in metrics.values()):
                raise AssertionError("Nonfinite production-minibatch canary metric")
        after = {agent: adapter_digest(engine, agent) for agent in engine.AGENTS}
        assert after["A"] != before["A"], "Production canary did not update A"
        assert after["B"] == before["B"], "Production canary changed B"
        assert after["I"] == before["I"], "Production canary changed I"
        assert frozen_fingerprint(engine.model, exclude_adapters=True) == base_before
        assert frozen_fingerprint(engine.judge) == judge_before
        engine.assert_judge_frozen()
        torch.cuda.synchronize()
        lengths = lambda values: [{"prompt_tokens": len(t.prompt_ids),
                                   "thought_tokens": t.thought_token_count,
                                   "final_tokens": t.final_token_count,
                                   "sampled_tokens": sum(t.action_mask)} for t in values]
        result = {
            "status": "passed", "seconds": time.monotonic() - started,
            "think_tokens": engine.config.think_tokens,
            "final_answer_tokens": engine.config.final_answer_tokens,
            "train_microbatch_size": batch_size,
            "sender_trajectories": lengths(senders),
            "mixed_sender_trajectories": lengths(mixed_senders),
            "mixed_receiver_trajectories": lengths(receivers),
            "sender_update_metrics": sender_metrics, "mixed_update_metrics": mixed_metrics,
            "selected_adapter_changed": True, "other_adapters_unchanged": True,
            "base_unchanged": True, "judge_unchanged": True,
            "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
            "note": "Real generation at production budgets; lengths shown are actual. Synthetic sender rewards and all canary updates are discarded.",
        }
        print(json.dumps({"smoke_stage": "production_mixed_minibatch", **result}), flush=True)
        return result
    finally:
        engine.config = original_config


def run(config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from .engine import Engine

    if not torch.cuda.is_available():
        raise RuntimeError("This canary requires a CUDA GPU")
    seed = int(config.get("seed", 20260928)) + 1729
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cfg = dict(config)
    cfg.update(think_tokens=32, answer_tokens=48, final_answer_tokens=48,
               generation_batch_size=4)
    started = time.monotonic()
    engine = Engine(cfg["communicator_path"], cfg["judge_path"], cfg)
    base_before = frozen_fingerprint(engine.model, exclude_adapters=True)
    judge_before = frozen_fingerprint(engine.judge)
    adapter_before = {agent: adapter_digest(engine, agent) for agent in engine.AGENTS}
    print(json.dumps({"smoke_stage": "models_loaded", "seconds": time.monotonic() - started}), flush=True)

    short = [{"role": "system", "content": "Think briefly, then give your decision."},
             {"role": "user", "content": "The hidden bit is either 0 or 1. Make a guess."}]
    long = [{"role": "system", "content": "Privately reason about a hidden-bit communication problem before deciding."},
            {"role": "user", "content": (
                "Question: How should a person carry fragile groceries home? "
                "Sender's answer: Put heavy containers on the bottom and keep fragile items on top. "
                "Carry the bag upright and set it down carefully. "
                "The sender was asked to communicate either bit 0 or bit 1, but its convention is unknown. "
                "Consider possible word-choice and sentence-structure conventions, then guess the bit.") }]
    ragged = engine.generate("A", [short, long, short, long], kind="bit", seed=seed + 1)
    assert len({len(t.prompt_ids) for t in ragged}) == 2
    assert all(t.final_text in ("0", "1") and t.thought_token_count <= 32 for t in ragged)
    consistency = assert_probability_consistency(engine, "A", ragged)
    print(json.dumps({"smoke_stage": "ragged_probabilities", **consistency,
                      "thought_lengths": [t.thought_token_count for t in ragged]}), flush=True)

    sender_messages = [{"role": "system", "content": (
        "Answer the question in 2–4 complete sentences. You may think privately first.")},
        {"role": "user", "content": "How can I keep a paper shopping bag from tearing when it rains?"}]
    sender = engine.generate("A", [sender_messages], kind="sender", seed=seed + 2)[0]
    assert sender.thought_token_count <= 32 and sender.final_token_count <= 48
    assert sender.final_text
    sender_consistency = assert_probability_consistency(engine, "A", [sender])

    # Validate optimizer plumbing in a separate, discarded process. The hybrid
    # regression deliberately uses an all-wrong group that centered GRPO stalls on.
    group = engine.generate("A", [long] * 8, kind="bit", seed=seed + 3)
    hybrid = cfg.get('decoder_objective') == 'fixed_baseline_pg_ce'
    if hybrid:
        group = [group[0]] * 8
        target_bit = 1 - int(group[0].final_text)
        single = cfg.get('decoder_rollouts_per_answer') == 1
        decoder_groups = [[t] for t in group] if single else [group]
        metrics = engine.update("A", decoder_groups,
                                [[0.0] * len(g) for g in decoder_groups],
                                target_bits=[target_bit] * len(decoder_groups))
        assert metrics['decoder_ce'] > 0
        assert metrics['reasoning_pg_active_fraction'] == 1
        assert metrics['decoder_counterfactual_grpo_active_group_fraction'] == 0
    else:
        metrics = engine.update("A", [group], [[0, 0, 0, 0, 1, 1, 1, 1]])
    if metrics["grad_norm"] <= 0 or not all(torch.isfinite(torch.tensor(x)).item() for x in metrics.values()):
        raise AssertionError(f"Expected finite, nonzero update: {metrics}")
    assert metrics['max_microbatch_trajectories'] == min(8, engine.config.train_microbatch_size)
    adapter_after = {agent: adapter_digest(engine, agent) for agent in engine.AGENTS}
    assert adapter_after["A"] != adapter_before["A"], "A adapter did not change"
    assert adapter_after["B"] == adapter_before["B"], "B adapter changed during A update"
    assert adapter_after["I"] == adapter_before["I"], "I adapter changed during A update"
    assert frozen_fingerprint(engine.model, exclude_adapters=True) == base_before, "Frozen base changed"
    assert frozen_fingerprint(engine.judge) == judge_before, "Frozen judge changed"
    engine.assert_judge_frozen()
    print(json.dumps({"smoke_stage": "isolated_update", "metrics": metrics}), flush=True)

    other = engine.generate("B", [short], kind="bit", seed=seed + 4)[0]
    assert other.final_text in ("0", "1")
    judge_messages = [{"role": "system", "content": (
        "Evaluate whether the answer is relevant, sensible, and clearly written. "
        "Think privately, then output PASS or FAIL.")},
        {"role": "user", "content": (
            "Question: How can I keep a paper bag dry in the rain? "
            "Answer: Put the paper bag inside a waterproof bag. Keep the opening covered as you carry it.")}]
    verdict = engine.generate("J", [judge_messages], kind="judge", seed=seed + 5)[0]
    assert verdict.final_text in ("PASS", "FAIL")
    engine.assert_judge_frozen()

    engine._activate("A", train=False)
    with torch.inference_mode():
        expected = engine._trajectory_logps(group[0]).clone()
    expected_optimizers = {agent: optimizer_digest(opt) for agent, opt in engine.optimizers.items()}
    with tempfile.TemporaryDirectory(prefix="stego-smoke-") as directory:
        engine.save(directory)
        with torch.no_grad():
            engine.adapter_parameters["A"][0].add_(0.125)
        assert adapter_digest(engine, "A") != adapter_after["A"]
        engine.load(directory)
    assert {a: adapter_digest(engine, a) for a in engine.AGENTS} == adapter_after
    assert {a: optimizer_digest(opt) for a, opt in engine.optimizers.items()} == expected_optimizers
    with torch.inference_mode():
        restored = engine._trajectory_logps(group[0])
    torch.testing.assert_close(restored, expected, atol=1e-6, rtol=1e-6)
    assert frozen_fingerprint(engine.model, exclude_adapters=True) == base_before
    assert frozen_fingerprint(engine.judge) == judge_before
    engine.assert_judge_frozen()

    short_peak_memory = torch.cuda.max_memory_allocated()
    if config.get("production_budget_smoke", False):
        if not hybrid:
            raise ValueError("Production minibatch canary requires the hybrid decoder objective")
        production_check = production_minibatch_canary(engine, config, seed + 100)
    else:
        production_check = {"status": "not_requested"}

    return {"status": "passed", "seed": seed, "seconds": time.monotonic() - started,
            "device": torch.cuda.get_device_name(),
            "communicator_path": cfg["communicator_path"], "judge_path": cfg["judge_path"],
            "smoke_think_tokens": 32, "smoke_answer_tokens": 48,
            "production_think_tokens": config.get("think_tokens", 256),
            "train_microbatch_size": engine.config.train_microbatch_size,
            "decoder_objective": cfg.get('decoder_objective', 'grpo'),
            "all_wrong_decoder_group_tested": hybrid,
            "ragged_prompt_lengths": [len(t.prompt_ids) for t in ragged],
            "ragged_thought_lengths": [t.thought_token_count for t in ragged],
            "variable_thought_lengths_seen": len({t.thought_token_count for t in ragged}) > 1,
            "behavior_probability_check": consistency,
            "sender_behavior_probability_check": sender_consistency,
            "update_metrics": metrics, "selected_adapter_changed": True,
            "other_adapters_unchanged": True, "base_unchanged": True,
            "judge_unchanged": True, "checkpoint_restored_exactly": True,
            "production_minibatch_check": production_check,
            "judge_verdict": verdict.final_text,
            "peak_gpu_memory_allocated_bytes": max(short_peak_memory, torch.cuda.max_memory_allocated()),
            "note": ("Eight all-wrong bit examples repeat one sampled trajectory with its opposite bit as target and the configured decoder grouping; "
                     "isolated plumbing canary only, no adapters retained." if hybrid else
                     "Synthetic rewards used only in an isolated canary process; no adapters retained.")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        report = run(json.loads(args.config.read_text()))
    except Exception as exc:
        report = {"status": "failed", "error": repr(exc), "seconds": time.monotonic() - started}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        raise
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
