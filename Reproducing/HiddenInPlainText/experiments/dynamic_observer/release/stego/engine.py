"""Single-GPU, three-adapter policy-gradient engine for the PIQA pilot.

Only ``Trajectory.final_text`` is public.  Prompt IDs, private thought, sampled
actions and behavior log-probabilities belong to the training controller.
No TRL trainer is needed: a rollout records the actual temperature-adjusted
sampling probabilities, including the restricted bit/verdict vocabulary.
"""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class EngineConfig:
    think_tokens: int = 256
    final_answer_tokens: int = 128
    generation_batch_size: int = 16
    train_microbatch_size: int = 1
    temperature: float = 0.6
    learning_rate: float = 1e-5
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    clip_epsilon: float = 0.2
    kl_beta: float = 0.01
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    device: str = "cuda:0"
    gradient_checkpointing: bool = True
    decoder_objective: str = "grpo"
    decoder_baseline: float = 0.5
    decoder_ce_weight: float = 1.0

    def __post_init__(self) -> None:
        if type(self.train_microbatch_size) is not int or self.train_microbatch_size < 1:
            raise ValueError("Training microbatch size must be a positive integer")
        if self.decoder_objective not in ("grpo", "fixed_baseline_pg_ce"):
            raise ValueError("Unknown decoder objective")
        if not math.isfinite(self.decoder_baseline) or not 0 <= self.decoder_baseline <= 1:
            raise ValueError("Decoder baseline must be a finite probability")
        if not math.isfinite(self.decoder_ce_weight) or self.decoder_ce_weight < 0:
            raise ValueError("Decoder CE weight must be finite and nonnegative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "EngineConfig":
        if value is None:
            return cls()
        # A caller may keep protocol and engine settings in the same mapping.
        aliases = {"lr": "learning_rate", "rank": "lora_rank",
                   "alpha": "lora_alpha", "batch_size": "generation_batch_size",
                   "final_tokens": "final_answer_tokens", "answer_tokens": "final_answer_tokens", "beta": "kl_beta",
                   "train_microbatch": "train_microbatch_size"}
        fields = cls.__dataclass_fields__
        values = {aliases.get(k, k): v for k, v in value.items()
                  if aliases.get(k, k) in fields}
        return cls(**values)


@dataclass
class Trajectory:
    agent: str
    kind: str
    prompt_ids: list[int]
    completion_ids: list[int]
    action_mask: list[bool]
    # None means the full vocabulary (except thought-stage EOS exclusions).
    supports: list[list[int] | None]
    modes: list[str]
    old_logps: list[float]
    final_text: str
    thought_text: str
    thought_token_count: int
    final_token_count: int
    forced_think_close: bool = False

    def validate(self) -> None:
        n = len(self.completion_ids)
        assert all(len(v) == n for v in
                   (self.action_mask, self.supports, self.modes, self.old_logps))
        assert self.prompt_ids and any(self.action_mask)
        for token, mask, support, logp in zip(
                self.completion_ids, self.action_mask, self.supports, self.old_logps):
            if mask:
                assert math.isfinite(logp)
                assert support is None or token in support


def centered_advantages(rewards: Sequence[float]) -> list[float]:
    """Sum reward components before calling; intentionally no SD division."""
    if not rewards:
        raise ValueError("A GRPO group cannot be empty")
    mean = sum(rewards) / len(rewards)
    return [float(r - mean) for r in rewards]


def allowed_continuations(sequences: Sequence[Sequence[int]],
                          prefix: Sequence[int]) -> list[int]:
    """Next-token support for a finite, prefix-free output language."""
    p = list(prefix)
    return sorted({seq[len(p)] for seq in sequences
                   if len(seq) > len(p) and list(seq[:len(p)]) == p})


class Engine:
    AGENTS = ("A", "B", "I")

    def __init__(self, communicator_path: str, judge_path: str,
                 config: EngineConfig | Mapping[str, Any] | None = None):
        # Lazy imports keep protocol/unit tests usable without a GPU runtime.
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.config = config if isinstance(config, EngineConfig) else EngineConfig.from_mapping(config)
        cfg = self.config
        if cfg.temperature <= 0 or cfg.think_tokens < 0 or cfg.final_answer_tokens < 1:
            raise ValueError("Invalid sampling configuration")
        if cfg.lora_dropout != 0:
            raise ValueError("Dropout must be zero for reproducible behavior-policy ratios")
        self.communicator_path = str(communicator_path)
        self.judge_path = str(judge_path)
        self.device = torch.device(cfg.device)
        self.tokenizer = AutoTokenizer.from_pretrained(communicator_path, local_files_only=True)
        self.judge_tokenizer = AutoTokenizer.from_pretrained(judge_path, local_files_only=True)
        if cfg.decoder_objective == "fixed_baseline_pg_ce":
            self._decoder_label_ids()
        for tokenizer in (self.tokenizer, self.judge_tokenizer):
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
        load_kw = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                       local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(communicator_path, **load_kw).to(self.device)
        lora = LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"])
        self.model = get_peft_model(base, lora, adapter_name="A")
        self.model.add_adapter("B", lora)
        self.model.add_adapter("I", lora)
        if cfg.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
            self.model.enable_input_require_grads()
        self.adapter_parameters = {
            agent: [p for name, p in self.model.named_parameters()
                    if "lora_" in name and f".{agent}." in name]
            for agent in self.AGENTS
        }
        if not all(self.adapter_parameters.values()):
            raise RuntimeError("Could not identify all independent adapter parameters")
        parameter_sets = [set(map(id, values)) for values in self.adapter_parameters.values()]
        assert not any(a & b for i, a in enumerate(parameter_sets) for b in parameter_sets[i + 1:])
        self.optimizers = {
            agent: torch.optim.AdamW(params, lr=cfg.learning_rate, betas=(0.9, 0.999),
                                    eps=1e-8, weight_decay=cfg.weight_decay)
            for agent, params in self.adapter_parameters.items()
        }
        self.judge = AutoModelForCausalLM.from_pretrained(judge_path, **load_kw).to(self.device)
        self.judge.requires_grad_(False)
        self.judge.eval()
        # Standard, non-inference-mode parameters expose mutation version counters.
        self._judge_versions = {name: p._version for name, p in self.judge.named_parameters()}
        self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(
            self.model.get_base_model().forward).parameters
        self._activate("A", train=False)
        self.assert_judge_frozen()

    def _activate(self, agent: str, train: bool) -> None:
        if agent not in self.AGENTS:
            raise ValueError(f"Unknown trainable agent {agent!r}")
        self.model.set_adapter(agent)
        own_ids = set(map(id, self.adapter_parameters[agent]))
        for parameter in self.model.parameters():
            parameter.requires_grad_(id(parameter) in own_ids)
        self.model.train(train)

    def assert_judge_frozen(self) -> None:
        if self.judge.training or any(p.requires_grad for p in self.judge.parameters()):
            raise RuntimeError("Judge must remain frozen and in eval mode")
        judge_ids = {id(p) for p in self.judge.parameters()}
        optimizer_ids = {id(p) for opt in self.optimizers.values()
                         for group in opt.param_groups for p in group["params"]}
        if judge_ids & optimizer_ids:
            raise RuntimeError("Judge parameter appeared in an optimizer")
        if any(p._version != self._judge_versions[name]
               for name, p in self.judge.named_parameters()):
            raise RuntimeError("Judge parameter mutation detected")

    @staticmethod
    def _eos_ids(model: Any, tokenizer: Any) -> list[int]:
        value = model.generation_config.eos_token_id
        if value is None:
            value = tokenizer.eos_token_id
        return list(value) if isinstance(value, (tuple, list)) else [int(value)]

    def generate(self, agent: str, messages_batch: Sequence[list[dict[str, str]]],
                 kind: str = "sender", seed: int | None = None) -> list[Trajectory]:
        """Generate independent trajectories; only the final_text may be sent on."""
        if agent == "J":
            if kind != "judge":
                raise ValueError("J only supports kind='judge'")
            self.assert_judge_frozen()
            model, tokenizer = self.judge, self.judge_tokenizer
        else:
            if kind not in ("sender", "bit"):
                raise ValueError("Communicators/interceptor support sender or bit outputs")
            self._activate(agent, train=False)
            model, tokenizer = self.model, self.tokenizer
        generator = None
        if seed is not None:
            generator = self.torch.Generator(device=self.device).manual_seed(seed)
        trajectories: list[Trajectory] = []
        with self.torch.inference_mode():
            for start in range(0, len(messages_batch), self.config.generation_batch_size):
                batch = messages_batch[start:start + self.config.generation_batch_size]
                trajectories.extend(self._generate_batch(agent, model, tokenizer, batch,
                                                         kind, generator))
        return trajectories

    def _generate_batch(self, agent: str, model: Any, tokenizer: Any,
                        batch: Sequence[list[dict[str, str]]], kind: str,
                        generator: Any) -> list[Trajectory]:
        thinking = self.config.think_tokens > 0
        prompts = [tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=thinking) for messages in batch]
        close_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if len(close_ids) != 1:
            raise RuntimeError("Expected Qwen's single-token </think> boundary")
        close_id = close_ids[0]
        eos_ids = self._eos_ids(model, tokenizer)
        if thinking:
            thoughts = self._sample_stage(model, tokenizer, prompts, "thought",
                self.config.think_tokens, [close_id], generator=generator,
                forbidden=eos_ids)
        else:
            thoughts = [([], [], []) for _ in prompts]
        completions, modes, masks, supports, old_logps, forced = [], [], [], [], [], []
        newline = tokenizer.encode("\n\n", add_special_tokens=False)
        for ids, logps, allowed in thoughts:
            force = thinking and (not ids or ids[-1] != close_id)
            injected = (close_ids if force else []) + (newline if thinking else [])
            completions.append(ids + injected)
            modes.append(["thought"] * len(ids) + ["forced"] * len(injected))
            masks.append([True] * len(ids) + [False] * len(injected))
            supports.append(allowed + [None] * len(injected))
            old_logps.append(logps + [0.0] * len(injected))
            forced.append(force)
        sequences = [prompt + completion for prompt, completion in zip(prompts, completions)]
        if kind == "sender":
            finals = self._sample_stage(model, tokenizer, sequences, "text",
                self.config.final_answer_tokens, eos_ids, generator=generator)
        else:
            labels = ["0", "1"] if kind == "bit" else ["PASS", "FAIL"]
            label_ids = [tokenizer.encode(label, add_special_tokens=False) for label in labels]
            for i, seq in enumerate(label_ids):
                if not seq or any(i != j and other[:len(seq)] == seq
                                  for j, other in enumerate(label_ids)):
                    raise RuntimeError("Constrained label token sequences must be prefix-free")
            finals = self._sample_stage(model, tokenizer, sequences, "constrained",
                max(map(len, label_ids)), [], generator=generator, label_ids=label_ids)
        result = []
        for i, (ids, logps, allowed) in enumerate(finals):
            thought_ids = thoughts[i][0]
            trajectory = Trajectory(
                agent=agent, kind=kind, prompt_ids=prompts[i],
                completion_ids=completions[i] + ids,
                action_mask=masks[i] + [True] * len(ids),
                supports=supports[i] + allowed,
                modes=modes[i] + [("text" if kind == "sender" else "constrained")] * len(ids),
                old_logps=old_logps[i] + logps,
                final_text=tokenizer.decode(ids, skip_special_tokens=True).strip(),
                thought_text=tokenizer.decode(thought_ids, skip_special_tokens=True),
                thought_token_count=len(thought_ids), final_token_count=len(ids),
                forced_think_close=forced[i])
            trajectory.validate()
            result.append(trajectory)
        return result

    def _sample_stage(self, model: Any, tokenizer: Any, sequences: list[list[int]],
                      mode: str, max_tokens: int, stop_ids: list[int], *,
                      generator: Any = None, forbidden: list[int] | None = None,
                      label_ids: list[list[int]] | None = None
                      ) -> list[tuple[list[int], list[float], list[list[int] | None]]]:
        """Cached batched decoding without retaining vocabulary-sized scores."""
        torch = self.torch
        size = len(sequences)
        width = max(map(len, sequences))
        pad = tokenizer.pad_token_id
        input_ids = torch.full((size, width), pad, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(input_ids)
        for i, seq in enumerate(sequences):
            input_ids[i, -len(seq):] = torch.tensor(seq, device=self.device)
            attention_mask[i, -len(seq):] = 1
        tokens = [[] for _ in sequences]
        logps = [[] for _ in sequences]
        supports = [[] for _ in sequences]
        active = [True] * size
        past = None
        stop_set = set(stop_ids)
        for _ in range(max_tokens):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past is not None:
                position_ids = position_ids[:, -1:]
            kwargs = dict(input_ids=input_ids, attention_mask=attention_mask,
                          position_ids=position_ids, past_key_values=past,
                          use_cache=True, return_dict=True)
            # Restrict the output projection to the latest token on both models.
            if "logits_to_keep" in inspect.signature(
                    model.get_base_model().forward if hasattr(model, "get_base_model")
                    else model.forward).parameters:
                kwargs["logits_to_keep"] = 1
            output = model(**kwargs)
            past = output.past_key_values
            scores = output.logits[:, -1, :].float() / self.config.temperature
            del output
            if forbidden:
                scores[:, forbidden] = -torch.inf
            row_supports: list[list[int] | None] = [None] * size
            if mode == "constrained":
                restricted = torch.full_like(scores, -torch.inf)
                for i in range(size):
                    allowed = allowed_continuations(label_ids, tokens[i]) if active[i] else [pad]
                    if not allowed:
                        raise RuntimeError("Empty constrained sampling support")
                    row_supports[i] = allowed
                    restricted[i, allowed] = scores[i, allowed]
                scores = restricted
            probabilities = torch.softmax(scores, dim=-1)
            sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            sampled_logps = scores.gather(1, sampled[:, None]).squeeze(1) - torch.logsumexp(scores, dim=-1)
            sampled_cpu, logps_cpu = sampled.tolist(), sampled_logps.tolist()
            next_active = active.copy()
            for i, was_active in enumerate(active):
                if not was_active:
                    continue
                token = sampled_cpu[i]
                tokens[i].append(token)
                logps[i].append(logps_cpu[i])
                supports[i].append(row_supports[i])
                if token in stop_set or (label_ids is not None and tokens[i] in label_ids):
                    next_active[i] = False
            if not any(next_active):
                break
            # Stopped rows continue as masked pads solely to retain rectangular KV caches.
            step_mask = torch.tensor(active, device=self.device, dtype=attention_mask.dtype)
            input_ids = sampled[:, None]
            input_ids = torch.where(step_mask[:, None].bool(), input_ids,
                                    torch.full_like(input_ids, pad))
            attention_mask = torch.cat((attention_mask, step_mask[:, None]), dim=1)
            active = next_active
        return list(zip(tokens, logps, supports))

    def _decoder_label_ids(self) -> list[int]:
        labels = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("0", "1")]
        if any(len(ids) != 1 for ids in labels) or labels[0] == labels[1]:
            raise ValueError("Hybrid decoder CE requires distinct single-token 0 and 1 labels")
        return [ids[0] for ids in labels]

    def _trajectory_scores(self, trajectory: Trajectory,
                           target_bit: int | None = None) -> tuple[Any, Any]:
        """Singleton wrapper retaining the public scoring helper's original API."""
        return self._trajectory_scores_batch([trajectory], [target_bit])[0]

    def _trajectory_scores_batch(self, trajectories: Sequence[Trajectory],
                                 target_bits: Sequence[int | None] | None = None
                                 ) -> list[tuple[Any, Any]]:
        """Score ragged trajectories in one model forward, preserving action masks.

        The target only selects a loss term from the final two-label distribution.
        The forward input remains the original rollout, including its sampled bit;
        the logits predicting that bit precede it in the causal sequence.

        Left padding aligns completion ends to limit the union of prediction
        positions projected by models supporting ``logits_to_keep``.  Explicit
        position IDs preserve each unpadded sequence's positions.  Each row then
        selects only its own actual actions, excluding padding and forced tokens.
        """
        torch = self.torch
        if not trajectories:
            raise ValueError("Cannot score an empty trajectory microbatch")
        labels = list(target_bits) if target_bits is not None else [None] * len(trajectories)
        if len(labels) != len(trajectories):
            raise ValueError("Expected one target bit or None per scored trajectory")
        for trajectory in trajectories:
            trajectory.validate()
        sequences = [t.prompt_ids + t.completion_ids for t in trajectories]
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), self.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        attention_mask = torch.zeros_like(ids)
        row_actions = []
        row_positions = []
        for row, (trajectory, sequence) in enumerate(zip(trajectories, sequences)):
            offset = width - len(sequence)
            ids[row, offset:] = torch.tensor(sequence, dtype=torch.long, device=self.device)
            attention_mask[row, offset:] = 1
            actions = [i for i, flag in enumerate(trajectory.action_mask) if flag]
            row_actions.append(actions)
            row_positions.append([offset + len(trajectory.prompt_ids) + i - 1 for i in actions])
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        kwargs = dict(input_ids=ids, attention_mask=attention_mask, position_ids=position_ids,
                      use_cache=False, return_dict=True)
        if self._supports_logits_to_keep:
            union_positions = sorted({position for positions in row_positions for position in positions})
            kwargs["logits_to_keep"] = torch.tensor(union_positions, dtype=torch.long,
                                                    device=self.device)
            projected_index = {position: index for index, position in enumerate(union_positions)}
        output = self.model(**kwargs)
        result = []
        for row, (trajectory, actions, positions, target_bit) in enumerate(
                zip(trajectories, row_actions, row_positions, labels)):
            selected = ([projected_index[position] for position in positions]
                        if self._supports_logits_to_keep else positions)
            selected = torch.tensor(selected, dtype=torch.long, device=self.device)
            scores = output.logits[row].index_select(0, selected).float() / self.config.temperature
            result.append(self._action_scores(trajectory, actions, scores, target_bit))
        return result

    def _action_scores(self, trajectory: Trajectory, actions: Sequence[int],
                       scores: Any, target_bit: int | None) -> tuple[Any, Any]:
        """Apply collection-time supports to one row of batched prediction logits."""
        torch = self.torch
        excluded = self._eos_ids(self.model, self.tokenizer)
        # Build a mask without assigning into tensors needed by autograd.
        constraint_mask = torch.zeros_like(scores, dtype=torch.bool)
        for row, action in enumerate(actions):
            support = trajectory.supports[action]
            if support is not None:
                constraint_mask[row] = True
                constraint_mask[row, support] = False
            elif trajectory.modes[action] == "thought":
                constraint_mask[row, excluded] = True
        scores = scores.masked_fill(constraint_mask, -torch.inf)
        targets = torch.tensor([trajectory.completion_ids[i] for i in actions],
                               dtype=torch.long, device=self.device)
        sampled_logps = scores.gather(1, targets[:, None]).squeeze(1) - torch.logsumexp(scores, dim=-1)
        target_logp = None
        if target_bit is not None:
            if type(target_bit) is not int or target_bit not in (0, 1) or trajectory.kind != "bit":
                raise ValueError("A supervised target must be an integer bit on a bit trajectory")
            label_ids = self._decoder_label_ids()
            finals = [row for row, action in enumerate(actions)
                      if trajectory.modes[action] == "constrained"]
            if (len(finals) != 1 or actions[finals[0]] != len(trajectory.completion_ids) - 1
                    or trajectory.final_token_count != 1):
                raise ValueError("Hybrid decoder CE expects exactly one final sampled bit")
            final_row = finals[0]
            final_action = actions[final_row]
            if set(trajectory.supports[final_action] or []) != set(label_ids):
                raise ValueError("Final bit sampling support differs from supervised CE support")
            if (trajectory.final_text not in ("0", "1") or
                    trajectory.completion_ids[final_action] != label_ids[int(trajectory.final_text)]):
                raise ValueError("Sampled bit token and public final text disagree")
            # log_softmax of both logits remains stable even when the sampled bit
            # has probability so close to one that 1 - exp(sampled_logp) rounds to 0.
            target_logp = torch.log_softmax(scores[final_row, label_ids], dim=-1)[target_bit]
        return sampled_logps, target_logp

    def _trajectory_logps(self, trajectory: Trajectory) -> Any:
        """Log probabilities under the same temperature and support as collection."""
        return self._trajectory_scores(trajectory)[0]

    def score_old(self, agent: str, trajectories: Sequence[Trajectory]) -> None:
        """Behavior probabilities are captured during generation; never overwrite them."""
        for trajectory in trajectories:
            if trajectory.agent != agent:
                raise ValueError("Trajectory belongs to a different adapter")
            trajectory.validate()

    def update(self, agent: str, groups: Sequence[Sequence[Trajectory]],
               rewards: Sequence[Sequence[float]],
               group_weights: Sequence[float] | None = None,
               target_bits: Sequence[int | None] | None = None) -> dict[str, float]:
        """Accumulate microbatched role losses, then take exactly one optimizer step.

        Sender trajectories always use the original group-centered GRPO loss.
        Opt-in hybrid bit trajectories use correctness minus a fixed baseline on
        sampled thought actions, and supervised CE on the final bit.  ``target_bits``
        supplies one true bit per decoder group and None for sender groups.  No
        ground-truth bit is passed to generation or added to a decoder's input.
        """
        torch = self.torch
        self._activate(agent, train=True)
        optimizer = self.optimizers[agent]
        optimizer.zero_grad(set_to_none=True)
        if not groups or len(groups) != len(rewards):
            raise ValueError("Expected nonempty matching rollout and reward groups")
        weights = list(group_weights) if group_weights is not None else [1.0] * len(groups)
        if len(weights) != len(groups) or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Invalid group weights")
        labels = list(target_bits) if target_bits is not None else [None] * len(groups)
        if len(labels) != len(groups):
            raise ValueError("Expected one target bit or None per rollout group")
        weight_total = sum(weights)
        metrics = dict(loss=0.0, policy_loss=0.0, kl=0.0, clip_fraction=0.0,
                       ratio_mean=0.0, reward_mean=0.0, active_group_fraction=0.0,
                       sampled_tokens=0.0, trajectories=0.0)
        active_groups = 0
        sender_groups = sender_active_groups = decoder_groups = decoder_active_groups = 0
        decoder_examples = 0
        decoder_ce_sum = decoder_correct_probability_sum = decoder_zero_probability_sum = 0.0
        reasoning_active = reasoning_advantage_sum = reasoning_advantage_abs_sum = 0.0
        decoder_ce_loss = 0.0
        sender_policy_loss = decoder_reasoning_policy_loss = 0.0
        entries = []
        for group, group_rewards, group_weight, target_bit in zip(groups, rewards, weights, labels):
            if len(group) != len(group_rewards) or not group:
                raise ValueError("Each rollout group must match its rewards")
            if any(t.prompt_ids != group[0].prompt_ids or t.kind != group[0].kind for t in group):
                raise ValueError("Every trajectory in a GRPO group needs the identical input and role")
            centered = centered_advantages(group_rewards)
            centered_active = int(any(abs(adv) > 1e-12 for adv in centered))
            active_groups += centered_active
            if group[0].kind == "sender":
                sender_groups += 1
                sender_active_groups += centered_active
            hybrid = (self.config.decoder_objective == "fixed_baseline_pg_ce"
                      and group[0].kind == "bit")
            if hybrid:
                decoder_groups += 1
                decoder_active_groups += centered_active
                if type(target_bit) is not int or target_bit not in (0, 1):
                    raise ValueError("Each hybrid decoder group needs a true integer bit")
                if any(t.final_text not in ("0", "1") or
                       float(r) != float(int(t.final_text) == target_bit)
                       for t, r in zip(group, group_rewards)):
                    raise ValueError("Hybrid decoder rewards must equal sampled-bit correctness")
                advantages = [float(r) - self.config.decoder_baseline for r in group_rewards]
            else:
                if target_bit is not None:
                    raise ValueError("Supervised targets are only accepted for hybrid decoder groups")
                advantages = centered
            sequence_weight = group_weight / weight_total / len(group)
            for trajectory, advantage, reward in zip(group, advantages, group_rewards):
                if trajectory.agent != agent:
                    raise ValueError("Cannot train an adapter on another agent's trajectory")
                trajectory.validate()
                entries.append((trajectory, advantage, reward, sequence_weight,
                                target_bit if hybrid else None, hybrid))
        metrics["microbatches"] = 0.0
        metrics["max_microbatch_trajectories"] = 0.0
        metrics["configured_train_microbatch_size"] = float(self.config.train_microbatch_size)
        for start in range(0, len(entries), self.config.train_microbatch_size):
            batch = entries[start:start + self.config.train_microbatch_size]
            trajectories = [item[0] for item in batch]
            if self.config.kl_beta:
                with torch.no_grad(), self.model.disable_adapter():
                    reference_scores = self._trajectory_scores_batch(trajectories)
            else:
                reference_scores = [(None, None)] * len(batch)
            current_scores = self._trajectory_scores_batch(trajectories, [item[4] for item in batch])
            batch_losses = []
            for entry, current, reference in zip(batch, current_scores, reference_scores):
                trajectory, advantage, reward, sequence_weight, target_bit, hybrid = entry
                current_logps, target_logp = current
                reference_logps = reference[0]
                old = torch.tensor([lp for lp, flag in zip(trajectory.old_logps, trajectory.action_mask)
                                    if flag], device=self.device, dtype=current_logps.dtype)
                ratios = torch.exp(current_logps - old)
                unclipped = ratios * advantage
                clipped = ratios.clamp(1 - self.config.clip_epsilon,
                                       1 + self.config.clip_epsilon) * advantage
                policy_loss = -torch.minimum(unclipped, clipped)
                if hybrid:
                    thought_mask = torch.tensor([mode == "thought" for mode, flag in
                        zip(trajectory.modes, trajectory.action_mask) if flag],
                        device=self.device, dtype=torch.bool)
                    # Forced boundary tokens are not actions; the sampled final bit
                    # receives CE rather than a sampled-action policy-gradient loss.
                    policy_loss = policy_loss.masked_fill(~thought_mask, 0.0)
                if self.config.kl_beta:
                    delta = reference_logps - current_logps
                    kl = torch.exp(delta) - delta - 1
                else:
                    kl = torch.zeros_like(current_logps)
                # Preserve the original role-specific fixed upper bound for PG/KL,
                # rather than normalizing by the rollout's actual sampled length.
                final_cap = self.config.final_answer_tokens if trajectory.kind == "sender" else max(
                    len(self.tokenizer.encode(s, add_special_tokens=False)) for s in ("0", "1"))
                normalizer = self.config.think_tokens + final_cap
                ce = -target_logp if hybrid else current_logps.new_zeros(())
                # Preserve the fixed role-length PG/KL normalizer.  Final-bit CE is
                # averaged over trajectories via sequence_weight, never divided by
                # the thinking budget.  Caller-supplied sender/receiver weights apply
                # to this complete role loss before the single shared-adapter step.
                loss = ((policy_loss + self.config.kl_beta * kl).sum() / normalizer
                        + self.config.decoder_ce_weight * ce) * sequence_weight
                if not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Nonfinite policy-gradient loss; no optimizer step taken")
                batch_losses.append(loss)
                with torch.no_grad():
                    metrics["loss"] += loss.item()
                    metrics["policy_loss"] += policy_loss.sum().item() / normalizer * sequence_weight
                    metrics["kl"] += kl.mean().item() * sequence_weight
                    metrics["clip_fraction"] += ((ratios - 1).abs() > self.config.clip_epsilon).float().mean().item() * sequence_weight
                    metrics["ratio_mean"] += ratios.mean().item() * sequence_weight
                    metrics["reward_mean"] += float(reward) * sequence_weight
                    metrics["sampled_tokens"] += len(old)
                    metrics["trajectories"] += 1
                    if hybrid:
                        decoder_reasoning_policy_loss += policy_loss.sum().item() / normalizer * sequence_weight
                        decoder_examples += 1
                        decoder_ce_sum += ce.item()
                        probability_correct = target_logp.exp().item()
                        decoder_correct_probability_sum += probability_correct
                        decoder_zero_probability_sum += (probability_correct if target_bit == 0
                                                         else 1.0 - probability_correct)
                        decoder_ce_loss += self.config.decoder_ce_weight * ce.item() * sequence_weight
                        reasoning_active += int(bool(thought_mask.any().item()) and abs(advantage) > 1e-12)
                        reasoning_advantage_sum += advantage
                        reasoning_advantage_abs_sum += abs(advantage)
                    elif trajectory.kind == "sender":
                        sender_policy_loss += policy_loss.sum().item() / normalizer * sequence_weight
                del loss, current_logps, policy_loss, kl, ratios, ce, target_logp, unclipped, clipped, old
            # sequence_weight already includes the full update's group and role
            # weights. Summing here (not averaging by microbatch size) preserves
            # the original gradient, including the final partial microbatch.
            batch_loss = torch.stack(batch_losses).sum()
            batch_loss.backward()
            metrics["microbatches"] += 1
            metrics["max_microbatch_trajectories"] = max(metrics["max_microbatch_trajectories"], len(batch))
            del batch_loss, batch_losses, current_scores, reference_scores, current, reference, reference_logps
        norm = torch.nn.utils.clip_grad_norm_(self.adapter_parameters[agent], self.config.max_grad_norm)
        if not torch.isfinite(norm):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite gradient norm; no optimizer step taken")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.model.eval()
        metrics["grad_norm"] = float(norm.item())
        metrics["active_group_fraction"] = active_groups / len(groups)
        # Kept for comparisons with old runs.  For hybrid decoders this is only the
        # counterfactual GRPO signal, not an indicator that the actual loss is active.
        metrics["counterfactual_grpo_active_group_fraction"] = active_groups / len(groups)
        if sender_groups:
            metrics["sender_grpo_active_group_fraction"] = sender_active_groups / sender_groups
            metrics["sender_policy_loss"] = sender_policy_loss
        if decoder_examples:
            metrics.update(
                decoder_examples=float(decoder_examples),
                decoder_ce=decoder_ce_sum / decoder_examples,
                decoder_ce_loss=decoder_ce_loss,
                decoder_reasoning_policy_loss=decoder_reasoning_policy_loss,
                decoder_counterfactual_grpo_active_group_fraction=decoder_active_groups / decoder_groups,
                decoder_correct_probability_mean=decoder_correct_probability_sum / decoder_examples,
                decoder_probability_zero_mean=decoder_zero_probability_sum / decoder_examples,
                reasoning_pg_active_fraction=reasoning_active / decoder_examples,
                reasoning_advantage_mean=reasoning_advantage_sum / decoder_examples,
                reasoning_advantage_abs_mean=reasoning_advantage_abs_sum / decoder_examples,
            )
        self.assert_judge_frozen()
        return metrics

    def save(self, path: str | Path) -> None:
        """Caller atomically publishes the completed-step checkpoint directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.assert_judge_frozen()
        self.model.save_pretrained(path / "adapters", selected_adapters=list(self.AGENTS),
                                   safe_serialization=True)
        self.torch.save({a: opt.state_dict() for a, opt in self.optimizers.items()},
                        path / "optimizers.pt")
        (path / "engine.json").write_text(json.dumps({
            "config": asdict(self.config), "communicator_path": self.communicator_path,
            "judge_path": self.judge_path}, indent=2) + "\n")

    def load(self, path: str | Path) -> None:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        path = Path(path)
        metadata = json.loads((path / "engine.json").read_text())
        if metadata["communicator_path"] != self.communicator_path or metadata["judge_path"] != self.judge_path:
            raise ValueError("Checkpoint base-model paths differ from the active models")
        # Older GRPO checkpoints predate the decoder objective fields; their omitted
        # settings mean the legacy defaults, never the opt-in hybrid objective.
        stored_config = asdict(EngineConfig(**metadata["config"]))
        if stored_config != asdict(self.config):
            raise ValueError("Checkpoint engine configuration differs from the active configuration")
        for agent in self.AGENTS:
            weights = load_file(str(path / "adapters" / agent / "adapter_model.safetensors"))
            set_peft_model_state_dict(self.model, weights, adapter_name=agent)
        states = self.torch.load(path / "optimizers.pt", map_location=self.device, weights_only=True)
        for agent, state in states.items():
            self.optimizers[agent].load_state_dict(state)
        self._activate("A", train=False)
        self.assert_judge_frozen()
