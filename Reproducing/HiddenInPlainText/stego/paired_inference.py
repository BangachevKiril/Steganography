"""Frozen inference with one independent random stream per supplied example.

The original model loader and trajectory construction stay unchanged. Only the
sampling draw is performed per active row, allowing matched seeds across prompt
conditions without coupling an example to the lengths of its batch companions.
Numerical model outputs can still vary across batch shapes on real accelerators;
this class guarantees stream isolation, not bitwise GPU arithmetic invariance.
"""
from __future__ import annotations

import inspect
from typing import Any, Sequence

from .engine import Trajectory, allowed_continuations
from .rejection_baseline import FrozenInferenceEngine


class SeededFrozenInferenceEngine(FrozenInferenceEngine):
    """The pretrained model, no adapters or updates, with per-example seeds."""

    def generate_seeded(
        self, agent: str, messages_batch: Sequence[list[dict[str, str]]],
        kind: str, seeds: list[int],
    ) -> list[Trajectory]:
        if len(seeds) != len(messages_batch):
            raise ValueError("Each input must have exactly one sampling seed")
        if any(isinstance(seed, bool) or not isinstance(seed, int)
               or not 0 <= seed < 2**64 for seed in seeds):
            raise ValueError("Sampling seeds must be unsigned 64-bit integers")
        if self.config.generation_batch_size < 1:
            raise ValueError("Generation batch size must be positive")
        if agent == "J":
            if kind != "judge":
                raise ValueError("J only supports kind='judge'")
            self.assert_judge_frozen()
            model, tokenizer = self.judge, self.judge_tokenizer
        else:
            if kind not in ("sender", "bit"):
                raise ValueError("The vanilla answerer supports sender or bit outputs")
            self._activate(agent, train=False)
            model, tokenizer = self.model, self.tokenizer
        trajectories: list[Trajectory] = []
        with self.torch.inference_mode():
            for start in range(0, len(messages_batch), self.config.generation_batch_size):
                stop = start + self.config.generation_batch_size
                # Equal seed values still require separate generator objects.
                generators = [self.torch.Generator(device=self.device).manual_seed(seed)
                              for seed in seeds[start:stop]]
                trajectories.extend(self._generate_batch(
                    agent, model, tokenizer, messages_batch[start:stop], kind, generators))
        self.assert_judge_frozen()
        return trajectories

    def _sample_stage(
        self, model: Any, tokenizer: Any, sequences: list[list[int]],
        mode: str, max_tokens: int, stop_ids: list[int], *,
        generator: Any = None, forbidden: list[int] | None = None,
        label_ids: list[list[int]] | None = None,
    ) -> list[tuple[list[int], list[float], list[list[int] | None]]]:
        # Preserve the inherited generate() behavior for callers that do not ask
        # for matched per-example streams.
        if not isinstance(generator, list):
            return super()._sample_stage(
                model, tokenizer, sequences, mode, max_tokens, stop_ids,
                generator=generator, forbidden=forbidden, label_ids=label_ids)
        if len(generator) != len(sequences):
            raise ValueError("Each sampling row must have its own generator")
        if len({id(item) for item in generator}) != len(generator):
            raise ValueError("Sampling rows cannot share generator objects")

        # This is Engine._sample_stage with only the multinomial draw changed.
        # Cache updates, support restrictions, probabilities and stopping rules
        # remain the same as in the training and vanilla-baseline experiments.
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
            sampled = torch.full((size,), pad, dtype=torch.long, device=self.device)
            for i, was_active in enumerate(active):
                if was_active:
                    sampled[i] = torch.multinomial(
                        probabilities[i], 1, generator=generator[i])[0]
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
            step_mask = torch.tensor(active, device=self.device, dtype=attention_mask.dtype)
            input_ids = sampled[:, None]
            input_ids = torch.where(step_mask[:, None].bool(), input_ids,
                                    torch.full_like(input_ids, pad))
            attention_mask = torch.cat((attention_mask, step_mask[:, None]), dim=1)
            active = next_active
        return list(zip(tokens, logps, supports))
