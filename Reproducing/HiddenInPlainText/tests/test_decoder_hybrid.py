"""CPU gradient checks for fixed-baseline reasoning PG plus final-bit supervision.

The toy model gives each preceding token its own logits. This separates thought,
forced-boundary, and final-label gradients without a GPU or pretrained model.
"""
from copy import deepcopy
from contextlib import contextmanager
from types import SimpleNamespace
import unittest

import torch

from stego.engine import Engine, EngineConfig, Trajectory


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def encode(self, text, add_special_tokens=False):
        return {"</think>": [3], "\n\n": [4], "0": [5], "1": [6]}[text]


class LogitTable(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.table = torch.nn.Parameter(torch.zeros(10, 10))
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.forward_inputs = []

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        self.forward_inputs.append(input_ids.detach().clone())
        logits = self.table[input_ids]
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        elif logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


class CausalLogitTable(LogitTable):
    """A small position- and prefix-sensitive model for padding/batching checks."""
    def __init__(self):
        super().__init__()
        with torch.no_grad():
            self.table.copy_(torch.sin(torch.arange(100).reshape(10, 10).float()) / 4)
            # Unmasked padding would materially alter subsequent predictions.
            self.table[0] = torch.arange(10).float()
        self.register_buffer("reference_table", torch.cos(torch.arange(100).reshape(10, 10).float()) / 5)
        self.adapter_enabled = True
        self.forward_details = []

    @contextmanager
    def disable_adapter(self):
        enabled = self.adapter_enabled
        self.adapter_enabled = False
        try:
            yield
        finally:
            self.adapter_enabled = enabled

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                logits_to_keep=0, **kwargs):
        self.forward_inputs.append(input_ids.detach().clone())
        attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        self.forward_details.append({
            "ids": input_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            "adapter_enabled": self.adapter_enabled,
        })
        table = self.table if self.adapter_enabled else self.reference_table
        embedded = table[input_ids]
        context = (embedded * attention_mask[..., None]).cumsum(dim=1)
        positions = position_ids[..., None].float()
        vocabulary_pattern = torch.linspace(-0.1, 0.1, 10)
        logits = embedded + 0.15 * context + positions * vocabulary_pattern
        if isinstance(logits_to_keep, torch.Tensor):
            logits = logits[:, logits_to_keep]
        elif logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return SimpleNamespace(logits=logits)


class RecordingSGD(torch.optim.SGD):
    def __init__(self, parameter):
        super().__init__([parameter], lr=0.1)
        self.parameter = parameter
        self.steps = 0
        self.gradient = None

    def step(self, closure=None):
        self.steps += 1
        self.gradient = self.parameter.grad.detach().clone()
        return super().step(closure)


def make_engine(**overrides):
    config = dict(device="cpu", think_tokens=256, final_answer_tokens=128,
                  temperature=0.6, kl_beta=0.0, max_grad_norm=1e6,
                  decoder_objective="fixed_baseline_pg_ce",
                  decoder_baseline=0.5, decoder_ce_weight=1.0)
    config.update(overrides)
    engine = object.__new__(Engine)
    engine.torch = torch
    engine.config = EngineConfig(**config)
    engine.device = torch.device("cpu")
    engine.tokenizer = Tokenizer()
    engine.model = LogitTable()
    engine._supports_logits_to_keep = True
    engine.adapter_parameters = {"A": [engine.model.table]}
    engine.optimizers = {"A": RecordingSGD(engine.model.table)}
    engine._activate = lambda agent, train: engine.model.train(train)
    engine.assert_judge_frozen = lambda: None
    return engine


def trajectory(engine, bit=0, *, kind="bit", thought_token=3, forced=False):
    thought = [thought_token] if engine.config.think_tokens else []
    boundary = ([3] if forced else []) + ([4] if thought else [])
    sampled_bit = 5 + bit
    modes = ["thought"] * len(thought) + ["forced"] * len(boundary)
    modes.append("constrained" if kind == "bit" else "text")
    mask = [True] * len(thought) + [False] * len(boundary) + [True]
    item = Trajectory(
        agent="A", kind=kind,
        prompt_ids=[1, 2] if thought else [1, 4],
        completion_ids=thought + boundary + [sampled_bit],
        action_mask=mask,
        supports=[None] * (len(thought) + len(boundary)) +
                 ([[5, 6]] if kind == "bit" else [None]),
        modes=modes, old_logps=[0.0] * len(mask), final_text=str(bit),
        thought_text="not public", thought_token_count=len(thought),
        final_token_count=1, forced_think_close=forced,
    )
    with torch.no_grad():
        behavior = iter(engine._trajectory_logps(item).tolist())
        item.old_logps = [next(behavior) if active else 0.0 for active in mask]
    return item


def update(engine, groups, rewards, *, targets, weights=None):
    metrics = engine.update("A", groups, rewards, group_weights=weights,
                            target_bits=targets)
    return metrics, engine.optimizers["A"].gradient


def batched_engine(batch_size, *, supports_logits_to_keep=True, kl_beta=0.0):
    engine = make_engine(train_microbatch_size=batch_size, think_tokens=8, kl_beta=kl_beta)
    engine.model = CausalLogitTable()
    engine.adapter_parameters = {"A": [engine.model.table]}
    engine.optimizers = {"A": RecordingSGD(engine.model.table)}
    engine._supports_logits_to_keep = supports_logits_to_keep
    return engine


def ragged_trajectory(engine, prompt, thoughts, bit, *, kind="bit"):
    item = trajectory(engine, bit, kind=kind)
    forced = bool(thoughts and thoughts[-1] != 3)
    boundary = ([3] if forced else []) + ([4] if thoughts else [])
    item.prompt_ids = list(prompt)
    item.completion_ids = list(thoughts) + boundary + [5 + bit]
    item.action_mask = [True] * len(thoughts) + [False] * len(boundary) + [True]
    item.supports = [None] * (len(thoughts) + len(boundary)) + ([[5, 6]] if kind == "bit" else [None])
    item.modes = ["thought"] * len(thoughts) + ["forced"] * len(boundary) + ["constrained" if kind == "bit" else "text"]
    item.old_logps = [0.0] * len(item.action_mask)
    item.thought_token_count = len(thoughts)
    item.forced_think_close = forced
    with torch.no_grad():
        behavior = iter(engine._trajectory_logps(item).tolist())
        item.old_logps = [next(behavior) if active else 0.0 for active in item.action_mask]
    return item


def mixed_ragged_groups(engine):
    groups = [
        [ragged_trajectory(engine, [1, 2], thoughts, bit, kind="sender")
         for thoughts, bit in [([3], 0), ([7, 3], 1), ([8, 7], 0)]],
        [ragged_trajectory(engine, [1, 7, 8, 2], thoughts, bit, kind="sender")
         for thoughts, bit in [([3], 1), ([7, 8, 3], 0)]],
    ]
    labels = [None, None, 1, 1, 0, 0, 1]
    decoder_specs = [
        ([2], [7, 3], 0),
        ([1, 8, 2], [3], 1),
        ([1, 1, 7, 2], [8, 7, 3], 0),
        ([8, 2], [7, 8], 1),
        ([1, 2], [8, 3], 0),
    ]
    groups.extend([[ragged_trajectory(engine, prompt, thoughts, bit)]
                   for prompt, thoughts, bit in decoder_specs])
    rewards = [[-1.0, 0.0, 1.0], [1.0, -1.0]] + [
        [float(bit == target)] for (_, _, bit), target in zip(decoder_specs, labels[2:])]
    return groups, rewards, labels, [0.25, 0.25] + [0.1] * 5


class HybridDecoderGradientTests(unittest.TestCase):
    def test_all_wrong_group_still_updates_thoughts_and_correct_label(self):
        engine = make_engine()
        item = trajectory(engine, bit=0)
        metrics, gradient = update(engine, [[item, deepcopy(item)]], [[0.0, 0.0]], targets=[1])
        # Gradient descent discourages this sampled thought and the wrong label.
        self.assertGreater(gradient[2, 3].item(), 0)
        self.assertGreater(gradient[4, 5].item(), 0)
        self.assertLess(gradient[4, 6].item(), 0)
        self.assertLess(engine.model.table[2, 3].item(), 0)
        self.assertGreater(engine.model.table[4, 6].item(), 0)
        self.assertEqual(metrics["counterfactual_grpo_active_group_fraction"], 0)
        self.assertEqual(metrics["reasoning_pg_active_fraction"], 1)
        self.assertEqual(engine.optimizers["A"].steps, 1)

    def test_all_correct_group_reinforces_sampled_thought(self):
        engine = make_engine(decoder_ce_weight=0.0)
        item = trajectory(engine, bit=1)
        _, gradient = update(engine, [[item, deepcopy(item)]], [[1.0, 1.0]], targets=[1])
        self.assertLess(gradient[2, 3].item(), 0)
        self.assertGreater(engine.model.table[2, 3].item(), 0)

    def test_fixed_baseline_does_not_depend_on_group_companions(self):
        def run(bits, thought_tokens):
            engine = make_engine(think_tokens=1, decoder_ce_weight=0.0)
            samples = [trajectory(engine, bit=bit, thought_token=token, forced=True)
                       for bit, token in zip(bits, thought_tokens)]
            return update(engine, [samples], [[float(bit == 1) for bit in bits]],
                          targets=[1])[1]

        wrong_alone = run([0], [7])
        right_alone = run([1], [8])
        together = run([0, 1], [7, 8])
        self.assertGreater(wrong_alone.abs().sum().item(), 0)
        self.assertGreater(right_alone.abs().sum().item(), 0)
        torch.testing.assert_close(together, (wrong_alone + right_alone) / 2)

    def test_final_bit_and_injected_boundaries_receive_no_reasoning_pg(self):
        engine = make_engine(think_tokens=1, decoder_ce_weight=0.0)
        item = trajectory(engine, bit=0, thought_token=7, forced=True)
        _, gradient = update(engine, [[item]], [[0.0]], targets=[1])
        self.assertGreater(gradient[2].abs().sum().item(), 0)
        # Rows 7 and 3 predict injected close/newline, and row 4 predicts the bit.
        for row in (7, 3, 4):
            torch.testing.assert_close(gradient[row], torch.zeros(10))

    def test_supervised_target_is_not_inserted_into_the_model_input(self):
        engine = make_engine()
        with torch.no_grad():
            engine.model.table[4, 5] = 0.7
            engine.model.table[4, 6] = -0.2
        item = trajectory(engine, bit=0)
        original = deepcopy(item)
        _, logp_zero = engine._trajectory_scores(item, target_bit=0)
        zero_input = engine.model.forward_inputs[-1]
        _, logp_one = engine._trajectory_scores(item, target_bit=1)
        one_input = engine.model.forward_inputs[-1]
        torch.testing.assert_close(zero_input, one_input)
        self.assertEqual(item, original)
        self.assertEqual(one_input[0, -1].item(), 5)  # still the sampled wrong bit
        expected = torch.log_softmax(engine.model.table[4, [5, 6]] / 0.6, dim=0)
        torch.testing.assert_close(logp_zero, expected[0])
        torch.testing.assert_close(logp_one, expected[1])

    def test_final_bit_ce_is_not_divided_by_the_thought_budget(self):
        gradients = []
        for budget in (1, 256):
            engine = make_engine(think_tokens=budget)
            item = trajectory(engine, bit=0)
            _, gradient = update(engine, [[item]], [[0.0]], targets=[1])
            gradients.append(gradient[4].clone())
        torch.testing.assert_close(gradients[0], gradients[1])
        self.assertAlmostEqual(gradients[1][5].item(), 0.5 / 0.6, places=6)
        self.assertAlmostEqual(gradients[1][6].item(), -0.5 / 0.6, places=6)

    def test_zero_thought_budget_still_allows_supervised_bit_training(self):
        engine = make_engine(think_tokens=0)
        item = trajectory(engine, bit=0)
        _, gradient = update(engine, [[item]], [[0.0]], targets=[1])
        self.assertLess(gradient[4, 6].item(), 0)
        torch.testing.assert_close(gradient[2], torch.zeros(10))

    def test_confident_wrong_decoder_has_finite_corrective_ce_gradient(self):
        engine = make_engine()
        with torch.no_grad():
            engine.model.table[4, 5] = 100.0
            engine.model.table[4, 6] = -100.0
        item = trajectory(engine, bit=0)
        metrics, gradient = update(engine, [[item]], [[0.0]], targets=[1])
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertAlmostEqual(metrics["decoder_ce"], 200.0 / 0.6, places=4)
        self.assertLess(gradient[4, 6].item(), -1.0)

    def test_sender_gradient_is_unchanged_from_legacy_grpo(self):
        gradients = []
        for objective in ("grpo", "fixed_baseline_pg_ce"):
            engine = make_engine(decoder_objective=objective)
            samples = [trajectory(engine, bit=bit, kind="sender") for bit in (0, 1)]
            gradients.append(update(engine, [samples], [[-1.0, 1.0]], targets=[None])[1])
        torch.testing.assert_close(gradients[0], gradients[1])
        self.assertGreater(gradients[0].abs().sum().item(), 0)

    def test_mixed_roles_keep_equal_weight_and_one_optimizer_step(self):
        def run(roles, weights):
            engine = make_engine()
            groups, rewards, labels = [], [], []
            for role in roles:
                if role == "sender":
                    groups.append([trajectory(engine, bit=b, kind="sender") for b in (0, 1)])
                    rewards.append([-1.0, 1.0])
                    labels.append(None)
                else:
                    groups.append([trajectory(engine, bit=0)] * 2)
                    rewards.append([0.0, 0.0])
                    labels.append(1)
            _, gradient = update(engine, groups, rewards, targets=labels, weights=weights)
            self.assertEqual(engine.optimizers["A"].steps, 1)
            return gradient

        sender = run(["sender"], [1.0])
        receiver = run(["receiver"], [1.0])
        combined = run(["sender", "sender", "receiver"], [0.25, 0.25, 0.5])
        torch.testing.assert_close(combined, 0.5 * sender + 0.5 * receiver)

    def test_decoder_labels_are_required_and_validated_before_an_optimizer_step(self):
        for targets in (None, [], [None], [2], [-1], [True], [1.0]):
            with self.subTest(targets=targets):
                engine = make_engine()
                item = trajectory(engine, bit=0)
                with self.assertRaises(ValueError):
                    update(engine, [[item]], [[0.0]], targets=targets)
                self.assertEqual(engine.optimizers["A"].steps, 0)

    def test_decoder_reward_must_agree_with_the_supervised_target(self):
        engine = make_engine()
        item = trajectory(engine, bit=0)
        with self.assertRaises(ValueError):
            update(engine, [[item]], [[1.0]], targets=[1])
        self.assertEqual(engine.optimizers["A"].steps, 0)

    def test_sender_cannot_receive_a_supervised_bit_target(self):
        engine = make_engine()
        item = trajectory(engine, bit=0, kind="sender")
        with self.assertRaises(ValueError):
            update(engine, [[item]], [[0.0]], targets=[0])
        self.assertEqual(engine.optimizers["A"].steps, 0)


class HybridMinibatchTests(unittest.TestCase):
    def test_ragged_batch_scores_match_individual_causal_scores(self):
        for optimized_projection in (False, True):
            with self.subTest(optimized_projection=optimized_projection):
                engine = batched_engine(4, supports_logits_to_keep=optimized_projection)
                samples = [
                    ragged_trajectory(engine, [1, 2], [3], 0),
                    ragged_trajectory(engine, [1, 7, 8, 2], [8, 7, 3], 1),
                    ragged_trajectory(engine, [2], [7, 8], 0),
                    ragged_trajectory(engine, [1, 8, 2], [3], 1, kind="sender"),
                ]
                labels = [1, 1, 0, None]
                with torch.no_grad():
                    expected = [engine._trajectory_scores(item, target_bit=label)
                                for item, label in zip(samples, labels)]
                    engine.model.forward_details.clear()
                    actual = engine._trajectory_scores_batch(samples, target_bits=labels)
                self.assertEqual(len(engine.model.forward_details), 1)
                forward = engine.model.forward_details[0]
                self.assertEqual(forward["ids"].shape[0], 4)
                self.assertGreater(int((forward["attention_mask"] == 0).sum()), 0)
                for row, sample, (logps, target), (expected_logps, expected_target) in zip(
                        range(4), samples, actual, expected):
                    mask = forward["attention_mask"][row].bool()
                    torch.testing.assert_close(forward["ids"][row][mask],
                                               torch.tensor(sample.prompt_ids + sample.completion_ids))
                    torch.testing.assert_close(forward["position_ids"][row][mask],
                                               torch.arange(int(mask.sum())))
                    self.assertEqual(logps.numel(), sum(sample.action_mask))
                    torch.testing.assert_close(logps, expected_logps, atol=1e-6, rtol=1e-6)
                    if expected_target is None:
                        self.assertIsNone(target)
                    else:
                        torch.testing.assert_close(target, expected_target, atol=1e-6, rtol=1e-6)

    def test_batch_four_matches_singleton_gradients_loss_and_parameter_update(self):
        for kl_beta in (0.0, 0.15):
            with self.subTest(kl_beta=kl_beta):
                outcomes = []
                for batch_size in (1, 4):
                    engine = batched_engine(batch_size, kl_beta=kl_beta)
                    groups, rewards, labels, weights = mixed_ragged_groups(engine)
                    # Move slightly away from the saved behavior policy to exercise
                    # probability ratios as well as CE and reference-model KL.
                    with torch.no_grad():
                        engine.model.table[2, 3] += 0.35
                        engine.model.table[4, 5] -= 0.15
                    engine.model.forward_details.clear()
                    backward_calls = []
                    hook = engine.model.table.register_hook(lambda gradient: backward_calls.append(gradient.detach().clone()))
                    metrics, gradient = update(engine, groups, rewards, targets=labels, weights=weights)
                    hook.remove()
                    outcomes.append((metrics, gradient, engine.model.table.detach().clone()))
                    self.assertEqual(engine.optimizers["A"].steps, 1)
                    current_sizes = [entry["ids"].shape[0] for entry in engine.model.forward_details
                                     if entry["adapter_enabled"]]
                    self.assertEqual(current_sizes, [1] * 10 if batch_size == 1 else [4, 4, 2])
                    self.assertEqual(len(backward_calls), len(current_sizes))
                    reference_sizes = [entry["ids"].shape[0] for entry in engine.model.forward_details
                                       if not entry["adapter_enabled"]]
                    self.assertEqual(reference_sizes, current_sizes if kl_beta else [])
                    self.assertEqual(metrics["trajectories"], 10)
                    if kl_beta:
                        self.assertGreater(metrics["kl"], 0)
                singleton_metrics, singleton_gradient, singleton_parameters = outcomes[0]
                batch_metrics, batch_gradient, batch_parameters = outcomes[1]
                torch.testing.assert_close(batch_gradient, singleton_gradient, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(batch_parameters, singleton_parameters, atol=2e-6, rtol=2e-6)
                for metric in ("loss", "policy_loss", "kl", "ratio_mean", "clip_fraction",
                               "reward_mean", "decoder_ce", "decoder_ce_loss",
                               "sender_policy_loss", "decoder_reasoning_policy_loss"):
                    self.assertAlmostEqual(batch_metrics[metric], singleton_metrics[metric], places=6,
                                           msg=f"{metric}: batch_size=4 differs from batch_size=1")

    def test_batched_reasoning_pg_excludes_final_forced_and_padded_positions(self):
        engine = make_engine(train_microbatch_size=4, think_tokens=1, decoder_ce_weight=0.0)
        samples = []
        for prompt in ([2], [1, 2], [1, 8, 2], [1, 1, 8, 2], [1, 7, 8, 2]):
            sample = trajectory(engine, bit=0, thought_token=7, forced=True)
            sample.prompt_ids = list(prompt)
            samples.append([sample])
        _, gradient = update(engine, samples, [[0.0]] * 5, targets=[1] * 5)
        self.assertGreater(gradient[2].abs().sum().item(), 0)
        for row in (0, 3, 4, 7):
            torch.testing.assert_close(gradient[row], torch.zeros(10))
        sizes = [ids.shape[0] for ids in engine.model.forward_inputs if ids.shape[0] > 1]
        self.assertEqual(sizes, [4])
        self.assertEqual(engine.optimizers["A"].steps, 1)


if __name__ == "__main__":
    unittest.main()
