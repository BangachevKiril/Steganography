"""CPU regression checks for the corrected experiment, without model assets."""
import copy
import json
import math
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from stego import university_evaluation as evaluation
from stego import university_training as training


class TokenBatch(dict):
    """A CPU stand-in for BatchEncoding at the tested CUDA boundary."""
    def to(self, _device):
        return self


class Classifier(torch.nn.Module):
    def __init__(self, features=4, dropout=0.0):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.classifier = torch.nn.Linear(features, 2)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.classifier(self.dropout(input_ids)))


class RecordingAdamW(torch.optim.AdamW):
    def __init__(self, parameters):
        super().__init__(parameters, lr=2e-5, weight_decay=0.01)
        self.gradients_at_steps = []

    def step(self, closure=None):
        self.gradients_at_steps.append([
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for group in self.param_groups for parameter in group['params']])
        return super().step(closure)


class UniversityCorrectionsTests(unittest.TestCase):
    def assert_tensors_close(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=1e-11, atol=1e-13)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tensors_close(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for lhs, rhs in zip(left, right):
                self.assert_tensors_close(lhs, rhs)
        else:
            self.assertEqual(left, right)

    def assert_rng_equal(self, before, after):
        self.assertEqual(before['python'], after['python'])
        self.assertEqual(before['numpy'][0], after['numpy'][0])
        np.testing.assert_array_equal(before['numpy'][1], after['numpy'][1])
        self.assertEqual(before['numpy'][2:], after['numpy'][2:])
        self.assertTrue(torch.equal(before['torch'], after['torch']))
        self.assertEqual(len(before['cuda']), len(after['cuda']))
        for lhs, rhs in zip(before['cuda'], after['cuda']):
            self.assertTrue(torch.equal(lhs, rhs))

    def test_accumulation_matches_full_mean_loss_gradients_parameters_and_adamw(self):
        # The second iteration detects stale/discarded gradients and compares
        # nonempty AdamW moments, rather than only a first sign-normalized step.
        for n in (256, 130):
            with self.subTest(n=n):
                torch.manual_seed(821)
                full = Classifier().double()
                accumulated = copy.deepcopy(full)
                full_optimizer = RecordingAdamW(full.parameters())
                accumulated_optimizer = RecordingAdamW(accumulated.parameters())
                for iteration in range(2):
                    features = torch.randn(n, 4, dtype=torch.float64)
                    targets = torch.randint(0, 2, (n,))
                    tokens = {'input_ids': features, 'attention_mask': torch.ones_like(features)}
                    full.train()
                    full_optimizer.zero_grad()
                    logits = full(**tokens).logits
                    loss = torch.nn.functional.cross_entropy(logits, targets)
                    loss.backward()
                    expected_correct = int((logits.argmax(-1) == targets).sum())
                    full_optimizer.step()
                    result = training.receiver_optimization_step(
                        accumulated, accumulated_optimizer, tokens, targets,
                        microbatch_size=64)
                    self.assertEqual(len(accumulated_optimizer.gradients_at_steps), iteration + 1)
                    self.assert_tensors_close(full_optimizer.gradients_at_steps[-1],
                                              accumulated_optimizer.gradients_at_steps[-1])
                    self.assert_tensors_close(full.state_dict(), accumulated.state_dict())
                    self.assert_tensors_close(full_optimizer.state_dict(),
                                              accumulated_optimizer.state_dict())
                    self.assertAlmostEqual(result['loss'], float(loss.detach()), places=12)
                    self.assertEqual(result['n'], n)
                    self.assertEqual(result['correct'], expected_correct)
                    self.assertEqual(result['optimizer_steps'], 1)
                    self.assertEqual(result['optimizer_sample_count'], n)

    def test_each_of_four_microbatches_contributes_to_the_single_step(self):
        # Each block controls its own input feature: discarding any microbatch
        # leaves an unmistakably zero gradient in the corresponding column.
        model = Classifier().double()
        with torch.no_grad():
            model.classifier.weight.zero_()
            model.classifier.bias.zero_()
        tokens = {'input_ids': torch.eye(4, dtype=torch.float64).repeat_interleave(64, 0),
                  'attention_mask': torch.ones(256, 4, dtype=torch.float64)}
        targets = torch.tensor(([0] * 64 + [1] * 64) * 2)
        optimizer = RecordingAdamW(model.parameters())
        training.receiver_optimization_step(model, optimizer, tokens, targets)
        self.assertEqual(len(optimizer.gradients_at_steps), 1)
        gradient = optimizer.gradients_at_steps[0][0]
        expected = torch.tensor([[-.125, .125, -.125, .125],
                                 [.125, -.125, .125, -.125]], dtype=torch.float64)
        torch.testing.assert_close(gradient, expected, rtol=0, atol=0)

    def test_observer_uses_each_rows_last_real_token_and_rejects_empty_rows(self):
        logits = torch.arange(3 * 5 * 4, dtype=torch.float64).reshape(3, 5, 4)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 1, 0]])
        expected = torch.stack([logits[0, 1], logits[1, 4], logits[2, 3]])
        torch.testing.assert_close(training.last_nonpadding_logits(logits, mask), expected)
        mask[1].zero_()
        with self.assertRaises(ValueError):
            training.last_nonpadding_logits(logits, mask)

    def test_observer_score_is_independent_of_padding_positions(self):
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
        logits = torch.zeros(2, 4, 10000)
        logits[0, 1, 2822], logits[0, 1, 9642] = 3, 0
        logits[0, 3, 2822], logits[0, 3, 9642] = 0, 10  # misleading pad logits
        logits[1, 3, 2822], logits[1, 3, 9642] = 0, 2
        tokenizer = Mock()
        tokenizer.apply_chat_template.side_effect = lambda *args, **kwargs: 'prompt'
        tokenizer.return_value = TokenBatch(input_ids=torch.ones_like(mask), attention_mask=mask)
        observer = training.Observer.__new__(training.Observer)
        observer.tokenizer = tokenizer
        observer.model = Mock(return_value=SimpleNamespace(logits=logits))
        rows = observer.score(['short', 'a longer input'])
        self.assertEqual([row['observer_flag'] for row in rows], [False, True])
        self.assertAlmostEqual(rows[0]['observer_score'], math.tanh(3 - .75), places=6)
        self.assertAlmostEqual(rows[1]['observer_score'], math.tanh(-2 - .75), places=6)

    def test_training_hands_trl_unpadded_queries_of_different_lengths(self):
        class Query:
            def __init__(self, ids):
                self.ids = ids

            def to(self, _device):
                return self.ids.clone()

        class Tokenizer:
            pad_token_id = 0

            def __init__(self):
                self.calls = []

            def __call__(self, text, **kwargs):
                self.calls.append(kwargs)
                ids = torch.arange(1, 4 if len(self.calls) == 1 else 8)
                if kwargs.get('padding') == 'max_length':
                    ids = torch.cat([torch.zeros(kwargs['max_length'] - len(ids), dtype=torch.long), ids])
                return {'input_ids': [Query(ids)]}

            def decode(self, response, **kwargs):
                return 'Public summary.'

        trainer = training.Trainer.__new__(training.Trainer)
        trainer.streams = {'sender': SimpleNamespace(next=lambda: [0, 1])}
        trainer.dataset = [{'input': 'Short CV.', 'label': 'FAIL'},
                           {'input': 'Longer applicant CV.', 'label': 'PASS'}]
        trainer.tokenizer = Tokenizer()
        trainer.query_max_length = 65
        trainer.ppo = SimpleNamespace(generate=Mock(return_value=[torch.tensor([8]), torch.tensor([9])]))
        result = trainer.generate('sender')
        queries = trainer.ppo.generate.call_args.args[0]
        self.assertEqual([len(query) for query in queries], [3, 7])
        self.assertTrue(all(not bool((query == 0).any()) for query in queries))
        self.assertTrue(all(call['padding'] is False for call in trainer.tokenizer.calls))
        self.assertTrue(all(call['max_length'] == 65 for call in trainer.tokenizer.calls))
        self.assertEqual(result[4], [0, 1])

    def test_sender_reward_receiver_eval_restores_mixed_modes_and_all_rng(self):
        trainer = training.Trainer.__new__(training.Trainer)
        trainer.receiver = Classifier(dropout=.8)
        trainer.receiver.train()
        trainer.receiver.classifier.eval()  # deliberately mixed caller modes
        original_modes = [module.training for module in trainer.receiver.modules()]
        trainer.log_status = Mock()
        trainer.generate = Mock(return_value=(
            [0, 1], [torch.tensor([1]), torch.tensor([2])],
            [torch.tensor([3]), torch.tensor([4])], ['summary A', 'summary B'], [0, 1], 18))
        trainer.observer = SimpleNamespace(score=lambda texts: [
            {'observer_score': .5, 'observer_flag': False} for _ in texts])
        seen_modes = []

        def receiver_logits(texts):
            seen_modes.append([module.training for module in trainer.receiver.modules()])
            # Deliberate RNG use verifies isolation even if a future scoring
            # implementation consumes a generator despite disabling dropout.
            random.random()
            np.random.rand()
            torch.rand(2)
            with torch.no_grad():
                return trainer.receiver(torch.ones(len(texts), 4)).logits

        trainer.receiver_logits = receiver_logits
        trainer.ppo = SimpleNamespace(step=Mock(return_value={'ppo/loss/total': 0.0}))
        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            kwargs.pop('device', None)
            return original_tensor(*args, **kwargs)

        before = training.capture_rng()
        with patch('torch.tensor', side_effect=cpu_tensor):
            trainer.sender_update()
        self.assertEqual(seen_modes, [[False] * len(original_modes)])
        self.assertEqual([module.training for module in trainer.receiver.modules()], original_modes)
        self.assert_rng_equal(before, training.capture_rng())
        first_rewards = copy.deepcopy(trainer.ppo.step.call_args.args[2])
        with patch('torch.tensor', side_effect=cpu_tensor):
            trainer.sender_update()
        self.assert_tensors_close(first_rewards, trainer.ppo.step.call_args.args[2])

        def failed_receiver_logits(texts):
            receiver_logits(texts)
            raise RuntimeError('scoring failed')

        trainer.receiver_logits = failed_receiver_logits
        with self.assertRaisesRegex(RuntimeError, 'scoring failed'):
            trainer.sender_update()
        self.assertEqual([module.training for module in trainer.receiver.modules()], original_modes)
        self.assert_rng_equal(before, training.capture_rng())

    def test_fresh_training_evaluates_committed_zero_before_any_receiver_update(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = training.Trainer.__new__(training.Trainer)
            trainer.run = Path(temporary)
            trainer.update = 0
            trainer.config_digest = 'corrected-config'
            trainer.config = {'evaluation_interval': 50, 'max_active_seconds': 27900,
                              'evaluate_initial_checkpoint': True}
            trainer.current_checkpoint = trainer.run / 'checkpoints/step_000000'
            trainer.current_checkpoint.mkdir(parents=True)
            (trainer.current_checkpoint / 'COMPLETE.json').write_text(json.dumps({
                'completed_update': 0, 'config_digest': trainer.config_digest}))
            trainer.active_seconds = lambda: 0
            events = []

            def evaluate():
                checkpoint = evaluation._checkpoint(trainer)
                self.assertEqual(checkpoint['completed_update'], 0)
                events.append('evaluation_zero')

            def first_update():
                events.append('first_update')
                raise RuntimeError('stop after testing order')

            trainer.evaluate = evaluate
            trainer.receiver_update = first_update
            with patch.object(training, 'STOP', False):
                with self.assertRaisesRegex(RuntimeError, 'stop after testing order'):
                    trainer.train()
            self.assertEqual(events, ['evaluation_zero', 'first_update'])

    def test_zero_evaluation_commits_its_own_baseline_and_rejects_tampering(self):
        # Exercise the public evaluation pipeline with all required cohort rows;
        # only heavyweight generation, model assets and judge process are mocked.
        with tempfile.TemporaryDirectory() as temporary:
            trainer = training.Trainer.__new__(training.Trainer)
            trainer.run = Path(temporary)
            trainer.update = 0
            trainer.config_digest = 'corrected-config'
            trainer.config = {'evaluation_seed': 20260930, 'judge_python': 'fake-judge',
                              'baseline_d': .08}
            trainer.current_checkpoint = trainer.run / 'checkpoints/step_000000'
            trainer.current_checkpoint.mkdir(parents=True)
            (trainer.current_checkpoint / 'COMPLETE.json').write_text(json.dumps({
                'completed_update': 0, 'config_digest': trainer.config_digest}))
            trainer.log_status = Mock()
            trainer.should_stop = lambda: False
            trainer.sender = torch.nn.Linear(2, 2)
            trainer.sender.pretrained_model = object()
            trainer.receiver = Classifier()
            trainer.receiver_tokenizer = lambda texts, **kwargs: TokenBatch(
                input_ids=torch.zeros(len(texts), 4), attention_mask=torch.ones(len(texts), 4))
            trainer.tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: 'PIQA prompt')
            trainer.observer = SimpleNamespace(score=lambda texts: [
                {'observer_score': .5, 'observer_flag': False} for text in texts])
            university = [{'example_id': i, 'input': f'Applicant {i}.', 'true_label': i % 2}
                          for i in range(2000)]
            piqa = [{'question_id': i, 'question': f'Question {i}?'} for i in range(100)]

            def generate(model, tokenizer, prompts, **kwargs):
                return [{'answer': 'A public answer. It has two sentences.'} for _ in prompts]

            def judge(trainer, config, run, evaluation_digest):
                payload = {'complete': True, 'n': 100, 'target_questions': 100,
                           'config_digest': evaluation_digest,
                           'source_generations_sha256': evaluation.sha256(run / 'PIQA_GENERATIONS.json'),
                           'rows': [{**row, 'answer': 'A public answer.',
                                     'judge_pass': i >= 7, 'format_ok': True}
                                    for i, row in enumerate(piqa)]}
                (run / 'PIQA_JUDGED.json').write_text(json.dumps(payload))

            before = copy.deepcopy(trainer.receiver.state_dict())
            with patch.object(evaluation, '_selection', return_value=(university, piqa, {'questions': piqa})), \
                    patch.object(evaluation, 'generate_hf', side_effect=generate) as generate_mock, \
                    patch.object(evaluation, '_run_judge', side_effect=judge):
                summary = evaluation.evaluate_training(trainer)
                generation_calls = generate_mock.call_count
                self.assertGreater(generation_calls, 0)
                self.assertEqual(evaluation.evaluate_training(trainer), summary)
                self.assertEqual(generate_mock.call_count, generation_calls)
            self.assertEqual(summary['completed_update'], 0)
            self.assertEqual(summary['parameter_updates'], 0)
            self.assertEqual(summary['university']['n'], 2000)
            self.assertEqual(summary['piqa']['n'], 100)
            self.assertEqual(summary['baseline_d'], .07)
            self.assertEqual(summary['delta_d_percentage_points'], 0)
            self.assertEqual(evaluation._baseline_d(trainer.config, trainer), .07)
            self.assert_tensors_close(before, trainer.receiver.state_dict())
            summary_path = trainer.run / 'evaluations/step_000000/SUMMARY.json'
            changed = json.loads(summary_path.read_text())
            changed['piqa']['d'] = .5
            summary_path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                evaluation._baseline_d(trainer.config, trainer)


if __name__ == '__main__':
    unittest.main()
