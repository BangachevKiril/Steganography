"""Public reporting retains real denominators and excludes uncommitted files."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from stego.training_report import aggregate, render_report


class TrainingReportTests(unittest.TestCase):
    @staticmethod
    def row(step, n, receiver, interceptor, rejected, phase='joint'):
        return {'global_step': step, 'phase': phase, 'n': n, 'seconds': 100,
                'correct_receiver': receiver, 'correct_interceptor': interceptor,
                'rejected_answers': rejected, 'C_R': receiver / n,
                'C_I': interceptor / n, 'd': rejected / n}

    def test_weighted_counts_and_separate_interceptor_denominators(self):
        totals = aggregate([self.row(1, 128, 64, 32, 16), self.row(2, 16, 16, 16, 0)])
        self.assertEqual(totals['C_R'], (80, 144))
        self.assertEqual(totals['C_I'], (48, 144))
        self.assertEqual(totals['d'], (16, 144))
        row = self.row(11, 16, 10, 8, 2, 'interceptor')
        row['interceptor_training'] = {'n': 128, 'correct_interceptor': 68, 'C_I': 68 / 128}
        totals = aggregate([row])
        self.assertEqual(totals['C_I'], (8, 16))
        self.assertEqual(totals['interceptor_training_C_I'], (68, 128))
        legacy = aggregate([{'phase': 'interceptor', 'n': 128, 'C_I': 0.5}])
        self.assertIsNone(legacy['C_R'])
        self.assertIsNone(legacy['d'])
        self.assertIsNone(legacy['C_I'])
        self.assertEqual(legacy['interceptor_training_C_I'], (64, 128))

    def test_report_uses_audited_bound_not_stale_files_or_pointer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'metrics').mkdir()
            (root / 'validation').mkdir()
            for step in range(1, 12):
                row = self.row(step, 128, 64, 64, 32)
                if step == 11:
                    row = self.row(step, 16, 12, 10, 2, 'interceptor')
                    row['interceptor_training'] = {'n': 128, 'correct_interceptor': 100}
                (root / 'metrics' / f'step_{step:06d}.json').write_text(json.dumps(row))
            # Even an unreadable future file is ignored before parsing.
            (root / 'metrics' / 'step_000012.json').write_text('uncommitted and invalid')
            (root / 'validation' / 'step_000020.json').write_text('uncommitted and invalid')
            (root / 'LATEST.json').write_text(json.dumps({'global_step': 100}))
            (root / 'TIMING.json').write_text(json.dumps({'completed_step': 12, 'eta_seconds_excluding_queue': 1000}))
            result = render_report(root, 11)
            self.assertEqual(result['observed_steps'], 11)
            self.assertEqual(result['blocks'][0]['C_R'], (640, 1280))
            self.assertEqual(result['blocks'][1]['C_R'], (12, 16))
            report = (root / 'report.md').read_text()
            self.assertIn('11–11 (partial)', report)
            self.assertIn('100/128 (78.1%)', report)
            self.assertIn('does not match the audited checkpoint', report)
            with (root / 'training_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 11)
            self.assertEqual(rows[-1]['C_I_n'], '16')
            self.assertEqual(rows[-1]['interceptor_training_C_I_n'], '128')

    def test_mismatched_step_or_counts_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, 'Count/rate mismatch'):
            aggregate([{'n': 16, 'correct_receiver': 8, 'C_R': 0.75}])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'metrics').mkdir()
            (root / 'metrics' / 'step_000001.json').write_text(json.dumps({'global_step': 2}))
            with self.assertRaisesRegex(ValueError, 'Step identity mismatch'):
                render_report(root, 1)

    def test_decoder_optimization_counts_and_ce_remain_separate(self):
        from stego.training_report import decoder_rows
        guesses = {'n': 128, 'zero': 100, 'correct': 70,
                   'by_true_bit': {'0': {'n': 64, 'zero': 53, 'correct': 53},
                                   '1': {'n': 64, 'zero': 47, 'correct': 17}}}
        row = self.row(11, 16, 8, 8, 1, 'interceptor')
        row['interceptor_training'] = {'decoder_guesses': guesses}
        row['updates'] = {'I': {'decoder_examples': 128, 'decoder_ce': 0.7,
                               'decoder_probability_zero_mean': 0.8,
                               'reasoning_pg_active_fraction': 1}}
        result = decoder_rows([row])[0]
        self.assertEqual((result['n'], result['zero'], result['correct']), (128, 100, 70))
        self.assertEqual((result['bit0_correct'], result['bit1_correct']), (53, 17))
        self.assertEqual(result['ce'], 0.7)
        self.assertEqual(result['role'], 'interceptor')

    def test_single_rollout_report_preserves_16_guess_optimization_denominator(self):
        from stego.train import guess_diagnostics
        row = self.row(11, 16, 10, 8, 1, 'interceptor')
        row.update(decoder_rollouts_per_answer=1, decoder_answer_contexts=16)
        row['interceptor_training'] = {'n': 16, 'correct_interceptor': 8, 'C_I': .5,
                'decoder_guesses': guess_diagnostics([(i % 2, 0, 'A' if i < 8 else 'B') for i in range(16)])}
        row['updates'] = {'I': {'decoder_examples': 16, 'decoder_ce': .75,
                'decoder_probability_zero_mean': .8, 'reasoning_pg_active_fraction': 1}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'metrics').mkdir()
            (root / 'metrics' / 'step_000011.json').write_text(json.dumps(row))
            render_report(root, 11)
            with (root / 'training_metrics.csv').open() as stream:
                output = next(csv.DictReader(stream))
            self.assertEqual(output['C_I_n'], '16')
            self.assertEqual(output['interceptor_training_C_I_n'], '16')
            with (root / 'decoder_metrics.csv').open() as stream:
                decoder = next(csv.DictReader(stream))
            self.assertEqual(decoder['n'], '16')
            self.assertEqual(decoder['rollouts_per_answer'], '1')
            self.assertEqual(decoder['answer_contexts'], '16')
            self.assertIn('one guess per answer gives 16', (root / 'report.md').read_text())
        # Disabling monitoring must not relabel optimization as a matched probe.
        row.pop('C_R')
        row.pop('correct_receiver')
        totals = aggregate([row])
        self.assertIsNone(totals['C_I'])
        self.assertEqual(totals['interceptor_training_C_I'], (8, 16))


if __name__ == '__main__':
    unittest.main()
