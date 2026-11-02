import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


_PATH = Path(__file__).resolve().parents[1] / 'orcd' / 'university_l40s_evaluation_memory.py'
_SPEC = importlib.util.spec_from_file_location('university_evaluation_memory_tested', _PATH)
cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cleanup)


class FakeCuda:
    def __init__(self):
        self.calls = []
        self.reserved = 1000

    def synchronize(self):
        self.calls.append('synchronize')

    def mem_get_info(self, device):
        return 2000 - self.reserved, 2000

    def memory_allocated(self, device):
        return 100

    def memory_reserved(self, device):
        return self.reserved

    def empty_cache(self):
        self.calls.append('empty_cache')
        self.reserved = 100


class UniversityEvaluationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.rng = torch.get_rng_state().clone()
        sender = torch.nn.Linear(2, 2)
        receiver = torch.nn.Linear(2, 2)
        sender_optimizer = torch.optim.AdamW(sender.parameters())
        receiver_optimizer = torch.optim.AdamW(receiver.parameters())
        parameter = next(sender.parameters())
        sender_optimizer.state[parameter]['exp_avg'] = torch.zeros_like(parameter)
        self.trainer = SimpleNamespace(
            sender=sender, receiver=receiver,
            observer=SimpleNamespace(model=torch.nn.Linear(2, 2)),
            sender_optimizer=sender_optimizer, receiver_optimizer=receiver_optimizer,
            ppo=SimpleNamespace(model=sender, ref_model=torch.nn.Linear(2, 2),
                                optimizer=sender_optimizer))
        self.cuda = FakeCuda()
        self.training = SimpleNamespace(capture_rng=lambda: torch.get_rng_state().clone())

    def tearDown(self):
        torch.set_rng_state(self.rng)

    def assert_rng_equal(self, actual, expected, label):
        self.assertTrue(torch.equal(actual, expected), label)

    def release(self, collect):
        return cleanup.release_unused_cuda_cache(
            self.trainer, self.training, self.assert_rng_equal,
            torch_module=SimpleNamespace(cuda=self.cuda), collect=collect)

    def test_cleanup_releases_only_unused_cache_and_preserves_live_metadata_rng(self):
        before = cleanup.live_state_metadata(self.trainer)

        def collect():
            self.cuda.calls.append('collect')
            return 7

        result = self.release(collect)
        self.assertEqual(self.cuda.calls, ['synchronize', 'collect', 'empty_cache', 'synchronize'])
        self.assertEqual(before, cleanup.live_state_metadata(self.trainer))
        self.assertTrue(result['rng_unchanged'])
        self.assertTrue(result['live_tensor_metadata_unchanged'])
        self.assertEqual(result['gc_collected_objects'], 7)
        self.assertEqual(result['before']['cuda_allocated_bytes'], result['after']['cuda_allocated_bytes'])
        self.assertLess(result['after']['cuda_reserved_bytes'], result['before']['cuda_reserved_bytes'])

    def test_optimizer_tensor_mutation_or_replacement_fails_closed(self):
        state = next(iter(self.trainer.sender_optimizer.state.values()))
        with self.assertRaisesRegex(RuntimeError, 'tensor metadata changed'):
            self.release(lambda: state['exp_avg'].add_(1))
        with self.assertRaisesRegex(RuntimeError, 'tensor metadata changed'):
            self.release(lambda: state.update(exp_avg=state['exp_avg'].clone()))

    def test_rng_consumption_fails_closed(self):
        with self.assertRaisesRegex(AssertionError, 'evaluation_cache_cleanup_rng'):
            self.release(lambda: torch.rand(1))


if __name__ == '__main__':
    unittest.main()
