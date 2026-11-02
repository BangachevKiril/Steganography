"""Exercise fresh retention against the original restore and rollback hooks."""
import fcntl
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_with_retention as retention

V1 = HERE.with_name('gemma_cv_nov02_v1')
sys.path.insert(0, str(V1))
from orcd import qwen_cycle_training, sparse_runtime
from stego.train import atomic_json


class RetentionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name).resolve()
        route = patch.object(retention, 'ATTEMPT', root)
        route.start()
        self.addCleanup(route.stop)
        cfg = json.loads((V1 / 'configs/gemma_cv_nov02_full_o9_s42.json').read_text())
        self.run = root / 'runs' / cfg['name']
        self.run.mkdir(parents=True)
        self.lock = (self.run / 'LOCK').open('w')
        self.addCleanup(self.lock.close)
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(self.run / 'config.json', cfg)
        self.digest = retention.digest(cfg)
        self.trainer = SimpleNamespace(run=self.run, config=cfg, config_digest=self.digest)
        self.ck = self.run / 'checkpoints'

    def published(self, step, latest=True):
        folder = self.ck / f'step_{step:06d}'
        folder.mkdir(parents=True)
        state = folder / 'state.pt'
        state.write_bytes(f'optimizer + weights {step}'.encode())
        atomic_json(folder / 'COMPLETE.json', dict(config_digest=self.digest, completed_update=step,
                                                  state_sha256=retention.sha(state)))
        self.trainer.update = step
        self.trainer.current_checkpoint = folder
        if latest:
            atomic_json(self.run / 'LATEST.json', dict(config_digest=self.digest, completed_update=step,
                                                       checkpoint=str(folder)))
        return state

    def test_initialization_and_post_commit_pruning(self):
        zero = self.published(0)
        retention.retain(self.trainer, 'save', atomic_json)
        self.assertTrue(zero.is_file())
        latest = self.published(280)
        def checkpoint(trainer):
            self.assertTrue(latest.exists() and zero.exists())
        training = SimpleNamespace(Trainer=type('T', (), {'checkpoint': checkpoint}), atomic_json=atomic_json)
        cycles = SimpleNamespace(install_cycle_training=lambda module: None)
        sparse = SimpleNamespace(archive_after_rollback=lambda t: None)
        retention.install(cycles, sparse)
        cycles.install_cycle_training(training)
        training.Trainer.checkpoint(self.trainer)
        self.assertEqual(list(self.ck.glob('step_*/state.pt')), [latest])
        self.assertTrue(zero.with_name('COMPLETE.json').exists())
        events = [retention.read(f) for f in (self.run / 'retention_events').glob('*.json')]
        self.assertTrue(any(e['status'] == 'applied' and e['retained_update'] == 280
                            and 'checkpoints/step_000000/state.pt' in e['retired_payloads'] for e in events))

    def test_restore_after_rename_before_latest_and_before_pruning(self):
        zero = self.published(0)
        top = self.published(280, latest=False)
        fragment = self.ck / '.step_000310.123.tmp'
        fragment.mkdir()
        (fragment / 'state.pt').write_bytes(b'interrupted')
        meta = 'COMPLETE.json.tmp-' + 'a' * 32
        (fragment / meta).write_bytes(b'partial metadata')
        ev = self.run / 'evaluations/step_000300/COMPLETE.json'
        atomic_json(ev, {'evidence': 'observed'})
        training = SimpleNamespace(atomic_json=atomic_json)
        witness = []
        class FakeTrainer:
            def checkpoint(self):
                raise AssertionError('Unexpected save')
            def restore(self, checkpoint):
                self.current_checkpoint = checkpoint
                witness.append('loaded')
        training.Trainer = FakeTrainer
        # Use the production sparse installer and its real archive operation.
        with patch.object(qwen_cycle_training, 'install_cycle_training'), \
                patch.object(sparse_runtime, 'archive_after_rollback', sparse_runtime.archive_after_rollback):
            sparse_runtime.install_sparse_runtime(training)
            retention.install(qwen_cycle_training, sparse_runtime)
            qwen_cycle_training.install_cycle_training(training)
            training.Trainer.restore(self.trainer, top.parent)
        self.assertEqual(witness, ['loaded'])
        self.assertFalse(zero.exists() or fragment.exists() or ev.exists())
        self.assertEqual(retention.read(self.run / 'LATEST.json')['checkpoint'], str(top.parent))
        archived = self.run / 'rolled_back_attempts'
        self.assertEqual(next(archived.rglob(meta)).read_bytes(), b'partial metadata')
        self.assertEqual(retention.read(next(archived.rglob('step_000300/COMPLETE.json'))), {'evidence': 'observed'})
        retention.retain(self.trainer, 'restore', atomic_json)
        self.assertTrue(top.exists())

    def test_bad_sha_and_unpublished_newer_state_preserve_payloads(self):
        old = self.published(0)
        top = self.published(280)
        good = top.read_bytes()
        top.write_bytes(good + b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            retention.retain(self.trainer, 'restore', atomic_json)
        self.assertTrue(old.exists())
        top.write_bytes(good)
        partial = self.ck / '.step_000560.11.tmp'
        partial.mkdir()
        (partial / 'state.pt').write_bytes(b'data')
        atomic_json(partial / 'COMPLETE.json', {})
        with self.assertRaisesRegex(ValueError, 'potentially complete'):
            retention.retain(self.trainer, 'restore', atomic_json)
        self.assertTrue(old.exists() and top.exists() and (partial / 'state.pt').exists())

    def test_previously_initialized_without_restart_still_refuses_to_initialize(self):
        atomic_json(self.run / 'INITIALIZATION.json', {'config_digest': self.digest})
        class NeverInitialize:
            def __init__(self, *args):
                raise AssertionError('Initialization was permitted')
            def restore(self, path):
                raise AssertionError('Restore was permitted')
        training = SimpleNamespace(Trainer=NeverInitialize)
        sparse_runtime.install_sparse_runtime(training)
        with self.assertRaisesRegex(RuntimeError, 'no complete restart'):
            training.Trainer(self.trainer.config, self.run)

    def test_real_trainer_roundtrip_with_two_cpu_models(self):
        import torch
        from orcd.ablation_validation import optimizer_digest
        from stego import university_training as frozen

        # The original writer invokes the report module via python -m from the
        # frozen working directory. Scope all hook installation to a subclass.
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(V1)
        class TinyTrainer(frozen.Trainer):
            pass
        training = SimpleNamespace(Trainer=TinyTrainer, atomic_json=atomic_json)
        cycles = SimpleNamespace(install_cycle_training=qwen_cycle_training.install_cycle_training)
        with patch.object(sparse_runtime, 'archive_after_rollback', sparse_runtime.archive_after_rollback):
            sparse_runtime.install_sparse_runtime(training)
            retention.install(cycles, sparse_runtime)
            cycles.install_cycle_training(training)
            t = object.__new__(TinyTrainer)
            t.run, t.config, t.config_digest = self.run, self.trainer.config, self.digest
            t.update, t.elapsed, t.started = 0, 0., time.monotonic()
            t.metrics, t.public_rows = [], []
            t.sender = torch.nn.Linear(2, 2)
            t.receiver = torch.nn.Linear(2, 2)
            t.sender_optimizer = torch.optim.AdamW(t.sender.parameters(), lr=.01)
            t.receiver_optimizer = torch.optim.AdamW(t.receiver.parameters(), lr=.01)
            pairs = [(t.sender, t.sender_optimizer), (t.receiver, t.receiver_optimizer)]
            t.ppo = SimpleNamespace(kl_ctl=SimpleNamespace(value=.1), current_step=0,
                                    accelerator=SimpleNamespace(scaler=None), lr_scheduler=None)
            t.streams = {'receiver': frozen.ShuffleStream([1, 2, 3, 4], 2)}
            t.checkpoint()
            first = t.current_checkpoint
            for model, optimizer in pairs:
                model(torch.tensor([[1., -2.]])).sum().backward()
                optimizer.step()
                optimizer.zero_grad()
            t.receiver.eval()
            t.streams['receiver'].next()
            t.update, t.ppo.current_step, t.metrics = 1, 1, [dict(kind='receiver', update=1)]
            t.checkpoint()
            second = t.current_checkpoint
            expected = [(optimizer_digest(model.state_dict()), optimizer_digest(opt.state_dict()))
                        for model, opt in pairs]
            self.assertFalse((first / 'state.pt').exists())
            self.assertTrue((first / 'COMPLETE.json').exists())
            self.assertEqual(frozen.latest_checkpoint(t.run, self.digest), second)
            for model, optimizer in pairs:
                with torch.no_grad():
                    for p in model.parameters():
                        p.add_(7.)
                self.assertTrue(optimizer.state)
                optimizer.state.clear()
            t.receiver.train()
            t.streams['receiver'].next()
            t.ppo.current_step = 12
            t.restore(second)
            self.assertEqual(t.update, 1)
            self.assertEqual(t.ppo.current_step, 1)
            self.assertFalse(t.receiver.training)
            self.assertEqual(t.streams['receiver'].cursor, 2)
            self.assertEqual(expected, [(optimizer_digest(model.state_dict()), optimizer_digest(opt.state_dict()))
                                        for model, opt in pairs])
            self.assertEqual(list(self.ck.glob('step_*/state.pt')), [second / 'state.pt'])


if __name__ == '__main__':
    unittest.main()
