"""CPU regression tests for the fixed-budget administrative continuation.

The real frozen train/checkpoint functions are patched and executed with small
fixtures. No model assets, remote commands or production run files are used.
"""
import ast
import copy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('UNIVERSITY_FROZEN_SOURCE',
                             ROOT / 'launches/uni_corrected_20261002T151448Z')).resolve()
WRAPPER = Path(os.environ.get('UNIVERSITY_EXTENSION_WRAPPER',
                              ROOT / 'orcd/university_corrected_extension.py')).resolve()
ORIGIN_ELAPSED = 27930.404972237535
ADDITIONAL_SECONDS = 27900
TARGET_SECONDS = ORIGIN_ELAPSED + ADDITIONAL_SECONDS


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.package_name = '_university_extension_frozen_fixture'
        package = types.ModuleType(cls.package_name)
        package.__path__ = [str(SOURCE / 'stego')]
        sys.modules[cls.package_name] = package
        cls.training = load_module(cls.package_name + '.university_training',
                                   SOURCE / 'stego/university_training.py')
        # The actual baseline was already verified by the administrative
        # wrapper. Avoid loading datasets/models in train-loop-only fixtures.
        evaluation = types.ModuleType(cls.package_name + '.university_evaluation')
        evaluation._baseline_d = Mock(return_value=.08)
        sys.modules[evaluation.__name__] = evaluation
        cls.extension = load_module('_university_extension_wrapper_fixture',
                                    WRAPPER)

    @classmethod
    def tearDownClass(cls):
        for name in list(sys.modules):
            if name == cls.package_name or name.startswith(cls.package_name + '.'):
                del sys.modules[name]
        sys.modules.pop('_university_extension_wrapper_fixture', None)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name).resolve()
        self.manifest = dict(origin_update=530,
                             origin_elapsed_seconds=ORIGIN_ELAPSED,
                             additional_active_seconds=ADDITIONAL_SECONDS,
                             target_active_seconds=TARGET_SECONDS)
        self.training.STOP = False
        self.addCleanup(setattr, self.training, 'STOP', False)

    def make_trainer(self, update_seconds=10000):
        trainer = self.training.Trainer.__new__(self.training.Trainer)
        trainer.run = self.run
        trainer.config = dict(evaluate_initial_checkpoint=True,
                              evaluation_interval=50, max_active_seconds=27900)
        trainer.config_digest = 'immutable-scientific-config'
        trainer.update = 530
        trainer.elapsed = ORIGIN_ELAPSED
        trainer.current_checkpoint = self.run / 'checkpoints/step_000530'
        trainer.current_checkpoint.mkdir(parents=True)
        trainer.metrics = [dict(update=i, kind=self.training.phase_for_update(i),
                                duration_seconds=1.0)
                           for i in range(1, 531)]
        trainer.public_rows = []
        trainer.events = []
        trainer.preempt_after_update = None
        trainer.active_seconds = lambda: trainer.elapsed
        trainer.log_status = Mock()

        def update(kind):
            trainer.events.append(('update', trainer.update + 1, kind))
            trainer.elapsed += update_seconds
            return {'n': 256 if kind == 'receiver' else 192}

        def checkpoint():
            trainer.events.append(('checkpoint', trainer.update))
            trainer.current_checkpoint = self.run / f'checkpoints/step_{trainer.update:06d}'
            trainer.current_checkpoint.mkdir(parents=True, exist_ok=True)
            (trainer.current_checkpoint / 'COMPLETE.json').write_text(json.dumps({
                'completed_update': trainer.update, 'config_digest': trainer.config_digest,
                'elapsed': trainer.elapsed}))
            if trainer.update == trainer.preempt_after_update:
                self.training.STOP = True

        def evaluate():
            trainer.events.append(('evaluation', trainer.update))
            evaluation = self.run / f'evaluations/step_{trainer.update:06d}'
            evaluation.mkdir(parents=True, exist_ok=True)
            (evaluation / 'COMPLETE.json').write_text(json.dumps({
                'completed_update': trainer.update, 'config_digest': trainer.config_digest}))

        trainer.receiver_update = lambda: update('receiver')
        trainer.sender_update = lambda: update('sender')
        trainer.checkpoint = checkpoint
        trainer.evaluate = evaluate
        return trainer

    def assert_code_equal(self, actual, expected):
        # Ignore only compiler location metadata. All executable instructions,
        # constants, name lookups, locals and closures must agree.
        for field in ('co_code', 'co_names', 'co_varnames',
                      'co_freevars', 'co_cellvars', 'co_argcount', 'co_kwonlyargcount'):
            self.assertEqual(getattr(actual, field), getattr(expected, field), field)
        self.assertEqual(len(actual.co_consts), len(expected.co_consts))
        for actual_constant, expected_constant in zip(actual.co_consts, expected.co_consts):
            if isinstance(actual_constant, types.CodeType):
                self.assertIsInstance(expected_constant, types.CodeType)
                self.assert_code_equal(actual_constant, expected_constant)
            else:
                self.assertEqual(actual_constant, expected_constant)

    def test_train_patch_changes_only_stop_threshold_and_uses_live_globals(self):
        original = self.training.Trainer.train
        original_bytes = Path(self.training.__file__).read_bytes()
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        replacements = []

        class ThresholdOnly(ast.NodeTransformer):
            def visit_Subscript(inner, node):
                if (isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == 'self' and node.value.attr == 'config'
                        and isinstance(node.slice, ast.Constant)
                        and node.slice.value == 'max_active_seconds'):
                    replacements.append(node)
                    return ast.copy_location(ast.Constant(TARGET_SECONDS), node)
                return inner.generic_visit(node)

        expected_tree = ast.fix_missing_locations(ThresholdOnly().visit(tree))
        namespace = dict(self.training.__dict__)
        exec(compile(expected_tree, '<independent expected threshold change>', 'exec'), namespace)
        patched = self.extension.patch_train_function(self.training, self.manifest)
        self.assertEqual(len(replacements), 1)
        self.assert_code_equal(patched.__code__, namespace['train'].__code__)
        self.assertIs(patched.__globals__, self.training.__dict__)
        self.assertIs(self.training.Trainer.train, original)
        self.assertEqual(Path(self.training.__file__).read_bytes(), original_bytes)

    def test_resume_531_preserves_schedule_and_stops_after_fixed_atomic_boundary(self):
        trainer = self.make_trainer(update_seconds=3986)
        config_before = copy.deepcopy(trainer.config)
        patched = self.extension.patch_train_function(self.training, self.manifest)
        patched(trainer)
        updates = [event for event in trainer.events if event[0] == 'update']
        self.assertEqual(updates, [
            ('update', 531, 'sender'), ('update', 532, 'sender'),
            ('update', 533, 'receiver'), ('update', 534, 'receiver'),
            ('update', 535, 'sender'), ('update', 536, 'sender'),
            ('update', 537, 'sender')])
        self.assertLess(trainer.elapsed - 3986, TARGET_SECONDS)
        self.assertGreaterEqual(trainer.elapsed, TARGET_SECONDS)
        self.assertEqual(trainer.config, config_before)
        complete = json.loads((self.run / 'COMPLETE.json').read_text())
        budget = json.loads((self.run / 'BUDGET_COMPLETE.json').read_text())
        self.assertEqual(complete['completed_update'], 537)
        self.assertEqual(budget['completed_update'], 537)
        self.assertEqual(complete['config_digest'], trainer.config_digest)
        self.assertEqual(complete['evaluation_complete_sha256'], self.training.sha256(
            self.run / 'evaluations/step_000537/COMPLETE.json'))

    def test_preemption_requeue_and_repeated_invocation_never_reset_allowance(self):
        trainer = self.make_trainer(update_seconds=10000)
        patched = self.extension.patch_train_function(self.training, self.manifest)
        trainer.preempt_after_update = 531
        with self.assertRaises(SystemExit) as stopped:
            patched(trainer)
        self.assertEqual(stopped.exception.code, 75)
        self.assertEqual(trainer.update, 531)
        self.assertEqual(trainer.elapsed, ORIGIN_ELAPSED + 10000)
        self.assertFalse((self.run / 'BUDGET_COMPLETE.json').exists())
        self.assertFalse((self.run / 'COMPLETE.json').exists())
        # Requeue restores the committed update and elapsed clock. Rebuilding
        # the wrapper from the same immutable manifest cannot add 27900 again.
        self.training.STOP = False
        trainer.preempt_after_update = None
        requeued = self.extension.patch_train_function(self.training, copy.deepcopy(self.manifest))
        requeued(trainer)
        self.assertEqual(trainer.update, 533)
        self.assertEqual(trainer.elapsed, ORIGIN_ELAPSED + 30000)
        self.assertEqual([e[1] for e in trainer.events if e[0] == 'update'], [531, 532, 533])
        events = list(trainer.events)
        complete_bytes = (self.run / 'COMPLETE.json').read_bytes()
        requeued(trainer)
        self.assertEqual(trainer.events, events)
        self.assertEqual((self.run / 'COMPLETE.json').read_bytes(), complete_bytes)
        self.assertEqual(self.manifest['target_active_seconds'], TARGET_SECONDS)

    def test_completed_run_is_untouched_and_pending_final_evaluation_adds_no_updates(self):
        trainer = self.make_trainer()
        patched = self.extension.patch_train_function(self.training, self.manifest)
        sentinel = b'preserved completion receipt'
        (self.run / 'COMPLETE.json').write_bytes(sentinel)
        patched(trainer)
        self.assertEqual(trainer.events, [])
        self.assertEqual((self.run / 'COMPLETE.json').read_bytes(), sentinel)
        (self.run / 'COMPLETE.json').unlink()
        trainer.elapsed = TARGET_SECONDS + 5
        evaluate = trainer.evaluate
        trainer.evaluate = Mock(side_effect=RuntimeError('judge interrupted'))
        with self.assertRaisesRegex(RuntimeError, 'judge interrupted'):
            patched(trainer)
        self.assertTrue((self.run / 'BUDGET_COMPLETE.json').exists())
        self.assertFalse((self.run / 'COMPLETE.json').exists())
        trainer.evaluate = evaluate
        patched(trainer)
        self.assertEqual([e for e in trainer.events if e[0] == 'update'], [])
        self.assertTrue((self.run / 'COMPLETE.json').exists())

    def test_real_checkpoint_retention_preserves_origin_530_and_normal_pruning(self):
        try:
            import torch
        except ImportError:
            self.skipTest('PyTorch unavailable; remaining administrative tests use stdlib fixtures')
        trainer = self.training.Trainer.__new__(self.training.Trainer)
        trainer.run = self.run
        trainer.config = {'checkpoint_keep_recent': 3, 'evaluation_interval': 50}
        trainer.config_digest = 'cpu-retention-fixture'
        trainer.update = 533
        trainer.metrics = []
        trainer.public_rows = []
        trainer.sender = torch.nn.Linear(2, 2)
        trainer.receiver = torch.nn.Linear(2, 2)
        trainer.sender_optimizer = torch.optim.AdamW(trainer.sender.parameters())
        trainer.receiver_optimizer = torch.optim.AdamW(trainer.receiver.parameters())
        trainer.ppo = types.SimpleNamespace(kl_ctl=types.SimpleNamespace(value=.1),
            current_step=0, lr_scheduler=None, accelerator=types.SimpleNamespace(scaler=None))
        trainer.streams = {}
        trainer.active_seconds = lambda: TARGET_SECONDS
        trainer.publish_metrics = Mock()
        for step in (500, 529, 530, 531, 532):
            checkpoint = self.run / f'checkpoints/step_{step:06d}'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'state.pt').write_bytes(f'saved fixture {step}'.encode())
        origin = self.run / 'checkpoints/step_000530/state.pt'
        origin_bytes = origin.read_bytes()
        original = self.training.Trainer.checkpoint
        patched = self.extension.patch_checkpoint_function(self.training, self.manifest)
        self.assertIs(patched.__globals__, self.training.__dict__)
        self.assertIs(self.training.Trainer.checkpoint, original)
        patched(trainer)
        self.assertEqual(origin.read_bytes(), origin_bytes)
        self.assertFalse((self.run / 'checkpoints/step_000529').exists())
        for step in (500, 530, 531, 532, 533):
            self.assertTrue((self.run / f'checkpoints/step_{step:06d}').exists())
        trainer.update = 534
        patched(trainer)
        self.assertEqual(origin.read_bytes(), origin_bytes)
        self.assertFalse((self.run / 'checkpoints/step_000531').exists())
        self.assertTrue((self.run / 'checkpoints/step_000500').exists())

    def test_checkpoint_patch_changes_only_origin_retention_predicate(self):
        original = self.training.Trainer.checkpoint
        tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
        replacements = []

        class RetainOriginOnly(ast.NodeTransformer):
            def visit_If(inner, node):
                test = node.test
                if (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
                        and len(test.values) == 3
                        and isinstance(test.values[-1], ast.Compare)
                        and isinstance(test.values[-1].left, ast.BinOp)
                        and isinstance(test.values[-1].left.op, ast.Mod)):
                    replacements.append(node)
                    test.values.append(ast.Compare(left=ast.Name(id='step', ctx=ast.Load()),
                        ops=[ast.NotEq()], comparators=[ast.Constant(530)]))
                return inner.generic_visit(node)

        expected_tree = ast.fix_missing_locations(RetainOriginOnly().visit(tree))
        namespace = dict(self.training.__dict__)
        exec(compile(expected_tree, '<independent expected retention change>', 'exec'), namespace)
        actual = self.extension.patch_checkpoint_function(self.training, self.manifest)
        self.assertEqual(len(replacements), 1)
        self.assert_code_equal(actual.__code__, namespace['checkpoint'].__code__)

    def valid_manifest(self):
        wrapper = self.extension
        return dict(self.manifest,
            schema_version=1, kind='authorized_corrected_university_extension',
            config_digest=wrapper.CONFIG_DIGEST, parent_state_sha256=wrapper.ORIGIN_STATE,
            parent_checkpoint_complete_sha256=wrapper.ORIGIN_MANIFEST,
            runtime_launcher_sha256=wrapper.RUNTIME_SHA256,
            run_dir=str(wrapper.BASE / 'runs' / wrapper.EXPERIMENT),
            training_source=str(wrapper.BASE / 'releases' / wrapper.EXPERIMENT),
            archive_dir=str(wrapper.BASE / 'archives' / (wrapper.EXPERIMENT + '_stage1_update530')),
            authorization='Human authorized another 27900 active seconds on the same corrected run.',
            extension_id='fixed-budget-test', wrapper_sha256='a' * 64,
            parent_complete_sha256='b' * 64, parent_budget_complete_sha256='c' * 64,
            parent_evaluation_complete_sha256='d' * 64)

    def test_manifest_rejects_budget_rebasing_wrong_science_and_missing_authorization(self):
        manifest = self.valid_manifest()
        self.extension.validate_manifest(manifest)
        changes = {
            'origin_update': 531,
            'origin_elapsed_seconds': ORIGIN_ELAPSED + 1,
            'additional_active_seconds': 27901,
            'target_active_seconds': TARGET_SECONDS + 1,
            'config_digest': 'different-config',
            'parent_state_sha256': '0' * 64,
            'parent_checkpoint_complete_sha256': '0' * 64,
            'runtime_launcher_sha256': '0' * 64,
            'run_dir': manifest['run_dir'] + '-different',
            'training_source': manifest['training_source'] + '-different',
            'archive_dir': manifest['archive_dir'] + '-different',
            'authorization': '',
            'extension_id': '',
        }
        for key, changed in changes.items():
            with self.subTest(key=key):
                altered = {**manifest, key: changed}
                with self.assertRaises(ValueError):
                    self.extension.validate_manifest(altered)
        self.assertEqual(manifest, self.valid_manifest())

    def activation_fixture(self):
        archive = self.run / 'preserved-stage-one'
        contents = {
            'COMPLETE.json': {'completed_update': 530, 'active_seconds': ORIGIN_ELAPSED + 136},
            'BUDGET_COMPLETE.json': {'completed_update': 530, 'active_seconds': ORIGIN_ELAPSED + 2},
            'evaluations/step_000530/COMPLETE.json': {'completed_update': 530, 'parameter_updates': 530},
            'checkpoints/step_000530/COMPLETE.json': {'completed_update': 530, 'elapsed': ORIGIN_ELAPSED},
        }
        for name, payload in contents.items():
            path = archive / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload))
        state = archive / 'checkpoints/step_000530/state.pt'
        state.write_bytes(b'CPU test fixture bytes; no model assets')
        manifest = dict(self.manifest, run_dir=str(self.run), archive_dir=str(archive),
            config_digest='fixture-config', extension_id='immutable-extension',
            authorization='test authorization',
            parent_complete_sha256=self.extension.sha256(archive / 'COMPLETE.json'),
            parent_budget_complete_sha256=self.extension.sha256(archive / 'BUDGET_COMPLETE.json'),
            parent_evaluation_complete_sha256=self.extension.sha256(archive / 'evaluations/step_000530/COMPLETE.json'),
            parent_checkpoint_complete_sha256=self.extension.sha256(archive / 'checkpoints/step_000530/COMPLETE.json'),
            parent_state_sha256=self.extension.sha256(state))
        for name in ('COMPLETE.json', 'BUDGET_COMPLETE.json'):
            (self.run / name).write_bytes((archive / name).read_bytes())
        current_checkpoint = self.run / 'checkpoints/step_000530'
        current_checkpoint.mkdir(parents=True)
        (current_checkpoint / 'COMPLETE.json').write_bytes(
            (archive / 'checkpoints/step_000530/COMPLETE.json').read_bytes())
        trainer = types.SimpleNamespace(run=self.run, config_digest='fixture-config',
                                        update=530, elapsed=ORIGIN_ELAPSED,
                                        current_checkpoint=current_checkpoint)
        return trainer, manifest, archive

    def activate(self, trainer, manifest, atomic_json=None):
        # activate imports only this helper from the production package name.
        shim = types.ModuleType('stego.university_training')
        shim.atomic_json = atomic_json or self.training.atomic_json
        with patch.dict(sys.modules, {'stego.university_training': shim}):
            return self.extension.activate(trainer, manifest)

    def test_activation_seals_manifest_before_removal_and_recovers_partial_deletion(self):
        trainer, manifest, archive = self.activation_fixture()
        archive_before = {p.relative_to(archive): p.read_bytes()
                          for p in archive.rglob('*') if p.is_file()}
        original_unlink = Path.unlink

        def persist(path, payload):
            self.assertTrue((self.run / 'COMPLETE.json').exists())
            self.assertTrue((self.run / 'BUDGET_COMPLETE.json').exists())
            self.training.atomic_json(path, payload)

        def interruption(path, *args, **kwargs):
            if path == self.run / 'BUDGET_COMPLETE.json':
                raise OSError('simulated crash after first marker deletion')
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, 'unlink', new=interruption):
            with self.assertRaisesRegex(OSError, 'simulated crash'):
                self.activate(trainer, manifest, atomic_json=persist)
        receipt = self.run / 'CONTINUATION.json'
        self.assertEqual(json.loads(receipt.read_text()), manifest)
        receipt_bytes = receipt.read_bytes()
        self.assertFalse((self.run / 'COMPLETE.json').exists())
        self.assertTrue((self.run / 'BUDGET_COMPLETE.json').exists())
        self.activate(trainer, manifest)
        self.activate(trainer, manifest)
        self.assertFalse((self.run / 'COMPLETE.json').exists())
        self.assertFalse((self.run / 'BUDGET_COMPLETE.json').exists())
        self.assertEqual(receipt.read_bytes(), receipt_bytes)
        self.assertEqual({p.relative_to(archive): p.read_bytes()
                          for p in archive.rglob('*') if p.is_file()}, archive_before)

    def test_activation_preserves_newer_completion_and_rejects_manifest_mutation(self):
        trainer, manifest, _ = self.activation_fixture()
        self.activate(trainer, manifest)
        trainer.update = 650
        trainer.elapsed = TARGET_SECONDS + 1
        for name in ('COMPLETE.json', 'BUDGET_COMPLETE.json'):
            (self.run / name).write_text(json.dumps({'completed_update': 650,
                                                   'active_seconds': trainer.elapsed}))
        before = {name: (self.run / name).read_bytes()
                  for name in ('COMPLETE.json', 'BUDGET_COMPLETE.json', 'CONTINUATION.json')}
        self.activate(trainer, manifest)
        with self.assertRaisesRegex(ValueError, 'authorization changed'):
            self.activate(trainer, {**manifest, 'target_active_seconds': TARGET_SECONDS + 27900})
        for name, data in before.items():
            self.assertEqual((self.run / name).read_bytes(), data)

    def test_activation_rejects_changed_archives_or_wrong_resume_before_any_mutation(self):
        trainer, manifest, archive = self.activation_fixture()
        markers = {name: (self.run / name).read_bytes()
                   for name in ('COMPLETE.json', 'BUDGET_COMPLETE.json')}
        for path in sorted(p for p in archive.rglob('*') if p.is_file()):
            with self.subTest(path=path.relative_to(archive)):
                before = path.read_bytes()
                path.write_bytes(before + b'corrupted')
                with self.assertRaisesRegex(ValueError, 'Preserved completed-stage identity changed'):
                    self.activate(trainer, manifest)
                path.write_bytes(before)
                self.assertFalse((self.run / 'CONTINUATION.json').exists())
                for name, data in markers.items():
                    self.assertEqual((self.run / name).read_bytes(), data)
        for update, elapsed in ((529, ORIGIN_ELAPSED), (530, ORIGIN_ELAPSED - 1),
                                (531, ORIGIN_ELAPSED + 1)):
            with self.subTest(update=update, elapsed=elapsed):
                trainer.update, trainer.elapsed = update, elapsed
                with self.assertRaises(ValueError):
                    self.activate(trainer, manifest)
                self.assertFalse((self.run / 'CONTINUATION.json').exists())
                for name, data in markers.items():
                    self.assertEqual((self.run / name).read_bytes(), data)


if __name__ == '__main__':
    unittest.main()
