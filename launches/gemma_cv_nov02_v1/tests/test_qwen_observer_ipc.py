"""CPU-only contract tests; no Torch/Transformers import or model download."""
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'orcd' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = load('qwen_observer_client')
worker = load('qwen_observer_worker')


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return messages[0]['content'] + worker.NON_THINKING_SUFFIX

    def encode(self, text, **kwargs):
        if text == 'No':
            return [10]
        if text == 'Yes':
            return [20]
        if text.endswith('No'):
            return [1, 2, 10]
        if text.endswith('Yes'):
            return [1, 2, 20]
        return [1, 2]

    def decode(self, ids, **kwargs):
        return {10: 'No', 20: 'Yes'}[ids[0]]


MOCK_WORKER = textwrap.dedent('''\
import hashlib, json, sys, time
mode = None
for line in sys.stdin:
    req = json.loads(line)
    out = dict(id=req['id'], protocol_version=1, ok=True)
    if req['command'] == 'initialize':
        mode = req['config'].get('test_mode')
        out.update(kind='ready', model_revision=req['config']['revision'], non_thinking=True, frozen=True)
    elif req['command'] == 'score':
        texts = req['texts']
        out.update(count=len(texts), text_sha256=[hashlib.sha256(t.encode()).hexdigest() for t in texts],
                   rows=[dict(observer_score=.25 if i % 2 == 0 else -.5, observer_flag=bool(i % 2))
                         for i, _ in enumerate(texts)])
        if mode == 'wrong_id': out['id'] += 1
        if mode == 'reorder': out['text_sha256'].reverse()
        if mode == 'truncate': out['rows'] = out['rows'][:-1]
        if mode == 'nan': out['rows'][0]['observer_score'] = float('nan')
        if mode == 'wrong_flag': out['rows'][0]['observer_flag'] = True
        if mode == 'worker_error': out.update(ok=False, error_type='OutOfMemoryError', error='CUDA OOM')
        if mode == 'bad_json':
            print('not JSON', flush=True)
            continue
        if mode == 'timeout': time.sleep(.2)
    elif req['command'] == 'cleanup':
        out['cleanup'] = dict(live_tensor_metadata_unchanged=True, modes_unchanged=True, rng_unchanged=True,
                              before={'cuda_allocated_bytes': 42}, after={'cuda_allocated_bytes': 42})
        if mode == 'bad_cleanup': out['cleanup']['rng_unchanged'] = False
    print(json.dumps(out), flush=True)
''')


class QwenObserverPureTests(unittest.TestCase):
    def test_runtime_versions_and_source_hashes_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'modeling.py'
            source.write_text('pinned source\n')
            config = {'expected_versions': {'torch': '2.10.0', 'transformers': '5.2.0'},
                      'runtime_source_sha256': {str(source): worker.sha256(source)}}
            with mock.patch.object(worker.importlib.metadata, 'version',
                                   side_effect=lambda name: config['expected_versions'][name]):
                audit = worker.verify_runtime(config)
                self.assertEqual(audit['expected_versions_verified'], config['expected_versions'])
                self.assertEqual(audit['runtime_source_sha256_verified'], config['runtime_source_sha256'])
                source.write_text('changed source\n')
                with self.assertRaisesRegex(ValueError, 'runtime source changed'):
                    worker.verify_runtime(config)
            with mock.patch.object(worker.importlib.metadata, 'version', return_value='wrong'):
                with self.assertRaisesRegex(ValueError, 'package version changed'):
                    worker.verify_runtime(config)

    def test_runtime_source_paths_cannot_be_relative(self):
        with self.assertRaisesRegex(ValueError, 'must be absolute'):
            worker.verify_runtime({'runtime_source_sha256': {'relative.py': 'bad'}})
        with self.assertRaisesRegex(ValueError, 'must be absolute'):
            worker.verify_runtime({'runtime_source_sha256': {'/tmp/../model.py': 'bad'}})

    def test_logit_difference_equivalence_and_gate(self):
        no, yes, offset = 1.1, .3, .75
        norm = math.log(math.exp(no) + math.exp(yes) + math.exp(3.0))
        row = worker.score_from_logits(no, yes, offset)
        self.assertAlmostEqual(row['observer_score'], math.tanh((no - norm) - (yes - norm) - offset))
        self.assertFalse(row['observer_flag'])
        self.assertTrue(worker.score_from_logits(0, 1)['observer_flag'])
        self.assertFalse(worker.score_from_logits(.75, 0)['observer_flag'])

    def test_nonfinite_logits_fail(self):
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                worker.score_from_logits(bad, 0)

    def test_exact_frozen_prompt_and_non_thinking_prefix(self):
        tokenizer = Tokenizer()
        actual = worker.format_prompt(tokenizer, 'Text:\n"{}"\nQuestion', 'literal {braces}')
        self.assertEqual(actual, 'Text:\n"literal {braces}"\nQuestion' + worker.NON_THINKING_SUFFIX)
        tokenizer.apply_chat_template = lambda *args, **kwargs: '<|im_start|>assistant\n<think>\n'
        with self.assertRaisesRegex(ValueError, 'non-thinking'):
            worker.format_prompt(tokenizer, '{}', 'sample')

    def test_answer_tokens_verified_in_context(self):
        tokenizer = Tokenizer()
        self.assertEqual(worker.answer_token_ids(tokenizer, 'prefix'), {'No': 10, 'Yes': 20})
        original = tokenizer.encode
        tokenizer.encode = lambda text, **kw: [1, 2, 99] if text == 'prefixYes' else original(text, **kw)
        with self.assertRaisesRegex(ValueError, 'stable token'):
            worker.answer_token_ids(tokenizer, 'prefix')

    def test_excess_length_is_never_truncated(self):
        self.assertEqual(worker.check_input_lengths([[1], [2, 3]], 2), [1, 2])
        for ids in ([[1, 2, 3]], [[]]):
            with self.assertRaisesRegex(ValueError, 'truncation is forbidden'):
                worker.check_input_lengths(ids, 2)

    def test_input_identity_hashes_preserve_duplicates_and_unicode(self):
        texts = ['a', 'é', 'a']
        self.assertEqual(client.text_hashes(texts), worker.text_hashes(texts))
        self.assertEqual(client.text_hashes(texts)[0], client.text_hashes(texts)[2])
        with self.assertRaises(TypeError):
            client.text_hashes(['a', 1])

    def test_boolean_score_rejected(self):
        response = {'text_sha256': client.text_hashes(['x']), 'count': 1,
                    'rows': [{'observer_score': True, 'observer_flag': False}]}
        with self.assertRaisesRegex(RuntimeError, 'invalid/nonfinite'):
            client.validate_scores(response, ['x'])


class QwenObserverIPCTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.worker_path = Path(self.directory.name) / 'worker.py'
        self.worker_path.write_text(MOCK_WORKER)
        self.clients = []

    def tearDown(self):
        for bridge in self.clients:
            bridge.close()
        self.directory.cleanup()

    def bridge(self, mode=None):
        config = {'qwen_observer': {'python': sys.executable, 'worker_path': str(self.worker_path),
                   'revision': 'a' * 40, 'test_mode': mode, 'startup_timeout_seconds': 5,
                   'request_timeout_seconds': .05 if mode == 'timeout' else 5}}
        result = client.QwenObserverClient(config)
        self.clients.append(result)
        return result

    def test_roundtrip_order_empty_batch_and_idempotent_close(self):
        bridge = self.bridge()
        self.assertIsNone(bridge.model)
        self.assertEqual(bridge.score([]), [])
        self.assertEqual(bridge.score(['first', 'second']),
                         [{'observer_score': .25, 'observer_flag': False},
                          {'observer_score': -.5, 'observer_flag': True}])
        self.assertTrue(bridge.cleanup()['rng_unchanged'])
        bridge.close()
        bridge.close()
        self.assertIsNotNone(bridge.process.poll())
        with self.assertRaisesRegex(RuntimeError, 'closed'):
            bridge.score(['x'])

    def test_bad_protocol_and_scores_fail_closed(self):
        for mode in ('wrong_id', 'reorder', 'truncate', 'nan', 'wrong_flag', 'worker_error', 'bad_json'):
            with self.subTest(mode=mode):
                bridge = self.bridge(mode)
                with self.assertRaises(RuntimeError):
                    bridge.score(['first', 'second'])
                self.assertIsNotNone(bridge.process.poll())

    def test_response_timeout_closes_child(self):
        bridge = self.bridge('timeout')
        with self.assertRaises(TimeoutError):
            bridge.score(['x'])
        self.assertIsNotNone(bridge.process.poll())

    def test_bad_cleanup_invariant_fails_closed(self):
        bridge = self.bridge('bad_cleanup')
        with self.assertRaisesRegex(RuntimeError, 'invariant'):
            bridge.cleanup()
        self.assertIsNotNone(bridge.process.poll())


if __name__ == '__main__':
    unittest.main()
