#!/usr/bin/env python3
"""Frozen Qwen3.5 FP8 observer; JSONL on stdout, diagnostics on stderr.

Runs only in its own modern Transformers environment. No model downloads, no
generation, no truncation, no KV cache, and no persistence of input texts.
Official interfaces: https://huggingface.co/docs/transformers/model_doc/qwen3_5
and https://huggingface.co/docs/transformers/quantization/finegrained_fp8 .
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import signal
import sys


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024 * 1024
NON_THINKING_SUFFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
SELF_TEST_TEXTS = [
    'An experienced software engineer who builds reliable systems.',
    'A graduate of a prestigious top-tier university with an outstanding academic pedigree.',
]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def text_hashes(texts):
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise TypeError('Observer inputs must be a list of strings')
    return [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in texts]


def score_from_logits(no_logit, yes_logit, offset=.75):
    if not all(math.isfinite(value) for value in (no_logit, yes_logit, offset)):
        raise ValueError('Nonfinite observer logits/offset')
    score = math.tanh(no_logit - yes_logit - offset)
    return {'observer_score': score, 'observer_flag': score < 0}


def format_prompt(tokenizer, template, text):
    # Support the frozen positional placeholder, without changing its contents.
    prompt = template.format(text, text=text)
    rendered = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    if not isinstance(rendered, str) or not rendered.endswith(NON_THINKING_SUFFIX):
        raise ValueError('Tokenizer did not produce the audited non-thinking assistant prefix')
    return rendered


def answer_token_ids(tokenizer, rendered):
    prefix = tokenizer.encode(rendered, add_special_tokens=False)
    result = {}
    for answer in ('No', 'Yes'):
        ids = tokenizer.encode(answer, add_special_tokens=False)
        extended = tokenizer.encode(rendered + answer, add_special_tokens=False)
        if (len(ids) != 1 or extended != prefix + ids
                or tokenizer.decode(ids, skip_special_tokens=False) != answer):
            raise ValueError('Observer answer is not one stable token at the response boundary: ' + answer)
        result[answer] = int(ids[0])
    if result['No'] == result['Yes']:
        raise ValueError('Observer Yes/No token IDs must differ')
    return result


def check_input_lengths(token_ids, limit):
    lengths = [len(row) for row in token_ids]
    if any(length == 0 or length > limit for length in lengths):
        raise ValueError('Observer input exceeds the fixed token limit; truncation is forbidden')
    return lengths


def verify_runtime(config):
    """Check the separately pinned worker environment before allocating a model."""
    versions, source_hashes = {}, {}
    for package, expected in config.get('expected_versions', {}).items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError('Observer runtime package version changed: ' + package)
        versions[package] = actual
    for name, expected in config.get('runtime_source_sha256', {}).items():
        path = Path(name)
        if not path.is_absolute() or '..' in path.parts:
            raise ValueError('Observer runtime source path must be absolute without traversal')
        actual = sha256(path)
        if actual != expected:
            raise ValueError('Observer runtime source changed: ' + name)
        source_hashes[name] = actual
    return {'expected_versions_verified': versions,
            'runtime_source_sha256_verified': source_hashes}


def parent_death_signal(expected_parent):
    """Linux kills this child if the legacy parent disappears, including SIGKILL."""
    if sys.platform == 'linux':
        import ctypes
        if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM) != 0:
            raise OSError('Unable to install observer parent-death signal')
    if os.getppid() != expected_parent:
        raise RuntimeError('Observer parent disappeared before initialization')


class FrozenQwenObserver:
    def __init__(self, config):
        runtime_audit = verify_runtime(config)
        import numpy as np
        import torch
        import transformers
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        self.torch, self.np, self.config = torch, np, config
        path = Path(config['model_path']).resolve()
        revision = config['revision']
        if not path.is_dir() or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Observer requires a local model directory and pinned 40-hex revision')
        self.micro_batch = int(config.get('micro_batch_size', 4))
        self.limit = int(config.get('max_input_tokens', 2048))
        self.offset = float(config.get('offset', .75))
        if self.micro_batch < 1 or self.limit < 1 or self.offset != .75:
            raise ValueError('Invalid observer batch/token limit or changed scientific offset')
        if not isinstance(config.get('prompt'), str) or not config['prompt']:
            raise ValueError('The exact frozen observer prompt is required')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Observer requires exactly one visible allocated GPU')
        prop = torch.cuda.get_device_properties(0)
        if ('H100' not in prop.name or prop.total_memory < 75 * 1024**3
                or torch.cuda.get_device_capability(0)[0] < 9):
            raise RuntimeError('Observer requires an H100 with at least 75 GiB')
        file_hashes = {}
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'):
            if (path / name).is_file():
                file_hashes[name] = sha256(path / name)
        for name, expected in config.get('artifact_sha256', {}).items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Invalid model artifact relative path')
            actual = sha256(path / relative)
            if actual != expected:
                raise ValueError('Pinned observer artifact changed: ' + name)
            file_hashes[name] = actual
        model_config = json.loads((path / 'config.json').read_text())
        quant = model_config.get('quantization_config', {})
        if (model_config.get('model_type') != 'qwen3_5'
                or quant.get('quant_method') != 'fp8'
                or quant.get('weight_block_size') != [128, 128]):
            raise ValueError('Observer is not the expected Qwen3.5 fine-grained FP8 checkpoint')
        seed = int(config.get('worker_seed', 0))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False, revision=revision)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            raise ValueError('Observer tokenizer has no padding/EOS token')
        rendered = format_prompt(self.tokenizer, config['prompt'], SELF_TEST_TEXTS[0])
        self.answer_ids = answer_token_ids(self.tokenizer, rendered)
        # Keep the official full architecture; do not remap/remove FP8 modules.
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(path), revision=revision, local_files_only=True,
            trust_remote_code=False, dtype='auto', device_map={'': 0},
            attn_implementation='sdpa')
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.config.use_cache = False
        if hasattr(self.model.config, 'text_config'):
            self.model.config.text_config.use_cache = False
        if any(p.device.type != 'cuda' or p.device.index != 0 for p in self.model.parameters()):
            raise RuntimeError('Observer parameters are not all on allocated CUDA device 0')
        dtype_counts = {}
        for parameter in self.model.parameters():
            key = str(parameter.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
        if not any('float8' in key for key in dtype_counts):
            raise RuntimeError('Loaded observer has no FP8 parameters; refusing a silent precision change')
        batched = self.score(SELF_TEST_TEXTS)
        individual = [self.score([text])[0] for text in SELF_TEST_TEXTS]
        differences = [abs(a['observer_score'] - b['observer_score']) for a, b in zip(batched, individual)]
        tolerance = config.get('self_test_score_atol')
        if tolerance is not None and max(differences) > float(tolerance):
            raise RuntimeError('Observer startup batch/individual difference exceeds configured tolerance')
        if abs(batched[0]['observer_logit_difference'] - batched[1]['observer_logit_difference']) < 1e-6:
            raise RuntimeError('Observer startup logits do not vary between different public control texts')
        cleanup = self.cleanup()
        self.ready = {
            'kind': 'ready', 'model_revision': revision, 'model_path': str(path),
            'model_class': type(self.model).__name__, 'model_artifact_sha256': file_hashes,
            'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'cuda_version': torch.version.cuda, 'gpu_name': prop.name,
            'gpu_total_bytes': prop.total_memory, 'device': 'cuda:0',
            'answer_token_ids': self.answer_ids, 'padding_side': 'left',
            'non_thinking': True, 'assistant_prefix_sha256': hashlib.sha256(NON_THINKING_SUFFIX.encode()).hexdigest(),
            'observer_prompt_sha256': hashlib.sha256(config['prompt'].encode()).hexdigest(),
            'offset': self.offset, 'frozen': True, 'parameter_dtype_numel': dtype_counts,
            'use_cache': False, 'logits_to_keep': 1, 'micro_batch_size': self.micro_batch,
            'max_input_tokens': self.limit, 'truncation': False,
            'self_test': {'batched': batched, 'individual': individual,
                          'max_abs_score_difference': max(differences),
                          'score_tolerance': tolerance,
                          'same_flags': all(a['observer_flag'] == b['observer_flag'] for a, b in zip(batched, individual)),
                          'full_model_exact_equivalence_claimed': False},
            'startup_cleanup': cleanup,
            'runtime_audit': runtime_audit,
        }

    def rng_digest(self):
        states = (random.getstate(), self.np.random.get_state(),
                  self.torch.get_rng_state().numpy().tobytes(),
                  [state.cpu().numpy().tobytes() for state in self.torch.cuda.get_rng_state_all()])
        return hashlib.sha256(pickle.dumps(states, protocol=4)).hexdigest()

    def metadata(self):
        def tensor(value):
            try:
                version = value._version
            except RuntimeError:
                version = 'untracked_inference_tensor'
            return (id(value), version, tuple(value.shape), str(value.dtype), str(value.device),
                    value.requires_grad, value.data_ptr())
        return {'parameters': {name: tensor(value) for name, value in self.model.named_parameters()},
                'buffers': {name: tensor(value) for name, value in self.model.named_buffers()},
                'modes': {name: module.training for name, module in self.model.named_modules()}}

    def memory(self):
        cuda = self.torch.cuda
        free, total = cuda.mem_get_info(0)
        return {'cuda_allocated_bytes': cuda.memory_allocated(0),
                'cuda_reserved_bytes': cuda.memory_reserved(0),
                'cuda_peak_allocated_bytes': cuda.max_memory_allocated(0),
                'device_free_bytes': free, 'device_total_bytes': total}

    def cleanup(self):
        cuda = self.torch.cuda
        cuda.synchronize()
        before, rng, memory = self.metadata(), self.rng_digest(), self.memory()
        collected = gc.collect()
        cuda.empty_cache()
        cuda.synchronize()
        after = self.metadata()
        if before != after or rng != self.rng_digest():
            raise RuntimeError('Observer cache cleanup changed live tensors, modes or RNG')
        if any(after['modes'].values()):
            raise RuntimeError('Observer unexpectedly entered training mode')
        return {'before': memory, 'after': self.memory(), 'gc_collected_objects': collected,
                'live_tensor_metadata_unchanged': True, 'modes_unchanged': True, 'rng_unchanged': True,
                'live_tensor_metadata_sha256': hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()}

    def score(self, texts):
        text_hashes(texts)
        rows = []
        torch = self.torch
        before_rng = self.rng_digest()
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Observer must remain frozen and in evaluation mode')
        for start in range(0, len(texts), self.micro_batch):
            prompts = [format_prompt(self.tokenizer, self.config['prompt'], text)
                       for text in texts[start:start + self.micro_batch]]
            encoded = self.tokenizer(prompts, add_special_tokens=False, padding=False, truncation=False)
            check_input_lengths(encoded['input_ids'], self.limit)
            batch = self.tokenizer.pad(encoded, padding=True, return_tensors='pt')
            # Left padding makes the final column the last attended token in every row.
            if not bool(batch['attention_mask'][:, -1].all()):
                raise RuntimeError('Observer final token is padding')
            batch = {key: value.to('cuda:0') for key, value in batch.items()
                     if key in ('input_ids', 'attention_mask')}
            with torch.inference_mode():
                output = self.model(**batch, use_cache=False, logits_to_keep=1, return_dict=True)
                logits = output.logits[:, -1, :]
                selected = logits[:, [self.answer_ids['No'], self.answer_ids['Yes']]].float().cpu().tolist()
            for no_logit, yes_logit in selected:
                row = score_from_logits(no_logit, yes_logit, self.offset)
                row['observer_logit_difference'] = no_logit - yes_logit
                rows.append(row)
            del batch, output, logits, selected
        if before_rng != self.rng_digest():
            raise RuntimeError('Frozen observer inference unexpectedly consumed RNG')
        return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-pid', required=True, type=int)
    args = parser.parse_args()
    parent_death_signal(args.parent_pid)
    protocol_stdout = sys.stdout
    worker = None
    previous_id = 0
    while True:
        line = sys.stdin.buffer.readline(MAX_LINE_BYTES + 1)
        if not line:
            return
        request_id = None
        try:
            if len(line) > MAX_LINE_BYTES or not line.endswith(b'\n'):
                raise ValueError('Observer request exceeds line limit/is incomplete')
            request = json.loads(line)
            request_id = request.get('id')
            if (type(request_id) is not int or request_id <= previous_id
                    or request.get('protocol_version') != PROTOCOL_VERSION):
                raise ValueError('Observer protocol request ID/version mismatch')
            previous_id = request_id
            command = request.get('command')
            # Third-party loader and kernels must never contaminate protocol stdout.
            with contextlib.redirect_stdout(sys.stderr):
                if command == 'initialize' and worker is None:
                    worker = FrozenQwenObserver(request['config'])
                    response = dict(worker.ready)
                elif command == 'score' and worker is not None:
                    texts = request['texts']
                    hashes = text_hashes(texts)
                    response = {'rows': worker.score(texts), 'text_sha256': hashes, 'count': len(texts)}
                elif command == 'cleanup' and worker is not None:
                    response = {'cleanup': worker.cleanup()}
                else:
                    raise ValueError('Unknown observer command or invalid initialization state')
            response.update(id=request_id, ok=True, protocol_version=PROTOCOL_VERSION)
        except BaseException as exc:
            response = {'id': request_id, 'ok': False, 'protocol_version': PROTOCOL_VERSION,
                        'error_type': type(exc).__name__, 'error': str(exc)}
            protocol_stdout.write(json.dumps(response, allow_nan=False) + '\n')
            protocol_stdout.flush()
            raise SystemExit(1)
        protocol_stdout.write(json.dumps(response, allow_nan=False) + '\n')
        protocol_stdout.flush()


if __name__ == '__main__':
    main()
