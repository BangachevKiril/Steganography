"""Dependency-free bridge from legacy TRL to a frozen, modern Qwen worker.

The worker shares the allocated GPU, but never imports into the training Python
environment. Only input strings and public scalar scores cross JSONL pipes.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import selectors
import signal
import subprocess
import threading
import time


PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 16 * 1024 * 1024


def text_hashes(texts):
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise TypeError('Observer inputs must be a list of strings')
    return [hashlib.sha256(text.encode('utf-8')).hexdigest() for text in texts]


def validate_scores(response, texts):
    """Fail closed on a reordered, truncated, nonfinite or inconsistent reply."""
    if response.get('text_sha256') != text_hashes(texts) or response.get('count') != len(texts):
        raise RuntimeError('Observer response input identity/count mismatch')
    rows = response.get('rows')
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise RuntimeError('Observer response row count mismatch')
    result = []
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError('Observer response row is not an object')
        score = row.get('observer_score')
        flag = row.get('observer_flag')
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not -1 <= score <= 1):
            raise RuntimeError('Observer returned an invalid/nonfinite score')
        if not isinstance(flag, bool) or flag != (score < 0):
            raise RuntimeError('Observer flag is inconsistent with its score')
        result.append({'observer_score': float(score), 'observer_flag': flag})
    return result


class QwenObserverClient:
    model = None  # Existing parent cache-cleanup code has no child tensors to visit.

    def __init__(self, config, run=None):
        self.config = dict(config['qwen_observer'])
        self.model = None
        self._lock = threading.RLock()
        self._closed = False
        self._next_id = 0
        self._messages = queue.Queue()
        self._stderr = None
        self.process = None
        self.request_timeout = float(self.config.get('request_timeout_seconds', 600))
        if self.request_timeout <= 0:
            raise ValueError('Observer timeout must be positive')
        if run is not None:
            events = Path(run) / 'runtime_events'
            events.mkdir(parents=True, exist_ok=True)
            self._stderr = (events / ('qwen_observer_%s_%s.stderr.log' %
                (os.environ.get('SLURM_JOB_ID', 'local'), os.getpid()))).open('ab', buffering=0)
        env = os.environ.copy()
        # Do not inherit the legacy launcher's Python import paths.
        for key in ('PYTHONPATH', 'PYTHONHOME'):
            env.pop(key, None)
        env.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        try:
            self.process = subprocess.Popen(
                [self.config['python'], '-u', self.config['worker_path'],
                 '--parent-pid', str(os.getpid())],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self._stderr, bufsize=0, start_new_session=True, env=env)
            os.set_blocking(self.process.stdin.fileno(), False)
            self._reader = threading.Thread(target=self._read_stdout, daemon=True)
            self._reader.start()
            atexit.register(self.close)
            self.ready = self._request('initialize', config=self.config,
                timeout=float(self.config.get('startup_timeout_seconds', 1800)))
            if (self.ready.get('kind') != 'ready'
                    or self.ready.get('protocol_version') != PROTOCOL_VERSION
                    or self.ready.get('model_revision') != self.config.get('revision')
                    or self.ready.get('non_thinking') is not True
                    or self.ready.get('frozen') != (self.config['k'] == 0)):
                raise RuntimeError('Observer startup audit did not match its configuration')
        except BaseException:
            self.close()
            raise

    def _read_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline(MAX_LINE_BYTES + 1)
                if not line:
                    raise RuntimeError('Observer worker closed stdout')
                if len(line) > MAX_LINE_BYTES or not line.endswith(b'\n'):
                    raise RuntimeError('Observer protocol line exceeds limit/is incomplete')
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError('Observer protocol response is not an object')
                self._messages.put(value)
        except BaseException as exc:
            self._messages.put(exc)

    def _write(self, encoded, deadline):
        view = memoryview(encoded)
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdin, selectors.EVENT_WRITE)
            while view:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError('Observer request write timed out')
                try:
                    count = os.write(self.process.stdin.fileno(), view)
                except BlockingIOError:
                    continue
                view = view[count:]

    def _request(self, command, timeout=None, **payload):
        with self._lock:
            if self._closed:
                raise RuntimeError('Observer worker is closed')
            self._next_id += 1
            request_id = self._next_id
            payload.update(id=request_id, command=command, protocol_version=PROTOCOL_VERSION)
            encoded = (json.dumps(payload, allow_nan=False, separators=(',', ':')) + '\n').encode()
            if len(encoded) > MAX_LINE_BYTES:
                raise ValueError('Observer request exceeds protocol byte limit')
            deadline = time.monotonic() + (self.request_timeout if timeout is None else timeout)
            try:
                self._write(encoded, deadline)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('Observer request timed out')
                try:
                    response = self._messages.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError('Observer request timed out') from exc
                if isinstance(response, BaseException):
                    raise RuntimeError('Observer protocol reader failed') from response
                if response.get('id') != request_id or response.get('protocol_version') != PROTOCOL_VERSION:
                    raise RuntimeError('Observer protocol identity/version mismatch')
                if response.get('ok') is not True:
                    raise RuntimeError('Observer worker failed: ' + str(response.get('error_type', 'unknown'))
                                       + ': ' + str(response.get('error', 'no details')))
                return response
            except BaseException:
                self.close()
                raise

    def score(self, texts):
        text_hashes(texts)
        if not texts:
            return []
        try:
            return validate_scores(self._request('score', texts=texts), texts)
        except BaseException:
            self.close()
            raise

    def cleanup(self):
        audit = self._request('cleanup').get('cleanup')
        if (not isinstance(audit, dict)
                or any(audit.get(key) is not True for key in
                       ('live_tensor_metadata_unchanged', 'modes_unchanged', 'rng_unchanged'))):
            self.close()
            raise RuntimeError('Observer cache-cleanup invariant failed')
        return audit

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            process = self.process
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait(timeout=5)
                if process.stdout is not None:
                    process.stdout.close()
            if self._stderr is not None:
                self._stderr.close()
            atexit.unregister(self.close)


# The launcher can assign training.Observer = QwenObserverClient directly.
Observer = QwenObserverClient
