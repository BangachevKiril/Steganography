#!/usr/bin/env python3
"""Read-only L40S restoration/observer diagnostic; never trains or generates.

Run in the unchanged historical training environment with bitsandbytes 0.43.3.
No dependency overlay is used. All outputs go to --output, outside the original source/run/checkpoint.
The imported training source and its Observer.score/Trainer.restore are unchanged.
Completion means the diagnostic ran, not approval of numerical equivalence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def equal_state(actual, expected, label='state'):
    """Exact recursive comparison without emitting checkpoint contents."""
    import numpy as np
    import torch
    if isinstance(expected, torch.Tensor):
        if (not isinstance(actual, torch.Tensor) or actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.equal(actual.detach().cpu(), expected.detach().cpu())):
            raise AssertionError('Tensor mismatch: ' + label)
    elif isinstance(expected, np.ndarray):
        if (not isinstance(actual, np.ndarray) or actual.dtype != expected.dtype
                or not np.array_equal(actual, expected)):
            raise AssertionError('Array mismatch: ' + label)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError('Mapping keys mismatch: ' + label)
        for key in expected:
            equal_state(actual[key], expected[key], label + '.' + str(key))
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, type(expected)) or len(actual) != len(expected):
            raise AssertionError('Sequence mismatch: ' + label)
        for index, value in enumerate(expected):
            equal_state(actual[index], value, label + '.' + str(index))
    elif actual != expected:
        raise AssertionError('Value mismatch: ' + label)


def verify_restoration(trainer, saved, training):
    parameters = {name: param for name, param in trainer.sender.named_parameters()
                  if param.requires_grad}
    equal_state(parameters, saved['sender_trainable'], 'sender_trainable')
    equal_state(trainer.receiver.state_dict(), saved['receiver'], 'receiver')
    equal_state(trainer.sender_optimizer.state_dict(), saved['sender_optimizer'], 'sender_optimizer')
    equal_state(trainer.receiver_optimizer.state_dict(), saved['receiver_optimizer'], 'receiver_optimizer')
    equal_state(trainer.ppo.optimizer.state_dict(), saved['sender_optimizer'], 'ppo_optimizer')
    equal_state({key: value.state_dict() for key, value in trainer.streams.items()}, saved['streams'], 'streams')
    equal_state(training.capture_rng(), saved['rng'], 'rng')
    equal_state(trainer.ppo.kl_ctl.value, saved['kl_coef'], 'kl_coef')
    equal_state(trainer.ppo.current_step, saved['ppo_current_step'], 'ppo_current_step')
    equal_state(trainer.elapsed, saved['elapsed'], 'elapsed')
    equal_state(trainer.update, saved['completed_update'], 'completed_update')
    equal_state(trainer.metrics, saved['metrics'], 'metrics')
    equal_state(trainer.public_rows, saved.get('public_rows', []), 'public_rows')
    for role in ('sender', 'receiver'):
        equal_state({name: module.training for name, module in getattr(trainer, role).named_modules()},
                    saved[role + '_modes'], role + '_modes')
    if 'scheduler' in saved:
        equal_state(trainer.ppo.lr_scheduler.state_dict(), saved['scheduler'], 'scheduler')
    if 'scaler' in saved:
        equal_state(trainer.ppo.accelerator.scaler.state_dict(), saved['scaler'], 'scaler')
    return {'exact_sender_trainable_state': True, 'exact_full_receiver_state': True,
            'exact_optimizer_states': True, 'exact_ppo_wrapped_optimizer_state': True,
            'exact_streams': True, 'exact_rng': True, 'exact_model_modes': True,
            'exact_elapsed_and_update': True, 'exact_metrics_and_public_rows': True,
            'exact_kl_coefficient_and_ppo_step': True,
            'scheduler_present': 'scheduler' in saved, 'scaler_present': 'scaler' in saved}


def reconstruct(config, original_run, sandbox, training, status):
    import torch
    from peft import LoraConfig
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from trl import AutoModelForCausalLMWithValueHead, create_reference_model, PPOConfig, PPOTrainer

    # No Trainer.__init__: its guard/writes are not suitable for this diagnostic.
    trainer = training.Trainer.__new__(training.Trainer)
    trainer.config, trainer.run = config, sandbox
    trainer.config_digest = training.digest(config)
    trainer.update, trainer.elapsed, trainer.started = 0, 0., time.monotonic()
    trainer.metrics, trainer.public_rows = [], []
    trainer.publish_metrics = lambda: None
    trainer.log_status = lambda stage, **extra: status(stage, **extra)
    baseline = Path(config['baseline_checkpoint'])
    initial_marker = read(baseline / 'COMPLETE.json')
    if (initial_marker['config_digest'] != config['baseline_config_digest']
            or initial_marker['parameter_updates'] != 0
            or file_sha(baseline / 'state.pt') != config['baseline_state_sha256']):
        raise ValueError('Baseline provenance mismatch')
    initial = torch.load(baseline / 'state.pt', map_location='cpu', weights_only=False)
    if initial['parameter_updates'] != 0 or any(initial['optimizer_states'].values()):
        raise ValueError('Baseline is not fresh step zero')
    training.seed_all(config['seed'])
    assets = config['assets']
    status('loading_sender_receiver')
    trainer.sender = AutoModelForCausalLMWithValueHead.from_pretrained(
        assets['sender']['path'], peft_config=LoraConfig(r=16, lora_alpha=16,
            lora_dropout=.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj', 'v_proj']),
        quantization_config=None, attn_implementation='eager', torch_dtype='float16',
        local_files_only=True).to('cuda:0')
    trainer.sender.config.use_cache = False
    trainer.sender.pretrained_model.config.use_cache = False
    named = dict(trainer.sender.named_parameters())
    if set(initial['sender_trainable']) != {name for name, param in named.items() if param.requires_grad}:
        raise ValueError('Sender initialization keys differ')
    with torch.no_grad():
        for name, value in initial['sender_trainable'].items():
            named[name].copy_(value)
    reference = create_reference_model(trainer.sender)
    trainer.receiver = AutoModelForSequenceClassification.from_pretrained(
        assets['receiver']['path'], num_labels=2, local_files_only=True).to('cuda:0')
    trainer.receiver.classifier = torch.nn.Linear(trainer.receiver.classifier.in_features, 2).to('cuda:0')
    trainer.receiver.load_state_dict(initial['receiver'])
    trainer.receiver.requires_grad_(True)
    trainer.tokenizer = AutoTokenizer.from_pretrained(assets['sender']['path'], local_files_only=True)
    trainer.tokenizer.padding_side = 'left'
    if trainer.tokenizer.pad_token_id is None:
        trainer.tokenizer.pad_token = trainer.tokenizer.eos_token
    trainer.receiver_tokenizer = AutoTokenizer.from_pretrained(assets['receiver']['path'], local_files_only=True)
    del initial

    split = read(original_run / 'TRAIN_SPLIT.json')
    query = read(original_run / 'QUERY_TOKENIZATION.json')
    if split['config_digest'] != trainer.config_digest or query['config_digest'] != trainer.config_digest:
        raise ValueError('Training split/query configuration mismatch')
    sender_ids, receiver_ids = split['sender_ids'], split['receiver_ids']
    if (len(sender_ids) != 9000 or len(receiver_ids) != 9000
            or set(sender_ids) & set(receiver_ids)
            or set(sender_ids) | set(receiver_ids) != set(range(18000))):
        raise ValueError('Invalid recorded training split')
    trainer.query_max_length = query['max_input_length']
    trainer.streams = {'sender': training.ShuffleStream(sender_ids, 192, True),
                       'receiver': training.ShuffleStream(receiver_ids, 256)}
    trainer.sender_optimizer = torch.optim.AdamW(trainer.sender.parameters(), lr=1.41e-5, weight_decay=.01)
    trainer.receiver_optimizer = torch.optim.AdamW(trainer.receiver.parameters(), lr=2e-5, weight_decay=.01)
    recorded_ppo_config = read(original_run / 'PPO_CONFIG.json')
    # exp_name defaults to the entry-point filename, which differs for a probe.
    ppo_config = PPOConfig(exp_name=recorded_ppo_config['exp_name'],
        model_name=assets['sender']['path'], learning_rate=1.41e-5,
        batch_size=192, mini_batch_size=64, gradient_accumulation_steps=3,
        init_kl_coef=.1, target=30, optimize_cuda_cache=True, early_stopping=False, log_with=None)
    trainer.ppo = PPOTrainer(ppo_config, trainer.sender, reference,
                            tokenizer=trainer.tokenizer, optimizer=trainer.sender_optimizer)
    # PPOTrainer fills the derived PEFT/world-size fields in this object.
    equal_state(json.loads(json.dumps(ppo_config.to_dict())), recorded_ppo_config, 'PPO_CONFIG')
    status('loading_observer')
    trainer.observer = training.Observer(config)

    # Fail closed if subsequent diagnostic code accidentally calls any updates.
    def forbidden(*args, **kwargs):
        raise RuntimeError('Training, generation, or optimizer step prohibited in L40S probe')
    for name in ('step',):
        setattr(trainer.sender_optimizer, name, forbidden)
        setattr(trainer.receiver_optimizer, name, forbidden)
        setattr(trainer.ppo.optimizer, name, forbidden)
    for name in ('step', 'generate'):
        setattr(trainer.ppo, name, forbidden)
    for name in ('train', 'generate', 'sender_update', 'receiver_update', 'evaluate', 'checkpoint'):
        setattr(trainer, name, forbidden)
    return trainer


def finite_smoke(trainer, saved, training, rows):
    import torch
    inspected = {}
    for role in ('sender', 'receiver'):
        inspected[role] = 0
        for name, param in getattr(trainer, role).named_parameters():
            if not torch.isfinite(param).all():
                raise FloatingPointError('Nonfinite model parameter: ' + role + '.' + name)
            inspected[role] += param.numel()
    versions = {role: {name: param._version for name, param in getattr(trainer, role).named_parameters()}
                for role in ('sender', 'receiver')}
    # One frozen forward, with actual fixed padded queries/ones-like TRL mask.
    # This is not an optimization or generation smoke and does not establish
    # backward/optimizer compatibility or full training peak-memory fitness.
    with training.isolated_models(trainer.sender, trainer.receiver):
        trainer.sender.eval()
        trainer.receiver.eval()
        texts = [row['answer'] for row in rows[:2]]
        prompts = [training.uni_prompt(row['input']) for row in rows[:2]]
        queries = trainer.tokenizer(prompts, truncation=True, padding='max_length',
            max_length=trainer.query_max_length, return_tensors='pt').to('cuda:0')
        with torch.no_grad():
            outputs = trainer.sender(input_ids=queries['input_ids'],
                attention_mask=torch.ones_like(queries['input_ids']))
            if not torch.isfinite(outputs[0]).all() or not torch.isfinite(outputs[2]).all():
                raise FloatingPointError('Nonfinite sender logits/value head')
            tokens = trainer.receiver_tokenizer(texts, padding=True, truncation=True,
                max_length=512, return_tensors='pt').to('cuda:0')
            receiver_logits = trainer.receiver(**tokens).logits
            if not torch.isfinite(receiver_logits).all():
                raise FloatingPointError('Nonfinite receiver logits')
        result = {'sender_logits_finite': True, 'sender_values_finite': True,
                  'receiver_logits_finite': True, 'forward_batch_size': 2,
                  'sender_query_length': trainer.query_max_length,
                  'finite_parameter_elements': inspected, 'optimizer_steps': 0,
                  'generations': 0, 'backward_passes': 0}
    for role, before in versions.items():
        equal_state({name: param._version for name, param in getattr(trainer, role).named_parameters()}, before,
                    role + '_parameter_versions')
    verify_restoration(trainer, saved, training)
    return result


def reference_populations(run, config_digest, training):
    evaluation = run / 'evaluations' / 'step_000050'
    marker = read(evaluation / 'COMPLETE.json')
    evaluation_config = read(evaluation / 'config.json')
    payload = read(evaluation / 'UNIVERSITY_RESULTS.json')
    if (marker['config_digest'] != config_digest or marker['completed_update'] != 50
            or marker['evaluation_config_digest'] != training.digest(evaluation_config)
            or payload['config_digest'] != marker['evaluation_config_digest']
            or file_sha(evaluation / 'SUMMARY.json') != marker['summary_sha256']):
        raise ValueError('Evaluation50 provenance mismatch')
    rows = payload['rows']
    if len(rows) != 2000 or [row['example_id'] for row in rows] != list(range(2000)):
        raise ValueError('Expected ordered full2000 evaluation50 rows')
    for index, row in enumerate(rows):
        start = index // 64 * 64
        if row['observer_batch_start'] != start or row['observer_batch_size'] != min(64, 2000 - start):
            raise ValueError('Evaluation observer batching changed')
    path = run / 'rollouts' / 'update_000082.json'
    rollout = read(path)
    if (rollout['config_digest'] != config_digest or rollout['update'] != 82
            or rollout['kind'] != 'sender' or len(rollout['rows']) != 192
            or training.phase_for_update(82) != 'sender'):
        raise ValueError('Expected recorded sender update82 with192 rollouts')
    populations = [('evaluation50', rows, evaluation / 'UNIVERSITY_RESULTS.json', 'example_id'),
                   ('training_update82', rollout['rows'], path, 'source_row_id')]
    for name, population, _, _ in populations:
        for row in population:
            score = row['observer_score']
            if (type(row['observer_flag']) is not bool or not math.isfinite(score)
                    or not -1 <= score <= 1 or row['observer_flag'] != (score < 0)):
                raise ValueError('Invalid reference observer result: ' + name)
    return populations


def comparison(trainer, name, rows, identity, output, status, training):
    import numpy as np
    compared = []
    # Both reference populations used unchanged Observer.score, contiguous64.
    # Its right padding and final-column scoring make the batch membership part
    # of the scientific inputs. Never call it per-row or reorder the examples.
    with training.isolated_models(trainer.sender, trainer.receiver):
        for start in range(0, len(rows), 64):
            status('observer_parity', population=name, completed=start, total=len(rows))
            batch = rows[start:start + 64]
            verdicts = trainer.observer.score([row['answer'] for row in batch])
            if len(verdicts) != len(batch):
                raise ValueError('Observer result count differs')
            for index, (row, verdict) in enumerate(zip(batch, verdicts), start):
                old, new = float(row['observer_score']), float(verdict['observer_score'])
                if not math.isfinite(new):
                    raise FloatingPointError('Nonfinite new observer score')
                delta = new - old
                compared.append({'index': index, identity: row[identity], 'true_label': row['true_label'],
                    'answer': row['answer'], 'batch_start': start, 'batch_size': len(batch),
                    'reference_score': old, 'new_score': new, 'score_delta': delta,
                    'absolute_score_delta': abs(delta), 'reference_flag': row['observer_flag'],
                    'new_flag': verdict['observer_flag'],
                    'flag_changed': row['observer_flag'] != verdict['observer_flag'],
                    'observer_reward_term_delta': .2 * delta})
            write(output / (name + '_ROWS.json'), {'population': name,
                'completed': len(compared), 'total': len(rows), 'batch_size': 64, 'rows': compared})
    errors = np.array([row['absolute_score_delta'] for row in compared], dtype=float)
    deltas = np.array([row['score_delta'] for row in compared], dtype=float)
    flips = [row for row in compared if row['flag_changed']]
    old_count = sum(row['reference_flag'] for row in compared)
    new_count = sum(row['new_flag'] for row in compared)
    quantiles = {str(q): float(np.quantile(errors, q)) for q in (0, .5, .9, .95, .99, 1)}
    confusion = {f'reference_{a}_new_{b}': sum(row['reference_flag'] == bool(a)
        and row['new_flag'] == bool(b) for row in compared) for a in (0, 1) for b in (0, 1)}
    return {'n': len(rows), 'batches': math.ceil(len(rows) / 64),
        'reference_C_I_count': old_count, 'reference_C_I': old_count / len(rows),
        'new_C_I_count': new_count, 'new_C_I': new_count / len(rows),
        'C_I_delta': (new_count - old_count) / len(rows),
        'flag_confusion': confusion, 'flag_changes': len(flips),
        'all_flags_identical': not flips, 'changed_indices': [row['index'] for row in flips],
        'changed_ids': [row[identity] for row in flips],
        'changed_scores': [{key: row[key] for key in ('index', identity, 'reference_score', 'new_score',
                            'reference_flag', 'new_flag')} for row in flips],
        'score_abs_error_quantiles': quantiles, 'score_abs_error_max': float(errors.max()),
        'score_abs_error_mean': float(errors.mean()), 'score_delta_mean': float(deltas.mean()),
        'observer_reward_term_delta_mean': float(.2 * deltas.mean()),
        'observer_reward_term_abs_delta_max': float(.2 * errors.max()),
        'reward_note': '0.2*score delta covers only the observer reward term. Any flag flip also switches the receiver-reward gate; that total change is not inferred here.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('training-source', 'training-run', 'checkpoint', 'output'):
        parser.add_argument('--' + argument, required=True, type=Path)
    args = parser.parse_args()
    source, run, checkpoint, output = (getattr(args, key).resolve()
        for key in ('training_source', 'training_run', 'checkpoint', 'output'))
    for protected in (source, run, checkpoint):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Probe output must be separate from source/training/checkpoint trees')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Probe output must be empty; use a fresh diagnostic directory')
    sandbox = output / 'restore_sandbox'
    sandbox.mkdir()
    started = time.monotonic()
    stage = 'starting'
    memory_enabled = False
    memory_phase = None
    memory_phases = []

    def finish_memory_phase():
        if not memory_enabled or memory_phase is None:
            return
        import torch
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info(0)
        memory_phases.append({'phase': memory_phase,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(0),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(0),
            'end_allocated_bytes': torch.cuda.memory_allocated(0),
            'end_reserved_bytes': torch.cuda.memory_reserved(0),
            'end_device_free_bytes': free, 'device_total_bytes': total})
        write(output / 'MEMORY.json', {'phases': memory_phases,
            'scope': 'Frozen restoration, forward smoke, and original-batch observer scoring only; no training.'})

    def status(new_stage, **extra):
        nonlocal stage, memory_phase
        key = new_stage + (':' + extra['population'] if 'population' in extra else '')
        if memory_enabled and key != memory_phase:
            finish_memory_phase()
            import torch
            torch.cuda.reset_peak_memory_stats(0)
            memory_phase = key
        stage = new_stage
        write(output / 'STATUS.json', {'status': 'running', 'stage': stage,
            'updated_at': time.time(), 'elapsed_seconds': time.monotonic() - started, **extra})

    try:
        status('verifying_inputs')
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(source))
        from stego import university_training as training
        if Path(training.__file__).resolve() != source / 'stego' / 'university_training.py':
            raise ValueError('Imported training module is not from requested frozen source')
        config = read(run / 'config.json')
        config_digest = training.digest(config)
        source_hashes = {}
        for relative, expected in config['source_sha256'].items():
            path = source / relative
            actual = file_sha(path)
            if actual != expected:
                raise ValueError('Frozen source differs: ' + relative)
            source_hashes[str(path)] = actual
        marker = read(checkpoint / 'COMPLETE.json')
        if (marker['config_digest'] != config_digest or marker['completed_update'] != 82
                or file_sha(checkpoint / 'state.pt') != marker['state_sha256']):
            raise ValueError('Requested checkpoint configuration/hash mismatch')
        populations = reference_populations(run, config_digest, training)
        monitored = [run / 'config.json', run / 'TRAIN_SPLIT.json', run / 'QUERY_TOKENIZATION.json',
            run / 'PPO_CONFIG.json', checkpoint / 'COMPLETE.json', checkpoint / 'state.pt']
        monitored += [path for _, _, path, _ in populations]
        monitored += [run / 'evaluations' / 'step_000050' / file for file in ('config.json', 'COMPLETE.json', 'SUMMARY.json')]
        input_hashes = {str(path): file_sha(path) for path in monitored}
        write(output / 'MANIFEST.json', {'training_source': str(source), 'training_run': str(run),
            'checkpoint': str(checkpoint), 'config_digest': config_digest,
            'completed_update': marker['completed_update'], 'state_sha256': marker['state_sha256'],
            'script_sha256': file_sha(__file__), 'source_sha256': source_hashes,
            'input_sha256': input_hashes, 'no_training_run_writes': True,
            'observer_implementation': 'Unchanged frozen training.Observer.score',
            'observer_batch_size': 64, 'observer_padding_side': 'right',
            'observer_scoring_column': -1, 'intended_runtime_change': 'A10080 to L40S48; original bitsandbytes0.43.3 retained',
            'acceptance_tolerance': None})
        import torch
        import bitsandbytes
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Probe requires exactly one visible CUDA GPU')
        props = torch.cuda.get_device_properties(0)
        if 'L40S' not in props.name or props.total_memory < 44 * 1024**3:
            raise RuntimeError('Probe requires L40S with at least44 GiB')
        versions = {package: importlib.metadata.version(package) for package in
            ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'bitsandbytes', 'numpy', 'datasets', 'tokenizers')}
        for package, expected in {'torch': '2.4.0', 'transformers': '4.43.2', 'trl': '0.9.6', 'bitsandbytes': '0.43.3'}.items():
            if versions[package].split('+')[0] != expected:
                raise RuntimeError(f'Unexpected {package} version; required {expected}')
        write(output / 'RUNTIME.json', {'versions': versions, 'python': sys.version,
            'executable': sys.executable, 'bitsandbytes_module': bitsandbytes.__file__,
            'cuda': torch.version.cuda, 'gpu': props.name, 'memory_bytes': props.total_memory,
            'capability': list(torch.cuda.get_device_capability(0)), 'visible_gpu_count': 1,
            'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        memory_enabled = True
        trainer = reconstruct(config, run, sandbox, training, status)
        status('restoring_checkpoint')
        saved = torch.load(checkpoint / 'state.pt', map_location='cpu', weights_only=False)
        if saved['config_digest'] != config_digest or saved['completed_update'] != marker['completed_update']:
            raise ValueError('Checkpoint inner state differs from completion marker')
        equal_state(saved['public_rows'], populations[1][1], 'checkpoint_public_rows')
        trainer.restore(checkpoint)
        restoration = verify_restoration(trainer, saved, training)
        write(output / 'RESTORATION.json', {'completed_update': trainer.update, **restoration})
        status('frozen_forward_smoke')
        smoke = finite_smoke(trainer, saved, training, populations[0][1])
        write(output / 'SMOKE.json', smoke)
        comparisons = {name: comparison(trainer, name, rows, identity, output, status, training)
                       for name, rows, _, identity in populations}
        status('verifying_final_state')
        final_restoration = verify_restoration(trainer, saved, training)
        for path, expected in {**source_hashes, **input_hashes}.items():
            if file_sha(path) != expected:
                raise RuntimeError('Original source or input changed during diagnostic: ' + path)
        torch.cuda.synchronize()
        finish_memory_phase()
        memory_enabled = False
        results = {'status': 'complete', 'stage': 'L40S_compatibility_diagnostic',
            'config_digest': config_digest, 'completed_update': trainer.update,
            'restoration': restoration, 'post_probe_restoration': final_restoration,
            'smoke': smoke, 'comparisons': comparisons, 'original_inputs_unchanged': True,
            'peak_allocated_bytes': max(row['peak_allocated_bytes'] for row in memory_phases),
            'peak_reserved_bytes': max(row['peak_reserved_bytes'] for row in memory_phases),
            'memory_phases': memory_phases,
            'parameter_updates': 0, 'backward_passes': 0, 'generations': 0,
            'acceptance_tolerance': None, 'elapsed_seconds': time.monotonic() - started,
            'limitations': ['This is not bit-identical equivalence or a training authorization.',
                'No backward pass or optimizer step was run; training memory/runtime compatibility remains untested.',
                'C_I is the static observer flag fraction, not ground-truth transmission-detection accuracy.',
                'Hardware differs from reference; the original bitsandbytes0.43.3 is retained.']}
        write(output / 'RESULTS.json', results)
        lines = ['# L40S university checkpoint/observer diagnostic', '',
            f"Restored committed update {trainer.update}; no updates, generation or backward passes.",
            'Sender, receiver, optimizers, stream positions, RNG and model modes matched the checkpoint exactly before and after the probe.', '',
            '| Population | Reference flagged | L40S flagged | Flag changes | Max absolute score change | Max absolute 0.2×score change |',
            '|---|---:|---:|---:|---:|---:|']
        for name, result in comparisons.items():
            lines.append(f"| {name} | {result['reference_C_I_count']}/{result['n']} | {result['new_C_I_count']}/{result['n']} | {result['flag_changes']} | {result['score_abs_error_max']:.9g} | {result['observer_reward_term_abs_delta_max']:.9g} |")
        lines += ['', 'Exact flag agreement and score-error quantiles are recorded separately in RESULTS.json. '
            'Public row files include every flag change and both scores. No acceptance tolerance was imposed.', '',
            'A flag change also switches the receiver-reward gate; the 0.2×score comparison alone is not the full reward change.', '',
            *results['limitations'], '']
        (output / 'report.md').write_text('\n'.join(lines))
        write(output / 'COMPLETE.json', {'status': 'complete', 'config_digest': config_digest,
            'completed_update': trainer.update, 'results_sha256': file_sha(output / 'RESULTS.json'),
            'manifest_sha256': file_sha(output / 'MANIFEST.json'), 'completed_at': time.time()})
        write(output / 'STATUS.json', {'status': 'complete', 'stage': 'complete',
            'completed_update': trainer.update, 'elapsed_seconds': time.monotonic() - started})
    except BaseException as exc:
        try:
            finish_memory_phase()
        except Exception:
            # Preserve the original diagnostic failure even after a CUDA error.
            pass
        write(output / 'FAILURE.json', {'status': 'failed', 'stage': stage,
            'error_type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started})
        write(output / 'STATUS.json', {'status': 'failed', 'stage': stage,
            'error_type': type(exc).__name__, 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
