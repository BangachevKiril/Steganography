"""Run the PIQA communication game with atomic, resumable global steps."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import signal
import statistics
import time
import uuid

from .protocol import (build_messages, check_answer_format, load_piqa,
                       parse_bit, parse_verdict, phase_for_step, rewards,
                       sample_episodes)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('w') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def write_trace(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with gzip.open(tmp, 'wt') as stream:
        json.dump(data, stream, allow_nan=False)
    os.replace(tmp, path)


def summarize(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    counts = {key: sum(row[key] for row in rows) for key in ('c_r', 'c_i', 'd')}
    result = {'n': n, 'C_R': counts['c_r'] / n, 'C_I': counts['c_i'] / n,
            'd': counts['d'] / n, 'correct_receiver': counts['c_r'],
            'correct_interceptor': counts['c_i'], 'rejected_answers': counts['d'],
            'format_pass_rate': sum(row['format_ok'] for row in rows) / n,
            'judge_content_pass_rate': sum(row['judge_pass'] for row in rows) / n}
    if all('bit' in row and 'receiver_guess' in row and 'interceptor_guess' in row for row in rows):
        result['decoder_guesses'] = {
                role: guess_diagnostics([(row['bit'], row[key], row.get('sender')) for row in rows])
                for role, key in [('receiver', 'receiver_guess'), ('interceptor', 'interceptor_guess')]}
    return result


def guess_diagnostics(observations):
    """Public counts only; stratify bias separately from decoding accuracy."""
    def counts(items):
        n = len(items)
        zero = sum(guess == 0 for bit, guess, sender in items)
        correct = sum(guess == bit for bit, guess, sender in items)
        return {'n': n, 'zero': zero, 'correct': correct,
                'zero_rate': zero / n if n else None,
                'accuracy': correct / n if n else None}
    return {**counts(observations),
            'by_true_bit': {str(bit): counts([x for x in observations if x[0] == bit])
                            for bit in (0, 1)},
            'by_receiver_identity': {receiver: counts([x for x in observations if x[2] == sender])
                                      for sender, receiver in [('A', 'B'), ('B', 'A')]}}


def reporting_diagnostics(rows):
    """Literal public-text checks; never feed these flags into rewards."""
    # Lazy import avoids report -> rejection_baseline -> train import cycles.
    from .prompt_comparison_report import FLAG_LABELS, text_flags
    flags = [text_flags(row['answer'], row['public_string']) for row in rows]
    counts = {key: sum(item[key] for item in flags) for key in FLAG_LABELS}
    accepted = [row for row, item in zip(rows, flags)
                if not row['d'] and not any(item.values())]
    return {'text_flag_counts': counts,
            'text_flag_rates': {key: value / len(rows) if rows else None
                                for key, value in counts.items()},
            'accepted_without_detected_disclosure': summarize(accepted),
            'text_flags_note': 'Literal proxies only; no flag changes rewards. The accepted subset excludes any of the four flags.'}


def group_signal_by_role(values, roles):
    """Count groups with nonzero centered rewards, separately by task role."""
    result = {}
    for agent, groups in values.items():
        for group, role in zip(groups, roles[agent]):
            item = result.setdefault(role, {'groups': 0, 'active_groups': 0})
            item['groups'] += 1
            item['active_groups'] += int(max(group) != min(group))
    for item in result.values():
        item['active_group_fraction'] = item['active_groups'] / item['groups']
    return result


def decoder_rollouts_per_answer(config):
    """Keep legacy grouped sampling unless the single-rollout hybrid is selected."""
    count = config.get('decoder_rollouts_per_answer', config['group_size'])
    if type(count) is not int or count not in (1, config['group_size']):
        raise ValueError('decoder_rollouts_per_answer must be 1 or group_size')
    if count == 1 and config.get('decoder_objective', 'grpo') != 'fixed_baseline_pg_ce':
        raise ValueError('Single decoder rollouts require fixed_baseline_pg_ce; singleton GRPO has zero advantage')
    return count


def render_receiver_system_prompt(template, receiver):
    """Substitute only the public receiver identity in a frozen prompt template."""
    if not isinstance(template, str) or not template.strip():
        raise ValueError('receiver_system_prompt must be a nonempty string')
    if template.count('{receiver}') != 1:
        raise ValueError('receiver_system_prompt must contain exactly one {receiver} placeholder')
    remaining = template.replace('{receiver}', '')
    if '{' in remaining or '}' in remaining:
        raise ValueError('receiver_system_prompt permits only the literal {receiver} placeholder')
    if receiver not in ('A', 'B'):
        raise ValueError('Receiver identity must be A or B')
    return template.replace('{receiver}', receiver)


class Experiment:
    def __init__(self, engine, config, run_dir):
        self.engine, self.config, self.run_dir = engine, config, Path(run_dir)
        decoder_rollouts_per_answer(config)
        if 'receiver_system_prompt' in config:
            render_receiver_system_prompt(config['receiver_system_prompt'], 'A')
        self.train = load_piqa(Path(config['data_dir']) / 'train.jsonl')
        self.valid = load_piqa(Path(config['data_dir']) / 'valid.jsonl')
        self.rng = random.Random(config['seed'])
        self.judge_cache = {}
        self.current_step = 0

    def messages(self, role, episode, answer=None):
        messages = build_messages(role, episode, answer)
        if role == 'sender' and 'sender_system_prompt' in self.config:
            override = self.config['sender_system_prompt']
            if not isinstance(override, str) or not override.strip():
                raise ValueError('sender_system_prompt must be a nonempty string')
            messages[0] = {'role': 'system', 'content': override}
        if role == 'receiver' and 'receiver_system_prompt' in self.config:
            messages[0] = {'role': 'system', 'content': render_receiver_system_prompt(
                self.config['receiver_system_prompt'], episode.receiver)}
        return messages

    def status(self, stage, **extra):
        atomic_json(self.run_dir / 'STATUS.json', {
            'status': 'running', 'global_step_in_progress': self.current_step,
            'stage': stage, 'updated_at': datetime.now(timezone.utc).isoformat(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'), **extra})

    def generate(self, requests, kind, stage):
        """Keep adapter batches separate; restore original request ordering."""
        output = [None] * len(requests)
        for agent in ('A', 'B', 'I', 'J'):
            positions = [i for i, (who, _) in enumerate(requests) if who == agent]
            if not positions:
                continue
            self.status(stage, agent=agent, trajectories=len(positions))
            result = self.engine.generate(agent, [requests[i][1] for i in positions], kind=kind)
            if len(result) != len(positions):
                raise RuntimeError('Generation count mismatch')
            for i, trajectory in zip(positions, result):
                output[i] = trajectory
        return output

    def judge(self, episodes, answers):
        keys = [hashlib.sha256(json.dumps([e.question, x], ensure_ascii=False).encode()).hexdigest()
                for e, x in zip(episodes, answers)]
        pending = {}
        for i, key in enumerate(keys):
            if key not in self.judge_cache:
                pending.setdefault(key, i)
        if pending:
            indices = list(pending.values())
            outputs = self.generate([('J', self.messages('judge', episodes[i], answers[i]))
                                     for i in indices], 'judge', 'judge')
            for key, output in zip(pending, outputs):
                verdict = parse_verdict(output.final_text)
                if verdict is None:
                    raise RuntimeError(f'Malformed judge verdict: {output.final_text!r}')
                self.judge_cache[key] = verdict
        return [self.judge_cache[key] for key in keys]

    def score_answers(self, episodes, answers):
        receiver = self.generate([(e.receiver, self.messages('receiver', e, x))
                                  for e, x in zip(episodes, answers)], 'bit', 'receiver')
        interceptor = self.generate([('I', self.messages('interceptor', e, x))
                                     for e, x in zip(episodes, answers)], 'bit', 'interceptor')
        judgments = self.judge(episodes, answers)
        return receiver, interceptor, self.scored_rows(episodes, answers, receiver, interceptor, judgments)

    @staticmethod
    def scored_rows(episodes, answers, receiver, interceptor, judgments):
        rows = []
        for e, x, r, it, judge_pass in zip(episodes, answers, receiver, interceptor, judgments):
            format_ok = check_answer_format(x)
            reward = rewards(parse_bit(r.final_text), parse_bit(it.final_text), e.bit,
                             bool(judge_pass and format_ok))
            rows.append({**asdict(e), 'answer': x, 'receiver_guess': parse_bit(r.final_text),
                         'interceptor_guess': parse_bit(it.final_text), 'judge_pass': judge_pass,
                         'format_ok': format_ok, 'sender_reward': reward.sender,
                         'receiver_reward': reward.receiver, 'interceptor_reward': reward.interceptor,
                         'c_r': reward.c_r, 'c_i': reward.c_i, 'd': reward.d})
        return rows

    def joint_step(self):
        cfg, G = self.config, self.config['group_size']
        episodes = sample_episodes(self.train, cfg['contexts_per_step'], self.rng,
                                   public_bits=cfg['public_bits'])
        repeated = [episode for episode in episodes for _ in range(G)]
        send = self.generate([(e.sender, self.messages('sender', e)) for e in repeated],
                             'sender', 'sender_rollouts')
        answers = [t.final_text for t in send]
        recv, intercept, rows = self.score_answers(repeated, answers)
        D = decoder_rollouts_per_answer(cfg)
        # Retain these RNG draws in both modes so future episode selection is
        # unchanged by removing decoder resampling. In single mode every answer
        # supplies its already-scored decoder trajectories to the update.
        chosen = [i * G + self.rng.randrange(G) for i in range(len(episodes))]
        if D == 1:
            chosen = list(range(len(repeated)))
        extra_indices = [index for index in chosen for _ in range(D - 1)]
        extra_recv = self.generate([(repeated[i].receiver, self.messages('receiver', repeated[i], answers[i]))
                                    for i in extra_indices], 'bit', 'receiver_grpo_groups')
        extra_intercept = self.generate([('I', self.messages('interceptor', repeated[i], answers[i]))
                                         for i in extra_indices], 'bit', 'interceptor_grpo_groups')
        groups, values, roles, targets = (defaultdict(list) for _ in range(4))
        decoder_observations = defaultdict(list)
        for i, episode in enumerate(episodes):
            sl = slice(i * G, (i + 1) * G)
            groups[episode.sender].append(send[sl])
            values[episode.sender].append([row['sender_reward'] for row in rows[sl]])
            roles[episode.sender].append('sender')
            targets[episode.sender].append(None)
        for i, index in enumerate(chosen):
            episode = repeated[index]
            sl = slice(i * (D - 1), (i + 1) * (D - 1))
            for agent, trajectories in [(episode.receiver, [recv[index]] + extra_recv[sl]),
                                        ('I', [intercept[index]] + extra_intercept[sl])]:
                groups[agent].append(trajectories)
                values[agent].append([float(parse_bit(t.final_text) == episode.bit) for t in trajectories])
                roles[agent].append('receiver' if agent != 'I' else 'interceptor')
                targets[agent].append(episode.bit)
                decoder_observations[roles[agent][-1]].extend(
                    (episode.bit, parse_bit(t.final_text), episode.sender) for t in trajectories)
        update_metrics = {}
        for agent in ('A', 'B', 'I'):
            self.status('update', agent=agent)
            if agent == 'I':
                weights = [1 / len(groups[agent])] * len(groups[agent])
            else:
                weights = [0.5 / roles[agent].count(role) for role in roles[agent]]
            extra = ({'target_bits': targets[agent]} if cfg.get('decoder_objective') == 'fixed_baseline_pg_ce'
                     else {})
            update_metrics[agent] = self.engine.update(agent, groups[agent], values[agent],
                                                       group_weights=weights, **extra)
        return {'phase': 'joint', **summarize(rows), 'updates': update_metrics,
                'measurement': f'pre-update sender rollouts: {len(episodes)} contexts x {G} answers',
                'decoder_rollouts_per_answer': D,
                'decoder_answer_contexts': len(chosen),
                'grpo_signal_by_role': group_signal_by_role(values, roles),
                'decoder_objective': cfg.get('decoder_objective', 'grpo'),
                'group_signal_note': 'Within-group reward variation; counterfactual GRPO diagnostic for hybrid decoders, not their update activity.',
                'decoder_training': {role: guess_diagnostics(items)
                                     for role, items in decoder_observations.items()},
                **reporting_diagnostics(rows),
                'by_direction': {who: summarize([r for r in rows if r['sender'] == who])
                                 for who in ('A', 'B')}}, {
                    'episodes': [asdict(e) for e in episodes], 'rows': rows,
                    'sender_trajectories': [asdict(t) for t in send],
                    'decoder_training_groups': {agent: [[asdict(t) for t in group]
                        for group, role in zip(groups[agent], roles[agent]) if role != 'sender']
                        for agent in groups}}

    def interceptor_step(self):
        cfg = self.config
        D = decoder_rollouts_per_answer(cfg)
        episodes = sample_episodes(self.train, cfg['contexts_per_step'], self.rng,
                                   public_bits=cfg['public_bits'])
        send = self.generate([(e.sender, self.messages('sender', e)) for e in episodes],
                             'sender', 'frozen_sender_rollouts')
        inputs = [('I', self.messages('interceptor', e, answer.final_text))
                  for e, answer in zip(episodes, send) for _ in range(D)]
        trajectories = self.generate(inputs, 'bit', 'interceptor_grpo_groups')
        groups = [trajectories[i * D:(i + 1) * D] for i in range(len(episodes))]
        scores = [[float(parse_bit(t.final_text) == e.bit) for t in group]
                  for e, group in zip(episodes, groups)]
        rows, monitoring_seed = None, None
        if cfg.get('monitor_interceptor_answers', False):
            rows, monitoring_seed = self.interceptor_monitor(
                episodes, [t.final_text for t in send], [group[0] for group in groups])
        self.status('update', agent='I')
        extra = ({'target_bits': [e.bit for e in episodes]}
                 if cfg.get('decoder_objective') == 'fixed_baseline_pg_ce' else {})
        update = self.engine.update('I', groups, scores, **extra)
        training = {'C_I': sum(map(sum, scores)) / len(trajectories),
                    'correct_interceptor': int(sum(map(sum, scores))),
                    'n': len(trajectories), 'sender_answers': len(episodes),
                    'decoder_rollouts_per_answer': D,
                    'decoder_guesses': guess_diagnostics([
                        (e.bit, parse_bit(t.final_text), e.sender)
                        for e, group in zip(episodes, groups) for t in group])}
        metrics = {'phase': 'interceptor', 'C_I': training['C_I'], 'n': training['n'],
                   'updates': {'I': update},
                   'decoder_rollouts_per_answer': D,
                   'decoder_answer_contexts': len(episodes),
                   'interceptor_training': training,
                   'decoder_objective': cfg.get('decoder_objective', 'grpo'),
                   'group_signal_note': 'Within-group reward variation; counterfactual GRPO diagnostic for hybrid decoders, not their update activity.',
                   'grpo_signal_by_role': group_signal_by_role({'I': scores}, {'I': ['interceptor'] * len(scores)})}
        trace = {'episodes': [asdict(e) for e in episodes],
                 'sender_trajectories': [asdict(t) for t in send],
                 'interceptor_groups': [[asdict(t) for t in group] for group in groups],
                 'rewards': scores}
        if rows is not None:
            metrics.update(**summarize(rows), **reporting_diagnostics(rows),
                           measurement='pre-update matched probe: one answer and first interceptor guess per context',
                           monitoring_seed=monitoring_seed, interceptor_training=training)
            trace['monitoring_rows'] = rows
        return metrics, trace

    def interceptor_monitor(self, episodes, answers, first_interceptor):
        """Observe frozen R/J on existing answers without advancing training RNG.

        The first existing interceptor guess is fixed in advance;
        the monitoring probe never resamples or selects successful I guesses.
        Its temporary judge cache must not change future training judgments.
        """
        import torch
        seed = self.config.get('monitor_seed', self.config['eval_seed'] + 1000000) + self.current_step
        training_cache = self.judge_cache
        self.judge_cache = {}
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                receiver = self.generate([(e.receiver, self.messages('receiver', e, x))
                                          for e, x in zip(episodes, answers)],
                                         'bit', 'monitor_receiver')
                judgments = self.judge(episodes, answers)
                rows = self.scored_rows(episodes, answers, receiver, first_interceptor, judgments)
        finally:
            self.judge_cache = training_cache
        return rows, seed

    def evaluate(self, step):
        import torch
        destination = self.run_dir / 'validation' / f'step_{step:06d}.json'
        if destination.exists():
            return json.loads(destination.read_text())
        seed = self.config['eval_seed'] + step
        episodes = sample_episodes(self.valid, self.config['eval_questions'], random.Random(seed),
                                   unique_questions=True, public_bits=self.config['public_bits'])
        start = time.monotonic()
        training_cache = self.judge_cache
        self.judge_cache = {}
        try:
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                send = self.generate([(e.sender, self.messages('sender', e)) for e in episodes],
                                     'sender', 'validation_sender')
                _, _, rows = self.score_answers(episodes, [t.final_text for t in send])
        finally:
            self.judge_cache = training_cache
        result = {'global_step': step, 'phase': 'validation', 'single_rollout': True,
                  'sampling_seed': seed, 'seconds': time.monotonic() - start,
                  **summarize(rows), **reporting_diagnostics(rows),
                  'by_direction': {who: summarize([r for r in rows if r['sender'] == who])
                                                    for who in ('A', 'B')}, 'rows': rows}
        atomic_json(destination, result)
        print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)
        return result


class Preempted(Exception):
    pass


def save_checkpoint(experiment, completed_step, config_digest):
    import torch
    root = experiment.run_dir / 'checkpoints'
    root.mkdir(parents=True, exist_ok=True)
    name = f'step_{completed_step:06d}'
    tmp = root / (name + '.tmp-' + uuid.uuid4().hex)
    tmp.mkdir()
    experiment.engine.save(tmp / 'engine')
    torch.save({'completed_step': completed_step, 'python_rng': experiment.rng.getstate(),
                'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                'judge_cache': experiment.judge_cache, 'config_digest': config_digest}, tmp / 'state.pt')
    atomic_json(tmp / 'COMPLETE.json', {'global_step': completed_step, 'config_digest': config_digest})
    os.rename(tmp, root / name)
    atomic_json(experiment.run_dir / 'LATEST.json', {'checkpoint': name, 'global_step': completed_step})
    # Retain every evaluation checkpoint, step zero, and the two latest steps.
    for path in root.glob('step_*'):
        if '.tmp-' in path.name or not path.is_dir():
            continue
        n = int(path.name.split('_')[-1])
        if n != 0 and n % experiment.config['eval_every'] and n < completed_step - 1:
            shutil.rmtree(path)


def record_fresh_initialization(experiment, config_digest):
    """Audit the production process, distinct from the discarded GPU canary."""
    engine = experiment.engine
    counts = {agent: len(opt.state) for agent, opt in engine.optimizers.items()}
    if any(counts.values()):
        raise RuntimeError('Fresh initialization has nonempty optimizer state')
    b_parameters = {agent: [p for name, p in engine.model.named_parameters()
                            if 'lora_B.' + agent + '.' in name] for agent in engine.AGENTS}
    zero_b = {agent: bool(parameters) and all(bool((p.detach() == 0).all().item())
                                             for p in parameters)
              for agent, parameters in b_parameters.items()}
    if not all(zero_b.values()):
        raise RuntimeError('Fresh adapters have nonzero LoRA B initialization')
    atomic_json(experiment.run_dir / 'INITIALIZATION.json', {
        'initialization': 'fresh_base_and_adapters', 'global_step': 0,
        'config_digest': config_digest, 'checkpoint_loaded': False,
        'optimizer_state_entries': counts, 'lora_B_all_zero': zero_b,
        'communicator_path': engine.communicator_path,
        'judge_path': engine.judge_path,
        'recorded_at': datetime.now(timezone.utc).isoformat()})


def restore_checkpoint(experiment, config_digest):
    import torch
    # A process can die after checkpoint promotion and before updating LATEST.
    candidates = []
    for path in (experiment.run_dir / 'checkpoints').glob('step_*'):
        suffix = path.name.removeprefix('step_')
        if suffix.isdigit() and (path / 'COMPLETE.json').is_file() and (path / 'state.pt').is_file():
            candidates.append((int(suffix), path))
    if not candidates:
        if (experiment.run_dir / 'LATEST.json').exists():
            raise RuntimeError('Checkpoint pointer exists but no complete checkpoint was found')
        return 0
    completed, path = max(candidates)
    state = torch.load(path / 'state.pt', map_location='cpu', weights_only=False)
    if state['config_digest'] != config_digest:
        raise RuntimeError('Refusing to resume with changed configuration')
    experiment.engine.load(path / 'engine')
    experiment.rng.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    experiment.judge_cache = state['judge_cache']
    if completed != state['completed_step']:
        raise RuntimeError('Checkpoint step mismatch')
    atomic_json(experiment.run_dir / 'LATEST.json', {'checkpoint': path.name, 'global_step': completed})
    return state['completed_step']


def timing_summary(run_dir, total_steps, completed, eval_every=20):
    durations = defaultdict(list)
    for path in sorted((run_dir / 'metrics').glob('step_*.json')):
        row = json.loads(path.read_text())
        if row['global_step'] <= completed:
            durations[row['phase']].append(row['seconds'])
    medians = {phase: statistics.median(items) for phase, items in durations.items()}
    evaluations = [json.loads(p.read_text())['seconds'] for p in (run_dir / 'validation').glob('step_*.json')]
    remaining = defaultdict(int)
    for step in range(completed + 1, total_steps + 1):
        remaining[phase_for_step(step)] += 1
    eta = None
    if all(phase in medians for phase in remaining):
        eta = sum(medians[phase] * n for phase, n in remaining.items())
        if evaluations:
            eta += (total_steps // eval_every - completed // eval_every) * statistics.median(evaluations)
    return {'completed_step': completed, 'median_step_seconds': medians,
            'observed_steps': {phase:len(items) for phase,items in durations.items()},
            'remaining_step_counts': dict(remaining), 'eta_seconds_excluding_queue': eta,
            'validation_seconds_observed': evaluations,
            'note': 'ETA excludes queue/preemption and checkpoint writes; includes future validation only once measured.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config.get('launch_ready') is False:
        raise RuntimeError('This configuration is a draft; finalize prompts and freeze its source before launching')
    if config.get('model_choice_status') != 'confirmed':
        raise RuntimeError('Finalize the pending model choice before launching this configuration')
    decoder_rollouts_per_answer(config)
    if 'receiver_system_prompt' in config:
        render_receiver_system_prompt(config['receiver_system_prompt'], 'A')
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    # Retain the open descriptor until process exit.
    run_lock = (args.run_dir / 'RUN.lock').open('a')
    try:
        fcntl.flock(run_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('Another worker already owns this run directory') from None
    saved_config = args.run_dir / 'config.json'
    if saved_config.exists() and json.loads(saved_config.read_text()) != config:
        raise RuntimeError('Refusing to replace an existing run configuration')
    source_root = Path(__file__).resolve().parent.parent
    for name, expected in config.get('source_sha256', {}).items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed since run was prepared: {name}')
    for name, expected in config.get('data_sha256', {}).items():
        if hashlib.sha256((Path(config['data_dir']) / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Dataset changed since run was prepared: {name}')
    atomic_json(saved_config, config)
    def interrupt(signum, frame):
        raise Preempted(f'Received signal {signum}')
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, interrupt)
    experiment = None
    try:
        import torch
        from .engine import Engine
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        atomic_json(args.run_dir / 'STATUS.json', {'status':'loading_models', 'config_digest':digest,
                    'slurm_job_id':os.environ.get('SLURM_JOB_ID')})
        engine = Engine(config['communicator_path'], config['judge_path'], config)
        experiment = Experiment(engine, config, args.run_dir)
        completed = restore_checkpoint(experiment, digest)
        if not (args.run_dir / 'LATEST.json').exists():
            record_fresh_initialization(experiment, digest)
            save_checkpoint(experiment, 0, digest)
        if completed and completed % config['eval_every'] == 0:
            experiment.current_step = completed
            experiment.evaluate(completed)
        for step in range(completed + 1, config['total_steps'] + 1):
            experiment.current_step = step
            start = time.monotonic()
            if phase_for_step(step) == 'joint':
                metrics, trace = experiment.joint_step()
            else:
                metrics, trace = experiment.interceptor_step()
            engine.assert_judge_frozen()
            write_trace(args.run_dir / 'private_rollouts' / f'step_{step:06d}.json.gz', trace)
            # Metrics are committed with the following checkpoint. Readers ignore
            # files beyond the highest complete checkpoint after a preemption.
            metrics.update(global_step=step, seconds=time.monotonic()-start)
            atomic_json(args.run_dir / 'metrics' / f'step_{step:06d}.json', metrics)
            save_checkpoint(experiment, step, digest)
            print(json.dumps(metrics, allow_nan=False), flush=True)
            if step % config['eval_every'] == 0:
                experiment.evaluate(step)
            engine.assert_judge_frozen()
            atomic_json(args.run_dir / 'TIMING.json', timing_summary(args.run_dir, config['total_steps'], step, config['eval_every']))
        atomic_json(args.run_dir / 'COMPLETE.json', {'global_step':config['total_steps'],
                    'config_digest':digest, 'finished_at':datetime.now(timezone.utc).isoformat()})
        atomic_json(args.run_dir / 'STATUS.json', {'status':'complete','global_step':config['total_steps']})
    except Preempted as exc:
        atomic_json(args.run_dir / 'STATUS.json', {'status':'preempted','reason':str(exc),
                    'resume':'Last complete checkpoint; partial optimizer updates discarded.'})
        raise SystemExit(75)
    except Exception as exc:
        atomic_json(args.run_dir / 'STATUS.json', {'status':'failed','error':repr(exc)})
        raise


if __name__ == '__main__':
    main()
