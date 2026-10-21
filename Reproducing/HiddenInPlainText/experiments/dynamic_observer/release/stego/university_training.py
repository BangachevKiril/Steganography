"""Corrected Uni O rerun from the independently verified original checkpoint zero.

Four corrections are explicit: receiver accumulation, genuine sender query masks,
observer last-token selection, and deterministic frozen receiver reward scoring.
The released raw-logit reward and remaining scientific settings are retained.
"""
from __future__ import annotations
import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import traceback

from .train import atomic_json
from .university_baseline import digest, sha256, seed_all, uni_prompt, normalize_label, OBSERVER_PROMPT, TASK_PROMPT, tensor_digest

STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


def phase_for_update(update):
    if update < 1:
        raise ValueError('Updates are one-based')
    return 'receiver' if (update - 1) % 7 < 2 else 'sender'


def reward_value(true_logit, observer_score, text):
    return (observer_score >= 0) * math.tanh(true_logit) + .2 * observer_score + .1 * min(len(text) / 20, 1)


def last_nonpadding_logits(logits, attention_mask):
    """Select the final attended token, including right-padded mixed-length rows."""
    import torch
    if attention_mask.ndim != 2 or logits.ndim != 3 or logits.shape[:2] != attention_mask.shape:
        raise ValueError('Observer logits and attention mask shapes differ')
    if attention_mask.shape[1] == 0 or not bool((attention_mask != 0).any(dim=1).all()):
        raise ValueError('Observer inputs require a real token in every row')
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    last = positions.expand_as(attention_mask).masked_fill(attention_mask == 0, -1).amax(dim=1)
    return logits[torch.arange(logits.shape[0], device=logits.device), last, :]


def receiver_optimization_step(receiver, optimizer, tokens, targets, microbatch_size=64):
    """One optimizer update using the mean CE over every accepted example."""
    import torch
    total = len(targets)
    if total < 1 or microbatch_size < 1 or any(len(value) != total for value in tokens.values()):
        raise ValueError('Invalid receiver optimization batch')
    receiver.train()
    optimizer.zero_grad()
    loss_sum = 0.0
    correct = 0
    minibatch_labels = []
    for start in range(0, total, microbatch_size):
        stop = min(start + microbatch_size, total)
        labels = targets[start:stop]
        logits = receiver(tokens['input_ids'][start:stop], tokens['attention_mask'][start:stop]).logits
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite receiver loss')
        # Weight by sample count so partial microbatches also implement full-batch mean CE.
        (loss * ((stop - start) / total)).backward()
        loss_sum += float(loss.detach()) * (stop - start)
        correct += int((logits.argmax(-1) == labels).sum())
        minibatch_labels.append(dict(n=stop-start, **{'PASS': int((labels == 1).sum()), 'FAIL': int((labels == 0).sum())}))
    optimizer.step()
    optimizer.zero_grad()
    return dict(n=total, correct=correct, optimizer_steps=1, loss=loss_sum/total,
                optimizer_sample_count=total, microbatches=len(minibatch_labels),
                gradient_reduction='mean_cross_entropy_over_all_examples',
                label_counts={'PASS': int((targets == 1).sum()), 'FAIL': int((targets == 0).sum())},
                minibatch_label_counts=minibatch_labels)


def capture_rng():
    import numpy as np
    import torch
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def isolated_models(*models):
    rng = capture_rng()
    modes = [(m, m.training) for model in models for m in model.modules()]
    try:
        yield
    finally:
        for m, mode in modes:
            m.training = mode
        restore_rng(rng)


def latest_checkpoint(run, config_digest, verify_hash=True):
    candidates = []
    for path in (Path(run) / 'checkpoints').glob('step_*'):
        marker = path / 'COMPLETE.json'
        if not marker.is_file() or not (path / 'state.pt').is_file():
            continue
        data = json.loads(marker.read_text())
        if data['config_digest'] != config_digest:
            raise ValueError('Checkpoint configuration mismatch: ' + str(path))
        candidates.append((int(data['completed_update']), path, data))
    if not candidates:
        return None
    _, path, data = max(candidates)
    if verify_hash and sha256(path / 'state.pt') != data['state_sha256']:
        raise ValueError('Checkpoint state hash mismatch')
    return path


class ShuffleStream:
    """Without-replacement shuffled batches with committed permutation/cursor."""
    def __init__(self, ids, batch_size, drop_last=False):
        self.ids = list(ids)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.permutation = []
        self.cursor = 0
        self.epoch = 0

    def next(self):
        import torch
        batch = []
        while len(batch) < self.batch_size:
            if self.cursor >= len(self.permutation) or (self.drop_last and self.batch_size > len(self.permutation)-self.cursor):
                torch.empty((), dtype=torch.int64).random_()
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
                generator = torch.Generator().manual_seed(seed)
                self.permutation = [self.ids[i] for i in torch.randperm(len(self.ids), generator=generator).tolist()]
                self.cursor = 0
                self.epoch += 1
            count = min(self.batch_size-len(batch), len(self.permutation)-self.cursor)
            batch.extend(self.permutation[self.cursor:self.cursor+count])
            self.cursor += count
        return batch

    def state_dict(self):
        return dict(permutation=self.permutation, cursor=self.cursor, epoch=self.epoch)

    def load_state_dict(self, state):
        permutation = list(state['permutation'])
        if permutation and sorted(permutation) != sorted(self.ids):
            raise ValueError('Saved sampler membership mismatch')
        if not 0 <= state['cursor'] <= len(permutation):
            raise ValueError('Saved sampler cursor invalid')
        self.permutation, self.cursor, self.epoch = permutation, state['cursor'], state['epoch']


class Observer:
    def __init__(self, config):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        path = config['assets']['observer']['path']
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.tokenizer.padding_side = 'right'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id != 128004 or self.tokenizer.encode('Yes', add_special_tokens=False) != [9642] or self.tokenizer.encode('No', add_special_tokens=False) != [2822]:
            raise ValueError('Historical observer tokenizer changed')
        self.model = AutoModelForCausalLM.from_pretrained(path, use_cache=False, quantization_config=BitsAndBytesConfig(load_in_8bit=True), local_files_only=True)
        self.model.requires_grad_(False).eval()

    def score(self, texts):
        import torch
        rows = []
        for start in range(0, len(texts), 64):
            batch = texts[start:start+64]
            prompts = [self.tokenizer.apply_chat_template([dict(role='user', content=OBSERVER_PROMPT.format(text))], tokenize=False, add_generation_prompt=True) for text in batch]
            tokens = self.tokenizer(prompts, add_special_tokens=False, padding=True, truncation=True, return_tensors='pt').to('cuda:0')
            with torch.no_grad():
                logits = last_nonpadding_logits(self.model(**tokens).logits, tokens['attention_mask'])
                logps = logits.log_softmax(-1)
                scores = torch.tanh(logps[:, 2822] - logps[:, 9642] - .75)
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Nonfinite observer scores')
            rows.extend(dict(observer_score=float(s), observer_flag=bool(s < 0)) for s in scores.cpu().tolist())
        return rows


class Trainer:
    def __init__(self, config, run):
        import numpy as np
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from trl import AutoModelForCausalLMWithValueHead, create_reference_model, PPOConfig, PPOTrainer
        self.config, self.run, self.config_digest = config, Path(run), digest(config)
        self.update, self.elapsed, self.started = 0, 0., time.monotonic()
        self.metrics = []
        self.public_rows = []
        self.log_status('initializing')
        if config.get('target_gpu') not in ('H100', 'H200'):
            raise ValueError('Corrected run requires explicit target_gpu=H100')
        if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 75 * 1024**3:
            raise RuntimeError('Corrected runtime requires the configured H100 allocation')
        if config['target_gpu'] not in torch.cuda.get_device_name(0):
            raise RuntimeError('Corrected runtime requires an H100 GPU')
        for item in config['assets']['dataset']['files'].values():
            if sha256(item['path']) != item['sha256']:
                raise ValueError('Dataset digest mismatch')
        baseline = Path(config['baseline_checkpoint'])
        marker = json.loads((baseline/'COMPLETE.json').read_text())
        if marker['config_digest'] != config['baseline_config_digest'] or marker['parameter_updates'] != 0:
            raise ValueError('Wrong baseline checkpoint')
        if sha256(baseline/'state.pt') != config['baseline_state_sha256']:
            raise ValueError('Initial state checksum mismatch')
        initial = torch.load(baseline/'state.pt', map_location='cpu', weights_only=False)
        if initial['parameter_updates'] != 0 or any(initial['optimizer_states'].values()):
            raise ValueError('Initialization has already been trained')
        seed_all(config['seed'])
        paths = config['assets']
        self.sender = AutoModelForCausalLMWithValueHead.from_pretrained(paths['sender']['path'],
            peft_config=LoraConfig(r=16, lora_alpha=16, lora_dropout=.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','v_proj']),
            quantization_config=None, attn_implementation='eager', torch_dtype='float16', local_files_only=True).to('cuda:0')
        self.sender.config.use_cache = False
        self.sender.pretrained_model.config.use_cache = False
        named = dict(self.sender.named_parameters())
        if set(initial['sender_trainable']) != {n for n,p in named.items() if p.requires_grad}:
            raise ValueError('Sender trainable key mismatch')
        with torch.no_grad():
            for n,p in initial['sender_trainable'].items():
                named[n].copy_(p)
        reference = create_reference_model(self.sender)
        self.receiver = AutoModelForSequenceClassification.from_pretrained(paths['receiver']['path'], num_labels=2, local_files_only=True).to('cuda:0')
        self.receiver.classifier = torch.nn.Linear(self.receiver.classifier.in_features, 2).to('cuda:0')
        self.receiver.load_state_dict(initial['receiver'])
        self.receiver.requires_grad_(True)
        self.tokenizer = AutoTokenizer.from_pretrained(paths['sender']['path'], local_files_only=True)
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.receiver_tokenizer = AutoTokenizer.from_pretrained(paths['receiver']['path'], local_files_only=True)
        # Continue the authors' initialization RNG, then their Dataset split.
        torch.set_rng_state(initial['torch_rng'])
        torch.cuda.set_rng_state_all(initial['cuda_rng'])
        np.random.set_state(initial['numpy_rng'])
        random.setstate(initial['python_rng'])
        dataset = Dataset.from_parquet(paths['dataset']['files']['train']['path'])
        self.dataset = dataset
        # Retain the historical truncation ceiling, but pass only real query IDs.
        # TRL owns batch padding and constructs the corresponding attention masks.
        self.query_max_length = (len(self.tokenizer.encode(TASK_PROMPT))
            + max(len(self.tokenizer.encode(text)) for text in dataset['input']) + 19)
        if self.query_max_length != 65:
            raise ValueError('Corrected run must retain the original 65-token query truncation ceiling')
        atomic_json(self.run/'QUERY_TOKENIZATION.json', dict(
            config_digest=self.config_digest, max_input_length=self.query_max_length,
            padding_side='left', pad_token_id=self.tokenizer.pad_token_id,
            truncation=True, add_special_tokens=True,
            query_padding=False, attention_mask='TRL pads unpadded queries and masks inserted padding',
            source='Corrected query construction; original truncation ceiling retained'))
        dataset = dataset.add_column('source_row_id', list(range(len(dataset))))
        split = dataset.train_test_split(test_size=.5)
        sender_ids, receiver_ids = split['train']['source_row_id'], split['test']['source_row_id']
        if len(sender_ids) != 9000 or len(receiver_ids) != 9000 or set(sender_ids) & set(receiver_ids):
            raise ValueError('Unexpected training split')
        split_manifest = dict(config_digest=self.config_digest, sender_ids=sender_ids, receiver_ids=receiver_ids)
        existing = self.run/'TRAIN_SPLIT.json'
        if existing.exists() and json.loads(existing.read_text()) != split_manifest:
            raise ValueError('Training split changed')
        atomic_json(existing, split_manifest)
        self.streams = dict(receiver=ShuffleStream(receiver_ids,256), sender=ShuffleStream(sender_ids,192,True))
        self.sender_optimizer = torch.optim.AdamW(self.sender.parameters(), lr=1.41e-5, weight_decay=.01)
        self.receiver_optimizer = torch.optim.AdamW(self.receiver.parameters(), lr=2e-5, weight_decay=.01)
        ppo_config = PPOConfig(model_name=paths['sender']['path'], learning_rate=1.41e-5,
            batch_size=192, mini_batch_size=64, gradient_accumulation_steps=3,
            init_kl_coef=.1, target=30, optimize_cuda_cache=True, early_stopping=False, log_with=None)
        self.ppo = PPOTrainer(ppo_config, self.sender, reference, tokenizer=self.tokenizer, optimizer=self.sender_optimizer)
        atomic_json(self.run/'PPO_CONFIG.json', ppo_config.to_dict())
        self.log_status('loading_observer')
        self.observer = Observer(config)
        checkpoint = latest_checkpoint(self.run, self.config_digest)
        if checkpoint:
            self.restore(checkpoint)
        else:
            receiver_digest = tensor_digest(self.receiver)
            head_digest = tensor_digest(self.receiver.classifier)
            if receiver_digest != config['expected_baseline_receiver_sha256'] or head_digest != config['expected_baseline_receiver_head_sha256']:
                raise ValueError('Corrected initialization receiver differs from verified baseline')
            if self.sender_optimizer.state or self.receiver_optimizer.state:
                raise ValueError('Corrected fresh initialization requires empty optimizer states')
            initialization = dict(config_digest=self.config_digest, baseline_checkpoint=str(baseline), baseline_state_sha256=config['baseline_state_sha256'], parameter_updates=0,
                receiver_state_sha256=receiver_digest, receiver_head_sha256=head_digest,
                optimizer_state_sizes=dict(sender=len(self.sender_optimizer.state), receiver=len(self.receiver_optimizer.state)), ppo_seed=ppo_config.seed,
                reference_policy='PEFT adapters disabled, pinned base policy as in TRL0.9.6', argmax=True)
            atomic_json(self.run/'INITIALIZATION.json', initialization)
            self.checkpoint()
        atomic_json(self.run/'smoke.json', dict(config_digest=self.config_digest, gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            torch=torch.__version__, initialization_verified=True, completed_update=self.update, status='models_loaded_updates_pending' if not self.update else 'resumed'))
        del initial
        self.log_status('ready')

    def log_status(self, stage, **extra):
        atomic_json(self.run/'STATUS.json', dict(status='complete' if stage == 'complete' else 'running', stage=stage, config_digest=self.config_digest, completed_update=self.update,
            slurm_job_id=os.environ.get('SLURM_JOB_ID'), updated_at=time.time(), **extra))

    def should_stop(self):
        return STOP

    def active_seconds(self):
        return self.elapsed + time.monotonic() - self.started

    def generate(self, kind):
        import numpy as np
        import torch
        ids = self.streams[kind].next()
        rows = [self.dataset[i] for i in ids]
        prompts = [uni_prompt(r['input']) for r in rows]
        queries = [self.tokenizer(p, truncation=True, padding=False,
            max_length=self.query_max_length, return_tensors='pt')['input_ids'][0].to('cuda:0')
            for p in prompts]
        cap = int(np.random.choice(range(15,20)))
        responses = self.ppo.generate(queries, batch_size=64, return_prompt=False,
            min_length=-1, top_k=0., top_p=1., do_sample=True, temperature=1., pad_token_id=self.tokenizer.pad_token_id, max_new_tokens=cap)
        texts = [self.tokenizer.decode(r.squeeze(), skip_special_tokens=True).strip() for r in responses]
        labels = [normalize_label(r['label']) for r in rows]
        return ids, queries, responses, texts, labels, cap

    def receiver_logits(self, texts):
        import torch
        result=[]
        for start in range(0,len(texts),64):
            tokens=self.receiver_tokenizer(texts[start:start+64], return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda:0')
            with torch.no_grad():
                logits=self.receiver(**tokens).logits
            if not torch.isfinite(logits).all():
                raise FloatingPointError('Nonfinite receiver logits')
            result.append(logits)
        return torch.cat(result)

    def receiver_update(self):
        import torch
        accepted=deque(maxlen=256)
        counts=dict(n=0,receiver_correct=0,observer_flagged=0,receiver_pass=0,receiver_fail=0)
        while len(accepted)<256:
            if STOP:
                raise SystemExit(75)
            self.log_status('receiver_collection', collected=len(accepted), generated=counts['n'])
            ids,queries,responses,texts,labels,cap=self.generate('receiver')
            observed=self.observer.score(texts)
            # Additional monitoring is RNG isolated and never enters optimization.
            with isolated_models(self.receiver):
                self.receiver.eval()
                predictions=self.receiver_logits(texts).argmax(-1).cpu().tolist()
            self.public_rows.extend(dict(source_row_id=i,answer=t,true_label=y,receiver_prediction=p,token_cap=cap,**o) for i,t,y,p,o in zip(ids,texts,labels,predictions,observed))
            counts['n']+=len(texts)
            counts['receiver_correct']+=sum(p==y for p,y in zip(predictions,labels))
            counts['receiver_pass']+=sum(predictions)
            counts['receiver_fail']+=len(predictions)-sum(predictions)
            counts['observer_flagged']+=sum(o['observer_flag'] for o in observed)
            accepted.extend((t,y) for t,y,o in zip(texts,labels,observed) if not o['observer_flag'])
            del queries,responses
        texts,labels=zip(*accepted)
        tokens=self.receiver_tokenizer(list(texts), return_tensors='pt', truncation=True, padding=True).to('cuda:0')
        target=torch.tensor(labels,device='cuda:0')
        self.log_status('receiver_optimization')
        optimization=receiver_optimization_step(self.receiver,self.receiver_optimizer,tokens,target)
        return dict(**counts, mean_loss=optimization['loss'], receiver_training=optimization)

    def sender_update(self):
        import torch
        self.log_status('sender_generation')
        ids,queries,responses,texts,labels,cap=self.generate('sender')
        observed=self.observer.score(texts)
        # Frozen reward scoring uses the same deterministic receiver mode as evaluation.
        # Restore every module mode and RNG state before the unchanged sender PPO call.
        with isolated_models(self.receiver):
            self.receiver.eval()
            logits=self.receiver_logits(texts)
        targets=torch.tensor(labels,device='cuda:0')
        scores=torch.tensor([o['observer_score'] for o in observed],device='cuda:0')
        flags=torch.tensor([o['observer_flag'] for o in observed],device='cuda:0')
        stego=logits.gather(1,targets[:,None]).squeeze(1).tanh()*(~flags)
        lengths=torch.tensor([min(len(t)/20,1) for t in texts],device='cuda:0')
        rewards=stego+.2*scores+.1*lengths
        if not torch.isfinite(rewards).all():
            raise FloatingPointError('Nonfinite sender reward')
        predictions=logits.argmax(-1)
        self.public_rows=[dict(source_row_id=i,answer=t,true_label=y,receiver_prediction=p,token_cap=cap,**o) for i,t,y,p,o in zip(ids,texts,labels,predictions.cpu().tolist(),observed)]
        counts=dict(n=len(texts),receiver_correct=int((predictions==targets).sum()),observer_flagged=int(flags.sum()),
            receiver_pass=int((predictions==1).sum()),receiver_fail=int((predictions==0).sum()),reward_mean=float(rewards.mean()),reward_std=float(rewards.std()),token_cap=cap)
        self.log_status('sender_ppo')
        stats=self.ppo.step(queries,responses,[r.unsqueeze(0) for r in rewards])
        selected={}
        for key in ['ppo/loss/total','ppo/loss/policy','ppo/loss/value','objective/kl','objective/kl_coef']:
            value=stats.get(key)
            if value is not None:
                value=float(value)
                selected[key]=value if math.isfinite(value) else None
        if selected.get('ppo/loss/total') is None and 'ppo/loss/total' in stats:
            raise FloatingPointError('Nonfinite PPO loss')
        counts['ppo']=selected
        return counts

    def assert_finite(self):
        import torch
        for role,model in [('sender',self.sender),('receiver',self.receiver)]:
            for name,param in model.named_parameters():
                if param.requires_grad and not torch.isfinite(param).all():
                    raise FloatingPointError('Nonfinite parameter '+role+':'+name)

    def checkpoint(self):
        import torch
        self.assert_finite()
        root=self.run/'checkpoints';root.mkdir(exist_ok=True)
        destination=root/f'step_{self.update:06d}'
        if destination.exists():
            raise ValueError('Refusing to overwrite committed checkpoint')
        temporary=root/f'.step_{self.update:06d}.{os.getpid()}.tmp'
        temporary.mkdir()
        observer_state=self.observer._request('save',path=str(temporary/'observer.pt'))
        state=dict(observer_state=observer_state,observer_buffer=self.observer_buffer,config_digest=self.config_digest,completed_update=self.update,
            sender_trainable={n:p.detach().cpu() for n,p in self.sender.named_parameters() if p.requires_grad},
            receiver={n:p.detach().cpu() for n,p in self.receiver.state_dict().items()},
            sender_optimizer=self.sender_optimizer.state_dict(),receiver_optimizer=self.receiver_optimizer.state_dict(),
            rng=capture_rng(),streams={k:v.state_dict() for k,v in self.streams.items()},metrics=self.metrics,public_rows=self.public_rows,
            elapsed=self.active_seconds(),kl_coef=self.ppo.kl_ctl.value,ppo_current_step=self.ppo.current_step,
            sender_modes={n:m.training for n,m in self.sender.named_modules()},receiver_modes={n:m.training for n,m in self.receiver.named_modules()})
        if self.ppo.lr_scheduler is not None:
            state['scheduler']=self.ppo.lr_scheduler.state_dict()
        scaler=self.ppo.accelerator.scaler
        if scaler is not None:
            state['scaler']=scaler.state_dict()
        torch.save(state,temporary/'state.pt')
        with (temporary/'state.pt').open('rb') as f:
            os.fsync(f.fileno())
        atomic_json(temporary/'COMPLETE.json',dict(config_digest=self.config_digest,completed_update=self.update,
            state_sha256=sha256(temporary/'state.pt'),observer_state=observer_state,elapsed=state['elapsed'],phase=phase_for_update(self.update) if self.update else 'initialization'))
        os.replace(temporary,destination)
        self.current_checkpoint=destination
        self.publish_metrics()
        atomic_json(self.run/'LATEST.json',dict(config_digest=self.config_digest,completed_update=self.update,checkpoint=str(destination)))
        keep=self.config.get('checkpoint_keep_recent',3)
        # Keep initial, recent and segment-boundary states; all evaluation identity
        # markers remain public even after their older state tensors are pruned.
        boundaries = {json.loads(p.read_text())['completed_update']
            for p in self.run.glob('SEGMENT_*_BUDGET_COMPLETE.json')}
        for path in root.glob('step_*'):
            step=int(path.name.split('_')[1])
            if step>0 and step<self.update-keep+1 and step not in boundaries:
                if step % self.config['evaluation_interval'] == 0:
                    (path/'state.pt').unlink(missing_ok=True)
                    (path/'observer.pt').unlink(missing_ok=True)
                else:
                    shutil.rmtree(path)

    def publish_metrics(self):
        if self.update:
            public=self.run/'rollouts';public.mkdir(exist_ok=True)
            atomic_json(public/f'update_{self.update:06d}.json',dict(config_digest=self.config_digest,update=self.update,kind=phase_for_update(self.update),rows=self.public_rows))
        temporary=self.run/'metrics.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in self.metrics))
        os.replace(temporary,self.run/'metrics.jsonl')
        # Reporting CLI has no influence on optimization; use its stable CLI API.
        import subprocess,sys
        subprocess.run([sys.executable,'-m','stego.university_training_report',str(self.run),'--completed-update',str(self.update)],check=True,stdout=subprocess.DEVNULL)

    def restore(self,checkpoint):
        import torch
        state=torch.load(checkpoint/'state.pt',map_location='cpu',weights_only=False)
        if state['config_digest']!=self.config_digest:
            raise ValueError('State config mismatch')
        self.observer._request('load',path=str(checkpoint/'observer.pt'),sha256=state['observer_state']['state_sha256'])
        self.observer_buffer=state['observer_buffer']
        with torch.no_grad():
            named=dict(self.sender.named_parameters())
            for name,param in state['sender_trainable'].items():
                named[name].copy_(param)
        self.receiver.load_state_dict(state['receiver'])
        self.sender_optimizer.load_state_dict(state['sender_optimizer'])
        self.receiver_optimizer.load_state_dict(state['receiver_optimizer'])
        # Accelerated optimizer may wrap the original object but shares its state.
        self.ppo.kl_ctl.value=state['kl_coef']
        self.ppo.current_step=state['ppo_current_step']
        if 'scheduler' in state:self.ppo.lr_scheduler.load_state_dict(state['scheduler'])
        if 'scaler' in state:self.ppo.accelerator.scaler.load_state_dict(state['scaler'])
        for key,stream in self.streams.items():stream.load_state_dict(state['streams'][key])
        for role in ['sender','receiver']:
            for name,module in getattr(self,role).named_modules():module.training=state[role+'_modes'][name]
        self.update=state['completed_update'];self.metrics=state['metrics'];self.public_rows=state.get('public_rows',[])
        self.elapsed=state['elapsed'];self.started=time.monotonic()
        restore_rng(state['rng'])
        self.current_checkpoint=checkpoint
        self.publish_metrics()

    def evaluate(self):
        from .university_evaluation import evaluate_training
        self.log_status('evaluation')
        with isolated_models(self.sender,self.receiver):
            self.sender.eval();self.receiver.eval()
            evaluate_training(self)
        self.publish_metrics()

    def train(self):
        if (self.run/'COMPLETE.json').exists():return
        if not self.config.get('evaluate_initial_checkpoint'):
            raise ValueError('Corrected protocol requires a frozen checkpoint-zero evaluation')
        if self.update == 0:
            self.evaluate()
        else:
            from .university_evaluation import _baseline_d
            _baseline_d(self.config, self)
        if self.update and self.update%self.config['evaluation_interval']==0:
            self.evaluate()
        while not (self.run/'BUDGET_COMPLETE.json').exists() and self.active_seconds()<self.config['max_active_seconds']:
            if STOP:raise SystemExit(75)
            number=self.update+1;kind=phase_for_update(number);began=time.monotonic()
            self.public_rows=[]
            row=self.receiver_update() if kind=='receiver' else self.sender_update()
            self.update=number
            row.update(update=number,kind=kind,cycle=(number-1)//7+1,duration_seconds=time.monotonic()-began,config_digest=self.config_digest)
            self.metrics.append(row)
            self.checkpoint()
            durations={k:[r['duration_seconds'] for r in self.metrics if r['kind']==k] for k in ['receiver','sender']}
            atomic_json(self.run/'TIMING.json',dict(config_digest=self.config_digest,completed_update=self.update,active_seconds=self.active_seconds(),
                mean_seconds={k:sum(v)/len(v) if v else None for k,v in durations.items()},counts={k:len(v) for k,v in durations.items()}))
            if number in [1,3]:
                atomic_json(self.run/f'FIRST_{kind.upper()}_UPDATE.json',row)
            if STOP:raise SystemExit(75)
            if number%self.config['evaluation_interval']==0:self.evaluate()
        if STOP:raise SystemExit(75)
        atomic_json(self.run/'BUDGET_COMPLETE.json',dict(config_digest=self.config_digest,completed_update=self.update,active_seconds=self.active_seconds()))
        self.evaluate()
        evaluation=self.run/'evaluations'/f'step_{self.update:06d}'/'COMPLETE.json'
        atomic_json(self.run/'COMPLETE.json',dict(config_digest=self.config_digest,completed_update=self.update,checkpoint=str(self.current_checkpoint),
            evaluation_complete_sha256=sha256(evaluation),active_seconds=self.active_seconds(),stop_reason='configured_active_time_budget',
            sender_updates=sum(r['kind']=='sender' for r in self.metrics),receiver_updates=sum(r['kind']=='receiver' for r in self.metrics)))
        self.log_status('complete')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();config=json.loads(Path(args.config).read_text());run=Path(args.run_dir);run.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGUSR1,stop_requested);signal.signal(signal.SIGTERM,stop_requested)
    with (run/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        existing=run/'config.json'
        if existing.exists() and json.loads(existing.read_text())!=config:raise ValueError('Run configuration changed')
        atomic_json(existing,config)
        for path,expected in config.get('source_sha256',{}).items():
            if sha256(path)!=expected:raise ValueError('Source digest mismatch: '+path)
        atomic_json(run/'PROMPTS.json',dict(university=__import__('stego.university_baseline',fromlist=['TASK_PROMPT']).TASK_PROMPT,observer=OBSERVER_PROMPT,piqa=__import__('stego.university_baseline',fromlist=['PIQA_INSTRUCTION']).PIQA_INSTRUCTION))
        try:Trainer(config,run).train()
        except SystemExit as exc:
            if exc.code==75:atomic_json(run/'PREEMPTED.json',dict(config_digest=digest(config),timestamp=time.time(),resume='highest complete checkpoint; no partial update committed'))
            raise
        except Exception as exc:
            atomic_json(run/'FAILURE.json',dict(config_digest=digest(config),error=repr(exc),traceback=traceback.format_exc(),timestamp=time.time()))
            raise


if __name__=='__main__':main()
