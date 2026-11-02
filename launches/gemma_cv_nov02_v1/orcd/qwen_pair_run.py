#!/usr/bin/env python3
"""Isolated Qwen3 sender/receiver validation and fixed-budget training entrypoint."""
import argparse
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--segment',type=int,choices=[1,2,3],default=1)
    parser.add_argument('--validation',required=True,type=Path)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    if args.validate and (os.environ.get('SPARSE_CPU_VALIDATION_PASSED')!='23' or os.environ.get('PROBABILITY_CPU_VALIDATION_PASSED')!='passed' or os.environ.get('ABLATION_CPU_VALIDATION_PASSED')!='passed'):
        raise ValueError('Mandatory CPU numerical and sparse tests have not passed')
    source=Path(__file__).resolve().parents[1]
    os.chdir(source);sys.path.insert(0,str(source))
    from stego import university_training as training
    from stego import university_evaluation as evaluation
    from orcd.qwen_observer_client import QwenObserverClient
    from orcd.qwen_cycle_training import install_cycle_training
    from orcd.university_l40s_memory_training import layout_preserving_cpu_offload,cuda_toy_parity
    config=json.loads(args.config.read_text());digest=training.digest(config)
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    for rel,expected in config['source_sha256'].items():
        if training.sha256(source/rel)!=expected:raise ValueError('Source mismatch: '+rel)
    for path,expected in config['input_sha256'].items():
        if training.sha256(path)!=expected:raise ValueError('Input mismatch: '+path)
    for name,expected in config['pair_model_change']['model_files_sha256'].items():
        if training.sha256(Path(config['assets']['sender']['path'])/name)!=expected:raise ValueError('Model mismatch: '+name)
    versions={k:importlib.metadata.version(k) for k in ['torch','transformers','trl','peft','accelerate','datasets','tokenizers','huggingface-hub']}
    for k,v in config['pair_model_change']['runtime_versions'].items():
        if versions[k]!=v:raise ValueError('Training runtime mismatch: '+k)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or config['target_gpu'] not in torch.cuda.get_device_name(0) or torch.cuda.get_device_properties(0).total_memory<44*1024**3:
        raise RuntimeError('One L40S >=44GiB required')
    full_receiver=config.get('receiver_parameter_training','lora')=='full'
    if not args.validate:
        validation=json.loads(args.validation.read_text())
        # Only the matched training seed and its derived run name may differ from the validated condition.
        import copy
        reference=copy.deepcopy(config)
        seed=reference['seed']
        if seed not in range(42,47) or not reference['name'].endswith(f'_s{seed}'):
            raise ValueError('Unexpected matched seed or run identity')
        reference['name']=reference['name'].rsplit('_s',1)[0]+'_s42'
        reference['seed']=reference['ablation']['seed']=42
        reference['protocol']['initialization'].update(global_seed=42,author_ppo_seed=42)
        reference['protocol']['training']['author_ppo_seed']=42
        if ((validation.get('receiver_full_updated') is not True or validation.get('fp32_master_restart_verified') is not True) if full_receiver else (validation.get('receiver_lora_updated') is not True or validation.get('receiver_base_unchanged') is not True)):
            raise ValueError('Receiver validation mode mismatch')
        if (validation.get('receiver_parameter_training')!=config['receiver_parameter_training']
                or validation.get('sender_trainable_updated') is not True or validation.get('cpu_tests_passed')!=23
                or validation.get('probability_tests_passed') is not True or validation.get('ablation_tests_passed') is not True
                or validation.get('receiver',{}).get('receiver_training',{}).get('optimizer_sample_count')!=256
                or validation.get('sender',{}).get('n')!=192
                or validation.get('sender',{}).get('objective')!=config['objective']
                or not validation.get('evaluation_weight_identity')):
            raise ValueError('Full validation invariants were not established')
        if validation.get('status')!='passed' or validation.get('config_digest')!=training.digest(reference) or validation.get('source_sha256')!=config['source_sha256']:
            raise ValueError('Matching full-batch L40S validation required')
    signal.signal(signal.SIGUSR1,training.stop_requested);signal.signal(signal.SIGTERM,training.stop_requested)
    observers=[]
    def observer(c):
        child=QwenObserverClient(c,run=run);observers.append(child)
        training.atomic_json(run/'QWEN_OBSERVER.json',dict(child.ready,config_digest=digest))
        return child
    training.Observer=observer
    original_init=training.Trainer.__init__
    original_checkpoint=training.Trainer.checkpoint
    def initialized(self,*a,**kw):
        parity=cuda_toy_parity(training)
        original_init(self,*a,**kw)
        step=self.ppo.step
        def offloaded_step(*sa,**sk):
            # Receiver has no role during PPO, and its optimizer masters already live on CPU.
            self.receiver.to('cpu');gc.collect();torch.cuda.empty_cache()
            stats={};began=time.monotonic()
            try:
                with layout_preserving_cpu_offload(stats):return step(*sa,**sk)
            finally:
                torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
                self.receiver.to('cuda:0')
                training.atomic_json(run/'PPO_MEMORY.json',dict(config_digest=digest,attempted_update=self.update+1,duration_seconds=time.monotonic()-began,transfers=stats,peak_allocated_bytes=torch.cuda.max_memory_allocated(),note='PPO return is not a checkpoint commit'))
        self.ppo.step=offloaded_step
        training.atomic_json(run/'RUNTIME.json',dict(config_digest=digest,versions=versions,gpu=torch.cuda.get_device_name(0),cuda_toy_parity=parity,restored_update=self.update,restored_elapsed=self.elapsed,receiver_full_training=full_receiver,receiver_lora_training=not full_receiver,receiver_optimizer='FP32 CPU AdamW lr1.41e-5; persistent FP32 masters for full SFT',receiver_base_dtype='bfloat16',activation_checkpointing='receiver nonreentrant; preserves RNG',sender_activation_offload='full-storage layout-preserving PPO only',slurm_job_id=os.environ.get('SLURM_JOB_ID')))
    def checkpoint(self):
        original_checkpoint(self)
        if self.update>=7 and (run/'evaluations/step_000000/COMPLETE.json').exists():
            if [r['kind'] for r in self.metrics[:7]]!=['receiver']*2+['sender']*5:raise ValueError('First cycle mismatch')
            training.atomic_json(run/'FULL_CYCLE_SMOKE.json',dict(status='passed',config_digest=digest,completed_update=self.update,first_cycle_updates=7,checkpoint_sha256=training.sha256(self.current_checkpoint/'COMPLETE.json'),evaluation0_sha256=training.sha256(run/'evaluations/step_000000/COMPLETE.json')))
    training.Trainer.__init__=initialized;training.Trainer.checkpoint=checkpoint
    from orcd.sparse_runtime import install_sparse_runtime
    install_sparse_runtime(training, args.validate)
    install_cycle_training(training)
    try:
        with (run/'LOCK').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=config:raise ValueError('Run config changed')
            training.atomic_json(run/'config.json',config)
            if args.validate and (run/'checkpoints').exists():
                raise ValueError('Validation directory already initialized; inspect rather than silently retry')
            trainer=training.Trainer(config,run)
            if args.validate:
                # Real 256-example receiver update and 192-example PPO update, including the observer.
                if full_receiver:
                    controls=['An experienced software engineer.','A graduate from Good University.']
                    before_scores=trainer.observer.score(controls)
                    trainer.observer.move('cpu');trainer.observer.move('cuda:0')
                    if before_scores!=trainer.observer.score(controls):
                        raise ValueError('Observer scores changed after CPU/CUDA roundtrip')
                frozen_before={n:p._version for n,p in trainer.receiver.named_parameters() if not p.requires_grad}
                adapter_before={n:p.detach().cpu().clone() for n,p in trainer.receiver.named_parameters() if p.requires_grad}
                sender_before={n:p.detach().cpu().clone() for n,p in trainer.sender.named_parameters() if p.requires_grad}
                receiver=trainer.receiver_update();sender=trainer.sender_update();trainer.assert_finite()
                if not any(not torch.equal(sender_before[n],p.detach().cpu()) for n,p in trainer.sender.named_parameters() if p.requires_grad):raise ValueError('Sender trainable parameters did not update')
                if frozen_before!={n:p._version for n,p in trainer.receiver.named_parameters() if not p.requires_grad}:raise ValueError('Frozen receiver parameters mutated')
                if not any(not torch.equal(adapter_before[n],p.detach().cpu()) for n,p in trainer.receiver.named_parameters() if p.requires_grad):raise ValueError('Receiver LoRA did not update')
                full_updated=False;master_restart=False;master_restart_hash=None;update_audit=None
                if full_receiver:
                    if frozen_before or any('lora_' in n for n in adapter_before):
                        raise ValueError('Full receiver contains frozen parameters or adapters')
                    from orcd.ablation_validation import receiver_update_audit
                    update_audit=receiver_update_audit(trainer.receiver,adapter_before)
                    training.atomic_json(run/'RECEIVER_UPDATE_AUDIT.json',dict(update_audit,config_digest=digest))
                    if not update_audit['passed']:
                        raise ValueError('Full receiver embedding/backbone/head did not update')
                    full_updated=True
                    del adapter_before
                    from orcd.ablation_validation import verify_receiver_restart
                    master_restart_hash=verify_receiver_restart(trainer,run/'validation_receiver_restart.pt')
                    master_restart=True
                from orcd.sparse_runtime import evaluation_identity
                identity=evaluation_identity(trainer)
                if (run/'checkpoints/step_000001').exists():raise ValueError('Lightweight evaluation saved a restart state')
                if receiver['receiver_training']['optimizer_sample_count']!=256 or sender['n']!=192:raise ValueError('Batch population mismatch')
                training.atomic_json(args.validation,dict(status='passed',config_digest=digest,source_sha256=config['source_sha256'],receiver=receiver,sender=sender,versions=versions,gpu=torch.cuda.get_device_name(0),scope='full receiver256 and sender192 PPO plus lightweight weight identity; not production evaluation0/full-cycle',evaluation_weight_identity=identity,receiver_parameter_training=config['receiver_parameter_training'],receiver_full_updated=full_updated,receiver_update_audit=update_audit,fp32_master_restart_verified=master_restart,validation_receiver_restart_sha256=master_restart_hash,receiver_lora_updated=not full_receiver,receiver_base_unchanged=not full_receiver,cpu_tests_passed=23,probability_tests_passed=True,ablation_tests_passed=True,sender_trainable_updated=True,slurm_job_id=os.environ.get('SLURM_JOB_ID'),completed_at=time.time()))
                # Only validation's disposable, never-trained checkpoint-zero tensor is removed.
                (run/'checkpoints/step_000000/state.pt').unlink()
            else:trainer.train()
    except SystemExit as e:
        if e.code==75:training.atomic_json(run/'PREEMPTED.json',dict(config_digest=digest,at=time.time()))
        raise
    except BaseException as e:
        training.atomic_json(run/'FAILURE.json',dict(config_digest=digest,error_type=type(e).__name__,failure_category=('oom' if 'out of memory' in str(e).lower() else 'nonfinite' if isinstance(e,FloatingPointError) else 'provenance_or_runtime'),slurm_job_id=os.environ.get('SLURM_JOB_ID'),at=time.time()))
        raise
    finally:
        for child in observers:child.close()


if __name__=='__main__':main()
