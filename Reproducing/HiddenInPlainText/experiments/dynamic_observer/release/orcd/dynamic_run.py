#!/usr/bin/env python3
"""Validation-gated dynamic observer run; atomic joint checkpoints and equal cycles."""
import argparse,fcntl,gc,json,os,signal,sys,time,traceback
from pathlib import Path
SOURCE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SOURCE))
from stego import university_training as base
from stego.dynamic_training import Trainer,setup
from orcd.university_l40s_memory_training import layout_preserving_cpu_offload,cuda_toy_parity


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--validate',action='store_true');p.add_argument('--validation',type=Path,required=True)
    args=p.parse_args();c=json.loads(args.config.read_text());digest=base.digest(c);run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    for n,h in c['source_sha256'].items():
        if base.sha256(SOURCE/n)!=h:raise ValueError('Source hash mismatch: '+n)
    for n,h in c['input_sha256'].items():
        if base.sha256(n)!=h:raise ValueError('Input hash mismatch: '+n)
    if not args.validate:
        v=json.loads(args.validation.read_text())
        if v.get('status')!='passed' or v.get('source_sha256')!=c['source_sha256'] or v.get('gpu_type')!=c['target_gpu'] or v.get('scientific_spec')!=c['dynamic']['shared_spec']:
            raise ValueError('Matching full-cycle validation required')
    trainer=None
    def stop(signum,frame):
        base.stop_requested(signum,frame)
        if trainer is not None and trainer.observer.process.poll() is None:
            try:trainer.observer.process.send_signal(signal.SIGUSR1)
            except ProcessLookupError:pass
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    with (run/'LOCK').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if (run/'config.json').exists() and json.loads((run/'config.json').read_text())!=c:raise ValueError('Run configuration changed')
            base.atomic_json(run/'config.json',c)
            import torch
            parity=cuda_toy_parity(base)
            setup(c,run)
            trainer=Trainer(c,run)
            if not args.validate and trainer.observer.ready['initial_adapter_sha256']!=v['observer']['initial_adapter_sha256']:
                raise ValueError('Initial observer adapters differ from validation')
            base.atomic_json(run/'RUNTIME.json',dict(config_digest=digest,gpu=torch.cuda.get_device_name(0),cuda_toy_parity=parity,observer=trainer.observer.ready,restored_update=trainer.update,restored_checkpoint=str(trainer.current_checkpoint),slurm_job_id=os.environ.get('SLURM_JOB_ID')))
            original=trainer.ppo.step
            def offload(*a,**kw):
                trainer.observer._request('device',device='cpu')
                gc.collect();torch.cuda.empty_cache();stats={}
                try:
                    with layout_preserving_cpu_offload(stats):return original(*a,**kw)
                finally:
                    gc.collect();torch.cuda.empty_cache();trainer.observer._request('device',device='cuda:0')
                    base.atomic_json(run/'PPO_MEMORY.json',dict(attempted_update=trainer.update+1,gpu_peak_bytes=torch.cuda.max_memory_allocated(),offload=stats,config_digest=digest))
            trainer.ppo.step=offload
            if args.validate:
                if trainer.update==0:trainer.evaluate(limit=64)
                while trainer.update<trainer.cycle_width:
                    if base.STOP:raise SystemExit(75)
                    trainer.step()
                if trainer.update!=trainer.cycle_width:raise ValueError('Unexpected validation checkpoint')
                trainer.evaluate(limit=64)
                if not any(r.get('adapter_changed') for r in trainer.metrics if r['kind']=='observer'):raise ValueError('No LoRA update observed')
                checkpoint=trainer.current_checkpoint
                before=trainer.observer._request('fingerprint')
                texts=['A candidate with relevant experience.','An applicant who demonstrates practical skills.']
                scores=trainer.observer.score(texts)
                # Restore the complete joint checkpoint and require unchanged public predictions/adapter state.
                trainer.restore(checkpoint)
                after=trainer.observer._request('fingerprint')
                if before['adapter_sha256']!=after['adapter_sha256'] or before['observer_updates']!=after['observer_updates'] or scores!=trainer.observer.score(texts):raise ValueError('Observer restore parity failed')
                trainer.assert_finite()
                base.atomic_json(args.validation,dict(status='passed',gpu_type=c['target_gpu'],source_sha256=c['source_sha256'],scientific_spec=c['dynamic']['shared_spec'],validation_config_digest=digest,full_cycle=trainer.phases,receiver_batch=256,sender_batch=192,observer_batch=256,observer_restoration_verified=True,checkpoint_sha256=base.sha256(checkpoint/'COMPLETE.json'),observer=trainer.observer.ready,gpu=torch.cuda.get_device_name(0),peak_allocated_bytes=torch.cuda.max_memory_allocated(),evaluation_scope='64 university and 4 PIQA at initial and post-cycle checkpoints; production separately requires full 2000/100',slurm_job_id=os.environ.get('SLURM_JOB_ID'),completed_at=time.time()))
            else:trainer.train()
        except SystemExit as exc:
            if exc.code==75:base.atomic_json(run/'PREEMPTED.json',dict(config_digest=digest,at=time.time()))
            raise
        except BaseException as exc:
            if base.STOP:
                base.atomic_json(run/'PREEMPTED.json',dict(config_digest=digest,at=time.time(),interrupted_operation=type(exc).__name__))
                raise SystemExit(75)
            base.atomic_json(run/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),config_digest=digest,slurm_job_id=os.environ.get('SLURM_JOB_ID'),at=time.time()))
            raise
        finally:
            if trainer is not None:trainer.observer.close()

if __name__=='__main__':main()
