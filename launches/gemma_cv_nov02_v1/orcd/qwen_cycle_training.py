"""Exactly 200 alternating cycles; sparse, hash-bound recovery under caller LOCK."""
import copy
import json
import math
from pathlib import Path
import time

TOTAL_CYCLES=200
TOTAL_UPDATES=7*TOTAL_CYCLES


def install_cycle_training(training):
    def read(path):
        return json.loads(Path(path).read_text())

    def identity(self, verify=False):
        path=Path(self.current_checkpoint)
        marker=read(path/'COMPLETE.json')
        if (marker['config_digest']!=self.config_digest
                or path.resolve()!=(Path(self.run)/'checkpoints'/f"step_{marker['completed_update']:06d}").resolve()
                or not 0<=marker['completed_update']<=TOTAL_UPDATES
                or not math.isfinite(marker['elapsed']) or marker['elapsed']<0):
            raise ValueError('Invalid checkpoint identity')
        if verify and training.sha256(path/'state.pt')!=marker['state_sha256']:
            raise ValueError('Checkpoint hash mismatch')
        return dict(path=str(path.resolve()),completed_update=marker['completed_update'],
                    state_sha256=marker['state_sha256'],complete_sha256=training.sha256(path/'COMPLETE.json'),elapsed=marker['elapsed'])

    def ensure_checkpoint(self):
        if identity(self)['completed_update']!=self.update:self.checkpoint()

    def persist_clock(self):
        value=dict(config_digest=self.config_digest,checkpoint=identity(self),active_seconds=self.active_seconds(),recorded_at=time.time())
        if not math.isfinite(value['active_seconds']) or value['active_seconds']<value['checkpoint']['elapsed']:
            raise ValueError('Invalid active clock')
        training.atomic_json(Path(self.run)/'CYCLE_ACTIVE_CLOCK.json',value)

    def restore_clock(self):
        path=Path(self.run)/'CYCLE_ACTIVE_CLOCK.json'
        if not path.exists():return
        clock=read(path);saved=identity(self);recorded=clock['checkpoint'];seconds=clock['active_seconds']
        if clock['config_digest']!=self.config_digest or not math.isfinite(seconds) or seconds<0 or recorded['completed_update']>saved['completed_update']:
            raise ValueError('Active clock provenance mismatch')
        if recorded['completed_update']==saved['completed_update']:
            if recorded!=saved or seconds<saved['elapsed']:raise ValueError('Clock checkpoint mismatch')
            self.elapsed=max(self.elapsed,seconds)
        elif seconds>saved['elapsed']:raise ValueError('New checkpoint lost previously observed active time')

    def evaluate(self):
        try:self.evaluate()
        except SystemExit as exc:
            if exc.code==75:ensure_checkpoint(self)
            raise
        finally:persist_clock(self)

    def publish_live(self):
        directory=Path(self.run)/'rollouts';directory.mkdir(exist_ok=True)
        training.atomic_json(directory/f'update_{self.update:06d}.json',dict(
            config_digest=self.config_digest,update=self.update,kind=training.phase_for_update(self.update),rows=self.public_rows))
        temporary=Path(self.run)/'metrics.jsonl.tmp'
        temporary.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in self.metrics))
        temporary.replace(Path(self.run)/'metrics.jsonl')

    def collection_snapshot(self):
        return dict(rng=training.capture_rng(),streams={k:copy.deepcopy(v.state_dict()) for k,v in self.streams.items()},
                    modes={role:{name:m.training for name,m in getattr(self,role).named_modules()} for role in ('sender','receiver')},
                    public_rows=self.public_rows)

    def rollback_collection(self,snapshot):
        for key,value in snapshot['streams'].items():self.streams[key].load_state_dict(value)
        for role,value in snapshot['modes'].items():
            for name,module in getattr(self,role).named_modules():module.training=value[name]
        training.restore_rng(snapshot['rng']);self.public_rows=snapshot['public_rows']

    def verify_complete(self):
        run=Path(self.run);marker=read(run/'COMPLETE.json');saved=identity(self,True)
        evaluation=run/'evaluations'/f'step_{TOTAL_UPDATES:06d}'/'COMPLETE.json'
        evaluated=read(evaluation)
        if (marker.get('config_digest')!=self.config_digest or marker.get('completed_update')!=TOTAL_UPDATES
                or marker.get('completed_cycles')!=TOTAL_CYCLES or marker.get('receiver_updates')!=400
                or marker.get('sender_updates')!=1000 or marker.get('checkpoint')!=saved
                or marker.get('evaluation_complete_sha256')!=training.sha256(evaluation)
                or evaluated.get('config_digest')!=self.config_digest
                or not training.sparse_evaluation_matches(self,evaluated,saved)
                or evaluated.get('summary_sha256')!=training.sha256(evaluation.parent/'SUMMARY.json')):
            raise ValueError('Final cycle completion provenance mismatch')

    def train(self):
        run=Path(self.run)
        if self.config.get('total_cycles')!=TOTAL_CYCLES or self.config.get('evaluate_initial_checkpoint') is not True:
            raise ValueError('Expected exactly 200 cycles and baseline evaluation')
        if identity(self,True)['completed_update']!=self.update:raise ValueError('Uncommitted initialization')
        if len(self.metrics)!=self.update or any(row['update']!=i or row['kind']!=training.phase_for_update(i) for i,row in enumerate(self.metrics,1)):
            raise ValueError('Restored update history/phase mismatch')
        if (run/'COMPLETE.json').exists():
            verify_complete(self);self.log_status('complete');return
        restore_clock(self)
        try:
            if self.update==0:evaluate(self)
            else:
                from stego.university_evaluation import _baseline_d
                _baseline_d(self.config,self)
            if self.update and self.update%self.config['evaluation_interval']==0:evaluate(self)
            while self.update<TOTAL_UPDATES:
                if training.STOP:
                    ensure_checkpoint(self);raise SystemExit(75)
                number=self.update+1;kind=training.phase_for_update(number);began=time.monotonic()
                snapshot=collection_snapshot(self) if kind=='receiver' else None
                self.public_rows=[]
                try:row=self.receiver_update() if kind=='receiver' else self.sender_update()
                except SystemExit as exc:
                    if exc.code==75 and snapshot is not None:
                        rollback_collection(self,snapshot);ensure_checkpoint(self)
                    raise
                self.assert_finite();self.update=number
                row.update(update=number,kind=kind,cycle=(number-1)//7+1,duration_seconds=time.monotonic()-began,config_digest=self.config_digest)
                self.metrics.append(row)
                if number%280==0 or training.STOP or number==TOTAL_UPDATES:ensure_checkpoint(self)
                else:publish_live(self)
                persist_clock(self)
                durations={k:[r['duration_seconds'] for r in self.metrics if r['kind']==k] for k in ('sender','receiver')}
                training.atomic_json(run/'TIMING.json',dict(config_digest=self.config_digest,completed_update=self.update,
                    completed_cycles=self.update//7,target_cycles=TOTAL_CYCLES,active_seconds=self.active_seconds(),
                    counts={k:len(v) for k,v in durations.items()},mean_seconds={k:sum(v)/len(v) if v else None for k,v in durations.items()}))
                if number in (1,3):training.atomic_json(run/f'FIRST_{kind.upper()}_UPDATE.json',row)
                if training.STOP:raise SystemExit(75)
                if number%self.config['evaluation_interval']==0:evaluate(self)
            ensure_checkpoint(self)
            if training.STOP:raise SystemExit(75)
            evaluate(self)
            evaluation=run/'evaluations'/f'step_{TOTAL_UPDATES:06d}'/'COMPLETE.json'
            counts={k:sum(r['kind']==k for r in self.metrics) for k in ('receiver','sender')}
            if counts!={'receiver':400,'sender':1000}:raise ValueError('Final cycle counts differ')
            saved=identity(self,True)
            if not training.sparse_evaluation_matches(self,read(evaluation),saved):raise ValueError('Final evaluated weights differ from restart')
            payload=dict(config_digest=self.config_digest,completed_update=self.update,completed_cycles=TOTAL_CYCLES,
                         receiver_updates=counts['receiver'],sender_updates=counts['sender'],checkpoint=saved,
                         evaluation_complete_sha256=training.sha256(evaluation),active_seconds=self.active_seconds(),
                         stop_reason='configured_200_cycles',completed_at=time.time())
            training.atomic_json(run/'BUDGET_COMPLETE.json',payload)
            training.atomic_json(run/'COMPLETE.json',payload)
            verify_complete(self);self.log_status('complete')
        finally:persist_clock(self)

    training.Trainer.train=train
    return train
