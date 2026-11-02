"""Sparse restart persistence with independently identified in-memory evaluations.

Public evaluation identities are weight fingerprints, not resumable checkpoints.
After rollback, results newer than the restored state move out of the live run.
"""
import json
import os
from pathlib import Path
import time


def model_identity(trainer):
    from stego.university_baseline import tensor_digest
    return {'sender_sha256': tensor_digest(trainer.sender),
            'receiver_sha256': tensor_digest(trainer.receiver)}


def evaluation_identity(trainer):
    from stego.train import atomic_json
    from stego.university_baseline import sha256
    directory=Path(trainer.run)/'evaluation_states'/f'step_{trainer.update:06d}'
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'COMPLETE.json'
    value=dict(config_digest=trainer.config_digest,completed_update=trainer.update,
               kind='in_memory_evaluation_weights',resumable=False,
               model_identity=model_identity(trainer))
    if marker.exists():
        if json.loads(marker.read_text())!=value:
            raise ValueError('Evaluation weight fingerprint changed at the same update')
    else:
        atomic_json(marker,value)
    return dict(path=str(directory.resolve()),completed_update=trainer.update,
                complete_sha256=sha256(marker))


def evaluation_matches(trainer, evaluation, checkpoint):
    from stego.university_baseline import sha256
    run=Path(trainer.run)
    step=checkpoint['completed_update']
    source=run/'evaluation_states'/f'step_{step:06d}'
    marker=source/'COMPLETE.json'
    value=json.loads(marker.read_text())
    saved=json.loads((Path(checkpoint['path'])/'COMPLETE.json').read_text())
    expected=dict(path=str(source.resolve()),completed_update=step,complete_sha256=sha256(marker))
    return (evaluation.get('source_checkpoint')==expected
            and value.get('config_digest')==trainer.config_digest
            and value.get('completed_update')==step
            and value.get('kind')=='in_memory_evaluation_weights'
            and value.get('model_identity')==saved.get('model_identity')
            and isinstance(value.get('model_identity'),dict))


def archive_after_rollback(trainer):
    """Called only while holding the run LOCK, after hash-verified restore."""
    from stego.train import atomic_json
    run=Path(trainer.run)
    discarded=[]
    for category,pattern,prefix in [('evaluations','step_*','step_'),
                                    ('evaluation_states','step_*','step_'),
                                    ('rollouts','update_*.json','update_')]:
        for path in sorted((run/category).glob(pattern)):
            suffix=path.stem[len(prefix):] if path.is_file() else path.name[len(prefix):]
            if suffix.isdigit() and int(suffix)>trainer.update:
                discarded.append(path)
    for name in ['FIRST_RECEIVER_UPDATE.json','FIRST_SENDER_UPDATE.json','FULL_CYCLE_SMOKE.json']:
        path=run/name
        if path.exists():
            value=json.loads(path.read_text())
            if value.get('completed_update',value.get('update',0))>trainer.update:
                discarded.append(path)
    if discarded:
        target=run/'rolled_back_attempts'/('%d_%s_%d'%(time.time_ns(),os.environ.get('SLURM_JOB_ID','local'),trainer.update))
        target.mkdir(parents=True)
        moved=[]
        for path in discarded:
            relative=path.relative_to(run);dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            path.rename(dest);moved.append(str(relative))
        atomic_json(target/'ROLLBACK.json',dict(restored_update=trainer.update,config_digest=trainer.config_digest,archived_public_artifacts=moved))
        atomic_json(run/'LAST_ROLLBACK.json',dict(restored_update=trainer.update,config_digest=trainer.config_digest,archive=str(target),archived_public_artifact_count=len(moved)))
    # Incomplete writes are never recoverable, and only this new run is in scope.
    import shutil
    removed=[]
    for path in (run/'checkpoints').glob('.step_*.tmp'):
        if path.is_dir() and not path.is_symlink():
            removed.append(str(path));shutil.rmtree(path)
    if removed:atomic_json(run/'INCOMPLETE_CHECKPOINT_CLEANUP.json',dict(restored_update=trainer.update,removed=removed))


def install_sparse_runtime(training, validation=False):
    original_init=training.Trainer.__init__
    original_restore=training.Trainer.restore
    def initialize(self,config,run,*args,**kwargs):
        path=Path(run)
        if (path/'INITIALIZATION.json').exists() and not any((p.parent/'state.pt').is_file() for p in (path/'checkpoints').glob('step_*/COMPLETE.json')):
            raise RuntimeError('Previously initialized run has no complete restart state; refusing fresh initialization')
        original_init(self,config,run,*args,**kwargs)
        training.atomic_json(self.run/'CHECKPOINT_POLICY.json',dict(config_digest=self.config_digest,**config['sparse_checkpoint_policy']))
    def restore(self,path):
        original_restore(self,path)
        archive_after_rollback(self)
    training.Trainer.__init__=initialize
    training.Trainer.restore=restore
    training.sparse_evaluation_matches=evaluation_matches
