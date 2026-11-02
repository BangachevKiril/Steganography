import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import test_qwen_segments as legacy
from test_qwen_segments import read,sha256,atomic_json
from orcd import qwen_cycle_training as cycles
from orcd.sparse_runtime import evaluation_matches

class CycleTests(unittest.TestCase):
    def setUp(self):
        legacy.SegmentTests.setUp(self)
        self.training.capture_rng=lambda:self.clock.now
        self.training.restore_rng=lambda state:None
        self.training.sparse_evaluation_matches=evaluation_matches
        original_init=self.training.Trainer.__init__
        original_save=self.training.Trainer.checkpoint
        original_evaluate=self.training.Trainer.evaluate
        def checkpoint(t):
            original_save(t)
            p=t.current_checkpoint/'COMPLETE.json';value=read(p);value['model_identity']={'sender_sha256':str(t.update),'receiver_sha256':str(t.update)};atomic_json(p,value)
        def initialize(t,*a,**kw):
            original_init(t,*a,**kw);t.config['total_cycles']=200
            t.assert_finite=lambda:None
            class Stream:
                def state_dict(self):return {'cursor':t.stream_cursor}
                def load_state_dict(self,state):t.stream_cursor=state['cursor']
            t.streams={'receiver':Stream()}
            t.sender=t.receiver=SimpleNamespace(named_modules=lambda:[])
        def evaluate(t):
            original_evaluate(t)
            root=t.run/'evaluations'/f'step_{t.update:06d}'
            state=t.run/'evaluation_states'/f'step_{t.update:06d}';state.mkdir(parents=True,exist_ok=True)
            atomic_json(state/'COMPLETE.json',dict(config_digest=t.config_digest,completed_update=t.update,kind='in_memory_evaluation_weights',resumable=False,model_identity={'sender_sha256':str(t.update),'receiver_sha256':str(t.update)}))
            atomic_json(root/'SUMMARY.json',{'n':2000})
            value=read(root/'COMPLETE.json');value.update(source_checkpoint=dict(path=str(state.resolve()),completed_update=t.update,complete_sha256=sha256(state/'COMPLETE.json')),summary_sha256=sha256(root/'SUMMARY.json'))
            atomic_json(root/'COMPLETE.json',value)
        self.training.Trainer.__init__=initialize;self.training.Trainer.checkpoint=checkpoint;self.training.Trainer.evaluate=evaluate
        cycles.install_cycle_training(self.training)

    def trainer(self):return self.training.Trainer(self.run)

    def test_exact_200_cycles_sparse_saves_and_evaluation_cadence(self):
        t=self.trainer();t.train()
        self.assertEqual(t.update,1400);self.assertEqual(t.optimizer_steps,1400)
        self.assertEqual([r['kind'] for r in t.metrics],(['receiver']*2+['sender']*5)*200)
        self.assertEqual(sorted(int(p.name[5:]) for p in (self.run/'checkpoints').glob('step_*')),[0,280,560,840,1120,1400])
        self.assertEqual(sorted(set(t.evaluation_calls)),list(range(0,1401,50)))
        marker=read(self.run/'COMPLETE.json');self.assertEqual(marker['receiver_updates'],400);self.assertEqual(marker['sender_updates'],1000)
        before=(self.run/'COMPLETE.json').read_bytes();self.trainer().train();self.assertEqual(before,(self.run/'COMPLETE.json').read_bytes())

    def test_graceful_resume_after_midcycle_update(self):
        t=self.trainer();original=t.sender_update
        def sender():
            value=original()
            if t.update==23:self.training.STOP=True
            return value
        t.sender_update=sender
        with self.assertRaises(SystemExit):t.train()
        self.assertEqual(t.update,24);self.assertTrue((self.run/'checkpoints/step_000024/state.pt').exists())
        self.training.STOP=False;t=self.trainer();t.train();self.assertEqual(t.optimizer_steps,1400)

    def test_interrupted_collection_restores_stream_and_saves_prior_updates(self):
        t=self.trainer();original=t.receiver_update
        def receiver():
            if t.update==8:
                t.stream_cursor+=10
                raise SystemExit(75)
            return original()
        t.receiver_update=receiver
        with self.assertRaises(SystemExit):t.train()
        self.assertEqual(t.update,8);self.assertEqual(t.stream_cursor,8)
        self.assertEqual(read(t.current_checkpoint/'state.pt')['stream_cursor'],8)
        t=self.trainer();t.train();self.assertEqual(t.optimizer_steps,1400)

    def test_final_evaluation_preemption_resumes_without_extra_updates(self):
        t=self.trainer();t.preempt_evaluation_update=1400
        with self.assertRaises(SystemExit):t.train()
        self.assertEqual(t.update,1400);self.assertFalse((self.run/'COMPLETE.json').exists())
        t=self.trainer();t.train();self.assertEqual(t.optimizer_steps,1400)

    def test_nonfinite_failure_never_publishes_completion(self):
        t=self.trainer()
        def fail():raise FloatingPointError('nonfinite')
        t.sender_update=fail
        with self.assertRaises(FloatingPointError):t.train()
        self.assertFalse((self.run/'COMPLETE.json').exists());self.assertEqual(t.update,2)

    def test_tampered_restart_hash_fails_before_updates(self):
        t=self.trainer();(t.current_checkpoint/'state.pt').write_text('tampered')
        with self.assertRaisesRegex(ValueError,'hash'):t.train()
        self.assertEqual(t.optimizer_steps,0)

    def test_completed_marker_counts_are_verified(self):
        t=self.trainer();t.train();p=self.run/'COMPLETE.json';value=read(p);value['completed_cycles']=199;atomic_json(p,value)
        with self.assertRaisesRegex(ValueError,'provenance'):self.trainer().train()
