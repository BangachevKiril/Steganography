import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent
sys.path.insert(0,str(SOURCE))
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sparse=load('sparse_test_segments',SOURCE/'orcd/qwen_segment_training.py')
legacy=load('legacy_segment_tests',SOURCE/'tests/test_qwen_segments.py')
legacy.segments=sparse

class SparseTests(legacy.SegmentTests):
 def prepare(self):
  t=self.trainer(1);t.update_cost=100.;t.evaluation_cost=5.
  self.saved=[];original=t.checkpoint
  def save():
   self.saved.append(t.update);return original()
  t.checkpoint=save
  return t
 def test_evaluations_do_not_save_restart_state(self):
  t=self.prepare();t.train()
  self.assertEqual(self.saved,[279])
  self.assertEqual(t.optimizer_steps,279)
  self.assertEqual(len(t.metrics),279)
 def test_shutdown_saves_dirty_update_before_exit(self):
  t=self.prepare();original=t.sender_update
  def sender():
   row=original()
   if t.update==23:self.training.STOP=True
   return row
  t.sender_update=sender
  with self.assertRaises(SystemExit):t.train()
  self.assertEqual(self.saved,[24])
  resumed=self.training.Trainer(self.run)
  self.assertEqual(resumed.update,24);self.assertEqual(resumed.optimizer_steps,24)
 def test_40_full_cycles_is_280_updates(self):
  t=self.prepare();t.update_cost=90.;t.train()
  self.assertIn(280,self.saved)
  self.assertNotIn(281,self.saved)
 def prepare_collection(self,t):
  class Stream:
   def state_dict(self):return {'cursor':t.stream_cursor}
   def load_state_dict(self,s):t.stream_cursor=s['cursor']
  t.streams={'receiver':Stream()}
  t.sender=t.receiver=SimpleNamespace(named_modules=lambda:[])
  self.training.capture_rng=lambda:self.clock.now
  self.training.restore_rng=lambda state:None
 def test_partial_collection_keeps_prior_dirty_completed_updates(self):
  t=self.prepare();self.prepare_collection(t)
  original=t.receiver_update
  def receiver():
   if t.update==8:
    self.clock.now=27899.;t.collection_batches=10
   return original()
  t.receiver_update=receiver;t.train()
  self.assertEqual(t.update,8);self.assertEqual(t.optimizer_steps,8)
  self.assertEqual(t.stream_cursor,8);self.assertEqual(self.saved,[8])
 def test_stop_during_collection_saves_prior_dirty_updates(self):
  t=self.prepare();self.prepare_collection(t)
  original=t.receiver_update
  def receiver():
   if t.update==8:
    t.stream_cursor+=3
    raise SystemExit(75)
   return original()
  t.receiver_update=receiver
  with self.assertRaises(SystemExit):t.train()
  self.assertEqual(t.update,8);self.assertEqual(t.optimizer_steps,8)
  self.assertEqual(t.stream_cursor,8);self.assertEqual(self.saved,[8])

if __name__=='__main__':unittest.main()
