import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from orcd import sparse_runtime as rt
from stego.train import atomic_json
from stego.university_baseline import sha256

class IdentityTests(unittest.TestCase):
 def setUp(self):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
  self.run=Path(tmp.name);self.t=SimpleNamespace(run=self.run,update=50,config_digest='test')
  self.hashes={'sender_sha256':'a'*64,'receiver_sha256':'b'*64}
  self.p=patch.object(rt,'model_identity',return_value=self.hashes);self.p.start();self.addCleanup(self.p.stop)
 def test_evaluation_hashes_without_saving_state(self):
  identity=rt.evaluation_identity(self.t)
  self.assertFalse((self.run/'checkpoints').exists())
  self.assertEqual(identity,rt.evaluation_identity(self.t))
  self.assertEqual(identity['complete_sha256'],sha256(Path(identity['path'])/'COMPLETE.json'))
  with patch.object(rt,'model_identity',return_value={'sender_sha256':'c'*64}):
   with self.assertRaisesRegex(ValueError,'fingerprint'):rt.evaluation_identity(self.t)
 def test_segment_requires_saved_and_evaluated_weights_match(self):
  identity=rt.evaluation_identity(self.t)
  saved=self.run/'checkpoints/step_000050';saved.mkdir(parents=True)
  atomic_json(saved/'COMPLETE.json',{'model_identity':self.hashes})
  checkpoint={'path':str(saved),'completed_update':50}
  self.assertTrue(rt.evaluation_matches(self.t,{'source_checkpoint':identity},checkpoint))
  atomic_json(saved/'COMPLETE.json',{'model_identity':{'sender_sha256':'x'}})
  self.assertFalse(rt.evaluation_matches(self.t,{'source_checkpoint':identity},checkpoint))
 def test_rollback_archives_lost_evaluations_and_removes_partial_state_only(self):
  self.t.update=280
  for update in [0,250,280,300,350]:
   for category in ['evaluations','evaluation_states']:
    atomic_json(self.run/category/f'step_{update:06d}'/'COMPLETE.json',{'update':update})
   atomic_json(self.run/'rollouts'/f'update_{update:06d}.json',{'update':update})
  partial=self.run/'checkpoints/.step_000560.123.tmp';partial.mkdir(parents=True);(partial/'state.pt').write_text('partial')
  atomic_json(self.run/'checkpoints/step_000280/COMPLETE.json',{'update':280})
  (self.run/'checkpoints/step_000280/state.pt').write_text('safe')
  rt.archive_after_rollback(self.t)
  self.assertTrue((self.run/'evaluations/step_000280/COMPLETE.json').exists())
  self.assertFalse((self.run/'evaluations/step_000300').exists())
  self.assertEqual(len(list((self.run/'rolled_back_attempts').glob('*/evaluations/step_000300/COMPLETE.json'))),1)
  self.assertFalse(partial.exists())
  self.assertEqual((self.run/'checkpoints/step_000280/state.pt').read_text(),'safe')

if __name__=='__main__':unittest.main()
