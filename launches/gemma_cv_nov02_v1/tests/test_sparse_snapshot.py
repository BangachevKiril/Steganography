import json,tempfile,unittest
from pathlib import Path
from orcd.monitoring.snapshot_qwen_university import snapshot,digest,sha256_bytes
from stego.train import atomic_json

class SnapshotTests(unittest.TestCase):
 def test_live_evaluation_is_valid_but_not_restart_recoverable(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp).resolve();config={'test':'sparse'};d=digest(config)
   atomic_json(root/'config.json',config)
   cp=root/'checkpoints/step_000000';cp.mkdir(parents=True);(cp/'state.pt').write_bytes(b'checkpoint')
   atomic_json(cp/'COMPLETE.json',dict(config_digest=d,completed_update=0,state_sha256=sha256_bytes(b'checkpoint')))
   (root/'metrics.jsonl').write_text(''.join(json.dumps(dict(update=i,config_digest=d))+'\n' for i in range(1,51)))
   source=root/'evaluation_states/step_000050';atomic_json(source/'COMPLETE.json',dict(config_digest=d,completed_update=50,kind='in_memory_evaluation_weights',resumable=False,model_identity={'sender_sha256':'a'*64,'receiver_sha256':'b'*64}))
   identity=dict(path=str(source),completed_update=50,complete_sha256=sha256_bytes((source/'COMPLETE.json').read_bytes()))
   ev=root/'evaluations/step_000050';ec=dict(training_config_digest=d,completed_update=50,source_checkpoint=identity);ed=digest(ec)
   atomic_json(ev/'config.json',ec)
   for filename,count,key in [('UNIVERSITY_GENERATIONS.json',2000,'example_id'),('UNIVERSITY_RECEIVER.json',2000,'example_id'),('UNIVERSITY_RESULTS.json',2000,'example_id')]:
    v=dict(config_digest=ed,rows=[{key:i} for i in range(count)])
    if filename=='PIQA_JUDGED.json':v.update(complete=True,n=100,target_questions=100,source_generations_sha256=sha256_bytes((ev/'PIQA_GENERATIONS.json').read_bytes()))
    atomic_json(ev/filename,v)
   summary=dict(config_digest=d,completed_update=50,parameter_updates=50,evaluation_config_digest=ed,source_checkpoint=identity,university={'n':2000},piqa_enabled=False)
   atomic_json(ev/'SUMMARY.json',summary)
   atomic_json(ev/'COMPLETE.json',dict(summary,status='complete',summary_sha256=sha256_bytes((ev/'SUMMARY.json').read_bytes())))
   out=snapshot(root,d)
   self.assertEqual(out['verified_update'],0)
   self.assertEqual(out['files']['metrics.jsonl'],'')
   self.assertEqual(len(out['files']['observed_metrics.jsonl'].splitlines()),50)
   self.assertFalse(out['verified_evaluations'][0]['recoverable'])
   self.assertEqual(out['verified_evaluations'][0]['completed_update'],50)
   (ev/'SUMMARY.json').write_text('{}')
   with self.assertRaises(ValueError):snapshot(root,d)

if __name__=='__main__':unittest.main()
