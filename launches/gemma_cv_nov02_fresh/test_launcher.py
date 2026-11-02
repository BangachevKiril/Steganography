"""Regression coverage for storage isolation, validated export, and retry safety."""
import csv
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize
import results
import submit


class LaunchTests(unittest.TestCase):
    def test_five_cross_score_columns(self):
        with tempfile.TemporaryDirectory() as base, patch.object(results, 'ATTEMPT', Path(base)):
            records = [dict(training_observer=size, seed=seed, source_sha256='sample',
                            observer_accuracy=(seed-42)/4, observer_pass=.65, observer_auc=.75)
                       for size in submit.SIZES for seed in range(42, 47)]
            folder = Path(base) / 'gemma_cv_nov02_results/cross_scores'
            results.publish_cross_scores('0.8', records)
            table = folder / 'condition_means.csv'
            self.assertFalse(table.exists())
            with ThreadPoolExecutor(max_workers=4) as workers:
                list(workers.map(lambda size: results.publish_cross_scores(size, records),
                                 submit.SIZES[1:]))
            with table.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 25)
            for r in rows:
                self.assertEqual(r['seeds'], '5')
                self.assertEqual(float(r['mean_observer_accuracy']), .5)
            saved = json.loads((folder / 'observer_27.json').read_text())
            self.assertEqual(saved['per_seed'], records)
            before = table.read_bytes()
            results.publish_cross_scores('27', records)
            self.assertEqual(before, table.read_bytes())
            with self.assertRaises(ValueError):
                results.publish_cross_scores('2', records[:-1])

    def test_batch_signal_preserves_job_id_and_child_failure(self):
        dispatch = Path(__file__).with_name('dispatch.sh').resolve()
        with tempfile.TemporaryDirectory() as base:
            root = Path(base)
            worker = root / 'worker.py'
            worker.write_text(
                "from pathlib import Path\nimport signal, sys, time\n"
                "def stop(sig, frame):\n    Path('saw_signal').touch()\n"
                "signal.signal(signal.SIGUSR1, stop)\nPath('ready').touch()\n"
                "while not Path('saw_signal').exists(): time.sleep(.01)\n"
                "time.sleep(.3)\nsys.exit(int(sys.argv[1]))\n")
            for status in (75, 1):
                shell = subprocess.Popen(['bash', '-c',
                    'source "$1"; scontrol() { printf "%s\\n" "$*" > requeued; }; '
                    'run_with_requeue "$2" "$3" "$4"', 'bash', str(dispatch),
                    sys.executable, str(worker), str(status)], cwd=root,
                    env={**os.environ, 'SLURM_JOB_ID': '4242'})
                try:
                    for path in ('ready','saw_signal'):
                        deadline = time.monotonic() + 5
                        while not (root / path).exists() and time.monotonic() < deadline:
                            time.sleep(.01)
                        self.assertTrue((root / path).exists())
                        shell.send_signal(signal.SIGUSR1)
                    self.assertEqual(shell.wait(timeout=5), 0 if status == 75 else 1)
                    receipt = root / 'requeued'
                    if status == 75:
                        self.assertEqual(receipt.read_text().strip(), 'requeue 4242')
                        receipt.unlink()
                    else:
                        self.assertFalse(receipt.exists())
                finally:
                    if shell.poll() is None:
                        shell.kill()
                        shell.wait()
                (root / 'saw_signal').unlink()
                (root / 'ready').unlink()

    def test_lanes_and_tasks(self):
        jobs = {}
        def capture(name, opts, argv):
            self.assertNotIn(name, jobs)
            jobs[name] = (opts, argv)
            return name
        submit.schedule(capture, gates_only=False)
        self.assertEqual(len(jobs), 48)
        names = [submit.job_name(suffix) for suffix in jobs]
        self.assertEqual(len(set(names)), 48)
        self.assertEqual(sum(name.startswith('gcv2r2_v_') for name in names), 7)
        self.assertEqual(sum(name.startswith('gcv2r2_keep1_') for name in names), 41)
        trains = {n: v for n, v in jobs.items() if n.startswith('t_')}
        self.assertEqual(len(trains), 35)
        for size in submit.SIZES:
            for seed in range(43, 47):
                opts, _ = trains[f't_full_{size}_s{seed}']
                self.assertIn(f'--dependency=afterok:t_full_{size}_s{seed-1}', opts)
        for opts, _ in trains.values():
            self.assertIn('--partition=mit_normal_gpu,mit_preemptable', opts)
            self.assertIn('--requeue', opts)
        self.assertIn('--gres=gpu:h200:1', jobs['v_full_27'][0])

    def test_compaction_guard_and_retry(self):
        import torch
        with tempfile.TemporaryDirectory() as base, patch.object(finalize, 'ATTEMPT', Path(base).resolve()):
            root = Path(base).resolve()
            run = root / 'runs' / 'demo'
            def put(path, value):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            cfg = {'name':'demo','seed':42,'assets':{'sender':{'revision':'s'},'receiver':{'revision':'r'}}}
            hd = finalize.digest(cfg)
            put(run/'config.json', cfg)
            (run/'LOCK').touch()
            final = run/'checkpoints/step_001400'
            final.mkdir(parents=True)
            state = final/'state.pt'
            weights = {'sender_trainable':{'policy.lora':torch.randn(4),'v_head.weight':torch.randn(1)},
                       'receiver_trainable':{'model.weight':torch.randn(5)}}
            torch.save({'config_digest':hd,'completed_update':1400,'receiver_optimizer':{},
                        'sender_optimizer':{},**weights}, state)
            modelid = {'sender_sha256':'s','receiver_sha256':'r'}
            cp = {'config_digest':hd,'completed_update':1400,'model_identity':modelid,
                  'state_sha256':finalize.sha(state)}
            put(final/'COMPLETE.json', cp)
            rows = [dict(example_id=i, input='cv', answer='full answer', true_label=i%2,
                         receiver_prediction=0, observer_prediction=1,
                         receiver_probabilities=[.5,.5], observer_good_probability=.6) for i in range(2000)]
            for step in range(0,1401,50):
                ev=run/f'evaluations/step_{step:06d}'
                ident=run/f'evaluation_states/step_{step:06d}'
                common={'config_digest':hd,'completed_update':step}
                put(ident/'COMPLETE.json',{**common,'model_identity':modelid})
                source={'path':str(ident),'completed_update':step,'complete_sha256':finalize.sha(ident/'COMPLETE.json')}
                put(ev/'SUMMARY.json',{**common,'source_checkpoint':source,'university':{'n':2000}})
                put(ev/'COMPLETE.json',{**common,'source_checkpoint':source,'evaluation_config_digest':'e',
                                       'summary_sha256':finalize.sha(ev/'SUMMARY.json')})
                put(ev/'UNIVERSITY_RESULTS.json',{'config_digest':'e','rows':rows})
            marker={'config_digest':hd,'completed_update':1400,'completed_cycles':200,
                    'receiver_updates':400,'sender_updates':1000,'checkpoint':{
                        'path':str(final),'complete_sha256':finalize.sha(final/'COMPLETE.json'),
                        'state_sha256':cp['state_sha256']},
                    'evaluation_complete_sha256':finalize.sha(ev/'COMPLETE.json')}
            put(run/'COMPLETE.json',marker)
            put(run/'BUDGET_COMPLETE.json',marker)
            p=run/'evaluations/step_000050/UNIVERSITY_RESULTS.json'
            original=p.read_bytes()
            put(p, {'config_digest':'e','rows':rows[:-1]})
            with self.assertRaises(ValueError):
                finalize.finalize(run,run/'config.json')
            self.assertTrue(state.is_file())
            p.write_bytes(original)
            finalize.finalize(run,run/'config.json')
            self.assertFalse(state.exists())
            restored=torch.load(run/'final_weights/weights.pt',map_location='cpu',weights_only=False)
            finalize.verify_export(weights,restored)
            finalize.finalize(run,run/'config.json')
            self.assertEqual(len(list((run/'evaluations').glob('*/UNIVERSITY_RESULTS.json'))),29)


if __name__ == '__main__':
    unittest.main()
