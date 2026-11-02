"""Score the same fixed final transmissions from the five observer-size conditions."""
import hashlib,json,sys
from pathlib import Path
from orcd.qwen_observer_client import QwenObserverClient
from evaluation_measures import measures
from stego.train import atomic_json
root=Path('/orcd/scratch/orcd/003/kirilb/Steganography/runs')
size,seed=sys.argv[1:]
cfg=json.loads(Path(f'configs/gemma_cv_nov02_full_o{size}_s{seed}.json').read_text())
judge=QwenObserverClient(cfg)
try:
    for trained in ['0.8','2','4','9','27']:
        run=root/f'gemma_cv_nov02_full_o{trained}_s{seed}'
        final=json.loads((run/'COMPLETE.json').read_text())
        path=run/'evaluations/step_001400'
        marker=(path/'COMPLETE.json').read_bytes()
        assert final['evaluation_complete_sha256']==hashlib.sha256(marker).hexdigest()
        src=(path/'UNIVERSITY_RESULTS.json').read_bytes();artifact=json.loads(src)
        assert artifact['config_digest']==json.loads(marker)['evaluation_config_digest']
        rows=artifact['rows'];out=[]
        for start in range(0,len(rows),64):
            batch=rows[start:start+64];scores=judge.score([r['answer'] for r in batch])
            out.extend({**r, **score} for r,score in zip(batch,scores))
        directory=run/'final_cross_scoring';directory.mkdir(exist_ok=True)
        atomic_json(directory/f'observer_{size}.json',dict(training_observer=trained,scoring_observer=cfg['assets']['observer'],seed=int(seed),source_sha256=hashlib.sha256(src).hexdigest(),metrics=measures(out),rows=[{k:r[k] for k in ('example_id','true_label','observer_prediction','observer_good_probability')} for r in out]))
finally:judge.close()
