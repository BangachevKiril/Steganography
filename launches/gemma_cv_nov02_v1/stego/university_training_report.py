"""Aggregate public university-only metrics, capped at a saved restart update."""
import argparse,json
from pathlib import Path
from .train import atomic_json

def render_report(run_dir,completed_update):
    root=Path(run_dir)
    rows=[json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines() if line.strip()]
    rows=[r for r in rows if r['update']<=completed_update]
    summary=dict(completed_update=completed_update,observed_updates=len(rows),piqa_enabled=False,
                 receiver_updates=sum(r['kind']=='receiver' for r in rows),sender_updates=sum(r['kind']=='sender' for r in rows))
    atomic_json(root/'TRAINING_SUMMARY.json',summary)
    return summary
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run_dir');parser.add_argument('--completed-update',type=int,required=True);args=parser.parse_args()
    print(json.dumps(render_report(args.run_dir,args.completed_update)))
