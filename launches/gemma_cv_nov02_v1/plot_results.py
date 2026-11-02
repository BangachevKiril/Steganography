"""Render per-condition curves from all five completed fixed-cohort evaluations."""
import json
from pathlib import Path
from statistics import mean

root=Path('/orcd/scratch/orcd/003/kirilb/Steganography/runs')
out=root/'gemma_cv_nov02_results'
out.mkdir(exist_ok=True)
steps=list(range(0,1401,50))
colors=['#316fa0','#ba6131','#277a68','#a34f81','#8163a3']
for mode,size in [('full',s) for s in ('0.8','2','4','9','27')]+[('r16','9'),('r32','9')]:
    runs=[]
    for seed in range(42,47):
        run=root/f'gemma_cv_nov02_{mode}_o{size}_s{seed}'
        assert json.loads((run/'COMPLETE.json').read_text())['completed_update']==1400
        runs.append([json.loads((run/f'evaluations/step_{step:06d}/SUMMARY.json').read_text())['university'] for step in steps])
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="680" font-family="sans-serif"><rect width="100%" height="100%" fill="white"/>',
         f'<text x="500" y="32" text-anchor="middle" font-size="22">Gemma receiver {mode} · Qwen3.5-{size}B observer</text>']
    for index,key in enumerate(['receiver_accuracy','observer_accuracy','receiver_pass','observer_pass']):
        x=65+(index%2)*480;y=92+(index//2)*265;w=385;h=180
        svg.append(f'<text x="{x}" y="{y-14}" font-size="17">{key.replace("_"," ").capitalize()}</text>')
        for tick in [0,.5,1]:
            yy=y+(1-tick)*h
            svg.append(f'<path d="M{x} {yy}h{w}" stroke="#ddd"/><text x="{x-6}" y="{yy+4}" text-anchor="end">{tick:.0%}</text>')
        for tick in [0,700,1400]:
            svg.append(f'<text x="{x+w*tick/1400}" y="{y+h+22}" text-anchor="middle">{tick}</text>')
        series=[[v[key] for v in run] for run in runs]
        for values,color,lw in zip(series+[[mean(v) for v in zip(*series)]],colors+['#151515'],[1.5]*5+[3]):
            pts=' '.join(f'{x+w*s/1400:.2f},{y+h*(1-v):.2f}' for s,v in zip(steps,values))
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{lw}"/>')
    svg.append('<text x="500" y="602" text-anchor="middle">Total parameter updates</text>')
    for i,(label,color) in enumerate(zip(['Seed '+str(s) for s in range(42,47)]+['Mean'],colors+['#151515'])):
        svg.append(f'<text x="{155+i*126}" y="646" fill="{color}">{label}</text>')
    (out/f'{mode}_o{size}.svg').write_text('\n'.join(svg+['</svg>']))
