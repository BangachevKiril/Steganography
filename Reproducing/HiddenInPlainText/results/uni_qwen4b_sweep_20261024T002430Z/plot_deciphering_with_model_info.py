"""Plot exact deciphering scores and predicted class balance from saved results."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
os.environ.setdefault('MPLCONFIGDIR', tempfile.mkdtemp(prefix='deciphering_plot_'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

root=Path(__file__).resolve().parent
source=root/'deciphering_with_model_info_scores.json'
s=json.loads(source.read_text())
rows=s['results']
assert [r['k'] for r in rows]==[0,1,2,4,8]
for r in rows:
    assert r['total']==2000 and r['true_good']==1001 and r['true_bad']==999
    assert r['predicted_good']+r['predicted_bad']==2000
    assert r['accuracy_percent']==r['correct']/20
    assert r['global_label_flip_diagnostic']['correct']==2000-r['correct']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14,
                     'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,(a,b)=plt.subplots(1,2,figsize=(15,7.3),gridspec_kw={'width_ratios':[1.15,1]})
blue='#0072B2';orange='#D97706';gray='#CBD5E1';ink='#1E293B'
ys=list(range(5))
for offset,flipped,color,label in [(-.18,False,blue,'Submitted labels'),(.18,True,orange,'All labels flipped (post-hoc)')]:
    vals=[r['global_label_flip_diagnostic']['accuracy_percent'] if flipped else r['accuracy_percent'] for r in rows]
    bars=a.barh([y+offset for y in ys],vals,height=.31,color=color,label=label,zorder=3)
    for bar,v,r in zip(bars,vals,rows):
        n=r['global_label_flip_diagnostic']['correct'] if flipped else r['correct']
        a.text(v+1.3,bar.get_y()+bar.get_height()/2,f'{v:.2f}%  ({n:,})',va='center',fontsize=9.5,color=ink)
a.set_yticks(ys,[f"k = {r['k']}" for r in rows]);a.invert_yaxis()
a.set_xlim(0,135);a.set_xticks([0,25,50,75,100]);a.xaxis.set_major_formatter(PercentFormatter(100))
a.set_xlabel('Correct university labels (% of 2,000)')
a.set_title('A. Deciphering accuracy',loc='left',pad=64,fontweight='bold')
a.axvline(50.05,color='#64748B',ls='--',lw=1.2,zorder=2,label='Always-good baseline: 50.05%')
a.legend(loc='lower left',bbox_to_anchor=(0,1.015),fontsize=10,frameon=False)
a.grid(axis='x',alpha=.18,zorder=0)
labels=['Ground truth']+[f"k = {r['k']}" for r in rows]
good=[1001]+[r['predicted_good'] for r in rows]
bad=[999]+[r['predicted_bad'] for r in rows]
yb=list(range(6))
b.barh(yb,good,height=.62,color=blue,label='Good',zorder=3)
b.barh(yb,bad,left=good,height=.62,color=gray,label='Bad',zorder=3)
for y,g,d in zip(yb,good,bad):
    b.text(g/2,y,f'{g:,}\n{g/20:.2f}%',ha='center',va='center',color='white',fontsize=10)
    b.text(g+d/2,y,f'{d:,}\n{d/20:.2f}%',ha='center',va='center',color=ink,fontsize=10)
b.axhline(.5,color='#94A3B8',linewidth=.7)
b.axvline(1001,color='#1E293B',ls='--',lw=1.2,zorder=4,label='True good count: 1,001')
b.set_yticks(yb,labels);b.invert_yaxis()
b.set_xlim(0,2000);b.set_xticks([0,500,1000,1500,2000]);b.set_xlabel('Label count (2,000 messages per dataset)')
b.set_title('B. Predicted good / bad balance',loc='left',pad=64,fontweight='bold')
b.legend(loc='lower left',bbox_to_anchor=(0,1.015),fontsize=10,frameon=False)
fig.suptitle('Steganography deciphering with model information',fontsize=20,fontweight='semibold',y=.98)
fig.text(.5,.925,'Final cycle 200 · 50/50 prior + sender/receiver identities supplied · 2,000 predictions per k',ha='center',color='#475569',fontsize=12)
fig.text(.06,.045,'Flipped labels are a diagnostic using ground truth, not the agent’s submitted accuracy.\nTrue labels for every dataset: 1,001 good (50.05%) and 999 bad (49.95%).',fontsize=11,color='#475569',linespacing=1.5)
fig.subplots_adjust(left=.06,right=.985,top=.70,bottom=.18,wspace=.27)
for ext in ['png','svg']:
    target=root/f'deciphering_with_model_info_results.{ext}'
    fig.savefig(target,dpi=180,facecolor='white')
    if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
plt.close(fig)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
provenance={'source_scores_sha256':sha(source),'script_sha256':sha(Path(__file__)),
    'artifacts':{f'deciphering_with_model_info_results.{e}':sha(root/f'deciphering_with_model_info_results.{e}') for e in ['png','svg']}}
(root/'DECIPHERING_WITH_MODEL_INFO_PLOT_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(root/'deciphering_with_model_info_results.png')
