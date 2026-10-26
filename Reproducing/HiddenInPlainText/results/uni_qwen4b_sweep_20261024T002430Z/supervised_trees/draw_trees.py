"""Render the saved supervised decision trees without refitting or simplifying rules."""
import hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/supervised-tree-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[5];OUT=Path(__file__).resolve().parent
SOURCE=OUT.parent/'supervised_coding_schemes.json'
saved=json.loads(SOURCE.read_text())
COLORS={0:'#555555',1:'#e69f00',2:'#0072b2',4:'#009e73',8:'#cc79a7'}
plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','svg.hashsalt':'supervised-scheme-trees','savefig.facecolor':'white'})
def isleaf(n):return 'feature_index' not in n
def depth(n):return 0 if isleaf(n) else 1+max(depth(n['yes']),depth(n['no']))
def leaves(n):return 1 if isleaf(n) else leaves(n['yes'])+leaves(n['no'])
def nodes(n):return 1 if isleaf(n) else 1+nodes(n['yes'])+nodes(n['no'])
for row in saved['results']:
 k=row['k'];tree=row['tree'];assert nodes(tree)==row['nodes']
 d=depth(tree);nleaf=leaves(tree)
 # Coordinates use fixed-size boxes and gaps, with YES branches above NO.
 positions={};cursor=[0]
 def place(n,level=0):
  if isleaf(n):y=cursor[0]*1.1;cursor[0]+=1
  else:
   y=(place(n['yes'],level+1)+place(n['no'],level+1))/2
  positions[id(n)]=(level*3.15+1.25,y)
  return y
 place(tree)
 graph_h=(nleaf-1)*1.1+1.05;graph_w=d*3.15+2.65
 fig=plt.figure(figsize=(graph_w+.7,graph_h+1.8))
 ax=fig.add_axes([.025,1.0/(graph_h+1.8),.95,graph_h/(graph_h+1.8)])
 ax.set_xlim(-.1,graph_w+.1);ax.set_ylim(graph_h-.525,-.525);ax.axis('off')
 def draw(n):
  x,y=positions[id(n)];leaf=isleaf(n);w=1.55 if leaf else 2.25;h=.66 if leaf else .72
  if leaf:
   good=bool(n['prediction']);edge='#0072b2' if good else '#b85c00';fill='#eaf4fb' if good else '#fff2e5'
   label=('GOOD' if good else 'BAD')+f"\nfit n = {n['n']:,}"
  else:
   feature=row['vocabulary'][n['feature_index']];assert feature==n['feature']
   kind,value=feature.split(':',1)
   label=f"Contains {'phrase' if kind=='phrase' else 'word' if kind=='word' else 'opening'}\n“{value}”?"
   edge='#718096';fill='#f3f5f8'
  ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.02,rounding_size=0.08',lw=1.3,edgecolor=edge,facecolor=fill,zorder=3))
  ax.text(x,y,label,ha='center',va='center',fontsize=11.5,color='#253247',linespacing=1.3,zorder=4)
  if not leaf:
   for branch in ['yes','no']:
    child=n[branch];cx,cy=positions[id(child)];cw=1.55 if isleaf(child) else 2.25
    start=(x+w/2+.02,y);end=(cx-cw/2-.025,cy)
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,lw=1.3,color='#718096',zorder=1))
    tx=start[0]+.52*(end[0]-start[0]);ty=y+.52*(cy-y)
    ax.text(tx,ty,branch.capitalize(),ha='center',va='center',fontsize=9.5,color='#253247',bbox={'facecolor':'white','edgecolor':'none','pad':1.6},zorder=2)
    draw(child)
 draw(tree)
 score=row['scores']['test']
 fig.text(.035,1-.15/(graph_h+1.8),f"Supervised Coding Scheme · k = {k}",ha='left',va='top',fontsize=19,weight='semibold',color=COLORS[k])
 fig.text(.035,1-.52/(graph_h+1.8),f"Internal held-out accuracy: {score['accuracy_percent']:.2f}% ({score['correct']}/{score['n']})",ha='left',va='top',fontsize=12,color='#253247')
 fig.text(.035,.22/(graph_h+1.8), 'YES = present; NO = absent. fit n = training examples at leaf.\nLowercase tokens; phrases are adjacent tokens.\nSplit: 1,214 train / 394 validation / 392 internal test.', fontsize=9.5, color='#596273', va='bottom', linespacing=1.35)
 fig.canvas.draw()
 for text in fig.texts:
  bbox=text.get_window_extent(fig.canvas.get_renderer())
  assert bbox.x0>=0 and bbox.x1<=fig.bbox.width and bbox.y0>=0 and bbox.y1<=fig.bbox.height
 for ext in ['png','svg']:
  target=OUT/f'supervised_tree_k{k}.{ext}'
  fig.savefig(target,dpi=160)
  if ext=='svg':target.write_text('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n')
 plt.close(fig)
(OUT/'PROVENANCE.json').write_text(json.dumps({'source':str(SOURCE.relative_to(ROOT)),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'method':'Direct rendering of saved pruned trees; no refitting, feature changes, polarity changes, or additional pruning.','qualification':saved['qualification']},indent=2)+'\n')
print('Rendered all five exact supervised trees.')
