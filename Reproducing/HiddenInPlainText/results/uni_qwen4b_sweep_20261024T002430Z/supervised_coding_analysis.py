"""Label-informed, shallow lexical rules with duplicate-grouped train/dev/test.
No test labels are used for vocabulary, fitting, rule selection, or polarity.
This is an internal split of previously inspected messages, not a fresh corpus.
"""
import argparse,hashlib,json,re
from pathlib import Path
from collections import Counter
import numpy as np
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[4], help='Repository root containing the original answer exports and verified ground truth')
parser.add_argument('--output', type=Path, help='Output directory; defaults to output/supervised_coding_schemes under root')
args=parser.parse_args()
ROOT=args.root.resolve(); OUT=args.output or ROOT/'output/supervised_coding_schemes'
OUT.mkdir(parents=True, exist_ok=True)
KS=[0,1,2,4,8]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
truth_path=ROOT/'output/answers_steg_only_scoring/verified_ground_truth.json'
gt=json.loads(truth_path.read_text());y=np.array(gt['true_labels'],dtype=int)
assert len(y)==2000 and sum(y)==1001
texts={k:json.loads((ROOT/f'output/answers_steg_only/answers_{k}.json').read_text()) for k in KS}
for k in KS:
 expected=next(r for r in gt['sources'] if r['k']==k)
 assert sha(ROOT/f'output/answers_steg_only/answers_{k}.json')==expected['answer_sha256']
# Union row IDs sharing any normalized answer within any setting. The same row
# split is used for all settings, and no identical normalized answer crosses it.
parent=list(range(2000))
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def union(i,j):
 a,b=find(i),find(j)
 if a!=b:parent[b]=a
for k in KS:
 seen={}
 for i,t in enumerate(texts[k]):
  key=' '.join(t.lower().split())
  if key in seen:union(i,seen[key])
  else:seen[key]=i
groups={}
for i in range(2000):groups.setdefault(find(i),[]).append(i)
rng=np.random.default_rng(20261025)
group_list=list(groups.values());rng.shuffle(group_list)
# Greedy random-group allocation toward 60/20/20 targets, without using labels.
target=np.array([1200,400,400]);sizes=np.zeros(3,dtype=int);splits=[[],[],[]]
for g in group_list:
 slot=int(np.argmax((target-sizes)/target))
 splits[slot].extend(g);sizes[slot]+=len(g)
train,dev,test=[np.array(sorted(a),dtype=int) for a in splits]
assert set(train)|set(dev)|set(test)==set(range(2000))
for k in KS:
 sets=[{' '.join(texts[k][i].lower().split()) for i in ids} for ids in splits]
 assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
def features(text):
 words=re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
 f={'word:'+w for w in words}
 f.update('phrase:'+a+' '+b for a,b in zip(words,words[1:]))
 if words:f.add('opening:'+words[0])
 return f

def fit(X,y,indices,depth,min_leaf):
 vals=y[indices];n=len(vals);ones=int(sum(vals));node={'prediction':int(ones*2>=n),'n':n,'good':ones}
 if depth==0 or ones in [0,n]:return node
 counts=X[indices].sum(axis=0).astype(float)
 pos=(X[indices]*vals[:,None]).sum(axis=0)
 neg_n=n-counts;neg_pos=ones-pos
 valid=(counts>=min_leaf)&(neg_n>=min_leaf)
 loss=np.full(X.shape[1],np.inf)
 loss[valid]=2*(pos[valid]*(counts[valid]-pos[valid])/counts[valid]+neg_pos[valid]*(neg_n[valid]-neg_pos[valid])/neg_n[valid])/n
 j=int(np.argmin(loss));base=2*ones*(n-ones)/(n*n)
 if not np.isfinite(loss[j]) or loss[j]>=base-1e-12:return node
 yes=indices[X[indices,j]];no=indices[~X[indices,j]]
 node.update(feature_index=j,no=fit(X,y,no,depth-1,min_leaf),yes=fit(X,y,yes,depth-1,min_leaf))
 return node

def predict(node,X):
 out=np.empty(len(X),dtype=int)
 def recurse(n,ids):
  if 'feature_index' not in n:out[ids]=n['prediction'];return
  m=X[ids,n['feature_index']];recurse(n['yes'],ids[m]);recurse(n['no'],ids[~m])
 recurse(node,np.arange(len(X)));return out

def prune_same_predictions(t):
 if 'feature_index' not in t:return t.copy()
 out=t.copy();out['yes']=prune_same_predictions(t['yes']);out['no']=prune_same_predictions(t['no'])
 if 'feature_index' not in out['yes'] and 'feature_index' not in out['no'] and out['yes']['prediction']==out['no']['prediction']:
  return {'prediction':out['yes']['prediction'],'n':out['n'],'good':out['good']}
 return out

def count_nodes(t):return 1 if 'feature_index' not in t else 1+count_nodes(t['yes'])+count_nodes(t['no'])
def describe(t,vocab,indent=''):
 if 'feature_index' not in t:return indent+('GOOD' if t['prediction'] else 'BAD')+f" (fit n={t['n']}, good={t['good']})"
 name=vocab[t['feature_index']];t['feature']=name
 return indent+'IF '+name+' PRESENT:\n'+describe(t['yes'],vocab,indent+'  ')+'\n'+indent+'ELSE:\n'+describe(t['no'],vocab,indent+'  ')
results=[]
for k in KS:
 fs=[features(t) for t in texts[k]]
 freq=Counter(f for i in train for f in fs[i])
 vocab=sorted(f for f,n in freq.items() if n>=5 and n<=len(train)-5)
 X=np.array([[f in row for f in vocab] for row in fs],dtype=bool)
 candidates=[]
 for depth in [1,2,3,4]:
  for leaf in [10,20]:
   tree=fit(X,y,train,depth,leaf);pred=predict(tree,X[dev]);acc=float(np.mean(pred==y[dev]))
   candidates.append(dict(depth=depth,min_leaf=leaf,validation_accuracy=acc,nodes=count_nodes(tree),tree=tree))
 best=max(c['validation_accuracy'] for c in candidates)
 # Prespecified simplicity preference: smallest tree within 1 percentage point
 # of the best validation accuracy; then accuracy, then more conservative leaf.
 eligible=[c for c in candidates if c['validation_accuracy']>=best-.01-1e-12]
 chosen=min(eligible,key=lambda c:(c['nodes'],-c['validation_accuracy'],-c['min_leaf'],c['depth']))
 tree=prune_same_predictions(chosen['tree']);rule=describe(tree,vocab);pred=predict(tree,X)
 assert np.array_equal(pred,predict(chosen['tree'],X))
 scores={}
 for name,ids in [('train',train),('validation',dev),('test',test),('all_descriptive',np.arange(2000))]:
  scores[name]=dict(correct=int(sum(pred[ids]==y[ids])),n=len(ids),accuracy_percent=float(np.mean(pred[ids]==y[ids])*100),good=int(sum(y[ids])),predicted_good=int(sum(pred[ids])))
 result=dict(k=k,selected_depth=chosen['depth'],min_leaf=chosen['min_leaf'],nodes_before_pruning=chosen['nodes'],nodes=count_nodes(tree),rule=rule,tree=tree,vocabulary=vocab,scores=scores,
 candidates=[{a:b for a,b in c.items() if a!='tree'} for c in candidates])
 results.append(result)
 (OUT/f'predictions_k{k}.json').write_text(json.dumps([dict(example_id=i,split='train' if i in set(train) else 'validation' if i in set(dev) else 'test',true_label=int(y[i]),prediction=int(pred[i])) for i in range(2000)],indent=2)+'\n')
 print('k=',k,'depth',chosen['depth'],'nodes',chosen['nodes'],'scores',scores)
 print(rule)
record=dict(method='Binary presence of words, adjacent word pairs, and first word; shallow Gini decision trees; training-only vocabulary min document frequency 5; depths 1..4 and min_leaf 10 or 20; choose fewest nodes within 1 percentage point of best validation accuracy; no refit after selection.',
seed=20261025,split_method='Greedy allocation of shuffled connected groups of duplicate normalized messages across all five k settings toward 60/20/20; no labels used for allocation.',
qualification='Internal grouped holdout of previously inspected final outputs. Neither a fresh independent corpus nor proof that the trained receiver uses these particular features. All-data accuracy is descriptive and includes fit/selection rows.',
source_truth_sha256=sha(truth_path),answer_sha256={str(k):sha(ROOT/f'output/answers_steg_only/answers_{k}.json') for k in KS},
splits={name:[int(i) for i in ids] for name,ids in zip(['train','validation','test'],[train,dev,test])},groups=len(groups),results=results,numpy_version=np.__version__,analysis_script_sha256=sha(Path(__file__)))
(OUT/'results.json').write_text(json.dumps(record,indent=2)+'\n')
