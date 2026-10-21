"""Descriptive activation-magnitude checks; no models or clusters are fitted."""
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

BASE=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def summary(values):
    values=np.asarray(values,dtype=np.float64)
    percentiles=np.percentile(values,[0,5,25,50,75,95,100])
    return {'n':len(values),'mean':float(values.mean()),'std_population':float(values.std()),
            'coefficient_of_variation':float(values.std()/values.mean()),
            'min':float(percentiles[0]),'p05':float(percentiles[1]),'p25':float(percentiles[2]),
            'median':float(percentiles[3]),'p75':float(percentiles[4]),'p95':float(percentiles[5]),
            'max':float(percentiles[6]),'p95_div_p05':float(percentiles[5]/percentiles[1])}

def group_effect(values,group):
    a=values[group==0];b=values[group==1]
    pooled=np.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/(len(a)+len(b)-2))
    return {'group0':summary(a),'group1':summary(b),'mean_difference_group1_minus_group0':float(b.mean()-a.mean()),
            'relative_mean_difference':float(b.mean()/a.mean()-1),
            'pooled_standardized_mean_difference':float((b.mean()-a.mean())/pooled)}

def main():
    out=BASE/'results'/('activation_magnitudes_1076_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(exist_ok=False)
    labels_path=BASE/'data/university_observer_checkpoint1076_split/test_hidden_labels.jsonl'
    assert sha(labels_path)=='c1c9d35c11f8e5cd5ffe87175a4c862bd6e24df8ac8941685a5da99a4faf7c09'
    labels={r['example_id']:r['hidden_label'] for r in map(json.loads,labels_path.read_text().splitlines())}
    report={'status':'complete','created_at':datetime.now(timezone.utc).isoformat(),'n_examples':1000,
            'scope':'Descriptive magnitude analysis of existing frozen test embeddings and existing frozen clusters; no fitting or selection.',
            'definitions':{'L2':'sqrt(sum_j x_j^2)','RMS':'L2 / sqrt(number of features)','coefficient_of_variation':'population std / mean',
                           'centered':'Subtract feature means computed over these1000unlabeled test examples',
                           'standardized':'Subtract feature means and divide by feature population std, exactly as previous k-means; zero std scale1',
                           'labels':'0=FAIL,1=PASS; group comparisons are descriptive'},
            'script_sha256':sha(__file__),'labels_sha256':sha(labels_path),'models':{}}
    data={}; all_rows=[]
    for tag,name in [('qwen25_3b','Qwen2.5-3B-Instruct'),('gemma_untrained','Original pretrained Gemma'),('gemma_trained','Trained Gemma sender')]:
        launch=BASE/'launches'/(tag+'_test_kmeans_1076_latest.json');record=json.loads(launch.read_text());plan=json.loads(Path(record['plan']).read_text());emb=Path(plan['input_embeddings']);assert sha(emb)==plan['input_embeddings_sha256']
        root=Path(record['local_results'])/'clusters';complete=json.loads((root/'CLUSTER_COMPLETE.json').read_text());assert sha(root/'CLUSTER_COMPLETE.json')==record['cluster_completion_sha256']
        for n in ['cluster_assignments.jsonl','clustering_model.npz']:
            assert sha(root/n)==complete['files'][n]
        with np.load(emb,allow_pickle=False) as f:
            x=f['embeddings'].astype(np.float64);ids=f['example_ids'].copy()
        assert x.shape[0]==1000 and np.isfinite(x).all() and len(set(ids))==1000
        y=np.array([labels[int(i)] for i in ids]);assert np.array_equal(np.bincount(y),[500,500])
        assignments=[json.loads(s) for s in (root/'cluster_assignments.jsonl').read_text().splitlines()]
        assert [r['example_id'] for r in assignments]==ids.tolist()
        c=np.array([r['cluster'] for r in assignments]);mean=x.mean(0);std=x.std(0);scale=np.where(std==0,1,std)
        with np.load(root/'clustering_model.npz',allow_pickle=False) as f:
            np.testing.assert_array_equal(mean,f['feature_mean']);np.testing.assert_array_equal(scale,f['feature_scale'])
        centered=x-mean;z=centered/scale
        norms={'raw':np.linalg.norm(x,axis=1),'centered':np.linalg.norm(centered,axis=1),'standardized':np.linalg.norm(z,axis=1)}
        entry={'name':name,'features':x.shape[1],'input_file':str(emb),'input_sha256':sha(emb),
               'raw_rms':summary(norms['raw']/np.sqrt(x.shape[1])),
               'raw_mean_absolute_activation':summary(np.abs(x).mean(1)),
               'feature_std_distribution':summary(std),
               'norm_of_common_mean_vector':float(np.linalg.norm(mean)),
               'common_mean_share_of_average_squared_norm':float(mean@mean/np.square(x).sum(1).mean()),
               'norms':{k:{'overall':summary(v),'by_hidden_label':group_effect(v,y),'by_frozen_cluster':group_effect(v,c)} for k,v in norms.items()},
               'correlation_raw_vs_standardized_norm':float(np.corrcoef(norms['raw'],norms['standardized'])[0,1])}
        report['models'][tag]=entry;data[tag]=(name,norms,y,c,ids)
        for i,eid in enumerate(ids):
            all_rows.append({'model':name,'example_id':int(eid),'hidden_label':int(y[i]),'frozen_cluster':int(c[i]),
                             **{k+'_l2_norm':float(v[i]) for k,v in norms.items()},'raw_rms':float(norms['raw'][i]/np.sqrt(x.shape[1]))})
    orig=data['gemma_untrained'];trained=data['gemma_trained'];assert np.array_equal(orig[4],trained[4])
    delta=trained[1]['raw']-orig[1]['raw'];rel=trained[1]['raw']/orig[1]['raw']-1
    report['paired_gemma_change_trained_minus_original']={'mean_l2_delta':float(delta.mean()),'mean_relative_change':float(rel.mean()),
        'median_relative_change':float(np.median(rel)),'relative_change_p05_p95':np.percentile(rel,[5,95]).tolist(),
        'raw_norm_correlation':float(np.corrcoef(orig[1]['raw'],trained[1]['raw'])[0,1])}
    (out/'magnitude_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with (out/'per_example_magnitudes.jsonl').open('w') as f:
        for r in all_rows:f.write(json.dumps(r,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(11,10),layout='constrained')
    for row,(name,norms,y,c,ids) in enumerate(data.values()):
        for col,(kind,g,labels_group) in enumerate([('raw',y,['FAIL','PASS']),('standardized',c,['Cluster0','Cluster1'])]):
            ax=axes[row,col];v=norms[kind];bins=np.histogram_bin_edges(v,bins=35)
            for k,color in [(0,'#4169a5'),(1,'#d57930')]:
                ax.hist(v[g==k],bins=bins,density=True,histtype='step',linewidth=1.8,color=color,label=labels_group[k])
            ax.set(title=name,xlabel=('Raw EOS L2 norm' if kind=='raw' else 'Standardized EOS L2 norm'),ylabel='Density')
            ax.legend(frameon=False);ax.grid(alpha=.18)
    fig.suptitle('Activation magnitude: hidden labels and existing clusters',fontsize=15)
    fig.savefig(out/'magnitude_distributions.png',dpi=160);plt.close(fig)
    names=['magnitude_summary.json','per_example_magnitudes.jsonl','magnitude_distributions.png']
    (out/'COMPLETE.json').write_text(json.dumps({'status':'complete','script_sha256':sha(__file__),'files':{n:sha(out/n) for n in names}},indent=2)+'\n')
    (BASE/'launches/activation_magnitudes_1076_latest.json').write_text(json.dumps({'status':'COMPLETED','local_results':str(out),'local_source':str(Path(__file__).parent),'completed_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
    print(json.dumps({'output':str(out),'models':{tag:{'raw_l2':r['norms']['raw']['overall'],'raw_rms_mean':r['raw_rms']['mean'],
        'raw_FAIL_mean':r['norms']['raw']['by_hidden_label']['group0']['mean'],'raw_PASS_mean':r['norms']['raw']['by_hidden_label']['group1']['mean'],
        'raw_label_difference_percent':100*r['norms']['raw']['by_hidden_label']['relative_mean_difference'],
        'raw_label_effect_d':r['norms']['raw']['by_hidden_label']['pooled_standardized_mean_difference'],
        'standardized_l2':r['norms']['standardized']['overall'],
        'standardized_cluster_means':[r['norms']['standardized']['by_frozen_cluster'][k]['mean'] for k in ['group0','group1']],
        'standardized_cluster_effect_d':r['norms']['standardized']['by_frozen_cluster']['pooled_standardized_mean_difference'],
        'mean_share':r['common_mean_share_of_average_squared_norm']} for tag,r in report['models'].items()},
        'paired_gemma':report['paired_gemma_change_trained_minus_original']},indent=2))

if __name__=='__main__':main()
