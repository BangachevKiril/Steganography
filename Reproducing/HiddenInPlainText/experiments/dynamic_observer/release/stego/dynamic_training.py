"""Equal-cycle dynamic observer sweep. Frozen PIQA judge remains evaluation-only."""
import gc,json,math,random,time
from pathlib import Path
from . import university_training as base
from .dynamic_objectives import schedule,sender_reward
from .university_baseline import generate_hf,uni_prompt,university_caps,piqa_messages,save_rows
from . import university_evaluation as evaluation
from orcd.qwen_observer_client import QwenObserverClient,text_hashes


class ObserverClient(QwenObserverClient):
    def score(self,texts):
        result=self._request('score',texts=texts)
        if result['text_sha256']!=text_hashes(texts) or result['count']!=len(texts):raise ValueError('Observer response identity mismatch')
        if len(result['rows'])!=len(texts):raise ValueError('Observer response row count mismatch')
        for r in result['rows']:
            p=r['observer_probabilities']
            if len(p)!=2 or any(not math.isfinite(v) or not 0<=v<=1 for v in p) or not math.isclose(sum(p),1.,abs_tol=1e-6):raise ValueError('Invalid observer probabilities')
            if r['observer_prediction']!=int(p[1]>p[0]):raise ValueError('Observer argmax mismatch')
            if not math.isclose(r['observer_confidence'],max(p),abs_tol=1e-6):raise ValueError('Observer confidence mismatch')
        return result['rows']
    def cleanup(self):return self._request('cleanup')


def setup(config,run):
    phases=schedule(config['dynamic']['k'])
    base.phase_for_update=lambda update:phases[(update-1)%len(phases)]
    def make_observer(c):
        cfg=dict(c);cfg['qwen_observer']=dict(c['qwen_observer'],config_digest=base.digest(c),run_root=str(run))
        return ObserverClient(cfg,run=run)
    base.Observer=make_observer


class Trainer(base.Trainer):
    def __init__(self,config,run):
        self.observer_buffer=[];self.phases=schedule(config['dynamic']['k']);self.cycle_width=len(self.phases)
        super().__init__(config,run)

    def observations(self,ids,texts,labels,cap):
        import torch
        with base.isolated_models(self.receiver):
            self.receiver.eval();rp=self.receiver_logits(texts).softmax(-1).cpu().tolist()
        observed=self.observer.score(texts)
        self.public_rows=[]
        for i,x,y,r,o in zip(ids,texts,labels,rp,observed):
            self.public_rows.append(dict(source_row_id=i,answer=x,true_label=y,receiver_probabilities=r,receiver_prediction=int(r[1]>r[0]),token_cap=cap,**o))
        return dict(n=len(texts),receiver_correct=sum(r['receiver_prediction']==r['true_label'] for r in self.public_rows),observer_correct=sum(r['observer_prediction']==r['true_label'] for r in self.public_rows),receiver_pass=sum(r['receiver_prediction'] for r in self.public_rows),receiver_fail=sum(r['receiver_prediction']==0 for r in self.public_rows),observer_confidence_mean=sum(r['observer_confidence'] for r in self.public_rows)/len(texts))

    def receiver_update(self):
        import torch
        ids,queries,responses,texts,labels,cap=self.generate('receiver')
        counts=self.observations(ids,texts,labels,cap)
        tokens=self.receiver_tokenizer(texts,return_tensors='pt',truncation=True,padding=True).to('cuda:0')
        result=base.receiver_optimization_step(self.receiver,self.receiver_optimizer,tokens,torch.tensor(labels,device='cuda:0'))
        if result['optimizer_sample_count']!=256:raise ValueError('Receiver batch changed')
        self.observer_buffer.extend(dict(answer=x,true_label=y) for x,y in zip(texts,labels))
        return dict(**counts,receiver_training=result)

    def sender_update(self):
        import torch
        ids,queries,responses,texts,labels,cap=self.generate('sender')
        counts=self.observations(ids,texts,labels,cap)
        components=[sender_reward(r['receiver_probabilities'],r['observer_probabilities'],r['true_label'],r['answer']) for r in self.public_rows]
        rewards=[torch.tensor([r['total']],device='cuda:0') for r in components]
        self.log_status('sender_ppo')
        stats=self.ppo.step(queries,responses,rewards)
        public_stats={}
        for key in ['ppo/loss/total','ppo/loss/policy','ppo/loss/value','objective/kl','objective/kl_coef']:
            if key in stats:
                value=float(stats[key])
                if not math.isfinite(value):raise FloatingPointError('Nonfinite PPO statistic '+key)
                public_stats[key]=value
        for row,component in zip(self.public_rows,components):row['reward']=component
        self.observer_buffer.extend(dict(answer=x,true_label=y) for x,y in zip(texts,labels))
        return dict(**counts,ppo=public_stats,reward_mean=sum(r['total'] for r in components)/len(components))

    def observer_update(self):
        # Independent selection RNG: observer k must not consume the sender's RNG stream.
        cycle=self.update//self.cycle_width;index=self.update%self.cycle_width-7
        if self.config['dynamic']['k']==0 or index<0:raise ValueError('Invalid observer phase')
        if len(self.observer_buffer)!=1472:raise ValueError('Observer pool must contain 2*256+5*192 current-cycle summaries')
        batch=random.Random(self.config['dynamic']['observer_sampling_seed']+cycle*100+index).sample(self.observer_buffer,256)
        result=self.observer._request('train',texts=[r['answer'] for r in batch],labels=[r['true_label'] for r in batch])
        self.public_rows=batch
        return {k:v for k,v in result.items() if k not in ['id','protocol_version','ok']}

    def publish_metrics(self):
        if self.update:
            public=self.run/'rollouts';public.mkdir(exist_ok=True)
            base.atomic_json(public/f'update_{self.update:06d}.json',dict(config_digest=self.config_digest,update=self.update,kind=base.phase_for_update(self.update),rows=self.public_rows))
        temporary=self.run/'metrics.jsonl.tmp';temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in self.metrics));temporary.replace(self.run/'metrics.jsonl')

    def evaluate(self,limit=None):
        import torch
        if self.update%self.cycle_width:raise ValueError('Evaluation must follow all observer steps in the cycle')
        cycle=self.update//self.cycle_width
        run=self.run/('validation_evaluation' if limit else 'evaluations')/f'cycle_{cycle:06d}';run.mkdir(parents=True,exist_ok=True)
        checkpoint=evaluation._checkpoint(self);marker=run/'COMPLETE.json'
        if marker.exists():
            m=json.loads(marker.read_text())
            if m['checkpoint']!=checkpoint or m['summary_sha256']!=base.sha256(run/'SUMMARY.json'):raise ValueError('Evaluation provenance mismatch')
            for name,expected in m['files_sha256'].items():
                if base.sha256(run/name)!=expected:raise ValueError('Evaluation file hash mismatch')
            return
        before=self.observer._request('fingerprint')
        with base.isolated_models(self.sender,self.receiver):
            self.sender.eval();self.receiver.eval()
            university,piqa,_=evaluation._selection(self.config)
            if limit:university=university[:limit];piqa=piqa[:min(limit,4)]
            caps=university_caps(2000,self.config['evaluation_seed'])
            rows=evaluation._prefix(run,'UNIVERSITY_RESULTS.json',self.config_digest,university,'example_id',64)
            for start in range(len(rows),len(university),64):
                if base.STOP:raise SystemExit(75)
                selected=university[start:start+64]
                answers=generate_hf(self.sender.pretrained_model,self.tokenizer,[uni_prompt(r['input']) for r in selected],seed=self.config['evaluation_seed']+10000+start,cap=caps[start//192],temperature=1.)
                texts=[r['answer'] for r in answers];rp=self.receiver_logits(texts).softmax(-1).cpu().tolist();observed=self.observer.score(texts)
                rows.extend(dict(**original,**answer,receiver_probabilities=r,receiver_prediction=int(r[1]>r[0]),**o) for original,answer,r,o in zip(selected,answers,rp,observed))
                save_rows(run,'UNIVERSITY_RESULTS.json',rows,self.config_digest)
                self.log_status('evaluation_university',cycle=cycle,completed=len(rows),total=len(university))
            piqa_rows=evaluation._prefix(run,'PIQA_GENERATIONS.json',self.config_digest,piqa,'question_id',16)
            for start in range(len(piqa_rows),len(piqa),16):
                if base.STOP:raise SystemExit(75)
                selected=piqa[start:start+16]
                prompts=[self.tokenizer.apply_chat_template(piqa_messages(r['question']),tokenize=False,add_generation_prompt=True) for r in selected]
                answers=generate_hf(self.sender.pretrained_model,self.tokenizer,prompts,seed=42+20000+start,cap=128,temperature=.6,add_special_tokens=False)
                piqa_rows.extend(dict(**original,**answer) for original,answer in zip(selected,answers));save_rows(run,'PIQA_GENERATIONS.json',piqa_rows,self.config_digest)
            base.atomic_json(run/'config.json',self.config)
            gc.collect();torch.cuda.empty_cache();self.observer.cleanup()
            evaluation._run_judge(self,self.config,run,self.config_digest)
            judged=json.loads((run/'PIQA_JUDGED.json').read_text())
            if (judged.get('complete') is not True or judged['n']!=len(piqa)
                or judged.get('target_questions')!=len(piqa)
                or judged.get('config_digest')!=self.config_digest
                or judged.get('source_generations_sha256')!=base.sha256(run/'PIQA_GENERATIONS.json')):raise ValueError('Incomplete or mismatched PIQA evaluation')
            evaluation.validate_prefix(judged['rows'],piqa,'question_id')
            after=self.observer._request('fingerprint')
            if {k:before[k] for k in ['adapter_sha256','observer_updates']}!={k:after[k] for k in ['adapter_sha256','observer_updates']}:raise ValueError('Observer changed during evaluation')
            counts=dict(receiver_correct=sum(r['receiver_prediction']==r['true_label'] for r in rows),observer_correct=sum(r['observer_prediction']==r['true_label'] for r in rows),receiver_PASS=sum(r['receiver_prediction'] for r in rows),judge_rejected=sum(not r['judge_pass'] for r in judged['rows']),format_rejected=sum(not r['format_ok'] for r in judged['rows']),union_rejected=sum(r['d'] for r in judged['rows']))
            summary=dict(config_digest=self.config_digest,cycle=cycle,completed_update=self.update,k=self.config['dynamic']['k'],university_n=len(rows),piqa_n=len(piqa),counts=counts,C_R=counts['receiver_correct']/len(rows),C_O=counts['observer_correct']/len(rows),observer_confidence=sum(r['observer_confidence'] for r in rows)/len(rows),d=counts['union_rejected']/len(piqa),observer_adapter_sha256=after['adapter_sha256'],observer_updates=after['observer_updates'])
            base.atomic_json(run/'SUMMARY.json',summary)
            names=['UNIVERSITY_RESULTS.json','PIQA_GENERATIONS.json','PIQA_JUDGED.json','SUMMARY.json','config.json']
            base.atomic_json(marker,dict(config_digest=self.config_digest,checkpoint=checkpoint,cycle=cycle,summary_sha256=base.sha256(run/'SUMMARY.json'),files_sha256={n:base.sha256(run/n) for n in names}))

    def step(self):
        if self.update%self.cycle_width==0:self.observer_buffer=[]
        kind=self.phases[self.update%self.cycle_width];self.log_status(kind,cycle=self.update//self.cycle_width+1)
        began=time.monotonic();row=getattr(self,kind+'_update')();self.update+=1
        row.update(kind=kind,update=self.update,cycle=(self.update-1)//self.cycle_width+1,duration_seconds=time.monotonic()-began,config_digest=self.config_digest)
        self.metrics.append(row);self.checkpoint()

    def train(self):
        target=self.config['dynamic']['cycles']*self.cycle_width
        if self.update>target:raise ValueError('Checkpoint exceeds equal-cycle target')
        if self.update%self.cycle_width==0 and (self.update//self.cycle_width)%self.config['dynamic']['evaluation_cycles']==0:self.evaluate()
        while self.update<target:
            if base.STOP:raise SystemExit(75)
            self.step()
            if self.update%self.cycle_width==0:
                cycle=self.update//self.cycle_width
                if cycle%self.config['dynamic']['evaluation_cycles']==0 or self.update==target:self.evaluate()
        self.evaluate()
        observer=self.observer._request('fingerprint')
        if self.config['dynamic']['k']==0 and (observer['observer_updates']!=0 or observer['adapter_sha256']!=self.observer.ready['initial_adapter_sha256']):raise ValueError('k=0 observer changed')
        base.atomic_json(self.run/'COMPLETE.json',dict(config_digest=self.config_digest,cycles=self.config['dynamic']['cycles'],completed_update=self.update,receiver_updates=sum(r['kind']=='receiver' for r in self.metrics),sender_updates=sum(r['kind']=='sender' for r in self.metrics),observer_updates=observer['observer_updates'],checkpoint=evaluation._checkpoint(self),elapsed=self.active_seconds()))
        self.log_status('complete')
