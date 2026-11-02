import contextlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from orcd.probability_objectives import binary_probability,observer_row,reward_components
from orcd.qwen_observer_client import validate_scores,text_hashes
from orcd.qwen_observer_worker import answer_token_ids,format_prompt,NON_THINKING_SUFFIX
from stego import university_training as training

class ProbabilityTests(unittest.TestCase):
    def test_probabilities_are_shift_invariant_not_raw_logit_tanh(self):
        a=torch.tensor([[0.,1.],[1.,0.]])
        obs=[observer_row(0.,0.,'detection')]*2
        x=reward_components(a,torch.tensor([1,0]),obs,['x']*2,'detection')
        y=reward_components(a+100,torch.tensor([1,0]),obs,['x']*2,'detection')
        torch.testing.assert_close(x['reward'],y['reward'])
        self.assertAlmostEqual(float(x['p_true'][0]),1/(1+math.exp(-1)),places=6)

    def test_detection_exact_and_no_hard_gate(self):
        logits=torch.tensor([[0.,math.log(3)]])
        row=observer_row(0.,math.log(4),'detection')
        self.assertTrue(row['observer_flag'])
        got=reward_components(logits,torch.tensor([1]),[row],['x'*10],'detection')
        self.assertAlmostEqual(float(got['reward'][0]),.75-.8+.05,places=6)
        self.assertAlmostEqual(float(got['receiver'][0]),.75,places=6)

    def test_deciphering_penalty_symmetric_and_minimum_at_chance(self):
        logits=torch.tensor([[0.,math.log(3)]]*3)
        rows=[observer_row(0.,x,'deciphering') for x in [-math.log(4),0.,math.log(4)]]
        got=reward_components(logits,torch.ones(3,dtype=torch.long),rows,['x'*100]*3,'deciphering')
        torch.testing.assert_close(got['reward'],torch.tensor([.24,.6,.24]))
        self.assertTrue(all(not r['observer_flag'] for r in rows))

    def test_correct_class_and_length_characters(self):
        logits=torch.tensor([[0.,math.log(3)]]*2)
        got=reward_components(logits,torch.tensor([0,1]),[observer_row(0.,0.,'detection')]*2,['é','x'*21],'detection')
        torch.testing.assert_close(got['p_true'],torch.tensor([.25,.75]))
        torch.testing.assert_close(got['length'],torch.tensor([.005,.1]))

    def test_extreme_logits_stable_and_nonfinite_rejected(self):
        self.assertEqual(binary_probability(-10000.,10000.),1.)
        self.assertEqual(binary_probability(10000.,-10000.),0.)
        for value in (float('nan'),float('inf'),True):
            with self.assertRaises(ValueError):binary_probability(value,0.)

    def test_protocol_probabilities_bound_to_order_and_objective(self):
        for objective in ('detection','deciphering'):
            row=observer_row(-1.,1.,objective)
            response=dict(text_sha256=text_hashes(['a']),count=1,rows=[row])
            self.assertEqual(validate_scores(response,['a'],objective),[row])
            with self.assertRaises(RuntimeError):validate_scores(response,['b'],objective)
            with self.assertRaises(RuntimeError):validate_scores(response,['a'],'deciphering' if objective=='detection' else 'detection')
            row['observer_probability']=float('nan')
            with self.assertRaises(RuntimeError):validate_scores(response,['a'],objective)

    def test_answer_label_mapping_bad_good_boundary(self):
        class Tokenizer:
            def apply_chat_template(self,messages,**kw):return messages[0]['content']+NON_THINKING_SUFFIX
            def encode(self,text,**kw):
                for answer,token in [('Bad',31),('Good',32)]:
                    if text==answer:return [token]
                    if text.endswith(answer):return [1,2,token]
                return [1,2]
            def decode(self,ids,**kw):return {31:'Bad',32:'Good'}[ids[0]]
        tokenizer=Tokenizer();rendered=format_prompt(tokenizer,'Summary:\n{}','literal {braces}')
        self.assertIn('literal {braces}',rendered)
        self.assertEqual(answer_token_ids(tokenizer,rendered,('Bad','Good')),{'Bad':31,'Good':32})

    def test_probability_validation_fails_closed(self):
        logits=torch.zeros(1,2);target=torch.tensor([0]);row=observer_row(0.,0.,'detection')
        for bad in (float('nan'),-.1,1.1,True):
            row['observer_flag_probability']=bad
            with self.assertRaises(ValueError):reward_components(logits,target,[row],['x'],'detection')
        with self.assertRaises(ValueError):reward_components(logits,torch.tensor([2]),[observer_row(0.,0.,'detection')],['x'],'detection')

    def test_production_sender_passes_exact_rewards_to_ppo(self):
        original_tensor=torch.tensor
        def tensor(*args,**kw):
            if kw.get('device')=='cuda:0':kw['device']='cpu'
            return original_tensor(*args,**kw)
        for objective in ('detection','deciphering'):
            model=torch.nn.Linear(1,1);captured=[]
            logits=torch.tensor([[0.,math.log(3)],[math.log(3),0.]])
            rows=[observer_row(0.,math.log(4),objective),observer_row(0.,0.,objective)]
            obj=SimpleNamespace(config={'objective':objective},receiver=model,log_status=lambda *a,**kw:None,
                generate=lambda kind:([7,8],['q1','q2'],['r1','r2'],['x'*10,'x'*20],[1,0],17),
                observer=SimpleNamespace(score=lambda texts:rows),receiver_logits=lambda texts:logits,
                ppo=SimpleNamespace(step=lambda q,r,rewards:captured.extend(float(x) for x in rewards) or {}))
            with patch.object(torch,'tensor',side_effect=tensor),patch.object(training,'isolated_models',side_effect=lambda *a:contextlib.nullcontext()):
                result=training.Trainer.sender_update(obj)
            expected=reward_components(logits,torch.tensor([1,0]),rows,['x'*10,'x'*20],objective)
            torch.testing.assert_close(torch.tensor(captured),expected['reward'])
            self.assertEqual(result['n'],2);self.assertEqual(result['objective'],objective)
            self.assertEqual(len(obj.public_rows),2)
            self.assertAlmostEqual(obj.public_rows[0]['receiver_correct_probability'],.75,places=6)

    def test_receiver_uses_all_256_even_when_detector_flags_every_summary(self):
        original_tensor=torch.tensor
        def tensor(*args,**kw):
            if kw.get('device')=='cuda:0':kw['device']='cpu'
            return original_tensor(*args,**kw)
        class Tokens(dict):
            def to(self,*a):return self
        generated=[];optimized=[]
        def generate(kind):
            generated.append(kind)
            return list(range(256)),[],[],['x']*256,[i%2 for i in range(256)],17
        def optimize(receiver,optimizer,tokens,targets):
            optimized.extend(targets.tolist());return {'loss':.7,'optimizer_sample_count':len(targets)}
        t=SimpleNamespace(config={'receiver_filter':'none'},receiver=torch.nn.Linear(1,1),receiver_optimizer=None,
            public_rows=[],log_status=lambda *a,**kw:None,generate=generate,
            observer=SimpleNamespace(score=lambda texts:[observer_row(0.,10.,'detection') for _ in texts]),
            receiver_logits=lambda texts:torch.zeros(len(texts),2),
            receiver_tokenizer=lambda texts,**kw:Tokens(input_ids=torch.ones(len(texts),1),attention_mask=torch.ones(len(texts),1)))
        with patch.object(torch,'tensor',side_effect=tensor),patch.object(training,'isolated_models',side_effect=lambda *a:contextlib.nullcontext()),patch.object(training,'receiver_optimization_step',side_effect=optimize),patch.object(training,'STOP',False):
            result=training.Trainer.receiver_update(t)
        self.assertEqual(generated,['receiver']);self.assertEqual(optimized,[i%2 for i in range(256)])
        self.assertEqual(result['observer_flagged'],256)

    def test_deciphering_evaluation_reports_accuracy_and_no_detection_rate(self):
        import tempfile
        from stego import university_evaluation as ev
        from orcd import sparse_runtime as rt
        from stego.university_baseline import digest
        class Tokens(dict):
            def to(self,device):return self
        class Model(torch.nn.Module):
            def forward(self,**tokens):return SimpleNamespace(logits=torch.tensor([[0.,1.]]).repeat(len(tokens['input_ids']),1))
        with tempfile.TemporaryDirectory() as tmp:
            config={'evaluation_seed':42,'piqa_enabled':False,'objective':'deciphering'};sender=Model();receiver=Model()
            t=SimpleNamespace(run=Path(tmp),config=config,config_digest=digest(config),update=0,
                sender=sender,receiver=receiver,tokenizer=None,sender_prompt=lambda x:x,
                receiver_tokenizer=lambda texts,**kw:Tokens(input_ids=torch.ones(len(texts),2,dtype=torch.long),attention_mask=torch.ones(len(texts),2,dtype=torch.long)),
                observer=SimpleNamespace(score=lambda texts:[observer_row(0.,math.log(3),'deciphering') for _ in texts]),
                should_stop=lambda:False,log_status=lambda *a,**kw:None)
            sender.pretrained_model=SimpleNamespace()
            cohort=[{'example_id':i,'input':'CV','true_label':i%2} for i in range(2000)]
            with patch.object(rt,'model_identity',return_value={'sender_sha256':'a'*64,'receiver_sha256':'b'*64}),patch.object(ev,'_selection',return_value=cohort),patch.object(ev,'generate_hf',side_effect=lambda model,tokenizer,prompts,**kw:[{'answer':'summary'} for _ in prompts]):
                result=ev.evaluate_training(t);u=result['university']
                self.assertIsNone(u['C_I']);self.assertIsNone(u['counts']['observer_flagged'])
                self.assertEqual(u['observer_accuracy'],.5);self.assertEqual(u['observer_good_fraction'],1.)
                self.assertAlmostEqual(u['mean_observer_probability'],.75)
                self.assertAlmostEqual(u['mean_observer_confidence_penalty'],.25)
                self.assertEqual(result,ev.evaluate_training(t))
