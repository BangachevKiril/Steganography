import copy
from types import SimpleNamespace
import unittest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from stego.qwen_pair import CPUAdamW, LabelReceiver
from stego.university_training import receiver_optimization_step, reward_value

class PairTests(unittest.TestCase):
    def test_selected_logits_and_padding(self):
        torch.manual_seed(4)
        c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,attention_dropout=0.)
        c._attn_implementation='eager'
        lm=Qwen3ForCausalLM(c).eval()
        wrapper=LabelReceiver.__new__(LabelReceiver);torch.nn.Module.__init__(wrapper);wrapper.model=lm;wrapper.set_label_ids([4,8])
        ids=torch.tensor([[0,0,2,3,5],[2,3,5,6,7]]);mask=(ids!=0).long()
        pos=mask.cumsum(-1)-1;pos.masked_fill_(mask==0,0)
        with torch.no_grad():
            full=lm(ids,attention_mask=mask,position_ids=pos).logits[:,-1][:,[4,8]]
            selected=wrapper(ids,mask).logits
        torch.testing.assert_close(full,selected)
        with torch.no_grad():single=wrapper(ids[:1,2:],mask[:1,2:]).logits
        torch.testing.assert_close(single,selected[:1])
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});wrapper.train()
        torch.nn.functional.cross_entropy(wrapper(ids,mask).logits,torch.tensor([0,1])).backward()
        self.assertTrue(torch.isfinite(lm.model.embed_tokens.weight.grad).all())

    def test_full_batch_ce_cpu_adam_and_resume(self):
        class Tiny(torch.nn.Module):
            def __init__(self):super().__init__();self.layer=torch.nn.Linear(3,2)
            def forward(self,input_ids,attention_mask):return SimpleNamespace(logits=self.layer(input_ids))
        torch.manual_seed(9);a=Tiny();b=copy.deepcopy(a)
        x=torch.randn(256,3);y=torch.arange(256)%2;mask=torch.ones_like(x)
        oa=CPUAdamW(a.parameters(),lr=2e-5,weight_decay=.01)
        ob=torch.optim.AdamW(b.parameters(),lr=2e-5,weight_decay=.01,foreach=False)
        r=receiver_optimization_step(a,oa,{'input_ids':x,'attention_mask':mask},y)
        torch.nn.functional.cross_entropy(b(x,mask).logits,y).backward();ob.step()
        self.assertEqual((r['optimizer_sample_count'],r['microbatches']),(256,4))
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)
        c=copy.deepcopy(a);oc=CPUAdamW(c.parameters(),lr=2e-5,weight_decay=.01);oc.load_state_dict(copy.deepcopy(oa.state_dict()))
        for model,opt in [(a,oa),(c,oc)]:receiver_optimization_step(model,opt,{'input_ids':x,'attention_mask':mask},y)
        for p,q in zip(a.parameters(),c.parameters()):torch.testing.assert_close(p,q,atol=0,rtol=0)

    def test_receiver_lora_only_independent_and_resumable(self):
        from peft import LoraConfig,get_peft_model
        torch.manual_seed(17)
        config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,attention_dropout=0.)
        config._attn_implementation='eager'
        def make():
            wrapper=LabelReceiver.__new__(LabelReceiver);torch.nn.Module.__init__(wrapper)
            wrapper.model=get_peft_model(Qwen3ForCausalLM(config),LoraConfig(r=16,lora_alpha=16,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','v_proj']))
            wrapper.set_label_ids([4,8]);return wrapper
        receiver=make();sender=make()
        trainable={n:p for n,p in receiver.named_parameters() if p.requires_grad}
        self.assertTrue(trainable);self.assertTrue(all('lora_' in n for n in trainable))
        self.assertFalse({p.data_ptr() for p in receiver.parameters()} & {p.data_ptr() for p in sender.parameters()})
        frozen={n:p.detach().clone() for n,p in receiver.named_parameters() if not p.requires_grad}
        before={n:p.detach().clone() for n,p in trainable.items()}
        opt=CPUAdamW(trainable.values(),lr=1.41e-5,weight_decay=.01)
        ids=torch.randint(1,32,(256,5));mask=torch.ones_like(ids);labels=torch.arange(256)%2
        receiver.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        receiver_optimization_step(receiver,opt,{'input_ids':ids,'attention_mask':mask},labels)
        self.assertTrue(any(not torch.equal(before[n],p) for n,p in trainable.items()))
        for n,p in receiver.named_parameters():
            if n in frozen:torch.testing.assert_close(p,frozen[n],rtol=0,atol=0)
        restored=copy.deepcopy(receiver);newopt=CPUAdamW((p for p in restored.parameters() if p.requires_grad),lr=1.41e-5,weight_decay=.01)
        newopt.load_state_dict(copy.deepcopy(opt.state_dict()))
        rng=torch.get_rng_state()
        receiver_optimization_step(receiver,opt,{'input_ids':ids,'attention_mask':mask},labels)
        torch.set_rng_state(rng)
        receiver_optimization_step(restored,newopt,{'input_ids':ids,'attention_mask':mask},labels)
        for a,b in zip(receiver.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_university_only_evaluation_and_resume(self):
        import tempfile,json
        from pathlib import Path
        from unittest.mock import patch
        from stego import university_evaluation as ev
        from orcd import sparse_runtime as rt
        from stego.university_baseline import digest
        class Tokens(dict):
            def to(self,device):return self
        class Model(torch.nn.Module):
            def forward(self,**tokens):return SimpleNamespace(logits=torch.tensor([[1.,0.]]).repeat(len(tokens['input_ids']),1))
        with tempfile.TemporaryDirectory() as tmp:
            config={'evaluation_seed':42,'piqa_enabled':False,'objective':'detection'};sender=Model();receiver=Model()
            t=SimpleNamespace(run=Path(tmp),config=config,config_digest=digest(config),update=0,
                sender=sender,receiver=receiver,tokenizer=None,sender_prompt=lambda x:x,
                receiver_tokenizer=lambda texts,**kw:Tokens(input_ids=torch.ones(len(texts),2,dtype=torch.long),attention_mask=torch.ones(len(texts),2,dtype=torch.long)),
                observer=SimpleNamespace(score=lambda texts:[{'observer_score':.5,'observer_flag':False,'observer_probability':.25,'observer_flag_probability':.25,'observer_objective':'detection'} for _ in texts]),
                should_stop=lambda:False,log_status=lambda *a,**kw:None)
            sender.pretrained_model=SimpleNamespace()
            cohort=[{'example_id':i,'input':'CV','true_label':i%2} for i in range(2000)]
            with patch.object(rt,'model_identity',return_value={'sender_sha256':'a'*64,'receiver_sha256':'b'*64}),patch.object(ev,'_selection',return_value=cohort),patch.object(ev,'generate_hf',side_effect=lambda model,tokenizer,prompts,**kw:[{'answer':'summary'} for _ in prompts]):
                result=ev.evaluate_training(t)
                self.assertEqual(result['university']['counts']['receiver_correct'],1000)
                self.assertFalse(result['piqa_enabled']);self.assertNotIn('piqa',result)
                self.assertFalse(list(Path(tmp).rglob('PIQA*')))
                self.assertEqual(result,ev.evaluate_training(t));self.assertIsNone(ev._baseline_d(config,t))

    def test_reward(self):
        import math
        self.assertAlmostEqual(reward_value(0.4,0.5,'abc'),math.tanh(.4)+.1+.015)
        self.assertAlmostEqual(reward_value(100,-.5,'abc'),-.1+.015)

if __name__=='__main__':unittest.main()
