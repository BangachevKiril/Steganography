import unittest
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from stego.qwen_pair import LabelReceiver,CPUAdamW
from stego.university_training import receiver_optimization_step
from orcd.ablation_validation import receiver_update_audit,verify_receiver_restart
from pathlib import Path
from types import SimpleNamespace
import tempfile

class TiedReceiverTests(unittest.TestCase):
    def make(self,tied=True):
        torch.manual_seed(17)
        c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,tie_word_embeddings=tied)
        c._attn_implementation='eager'
        r=LabelReceiver.__new__(LabelReceiver);torch.nn.Module.__init__(r)
        r.model=Qwen3ForCausalLM(c).to(torch.bfloat16);r.set_label_ids([4,8])
        r.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        return r

    def test_reproduces_false_failure_and_accepts_real_tied_update(self):
        r=self.make();before={n:p.detach().clone() for n,p in r.named_parameters()}
        opt=CPUAdamW(r.parameters(),lr=1.41e-5,weight_decay=.01,fp32_master_state=True)
        ids=torch.randint(1,32,(256,5));y=torch.arange(256)%2
        stats=receiver_optimization_step(r,opt,dict(input_ids=ids,attention_mask=torch.ones_like(ids)),y)
        changed=[n for n,p in r.named_parameters() if not torch.equal(before[n],p)]
        self.assertFalse(any('lm_head' in n for n in changed))  # Old GPU gate always failed here.
        audit=receiver_update_audit(r,before)
        self.assertTrue(audit['passed']);self.assertTrue(audit['input_output_weights_tied'])
        self.assertEqual(audit['embedding_parameter'],audit['output_head_parameter'])
        self.assertEqual(len(opt.params),len({id(p) for p in opt.params}))
        self.assertEqual(stats['optimizer_sample_count'],256)
        with tempfile.TemporaryDirectory() as tmp:
            verify_receiver_restart(SimpleNamespace(receiver=r,receiver_optimizer=opt),Path(tmp)/'state.pt')
        self.assertIs(r.model.get_input_embeddings().weight,r.model.get_output_embeddings().weight)

    def test_no_update_still_rejected(self):
        r=self.make();before={n:p.detach().clone() for n,p in r.named_parameters()}
        self.assertFalse(receiver_update_audit(r,before)['passed'])

    def test_untied_unchanged_head_still_rejected(self):
        r=self.make(False);before={n:p.detach().clone() for n,p in r.named_parameters()}
        with torch.no_grad():
            r.model.get_input_embeddings().weight.add_(.1)
            next(r.model.model.layers[0].parameters()).add_(.1)
        audit=receiver_update_audit(r,before)
        self.assertTrue(audit['embedding_updated']);self.assertTrue(audit['first_backbone_layer_updated'])
        self.assertFalse(audit['output_head_updated']);self.assertFalse(audit['passed'])

    def test_frozen_parameter_rejected(self):
        r=self.make();before={n:p.detach().clone() for n,p in r.named_parameters()}
        next(r.model.model.layers[0].parameters()).requires_grad_(False)
        with self.assertRaises(ValueError):receiver_update_audit(r,before)
