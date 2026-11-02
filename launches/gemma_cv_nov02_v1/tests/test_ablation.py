import copy
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from transformers import Qwen3Config,Qwen3ForCausalLM
from stego.qwen_pair import CPUAdamW,LabelReceiver
from stego.university_training import receiver_optimization_step
from orcd.ablation_validation import optimizer_digest,verify_receiver_restart

class AblationTests(unittest.TestCase):
    def test_full_receiver_constructor_and_backbone_update(self):
        c=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8)
        c._attn_implementation='eager'
        torch.manual_seed(17)
        with patch('stego.qwen_pair.AutoModelForCausalLM.from_pretrained',return_value=Qwen3ForCausalLM(c)):
            model=LabelReceiver('unused',full_training=True)
        model.set_label_ids([4,8])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(any('lora_' in n for n,p in model.named_parameters()))
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        opt=CPUAdamW(model.parameters(),lr=1.41e-5,weight_decay=.01,fp32_master_state=True)
        ids=torch.randint(1,32,(256,5));labels=torch.arange(256)%2
        stats=receiver_optimization_step(model,opt,dict(input_ids=ids,attention_mask=torch.ones_like(ids)),labels)
        self.assertEqual((stats['optimizer_sample_count'],stats['microbatches']),(256,4))
        changed=[n for n,p in model.named_parameters() if not torch.equal(before[n],p)]
        for part in ['embed_tokens','layers.0.','lm_head']:self.assertTrue(any(part in n for n in changed))

    def test_bfloat16_cpu_master_resume_preserves_subrounding_updates(self):
        p=torch.nn.Parameter(torch.tensor([1.,-.5],dtype=torch.bfloat16))
        opt=CPUAdamW([p],lr=1.41e-5,weight_decay=.01,fp32_master_state=True)
        for _ in range(5):p.grad=torch.tensor([.2,-.3],dtype=p.dtype);opt.step()
        self.assertFalse(torch.equal(opt.masters[0],p.float()))
        buf=io.BytesIO();torch.save(opt.state_dict(),buf);buf.seek(0)
        q=torch.nn.Parameter(p.detach().clone())
        other=CPUAdamW([q],lr=1.41e-5,weight_decay=.01,fp32_master_state=True)
        other.load_state_dict(torch.load(buf,weights_only=False))
        for _ in range(200):
            for param,optimizer in [(p,opt),(q,other)]:param.grad=torch.tensor([.2,-.3],dtype=param.dtype);optimizer.step()
        self.assertTrue(torch.equal(p,q));self.assertEqual(optimizer_digest(opt.state_dict()),optimizer_digest(other.state_dict()))
        self.assertTrue(all(m.dtype==torch.float32 for m in opt.masters))

    def test_missing_or_corrupted_master_restart_rejected(self):
        p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.bfloat16))
        opt=CPUAdamW([p],lr=1e-5,weight_decay=.01,fp32_master_state=True)
        with self.assertRaises(ValueError):opt.load_state_dict(opt.optimizer.state_dict())
        state=copy.deepcopy(opt.state_dict());state['masters'][0].add_(1)
        with self.assertRaises(ValueError):opt.load_state_dict(state)

    def test_real_disk_receiver_restart_roundtrip(self):
        model=torch.nn.Linear(3,2).to(torch.bfloat16)
        opt=CPUAdamW(model.parameters(),lr=1.41e-5,weight_decay=.01,fp32_master_state=True)
        for p in model.parameters():p.grad=torch.ones_like(p)
        opt.step();expected={n:p.clone() for n,p in model.named_parameters()}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'state.pt'
            digest=verify_receiver_restart(SimpleNamespace(receiver=model,receiver_optimizer=opt),path)
            self.assertEqual(len(digest),64);self.assertFalse(path.exists())
        for n,p in model.named_parameters():self.assertTrue(torch.equal(p,expected[n]))

    def test_observer_transfer_rejects_bad_audit(self):
        from orcd.qwen_observer_client import QwenObserverClient
        from unittest.mock import Mock
        client=QwenObserverClient.__new__(QwenObserverClient)
        client._request=Mock(return_value={'move':{'device':'cpu','rng_unchanged':True,'modes_unchanged':True,'parameters_unchanged':False}})
        client.close=Mock()
        with self.assertRaises(RuntimeError):client.move('cpu')
        client.close.assert_called_once()

    def test_offload_changes_only_device_and_preserves_scores(self):
        # A real CUDA worker additionally executes this roundtrip in its GPU validation.
        from orcd.qwen_observer_worker import FrozenQwenObserver
        import numpy as np
        worker=FrozenQwenObserver.__new__(FrozenQwenObserver)
        worker.model=torch.nn.Linear(3,2).eval().requires_grad_(False);worker.torch=torch;worker.np=np
        inputs=torch.ones(2,3);before=worker.model(inputs).clone()
        with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'empty_cache'):
            audit=worker.move('cpu')
        self.assertTrue(audit['parameters_unchanged']);self.assertTrue(torch.equal(before,worker.model(inputs)))
