import unittest
import torch
from transformers import Gemma2Config,Gemma2ForCausalLM
from stego.qwen_pair import LabelReceiver
class GemmaParity(unittest.TestCase):
    def test_selected_softcapped_logits_match_native_head(self):
        cfg=Gemma2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=16,final_logit_softcapping=0.2,pad_token_id=0)
        cfg._attn_implementation='eager'
        model=Gemma2ForCausalLM(cfg).eval()
        wrapper=LabelReceiver.__new__(LabelReceiver)
        torch.nn.Module.__init__(wrapper)
        wrapper.model=model
        wrapper.set_label_ids([8,12])
        ids=torch.tensor([[0,0,2,3,4],[2,4,5,6,7]])
        mask=ids.ne(0).long()
        pos=(mask.cumsum(-1)-1).masked_fill(mask==0,0)
        with torch.no_grad():
            expected=model(ids,attention_mask=mask,position_ids=pos,use_cache=False).logits[:,-1,[8,12]]
            torch.testing.assert_close(wrapper(ids,mask).logits,expected,rtol=1e-5,atol=1e-6)
