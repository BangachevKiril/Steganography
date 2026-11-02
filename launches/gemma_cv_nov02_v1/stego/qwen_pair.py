"""Qwen3 role adapters, retaining binary CE and original raw-logit rewards."""
from types import SimpleNamespace
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .university_baseline import TASK_PROMPT


def sender_prompt(tokenizer, cv):
    return tokenizer.apply_chat_template([{'role':'user','content':TASK_PROMPT+'\n"'+cv+'"'}], tokenize=False, add_generation_prompt=True, enable_thinking=False)


class ReceiverTokenizer:
    def __init__(self, path, instruction):
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.tokenizer.padding_side='left'
        self.instruction=instruction
        self.label_ids=[self.tokenizer.encode(x,add_special_tokens=False) for x in ['BAD','GOOD']]
        if any(len(x)!=1 for x in self.label_ids):
            raise ValueError('BAD/GOOD must each be a single token')
        self.label_ids=[x[0] for x in self.label_ids]
        if self.label_ids != [33767,46775]:
            raise ValueError('Unexpected Gemma label-token IDs')
        for text in ['short CV', 'A candidate with several years of experience.']:
            prompt=self.prompt(text)
            for label,idx in zip(['BAD','GOOD'],self.label_ids):
                prefix=self.tokenizer.encode(prompt,add_special_tokens=False)
                combined=self.tokenizer.encode(prompt+label,add_special_tokens=False)
                if combined != prefix+[idx]:
                    raise ValueError('Label tokenization changes at assistant boundary')

    def prompt(self,text):
        return self.tokenizer.apply_chat_template([
            {'role':'user','content':self.instruction+'\n\n'+json.dumps({'summary':text},ensure_ascii=False)}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def __call__(self,texts,**kwargs):
        kwargs.pop('max_length',None)
        kwargs['truncation']=False
        kwargs['add_special_tokens']=False
        result=self.tokenizer([self.prompt(t) for t in texts],**kwargs)
        if result['input_ids'].shape[1]>512:
            raise ValueError('Receiver input exceeds 512-token bound; never truncate label prefix')
        return result


class LabelReceiver(torch.nn.Module):
    def __init__(self,path,full_training=False,rank=16):
        super().__init__()
        self.model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=torch.bfloat16,
            attn_implementation='eager',local_files_only=True)
        if full_training:
            self.model.requires_grad_(True)
        else:
            from peft import LoraConfig, get_peft_model
            self.model=get_peft_model(self.model,LoraConfig(r=rank,lora_alpha=rank,lora_dropout=.05,bias="none",task_type="CAUSAL_LM",target_modules=["q_proj","v_proj"]))
        self.model.config.use_cache=False
        self.label_ids=None

    def set_label_ids(self,ids):
        self.label_ids=tuple(ids)

    def forward(self,input_ids,attention_mask,**kwargs):
        if self.label_ids is None:raise ValueError('Missing verified label IDs')
        # Position IDs must match generation semantics for mixed left-padded batches.
        position_ids=attention_mask.long().cumsum(-1)-1
        position_ids.masked_fill_(attention_mask==0,0)
        base=self.model.get_base_model() if hasattr(self.model,"get_base_model") else self.model
        output=base.model(input_ids=input_ids,attention_mask=attention_mask,
            position_ids=position_ids,use_cache=False,return_dict=True)
        positions=torch.arange(attention_mask.shape[1],device=attention_mask.device)
        last=positions.expand_as(attention_mask).masked_fill(attention_mask==0,-1).amax(1)
        hidden=output.last_hidden_state[torch.arange(input_ids.shape[0],device=input_ids.device),last]
        # Equivalent selected LM-head logits without allocating B*S*V logits.
        weights=base.lm_head.weight[list(self.label_ids)]
        logits=torch.nn.functional.linear(hidden,weights)
        cap=getattr(base.config,'final_logit_softcapping',None)
        if cap is not None:
            logits=torch.tanh(logits/cap)*cap
        logits=logits.float()
        if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite receiver logits')
        return SimpleNamespace(logits=logits)


class CPUAdamW:
    """Full FP32 AdamW with CPU masters/states and exact gradient copies.

    All model parameters remain trainable. CPU execution is capacity management,
    not quantization or an optimizer change; CPU/CUDA bitwise parity is not claimed.
    """
    def __init__(self,params,lr,weight_decay,fp32_master_state=False):
        self.fp32_master_state=fp32_master_state
        self.params=list(params)
        self.masters=[torch.nn.Parameter(p.detach().to(device="cpu",dtype=torch.float32).clone(),requires_grad=True) for p in self.params]
        self.optimizer=torch.optim.AdamW(self.masters,lr=lr,weight_decay=weight_decay,foreach=False)
    @property
    def state(self):return self.optimizer.state
    def zero_grad(self):
        for p in self.params:p.grad=None
        self.optimizer.zero_grad(set_to_none=True)
    def step(self):
        for p,m in zip(self.params,self.masters):
            if p.grad is None:m.grad=None
            else:
                if not torch.isfinite(p.grad).all():raise FloatingPointError('Nonfinite receiver gradient')
                m.grad=p.grad.detach().to(device="cpu",dtype=torch.float32)
        self.optimizer.step()
        with torch.no_grad():
            for p,m in zip(self.params,self.masters):
                if not torch.isfinite(m).all():raise FloatingPointError('Nonfinite CPU receiver parameter')
                p.copy_(m)
        self.zero_grad()
    def state_dict(self):
        state=self.optimizer.state_dict()
        if not self.fp32_master_state:return state
        return dict(format='fp32_cpu_master_adamw_v1',optimizer=state,
                    masters=[m.detach() for m in self.masters])
    def load_state_dict(self,state):
        if self.fp32_master_state:
            if state.get('format')!='fp32_cpu_master_adamw_v1' or len(state.get('masters',[]))!=len(self.masters):
                raise ValueError('Full receiver restart requires all FP32 master weights')
            with torch.no_grad():
                for p,m,saved in zip(self.params,self.masters,state['masters']):
                    if saved.dtype!=torch.float32 or saved.shape!=m.shape or not torch.isfinite(saved).all():
                        raise ValueError('Invalid receiver FP32 master tensor')
                    if not torch.equal(saved.to(dtype=p.dtype),p.detach().cpu()):
                        raise ValueError('Receiver master/model restart mismatch')
                    m.copy_(saved)
            self.optimizer.load_state_dict(state['optimizer'])
        else:
            with torch.no_grad():
                for p,m in zip(self.params,self.masters):m.copy_(p.detach().cpu())
            self.optimizer.load_state_dict(state)
