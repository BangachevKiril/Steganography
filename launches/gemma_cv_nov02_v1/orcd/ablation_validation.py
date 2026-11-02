"""Hash nested optimizer state for exact save/restore validation."""
import hashlib
import torch

def optimizer_digest(state):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode())
            t=x.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
            for block in t.split(8*1024*1024):h.update(block.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for item in x:visit(item)
        else:h.update(repr(x).encode())
    visit(state)
    return h.hexdigest()

def verify_receiver_restart(trainer,path):
    import os
    from stego.university_training import sha256
    optimizer=trainer.receiver_optimizer
    expected=optimizer_digest(optimizer.state_dict())
    model={n:p.detach().cpu() for n,p in trainer.receiver.named_parameters() if p.requires_grad}
    torch.save(dict(receiver_trainable=model,receiver_optimizer=optimizer.state_dict()),path)
    del model
    with path.open('rb') as stream:os.fsync(stream.fileno())
    expected_file=sha256(path)
    restored=torch.load(path,map_location='cpu',weights_only=False)
    if sha256(path)!=expected_file:raise ValueError('Validation restart file changed')
    named=dict(trainer.receiver.named_parameters())
    with torch.no_grad():
        optimizer.masters[0].add_(1)
        next(iter(named.values())).add_(1)
        for name,value in restored['receiver_trainable'].items():named[name].copy_(value)
    optimizer.load_state_dict(restored['receiver_optimizer'])
    if optimizer_digest(optimizer.state_dict())!=expected:
        raise ValueError('Full receiver optimizer/master checkpoint roundtrip failed')
    path.unlink()  # Disposable validation restart; production restarts use standard hash-committed state.pt.
    return expected_file


def receiver_update_audit(receiver,before):
    """Check actual parameter identity, including Qwen's tied embedding/head."""
    named=dict(receiver.named_parameters())
    if set(named)!=set(before):raise ValueError('Receiver parameter population changed')
    if not all(p.requires_grad for p in named.values()):raise ValueError('Full receiver contains frozen parameters')
    changed={n for n,p in named.items() if not torch.equal(before[n],p.detach().cpu())}
    canonical={id(p):n for n,p in named.items()}
    model=receiver.model
    embedding=model.get_input_embeddings().weight
    head=model.get_output_embeddings().weight
    embedding_name=canonical[id(embedding)];head_name=canonical[id(head)]
    result=dict(embedding_parameter=embedding_name,output_head_parameter=head_name,
        input_output_weights_tied=embedding is head,
        embedding_updated=embedding_name in changed,output_head_updated=head_name in changed,
        first_backbone_layer_updated=any('layers.0.' in n for n in changed),
        changed_unique_parameters=len(changed),total_unique_parameters=len(named))
    result['passed']=all(result[k] for k in ('embedding_updated','output_head_updated','first_backbone_layer_updated'))
    return result
