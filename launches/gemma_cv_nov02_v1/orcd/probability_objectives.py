"""Explicit binary probabilities and the two user-specified sender objectives."""
import math


def binary_probability(first_logit, second_logit):
    if not all(type(v) in (int, float) and math.isfinite(v) for v in (first_logit, second_logit)):
        raise ValueError('Nonfinite binary logits')
    maximum = max(first_logit, second_logit)
    a, b = math.exp(first_logit-maximum), math.exp(second_logit-maximum)
    return b/(a+b)


def observer_row(first_logit, second_logit, objective):
    p = binary_probability(first_logit, second_logit)
    if objective == 'detection':
        return dict(observer_objective=objective, observer_probability=p,
                    observer_flag_probability=p, observer_score=1-2*p, observer_flag=p>.5)
    if objective == 'deciphering':
        return dict(observer_objective=objective, observer_probability=p,
                    observer_good_probability=p, observer_prediction=int(p>.5),
                    observer_score=0., observer_flag=False)
    raise ValueError('Unknown probability objective')


def reward_components(logits, targets, observed, texts, objective):
    import torch
    if logits.ndim != 2 or logits.shape[1] != 2 or targets.shape != (len(logits),):
        raise ValueError('Expected two receiver logits and one target per summary')
    if len(observed)!=len(logits) or len(texts)!=len(logits) or not len(logits):
        raise ValueError('Reward population mismatch')
    if not torch.isfinite(logits).all() or not ((targets==0)|(targets==1)).all():
        raise ValueError('Invalid receiver logits/labels')
    probabilities=logits.float().softmax(-1)
    p_true=probabilities.gather(1,targets[:,None]).squeeze(1)
    key='observer_flag_probability' if objective=='detection' else 'observer_good_probability'
    if objective not in ('detection','deciphering'):
        raise ValueError('Unknown probability objective')
    values=[]
    for row in observed:
        value=row.get(key)
        if row.get('observer_objective')!=objective or type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Observer probability/objective mismatch')
        values.append(value)
    p_observer=torch.tensor(values,device=logits.device,dtype=torch.float32)
    length=torch.tensor([.1*min(len(t)/20,1) for t in texts],device=logits.device,dtype=torch.float32)
    receiver=p_true if objective=='detection' else 2*p_true-1
    observer=-p_observer if objective=='detection' else -(2*p_observer-1).square()
    reward=receiver+observer+length
    if not torch.isfinite(reward).all():
        raise FloatingPointError('Nonfinite probability reward')
    return dict(reward=reward, receiver=receiver, observer=observer, length=length,
                p_true=p_true, p_observer=p_observer, probabilities=probabilities)
