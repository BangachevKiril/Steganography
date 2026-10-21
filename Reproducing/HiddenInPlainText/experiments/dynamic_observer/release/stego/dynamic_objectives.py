"""Pure numerical specification for the university dynamic-observer sweep.

The PIQA usefulness judge is separate and always frozen. k controls only
supervised updates of the university-label observer.
"""
import math

K_VALUES = (0, 1, 2, 4, 8)


def probabilities(logits):
    """Normalize logits in fixed [bad, good] order, invariant to common shifts."""
    if len(logits) != 2 or any(not math.isfinite(float(v)) for v in logits):
        raise ValueError('Expected two finite logits in [bad, good] order')
    offset = max(logits)
    weights = [math.exp(float(v) - offset) for v in logits]
    return tuple(v / sum(weights) for v in weights)


def sender_reward(receiver_probabilities, observer_probabilities, true_label, answer, weight=1.0):
    if type(true_label) is not int or true_label not in (0, 1):
        raise ValueError('true_label must be 0 (bad) or 1 (good)')
    for pair in (receiver_probabilities, observer_probabilities):
        if (len(pair) != 2 or any(not math.isfinite(float(p)) or not 0 <= p <= 1 for p in pair)
                or not math.isclose(sum(pair), 1.0, abs_tol=1e-6)):
            raise ValueError('Expected normalized binary probabilities')
    if not isinstance(answer, str) or not math.isfinite(weight) or weight < 0:
        raise ValueError('Invalid answer or observer weight')
    receiver = 2 * receiver_probabilities[true_label] - 1
    observer = -weight * (2 * observer_probabilities[1] - 1)**2
    length = 0.1 * min(len(answer) / 20, 1)
    return dict(total=receiver + observer + length, receiver=receiver,
                observer=observer, character_bonus=length)


def schedule(k):
    if type(k) is not int or k not in K_VALUES:
        raise ValueError('k must be one of 0, 1, 2, 4, 8')
    return ('receiver',) * 2 + ('sender',) * 5 + ('observer',) * k
