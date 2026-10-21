# L40S university checkpoint/observer diagnostic

Restored committed update 82; no updates, generation or backward passes.
Sender, receiver, optimizers, stream positions, RNG and model modes matched the checkpoint exactly before and after the probe.

| Population | Reference flagged | L40S flagged | Flag changes | Max absolute score change | Max absolute 0.2×score change |
|---|---:|---:|---:|---:|---:|
| evaluation50 | 208/2000 | 207/2000 | 29 | 1.43430638 | 0.286861277 |
| training_update82 | 44/192 | 44/192 | 2 | 0.63225361 | 0.126450722 |

Exact flag agreement and score-error quantiles are recorded separately in RESULTS.json. Public row files include every flag change and both scores. No acceptance tolerance was imposed.

A flag change also switches the receiver-reward gate; the 0.2×score comparison alone is not the full reward change.

This is not bit-identical equivalence or a training authorization.
No backward pass or optimizer step was run; training memory/runtime compatibility remains untested.
C_I is the static observer flag fraction, not ground-truth transmission-detection accuracy.
Hardware differs from reference; the original bitsandbytes0.43.3 is retained.
