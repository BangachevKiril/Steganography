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

## Verified memory measurements

Frozen probe only; these peaks do not establish PPO backward-pass or full evaluation capacity. GiB denotes 2^30 bytes.

| Phase | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|
| loading_sender_receiver | 10.01 | 10.03 |
| loading_observer | 18.64 | 18.78 |
| restoring_checkpoint | 19.00 | 19.04 |
| frozen_forward_smoke | 21.74 | 21.81 |
| observer_parity:evaluation50 | 26.03 | 26.68 |
| observer_parity:training_update82 | 25.84 | 26.68 |
| verifying_final_state | 19.03 | 26.68 |

Public rows were independently matched to all original answers, labels, batch boundaries, scores and flags. The completion/results/manifest and staged script hashes were verified. Differences include sizeable score changes and individual flag flips despite similar aggregate flag fractions. This test establishes successful frozen execution and exact checkpoint restoration, not numerical equivalence or approval to move training.
