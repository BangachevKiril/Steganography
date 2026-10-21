# Corrected university continuation: additional 7h45m

The user explicitly authorized another 27,900 active seconds on October 2, 2026, continuing the completed corrected experiment from update 530. The original stage is preserved in a remote archive, and its public local results remain unchanged. The older uncorrected run, baseline, probes and unrelated jobs are untouched.

The same frozen corrected source and configuration are used. The separate administrative launcher changes only the training-loop stopping threshold and checkpoint retention for the origin checkpoint 530. Original disk source, config digest, scientific updates, rewards, batch sizes, precision, models and evaluation cohorts are unchanged. The audited L40S launcher still performs exact CUDA toy parity before model loading, full checkpoint restoration verification, layout-preserving PPO activation offload, and unused-cache cleanup before the judge.

Checkpoint530 has elapsed 27,930.404972237535 seconds. The additional segment stops at the fixed cumulative checkpoint clock 55,830.404972237535 seconds, checked between atomic updates, followed by final evaluation. Requeues never reset this threshold. Prior finalization (136.04994407203 seconds after the checkpoint clock) is preserved separately and does not increase the new allowance. The saved elapsed time itself is not edited.

Both optimizer states, random state, data-stream cursors, module modes, PPO/KL state and checkpoint contents are restored unchanged. Updates 531 and 532 are sender PPO updates, completing cycle 76; evaluation 550 is next. Every frozen evaluation uses the same 2,000 university rows and 100 PIQA questions. PIQA delta remains relative to corrected evaluation 0 (8/100), with update 530 (4/100) available as the continuation starting point. C_I is the static observer flag fraction.

The remote completed-stage archive copies public artifacts before restoration can regenerate reports, hardlinks all immutable saved states on the cluster, and has a hash manifest. Checkpoint530 is excluded from live pruning, preserving evaluation 530's original checkpoint path. The new launcher seals CONTINUATION.json under the existing exclusive run lock before removing only the exact original COMPLETE/BUDGET_COMPLETE markers. Repeated activation is idempotent and cannot clear continuation completion markers.

Hardware remains one L40S with 128G RAM and six-hour allocations on mit_normal_gpu,mit_preemptable, using the original environment and checkpointed requeue. Weights and private reasoning never leave the cluster.
