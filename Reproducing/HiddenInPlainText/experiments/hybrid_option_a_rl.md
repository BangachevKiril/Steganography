# Fresh PIQA hybrid pilot with receiver prompt A

The user selected option A and authorized this fresh 100-step launch on
2026-09-30. Receiver A/B use the exact `concise_plus_prior` system prompts from
`results/decoder_prompts_step40_20260930T005259Z/PROMPTS.json`. The interceptor
keeps the original prompt from that same verified comparison. The sender keeps
the revised prompt from the previous hybrid run. The frozen `PROMPTS.json`
records all role-specific system messages and their SHA-256 hashes.

Use `configs/piqa_hybrid_option_a.json` and `orcd/hybrid_option_a_run.slurm` from
the new immutable release. Start new A/B/I adapters with empty optimizer state;
do not resume the stopped hybrid pilot or any earlier run. Preserve the stopped
pilot at checkpoint 23 and its launch record separately.

The hybrid objective remains reasoning-only policy gradient with correctness
minus 0.5, plus two-label final-bit cross-entropy of weight 1 per example. Sender
GRPO remains eight rollouts per each of 16 contexts. All sampled reasoning and
generation settings, reward definitions, reference KL, learning rate, and role
weights remain as in the previous hybrid pilot.

The two approved efficiency changes are implemented as described in
`experiments/single_decoder_minibatches.md`: accumulate gradients in batches of
four trajectories and use one decoder rollout per answer. Joint steps train on
128 sender answers, reusing their 128 receiver and 128 interceptor scoring
trajectories. Interceptor-only steps train on 16 guesses on 16 fresh answers;
the passive receiver/judge probe measures those same answers. Keep optimization
and monitoring metrics separately labeled even when their I counts coincide.

Ten joint steps alternate with ten interceptor-only steps. Validation at steps
20, 40, 60, 80, and 100 uses 32 questions and a single rollout. Report C_R, C_I,
and d with raw counts and decoder zero rates, per-bit and per-identity accuracy,
CE, conditional final-bit probabilities, actual reasoning PG activity, sender
reward variation, and literal disclosure diagnostics. Singleton decoder group
variation is zero by construction and does not indicate an inactive hybrid loss.

Before production, run the discarded GPU canary, including ragged batched
probability checks and production-budget minibatch updates. Production starts
in a separate process after smoke passes. Checkpoint zero and INITIALIZATION.json
verify fresh state. Resume only this new run's complete, digest-matching
checkpoints after preemption. Never report metrics beyond a verified committed
checkpoint. Public monitoring excludes private reasoning and full training state.

This run changes both receiver wording and decoder sampling relative to the
stopped hybrid run; comparisons cannot isolate either effect by itself. A more
balanced distribution of guesses alone does not establish successful decoding
or a receiver advantage over the interceptor.
