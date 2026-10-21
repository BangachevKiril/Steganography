# Fresh PIQA pilot with supervised decoder bits

This authorized retry compares against `piqa_revised_fresh_20260929T173207Z`.
It starts fresh A/B/I adapters and empty optimizers. It does not load the previous
run's checkpoints. Dataset, model snapshot, prompts, seed, judge, rewards,
generation limits, learning rate, adapter settings, phase schedule, group sizes,
and validation schedule are unchanged.

The sender still receives `C_R - C_I - d` and uses the existing centered GRPO
objective on its sampled reasoning and public answer. No soft probability reward
is introduced. Receiver and interceptor use the same new decoder objective.

For each decoder trajectory, let `r` be sampled final-bit correctness and `m`
the true bit. Reasoning actions use advantage `r - 0.5`, without group centering.
The policy-gradient surrogate retains the existing importance-ratio clipping
at 0.2. It applies only to sampled thought tokens, including a sampled closing
thinking token. Injected boundary tokens and the final sampled bit receive no
policy-gradient term. The final bit receives `-log p(m | input, sampled thoughts)`
over the same two-label support and temperature 0.6 used for generation.
The true bit selects the loss target only; it is never inserted into a decoder
prompt or used to generate its reasoning prefix.

The per-trajectory loss is the sum of the reasoning PG term and the unchanged
0.01-weighted sampled KL term, divided by the existing fixed decoder length
bound 257, plus final-bit cross-entropy with coefficient **1.0**. Cross-entropy
is averaged per trajectory, not divided by 257. KL still applies to all sampled
actions, including the final bit. This is a hybrid objective: CE teaches the final
decision on sampled prefixes; outcome PG supplies credit to discrete reasoning
choices. It does not provide token-specific labels for reasoning quality.

Both decoder conditions retain eight rollouts for each selected answer context.
A/B combine sender and receiver role losses with equal 0.5 weights and take one
optimizer step each; because roles share adapters, receiver updates may also
change subsequent sender behavior. I takes its own update. Ten joint steps
alternate with ten interceptor-only steps, for 100 global steps total.

The isolated GPU canary repeats one sampled trajectory eight times and labels
it with the opposite bit to check the formerly stalled all-wrong case. Its
adapters are discarded; training loads fresh base weights in a new process.
The production initialization audit and checkpoint zero verify empty A/B/I
optimizer states and zero LoRA B matrices.

Reporting keeps joint observations (128 answers), interceptor-only monitoring
(16 answers), and decoder optimization guesses (128 per role) separate. Report
sampled `C_R`, `C_I`, and `d`, zero guesses, accuracy by true bit and receiver
identity, conditional final-bit probabilities, CE, and reasoning PG activity.
The retained within-group reward-variation statistic is counterfactual for hybrid
decoders: zero variation no longer implies no task update. Literal disclosure
flags remain diagnostics, not evidence of secrecy or additional reward terms.

Validation remains 32 unique held-out questions with one rollout at steps 20,
40, 60, 80, and 100. Compare these with the previous run, including zero rates;
a more balanced guess distribution alone is not evidence of communication.
Resume preemptions only from complete checkpoints within this new run whose
configuration digest matches. Never reuse a checkpoint from the comparison run.
