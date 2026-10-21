# Fresh RL pilot with the revised sender prompt

This is a new 100-step experiment. Communicators A/B and interceptor I start
with fresh LoRA adapters on the original frozen Qwen3-4B base; no trained
checkpoint or optimizer state from the previous pilot is loaded. The judge
remains frozen. A separate empty run directory and a committed step-zero
checkpoint establish the initialization. Preemption resumes only this run's
own latest complete checkpoint.

The sender system message is exactly the revised prompt tested in the frozen
comparison `20260929T162649Z`. The configuration stores its full text and hash.
Receiver, interceptor and judge prompts, payload fields, rewards, sampling,
token budgets, training seeds and optimizer settings remain unchanged. There
are eight sender rollouts per context, 16 contexts per joint step, 256 private
thinking tokens, and at most 128 public answer tokens. Ten joint updates
alternate with ten interceptor-only updates, for five cycles. Validation uses
32 random distinct questions and one rollout at steps 20, 40, 60, 80 and 100.

## Training measurements

Joint-step C_R, C_I and d use the 128 scored sender answers before that step's
updates. During interceptor-only steps, all three metrics use the existing 16
sender answers, the first of the eight already-sampled interceptor decisions
per answer, and added receiver/judge inference. This monitoring inference has
an isolated deterministic Torch RNG stream and judge cache; its results do
not alter training rewards, samples or updates. The original interceptor
optimization accuracy over all 128 decisions is recorded separately.

Counts and denominators accompany each rate. Reports keep joint and
interceptor-only phases separate rather than pooling their unequal samples.
Additional passive diagnostics record sender reward variation within groups,
literal disclosure flags, and receiver/interceptor performance on answers
that pass both quality checks without detected disclosure. Literal flags are
imperfect text diagnostics and do not affect rewards.

Only metrics at or below the highest complete checkpoint are reportable.
The monitor preserves per-step metrics and reports the first completed step,
each completed ten-step phase, validation milestones, useful measured timing,
completion and actionable failures. Routine unchanged status stays quiet.
The model's private reasoning is never included in displayed reports.
