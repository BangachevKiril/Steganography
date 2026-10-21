# Minibatches and single decoder rollouts

Implementation prepared after the user stopped the hybrid pilot at checkpoint
23. The stopped run, its configuration, and its results remain unchanged. No
training or monitoring is restarted by this change. The prepared configuration
is `configs/piqa_hybrid_single_decoder_draft.json`; it deliberately has
`launch_ready: false` while decoder prompts still await revision. Freeze the
revised prompts and source, and validate GPU behavior before a future launch.

## Sampling and updates

Keep 16 balanced question/sender/bit contexts per step and eight sender rollouts
per context during joint training. The receiver and interceptor each already
produce one guess on every sender answer for scoring. Reuse those exact sampled
trajectories for decoder training, without generating another decoder rollout.
Thus each joint step still trains on 128 sender trajectories, 128 receiver
trajectories, and 128 interceptor trajectories. Decoder observations now cover
128 answer contexts once each, rather than 16 selected answers eight times each.
All scoring and sampling finish before any adapter update.

In an interceptor-only step, generate 16 sender answers and one interceptor
rollout per answer, giving **16 optimization examples rather than 128**. The
matched receiver/judge monitoring pass uses these same answers and I guesses.
The optimization and monitoring measurements remain labeled separately even
when their interceptor counts coincide. Do not increase the number of contexts
or optimizer steps to compensate for the reduced decoder sample count.

| Per global step | Previous joint | New joint | Previous I-only | New I-only |
|---|---:|---:|---:|---:|
| Sender trajectories generated | 128 | 128 | 16 | 16 |
| Receiver trajectories generated | 240 | 128 | 16 | 16 |
| Interceptor trajectories generated | 240 | 128 | 128 | 16 |
| Judge trajectories, at most | 128 | 128 | 16 | 16 |
| Decoder examples optimized, R / I | 128 / 128 | 128 / 128 | 0 / 128 | 0 / 16 |

The receiver rows during I-only steps are passive monitoring. Cached identical
question/answer judgments can reduce the judge generation counts.

Sender GRPO is unchanged. Each decoder trajectory retains reasoning-only policy
gradient with correctness minus 0.5 and final-bit two-label CE with coefficient
1.0. There is no centering of singleton decoder rewards. Decoder singleton
groups are rejected with the legacy GRPO objective because their centered
advantage would always be zero. The retained within-group variation diagnostic
is consequently zero for singleton hybrid decoders and does not measure their
actual update activity.

A/B each retain equal 0.5 sender/receiver loss weights. I uses its mean decoder
loss. Prompts, temperature, token budgets, KL, learning rate, 10/10 alternating
phases, and validation settings are unchanged by these implementation changes.
Future prompt changes must be recorded separately. The controller retains the
old answer-selection RNG draws so removal of decoder resampling does not itself
change the separate Python RNG stream used for future episode selection.

## Batched gradient accumulation

The prepared configuration uses `train_microbatch_size: 4`. Flatten the weighted
trajectory groups and score up to four trajectories in a single policy forward,
plus a single frozen-reference forward when KL is enabled. Left padding, explicit
attention masks and position IDs preserve each trajectory's causal context.
Only real sampled action positions contribute; padding and forced boundaries
are excluded. Models supporting `logits_to_keep` project the union of required
prediction positions, then each row selects its own positions.

Sum the already normalized trajectory losses within a microbatch and perform
one backward pass. Accumulate gradients across microbatches, then clip and take
exactly one optimizer step per participating adapter. There is no extra division
by microbatch size, including the final partial batch. This is batched gradient
accumulation, not additional optimizer updates on parts of a rollout group.

For the configured counts, this reduces policy forward/backward calls during a
joint step from 384 to 96, and reference forward calls from 384 to 96. Combined
with single rollouts, an I-only step uses four policy forward/backward batches
instead of 128 individual examples. Generation removes 224 decoder sequences
per joint step and 112 per interceptor-only step. These are operation counts,
not measured wall-clock speedups; GPU throughput and peak memory still need a
benchmark. No GPU job is launched by preparing this implementation.

## Compatibility and validation

Omitting `decoder_rollouts_per_answer` retains the legacy grouped decoder
sampling. The archived hybrid config keeps its original hashes and batch size.
The draft cannot start training until `launch_ready` is changed after preparation.
An existing checkpoint cannot be silently resumed with a changed configuration.

CPU tests cover minibatch-versus-singleton losses and parameter updates, ragged
sequences, masks, mixed role weights, KL, final partial batches, true batching,
and one optimizer step. Controller tests verify exact scoring-rollout reuse,
generation counts, bit targets, role weights, unchanged sender rewards and
question RNG stream, and 32-question single-rollout validation. Reports read real
optimization denominators and record decoder rollouts and answer-context counts.
The isolated GPU smoke test honors the configured training microbatch size.
