# Frozen decoder bias comparison

This diagnostic compares vanilla Qwen3-4B with the receiver and interceptor adapters saved at global step 40 of `piqa_revised_fresh_20260929T173207Z`. It runs in a separate Slurm allocation and never updates, pauses, or resumes the training job.

Use all 32 public answers from that run's step-40 validation, without filtering. Keep each question, answer, public random string, sender/receiver identities, and decoder prompt fixed across model conditions. Communicator A receives B's answers, communicator B receives A's answers, and interceptor I receives all answers. Ground-truth bits are used only for scoring and are absent from decoder inputs. Keep saved judge/format outcomes for subgroup reporting; do not regenerate answers or judge verdicts.

For each answer and decoder role, generate eight independent private reasoning/guess replicates under both vanilla and trained conditions, matching each replicate's initial sampling seed across conditions. This yields 1,024 decoder trajectories on 32 answer contexts. Preserve the training sampler's 256-token thinking limit, forced closure, temperature 0.6, and binary final-output constraint. No optimizer is constructed and every parameter remains frozen. Private reasoning text and token sequences are neither displayed nor saved by this diagnostic.

The recorded probability of 0 is the exact constrained final-token probability conditional on that replicate's sampled private reasoning. Its average across eight replicates estimates the probability of 0 after marginalizing over the model's reasoning. Matched seeds do not imply identical reasoning between vanilla and trained models. Replicates are not independent questions; summarize across the 32 contexts and show per-context values and paired differences.

Report sampled zero counts, accuracy, mean probability of zero, counts by true bit, receiver identity, and the saved accepted subset. A trained-minus-vanilla shift identifies a change associated with the trained adapters on these fixed answers. This diagnostic cannot by itself separate effects of the base model from the unchanged prompt, identify which training mechanism caused a shift, or establish the existence or absence of a hidden code.

Input and source hashes, the checkpoint's completion marker and matching configuration digest, and adapter provenance are recorded in the manifest. Atomic batch commits allow preemption to resume this diagnostic without repeating completed batches. Keep its launch record and result directory separate from the main training run.
