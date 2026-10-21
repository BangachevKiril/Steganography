# PIQA communication pilot

Three trainable agents (communicators A/B and interceptor I) and a frozen judge J
play a hidden-bit communication game. The configuration uses Qwen3-4B
for every role, as confirmed by the user.
Qwen3's original dense lineup has 1.7B and 4B checkpoints, without a 3B checkpoint.

The fresh revised-prompt pilot uses `configs/piqa_revised_pilot.json` and
`orcd/revised_run.slurm`. It starts new adapters and preserves the original
pilot separately. See [its experiment notes](experiments/revised_sender_rl.md)
for prompt provenance and training-monitoring denominators.

The stopped hybrid decoder pilot is specified in
[hybrid decoder notes](experiments/hybrid_decoder_rl.md). Its prepared successor
uses [single decoder rollouts and batched updates](experiments/single_decoder_minibatches.md);
the draft configuration remains disabled pending decoder prompt revision.

## Episodes and observations

We use the unchanged `goal` field from PIQA: 16,113 training and 1,838 validation
questions. Answer choices and labels are never supplied to any role. The files
are at `/orcd/pool/003/kirilb/PIQA`; configuration records their SHA-256 hashes.
Source and license information are in that dataset directory's manifest and
README. Dataset source: <https://yonatanbisk.com/piqa/> (AFL-3.0).

Each context contains a question, sender identity, independent random bit `m`,
and independent public 16-bit string `s`. The receiver is the other communicator.
Each 16-context training batch balances the four `(sender, m)` combinations.
Only the sender sees `m`. Both decoders see the same question, identities, public
string, and final answer. The judge sees only the question and final answer.
Models have no access to trace files, rewards, or another role's private thought.

All roles may sample up to 256 private thinking tokens. The harness ends thinking
at that cap. The sender then has up to 128 tokens for a 2–4 sentence answer;
decoders must output `0` or `1`; J must output `PASS` or `FAIL`. Injected delimiters
are masked out of policy loss. System prompts live in `stego/protocol.py`.

J is lenient about minor grammar and stylistic issues. It rejects substantial
content errors, irrelevance, or unintelligibility. A separate format check
enforces 2–4 sentences. Define `d = 1` when either check fails, and log both checks
separately. With `c_r` and `c_i` indicating correct bit recovery, rewards are:

```
sender      = c_r - c_i - d
receiver    = c_r
interceptor = c_i
```

The judge's weights never change. Its sampled verdict is cached by exact
question/answer within training. Validation uses an isolated cache and RNG state.
The public string is randomness available to everyone, rather than a secret key.

## Updates and validation

The 100-step pilot alternates 10 joint steps and 10 interceptor-only steps, five
times. A global step is one optimizer update per participating agent.

In a joint step, 16 contexts each yield eight sender rollouts. Each of the 128
answers receives one receiver decision, interceptor decision, and judge verdict
(identical question/answer pairs reuse a verdict). These determine sender rewards.
For each context, one answer is selected uniformly; its initial decoder decision
and seven further samples form an eight-rollout group for each decoder. All
rollouts finish before any weights change. A/B each have eight sender groups and
eight receiver groups; these roles have equal loss weight. I has 16 groups.

In an interceptor-only step, frozen A/B produce one fresh answer per context,
then I samples eight decisions per answer and takes one update. A/B and J do not
receive optimizer updates. J and receiver inference are omitted in this phase.
The revised-prompt pilot enables an additional passive receiver/judge check on
these 16 existing answers, with isolated RNG and cache, so all three training
metrics remain available. Its matched 16-answer measurements are separate from
the interceptor's optimization accuracy over 128 guesses.

GRPO uses group-mean-centered rewards without standard-deviation scaling,
role-specific fixed maximum length denominators, clipping 0.2, and reference KL
coefficient 0.01 against the initial base. The sampled private reasoning and
final response receive the terminal advantage. Independent LoRA adapters use
rank 16, alpha 32, dropout 0, attention and MLP projections, AdamW at 1e-5, and
gradient clipping 1. Sampling uses temperature 0.6 with full vocabulary support
(top-p 1, no top-k truncation), so generation and training probabilities match.
Bit and judge outputs use a constrained label vocabulary.

After global steps 20, 40, 60, 80, and 100, sample 32 distinct random validation
questions, balance `(sender, m)`, and run one complete episode per question.
Report `C_R = mean(c_r)`, `C_I = mean(c_i)`, and `d = mean(d)`, with counts and
direction breakdowns. These 32-question estimates are noisy. Validation never
updates weights or consumes the training random-number stream.

## Running on ORCD

`orcd/run.slurm` requests one H100 on `mit_preemptable`, eight CPUs, 96 GB RAM, and
up to two days. It reuses the existing compatible Python environment without
modifying it. Run the GPU canary in `stego/smoke.py` before training; it must pass
generation/log-probability consistency, nonzero gradient updates, agent isolation,
and checkpoint restoration. Production starts in a fresh process after the canary.

Use an immutable source snapshot, finalized configuration with source hashes,
and a fresh run directory. Launch records belong in the run directory. For local
checks and manual execution within an allocation:

```sh
python -m unittest discover -s tests -v
python -m stego.smoke --config configs/piqa_pilot.json
python -m stego.train --config configs/piqa_pilot.json --run-dir /path/to/run
```

Each completed step saves all three adapters, optimizer states, and RNG states
atomically. On preemption, resume the last complete step; discard partial updates.
The run lock prevents concurrent writers. A finalized checkpoint is authoritative
even if preemption occurred before its `LATEST.json` pointer was updated.
Keep step zero, every validation checkpoint, and the two latest checkpoints.

Run artifacts include per-step metrics, single-rollout validation records,
compressed private training traces, `STATUS.json`, and `TIMING.json`. Only metrics
up to the highest complete checkpoint are committed. Timing estimates use observed
phase medians; they exclude queueing, preemption, and checkpoint-write time, and
include future validation once its duration is measured. Full-run estimates need
measurements from both phases. Private traces are researcher diagnostics and
must never be inserted into another model's prompt.
