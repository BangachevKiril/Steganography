# Vanilla answer rejection baseline

Sample 100 distinct PIQA training rows uniformly with seed 20261028, without
filtering or rewriting their goal text. Give a frozen pretrained Qwen3-4B a
plain helpful-answer prompt requesting 2–4 sentences. Provide only the question;
include no hidden bit, public random string, communication role, or game reward.
No LoRA adapter or trained checkpoint is loaded, and no optimization occurs.

Use the existing experiment's sampler and settings: temperature 0.6, full
vocabulary support, up to 256 private thinking tokens and 128 answer tokens.
Pass only the extracted final answer and original question to the exact frozen
Qwen3-4B judge prompt used in the ongoing pilot. Judge once per answer; retain
every verdict. Use the unchanged sentence-format checker separately.

Report judge-only rejection, format failure, their overlap, and their union
(the pilot's `d`), all out of 100. Preserve all answers and verdicts, plus
separate private diagnostics. Record forced thinking closure separately for
answer generation and judging, and final-answer truncation at the token cap.
These diagnostics help interpret a high rate; the baseline alone does not
establish whether every judge decision is correct.

Run on a separate preemptable H100 allocation using
`orcd/rejection_baseline.slurm`. The existing training job continues independently.
Save each completed batch so preemption does not require redoing finished batches.
Use a separate configuration, immutable source release, run directory, and
`launches/rejection_baseline_latest.json` launch record.
