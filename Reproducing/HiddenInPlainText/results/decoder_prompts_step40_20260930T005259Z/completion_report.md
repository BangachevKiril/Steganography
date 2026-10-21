# Decoder prompt comparison: verified results and interpretation

The concise prompt with an explicit equal prior substantially reduces receiver zero bias in both vanilla and frozen checkpoint-40 models. It does not improve overall sampled decoding accuracy on these saved answers. Adding the same prior to the original prompt increases the bias, and adding uncertainty guidance to concise + prior largely reverses its benefit. This is evidence of sensitivity to wording and context, not a universal correction from stating the prior.

There are 32 saved checkpoint-40 answers and eight matched-seed replicates per answer in each role/model/prompt cell: 256 decisions, with 128 for each true bit. Four new prompts contribute 4,096 new decisions; the original prompt contributes 1,024 verified reused decisions. Vanilla means no trained adapter, and trained means the frozen checkpoint-40 adapters. No training, answer generation, or judge scoring was rerun.

Mean P(0) is the conditional final-label probability at temperature 0.6, averaged over the eight sampled thinking paths per context. These are 32 answer contexts, not 256 independent questions. Every decision hit the 256-token thinking cutoff. This comparison does not isolate a causal effect of that cutoff.

| Prompt | Role | Model | Mean P(0) | Zeros /256 | Correct /256 | Correct bit 0 /128 | Correct bit 1 /128 | Mixed groups /32 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Original (reused control) | receiver | vanilla | 0.8240 | 220 | 136 | 114 | 22 | 21 |
| Original (reused control) | receiver | trained | 0.8663 | 224 | 136 | 116 | 20 | 16 |
| Original (reused control) | interceptor | vanilla | 0.6826 | 175 | 127 | 87 | 40 | 27 |
| Original (reused control) | interceptor | trained | 0.7076 | 185 | 141 | 99 | 42 | 23 |
| Original + equal prior | receiver | vanilla | 0.9256 | 233 | 143 | 124 | 19 | 11 |
| Original + equal prior | receiver | trained | 0.9223 | 232 | 142 | 123 | 19 | 11 |
| Original + equal prior | interceptor | vanilla | 0.7428 | 190 | 144 | 103 | 41 | 25 |
| Original + equal prior | interceptor | trained | 0.7803 | 203 | 139 | 107 | 32 | 19 |
| Concise | receiver | vanilla | 0.7714 | 197 | 125 | 97 | 28 | 19 |
| Concise | receiver | trained | 0.7473 | 191 | 121 | 92 | 29 | 17 |
| Concise | interceptor | vanilla | 0.8769 | 223 | 135 | 115 | 20 | 16 |
| Concise | interceptor | trained | 0.9034 | 234 | 132 | 119 | 13 | 10 |
| Concise + equal prior | receiver | vanilla | 0.5399 | 136 | 126 | 67 | 59 | 27 |
| Concise + equal prior | receiver | trained | 0.5483 | 141 | 135 | 74 | 61 | 23 |
| Concise + equal prior | interceptor | vanilla | 0.7656 | 201 | 127 | 100 | 27 | 23 |
| Concise + equal prior | interceptor | trained | 0.8216 | 212 | 146 | 115 | 31 | 16 |
| Concise + prior + uncertainty | receiver | vanilla | 0.8608 | 222 | 126 | 110 | 16 | 9 |
| Concise + prior + uncertainty | receiver | trained | 0.8631 | 221 | 125 | 109 | 16 | 13 |
| Concise + prior + uncertainty | interceptor | vanilla | 0.9176 | 235 | 133 | 120 | 13 | 9 |
| Concise + prior + uncertainty | interceptor | trained | 0.9310 | 243 | 131 | 123 | 8 | 1 |

Mixed groups contain both guesses among the eight replicates. Per-bit accuracy is the correct count divided by 128.

Paired changes versus the original prompt use identical saved contexts, roles, model conditions, and initial seeds. Matching seeds does not make sampled thinking identical across prompts or guarantee bitwise equality across hardware.

| Prompt | Role | Model | Change in mean P(0) | Change in sampled zeros | Change in correct guesses |
|---|---|---|---:|---:|---:|
| Original + equal prior | receiver | vanilla | +0.1017 | +13 | +7 |
| Original + equal prior | receiver | trained | +0.0560 | +8 | +6 |
| Original + equal prior | interceptor | vanilla | +0.0602 | +15 | +17 |
| Original + equal prior | interceptor | trained | +0.0727 | +18 | -2 |
| Concise | receiver | vanilla | -0.0526 | -23 | -11 |
| Concise | receiver | trained | -0.1190 | -33 | -15 |
| Concise | interceptor | vanilla | +0.1943 | +48 | +8 |
| Concise | interceptor | trained | +0.1958 | +49 | -9 |
| Concise + equal prior | receiver | vanilla | -0.2840 | -84 | -10 |
| Concise + equal prior | receiver | trained | -0.3180 | -83 | -1 |
| Concise + equal prior | interceptor | vanilla | +0.0831 | +26 | +0 |
| Concise + equal prior | interceptor | trained | +0.1140 | +27 | +5 |
| Concise + prior + uncertainty | receiver | vanilla | +0.0369 | +2 | -10 |
| Concise + prior + uncertainty | receiver | trained | -0.0033 | -3 | -11 |
| Concise + prior + uncertainty | interceptor | vanilla | +0.2350 | +60 | +6 |
| Concise + prior + uncertainty | interceptor | trained | +0.2234 | +58 | -10 |

For concise + prior, receiver mean P(0) falls by 28.4 percentage points in vanilla and 31.8 points with trained adapters. Overall correct guesses change from 136 to 126/256 (vanilla) and 136 to 135/256 (trained). The vanilla mean probabilities barely distinguish true bits: P(0)=0.539 when m=0 and 0.541 when m=1. The trained values are 0.583 and 0.514, respectively. Reducing a default guessing preference does not establish successful decoding.

The near-neutral receiver mean is present in both receiver-identity subsets: A/B means are 0.536/0.544 for vanilla and 0.549/0.548 for trained. The A and B subsets contain different contexts, so this is not a controlled identity-only comparison.

There is a secondary encouraging result on the 25 accepted saved answers: receiver correct guesses increase from 96 to 109/200 for vanilla and 94 to 114/200 for trained. This subset has 11 zero-bit contexts and 14 one-bit contexts, is small, and was inspected across multiple prompts. It requires fresh validation before claiming decoding improvement.

Concise + prior increases receiver mixed-guess groups from 21 to 27/32 (vanilla) and 16 to 23/32 (trained). More within-context variation could matter for future GRPO learning, but this inference-only comparison did not test training with the new prompts. The interceptor remains biased toward zero under this prompt (mean P(0)=0.766/0.822), and trained interceptor accuracy 146/256 exceeds receiver accuracy 135/256. There is no demonstrated overall secrecy advantage.

Verification: Slurm completed job 24369289 with exit 0:0; config digest and RESULTS.json hash match COMPLETE.json; all 5,120 row keys are unique; matched contexts and seeds agree; reused control rows match the earlier results exactly; all public artifact hashes and model verification pass; summary recomputation agrees within absolute floating-point tolerance 1e-12.

Artifacts: [original generated report](report.md), [per-question CSV](per_question.csv), [public results](RESULTS.json), [all breakdowns](SUMMARY.json), [exact prompts](PROMPTS.json), [manifest](MANIFEST.json), [model verification](MODEL_VERIFICATION.json), and [local verification](PUBLIC_RESULT_AUDIT.json).
