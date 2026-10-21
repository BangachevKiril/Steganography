# Frozen decoder prompt comparison

Four new prompts produced 4,096 decodes. The unchanged original prompt control reuses its 1,024 previously completed decodes. Every condition uses the same 32 saved checkpoint-40 answers, eight replicates per answer, matched seeds, and frozen vanilla or trained models. No answers or judge verdicts were regenerated and no model was updated.

conditional_p0 is the two-label probability of 0 after that arm's own sampled private thinking, at temperature 0.6. Mean P(0) over eight independently sampled thinking paths estimates the marginal decoder probability. The experiment contains 32 distinct saved answers, not 256 independent questions. No private thinking is retained.

The trained adapters remain checkpoint 40 from training with the original decoder prompts; new prompts are applied only at inference. Matching seeds does not guarantee identical floating-point outputs across GPU models or software execution conditions. A P(0) closer to one half alone does not establish better decoding.

| Prompt | Role | Model | Mean P(0) | Change vs original | Sampled 0 | Accuracy | Thinking cutoff | Groups with both guesses |
|---|---|---|---:|---:|---:|---:|---:|---:|
| original | receiver | vanilla | 0.8240 | +0.0000 | 220/256 | 136/256 (53.1%) | 256/256 | 21/32 |
| original | receiver | trained | 0.8663 | +0.0000 | 224/256 | 136/256 (53.1%) | 256/256 | 16/32 |
| original | interceptor | vanilla | 0.6826 | +0.0000 | 175/256 | 127/256 (49.6%) | 256/256 | 27/32 |
| original | interceptor | trained | 0.7076 | +0.0000 | 185/256 | 141/256 (55.1%) | 256/256 | 23/32 |
| original_plus_prior | receiver | vanilla | 0.9256 | +0.1017 | 233/256 | 143/256 (55.9%) | 256/256 | 11/32 |
| original_plus_prior | receiver | trained | 0.9223 | +0.0560 | 232/256 | 142/256 (55.5%) | 256/256 | 11/32 |
| original_plus_prior | interceptor | vanilla | 0.7428 | +0.0602 | 190/256 | 144/256 (56.2%) | 256/256 | 25/32 |
| original_plus_prior | interceptor | trained | 0.7803 | +0.0727 | 203/256 | 139/256 (54.3%) | 256/256 | 19/32 |
| concise | receiver | vanilla | 0.7714 | -0.0526 | 197/256 | 125/256 (48.8%) | 256/256 | 19/32 |
| concise | receiver | trained | 0.7473 | -0.1190 | 191/256 | 121/256 (47.3%) | 256/256 | 17/32 |
| concise | interceptor | vanilla | 0.8769 | +0.1943 | 223/256 | 135/256 (52.7%) | 256/256 | 16/32 |
| concise | interceptor | trained | 0.9034 | +0.1958 | 234/256 | 132/256 (51.6%) | 256/256 | 10/32 |
| concise_plus_prior | receiver | vanilla | 0.5399 | -0.2840 | 136/256 | 126/256 (49.2%) | 256/256 | 27/32 |
| concise_plus_prior | receiver | trained | 0.5483 | -0.3180 | 141/256 | 135/256 (52.7%) | 256/256 | 23/32 |
| concise_plus_prior | interceptor | vanilla | 0.7656 | +0.0831 | 201/256 | 127/256 (49.6%) | 256/256 | 23/32 |
| concise_plus_prior | interceptor | trained | 0.8216 | +0.1140 | 212/256 | 146/256 (57.0%) | 256/256 | 16/32 |
| concise_plus_prior_uncertainty | receiver | vanilla | 0.8608 | +0.0369 | 222/256 | 126/256 (49.2%) | 256/256 | 9/32 |
| concise_plus_prior_uncertainty | receiver | trained | 0.8631 | -0.0033 | 221/256 | 125/256 (48.8%) | 256/256 | 13/32 |
| concise_plus_prior_uncertainty | interceptor | vanilla | 0.9176 | +0.2350 | 235/256 | 133/256 (52.0%) | 256/256 | 9/32 |
| concise_plus_prior_uncertainty | interceptor | trained | 0.9310 | +0.2234 | 243/256 | 131/256 (51.2%) | 256/256 | 1/32 |

Both-guesses rate is the fraction of complete eight-replicate answer groups containing at least one 0 and at least one 1.

`SUMMARY.json` also gives results by true bit, receiver identity, and saved answer quality; paired changes from the original prompt; and trained-minus-vanilla differences. `per_question.csv` contains all question-level averages. The 32 saved answers are reused across conditions; the many replicates are not independent test questions.
