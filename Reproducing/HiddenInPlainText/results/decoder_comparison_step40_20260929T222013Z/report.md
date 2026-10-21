# Frozen decoder comparison

The same 32 checkpoint-40 validation answers were decoded by the vanilla base and trained checkpoint-40 adapters. Each answer has eight independent replicates for each role and arm. Seeds, questions, public strings, sender identities, and answers are matched across arms. Receiver decoding uses the opposite communicator's adapter; interceptor decoding uses I.

No training updates were performed. Saved answer-quality verdicts were retained; the judge was not rerun.

conditional_p0 is the two-label probability of 0 after that arm's own sampled private thinking, at temperature 0.6. Mean P(0) over eight independently sampled thinking paths estimates the marginal decoder probability. The experiment contains 32 distinct saved answers, not 256 independent questions. No private thinking is retained.

| Role | Model | Mean P(0) | Sampled 0 | Correct | Thinking cutoff |
|---|---|---:|---:|---:|---:|
| receiver | vanilla | 0.8240 | 220/256 (85.9%) | 136/256 (53.1%) | 256/256 |
| receiver | trained | 0.8663 | 224/256 (87.5%) | 136/256 (53.1%) | 256/256 |
| interceptor | vanilla | 0.6826 | 175/256 (68.4%) | 127/256 (49.6%) | 256/256 |
| interceptor | trained | 0.7076 | 185/256 (72.3%) | 141/256 (55.1%) | 256/256 |

Trained minus vanilla mean P(0): receiver +0.0424; interceptor +0.0250.

`SUMMARY.json` includes results by true bit, receiver identity, and saved answer quality. `per_question.csv` contains each question/role/arm average, and `RESULTS.json` contains all public replicate decisions. Multiple replicates reduce sampling noise on these same answers; they do not establish performance on additional questions.
