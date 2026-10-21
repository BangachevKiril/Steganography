# Training metrics

Only steps at or below the independently verified committed checkpoint **100** are included.

`C_R` is receiver accuracy, `C_I` is interceptor accuracy, and `d` is the fraction of answers rejected by the judge or format check. Counts are pooled within each ten-step phase; rates are weighted by their measured denominators.

Joint steps measure all sender answers (normally 128 per step). Interceptor-only monitoring measures the same 16 answers with receiver, interceptor, and judge; interceptor optimization accuracy is labeled separately. Optimization uses the configured decoder rollout count: one guess per answer gives 16 interceptor-only guesses, while eight gives 128. These measurements are taken before the step’s updates. Blank CSV cells and “not measured” mean the metric was unavailable, never zero.

| Steps | Phase | Steps observed | C_R | C_I | d | Interceptor optimization C_I |
|---|---|---:|---|---|---|---|
| 1–10 | joint | 10/10 | 623/1280 (48.7%) | 667/1280 (52.1%) | 295/1280 (23.0%) | not measured |
| 11–20 | interceptor | 10/10 | 75/160 (46.9%) | 84/160 (52.5%) | 32/160 (20.0%) | 84/160 (52.5%) |
| 21–30 | joint | 10/10 | 660/1280 (51.6%) | 649/1280 (50.7%) | 290/1280 (22.7%) | not measured |
| 31–40 | interceptor | 10/10 | 93/160 (58.1%) | 83/160 (51.9%) | 32/160 (20.0%) | 83/160 (51.9%) |
| 41–50 | joint | 10/10 | 659/1280 (51.5%) | 651/1280 (50.9%) | 288/1280 (22.5%) | not measured |
| 51–60 | interceptor | 10/10 | 69/160 (43.1%) | 76/160 (47.5%) | 42/160 (26.2%) | 76/160 (47.5%) |
| 61–70 | joint | 10/10 | 616/1280 (48.1%) | 617/1280 (48.2%) | 289/1280 (22.6%) | not measured |
| 71–80 | interceptor | 10/10 | 82/160 (51.2%) | 82/160 (51.2%) | 36/160 (22.5%) | 82/160 (51.2%) |
| 81–90 | joint | 10/10 | 675/1280 (52.7%) | 658/1280 (51.4%) | 283/1280 (22.1%) | not measured |
| 91–100 | interceptor | 10/10 | 80/160 (50.0%) | 76/160 (47.5%) | 31/160 (19.4%) | 76/160 (47.5%) |

## Validation

Validation uses a single rollout per sampled question; these results are kept separate from training.

| Step | C_R | C_I | d |
|---:|---|---|---|
| 20 | 20/32 (62.5%) | 10/32 (31.2%) | 9/32 (28.1%) |
| 40 | 16/32 (50.0%) | 11/32 (34.4%) | 13/32 (40.6%) |
| 60 | 22/32 (68.8%) | 17/32 (53.1%) | 7/32 (21.9%) |
| 80 | 15/32 (46.9%) | 21/32 (65.6%) | 10/32 (31.2%) |
| 100 | 11/32 (34.4%) | 17/32 (53.1%) | 9/32 (28.1%) |

## Decoder optimization diagnostics

These diagnostics describe the guesses used for optimization. With single decoder rollouts, joint updates reuse one guess for each sender answer and interceptor-only updates use one guess for each of 16 answers. Legacy grouped sampling uses eight guesses per selected answer context. Counts are pooled within a phase. CE and conditional final-bit probabilities are measured before the update; the probability conditions on each sampled reasoning prefix. Reasoning PG activity is a nonzero-advantage fraction, not evidence of a useful learned code.

| Steps | Phase | Role | Zero guesses | Correct guesses | Correct, true bit 0 | Correct, true bit 1 | Mean CE |
|---|---|---|---|---|---|---|---|
| 1–10 | joint | interceptor | 585/1280 (45.7%) | 667/1280 (52.1%) | 306/640 (47.8%) | 361/640 (56.4%) | 2.1282 |
| 1–10 | joint | receiver | 619/1280 (48.4%) | 623/1280 (48.7%) | 301/640 (47.0%) | 322/640 (50.3%) | 1.9705 |
| 11–20 | interceptor | interceptor | 82/160 (51.2%) | 84/160 (52.5%) | 43/80 (53.8%) | 41/80 (51.2%) | 1.4782 |
| 21–30 | joint | interceptor | 613/1280 (47.9%) | 649/1280 (50.7%) | 311/640 (48.6%) | 338/640 (52.8%) | 1.0167 |
| 21–30 | joint | receiver | 652/1280 (50.9%) | 660/1280 (51.6%) | 336/640 (52.5%) | 324/640 (50.6%) | 1.0232 |
| 31–40 | interceptor | interceptor | 97/160 (60.6%) | 83/160 (51.9%) | 50/80 (62.5%) | 33/80 (41.2%) | 0.7033 |
| 41–50 | joint | interceptor | 639/1280 (49.9%) | 651/1280 (50.9%) | 325/640 (50.8%) | 326/640 (50.9%) | 0.7089 |
| 41–50 | joint | receiver | 609/1280 (47.6%) | 659/1280 (51.5%) | 314/640 (49.1%) | 345/640 (53.9%) | 0.7390 |
| 51–60 | interceptor | interceptor | 90/160 (56.2%) | 76/160 (47.5%) | 43/80 (53.8%) | 33/80 (41.2%) | 0.7442 |
| 61–70 | joint | interceptor | 667/1280 (52.1%) | 617/1280 (48.2%) | 322/640 (50.3%) | 295/640 (46.1%) | 0.6867 |
| 61–70 | joint | receiver | 654/1280 (51.1%) | 616/1280 (48.1%) | 315/640 (49.2%) | 301/640 (47.0%) | 0.7022 |
| 71–80 | interceptor | interceptor | 82/160 (51.2%) | 82/160 (51.2%) | 42/80 (52.5%) | 40/80 (50.0%) | 0.6878 |
| 81–90 | joint | interceptor | 654/1280 (51.1%) | 658/1280 (51.4%) | 336/640 (52.5%) | 322/640 (50.3%) | 0.7007 |
| 81–90 | joint | receiver | 639/1280 (49.9%) | 675/1280 (52.7%) | 337/640 (52.7%) | 338/640 (52.8%) | 0.6971 |
| 91–100 | interceptor | interceptor | 96/160 (60.0%) | 76/160 (47.5%) | 46/80 (57.5%) | 30/80 (37.5%) | 0.6922 |

[Per-step decoder diagnostics](decoder_metrics.csv)


## Timing

- interceptor: median 60.0 seconds per step, measured over 50 committed steps.
- joint: median 349.7 seconds per step, measured over 50 committed steps.

Recorded remaining-time estimate: 0.00 hours. ETA excludes queue/preemption and checkpoint writes; includes future validation only once measured.

[Per-step counts and rates](training_metrics.csv)

## Completion assessment

Completed all 100 steps at 2026-10-01 02:27 UTC (September 30, 10:27 p.m. EDT). Slurm job 24467609 completed with exit 0:0 after two preemptions. The final checkpoint has a complete manifest, nonempty state.pt on the cluster, and the frozen configuration digest. All five scheduled validations are present. Production smoke and fresh step-zero initialization were verified; public artifacts contain no private rollouts or full training state.

Final validation: receiver 11/32 (34.4%), interceptor 17/32 (53.1%), rejected answers 9/32 (28.1%). The receiver scored 6/16 on true bit 0 and 5/16 on true bit 1; receiver A/B scored 5/16 and 6/16. Both receiver and interceptor guessed zero 17/32 times. Among 23 accepted answers without detected literal disclosure, receiver/interceptor accuracy was 8/23 versus 11/23. No final-validation literal flags were detected; these proxies do not prove secrecy.

Zero bias was lower than in revised-GRPO final validation (receiver 29/32 zero guesses, interceptor 25/32), but both the receiver prompt and decoder sampling changed, so this is not an isolated causal comparison. The stopped hybrid has only a step-20 validation and is not a final-checkpoint comparator.

In the last joint phase (81–90), receiver/interceptor zero counts were 639/1280 and 654/1280. Mean decoder CE for A/B/I was 0.698/0.696/0.701 nats, near ln(2), and conditional P(0) was 0.489/0.496/0.495. Sender GRPO had active variation in 798/800 groups across the run; decoder reasoning-PG activity was 100%, with finite nonzero gradients. Singleton decoder GRPO variation is zero by construction and does not measure hybrid update activity. Active updates and balanced guesses did not establish reliable hidden communication.
