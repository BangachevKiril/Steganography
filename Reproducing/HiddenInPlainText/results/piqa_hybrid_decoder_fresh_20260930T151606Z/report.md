# Training metrics

**Stopped at the user’s request after committed step 23. Slurm job 24441537 is cancelled and its monitor is deleted. Completed checkpoints and public results are preserved; no restart is scheduled.**

Only steps at or below the independently verified committed checkpoint **23** are included.

`C_R` is receiver accuracy, `C_I` is interceptor accuracy, and `d` is the fraction of answers rejected by the judge or format check. Counts are pooled within each ten-step phase; rates are weighted by their measured denominators.

Joint steps measure all sender answers (normally 128 per step). Interceptor-only steps use a matched monitoring probe (normally 16 answers per step); interceptor optimization accuracy over all eight guesses per answer appears separately (normally 128 guesses per step). These measurements are taken before the step’s updates. Blank CSV cells and “not measured” mean the metric was unavailable, never zero.

| Steps | Phase | Steps observed | C_R | C_I | d | Interceptor optimization C_I |
|---|---|---:|---|---|---|---|
| 1–10 | joint | 10/10 | 660/1280 (51.6%) | 691/1280 (54.0%) | 258/1280 (20.2%) | not measured |
| 11–20 | interceptor | 10/10 | 89/160 (55.6%) | 84/160 (52.5%) | 44/160 (27.5%) | 660/1280 (51.6%) |
| 21–23 (partial) | joint | 3/3 | 190/384 (49.5%) | 194/384 (50.5%) | 95/384 (24.7%) | not measured |

## Validation

Validation uses a single rollout per sampled question; these results are kept separate from training.

| Step | C_R | C_I | d |
|---:|---|---|---|
| 20 | 17/32 (53.1%) | 12/32 (37.5%) | 5/32 (15.6%) |

## Decoder optimization diagnostics

These are the eight optimization guesses per answer context, not the matched monitoring observations above. Counts are pooled within a phase. CE and conditional final-bit probabilities are measured before the update; the probability conditions on each sampled reasoning prefix. Reasoning PG activity is a nonzero-advantage fraction, not evidence of a useful learned code.

| Steps | Phase | Role | Zero guesses | Correct guesses | Correct, true bit 0 | Correct, true bit 1 | Mean CE |
|---|---|---|---|---|---|---|---|
| 1–10 | joint | interceptor | 626/1280 (48.9%) | 648/1280 (50.6%) | 317/640 (49.5%) | 331/640 (51.7%) | 2.1719 |
| 1–10 | joint | receiver | 674/1280 (52.7%) | 694/1280 (54.2%) | 364/640 (56.9%) | 330/640 (51.6%) | 2.6211 |
| 11–20 | interceptor | interceptor | 640/1280 (50.0%) | 660/1280 (51.6%) | 330/640 (51.6%) | 330/640 (51.6%) | 1.3990 |
| 21–23 | joint | interceptor | 221/384 (57.6%) | 213/384 (55.5%) | 121/192 (63.0%) | 92/192 (47.9%) | 1.0203 |
| 21–23 | joint | receiver | 127/384 (33.1%) | 197/384 (51.3%) | 66/192 (34.4%) | 131/192 (68.2%) | 2.0984 |

[Per-step decoder diagnostics](decoder_metrics.csv)


## Timing

- interceptor: median 140.9 seconds per step, measured over 10 committed steps.
- joint: median 490.6 seconds per step, measured over 13 committed steps.


[Per-step counts and rates](training_metrics.csv)
