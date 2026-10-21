# Training metrics

Only steps at or below the independently verified committed checkpoint **100** are included.

`C_R` is receiver accuracy, `C_I` is interceptor accuracy, and `d` is the fraction of answers rejected by the judge or format check. Counts are pooled within each ten-step phase; rates are weighted by their measured denominators.

Joint steps measure all sender answers (normally 128 per step). Interceptor-only steps use a matched monitoring probe (normally 16 answers per step); interceptor optimization accuracy over all eight guesses per answer appears separately (normally 128 guesses per step). These measurements are taken before the step’s updates. Blank CSV cells and “not measured” mean the metric was unavailable, never zero.

| Steps | Phase | Steps observed | C_R | C_I | d | Interceptor optimization C_I |
|---|---|---:|---|---|---|---|
| 1–10 | joint | 10/10 | 643/1280 (50.2%) | 652/1280 (50.9%) | 258/1280 (20.2%) | not measured |
| 11–20 | interceptor | 10/10 | 82/160 (51.2%) | 78/160 (48.8%) | 36/160 (22.5%) | 619/1280 (48.4%) |
| 21–30 | joint | 10/10 | 642/1280 (50.2%) | 661/1280 (51.6%) | 243/1280 (19.0%) | not measured |
| 31–40 | interceptor | 10/10 | 81/160 (50.6%) | 84/160 (52.5%) | 29/160 (18.1%) | 659/1280 (51.5%) |
| 41–50 | joint | 10/10 | 645/1280 (50.4%) | 631/1280 (49.3%) | 187/1280 (14.6%) | not measured |
| 51–60 | interceptor | 10/10 | 77/160 (48.1%) | 86/160 (53.8%) | 22/160 (13.8%) | 675/1280 (52.7%) |
| 61–70 | joint | 10/10 | 662/1280 (51.7%) | 664/1280 (51.9%) | 233/1280 (18.2%) | not measured |
| 71–80 | interceptor | 10/10 | 82/160 (51.2%) | 87/160 (54.4%) | 26/160 (16.2%) | 674/1280 (52.7%) |
| 81–90 | joint | 10/10 | 627/1280 (49.0%) | 644/1280 (50.3%) | 183/1280 (14.3%) | not measured |
| 91–100 | interceptor | 10/10 | 85/160 (53.1%) | 81/160 (50.6%) | 22/160 (13.8%) | 648/1280 (50.6%) |

## Validation

Validation uses a single rollout per sampled question; these results are kept separate from training.

| Step | C_R | C_I | d |
|---:|---|---|---|
| 20 | 17/32 (53.1%) | 17/32 (53.1%) | 6/32 (18.8%) |
| 40 | 18/32 (56.2%) | 15/32 (46.9%) | 7/32 (21.9%) |
| 60 | 16/32 (50.0%) | 16/32 (50.0%) | 6/32 (18.8%) |
| 80 | 17/32 (53.1%) | 17/32 (53.1%) | 7/32 (21.9%) |
| 100 | 15/32 (46.9%) | 15/32 (46.9%) | 2/32 (6.2%) |

## Timing

- interceptor: median 156.4 seconds per step, measured over 50 committed steps.
- joint: median 551.6 seconds per step, measured over 50 committed steps.

Recorded remaining-time estimate: 0.00 hours. ETA excludes queue/preemption and checkpoint writes; includes future validation only once measured.

[Per-step counts and rates](training_metrics.csv)
