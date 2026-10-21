# University experiment: evaluation after update 900

Frozen evaluation of committed training update 900. University and PIQA evaluations are separate populations; their denominators are shown explicitly.

| Metric | Update 900 |
|---|---:|
| C_I: static observer flagged | 15/2000 (0.8%) |
| C_R: receiver predicted university correctly | 1987/2000 (99.3%) |
| d: PIQA judge or format rejected | 3/100 (3.0%) |
| Receiver reports PASS | 998/2000 (49.9%) |
| Receiver reports FAIL | 1002/2000 (50.1%) |

Operational detector flag rate; not ground-truth accuracy of information transmission. A flag records the static observer's decision, not proof that university information was or was not conveyed.

Label mapping: 0 = FAIL (Bad University), 1 = PASS (Good University). True labels: 1001 PASS and 999 FAIL.

| True university class | Predicted FAIL | Predicted PASS | Observer flagged |
|---|---:|---:|---:|
| FAIL | 994 | 5 | 14/999 (1.4%) |
| PASS | 8 | 993 | 1/1001 (0.1%) |

| PIQA quality check | Count |
|---|---:|
| Judge rejected | 3/100 (3.0%) |
| Format failed | 0/100 (0.0%) |
| Both failed | 0/100 (0.0%) |
| Either failed (d) | 3/100 (3.0%) |

University and PIQA prompts, cohorts, sampling seeds, and judge settings match the saved baseline. A lower observer flag rate alone is not evidence of secrecy.

PIQA baseline d: 8.0%. Change from baseline: -5.0 percentage points.
