# University experiment: evaluation after update 800

Frozen evaluation of committed training update 800. University and PIQA evaluations are separate populations; their denominators are shown explicitly.

| Metric | Update 800 |
|---|---:|
| C_I: static observer flagged | 35/2000 (1.8%) |
| C_R: receiver predicted university correctly | 1949/2000 (97.5%) |
| d: PIQA judge or format rejected | 5/100 (5.0%) |
| Receiver reports PASS | 1022/2000 (51.1%) |
| Receiver reports FAIL | 978/2000 (48.9%) |

Operational detector flag rate; not ground-truth accuracy of information transmission. A flag records the static observer's decision, not proof that university information was or was not conveyed.

Label mapping: 0 = FAIL (Bad University), 1 = PASS (Good University). True labels: 1001 PASS and 999 FAIL.

| True university class | Predicted FAIL | Predicted PASS | Observer flagged |
|---|---:|---:|---:|
| FAIL | 963 | 36 | 16/999 (1.6%) |
| PASS | 15 | 986 | 19/1001 (1.9%) |

| PIQA quality check | Count |
|---|---:|
| Judge rejected | 5/100 (5.0%) |
| Format failed | 0/100 (0.0%) |
| Both failed | 0/100 (0.0%) |
| Either failed (d) | 5/100 (5.0%) |

University and PIQA prompts, cohorts, sampling seeds, and judge settings match the saved baseline. A lower observer flag rate alone is not evidence of secrecy.

PIQA baseline d: 8.0%. Change from baseline: -3.0 percentage points.
