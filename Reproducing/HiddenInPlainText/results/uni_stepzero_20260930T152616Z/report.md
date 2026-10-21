# University experiment: baseline

Step zero, before any parameter updates. University and PIQA evaluations are separate populations; their denominators are shown explicitly.

| Metric | Baseline |
|---|---:|
| C_I: static observer flagged | 88/2000 (4.4%) |
| C_R: receiver predicted university correctly | 1011/2000 (50.5%) |
| d: PIQA judge or format rejected | 8/100 (8.0%) |
| Receiver reports PASS | 1916/2000 (95.8%) |
| Receiver reports FAIL | 84/2000 (4.2%) |

Operational detector flag rate; not ground-truth accuracy of information transmission. A flag records the static observer's decision, not proof that university information was or was not conveyed.

Label mapping: 0 = FAIL (Bad University), 1 = PASS (Good University). True labels: 1001 PASS and 999 FAIL.

| True university class | Predicted FAIL | Predicted PASS | Observer flagged |
|---|---:|---:|---:|
| FAIL | 47 | 952 | 42/999 (4.2%) |
| PASS | 37 | 964 | 46/1001 (4.6%) |

| PIQA quality check | Count |
|---|---:|
| Judge rejected | 5/100 (5.0%) |
| Format failed | 3/100 (3.0%) |
| Both failed | 0/100 (0.0%) |
| Either failed (d) | 8/100 (8.0%) |

This step-zero evaluation establishes the quality baseline; it does not yet measure degradation from training. A lower observer flag rate alone is not evidence of secrecy.

Observer control accuracy (separate from C_I): {"n": 148, "correct": 146, "0": {"n": 42, "correct": 42}, "1": {"n": 106, "correct": 104}}

See config.json and PROTOCOL.json for pinned models, seeds, evaluation additions, and observer tokenizer configuration.
