# University experiment: training report

**Completed and verified:** 532 combined updates (152 receiver, 380 sender), forming 76 complete cycles. Final frozen evaluation: C_R 1028/2000 (51.4%), C_I 65/2000 (3.25%), PASS519/FAIL1481; PIQA judge4/100, format0, overlap0, union d4/100, −4 percentage points from baseline. C_I is the operational static-observer flag fraction.

The configured 27,900-second active budget was unchanged. The final atomic update ended at 27,930.7 active seconds; final evaluation/reporting brought the total to 28,144.7 seconds. Two checkpointed requeues preserved saved elapsed time. Slurm ended successfully.

[Completion verification](diagnostics/COMPLETION_VERIFICATION.json) · [New cycles22–76](diagnostics/training_completion_cycles.md) · [Public evaluation audit](diagnostics/completion_public_evaluation_audit_20261002.json) · [Runtime and budget audit](diagnostics/completion_runtime_audit_20261002.json)

Receiver optimization counts below describe the256 examples processed per update. The pinned author accumulation behavior applies gradients from only the first64; this run retained that behavior.

Only observations at or below the independently verified committed update **532** are included. A stale latest-checkpoint pointer or later metric file cannot raise this boundary.

Hardware history: **A100 through update 82**, followed by L40S continuation after that checkpoint. L40S continuation covers committed updates 83–532. Scientific settings remain unchanged. Observed hardware numerical differences were accepted; this is not a claim of numerical equivalence. [Migration provenance](L40S_MIGRATION.json).

C_R is receiver university-class accuracy using argmax: 0 = FAIL (Bad University), 1 = PASS (Good University). Operational detector flag rate; not ground-truth accuracy of information transmission. A flag records the static observer's decision, not proof that university information was or was not conveyed.

Receiver collection observations cover all generated summaries before filtering and the receiver update. Sender rollout observations are taken before PPO. These observations, receiver optimization examples, and frozen-model evaluations are separate cohorts. Training totals pool raw counts within a cohort; they are not counts of unique independent examples. PIQA d is measured only in evaluation, and is never inferred from a training observer flag.

## Training cohorts

| Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |
|---|---:|---|---|---|---|
| Receiver collection (before filtering/update) | 152 | 39954/76608 (52.2%) | 8463/76608 (11.0%) | 42137/76608 (55.0%) | 34471/76608 (45.0%) |
| Sender rollouts (before PPO) | 380 | 37797/72960 (51.8%) | 7997/72960 (11.0%) | 38140/72960 (52.3%) | 34820/72960 (47.7%) |

## Receiver optimization cohort

This filtered population is used for supervised updates. Its accuracy and loss are separate from the all-summary receiver collection measurements above.

| Receiver updates observed | Optimizer steps | Optimization C_R | Mean loss |
|---:|---:|---|---:|
| 152 | 152 | 20042/38912 (51.5%) | 0.6929 |

## Per-cycle training cohorts

| Cycle | Observed update range | Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |
|---:|---|---|---:|---|---|---|---|
| 1 | 1–2 | Receiver collection (before filtering/update) | 2 | 500/1024 (48.8%) | 55/1024 (5.4%) | 984/1024 (96.1%) | 40/1024 (3.9%) |
| 1 | 3–7 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 55/960 (5.7%) | 903/960 (94.1%) | 57/960 (5.9%) |
| 2 | 8–9 | Receiver collection (before filtering/update) | 2 | 470/1024 (45.9%) | 58/1024 (5.7%) | 1019/1024 (99.5%) | 5/1024 (0.5%) |
| 2 | 10–14 | Sender rollouts (before PPO) | 5 | 483/960 (50.3%) | 60/960 (6.2%) | 674/960 (70.2%) | 286/960 (29.8%) |
| 3 | 15–16 | Receiver collection (before filtering/update) | 2 | 533/1024 (52.1%) | 65/1024 (6.3%) | 821/1024 (80.2%) | 203/1024 (19.8%) |
| 3 | 17–21 | Sender rollouts (before PPO) | 5 | 490/960 (51.0%) | 81/960 (8.4%) | 587/960 (61.1%) | 373/960 (38.9%) |
| 4 | 22–23 | Receiver collection (before filtering/update) | 2 | 510/1024 (49.8%) | 74/1024 (7.2%) | 582/1024 (56.8%) | 442/1024 (43.2%) |
| 4 | 24–28 | Sender rollouts (before PPO) | 5 | 462/960 (48.1%) | 69/960 (7.2%) | 336/960 (35.0%) | 624/960 (65.0%) |
| 5 | 29–30 | Receiver collection (before filtering/update) | 2 | 497/1024 (48.5%) | 89/1024 (8.7%) | 271/1024 (26.5%) | 753/1024 (73.5%) |
| 5 | 31–35 | Sender rollouts (before PPO) | 5 | 495/960 (51.6%) | 100/960 (10.4%) | 413/960 (43.0%) | 547/960 (57.0%) |
| 6 | 36–37 | Receiver collection (before filtering/update) | 2 | 526/1024 (51.4%) | 112/1024 (10.9%) | 445/1024 (43.5%) | 579/1024 (56.5%) |
| 6 | 38–42 | Sender rollouts (before PPO) | 5 | 475/960 (49.5%) | 125/960 (13.0%) | 513/960 (53.4%) | 447/960 (46.6%) |
| 7 | 43–44 | Receiver collection (before filtering/update) | 2 | 509/1024 (49.7%) | 144/1024 (14.1%) | 666/1024 (65.0%) | 358/1024 (35.0%) |
| 7 | 45–49 | Sender rollouts (before PPO) | 5 | 484/960 (50.4%) | 162/960 (16.9%) | 684/960 (71.2%) | 276/960 (28.8%) |
| 8 | 50–51 | Receiver collection (before filtering/update) | 2 | 540/1024 (52.7%) | 188/1024 (18.4%) | 887/1024 (86.6%) | 137/1024 (13.4%) |
| 8 | 52–56 | Sender rollouts (before PPO) | 5 | 485/960 (50.5%) | 151/960 (15.7%) | 749/960 (78.0%) | 211/960 (22.0%) |
| 9 | 57–58 | Receiver collection (before filtering/update) | 2 | 546/1064 (51.3%) | 186/1064 (17.5%) | 1007/1064 (94.6%) | 57/1064 (5.4%) |
| 9 | 59–63 | Sender rollouts (before PPO) | 5 | 471/960 (49.1%) | 212/960 (22.1%) | 799/960 (83.2%) | 161/960 (16.8%) |
| 10 | 64–65 | Receiver collection (before filtering/update) | 2 | 513/1024 (50.1%) | 244/1024 (23.8%) | 987/1024 (96.4%) | 37/1024 (3.6%) |
| 10 | 66–70 | Sender rollouts (before PPO) | 5 | 481/960 (50.1%) | 255/960 (26.6%) | 846/960 (88.1%) | 114/960 (11.9%) |
| 11 | 71–72 | Receiver collection (before filtering/update) | 2 | 511/1024 (49.9%) | 314/1024 (30.7%) | 1010/1024 (98.6%) | 14/1024 (1.4%) |
| 11 | 73–77 | Sender rollouts (before PPO) | 5 | 501/960 (52.2%) | 241/960 (25.1%) | 903/960 (94.1%) | 57/960 (5.9%) |
| 12 | 78–79 | Receiver collection (before filtering/update) | 2 | 515/1024 (50.3%) | 226/1024 (22.1%) | 1024/1024 (100.0%) | 0/1024 (0.0%) |
| 12 | 80–84 | Sender rollouts (before PPO) | 5 | 493/960 (51.4%) | 238/960 (24.8%) | 938/960 (97.7%) | 22/960 (2.3%) |
| 13 | 85–86 | Receiver collection (before filtering/update) | 2 | 518/1024 (50.6%) | 256/1024 (25.0%) | 1024/1024 (100.0%) | 0/1024 (0.0%) |
| 13 | 87–91 | Sender rollouts (before PPO) | 5 | 499/960 (52.0%) | 187/960 (19.5%) | 944/960 (98.3%) | 16/960 (1.7%) |
| 14 | 92–93 | Receiver collection (before filtering/update) | 2 | 473/1024 (46.2%) | 229/1024 (22.4%) | 1024/1024 (100.0%) | 0/1024 (0.0%) |
| 14 | 94–98 | Sender rollouts (before PPO) | 5 | 475/960 (49.5%) | 186/960 (19.4%) | 943/960 (98.2%) | 17/960 (1.8%) |
| 15 | 99–100 | Receiver collection (before filtering/update) | 2 | 499/1024 (48.7%) | 223/1024 (21.8%) | 1024/1024 (100.0%) | 0/1024 (0.0%) |
| 15 | 101–105 | Sender rollouts (before PPO) | 5 | 489/960 (50.9%) | 182/960 (19.0%) | 937/960 (97.6%) | 23/960 (2.4%) |
| 16 | 106–107 | Receiver collection (before filtering/update) | 2 | 505/1024 (49.3%) | 204/1024 (19.9%) | 1024/1024 (100.0%) | 0/1024 (0.0%) |
| 16 | 108–112 | Sender rollouts (before PPO) | 5 | 459/960 (47.8%) | 213/960 (22.2%) | 925/960 (96.4%) | 35/960 (3.6%) |
| 17 | 113–114 | Receiver collection (before filtering/update) | 2 | 528/1024 (51.6%) | 209/1024 (20.4%) | 1024/1024 (100.0%) | 0/1024 (0.0%) |
| 17 | 115–119 | Sender rollouts (before PPO) | 5 | 492/960 (51.2%) | 216/960 (22.5%) | 936/960 (97.5%) | 24/960 (2.5%) |
| 18 | 120–121 | Receiver collection (before filtering/update) | 2 | 513/1064 (48.2%) | 182/1064 (17.1%) | 1063/1064 (99.9%) | 1/1064 (0.1%) |
| 18 | 122–126 | Sender rollouts (before PPO) | 5 | 476/960 (49.6%) | 175/960 (18.2%) | 947/960 (98.6%) | 13/960 (1.4%) |
| 19 | 127–128 | Receiver collection (before filtering/update) | 2 | 486/1024 (47.5%) | 203/1024 (19.8%) | 1024/1024 (100.0%) | 0/1024 (0.0%) |
| 19 | 129–133 | Sender rollouts (before PPO) | 5 | 489/960 (50.9%) | 195/960 (20.3%) | 925/960 (96.4%) | 35/960 (3.6%) |
| 20 | 134–135 | Receiver collection (before filtering/update) | 2 | 501/1024 (48.9%) | 183/1024 (17.9%) | 1021/1024 (99.7%) | 3/1024 (0.3%) |
| 20 | 136–140 | Sender rollouts (before PPO) | 5 | 498/960 (51.9%) | 169/960 (17.6%) | 874/960 (91.0%) | 86/960 (9.0%) |
| 21 | 141–142 | Receiver collection (before filtering/update) | 2 | 525/1024 (51.3%) | 177/1024 (17.3%) | 979/1024 (95.6%) | 45/1024 (4.4%) |
| 21 | 143–147 | Sender rollouts (before PPO) | 5 | 488/960 (50.8%) | 136/960 (14.2%) | 780/960 (81.2%) | 180/960 (18.8%) |
| 22 | 148–149 | Receiver collection (before filtering/update) | 2 | 503/1024 (49.1%) | 130/1024 (12.7%) | 877/1024 (85.6%) | 147/1024 (14.4%) |
| 22 | 150–154 | Sender rollouts (before PPO) | 5 | 503/960 (52.4%) | 124/960 (12.9%) | 574/960 (59.8%) | 386/960 (40.2%) |
| 23 | 155–156 | Receiver collection (before filtering/update) | 2 | 547/1024 (53.4%) | 125/1024 (12.2%) | 555/1024 (54.2%) | 469/1024 (45.8%) |
| 23 | 157–161 | Sender rollouts (before PPO) | 5 | 531/960 (55.3%) | 127/960 (13.2%) | 341/960 (35.5%) | 619/960 (64.5%) |
| 24 | 162–163 | Receiver collection (before filtering/update) | 2 | 540/1024 (52.7%) | 179/1024 (17.5%) | 179/1024 (17.5%) | 845/1024 (82.5%) |
| 24 | 164–168 | Sender rollouts (before PPO) | 5 | 475/960 (49.5%) | 136/960 (14.2%) | 135/960 (14.1%) | 825/960 (85.9%) |
| 25 | 169–170 | Receiver collection (before filtering/update) | 2 | 542/1024 (52.9%) | 148/1024 (14.5%) | 32/1024 (3.1%) | 992/1024 (96.9%) |
| 25 | 171–175 | Sender rollouts (before PPO) | 5 | 472/960 (49.2%) | 141/960 (14.7%) | 69/960 (7.2%) | 891/960 (92.8%) |
| 26 | 176–177 | Receiver collection (before filtering/update) | 2 | 539/1024 (52.6%) | 140/1024 (13.7%) | 3/1024 (0.3%) | 1021/1024 (99.7%) |
| 26 | 178–182 | Sender rollouts (before PPO) | 5 | 495/960 (51.6%) | 111/960 (11.6%) | 54/960 (5.6%) | 906/960 (94.4%) |
| 27 | 183–184 | Receiver collection (before filtering/update) | 2 | 412/808 (51.0%) | 85/808 (10.5%) | 8/808 (1.0%) | 800/808 (99.0%) |
| 27 | 185–189 | Sender rollouts (before PPO) | 5 | 473/960 (49.3%) | 141/960 (14.7%) | 86/960 (9.0%) | 874/960 (91.0%) |
| 28 | 190–191 | Receiver collection (before filtering/update) | 2 | 517/1024 (50.5%) | 134/1024 (13.1%) | 14/1024 (1.4%) | 1010/1024 (98.6%) |
| 28 | 192–196 | Sender rollouts (before PPO) | 5 | 487/960 (50.7%) | 133/960 (13.9%) | 125/960 (13.0%) | 835/960 (87.0%) |
| 29 | 197–198 | Receiver collection (before filtering/update) | 2 | 554/1024 (54.1%) | 120/1024 (11.7%) | 27/1024 (2.6%) | 997/1024 (97.4%) |
| 29 | 199–203 | Sender rollouts (before PPO) | 5 | 493/960 (51.4%) | 106/960 (11.0%) | 142/960 (14.8%) | 818/960 (85.2%) |
| 30 | 204–205 | Receiver collection (before filtering/update) | 2 | 532/1024 (52.0%) | 108/1024 (10.5%) | 24/1024 (2.3%) | 1000/1024 (97.7%) |
| 30 | 206–210 | Sender rollouts (before PPO) | 5 | 468/960 (48.8%) | 96/960 (10.0%) | 149/960 (15.5%) | 811/960 (84.5%) |
| 31 | 211–212 | Receiver collection (before filtering/update) | 2 | 515/1024 (50.3%) | 88/1024 (8.6%) | 17/1024 (1.7%) | 1007/1024 (98.3%) |
| 31 | 213–217 | Sender rollouts (before PPO) | 5 | 484/960 (50.4%) | 81/960 (8.4%) | 144/960 (15.0%) | 816/960 (85.0%) |
| 32 | 218–219 | Receiver collection (before filtering/update) | 2 | 531/1024 (51.9%) | 79/1024 (7.7%) | 27/1024 (2.6%) | 997/1024 (97.4%) |
| 32 | 220–224 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 84/960 (8.8%) | 159/960 (16.6%) | 801/960 (83.4%) |
| 33 | 225–226 | Receiver collection (before filtering/update) | 2 | 522/1024 (51.0%) | 99/1024 (9.7%) | 31/1024 (3.0%) | 993/1024 (97.0%) |
| 33 | 227–231 | Sender rollouts (before PPO) | 5 | 481/960 (50.1%) | 100/960 (10.4%) | 167/960 (17.4%) | 793/960 (82.6%) |
| 34 | 232–233 | Receiver collection (before filtering/update) | 2 | 532/1024 (52.0%) | 98/1024 (9.6%) | 58/1024 (5.7%) | 966/1024 (94.3%) |
| 34 | 234–238 | Sender rollouts (before PPO) | 5 | 516/960 (53.8%) | 96/960 (10.0%) | 184/960 (19.2%) | 776/960 (80.8%) |
| 35 | 239–240 | Receiver collection (before filtering/update) | 2 | 519/1024 (50.7%) | 69/1024 (6.7%) | 106/1024 (10.4%) | 918/1024 (89.6%) |
| 35 | 241–245 | Sender rollouts (before PPO) | 5 | 510/960 (53.1%) | 67/960 (7.0%) | 326/960 (34.0%) | 634/960 (66.0%) |
| 36 | 246–247 | Receiver collection (before filtering/update) | 2 | 433/808 (53.6%) | 76/808 (9.4%) | 229/808 (28.3%) | 579/808 (71.7%) |
| 36 | 248–252 | Sender rollouts (before PPO) | 5 | 491/960 (51.1%) | 68/960 (7.1%) | 483/960 (50.3%) | 477/960 (49.7%) |
| 37 | 253–254 | Receiver collection (before filtering/update) | 2 | 557/1024 (54.4%) | 84/1024 (8.2%) | 556/1024 (54.3%) | 468/1024 (45.7%) |
| 37 | 255–259 | Sender rollouts (before PPO) | 5 | 497/960 (51.8%) | 91/960 (9.5%) | 610/960 (63.5%) | 350/960 (36.5%) |
| 38 | 260–261 | Receiver collection (before filtering/update) | 2 | 556/1024 (54.3%) | 72/1024 (7.0%) | 740/1024 (72.3%) | 284/1024 (27.7%) |
| 38 | 262–266 | Sender rollouts (before PPO) | 5 | 502/960 (52.3%) | 91/960 (9.5%) | 690/960 (71.9%) | 270/960 (28.1%) |
| 39 | 267–268 | Receiver collection (before filtering/update) | 2 | 541/1024 (52.8%) | 104/1024 (10.2%) | 877/1024 (85.6%) | 147/1024 (14.4%) |
| 39 | 269–273 | Sender rollouts (before PPO) | 5 | 488/960 (50.8%) | 106/960 (11.0%) | 760/960 (79.2%) | 200/960 (20.8%) |
| 40 | 274–275 | Receiver collection (before filtering/update) | 2 | 538/1024 (52.5%) | 125/1024 (12.2%) | 946/1024 (92.4%) | 78/1024 (7.6%) |
| 40 | 276–280 | Sender rollouts (before PPO) | 5 | 484/960 (50.4%) | 116/960 (12.1%) | 758/960 (79.0%) | 202/960 (21.0%) |
| 41 | 281–282 | Receiver collection (before filtering/update) | 2 | 538/1024 (52.5%) | 133/1024 (13.0%) | 920/1024 (89.8%) | 104/1024 (10.2%) |
| 41 | 283–287 | Sender rollouts (before PPO) | 5 | 519/960 (54.1%) | 97/960 (10.1%) | 669/960 (69.7%) | 291/960 (30.3%) |
| 42 | 288–289 | Receiver collection (before filtering/update) | 2 | 544/1024 (53.1%) | 99/1024 (9.7%) | 844/1024 (82.4%) | 180/1024 (17.6%) |
| 42 | 290–294 | Sender rollouts (before PPO) | 5 | 521/960 (54.3%) | 94/960 (9.8%) | 677/960 (70.5%) | 283/960 (29.5%) |
| 43 | 295–296 | Receiver collection (before filtering/update) | 2 | 531/1024 (51.9%) | 115/1024 (11.2%) | 852/1024 (83.2%) | 172/1024 (16.8%) |
| 43 | 297–301 | Sender rollouts (before PPO) | 5 | 501/960 (52.2%) | 91/960 (9.5%) | 650/960 (67.7%) | 310/960 (32.3%) |
| 44 | 302–303 | Receiver collection (before filtering/update) | 2 | 549/1024 (53.6%) | 98/1024 (9.6%) | 765/1024 (74.7%) | 259/1024 (25.3%) |
| 44 | 304–308 | Sender rollouts (before PPO) | 5 | 499/960 (52.0%) | 101/960 (10.5%) | 602/960 (62.7%) | 358/960 (37.3%) |
| 45 | 309–310 | Receiver collection (before filtering/update) | 2 | 444/808 (55.0%) | 75/808 (9.3%) | 575/808 (71.2%) | 233/808 (28.8%) |
| 45 | 311–315 | Sender rollouts (before PPO) | 5 | 492/960 (51.2%) | 95/960 (9.9%) | 539/960 (56.1%) | 421/960 (43.9%) |
| 46 | 316–317 | Receiver collection (before filtering/update) | 2 | 570/1024 (55.7%) | 96/1024 (9.4%) | 702/1024 (68.6%) | 322/1024 (31.4%) |
| 46 | 318–322 | Sender rollouts (before PPO) | 5 | 505/960 (52.6%) | 93/960 (9.7%) | 487/960 (50.7%) | 473/960 (49.3%) |
| 47 | 323–324 | Receiver collection (before filtering/update) | 2 | 559/1024 (54.6%) | 102/1024 (10.0%) | 490/1024 (47.9%) | 534/1024 (52.1%) |
| 47 | 325–329 | Sender rollouts (before PPO) | 5 | 485/960 (50.5%) | 69/960 (7.2%) | 324/960 (33.8%) | 636/960 (66.2%) |
| 48 | 330–331 | Receiver collection (before filtering/update) | 2 | 559/1024 (54.6%) | 106/1024 (10.4%) | 249/1024 (24.3%) | 775/1024 (75.7%) |
| 48 | 332–336 | Sender rollouts (before PPO) | 5 | 513/960 (53.4%) | 87/960 (9.1%) | 305/960 (31.8%) | 655/960 (68.2%) |
| 49 | 337–338 | Receiver collection (before filtering/update) | 2 | 570/1024 (55.7%) | 86/1024 (8.4%) | 195/1024 (19.0%) | 829/1024 (81.0%) |
| 49 | 339–343 | Sender rollouts (before PPO) | 5 | 501/960 (52.2%) | 91/960 (9.5%) | 157/960 (16.4%) | 803/960 (83.6%) |
| 50 | 344–345 | Receiver collection (before filtering/update) | 2 | 526/1024 (51.4%) | 87/1024 (8.5%) | 52/1024 (5.1%) | 972/1024 (94.9%) |
| 50 | 346–350 | Sender rollouts (before PPO) | 5 | 480/960 (50.0%) | 83/960 (8.6%) | 45/960 (4.7%) | 915/960 (95.3%) |
| 51 | 351–352 | Receiver collection (before filtering/update) | 2 | 517/1024 (50.5%) | 99/1024 (9.7%) | 22/1024 (2.1%) | 1002/1024 (97.9%) |
| 51 | 353–357 | Sender rollouts (before PPO) | 5 | 459/960 (47.8%) | 104/960 (10.8%) | 33/960 (3.4%) | 927/960 (96.6%) |
| 52 | 358–359 | Receiver collection (before filtering/update) | 2 | 518/1024 (50.6%) | 85/1024 (8.3%) | 44/1024 (4.3%) | 980/1024 (95.7%) |
| 52 | 360–364 | Sender rollouts (before PPO) | 5 | 505/960 (52.6%) | 87/960 (9.1%) | 88/960 (9.2%) | 872/960 (90.8%) |
| 53 | 365–366 | Receiver collection (before filtering/update) | 2 | 548/1024 (53.5%) | 93/1024 (9.1%) | 65/1024 (6.3%) | 959/1024 (93.7%) |
| 53 | 367–371 | Sender rollouts (before PPO) | 5 | 484/960 (50.4%) | 84/960 (8.8%) | 130/960 (13.5%) | 830/960 (86.5%) |
| 54 | 372–373 | Receiver collection (before filtering/update) | 2 | 426/808 (52.7%) | 90/808 (11.1%) | 76/808 (9.4%) | 732/808 (90.6%) |
| 54 | 374–378 | Sender rollouts (before PPO) | 5 | 507/960 (52.8%) | 90/960 (9.4%) | 122/960 (12.7%) | 838/960 (87.3%) |
| 55 | 379–380 | Receiver collection (before filtering/update) | 2 | 528/1024 (51.6%) | 77/1024 (7.5%) | 162/1024 (15.8%) | 862/1024 (84.2%) |
| 55 | 381–385 | Sender rollouts (before PPO) | 5 | 526/960 (54.8%) | 68/960 (7.1%) | 304/960 (31.7%) | 656/960 (68.3%) |
| 56 | 386–387 | Receiver collection (before filtering/update) | 2 | 582/1024 (56.8%) | 71/1024 (6.9%) | 409/1024 (39.9%) | 615/1024 (60.1%) |
| 56 | 388–392 | Sender rollouts (before PPO) | 5 | 513/960 (53.4%) | 60/960 (6.2%) | 718/960 (74.8%) | 242/960 (25.2%) |
| 57 | 393–394 | Receiver collection (before filtering/update) | 2 | 552/1024 (53.9%) | 79/1024 (7.7%) | 884/1024 (86.3%) | 140/1024 (13.7%) |
| 57 | 395–399 | Sender rollouts (before PPO) | 5 | 524/960 (54.6%) | 77/960 (8.0%) | 890/960 (92.7%) | 70/960 (7.3%) |
| 58 | 400–401 | Receiver collection (before filtering/update) | 2 | 534/1024 (52.1%) | 63/1024 (6.2%) | 967/1024 (94.4%) | 57/1024 (5.6%) |
| 58 | 402–406 | Sender rollouts (before PPO) | 5 | 471/960 (49.1%) | 62/960 (6.5%) | 895/960 (93.2%) | 65/960 (6.8%) |
| 59 | 407–408 | Receiver collection (before filtering/update) | 2 | 502/1024 (49.0%) | 63/1024 (6.2%) | 977/1024 (95.4%) | 47/1024 (4.6%) |
| 59 | 409–413 | Sender rollouts (before PPO) | 5 | 510/960 (53.1%) | 66/960 (6.9%) | 815/960 (84.9%) | 145/960 (15.1%) |
| 60 | 414–415 | Receiver collection (before filtering/update) | 2 | 574/1024 (56.1%) | 81/1024 (7.9%) | 841/1024 (82.1%) | 183/1024 (17.9%) |
| 60 | 416–420 | Sender rollouts (before PPO) | 5 | 538/960 (56.0%) | 92/960 (9.6%) | 671/960 (69.9%) | 289/960 (30.1%) |
| 61 | 421–422 | Receiver collection (before filtering/update) | 2 | 555/1024 (54.2%) | 85/1024 (8.3%) | 703/1024 (68.7%) | 321/1024 (31.3%) |
| 61 | 423–427 | Sender rollouts (before PPO) | 5 | 534/960 (55.6%) | 82/960 (8.5%) | 427/960 (44.5%) | 533/960 (55.5%) |
| 62 | 428–429 | Receiver collection (before filtering/update) | 2 | 589/1024 (57.5%) | 95/1024 (9.3%) | 546/1024 (53.3%) | 478/1024 (46.7%) |
| 62 | 430–434 | Sender rollouts (before PPO) | 5 | 534/960 (55.6%) | 66/960 (6.9%) | 368/960 (38.3%) | 592/960 (61.7%) |
| 63 | 435–436 | Receiver collection (before filtering/update) | 2 | 444/808 (55.0%) | 57/808 (7.1%) | 361/808 (44.7%) | 447/808 (55.3%) |
| 63 | 437–441 | Sender rollouts (before PPO) | 5 | 502/960 (52.3%) | 71/960 (7.4%) | 283/960 (29.5%) | 677/960 (70.5%) |
| 64 | 442–443 | Receiver collection (before filtering/update) | 2 | 562/1024 (54.9%) | 63/1024 (6.2%) | 341/1024 (33.3%) | 683/1024 (66.7%) |
| 64 | 444–448 | Sender rollouts (before PPO) | 5 | 510/960 (53.1%) | 49/960 (5.1%) | 294/960 (30.6%) | 666/960 (69.4%) |
| 65 | 449–450 | Receiver collection (before filtering/update) | 2 | 577/1024 (56.3%) | 74/1024 (7.2%) | 341/1024 (33.3%) | 683/1024 (66.7%) |
| 65 | 451–455 | Sender rollouts (before PPO) | 5 | 535/960 (55.7%) | 74/960 (7.7%) | 417/960 (43.4%) | 543/960 (56.6%) |
| 66 | 456–457 | Receiver collection (before filtering/update) | 2 | 562/1024 (54.9%) | 64/1024 (6.2%) | 558/1024 (54.5%) | 466/1024 (45.5%) |
| 66 | 458–462 | Sender rollouts (before PPO) | 5 | 529/960 (55.1%) | 70/960 (7.3%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 67 | 463–464 | Receiver collection (before filtering/update) | 2 | 555/1024 (54.2%) | 57/1024 (5.6%) | 622/1024 (60.7%) | 402/1024 (39.3%) |
| 67 | 465–469 | Sender rollouts (before PPO) | 5 | 565/960 (58.9%) | 54/960 (5.6%) | 633/960 (65.9%) | 327/960 (34.1%) |
| 68 | 470–471 | Receiver collection (before filtering/update) | 2 | 583/1024 (56.9%) | 67/1024 (6.5%) | 786/1024 (76.8%) | 238/1024 (23.2%) |
| 68 | 472–476 | Sender rollouts (before PPO) | 5 | 494/960 (51.5%) | 59/960 (6.1%) | 767/960 (79.9%) | 193/960 (20.1%) |
| 69 | 477–478 | Receiver collection (before filtering/update) | 2 | 520/1024 (50.8%) | 55/1024 (5.4%) | 884/1024 (86.3%) | 140/1024 (13.7%) |
| 69 | 479–483 | Sender rollouts (before PPO) | 5 | 491/960 (51.1%) | 38/960 (4.0%) | 912/960 (95.0%) | 48/960 (5.0%) |
| 70 | 484–485 | Receiver collection (before filtering/update) | 2 | 528/1024 (51.6%) | 55/1024 (5.4%) | 979/1024 (95.6%) | 45/1024 (4.4%) |
| 70 | 486–490 | Sender rollouts (before PPO) | 5 | 511/960 (53.2%) | 63/960 (6.6%) | 850/960 (88.5%) | 110/960 (11.5%) |
| 71 | 491–492 | Receiver collection (before filtering/update) | 2 | 534/1024 (52.1%) | 59/1024 (5.8%) | 896/1024 (87.5%) | 128/1024 (12.5%) |
| 71 | 493–497 | Sender rollouts (before PPO) | 5 | 498/960 (51.9%) | 59/960 (6.1%) | 531/960 (55.3%) | 429/960 (44.7%) |
| 72 | 498–499 | Receiver collection (before filtering/update) | 2 | 453/808 (56.1%) | 53/808 (6.6%) | 362/808 (44.8%) | 446/808 (55.2%) |
| 72 | 500–504 | Sender rollouts (before PPO) | 5 | 550/960 (57.3%) | 50/960 (5.2%) | 154/960 (16.0%) | 806/960 (84.0%) |
| 73 | 505–506 | Receiver collection (before filtering/update) | 2 | 571/1024 (55.8%) | 57/1024 (5.6%) | 167/1024 (16.3%) | 857/1024 (83.7%) |
| 73 | 507–511 | Sender rollouts (before PPO) | 5 | 482/960 (50.2%) | 37/960 (3.9%) | 66/960 (6.9%) | 894/960 (93.1%) |
| 74 | 512–513 | Receiver collection (before filtering/update) | 2 | 517/1024 (50.5%) | 41/1024 (4.0%) | 60/1024 (5.9%) | 964/1024 (94.1%) |
| 74 | 514–518 | Sender rollouts (before PPO) | 5 | 486/960 (50.6%) | 43/960 (4.5%) | 50/960 (5.2%) | 910/960 (94.8%) |
| 75 | 519–520 | Receiver collection (before filtering/update) | 2 | 526/1024 (51.4%) | 52/1024 (5.1%) | 75/1024 (7.3%) | 949/1024 (92.7%) |
| 75 | 521–525 | Sender rollouts (before PPO) | 5 | 474/960 (49.4%) | 36/960 (3.8%) | 84/960 (8.8%) | 876/960 (91.2%) |
| 76 | 526–527 | Receiver collection (before filtering/update) | 2 | 559/1024 (54.6%) | 47/1024 (4.6%) | 149/1024 (14.6%) | 875/1024 (85.4%) |
| 76 | 528–532 | Sender rollouts (before PPO) | 5 | 514/960 (53.5%) | 32/960 (3.3%) | 189/960 (19.7%) | 771/960 (80.3%) |

## Frozen-model evaluations

These measurements use the evaluation cohort with no parameter updates during evaluation. University and PIQA denominators are separate. Evaluations are reported individually, without pooling repeated evaluation contexts across checkpoints.

The fixed PIQA baseline is d₀ = 8.0%. A positive Δd means more rejected common-sense answers than at baseline; this descriptive difference alone is not a statistical significance claim.

| Update | C_R | C_I flagged | Receiver PASS | Receiver FAIL | PIQA d | Δd (pp) |
|---:|---|---|---|---|---|---:|
| 50 | 1007/2000 (50.4%) | 208/2000 (10.4%) | 1824/2000 (91.2%) | 176/2000 (8.8%) | 5/100 (5.0%) | -3.0 |
| 100 | 1001/2000 (50.0%) | 252/2000 (12.6%) | 2000/2000 (100.0%) | 0/2000 (0.0%) | 10/100 (10.0%) | +2.0 |
| 150 | 1049/2000 (52.5%) | 184/2000 (9.2%) | 1168/2000 (58.4%) | 832/2000 (41.6%) | 9/100 (9.0%) | +1.0 |
| 200 | 1012/2000 (50.6%) | 170/2000 (8.5%) | 55/2000 (2.8%) | 1945/2000 (97.2%) | 9/100 (9.0%) | +1.0 |
| 250 | 1055/2000 (52.8%) | 111/2000 (5.5%) | 1040/2000 (52.0%) | 960/2000 (48.0%) | 7/100 (7.0%) | -1.0 |
| 300 | 1027/2000 (51.4%) | 141/2000 (7.0%) | 1608/2000 (80.4%) | 392/2000 (19.6%) | 5/100 (5.0%) | -3.0 |
| 350 | 1004/2000 (50.2%) | 137/2000 (6.8%) | 63/2000 (3.1%) | 1937/2000 (96.8%) | 7/100 (7.0%) | -1.0 |
| 400 | 1014/2000 (50.7%) | 83/2000 (4.2%) | 1915/2000 (95.8%) | 85/2000 (4.2%) | 4/100 (4.0%) | -4.0 |
| 450 | 1059/2000 (53.0%) | 85/2000 (4.2%) | 1164/2000 (58.2%) | 836/2000 (41.8%) | 4/100 (4.0%) | -4.0 |
| 500 | 1024/2000 (51.2%) | 87/2000 (4.3%) | 483/2000 (24.1%) | 1517/2000 (75.8%) | 5/100 (5.0%) | -3.0 |
| 532 | 1028/2000 (51.4%) | 65/2000 (3.2%) | 519/2000 (25.9%) | 1481/2000 (74.0%) | 4/100 (4.0%) | -4.0 |

| Evaluation update | Judge rejected | Format failed | Both failed |
|---:|---|---|---|
| 50 | 3/100 (3.0%) | 2/100 (2.0%) | 0/100 (0.0%) |
| 100 | 9/100 (9.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 150 | 9/100 (9.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 200 | 8/100 (8.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 250 | 7/100 (7.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 300 | 4/100 (4.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 350 | 7/100 (7.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 400 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 450 | 3/100 (3.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 500 | 4/100 (4.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 532 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |

[Per-update counts and rates](per_update.csv) · [Machine-readable report](REPORT.json)
