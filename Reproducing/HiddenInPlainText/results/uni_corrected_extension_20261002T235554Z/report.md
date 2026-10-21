# University experiment: training report

Only observations at or below the independently verified committed update **1076** are included. A stale latest-checkpoint pointer or later metric file cannot raise this boundary.

This separately initialized L40S experiment corrects receiver gradient accumulation, sender attention masking, observer scoring position and receiver reward-scoring mode. All 256 filtered examples contribute to each receiver optimizer step. The released raw-logit reward and other hyperparameters are retained.

Baseline comparisons use this run's completed step-zero evaluation under the corrected scorer. Historical observer rates use a different scoring position and are not directly comparable.

C_R is receiver university-class accuracy using argmax: 0 = FAIL (Bad University), 1 = PASS (Good University). Operational detector flag rate; not ground-truth accuracy of information transmission. A flag records the static observer's decision, not proof that university information was or was not conveyed.

Receiver collection observations cover all generated summaries before filtering and the receiver update. Sender rollout observations are taken before PPO. These observations, receiver optimization examples, and frozen-model evaluations are separate cohorts. Training totals pool raw counts within a cohort; they are not counts of unique independent examples. PIQA d is measured only in evaluation, and is never inferred from a training observer flag.

## Training cohorts

| Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |
|---|---:|---|---|---|---|
| Receiver collection (before filtering/update) | 308 | 100918/149632 (67.4%) | 9401/149632 (6.3%) | 72205/149632 (48.3%) | 77427/149632 (51.7%) |
| Sender rollouts (before PPO) | 768 | 101030/147456 (68.5%) | 9135/147456 (6.2%) | 71189/147456 (48.3%) | 76267/147456 (51.7%) |

## Receiver optimization cohort

This filtered population is used for supervised updates. Its accuracy and loss are separate from the all-summary receiver collection measurements above.

| Receiver updates observed | Optimizer steps | Optimization C_R | Mean loss |
|---:|---:|---|---:|
| 308 | 308 | 53941/78848 (68.4%) | 0.4894 |

## Per-cycle training cohorts

| Cycle | Observed update range | Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |
|---:|---|---|---:|---|---|---|---|
| 1 | 1–2 | Receiver collection (before filtering/update) | 2 | 499/1024 (48.7%) | 60/1024 (5.9%) | 667/1024 (65.1%) | 357/1024 (34.9%) |
| 1 | 3–7 | Sender rollouts (before PPO) | 5 | 480/960 (50.0%) | 80/960 (8.3%) | 625/960 (65.1%) | 335/960 (34.9%) |
| 2 | 8–9 | Receiver collection (before filtering/update) | 2 | 520/1024 (50.8%) | 62/1024 (6.1%) | 447/1024 (43.7%) | 577/1024 (56.3%) |
| 2 | 10–14 | Sender rollouts (before PPO) | 5 | 475/960 (49.5%) | 52/960 (5.4%) | 46/960 (4.8%) | 914/960 (95.2%) |
| 3 | 15–16 | Receiver collection (before filtering/update) | 2 | 506/1024 (49.4%) | 68/1024 (6.6%) | 32/1024 (3.1%) | 992/1024 (96.9%) |
| 3 | 17–21 | Sender rollouts (before PPO) | 5 | 480/960 (50.0%) | 71/960 (7.4%) | 5/960 (0.5%) | 955/960 (99.5%) |
| 4 | 22–23 | Receiver collection (before filtering/update) | 2 | 552/1024 (53.9%) | 87/1024 (8.5%) | 2/1024 (0.2%) | 1022/1024 (99.8%) |
| 4 | 24–28 | Sender rollouts (before PPO) | 5 | 469/960 (48.9%) | 89/960 (9.3%) | 3/960 (0.3%) | 957/960 (99.7%) |
| 5 | 29–30 | Receiver collection (before filtering/update) | 2 | 504/1024 (49.2%) | 92/1024 (9.0%) | 0/1024 (0.0%) | 1024/1024 (100.0%) |
| 5 | 31–35 | Sender rollouts (before PPO) | 5 | 473/960 (49.3%) | 89/960 (9.3%) | 5/960 (0.5%) | 955/960 (99.5%) |
| 6 | 36–37 | Receiver collection (before filtering/update) | 2 | 530/1024 (51.8%) | 120/1024 (11.7%) | 19/1024 (1.9%) | 1005/1024 (98.1%) |
| 6 | 38–42 | Sender rollouts (before PPO) | 5 | 497/960 (51.8%) | 106/960 (11.0%) | 83/960 (8.6%) | 877/960 (91.4%) |
| 7 | 43–44 | Receiver collection (before filtering/update) | 2 | 486/1024 (47.5%) | 119/1024 (11.6%) | 127/1024 (12.4%) | 897/1024 (87.6%) |
| 7 | 45–49 | Sender rollouts (before PPO) | 5 | 502/960 (52.3%) | 119/960 (12.4%) | 370/960 (38.5%) | 590/960 (61.5%) |
| 8 | 50–51 | Receiver collection (before filtering/update) | 2 | 529/1024 (51.7%) | 143/1024 (14.0%) | 488/1024 (47.7%) | 536/1024 (52.3%) |
| 8 | 52–56 | Sender rollouts (before PPO) | 5 | 514/960 (53.5%) | 133/960 (13.9%) | 772/960 (80.4%) | 188/960 (19.6%) |
| 9 | 57–58 | Receiver collection (before filtering/update) | 2 | 558/1064 (52.4%) | 153/1064 (14.4%) | 881/1064 (82.8%) | 183/1064 (17.2%) |
| 9 | 59–63 | Sender rollouts (before PPO) | 5 | 483/960 (50.3%) | 150/960 (15.6%) | 845/960 (88.0%) | 115/960 (12.0%) |
| 10 | 64–65 | Receiver collection (before filtering/update) | 2 | 525/1024 (51.3%) | 153/1024 (14.9%) | 911/1024 (89.0%) | 113/1024 (11.0%) |
| 10 | 66–70 | Sender rollouts (before PPO) | 5 | 465/960 (48.4%) | 140/960 (14.6%) | 794/960 (82.7%) | 166/960 (17.3%) |
| 11 | 71–72 | Receiver collection (before filtering/update) | 2 | 516/1024 (50.4%) | 164/1024 (16.0%) | 829/1024 (81.0%) | 195/1024 (19.0%) |
| 11 | 73–77 | Sender rollouts (before PPO) | 5 | 510/960 (53.1%) | 176/960 (18.3%) | 548/960 (57.1%) | 412/960 (42.9%) |
| 12 | 78–79 | Receiver collection (before filtering/update) | 2 | 516/1024 (50.4%) | 145/1024 (14.2%) | 551/1024 (53.8%) | 473/1024 (46.2%) |
| 12 | 80–84 | Sender rollouts (before PPO) | 5 | 509/960 (53.0%) | 147/960 (15.3%) | 480/960 (50.0%) | 480/960 (50.0%) |
| 13 | 85–86 | Receiver collection (before filtering/update) | 2 | 528/1024 (51.6%) | 178/1024 (17.4%) | 480/1024 (46.9%) | 544/1024 (53.1%) |
| 13 | 87–91 | Sender rollouts (before PPO) | 5 | 532/960 (55.4%) | 119/960 (12.4%) | 497/960 (51.8%) | 463/960 (48.2%) |
| 14 | 92–93 | Receiver collection (before filtering/update) | 2 | 529/1024 (51.7%) | 130/1024 (12.7%) | 534/1024 (52.1%) | 490/1024 (47.9%) |
| 14 | 94–98 | Sender rollouts (before PPO) | 5 | 508/960 (52.9%) | 123/960 (12.8%) | 346/960 (36.0%) | 614/960 (64.0%) |
| 15 | 99–100 | Receiver collection (before filtering/update) | 2 | 554/1024 (54.1%) | 137/1024 (13.4%) | 335/1024 (32.7%) | 689/1024 (67.3%) |
| 15 | 101–105 | Sender rollouts (before PPO) | 5 | 486/960 (50.6%) | 119/960 (12.4%) | 194/960 (20.2%) | 766/960 (79.8%) |
| 16 | 106–107 | Receiver collection (before filtering/update) | 2 | 523/1024 (51.1%) | 114/1024 (11.1%) | 194/1024 (18.9%) | 830/1024 (81.1%) |
| 16 | 108–112 | Sender rollouts (before PPO) | 5 | 524/960 (54.6%) | 128/960 (13.3%) | 186/960 (19.4%) | 774/960 (80.6%) |
| 17 | 113–114 | Receiver collection (before filtering/update) | 2 | 512/1024 (50.0%) | 115/1024 (11.2%) | 212/1024 (20.7%) | 812/1024 (79.3%) |
| 17 | 115–119 | Sender rollouts (before PPO) | 5 | 469/960 (48.9%) | 111/960 (11.6%) | 367/960 (38.2%) | 593/960 (61.8%) |
| 18 | 120–121 | Receiver collection (before filtering/update) | 2 | 421/808 (52.1%) | 85/808 (10.5%) | 305/808 (37.7%) | 503/808 (62.3%) |
| 18 | 122–126 | Sender rollouts (before PPO) | 5 | 513/960 (53.4%) | 143/960 (14.9%) | 552/960 (57.5%) | 408/960 (42.5%) |
| 19 | 127–128 | Receiver collection (before filtering/update) | 2 | 524/1024 (51.2%) | 129/1024 (12.6%) | 686/1024 (67.0%) | 338/1024 (33.0%) |
| 19 | 129–133 | Sender rollouts (before PPO) | 5 | 517/960 (53.9%) | 110/960 (11.5%) | 727/960 (75.7%) | 233/960 (24.3%) |
| 20 | 134–135 | Receiver collection (before filtering/update) | 2 | 530/1024 (51.8%) | 132/1024 (12.9%) | 679/1024 (66.3%) | 345/1024 (33.7%) |
| 20 | 136–140 | Sender rollouts (before PPO) | 5 | 530/960 (55.2%) | 123/960 (12.8%) | 492/960 (51.2%) | 468/960 (48.8%) |
| 21 | 141–142 | Receiver collection (before filtering/update) | 2 | 550/1024 (53.7%) | 154/1024 (15.0%) | 530/1024 (51.8%) | 494/1024 (48.2%) |
| 21 | 143–147 | Sender rollouts (before PPO) | 5 | 502/960 (52.3%) | 131/960 (13.6%) | 422/960 (44.0%) | 538/960 (56.0%) |
| 22 | 148–149 | Receiver collection (before filtering/update) | 2 | 531/1024 (51.9%) | 143/1024 (14.0%) | 448/1024 (43.8%) | 576/1024 (56.2%) |
| 22 | 150–154 | Sender rollouts (before PPO) | 5 | 482/960 (50.2%) | 117/960 (12.2%) | 357/960 (37.2%) | 603/960 (62.8%) |
| 23 | 155–156 | Receiver collection (before filtering/update) | 2 | 527/1024 (51.5%) | 105/1024 (10.3%) | 475/1024 (46.4%) | 549/1024 (53.6%) |
| 23 | 157–161 | Sender rollouts (before PPO) | 5 | 517/960 (53.9%) | 90/960 (9.4%) | 555/960 (57.8%) | 405/960 (42.2%) |
| 24 | 162–163 | Receiver collection (before filtering/update) | 2 | 537/1024 (52.4%) | 107/1024 (10.4%) | 591/1024 (57.7%) | 433/1024 (42.3%) |
| 24 | 164–168 | Sender rollouts (before PPO) | 5 | 510/960 (53.1%) | 110/960 (11.5%) | 548/960 (57.1%) | 412/960 (42.9%) |
| 25 | 169–170 | Receiver collection (before filtering/update) | 2 | 566/1024 (55.3%) | 136/1024 (13.3%) | 555/1024 (54.2%) | 469/1024 (45.8%) |
| 25 | 171–175 | Sender rollouts (before PPO) | 5 | 483/960 (50.3%) | 109/960 (11.4%) | 494/960 (51.5%) | 466/960 (48.5%) |
| 26 | 176–177 | Receiver collection (before filtering/update) | 2 | 529/1024 (51.7%) | 120/1024 (11.7%) | 487/1024 (47.6%) | 537/1024 (52.4%) |
| 26 | 178–182 | Sender rollouts (before PPO) | 5 | 493/960 (51.4%) | 105/960 (10.9%) | 332/960 (34.6%) | 628/960 (65.4%) |
| 27 | 183–184 | Receiver collection (before filtering/update) | 2 | 449/808 (55.6%) | 84/808 (10.4%) | 273/808 (33.8%) | 535/808 (66.2%) |
| 27 | 185–189 | Sender rollouts (before PPO) | 5 | 487/960 (50.7%) | 108/960 (11.2%) | 280/960 (29.2%) | 680/960 (70.8%) |
| 28 | 190–191 | Receiver collection (before filtering/update) | 2 | 521/1024 (50.9%) | 113/1024 (11.0%) | 314/1024 (30.7%) | 710/1024 (69.3%) |
| 28 | 192–196 | Sender rollouts (before PPO) | 5 | 531/960 (55.3%) | 112/960 (11.7%) | 421/960 (43.9%) | 539/960 (56.1%) |
| 29 | 197–198 | Receiver collection (before filtering/update) | 2 | 537/1024 (52.4%) | 102/1024 (10.0%) | 509/1024 (49.7%) | 515/1024 (50.3%) |
| 29 | 199–203 | Sender rollouts (before PPO) | 5 | 492/960 (51.2%) | 107/960 (11.1%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 30 | 204–205 | Receiver collection (before filtering/update) | 2 | 557/1024 (54.4%) | 112/1024 (10.9%) | 513/1024 (50.1%) | 511/1024 (49.9%) |
| 30 | 206–210 | Sender rollouts (before PPO) | 5 | 493/960 (51.4%) | 81/960 (8.4%) | 492/960 (51.2%) | 468/960 (48.8%) |
| 31 | 211–212 | Receiver collection (before filtering/update) | 2 | 539/1024 (52.6%) | 96/1024 (9.4%) | 486/1024 (47.5%) | 538/1024 (52.5%) |
| 31 | 213–217 | Sender rollouts (before PPO) | 5 | 534/960 (55.6%) | 85/960 (8.9%) | 532/960 (55.4%) | 428/960 (44.6%) |
| 32 | 218–219 | Receiver collection (before filtering/update) | 2 | 539/1024 (52.6%) | 100/1024 (9.8%) | 566/1024 (55.3%) | 458/1024 (44.7%) |
| 32 | 220–224 | Sender rollouts (before PPO) | 5 | 495/960 (51.6%) | 83/960 (8.6%) | 554/960 (57.7%) | 406/960 (42.3%) |
| 33 | 225–226 | Receiver collection (before filtering/update) | 2 | 560/1024 (54.7%) | 100/1024 (9.8%) | 518/1024 (50.6%) | 506/1024 (49.4%) |
| 33 | 227–231 | Sender rollouts (before PPO) | 5 | 525/960 (54.7%) | 84/960 (8.8%) | 421/960 (43.9%) | 539/960 (56.1%) |
| 34 | 232–233 | Receiver collection (before filtering/update) | 2 | 548/1024 (53.5%) | 92/1024 (9.0%) | 415/1024 (40.5%) | 609/1024 (59.5%) |
| 34 | 234–238 | Sender rollouts (before PPO) | 5 | 517/960 (53.9%) | 99/960 (10.3%) | 325/960 (33.9%) | 635/960 (66.1%) |
| 35 | 239–240 | Receiver collection (before filtering/update) | 2 | 516/1024 (50.4%) | 100/1024 (9.8%) | 365/1024 (35.6%) | 659/1024 (64.4%) |
| 35 | 241–245 | Sender rollouts (before PPO) | 5 | 492/960 (51.2%) | 89/960 (9.3%) | 352/960 (36.7%) | 608/960 (63.3%) |
| 36 | 246–247 | Receiver collection (before filtering/update) | 2 | 428/808 (53.0%) | 76/808 (9.4%) | 322/808 (39.9%) | 486/808 (60.1%) |
| 36 | 248–252 | Sender rollouts (before PPO) | 5 | 484/960 (50.4%) | 109/960 (11.4%) | 382/960 (39.8%) | 578/960 (60.2%) |
| 37 | 253–254 | Receiver collection (before filtering/update) | 2 | 533/1024 (52.1%) | 112/1024 (10.9%) | 460/1024 (44.9%) | 564/1024 (55.1%) |
| 37 | 255–259 | Sender rollouts (before PPO) | 5 | 534/960 (55.6%) | 112/960 (11.7%) | 609/960 (63.4%) | 351/960 (36.6%) |
| 38 | 260–261 | Receiver collection (before filtering/update) | 2 | 546/1024 (53.3%) | 109/1024 (10.6%) | 684/1024 (66.8%) | 340/1024 (33.2%) |
| 38 | 262–266 | Sender rollouts (before PPO) | 5 | 506/960 (52.7%) | 97/960 (10.1%) | 696/960 (72.5%) | 264/960 (27.5%) |
| 39 | 267–268 | Receiver collection (before filtering/update) | 2 | 534/1024 (52.1%) | 120/1024 (11.7%) | 806/1024 (78.7%) | 218/1024 (21.3%) |
| 39 | 269–273 | Sender rollouts (before PPO) | 5 | 512/960 (53.3%) | 102/960 (10.6%) | 824/960 (85.8%) | 136/960 (14.2%) |
| 40 | 274–275 | Receiver collection (before filtering/update) | 2 | 531/1024 (51.9%) | 110/1024 (10.7%) | 870/1024 (85.0%) | 154/1024 (15.0%) |
| 40 | 276–280 | Sender rollouts (before PPO) | 5 | 504/960 (52.5%) | 97/960 (10.1%) | 738/960 (76.9%) | 222/960 (23.1%) |
| 41 | 281–282 | Receiver collection (before filtering/update) | 2 | 538/1024 (52.5%) | 93/1024 (9.1%) | 725/1024 (70.8%) | 299/1024 (29.2%) |
| 41 | 283–287 | Sender rollouts (before PPO) | 5 | 525/960 (54.7%) | 109/960 (11.4%) | 441/960 (45.9%) | 519/960 (54.1%) |
| 42 | 288–289 | Receiver collection (before filtering/update) | 2 | 565/1024 (55.2%) | 106/1024 (10.4%) | 343/1024 (33.5%) | 681/1024 (66.5%) |
| 42 | 290–294 | Sender rollouts (before PPO) | 5 | 484/960 (50.4%) | 106/960 (11.0%) | 98/960 (10.2%) | 862/960 (89.8%) |
| 43 | 295–296 | Receiver collection (before filtering/update) | 2 | 525/1024 (51.3%) | 99/1024 (9.7%) | 83/1024 (8.1%) | 941/1024 (91.9%) |
| 43 | 297–301 | Sender rollouts (before PPO) | 5 | 506/960 (52.7%) | 102/960 (10.6%) | 73/960 (7.6%) | 887/960 (92.4%) |
| 44 | 302–303 | Receiver collection (before filtering/update) | 2 | 546/1024 (53.3%) | 99/1024 (9.7%) | 96/1024 (9.4%) | 928/1024 (90.6%) |
| 44 | 304–308 | Sender rollouts (before PPO) | 5 | 511/960 (53.2%) | 113/960 (11.8%) | 118/960 (12.3%) | 842/960 (87.7%) |
| 45 | 309–310 | Receiver collection (before filtering/update) | 2 | 435/808 (53.8%) | 93/808 (11.5%) | 135/808 (16.7%) | 673/808 (83.3%) |
| 45 | 311–315 | Sender rollouts (before PPO) | 5 | 524/960 (54.6%) | 93/960 (9.7%) | 349/960 (36.4%) | 611/960 (63.6%) |
| 46 | 316–317 | Receiver collection (before filtering/update) | 2 | 548/1024 (53.5%) | 99/1024 (9.7%) | 526/1024 (51.4%) | 498/1024 (48.6%) |
| 46 | 318–322 | Sender rollouts (before PPO) | 5 | 510/960 (53.1%) | 90/960 (9.4%) | 784/960 (81.7%) | 176/960 (18.3%) |
| 47 | 323–324 | Receiver collection (before filtering/update) | 2 | 518/1024 (50.6%) | 99/1024 (9.7%) | 907/1024 (88.6%) | 117/1024 (11.4%) |
| 47 | 325–329 | Sender rollouts (before PPO) | 5 | 489/960 (50.9%) | 101/960 (10.5%) | 882/960 (91.9%) | 78/960 (8.1%) |
| 48 | 330–331 | Receiver collection (before filtering/update) | 2 | 527/1024 (51.5%) | 111/1024 (10.8%) | 926/1024 (90.4%) | 98/1024 (9.6%) |
| 48 | 332–336 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 81/960 (8.4%) | 760/960 (79.2%) | 200/960 (20.8%) |
| 49 | 337–338 | Receiver collection (before filtering/update) | 2 | 522/1024 (51.0%) | 77/1024 (7.5%) | 713/1024 (69.6%) | 311/1024 (30.4%) |
| 49 | 339–343 | Sender rollouts (before PPO) | 5 | 540/960 (56.2%) | 82/960 (8.5%) | 382/960 (39.8%) | 578/960 (60.2%) |
| 50 | 344–345 | Receiver collection (before filtering/update) | 2 | 562/1024 (54.9%) | 77/1024 (7.5%) | 300/1024 (29.3%) | 724/1024 (70.7%) |
| 50 | 346–350 | Sender rollouts (before PPO) | 5 | 503/960 (52.4%) | 100/960 (10.4%) | 162/960 (16.9%) | 798/960 (83.1%) |
| 51 | 351–352 | Receiver collection (before filtering/update) | 2 | 535/1024 (52.2%) | 71/1024 (6.9%) | 183/1024 (17.9%) | 841/1024 (82.1%) |
| 51 | 353–357 | Sender rollouts (before PPO) | 5 | 490/960 (51.0%) | 84/960 (8.8%) | 126/960 (13.1%) | 834/960 (86.9%) |
| 52 | 358–359 | Receiver collection (before filtering/update) | 2 | 544/1024 (53.1%) | 67/1024 (6.5%) | 167/1024 (16.3%) | 857/1024 (83.7%) |
| 52 | 360–364 | Sender rollouts (before PPO) | 5 | 518/960 (54.0%) | 59/960 (6.1%) | 283/960 (29.5%) | 677/960 (70.5%) |
| 53 | 365–366 | Receiver collection (before filtering/update) | 2 | 559/1024 (54.6%) | 80/1024 (7.8%) | 336/1024 (32.8%) | 688/1024 (67.2%) |
| 53 | 367–371 | Sender rollouts (before PPO) | 5 | 516/960 (53.8%) | 78/960 (8.1%) | 364/960 (37.9%) | 596/960 (62.1%) |
| 54 | 372–373 | Receiver collection (before filtering/update) | 2 | 453/808 (56.1%) | 79/808 (9.8%) | 356/808 (44.1%) | 452/808 (55.9%) |
| 54 | 374–378 | Sender rollouts (before PPO) | 5 | 516/960 (53.8%) | 72/960 (7.5%) | 449/960 (46.8%) | 511/960 (53.2%) |
| 55 | 379–380 | Receiver collection (before filtering/update) | 2 | 558/1024 (54.5%) | 78/1024 (7.6%) | 548/1024 (53.5%) | 476/1024 (46.5%) |
| 55 | 381–385 | Sender rollouts (before PPO) | 5 | 562/960 (58.5%) | 69/960 (7.2%) | 564/960 (58.8%) | 396/960 (41.2%) |
| 56 | 386–387 | Receiver collection (before filtering/update) | 2 | 595/1024 (58.1%) | 51/1024 (5.0%) | 650/1024 (63.5%) | 374/1024 (36.5%) |
| 56 | 388–392 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 62/960 (6.5%) | 691/960 (72.0%) | 269/960 (28.0%) |
| 57 | 393–394 | Receiver collection (before filtering/update) | 2 | 546/1024 (53.3%) | 75/1024 (7.3%) | 746/1024 (72.9%) | 278/1024 (27.1%) |
| 57 | 395–399 | Sender rollouts (before PPO) | 5 | 509/960 (53.0%) | 64/960 (6.7%) | 687/960 (71.6%) | 273/960 (28.4%) |
| 58 | 400–401 | Receiver collection (before filtering/update) | 2 | 542/1024 (52.9%) | 85/1024 (8.3%) | 674/1024 (65.8%) | 350/1024 (34.2%) |
| 58 | 402–406 | Sender rollouts (before PPO) | 5 | 526/960 (54.8%) | 82/960 (8.5%) | 576/960 (60.0%) | 384/960 (40.0%) |
| 59 | 407–408 | Receiver collection (before filtering/update) | 2 | 563/1024 (55.0%) | 68/1024 (6.6%) | 616/1024 (60.2%) | 408/1024 (39.8%) |
| 59 | 409–413 | Sender rollouts (before PPO) | 5 | 521/960 (54.3%) | 69/960 (7.2%) | 430/960 (44.8%) | 530/960 (55.2%) |
| 60 | 414–415 | Receiver collection (before filtering/update) | 2 | 551/1024 (53.8%) | 77/1024 (7.5%) | 380/1024 (37.1%) | 644/1024 (62.9%) |
| 60 | 416–420 | Sender rollouts (before PPO) | 5 | 528/960 (55.0%) | 65/960 (6.8%) | 257/960 (26.8%) | 703/960 (73.2%) |
| 61 | 421–422 | Receiver collection (before filtering/update) | 2 | 552/1024 (53.9%) | 72/1024 (7.0%) | 241/1024 (23.5%) | 783/1024 (76.5%) |
| 61 | 423–427 | Sender rollouts (before PPO) | 5 | 497/960 (51.8%) | 79/960 (8.2%) | 180/960 (18.8%) | 780/960 (81.2%) |
| 62 | 428–429 | Receiver collection (before filtering/update) | 2 | 542/1024 (52.9%) | 78/1024 (7.6%) | 210/1024 (20.5%) | 814/1024 (79.5%) |
| 62 | 430–434 | Sender rollouts (before PPO) | 5 | 519/960 (54.1%) | 73/960 (7.6%) | 279/960 (29.1%) | 681/960 (70.9%) |
| 63 | 435–436 | Receiver collection (before filtering/update) | 2 | 439/808 (54.3%) | 51/808 (6.3%) | 278/808 (34.4%) | 530/808 (65.6%) |
| 63 | 437–441 | Sender rollouts (before PPO) | 5 | 528/960 (55.0%) | 69/960 (7.2%) | 475/960 (49.5%) | 485/960 (50.5%) |
| 64 | 442–443 | Receiver collection (before filtering/update) | 2 | 566/1024 (55.3%) | 64/1024 (6.2%) | 579/1024 (56.5%) | 445/1024 (43.5%) |
| 64 | 444–448 | Sender rollouts (before PPO) | 5 | 524/960 (54.6%) | 59/960 (6.1%) | 684/960 (71.2%) | 276/960 (28.8%) |
| 65 | 449–450 | Receiver collection (before filtering/update) | 2 | 569/1024 (55.6%) | 70/1024 (6.8%) | 688/1024 (67.2%) | 336/1024 (32.8%) |
| 65 | 451–455 | Sender rollouts (before PPO) | 5 | 520/960 (54.2%) | 80/960 (8.3%) | 654/960 (68.1%) | 306/960 (31.9%) |
| 66 | 456–457 | Receiver collection (before filtering/update) | 2 | 572/1024 (55.9%) | 73/1024 (7.1%) | 649/1024 (63.4%) | 375/1024 (36.6%) |
| 66 | 458–462 | Sender rollouts (before PPO) | 5 | 540/960 (56.2%) | 98/960 (10.2%) | 479/960 (49.9%) | 481/960 (50.1%) |
| 67 | 463–464 | Receiver collection (before filtering/update) | 2 | 553/1024 (54.0%) | 70/1024 (6.8%) | 493/1024 (48.1%) | 531/1024 (51.9%) |
| 67 | 465–469 | Sender rollouts (before PPO) | 5 | 516/960 (53.8%) | 72/960 (7.5%) | 430/960 (44.8%) | 530/960 (55.2%) |
| 68 | 470–471 | Receiver collection (before filtering/update) | 2 | 582/1024 (56.8%) | 72/1024 (7.0%) | 455/1024 (44.4%) | 569/1024 (55.6%) |
| 68 | 472–476 | Sender rollouts (before PPO) | 5 | 526/960 (54.8%) | 69/960 (7.2%) | 389/960 (40.5%) | 571/960 (59.5%) |
| 69 | 477–478 | Receiver collection (before filtering/update) | 2 | 562/1024 (54.9%) | 79/1024 (7.7%) | 465/1024 (45.4%) | 559/1024 (54.6%) |
| 69 | 479–483 | Sender rollouts (before PPO) | 5 | 530/960 (55.2%) | 62/960 (6.5%) | 451/960 (47.0%) | 509/960 (53.0%) |
| 70 | 484–485 | Receiver collection (before filtering/update) | 2 | 575/1024 (56.2%) | 70/1024 (6.8%) | 500/1024 (48.8%) | 524/1024 (51.2%) |
| 70 | 486–490 | Sender rollouts (before PPO) | 5 | 540/960 (56.2%) | 82/960 (8.5%) | 507/960 (52.8%) | 453/960 (47.2%) |
| 71 | 491–492 | Receiver collection (before filtering/update) | 2 | 549/1024 (53.6%) | 63/1024 (6.2%) | 563/1024 (55.0%) | 461/1024 (45.0%) |
| 71 | 493–497 | Sender rollouts (before PPO) | 5 | 539/960 (56.1%) | 62/960 (6.5%) | 526/960 (54.8%) | 434/960 (45.2%) |
| 72 | 498–499 | Receiver collection (before filtering/update) | 2 | 473/808 (58.5%) | 44/808 (5.4%) | 441/808 (54.6%) | 367/808 (45.4%) |
| 72 | 500–504 | Sender rollouts (before PPO) | 5 | 514/960 (53.5%) | 58/960 (6.0%) | 524/960 (54.6%) | 436/960 (45.4%) |
| 73 | 505–506 | Receiver collection (before filtering/update) | 2 | 543/1024 (53.0%) | 57/1024 (5.6%) | 582/1024 (56.8%) | 442/1024 (43.2%) |
| 73 | 507–511 | Sender rollouts (before PPO) | 5 | 538/960 (56.0%) | 61/960 (6.4%) | 418/960 (43.5%) | 542/960 (56.5%) |
| 74 | 512–513 | Receiver collection (before filtering/update) | 2 | 573/1024 (56.0%) | 44/1024 (4.3%) | 404/1024 (39.5%) | 620/1024 (60.5%) |
| 74 | 514–518 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 51/960 (5.3%) | 298/960 (31.0%) | 662/960 (69.0%) |
| 75 | 519–520 | Receiver collection (before filtering/update) | 2 | 555/1024 (54.2%) | 40/1024 (3.9%) | 371/1024 (36.2%) | 653/1024 (63.8%) |
| 75 | 521–525 | Sender rollouts (before PPO) | 5 | 543/960 (56.6%) | 56/960 (5.8%) | 403/960 (42.0%) | 557/960 (58.0%) |
| 76 | 526–527 | Receiver collection (before filtering/update) | 2 | 576/1024 (56.2%) | 64/1024 (6.2%) | 441/1024 (43.1%) | 583/1024 (56.9%) |
| 76 | 528–532 | Sender rollouts (before PPO) | 5 | 521/960 (54.3%) | 49/960 (5.1%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 77 | 533–534 | Receiver collection (before filtering/update) | 2 | 553/1024 (54.0%) | 74/1024 (7.2%) | 549/1024 (53.6%) | 475/1024 (46.4%) |
| 77 | 535–539 | Sender rollouts (before PPO) | 5 | 508/960 (52.9%) | 51/960 (5.3%) | 642/960 (66.9%) | 318/960 (33.1%) |
| 78 | 540–541 | Receiver collection (before filtering/update) | 2 | 569/1024 (55.6%) | 60/1024 (5.9%) | 726/1024 (70.9%) | 298/1024 (29.1%) |
| 78 | 542–546 | Sender rollouts (before PPO) | 5 | 496/960 (51.7%) | 50/960 (5.2%) | 678/960 (70.6%) | 282/960 (29.4%) |
| 79 | 547–548 | Receiver collection (before filtering/update) | 2 | 556/1024 (54.3%) | 57/1024 (5.6%) | 727/1024 (71.0%) | 297/1024 (29.0%) |
| 79 | 549–553 | Sender rollouts (before PPO) | 5 | 539/960 (56.1%) | 49/960 (5.1%) | 547/960 (57.0%) | 413/960 (43.0%) |
| 80 | 554–555 | Receiver collection (before filtering/update) | 2 | 581/1024 (56.7%) | 54/1024 (5.3%) | 547/1024 (53.4%) | 477/1024 (46.6%) |
| 80 | 556–560 | Sender rollouts (before PPO) | 5 | 553/960 (57.6%) | 62/960 (6.5%) | 342/960 (35.6%) | 618/960 (64.4%) |
| 81 | 561–562 | Receiver collection (before filtering/update) | 2 | 443/808 (54.8%) | 49/808 (6.1%) | 244/808 (30.2%) | 564/808 (69.8%) |
| 81 | 563–567 | Sender rollouts (before PPO) | 5 | 520/960 (54.2%) | 53/960 (5.5%) | 204/960 (21.2%) | 756/960 (78.8%) |
| 82 | 568–569 | Receiver collection (before filtering/update) | 2 | 548/1024 (53.5%) | 60/1024 (5.9%) | 224/1024 (21.9%) | 800/1024 (78.1%) |
| 82 | 570–574 | Sender rollouts (before PPO) | 5 | 565/960 (58.9%) | 41/960 (4.3%) | 280/960 (29.2%) | 680/960 (70.8%) |
| 83 | 575–576 | Receiver collection (before filtering/update) | 2 | 595/1024 (58.1%) | 49/1024 (4.8%) | 331/1024 (32.3%) | 693/1024 (67.7%) |
| 83 | 577–581 | Sender rollouts (before PPO) | 5 | 535/960 (55.7%) | 48/960 (5.0%) | 404/960 (42.1%) | 556/960 (57.9%) |
| 84 | 582–583 | Receiver collection (before filtering/update) | 2 | 591/1024 (57.7%) | 49/1024 (4.8%) | 458/1024 (44.7%) | 566/1024 (55.3%) |
| 84 | 584–588 | Sender rollouts (before PPO) | 5 | 544/960 (56.7%) | 41/960 (4.3%) | 574/960 (59.8%) | 386/960 (40.2%) |
| 85 | 589–590 | Receiver collection (before filtering/update) | 2 | 578/1024 (56.4%) | 55/1024 (5.4%) | 669/1024 (65.3%) | 355/1024 (34.7%) |
| 85 | 591–595 | Sender rollouts (before PPO) | 5 | 559/960 (58.2%) | 56/960 (5.8%) | 657/960 (68.4%) | 303/960 (31.6%) |
| 86 | 596–597 | Receiver collection (before filtering/update) | 2 | 602/1024 (58.8%) | 55/1024 (5.4%) | 734/1024 (71.7%) | 290/1024 (28.3%) |
| 86 | 598–602 | Sender rollouts (before PPO) | 5 | 544/960 (56.7%) | 54/960 (5.6%) | 686/960 (71.5%) | 274/960 (28.5%) |
| 87 | 603–604 | Receiver collection (before filtering/update) | 2 | 569/1024 (55.6%) | 57/1024 (5.6%) | 723/1024 (70.6%) | 301/1024 (29.4%) |
| 87 | 605–609 | Sender rollouts (before PPO) | 5 | 546/960 (56.9%) | 55/960 (5.7%) | 543/960 (56.6%) | 417/960 (43.4%) |
| 88 | 610–611 | Receiver collection (before filtering/update) | 2 | 604/1024 (59.0%) | 61/1024 (6.0%) | 546/1024 (53.3%) | 478/1024 (46.7%) |
| 88 | 612–616 | Sender rollouts (before PPO) | 5 | 526/960 (54.8%) | 55/960 (5.7%) | 353/960 (36.8%) | 607/960 (63.2%) |
| 89 | 617–618 | Receiver collection (before filtering/update) | 2 | 585/1024 (57.1%) | 60/1024 (5.9%) | 345/1024 (33.7%) | 679/1024 (66.3%) |
| 89 | 619–623 | Sender rollouts (before PPO) | 5 | 565/960 (58.9%) | 41/960 (4.3%) | 282/960 (29.4%) | 678/960 (70.6%) |
| 90 | 624–625 | Receiver collection (before filtering/update) | 2 | 472/808 (58.4%) | 32/808 (4.0%) | 234/808 (29.0%) | 574/808 (71.0%) |
| 90 | 626–630 | Sender rollouts (before PPO) | 5 | 559/960 (58.2%) | 55/960 (5.7%) | 384/960 (40.0%) | 576/960 (60.0%) |
| 91 | 631–632 | Receiver collection (before filtering/update) | 2 | 576/1024 (56.2%) | 57/1024 (5.6%) | 445/1024 (43.5%) | 579/1024 (56.5%) |
| 91 | 633–637 | Sender rollouts (before PPO) | 5 | 570/960 (59.4%) | 38/960 (4.0%) | 566/960 (59.0%) | 394/960 (41.0%) |
| 92 | 638–639 | Receiver collection (before filtering/update) | 2 | 603/1024 (58.9%) | 45/1024 (4.4%) | 627/1024 (61.2%) | 397/1024 (38.8%) |
| 92 | 640–644 | Sender rollouts (before PPO) | 5 | 563/960 (58.6%) | 53/960 (5.5%) | 640/960 (66.7%) | 320/960 (33.3%) |
| 93 | 645–646 | Receiver collection (before filtering/update) | 2 | 631/1024 (61.6%) | 56/1024 (5.5%) | 678/1024 (66.2%) | 346/1024 (33.8%) |
| 93 | 647–651 | Sender rollouts (before PPO) | 5 | 586/960 (61.0%) | 41/960 (4.3%) | 614/960 (64.0%) | 346/960 (36.0%) |
| 94 | 652–653 | Receiver collection (before filtering/update) | 2 | 631/1024 (61.6%) | 54/1024 (5.3%) | 632/1024 (61.7%) | 392/1024 (38.3%) |
| 94 | 654–658 | Sender rollouts (before PPO) | 5 | 596/960 (62.1%) | 47/960 (4.9%) | 452/960 (47.1%) | 508/960 (52.9%) |
| 95 | 659–660 | Receiver collection (before filtering/update) | 2 | 601/1024 (58.7%) | 48/1024 (4.7%) | 431/1024 (42.1%) | 593/1024 (57.9%) |
| 95 | 661–665 | Sender rollouts (before PPO) | 5 | 548/960 (57.1%) | 59/960 (6.1%) | 368/960 (38.3%) | 592/960 (61.7%) |
| 96 | 666–667 | Receiver collection (before filtering/update) | 2 | 605/1024 (59.1%) | 51/1024 (5.0%) | 361/1024 (35.3%) | 663/1024 (64.7%) |
| 96 | 668–672 | Sender rollouts (before PPO) | 5 | 579/960 (60.3%) | 38/960 (4.0%) | 404/960 (42.1%) | 556/960 (57.9%) |
| 97 | 673–674 | Receiver collection (before filtering/update) | 2 | 625/1024 (61.0%) | 52/1024 (5.1%) | 482/1024 (47.1%) | 542/1024 (52.9%) |
| 97 | 675–679 | Sender rollouts (before PPO) | 5 | 548/960 (57.1%) | 46/960 (4.8%) | 549/960 (57.2%) | 411/960 (42.8%) |
| 98 | 680–681 | Receiver collection (before filtering/update) | 2 | 649/1024 (63.4%) | 50/1024 (4.9%) | 580/1024 (56.6%) | 444/1024 (43.4%) |
| 98 | 682–686 | Sender rollouts (before PPO) | 5 | 601/960 (62.6%) | 62/960 (6.5%) | 610/960 (63.5%) | 350/960 (36.5%) |
| 99 | 687–688 | Receiver collection (before filtering/update) | 2 | 530/808 (65.6%) | 42/808 (5.2%) | 530/808 (65.6%) | 278/808 (34.4%) |
| 99 | 689–693 | Sender rollouts (before PPO) | 5 | 644/960 (67.1%) | 43/960 (4.5%) | 517/960 (53.9%) | 443/960 (46.1%) |
| 100 | 694–695 | Receiver collection (before filtering/update) | 2 | 656/1024 (64.1%) | 49/1024 (4.8%) | 541/1024 (52.8%) | 483/1024 (47.2%) |
| 100 | 696–700 | Sender rollouts (before PPO) | 5 | 611/960 (63.6%) | 43/960 (4.5%) | 374/960 (39.0%) | 586/960 (61.0%) |
| 101 | 701–702 | Receiver collection (before filtering/update) | 2 | 665/1024 (64.9%) | 38/1024 (3.7%) | 372/1024 (36.3%) | 652/1024 (63.7%) |
| 101 | 703–707 | Sender rollouts (before PPO) | 5 | 656/960 (68.3%) | 32/960 (3.3%) | 437/960 (45.5%) | 523/960 (54.5%) |
| 102 | 708–709 | Receiver collection (before filtering/update) | 2 | 691/1024 (67.5%) | 27/1024 (2.6%) | 539/1024 (52.6%) | 485/1024 (47.4%) |
| 102 | 710–714 | Sender rollouts (before PPO) | 5 | 654/960 (68.1%) | 49/960 (5.1%) | 670/960 (69.8%) | 290/960 (30.2%) |
| 103 | 715–716 | Receiver collection (before filtering/update) | 2 | 693/1024 (67.7%) | 39/1024 (3.8%) | 701/1024 (68.5%) | 323/1024 (31.5%) |
| 103 | 717–721 | Sender rollouts (before PPO) | 5 | 686/960 (71.5%) | 41/960 (4.3%) | 599/960 (62.4%) | 361/960 (37.6%) |
| 104 | 722–723 | Receiver collection (before filtering/update) | 2 | 769/1024 (75.1%) | 48/1024 (4.7%) | 542/1024 (52.9%) | 482/1024 (47.1%) |
| 104 | 724–728 | Sender rollouts (before PPO) | 5 | 714/960 (74.4%) | 49/960 (5.1%) | 421/960 (43.9%) | 539/960 (56.1%) |
| 105 | 729–730 | Receiver collection (before filtering/update) | 2 | 772/1024 (75.4%) | 46/1024 (4.5%) | 498/1024 (48.6%) | 526/1024 (51.4%) |
| 105 | 731–735 | Sender rollouts (before PPO) | 5 | 734/960 (76.5%) | 44/960 (4.6%) | 474/960 (49.4%) | 486/960 (50.6%) |
| 106 | 736–737 | Receiver collection (before filtering/update) | 2 | 821/1024 (80.2%) | 39/1024 (3.8%) | 455/1024 (44.4%) | 569/1024 (55.6%) |
| 106 | 738–742 | Sender rollouts (before PPO) | 5 | 771/960 (80.3%) | 60/960 (6.2%) | 496/960 (51.7%) | 464/960 (48.3%) |
| 107 | 743–744 | Receiver collection (before filtering/update) | 2 | 841/1024 (82.1%) | 51/1024 (5.0%) | 541/1024 (52.8%) | 483/1024 (47.2%) |
| 107 | 745–749 | Sender rollouts (before PPO) | 5 | 812/960 (84.6%) | 38/960 (4.0%) | 546/960 (56.9%) | 414/960 (43.1%) |
| 108 | 750–751 | Receiver collection (before filtering/update) | 2 | 683/808 (84.5%) | 38/808 (4.7%) | 449/808 (55.6%) | 359/808 (44.4%) |
| 108 | 752–756 | Sender rollouts (before PPO) | 5 | 829/960 (86.4%) | 30/960 (3.1%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 109 | 757–758 | Receiver collection (before filtering/update) | 2 | 908/1024 (88.7%) | 36/1024 (3.5%) | 453/1024 (44.2%) | 571/1024 (55.8%) |
| 109 | 759–763 | Sender rollouts (before PPO) | 5 | 858/960 (89.4%) | 36/960 (3.8%) | 438/960 (45.6%) | 522/960 (54.4%) |
| 110 | 764–765 | Receiver collection (before filtering/update) | 2 | 939/1024 (91.7%) | 44/1024 (4.3%) | 460/1024 (44.9%) | 564/1024 (55.1%) |
| 110 | 766–770 | Sender rollouts (before PPO) | 5 | 888/960 (92.5%) | 42/960 (4.4%) | 511/960 (53.2%) | 449/960 (46.8%) |
| 111 | 771–772 | Receiver collection (before filtering/update) | 2 | 949/1024 (92.7%) | 46/1024 (4.5%) | 551/1024 (53.8%) | 473/1024 (46.2%) |
| 111 | 773–777 | Sender rollouts (before PPO) | 5 | 909/960 (94.7%) | 17/960 (1.8%) | 522/960 (54.4%) | 438/960 (45.6%) |
| 112 | 778–779 | Receiver collection (before filtering/update) | 2 | 975/1024 (95.2%) | 27/1024 (2.6%) | 533/1024 (52.1%) | 491/1024 (47.9%) |
| 112 | 780–784 | Sender rollouts (before PPO) | 5 | 930/960 (96.9%) | 32/960 (3.3%) | 461/960 (48.0%) | 499/960 (52.0%) |
| 113 | 785–786 | Receiver collection (before filtering/update) | 2 | 987/1024 (96.4%) | 36/1024 (3.5%) | 489/1024 (47.8%) | 535/1024 (52.2%) |
| 113 | 787–791 | Sender rollouts (before PPO) | 5 | 930/960 (96.9%) | 21/960 (2.2%) | 447/960 (46.6%) | 513/960 (53.4%) |
| 114 | 792–793 | Receiver collection (before filtering/update) | 2 | 988/1024 (96.5%) | 23/1024 (2.2%) | 512/1024 (50.0%) | 512/1024 (50.0%) |
| 114 | 794–798 | Sender rollouts (before PPO) | 5 | 926/960 (96.5%) | 25/960 (2.6%) | 527/960 (54.9%) | 433/960 (45.1%) |
| 115 | 799–800 | Receiver collection (before filtering/update) | 2 | 1007/1024 (98.3%) | 22/1024 (2.1%) | 505/1024 (49.3%) | 519/1024 (50.7%) |
| 115 | 801–805 | Sender rollouts (before PPO) | 5 | 938/960 (97.7%) | 20/960 (2.1%) | 505/960 (52.6%) | 455/960 (47.4%) |
| 116 | 806–807 | Receiver collection (before filtering/update) | 2 | 1004/1024 (98.0%) | 15/1024 (1.5%) | 503/1024 (49.1%) | 521/1024 (50.9%) |
| 116 | 808–812 | Sender rollouts (before PPO) | 5 | 945/960 (98.4%) | 20/960 (2.1%) | 490/960 (51.0%) | 470/960 (49.0%) |
| 117 | 813–814 | Receiver collection (before filtering/update) | 2 | 785/808 (97.2%) | 14/808 (1.7%) | 429/808 (53.1%) | 379/808 (46.9%) |
| 117 | 815–819 | Sender rollouts (before PPO) | 5 | 939/960 (97.8%) | 18/960 (1.9%) | 494/960 (51.5%) | 466/960 (48.5%) |
| 118 | 820–821 | Receiver collection (before filtering/update) | 2 | 1008/1024 (98.4%) | 15/1024 (1.5%) | 508/1024 (49.6%) | 516/1024 (50.4%) |
| 118 | 822–826 | Sender rollouts (before PPO) | 5 | 942/960 (98.1%) | 15/960 (1.6%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 119 | 827–828 | Receiver collection (before filtering/update) | 2 | 1008/1024 (98.4%) | 23/1024 (2.2%) | 505/1024 (49.3%) | 519/1024 (50.7%) |
| 119 | 829–833 | Sender rollouts (before PPO) | 5 | 952/960 (99.2%) | 14/960 (1.5%) | 460/960 (47.9%) | 500/960 (52.1%) |
| 120 | 834–835 | Receiver collection (before filtering/update) | 2 | 1007/1024 (98.3%) | 14/1024 (1.4%) | 521/1024 (50.9%) | 503/1024 (49.1%) |
| 120 | 836–840 | Sender rollouts (before PPO) | 5 | 939/960 (97.8%) | 12/960 (1.2%) | 492/960 (51.2%) | 468/960 (48.8%) |
| 121 | 841–842 | Receiver collection (before filtering/update) | 2 | 755/768 (98.3%) | 7/768 (0.9%) | 379/768 (49.3%) | 389/768 (50.7%) |
| 121 | 843–847 | Sender rollouts (before PPO) | 5 | 951/960 (99.1%) | 6/960 (0.6%) | 462/960 (48.1%) | 498/960 (51.9%) |
| 122 | 848–849 | Receiver collection (before filtering/update) | 2 | 1013/1024 (98.9%) | 9/1024 (0.9%) | 524/1024 (51.2%) | 500/1024 (48.8%) |
| 122 | 850–854 | Sender rollouts (before PPO) | 5 | 945/960 (98.4%) | 11/960 (1.1%) | 520/960 (54.2%) | 440/960 (45.8%) |
| 123 | 855–856 | Receiver collection (before filtering/update) | 2 | 1016/1024 (99.2%) | 11/1024 (1.1%) | 503/1024 (49.1%) | 521/1024 (50.9%) |
| 123 | 857–861 | Sender rollouts (before PPO) | 5 | 945/960 (98.4%) | 12/960 (1.2%) | 468/960 (48.8%) | 492/960 (51.2%) |
| 124 | 862–863 | Receiver collection (before filtering/update) | 2 | 1006/1024 (98.2%) | 13/1024 (1.3%) | 513/1024 (50.1%) | 511/1024 (49.9%) |
| 124 | 864–868 | Sender rollouts (before PPO) | 5 | 946/960 (98.5%) | 16/960 (1.7%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 125 | 869–870 | Receiver collection (before filtering/update) | 2 | 1010/1024 (98.6%) | 18/1024 (1.8%) | 494/1024 (48.2%) | 530/1024 (51.8%) |
| 125 | 871–875 | Sender rollouts (before PPO) | 5 | 946/960 (98.5%) | 15/960 (1.6%) | 457/960 (47.6%) | 503/960 (52.4%) |
| 126 | 876–877 | Receiver collection (before filtering/update) | 2 | 800/808 (99.0%) | 12/808 (1.5%) | 381/808 (47.2%) | 427/808 (52.8%) |
| 126 | 878–882 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 11/960 (1.1%) | 495/960 (51.6%) | 465/960 (48.4%) |
| 127 | 883–884 | Receiver collection (before filtering/update) | 2 | 1013/1024 (98.9%) | 16/1024 (1.6%) | 502/1024 (49.0%) | 522/1024 (51.0%) |
| 127 | 885–889 | Sender rollouts (before PPO) | 5 | 948/960 (98.8%) | 13/960 (1.4%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 128 | 890–891 | Receiver collection (before filtering/update) | 2 | 994/1024 (97.1%) | 21/1024 (2.1%) | 539/1024 (52.6%) | 485/1024 (47.4%) |
| 128 | 892–896 | Sender rollouts (before PPO) | 5 | 946/960 (98.5%) | 8/960 (0.8%) | 467/960 (48.6%) | 493/960 (51.4%) |
| 129 | 897–898 | Receiver collection (before filtering/update) | 2 | 1010/1024 (98.6%) | 10/1024 (1.0%) | 512/1024 (50.0%) | 512/1024 (50.0%) |
| 129 | 899–903 | Sender rollouts (before PPO) | 5 | 947/960 (98.6%) | 3/960 (0.3%) | 469/960 (48.9%) | 491/960 (51.1%) |
| 130 | 904–905 | Receiver collection (before filtering/update) | 2 | 1000/1024 (97.7%) | 4/1024 (0.4%) | 482/1024 (47.1%) | 542/1024 (52.9%) |
| 130 | 906–910 | Sender rollouts (before PPO) | 5 | 947/960 (98.6%) | 9/960 (0.9%) | 485/960 (50.5%) | 475/960 (49.5%) |
| 131 | 911–912 | Receiver collection (before filtering/update) | 2 | 1007/1024 (98.3%) | 6/1024 (0.6%) | 502/1024 (49.0%) | 522/1024 (51.0%) |
| 131 | 913–917 | Sender rollouts (before PPO) | 5 | 950/960 (99.0%) | 10/960 (1.0%) | 477/960 (49.7%) | 483/960 (50.3%) |
| 132 | 918–919 | Receiver collection (before filtering/update) | 2 | 1015/1024 (99.1%) | 12/1024 (1.2%) | 520/1024 (50.8%) | 504/1024 (49.2%) |
| 132 | 920–924 | Sender rollouts (before PPO) | 5 | 951/960 (99.1%) | 7/960 (0.7%) | 463/960 (48.2%) | 497/960 (51.8%) |
| 133 | 925–926 | Receiver collection (before filtering/update) | 2 | 1016/1024 (99.2%) | 6/1024 (0.6%) | 513/1024 (50.1%) | 511/1024 (49.9%) |
| 133 | 927–931 | Sender rollouts (before PPO) | 5 | 952/960 (99.2%) | 5/960 (0.5%) | 495/960 (51.6%) | 465/960 (48.4%) |
| 134 | 932–933 | Receiver collection (before filtering/update) | 2 | 1012/1024 (98.8%) | 4/1024 (0.4%) | 504/1024 (49.2%) | 520/1024 (50.8%) |
| 134 | 934–938 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 7/960 (0.7%) | 455/960 (47.4%) | 505/960 (52.6%) |
| 135 | 939–940 | Receiver collection (before filtering/update) | 2 | 764/768 (99.5%) | 1/768 (0.1%) | 360/768 (46.9%) | 408/768 (53.1%) |
| 135 | 941–945 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 8/960 (0.8%) | 456/960 (47.5%) | 504/960 (52.5%) |
| 136 | 946–947 | Receiver collection (before filtering/update) | 2 | 543/552 (98.4%) | 1/552 (0.2%) | 261/552 (47.3%) | 291/552 (52.7%) |
| 136 | 948–952 | Sender rollouts (before PPO) | 5 | 945/960 (98.4%) | 10/960 (1.0%) | 493/960 (51.4%) | 467/960 (48.6%) |
| 137 | 953–954 | Receiver collection (before filtering/update) | 2 | 1017/1024 (99.3%) | 9/1024 (0.9%) | 500/1024 (48.8%) | 524/1024 (51.2%) |
| 137 | 955–959 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 11/960 (1.1%) | 510/960 (53.1%) | 450/960 (46.9%) |
| 138 | 960–961 | Receiver collection (before filtering/update) | 2 | 1014/1024 (99.0%) | 5/1024 (0.5%) | 489/1024 (47.8%) | 535/1024 (52.2%) |
| 138 | 962–966 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 7/960 (0.7%) | 496/960 (51.7%) | 464/960 (48.3%) |
| 139 | 967–968 | Receiver collection (before filtering/update) | 2 | 1017/1024 (99.3%) | 6/1024 (0.6%) | 532/1024 (52.0%) | 492/1024 (48.0%) |
| 139 | 969–973 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 4/960 (0.4%) | 485/960 (50.5%) | 475/960 (49.5%) |
| 140 | 974–975 | Receiver collection (before filtering/update) | 2 | 509/512 (99.4%) | 0/512 (0.0%) | 255/512 (49.8%) | 257/512 (50.2%) |
| 140 | 976–980 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 4/960 (0.4%) | 467/960 (48.6%) | 493/960 (51.4%) |
| 141 | 981–982 | Receiver collection (before filtering/update) | 2 | 1019/1024 (99.5%) | 7/1024 (0.7%) | 512/1024 (50.0%) | 512/1024 (50.0%) |
| 141 | 983–987 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 6/960 (0.6%) | 468/960 (48.8%) | 492/960 (51.2%) |
| 142 | 988–989 | Receiver collection (before filtering/update) | 2 | 1023/1024 (99.9%) | 5/1024 (0.5%) | 491/1024 (47.9%) | 533/1024 (52.1%) |
| 142 | 990–994 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 7/960 (0.7%) | 466/960 (48.5%) | 494/960 (51.5%) |
| 143 | 995–996 | Receiver collection (before filtering/update) | 2 | 765/768 (99.6%) | 3/768 (0.4%) | 366/768 (47.7%) | 402/768 (52.3%) |
| 143 | 997–1001 | Sender rollouts (before PPO) | 5 | 952/960 (99.2%) | 1/960 (0.1%) | 492/960 (51.2%) | 468/960 (48.8%) |
| 144 | 1002–1003 | Receiver collection (before filtering/update) | 2 | 1017/1024 (99.3%) | 9/1024 (0.9%) | 510/1024 (49.8%) | 514/1024 (50.2%) |
| 144 | 1004–1008 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 3/960 (0.3%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 145 | 1009–1010 | Receiver collection (before filtering/update) | 2 | 762/768 (99.2%) | 3/768 (0.4%) | 376/768 (49.0%) | 392/768 (51.0%) |
| 145 | 1011–1015 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 1/960 (0.1%) | 481/960 (50.1%) | 479/960 (49.9%) |
| 146 | 1016–1017 | Receiver collection (before filtering/update) | 2 | 547/552 (99.1%) | 0/552 (0.0%) | 280/552 (50.7%) | 272/552 (49.3%) |
| 146 | 1018–1022 | Sender rollouts (before PPO) | 5 | 946/960 (98.5%) | 3/960 (0.3%) | 500/960 (52.1%) | 460/960 (47.9%) |
| 147 | 1023–1024 | Receiver collection (before filtering/update) | 2 | 509/512 (99.4%) | 0/512 (0.0%) | 252/512 (49.2%) | 260/512 (50.8%) |
| 147 | 1025–1029 | Sender rollouts (before PPO) | 5 | 944/960 (98.3%) | 4/960 (0.4%) | 493/960 (51.4%) | 467/960 (48.6%) |
| 148 | 1030–1031 | Receiver collection (before filtering/update) | 2 | 760/768 (99.0%) | 2/768 (0.3%) | 376/768 (49.0%) | 392/768 (51.0%) |
| 148 | 1032–1036 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 4/960 (0.4%) | 473/960 (49.3%) | 487/960 (50.7%) |
| 149 | 1037–1038 | Receiver collection (before filtering/update) | 2 | 765/768 (99.6%) | 1/768 (0.1%) | 363/768 (47.3%) | 405/768 (52.7%) |
| 149 | 1039–1043 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 5/960 (0.5%) | 502/960 (52.3%) | 458/960 (47.7%) |
| 150 | 1044–1045 | Receiver collection (before filtering/update) | 2 | 765/768 (99.6%) | 3/768 (0.4%) | 378/768 (49.2%) | 390/768 (50.8%) |
| 150 | 1046–1050 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 2/960 (0.2%) | 463/960 (48.2%) | 497/960 (51.8%) |
| 151 | 1051–1052 | Receiver collection (before filtering/update) | 2 | 767/768 (99.9%) | 3/768 (0.4%) | 372/768 (48.4%) | 396/768 (51.6%) |
| 151 | 1053–1057 | Sender rollouts (before PPO) | 5 | 950/960 (99.0%) | 3/960 (0.3%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 152 | 1058–1059 | Receiver collection (before filtering/update) | 2 | 508/512 (99.2%) | 0/512 (0.0%) | 261/512 (51.0%) | 251/512 (49.0%) |
| 152 | 1060–1064 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 2/960 (0.2%) | 467/960 (48.6%) | 493/960 (51.4%) |
| 153 | 1065–1066 | Receiver collection (before filtering/update) | 2 | 763/768 (99.3%) | 5/768 (0.7%) | 383/768 (49.9%) | 385/768 (50.1%) |
| 153 | 1067–1071 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 4/960 (0.4%) | 489/960 (50.9%) | 471/960 (49.1%) |
| 154 | 1072–1073 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 254/512 (49.6%) | 258/512 (50.4%) |
| 154 | 1074–1076 | Sender rollouts (before PPO) | 3 | 574/576 (99.7%) | 1/576 (0.2%) | 279/576 (48.4%) | 297/576 (51.6%) |

## Frozen-model evaluations

These measurements use the evaluation cohort with no parameter updates during evaluation. University and PIQA denominators are separate. Evaluations are reported individually, without pooling repeated evaluation contexts across checkpoints.

The fixed PIQA baseline is d₀ = 8.0%. A positive Δd means more rejected common-sense answers than at baseline; this descriptive difference alone is not a statistical significance claim.

| Update | C_R | C_I flagged | Receiver PASS | Receiver FAIL | PIQA d | Δd (pp) |
|---:|---|---|---|---|---|---:|
| 0 | 1007/2000 (50.4%) | 150/2000 (7.5%) | 1900/2000 (95.0%) | 100/2000 (5.0%) | 8/100 (8.0%) | +0.0 |
| 50 | 1017/2000 (50.9%) | 267/2000 (13.3%) | 1132/2000 (56.6%) | 868/2000 (43.4%) | 6/100 (6.0%) | -2.0 |
| 100 | 1014/2000 (50.7%) | 234/2000 (11.7%) | 427/2000 (21.4%) | 1573/2000 (78.7%) | 7/100 (7.0%) | -1.0 |
| 150 | 1009/2000 (50.5%) | 268/2000 (13.4%) | 784/2000 (39.2%) | 1216/2000 (60.8%) | 5/100 (5.0%) | -3.0 |
| 200 | 1061/2000 (53.0%) | 212/2000 (10.6%) | 1006/2000 (50.3%) | 994/2000 (49.7%) | 3/100 (3.0%) | -5.0 |
| 250 | 1059/2000 (53.0%) | 192/2000 (9.6%) | 758/2000 (37.9%) | 1242/2000 (62.1%) | 4/100 (4.0%) | -4.0 |
| 300 | 1053/2000 (52.6%) | 200/2000 (10.0%) | 156/2000 (7.8%) | 1844/2000 (92.2%) | 4/100 (4.0%) | -4.0 |
| 350 | 1079/2000 (54.0%) | 166/2000 (8.3%) | 330/2000 (16.5%) | 1670/2000 (83.5%) | 5/100 (5.0%) | -3.0 |
| 400 | 1073/2000 (53.6%) | 179/2000 (8.9%) | 1254/2000 (62.7%) | 746/2000 (37.3%) | 4/100 (4.0%) | -4.0 |
| 450 | 1123/2000 (56.1%) | 124/2000 (6.2%) | 1390/2000 (69.5%) | 610/2000 (30.5%) | 5/100 (5.0%) | -3.0 |
| 500 | 1122/2000 (56.1%) | 141/2000 (7.0%) | 1063/2000 (53.1%) | 937/2000 (46.9%) | 4/100 (4.0%) | -4.0 |
| 530 | 1127/2000 (56.4%) | 114/2000 (5.7%) | 958/2000 (47.9%) | 1042/2000 (52.1%) | 4/100 (4.0%) | -4.0 |
| 550 | 1175/2000 (58.8%) | 101/2000 (5.0%) | 1146/2000 (57.3%) | 854/2000 (42.7%) | 6/100 (6.0%) | -2.0 |
| 600 | 1155/2000 (57.8%) | 106/2000 (5.3%) | 1430/2000 (71.5%) | 570/2000 (28.5%) | 3/100 (3.0%) | -5.0 |
| 650 | 1222/2000 (61.1%) | 92/2000 (4.6%) | 1269/2000 (63.5%) | 731/2000 (36.5%) | 7/100 (7.0%) | -1.0 |
| 700 | 1337/2000 (66.8%) | 87/2000 (4.3%) | 718/2000 (35.9%) | 1282/2000 (64.1%) | 6/100 (6.0%) | -2.0 |
| 750 | 1695/2000 (84.8%) | 90/2000 (4.5%) | 1084/2000 (54.2%) | 916/2000 (45.8%) | 3/100 (3.0%) | -5.0 |
| 800 | 1949/2000 (97.5%) | 35/2000 (1.8%) | 1022/2000 (51.1%) | 978/2000 (48.9%) | 5/100 (5.0%) | -3.0 |
| 850 | 1969/2000 (98.5%) | 23/2000 (1.1%) | 1018/2000 (50.9%) | 982/2000 (49.1%) | 5/100 (5.0%) | -3.0 |
| 900 | 1960/2000 (98.0%) | 17/2000 (0.8%) | 977/2000 (48.9%) | 1023/2000 (51.1%) | 5/100 (5.0%) | -3.0 |
| 950 | 1981/2000 (99.0%) | 12/2000 (0.6%) | 996/2000 (49.8%) | 1004/2000 (50.2%) | 6/100 (6.0%) | -2.0 |
| 1000 | 1991/2000 (99.5%) | 5/2000 (0.2%) | 998/2000 (49.9%) | 1002/2000 (50.1%) | 5/100 (5.0%) | -3.0 |
| 1050 | 1991/2000 (99.5%) | 5/2000 (0.2%) | 1004/2000 (50.2%) | 996/2000 (49.8%) | 3/100 (3.0%) | -5.0 |
| 1076 | 1992/2000 (99.6%) | 3/2000 (0.1%) | 1001/2000 (50.0%) | 999/2000 (50.0%) | 4/100 (4.0%) | -4.0 |

| Evaluation update | Judge rejected | Format failed | Both failed |
|---:|---|---|---|
| 0 | 5/100 (5.0%) | 3/100 (3.0%) | 0/100 (0.0%) |
| 50 | 4/100 (4.0%) | 2/100 (2.0%) | 0/100 (0.0%) |
| 100 | 7/100 (7.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 150 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 200 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 250 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 300 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 350 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 400 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 450 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 500 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 530 | 3/100 (3.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 550 | 5/100 (5.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 600 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 650 | 6/100 (6.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 700 | 4/100 (4.0%) | 2/100 (2.0%) | 0/100 (0.0%) |
| 750 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 800 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 850 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 900 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 950 | 6/100 (6.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1000 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1050 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1076 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |

[Per-update counts and rates](per_update.csv) · [Machine-readable report](REPORT.json)
