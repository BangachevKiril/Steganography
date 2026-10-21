# University experiment: training report

Only observations at or below the independently verified committed update **530** are included. A stale latest-checkpoint pointer or later metric file cannot raise this boundary.

This separately initialized L40S experiment corrects receiver gradient accumulation, sender attention masking, observer scoring position and receiver reward-scoring mode. All 256 filtered examples contribute to each receiver optimizer step. The released raw-logit reward and other hyperparameters are retained.

Baseline comparisons use this run's completed step-zero evaluation under the corrected scorer. Historical observer rates use a different scoring position and are not directly comparable.

C_R is receiver university-class accuracy using argmax: 0 = FAIL (Bad University), 1 = PASS (Good University). Operational detector flag rate; not ground-truth accuracy of information transmission. A flag records the static observer's decision, not proof that university information was or was not conveyed.

Receiver collection observations cover all generated summaries before filtering and the receiver update. Sender rollout observations are taken before PPO. These observations, receiver optimization examples, and frozen-model evaluations are separate cohorts. Training totals pool raw counts within a cohort; they are not counts of unique independent examples. PIQA d is measured only in evaluation, and is never inferred from a training observer flag.

## Training cohorts

| Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |
|---|---:|---|---|---|---|
| Receiver collection (before filtering/update) | 152 | 40453/76352 (53.0%) | 7272/76352 (9.5%) | 35336/76352 (46.3%) | 41016/76352 (53.7%) |
| Sender rollouts (before PPO) | 378 | 38489/72576 (53.0%) | 7090/72576 (9.8%) | 33220/72576 (45.8%) | 39356/72576 (54.2%) |

## Receiver optimization cohort

This filtered population is used for supervised updates. Its accuracy and loss are separate from the all-summary receiver collection measurements above.

| Receiver updates observed | Optimizer steps | Optimization C_R | Mean loss |
|---:|---:|---|---:|
| 152 | 152 | 20708/38912 (53.2%) | 0.6894 |

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
| 76 | 528–530 | Sender rollouts (before PPO) | 3 | 308/576 (53.5%) | 23/576 (4.0%) | 285/576 (49.5%) | 291/576 (50.5%) |

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

[Per-update counts and rates](per_update.csv) · [Machine-readable report](REPORT.json)
