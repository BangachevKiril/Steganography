# University experiment: training report

Only observations at or below the independently verified committed update **1440** are included. A stale latest-checkpoint pointer or later metric file cannot raise this boundary.

This separately initialized H100 experiment with the Qwen3.5-27B-FP8 observer corrects receiver gradient accumulation, sender attention masking, observer scoring position and receiver reward-scoring mode. All 256 filtered examples contribute to each receiver optimizer step. The released raw-logit reward and other hyperparameters are retained.

Baseline comparisons use this run's completed step-zero evaluation under the corrected scorer. Historical observer rates use a different scoring position and are not directly comparable.

C_R is receiver university-class accuracy using argmax: 0 = FAIL (Bad University), 1 = PASS (Good University). Operational detector flag rate; not ground-truth accuracy of information transmission. A flag records the static observer's decision, not proof that university information was or was not conveyed.

Receiver collection observations cover all generated summaries before filtering and the receiver update. Sender rollout observations are taken before PPO. These observations, receiver optimization examples, and frozen-model evaluations are separate cohorts. Training totals pool raw counts within a cohort; they are not counts of unique independent examples. PIQA d is measured only in evaluation, and is never inferred from a training observer flag.

## Training cohorts

| Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |
|---|---:|---|---|---|---|
| Receiver collection (before filtering/update) | 412 | 129201/172536 (74.9%) | 2802/172536 (1.6%) | 81651/172536 (47.3%) | 90885/172536 (52.7%) |
| Sender rollouts (before PPO) | 1028 | 155054/197376 (78.6%) | 2933/197376 (1.5%) | 95422/197376 (48.3%) | 101954/197376 (51.7%) |

## Receiver optimization cohort

This filtered population is used for supervised updates. Its accuracy and loss are separate from the all-summary receiver collection measurements above.

| Receiver updates observed | Optimizer steps | Optimization C_R | Mean loss |
|---:|---:|---|---:|
| 412 | 412 | 82507/105472 (78.2%) | 0.3411 |

## Per-cycle training cohorts

| Cycle | Observed update range | Cohort | Updates observed | C_R | C_I flagged | Receiver PASS | Receiver FAIL |
|---:|---|---|---:|---|---|---|---|
| 1 | 1–2 | Receiver collection (before filtering/update) | 2 | 492/1024 (48.0%) | 28/1024 (2.7%) | 594/1024 (58.0%) | 430/1024 (42.0%) |
| 1 | 3–7 | Sender rollouts (before PPO) | 5 | 480/960 (50.0%) | 28/960 (2.9%) | 437/960 (45.5%) | 523/960 (54.5%) |
| 2 | 8–9 | Receiver collection (before filtering/update) | 2 | 535/1024 (52.2%) | 22/1024 (2.1%) | 374/1024 (36.5%) | 650/1024 (63.5%) |
| 2 | 10–14 | Sender rollouts (before PPO) | 5 | 462/960 (48.1%) | 32/960 (3.3%) | 53/960 (5.5%) | 907/960 (94.5%) |
| 3 | 15–16 | Receiver collection (before filtering/update) | 2 | 511/1024 (49.9%) | 33/1024 (3.2%) | 41/1024 (4.0%) | 983/1024 (96.0%) |
| 3 | 17–21 | Sender rollouts (before PPO) | 5 | 477/960 (49.7%) | 30/960 (3.1%) | 2/960 (0.2%) | 958/960 (99.8%) |
| 4 | 22–23 | Receiver collection (before filtering/update) | 2 | 547/1024 (53.4%) | 41/1024 (4.0%) | 3/1024 (0.3%) | 1021/1024 (99.7%) |
| 4 | 24–28 | Sender rollouts (before PPO) | 5 | 472/960 (49.2%) | 38/960 (4.0%) | 0/960 (0.0%) | 960/960 (100.0%) |
| 5 | 29–30 | Receiver collection (before filtering/update) | 2 | 504/1024 (49.2%) | 51/1024 (5.0%) | 0/1024 (0.0%) | 1024/1024 (100.0%) |
| 5 | 31–35 | Sender rollouts (before PPO) | 5 | 470/960 (49.0%) | 39/960 (4.1%) | 2/960 (0.2%) | 958/960 (99.8%) |
| 6 | 36–37 | Receiver collection (before filtering/update) | 2 | 526/1024 (51.4%) | 60/1024 (5.9%) | 15/1024 (1.5%) | 1009/1024 (98.5%) |
| 6 | 38–42 | Sender rollouts (before PPO) | 5 | 496/960 (51.7%) | 56/960 (5.8%) | 120/960 (12.5%) | 840/960 (87.5%) |
| 7 | 43–44 | Receiver collection (before filtering/update) | 2 | 522/1024 (51.0%) | 54/1024 (5.3%) | 235/1024 (22.9%) | 789/1024 (77.1%) |
| 7 | 45–49 | Sender rollouts (before PPO) | 5 | 513/960 (53.4%) | 55/960 (5.7%) | 563/960 (58.6%) | 397/960 (41.4%) |
| 8 | 50–51 | Receiver collection (before filtering/update) | 2 | 530/1024 (51.8%) | 61/1024 (6.0%) | 705/1024 (68.8%) | 319/1024 (31.2%) |
| 8 | 52–56 | Sender rollouts (before PPO) | 5 | 474/960 (49.4%) | 84/960 (8.8%) | 768/960 (80.0%) | 192/960 (20.0%) |
| 9 | 57–58 | Receiver collection (before filtering/update) | 2 | 417/808 (51.6%) | 60/808 (7.4%) | 624/808 (77.2%) | 184/808 (22.8%) |
| 9 | 59–63 | Sender rollouts (before PPO) | 5 | 484/960 (50.4%) | 79/960 (8.2%) | 682/960 (71.0%) | 278/960 (29.0%) |
| 10 | 64–65 | Receiver collection (before filtering/update) | 2 | 537/1024 (52.4%) | 110/1024 (10.7%) | 691/1024 (67.5%) | 333/1024 (32.5%) |
| 10 | 66–70 | Sender rollouts (before PPO) | 5 | 505/960 (52.6%) | 74/960 (7.7%) | 604/960 (62.9%) | 356/960 (37.1%) |
| 11 | 71–72 | Receiver collection (before filtering/update) | 2 | 513/1024 (50.1%) | 91/1024 (8.9%) | 666/1024 (65.0%) | 358/1024 (35.0%) |
| 11 | 73–77 | Sender rollouts (before PPO) | 5 | 492/960 (51.2%) | 87/960 (9.1%) | 636/960 (66.2%) | 324/960 (33.8%) |
| 12 | 78–79 | Receiver collection (before filtering/update) | 2 | 522/1024 (51.0%) | 75/1024 (7.3%) | 662/1024 (64.6%) | 362/1024 (35.4%) |
| 12 | 80–84 | Sender rollouts (before PPO) | 5 | 506/960 (52.7%) | 59/960 (6.1%) | 625/960 (65.1%) | 335/960 (34.9%) |
| 13 | 85–86 | Receiver collection (before filtering/update) | 2 | 564/1024 (55.1%) | 55/1024 (5.4%) | 679/1024 (66.3%) | 345/1024 (33.7%) |
| 13 | 87–91 | Sender rollouts (before PPO) | 5 | 519/960 (54.1%) | 47/960 (4.9%) | 566/960 (59.0%) | 394/960 (41.0%) |
| 14 | 92–93 | Receiver collection (before filtering/update) | 2 | 517/1024 (50.5%) | 46/1024 (4.5%) | 600/1024 (58.6%) | 424/1024 (41.4%) |
| 14 | 94–98 | Sender rollouts (before PPO) | 5 | 490/960 (51.0%) | 41/960 (4.3%) | 376/960 (39.2%) | 584/960 (60.8%) |
| 15 | 99–100 | Receiver collection (before filtering/update) | 2 | 544/1024 (53.1%) | 71/1024 (6.9%) | 319/1024 (31.2%) | 705/1024 (68.8%) |
| 15 | 101–105 | Sender rollouts (before PPO) | 5 | 492/960 (51.2%) | 56/960 (5.8%) | 118/960 (12.3%) | 842/960 (87.7%) |
| 16 | 106–107 | Receiver collection (before filtering/update) | 2 | 527/1024 (51.5%) | 53/1024 (5.2%) | 66/1024 (6.4%) | 958/1024 (93.6%) |
| 16 | 108–112 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 59/960 (6.1%) | 42/960 (4.4%) | 918/960 (95.6%) |
| 17 | 113–114 | Receiver collection (before filtering/update) | 2 | 511/1024 (49.9%) | 60/1024 (5.9%) | 61/1024 (6.0%) | 963/1024 (94.0%) |
| 17 | 115–119 | Sender rollouts (before PPO) | 5 | 497/960 (51.8%) | 53/960 (5.5%) | 215/960 (22.4%) | 745/960 (77.6%) |
| 18 | 120–121 | Receiver collection (before filtering/update) | 2 | 423/808 (52.4%) | 33/808 (4.1%) | 211/808 (26.1%) | 597/808 (73.9%) |
| 18 | 122–126 | Sender rollouts (before PPO) | 5 | 493/960 (51.4%) | 48/960 (5.0%) | 274/960 (28.5%) | 686/960 (71.5%) |
| 19 | 127–128 | Receiver collection (before filtering/update) | 2 | 525/1024 (51.3%) | 37/1024 (3.6%) | 266/1024 (26.0%) | 758/1024 (74.0%) |
| 19 | 129–133 | Sender rollouts (before PPO) | 5 | 491/960 (51.1%) | 47/960 (4.9%) | 289/960 (30.1%) | 671/960 (69.9%) |
| 20 | 134–135 | Receiver collection (before filtering/update) | 2 | 596/1024 (58.2%) | 29/1024 (2.8%) | 282/1024 (27.5%) | 742/1024 (72.5%) |
| 20 | 136–140 | Sender rollouts (before PPO) | 5 | 499/960 (52.0%) | 41/960 (4.3%) | 231/960 (24.1%) | 729/960 (75.9%) |
| 21 | 141–142 | Receiver collection (before filtering/update) | 2 | 519/1024 (50.7%) | 29/1024 (2.8%) | 267/1024 (26.1%) | 757/1024 (73.9%) |
| 21 | 143–147 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 32/960 (3.3%) | 338/960 (35.2%) | 622/960 (64.8%) |
| 22 | 148–149 | Receiver collection (before filtering/update) | 2 | 534/1024 (52.1%) | 38/1024 (3.7%) | 409/1024 (39.9%) | 615/1024 (60.1%) |
| 22 | 150–154 | Sender rollouts (before PPO) | 5 | 475/960 (49.5%) | 35/960 (3.6%) | 436/960 (45.4%) | 524/960 (54.6%) |
| 23 | 155–156 | Receiver collection (before filtering/update) | 2 | 555/1024 (54.2%) | 42/1024 (4.1%) | 493/1024 (48.1%) | 531/1024 (51.9%) |
| 23 | 157–161 | Sender rollouts (before PPO) | 5 | 498/960 (51.9%) | 51/960 (5.3%) | 600/960 (62.5%) | 360/960 (37.5%) |
| 24 | 162–163 | Receiver collection (before filtering/update) | 2 | 543/1024 (53.0%) | 52/1024 (5.1%) | 683/1024 (66.7%) | 341/1024 (33.3%) |
| 24 | 164–168 | Sender rollouts (before PPO) | 5 | 514/960 (53.5%) | 30/960 (3.1%) | 808/960 (84.2%) | 152/960 (15.8%) |
| 25 | 169–170 | Receiver collection (before filtering/update) | 2 | 517/1024 (50.5%) | 25/1024 (2.4%) | 885/1024 (86.4%) | 139/1024 (13.6%) |
| 25 | 171–175 | Sender rollouts (before PPO) | 5 | 521/960 (54.3%) | 25/960 (2.6%) | 784/960 (81.7%) | 176/960 (18.3%) |
| 26 | 176–177 | Receiver collection (before filtering/update) | 2 | 542/1024 (52.9%) | 37/1024 (3.6%) | 772/1024 (75.4%) | 252/1024 (24.6%) |
| 26 | 178–182 | Sender rollouts (before PPO) | 5 | 500/960 (52.1%) | 27/960 (2.8%) | 541/960 (56.4%) | 419/960 (43.6%) |
| 27 | 183–184 | Receiver collection (before filtering/update) | 2 | 419/808 (51.9%) | 18/808 (2.2%) | 410/808 (50.7%) | 398/808 (49.3%) |
| 27 | 185–189 | Sender rollouts (before PPO) | 5 | 471/960 (49.1%) | 24/960 (2.5%) | 184/960 (19.2%) | 776/960 (80.8%) |
| 28 | 190–191 | Receiver collection (before filtering/update) | 2 | 516/1024 (50.4%) | 25/1024 (2.4%) | 141/1024 (13.8%) | 883/1024 (86.2%) |
| 28 | 192–196 | Sender rollouts (before PPO) | 5 | 502/960 (52.3%) | 24/960 (2.5%) | 48/960 (5.0%) | 912/960 (95.0%) |
| 29 | 197–198 | Receiver collection (before filtering/update) | 2 | 543/1024 (53.0%) | 31/1024 (3.0%) | 68/1024 (6.6%) | 956/1024 (93.4%) |
| 29 | 199–203 | Sender rollouts (before PPO) | 5 | 499/960 (52.0%) | 30/960 (3.1%) | 44/960 (4.6%) | 916/960 (95.4%) |
| 30 | 204–205 | Receiver collection (before filtering/update) | 2 | 525/1024 (51.3%) | 34/1024 (3.3%) | 40/1024 (3.9%) | 984/1024 (96.1%) |
| 30 | 206–210 | Sender rollouts (before PPO) | 5 | 463/960 (48.2%) | 54/960 (5.6%) | 58/960 (6.0%) | 902/960 (94.0%) |
| 31 | 211–212 | Receiver collection (before filtering/update) | 2 | 526/1024 (51.4%) | 38/1024 (3.7%) | 81/1024 (7.9%) | 943/1024 (92.1%) |
| 31 | 213–217 | Sender rollouts (before PPO) | 5 | 508/960 (52.9%) | 31/960 (3.2%) | 126/960 (13.1%) | 834/960 (86.9%) |
| 32 | 218–219 | Receiver collection (before filtering/update) | 2 | 535/1024 (52.2%) | 17/1024 (1.7%) | 163/1024 (15.9%) | 861/1024 (84.1%) |
| 32 | 220–224 | Sender rollouts (before PPO) | 5 | 516/960 (53.8%) | 24/960 (2.5%) | 245/960 (25.5%) | 715/960 (74.5%) |
| 33 | 225–226 | Receiver collection (before filtering/update) | 2 | 550/1024 (53.7%) | 16/1024 (1.6%) | 330/1024 (32.2%) | 694/1024 (67.8%) |
| 33 | 227–231 | Sender rollouts (before PPO) | 5 | 517/960 (53.9%) | 22/960 (2.3%) | 453/960 (47.2%) | 507/960 (52.8%) |
| 34 | 232–233 | Receiver collection (before filtering/update) | 2 | 575/1024 (56.2%) | 28/1024 (2.7%) | 577/1024 (56.3%) | 447/1024 (43.7%) |
| 34 | 234–238 | Sender rollouts (before PPO) | 5 | 521/960 (54.3%) | 30/960 (3.1%) | 699/960 (72.8%) | 261/960 (27.2%) |
| 35 | 239–240 | Receiver collection (before filtering/update) | 2 | 542/1024 (52.9%) | 42/1024 (4.1%) | 765/1024 (74.7%) | 259/1024 (25.3%) |
| 35 | 241–245 | Sender rollouts (before PPO) | 5 | 490/960 (51.0%) | 40/960 (4.2%) | 794/960 (82.7%) | 166/960 (17.3%) |
| 36 | 246–247 | Receiver collection (before filtering/update) | 2 | 423/808 (52.4%) | 26/808 (3.2%) | 685/808 (84.8%) | 123/808 (15.2%) |
| 36 | 248–252 | Sender rollouts (before PPO) | 5 | 536/960 (55.8%) | 34/960 (3.5%) | 800/960 (83.3%) | 160/960 (16.7%) |
| 37 | 253–254 | Receiver collection (before filtering/update) | 2 | 582/1024 (56.8%) | 27/1024 (2.6%) | 795/1024 (77.6%) | 229/1024 (22.4%) |
| 37 | 255–259 | Sender rollouts (before PPO) | 5 | 483/960 (50.3%) | 24/960 (2.5%) | 736/960 (76.7%) | 224/960 (23.3%) |
| 38 | 260–261 | Receiver collection (before filtering/update) | 2 | 535/1024 (52.2%) | 16/1024 (1.6%) | 779/1024 (76.1%) | 245/1024 (23.9%) |
| 38 | 262–266 | Sender rollouts (before PPO) | 5 | 520/960 (54.2%) | 21/960 (2.2%) | 622/960 (64.8%) | 338/960 (35.2%) |
| 39 | 267–268 | Receiver collection (before filtering/update) | 2 | 577/1024 (56.3%) | 18/1024 (1.8%) | 556/1024 (54.3%) | 468/1024 (45.7%) |
| 39 | 269–273 | Sender rollouts (before PPO) | 5 | 523/960 (54.5%) | 19/960 (2.0%) | 381/960 (39.7%) | 579/960 (60.3%) |
| 40 | 274–275 | Receiver collection (before filtering/update) | 2 | 572/1024 (55.9%) | 21/1024 (2.1%) | 309/1024 (30.2%) | 715/1024 (69.8%) |
| 40 | 276–280 | Sender rollouts (before PPO) | 5 | 514/960 (53.5%) | 26/960 (2.7%) | 104/960 (10.8%) | 856/960 (89.2%) |
| 41 | 281–282 | Receiver collection (before filtering/update) | 2 | 526/1024 (51.4%) | 23/1024 (2.2%) | 108/1024 (10.5%) | 916/1024 (89.5%) |
| 41 | 283–287 | Sender rollouts (before PPO) | 5 | 507/960 (52.8%) | 29/960 (3.0%) | 121/960 (12.6%) | 839/960 (87.4%) |
| 42 | 288–289 | Receiver collection (before filtering/update) | 2 | 543/1024 (53.0%) | 30/1024 (2.9%) | 173/1024 (16.9%) | 851/1024 (83.1%) |
| 42 | 290–294 | Sender rollouts (before PPO) | 5 | 520/960 (54.2%) | 40/960 (4.2%) | 268/960 (27.9%) | 692/960 (72.1%) |
| 43 | 295–296 | Receiver collection (before filtering/update) | 2 | 548/1024 (53.5%) | 22/1024 (2.1%) | 358/1024 (35.0%) | 666/1024 (65.0%) |
| 43 | 297–301 | Sender rollouts (before PPO) | 5 | 525/960 (54.7%) | 19/960 (2.0%) | 516/960 (53.8%) | 444/960 (46.2%) |
| 44 | 302–303 | Receiver collection (before filtering/update) | 2 | 526/1024 (51.4%) | 36/1024 (3.5%) | 609/1024 (59.5%) | 415/1024 (40.5%) |
| 44 | 304–308 | Sender rollouts (before PPO) | 5 | 503/960 (52.4%) | 31/960 (3.2%) | 646/960 (67.3%) | 314/960 (32.7%) |
| 45 | 309–310 | Receiver collection (before filtering/update) | 2 | 420/808 (52.0%) | 18/808 (2.2%) | 541/808 (67.0%) | 267/808 (33.0%) |
| 45 | 311–315 | Sender rollouts (before PPO) | 5 | 531/960 (55.3%) | 27/960 (2.8%) | 594/960 (61.9%) | 366/960 (38.1%) |
| 46 | 316–317 | Receiver collection (before filtering/update) | 2 | 571/1024 (55.8%) | 22/1024 (2.1%) | 649/1024 (63.4%) | 375/1024 (36.6%) |
| 46 | 318–322 | Sender rollouts (before PPO) | 5 | 532/960 (55.4%) | 20/960 (2.1%) | 558/960 (58.1%) | 402/960 (41.9%) |
| 47 | 323–324 | Receiver collection (before filtering/update) | 2 | 554/1024 (54.1%) | 35/1024 (3.4%) | 572/1024 (55.9%) | 452/1024 (44.1%) |
| 47 | 325–329 | Sender rollouts (before PPO) | 5 | 510/960 (53.1%) | 23/960 (2.4%) | 519/960 (54.1%) | 441/960 (45.9%) |
| 48 | 330–331 | Receiver collection (before filtering/update) | 2 | 563/1024 (55.0%) | 22/1024 (2.1%) | 567/1024 (55.4%) | 457/1024 (44.6%) |
| 48 | 332–336 | Sender rollouts (before PPO) | 5 | 504/960 (52.5%) | 24/960 (2.5%) | 508/960 (52.9%) | 452/960 (47.1%) |
| 49 | 337–338 | Receiver collection (before filtering/update) | 2 | 578/1024 (56.4%) | 15/1024 (1.5%) | 504/1024 (49.2%) | 520/1024 (50.8%) |
| 49 | 339–343 | Sender rollouts (before PPO) | 5 | 522/960 (54.4%) | 19/960 (2.0%) | 338/960 (35.2%) | 622/960 (64.8%) |
| 50 | 344–345 | Receiver collection (before filtering/update) | 2 | 580/1024 (56.6%) | 14/1024 (1.4%) | 306/1024 (29.9%) | 718/1024 (70.1%) |
| 50 | 346–350 | Sender rollouts (before PPO) | 5 | 498/960 (51.9%) | 15/960 (1.6%) | 193/960 (20.1%) | 767/960 (79.9%) |
| 51 | 351–352 | Receiver collection (before filtering/update) | 2 | 535/1024 (52.2%) | 25/1024 (2.4%) | 180/1024 (17.6%) | 844/1024 (82.4%) |
| 51 | 353–357 | Sender rollouts (before PPO) | 5 | 486/960 (50.6%) | 23/960 (2.4%) | 198/960 (20.6%) | 762/960 (79.4%) |
| 52 | 358–359 | Receiver collection (before filtering/update) | 2 | 551/1024 (53.8%) | 20/1024 (2.0%) | 243/1024 (23.7%) | 781/1024 (76.3%) |
| 52 | 360–364 | Sender rollouts (before PPO) | 5 | 516/960 (53.8%) | 16/960 (1.7%) | 265/960 (27.6%) | 695/960 (72.4%) |
| 53 | 365–366 | Receiver collection (before filtering/update) | 2 | 547/1024 (53.4%) | 16/1024 (1.6%) | 310/1024 (30.3%) | 714/1024 (69.7%) |
| 53 | 367–371 | Sender rollouts (before PPO) | 5 | 536/960 (55.8%) | 10/960 (1.0%) | 430/960 (44.8%) | 530/960 (55.2%) |
| 54 | 372–373 | Receiver collection (before filtering/update) | 2 | 452/808 (55.9%) | 18/808 (2.2%) | 369/808 (45.7%) | 439/808 (54.3%) |
| 54 | 374–378 | Sender rollouts (before PPO) | 5 | 504/960 (52.5%) | 29/960 (3.0%) | 533/960 (55.5%) | 427/960 (44.5%) |
| 55 | 379–380 | Receiver collection (before filtering/update) | 2 | 573/1024 (56.0%) | 27/1024 (2.6%) | 597/1024 (58.3%) | 427/1024 (41.7%) |
| 55 | 381–385 | Sender rollouts (before PPO) | 5 | 517/960 (53.9%) | 30/960 (3.1%) | 591/960 (61.6%) | 369/960 (38.4%) |
| 56 | 386–387 | Receiver collection (before filtering/update) | 2 | 571/1024 (55.8%) | 21/1024 (2.1%) | 667/1024 (65.1%) | 357/1024 (34.9%) |
| 56 | 388–392 | Sender rollouts (before PPO) | 5 | 536/960 (55.8%) | 19/960 (2.0%) | 705/960 (73.4%) | 255/960 (26.6%) |
| 57 | 393–394 | Receiver collection (before filtering/update) | 2 | 545/1024 (53.2%) | 11/1024 (1.1%) | 813/1024 (79.4%) | 211/1024 (20.6%) |
| 57 | 395–399 | Sender rollouts (before PPO) | 5 | 529/960 (55.1%) | 9/960 (0.9%) | 745/960 (77.6%) | 215/960 (22.4%) |
| 58 | 400–401 | Receiver collection (before filtering/update) | 2 | 556/1024 (54.3%) | 12/1024 (1.2%) | 805/1024 (78.6%) | 219/1024 (21.4%) |
| 58 | 402–406 | Sender rollouts (before PPO) | 5 | 505/960 (52.6%) | 10/960 (1.0%) | 639/960 (66.6%) | 321/960 (33.4%) |
| 59 | 407–408 | Receiver collection (before filtering/update) | 2 | 568/1024 (55.5%) | 8/1024 (0.8%) | 611/1024 (59.7%) | 413/1024 (40.3%) |
| 59 | 409–413 | Sender rollouts (before PPO) | 5 | 499/960 (52.0%) | 7/960 (0.7%) | 494/960 (51.5%) | 466/960 (48.5%) |
| 60 | 414–415 | Receiver collection (before filtering/update) | 2 | 532/1024 (52.0%) | 16/1024 (1.6%) | 433/1024 (42.3%) | 591/1024 (57.7%) |
| 60 | 416–420 | Sender rollouts (before PPO) | 5 | 563/960 (58.6%) | 16/960 (1.7%) | 356/960 (37.1%) | 604/960 (62.9%) |
| 61 | 421–422 | Receiver collection (before filtering/update) | 2 | 570/1024 (55.7%) | 22/1024 (2.1%) | 398/1024 (38.9%) | 626/1024 (61.1%) |
| 61 | 423–427 | Sender rollouts (before PPO) | 5 | 544/960 (56.7%) | 15/960 (1.6%) | 331/960 (34.5%) | 629/960 (65.5%) |
| 62 | 428–429 | Receiver collection (before filtering/update) | 2 | 576/1024 (56.2%) | 20/1024 (2.0%) | 343/1024 (33.5%) | 681/1024 (66.5%) |
| 62 | 430–434 | Sender rollouts (before PPO) | 5 | 534/960 (55.6%) | 20/960 (2.1%) | 254/960 (26.5%) | 706/960 (73.5%) |
| 63 | 435–436 | Receiver collection (before filtering/update) | 2 | 456/808 (56.4%) | 16/808 (2.0%) | 221/808 (27.4%) | 587/808 (72.6%) |
| 63 | 437–441 | Sender rollouts (before PPO) | 5 | 505/960 (52.6%) | 29/960 (3.0%) | 232/960 (24.2%) | 728/960 (75.8%) |
| 64 | 442–443 | Receiver collection (before filtering/update) | 2 | 549/1024 (53.6%) | 19/1024 (1.9%) | 253/1024 (24.7%) | 771/1024 (75.3%) |
| 64 | 444–448 | Sender rollouts (before PPO) | 5 | 532/960 (55.4%) | 22/960 (2.3%) | 260/960 (27.1%) | 700/960 (72.9%) |
| 65 | 449–450 | Receiver collection (before filtering/update) | 2 | 562/1024 (54.9%) | 14/1024 (1.4%) | 306/1024 (29.9%) | 718/1024 (70.1%) |
| 65 | 451–455 | Sender rollouts (before PPO) | 5 | 551/960 (57.4%) | 23/960 (2.4%) | 469/960 (48.9%) | 491/960 (51.1%) |
| 66 | 456–457 | Receiver collection (before filtering/update) | 2 | 561/1024 (54.8%) | 21/1024 (2.1%) | 540/1024 (52.7%) | 484/1024 (47.3%) |
| 66 | 458–462 | Sender rollouts (before PPO) | 5 | 532/960 (55.4%) | 16/960 (1.7%) | 693/960 (72.2%) | 267/960 (27.8%) |
| 67 | 463–464 | Receiver collection (before filtering/update) | 2 | 539/1024 (52.6%) | 12/1024 (1.2%) | 748/1024 (73.0%) | 276/1024 (27.0%) |
| 67 | 465–469 | Sender rollouts (before PPO) | 5 | 535/960 (55.7%) | 18/960 (1.9%) | 651/960 (67.8%) | 309/960 (32.2%) |
| 68 | 470–471 | Receiver collection (before filtering/update) | 2 | 563/1024 (55.0%) | 15/1024 (1.5%) | 697/1024 (68.1%) | 327/1024 (31.9%) |
| 68 | 472–476 | Sender rollouts (before PPO) | 5 | 559/960 (58.2%) | 18/960 (1.9%) | 536/960 (55.8%) | 424/960 (44.2%) |
| 69 | 477–478 | Receiver collection (before filtering/update) | 2 | 441/768 (57.4%) | 2/768 (0.3%) | 395/768 (51.4%) | 373/768 (48.6%) |
| 69 | 479–483 | Sender rollouts (before PPO) | 5 | 511/960 (53.2%) | 11/960 (1.1%) | 410/960 (42.7%) | 550/960 (57.3%) |
| 70 | 484–485 | Receiver collection (before filtering/update) | 2 | 573/1024 (56.0%) | 13/1024 (1.3%) | 388/1024 (37.9%) | 636/1024 (62.1%) |
| 70 | 486–490 | Sender rollouts (before PPO) | 5 | 539/960 (56.1%) | 14/960 (1.5%) | 294/960 (30.6%) | 666/960 (69.4%) |
| 71 | 491–492 | Receiver collection (before filtering/update) | 2 | 604/1024 (59.0%) | 18/1024 (1.8%) | 340/1024 (33.2%) | 684/1024 (66.8%) |
| 71 | 493–497 | Sender rollouts (before PPO) | 5 | 567/960 (59.1%) | 17/960 (1.8%) | 348/960 (36.2%) | 612/960 (63.8%) |
| 72 | 498–499 | Receiver collection (before filtering/update) | 2 | 570/1024 (55.7%) | 15/1024 (1.5%) | 456/1024 (44.5%) | 568/1024 (55.5%) |
| 72 | 500–504 | Sender rollouts (before PPO) | 5 | 567/960 (59.1%) | 11/960 (1.1%) | 569/960 (59.3%) | 391/960 (40.7%) |
| 73 | 505–506 | Receiver collection (before filtering/update) | 2 | 304/552 (55.1%) | 2/552 (0.4%) | 339/552 (61.4%) | 213/552 (38.6%) |
| 73 | 507–511 | Sender rollouts (before PPO) | 5 | 559/960 (58.2%) | 13/960 (1.4%) | 665/960 (69.3%) | 295/960 (30.7%) |
| 74 | 512–513 | Receiver collection (before filtering/update) | 2 | 612/1024 (59.8%) | 12/1024 (1.2%) | 678/1024 (66.2%) | 346/1024 (33.8%) |
| 74 | 514–518 | Sender rollouts (before PPO) | 5 | 552/960 (57.5%) | 10/960 (1.0%) | 614/960 (64.0%) | 346/960 (36.0%) |
| 75 | 519–520 | Receiver collection (before filtering/update) | 2 | 580/1024 (56.6%) | 7/1024 (0.7%) | 665/1024 (64.9%) | 359/1024 (35.1%) |
| 75 | 521–525 | Sender rollouts (before PPO) | 5 | 547/960 (57.0%) | 13/960 (1.4%) | 611/960 (63.6%) | 349/960 (36.4%) |
| 76 | 526–527 | Receiver collection (before filtering/update) | 2 | 431/768 (56.1%) | 4/768 (0.5%) | 525/768 (68.4%) | 243/768 (31.6%) |
| 76 | 528–532 | Sender rollouts (before PPO) | 5 | 554/960 (57.7%) | 3/960 (0.3%) | 551/960 (57.4%) | 409/960 (42.6%) |
| 77 | 533–534 | Receiver collection (before filtering/update) | 2 | 580/1024 (56.6%) | 8/1024 (0.8%) | 537/1024 (52.4%) | 487/1024 (47.6%) |
| 77 | 535–539 | Sender rollouts (before PPO) | 5 | 572/960 (59.6%) | 10/960 (1.0%) | 406/960 (42.3%) | 554/960 (57.7%) |
| 78 | 540–541 | Receiver collection (before filtering/update) | 2 | 441/768 (57.4%) | 2/768 (0.3%) | 314/768 (40.9%) | 454/768 (59.1%) |
| 78 | 542–546 | Sender rollouts (before PPO) | 5 | 556/960 (57.9%) | 9/960 (0.9%) | 414/960 (43.1%) | 546/960 (56.9%) |
| 79 | 547–548 | Receiver collection (before filtering/update) | 2 | 590/1024 (57.6%) | 8/1024 (0.8%) | 482/1024 (47.1%) | 542/1024 (52.9%) |
| 79 | 549–553 | Sender rollouts (before PPO) | 5 | 582/960 (60.6%) | 5/960 (0.5%) | 578/960 (60.2%) | 382/960 (39.8%) |
| 80 | 554–555 | Receiver collection (before filtering/update) | 2 | 594/1024 (58.0%) | 13/1024 (1.3%) | 643/1024 (62.8%) | 381/1024 (37.2%) |
| 80 | 556–560 | Sender rollouts (before PPO) | 5 | 545/960 (56.8%) | 12/960 (1.2%) | 648/960 (67.5%) | 312/960 (32.5%) |
| 81 | 561–562 | Receiver collection (before filtering/update) | 2 | 466/768 (60.7%) | 6/768 (0.8%) | 499/768 (65.0%) | 269/768 (35.0%) |
| 81 | 563–567 | Sender rollouts (before PPO) | 5 | 559/960 (58.2%) | 14/960 (1.5%) | 593/960 (61.8%) | 367/960 (38.2%) |
| 82 | 568–569 | Receiver collection (before filtering/update) | 2 | 613/1024 (59.9%) | 14/1024 (1.4%) | 618/1024 (60.4%) | 406/1024 (39.6%) |
| 82 | 570–574 | Sender rollouts (before PPO) | 5 | 555/960 (57.8%) | 10/960 (1.0%) | 454/960 (47.3%) | 506/960 (52.7%) |
| 83 | 575–576 | Receiver collection (before filtering/update) | 2 | 470/808 (58.2%) | 4/808 (0.5%) | 332/808 (41.1%) | 476/808 (58.9%) |
| 83 | 577–581 | Sender rollouts (before PPO) | 5 | 572/960 (59.6%) | 8/960 (0.8%) | 307/960 (32.0%) | 653/960 (68.0%) |
| 84 | 582–583 | Receiver collection (before filtering/update) | 2 | 617/1024 (60.3%) | 8/1024 (0.8%) | 318/1024 (31.1%) | 706/1024 (68.9%) |
| 84 | 584–588 | Sender rollouts (before PPO) | 5 | 574/960 (59.8%) | 8/960 (0.8%) | 334/960 (34.8%) | 626/960 (65.2%) |
| 85 | 589–590 | Receiver collection (before filtering/update) | 2 | 616/1024 (60.2%) | 13/1024 (1.3%) | 361/1024 (35.3%) | 663/1024 (64.7%) |
| 85 | 591–595 | Sender rollouts (before PPO) | 5 | 599/960 (62.4%) | 8/960 (0.8%) | 413/960 (43.0%) | 547/960 (57.0%) |
| 86 | 596–597 | Receiver collection (before filtering/update) | 2 | 629/1024 (61.4%) | 13/1024 (1.3%) | 446/1024 (43.6%) | 578/1024 (56.4%) |
| 86 | 598–602 | Sender rollouts (before PPO) | 5 | 583/960 (60.7%) | 10/960 (1.0%) | 543/960 (56.6%) | 417/960 (43.4%) |
| 87 | 603–604 | Receiver collection (before filtering/update) | 2 | 635/1024 (62.0%) | 18/1024 (1.8%) | 603/1024 (58.9%) | 421/1024 (41.1%) |
| 87 | 605–609 | Sender rollouts (before PPO) | 5 | 595/960 (62.0%) | 26/960 (2.7%) | 648/960 (67.5%) | 312/960 (32.5%) |
| 88 | 610–611 | Receiver collection (before filtering/update) | 2 | 639/1024 (62.4%) | 18/1024 (1.8%) | 694/1024 (67.8%) | 330/1024 (32.2%) |
| 88 | 612–616 | Sender rollouts (before PPO) | 5 | 612/960 (63.8%) | 14/960 (1.5%) | 597/960 (62.2%) | 363/960 (37.8%) |
| 89 | 617–618 | Receiver collection (before filtering/update) | 2 | 664/1024 (64.8%) | 29/1024 (2.8%) | 569/1024 (55.6%) | 455/1024 (44.4%) |
| 89 | 619–623 | Sender rollouts (before PPO) | 5 | 633/960 (65.9%) | 37/960 (3.9%) | 478/960 (49.8%) | 482/960 (50.2%) |
| 90 | 624–625 | Receiver collection (before filtering/update) | 2 | 680/1024 (66.4%) | 24/1024 (2.3%) | 522/1024 (51.0%) | 502/1024 (49.0%) |
| 90 | 626–630 | Sender rollouts (before PPO) | 5 | 630/960 (65.6%) | 16/960 (1.7%) | 453/960 (47.2%) | 507/960 (52.8%) |
| 91 | 631–632 | Receiver collection (before filtering/update) | 2 | 693/1024 (67.7%) | 6/1024 (0.6%) | 474/1024 (46.3%) | 550/1024 (53.7%) |
| 91 | 633–637 | Sender rollouts (before PPO) | 5 | 669/960 (69.7%) | 12/960 (1.2%) | 555/960 (57.8%) | 405/960 (42.2%) |
| 92 | 638–639 | Receiver collection (before filtering/update) | 2 | 578/808 (71.5%) | 19/808 (2.4%) | 483/808 (59.8%) | 325/808 (40.2%) |
| 92 | 640–644 | Sender rollouts (before PPO) | 5 | 682/960 (71.0%) | 21/960 (2.2%) | 629/960 (65.5%) | 331/960 (34.5%) |
| 93 | 645–646 | Receiver collection (before filtering/update) | 2 | 754/1024 (73.6%) | 21/1024 (2.1%) | 581/1024 (56.7%) | 443/1024 (43.3%) |
| 93 | 647–651 | Sender rollouts (before PPO) | 5 | 680/960 (70.8%) | 15/960 (1.6%) | 550/960 (57.3%) | 410/960 (42.7%) |
| 94 | 652–653 | Receiver collection (before filtering/update) | 2 | 722/1024 (70.5%) | 13/1024 (1.3%) | 578/1024 (56.4%) | 446/1024 (43.6%) |
| 94 | 654–658 | Sender rollouts (before PPO) | 5 | 708/960 (73.8%) | 13/960 (1.4%) | 480/960 (50.0%) | 480/960 (50.0%) |
| 95 | 659–660 | Receiver collection (before filtering/update) | 2 | 767/1024 (74.9%) | 10/1024 (1.0%) | 467/1024 (45.6%) | 557/1024 (54.4%) |
| 95 | 661–665 | Sender rollouts (before PPO) | 5 | 741/960 (77.2%) | 13/960 (1.4%) | 463/960 (48.2%) | 497/960 (51.8%) |
| 96 | 666–667 | Receiver collection (before filtering/update) | 2 | 832/1024 (81.2%) | 30/1024 (2.9%) | 506/1024 (49.4%) | 518/1024 (50.6%) |
| 96 | 668–672 | Sender rollouts (before PPO) | 5 | 795/960 (82.8%) | 32/960 (3.3%) | 532/960 (55.4%) | 428/960 (44.6%) |
| 97 | 673–674 | Receiver collection (before filtering/update) | 2 | 869/1024 (84.9%) | 22/1024 (2.1%) | 506/1024 (49.4%) | 518/1024 (50.6%) |
| 97 | 675–679 | Sender rollouts (before PPO) | 5 | 829/960 (86.4%) | 31/960 (3.2%) | 522/960 (54.4%) | 438/960 (45.6%) |
| 98 | 680–681 | Receiver collection (before filtering/update) | 2 | 910/1024 (88.9%) | 21/1024 (2.1%) | 539/1024 (52.6%) | 485/1024 (47.4%) |
| 98 | 682–686 | Sender rollouts (before PPO) | 5 | 856/960 (89.2%) | 22/960 (2.3%) | 563/960 (58.6%) | 397/960 (41.4%) |
| 99 | 687–688 | Receiver collection (before filtering/update) | 2 | 911/1024 (89.0%) | 16/1024 (1.6%) | 551/1024 (53.8%) | 473/1024 (46.2%) |
| 99 | 689–693 | Sender rollouts (before PPO) | 5 | 846/960 (88.1%) | 5/960 (0.5%) | 455/960 (47.4%) | 505/960 (52.6%) |
| 100 | 694–695 | Receiver collection (before filtering/update) | 2 | 696/768 (90.6%) | 3/768 (0.4%) | 376/768 (49.0%) | 392/768 (51.0%) |
| 100 | 696–700 | Sender rollouts (before PPO) | 5 | 860/960 (89.6%) | 10/960 (1.0%) | 483/960 (50.3%) | 477/960 (49.7%) |
| 101 | 701–702 | Receiver collection (before filtering/update) | 2 | 754/808 (93.3%) | 12/808 (1.5%) | 402/808 (49.8%) | 406/808 (50.2%) |
| 101 | 703–707 | Sender rollouts (before PPO) | 5 | 896/960 (93.3%) | 24/960 (2.5%) | 509/960 (53.0%) | 451/960 (47.0%) |
| 102 | 708–709 | Receiver collection (before filtering/update) | 2 | 496/512 (96.9%) | 0/512 (0.0%) | 246/512 (48.0%) | 266/512 (52.0%) |
| 102 | 710–714 | Sender rollouts (before PPO) | 5 | 925/960 (96.4%) | 9/960 (0.9%) | 501/960 (52.2%) | 459/960 (47.8%) |
| 103 | 715–716 | Receiver collection (before filtering/update) | 2 | 988/1024 (96.5%) | 11/1024 (1.1%) | 505/1024 (49.3%) | 519/1024 (50.7%) |
| 103 | 717–721 | Sender rollouts (before PPO) | 5 | 921/960 (95.9%) | 6/960 (0.6%) | 494/960 (51.5%) | 466/960 (48.5%) |
| 104 | 722–723 | Receiver collection (before filtering/update) | 2 | 741/768 (96.5%) | 5/768 (0.7%) | 400/768 (52.1%) | 368/768 (47.9%) |
| 104 | 724–728 | Sender rollouts (before PPO) | 5 | 933/960 (97.2%) | 9/960 (0.9%) | 500/960 (52.1%) | 460/960 (47.9%) |
| 105 | 729–730 | Receiver collection (before filtering/update) | 2 | 749/768 (97.5%) | 5/768 (0.7%) | 391/768 (50.9%) | 377/768 (49.1%) |
| 105 | 731–735 | Sender rollouts (before PPO) | 5 | 932/960 (97.1%) | 6/960 (0.6%) | 510/960 (53.1%) | 450/960 (46.9%) |
| 106 | 736–737 | Receiver collection (before filtering/update) | 2 | 737/768 (96.0%) | 2/768 (0.3%) | 375/768 (48.8%) | 393/768 (51.2%) |
| 106 | 738–742 | Sender rollouts (before PPO) | 5 | 934/960 (97.3%) | 9/960 (0.9%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 107 | 743–744 | Receiver collection (before filtering/update) | 2 | 742/768 (96.6%) | 1/768 (0.1%) | 393/768 (51.2%) | 375/768 (48.8%) |
| 107 | 745–749 | Sender rollouts (before PPO) | 5 | 938/960 (97.7%) | 4/960 (0.4%) | 516/960 (53.8%) | 444/960 (46.2%) |
| 108 | 750–751 | Receiver collection (before filtering/update) | 2 | 738/768 (96.1%) | 1/768 (0.1%) | 394/768 (51.3%) | 374/768 (48.7%) |
| 108 | 752–756 | Sender rollouts (before PPO) | 5 | 926/960 (96.5%) | 3/960 (0.3%) | 487/960 (50.7%) | 473/960 (49.3%) |
| 109 | 757–758 | Receiver collection (before filtering/update) | 2 | 745/768 (97.0%) | 4/768 (0.5%) | 393/768 (51.2%) | 375/768 (48.8%) |
| 109 | 759–763 | Sender rollouts (before PPO) | 5 | 924/960 (96.2%) | 4/960 (0.4%) | 474/960 (49.4%) | 486/960 (50.6%) |
| 110 | 764–765 | Receiver collection (before filtering/update) | 2 | 739/768 (96.2%) | 3/768 (0.4%) | 379/768 (49.3%) | 389/768 (50.7%) |
| 110 | 766–770 | Sender rollouts (before PPO) | 5 | 920/960 (95.8%) | 3/960 (0.3%) | 487/960 (50.7%) | 473/960 (49.3%) |
| 111 | 771–772 | Receiver collection (before filtering/update) | 2 | 989/1024 (96.6%) | 4/1024 (0.4%) | 529/1024 (51.7%) | 495/1024 (48.3%) |
| 111 | 773–777 | Sender rollouts (before PPO) | 5 | 934/960 (97.3%) | 3/960 (0.3%) | 488/960 (50.8%) | 472/960 (49.2%) |
| 112 | 778–779 | Receiver collection (before filtering/update) | 2 | 795/808 (98.4%) | 5/808 (0.6%) | 414/808 (51.2%) | 394/808 (48.8%) |
| 112 | 780–784 | Sender rollouts (before PPO) | 5 | 937/960 (97.6%) | 2/960 (0.2%) | 485/960 (50.5%) | 475/960 (49.5%) |
| 113 | 785–786 | Receiver collection (before filtering/update) | 2 | 1011/1024 (98.7%) | 4/1024 (0.4%) | 494/1024 (48.2%) | 530/1024 (51.8%) |
| 113 | 787–791 | Sender rollouts (before PPO) | 5 | 942/960 (98.1%) | 1/960 (0.1%) | 485/960 (50.5%) | 475/960 (49.5%) |
| 114 | 792–793 | Receiver collection (before filtering/update) | 2 | 505/512 (98.6%) | 0/512 (0.0%) | 270/512 (52.7%) | 242/512 (47.3%) |
| 114 | 794–798 | Sender rollouts (before PPO) | 5 | 934/960 (97.3%) | 7/960 (0.7%) | 498/960 (51.9%) | 462/960 (48.1%) |
| 115 | 799–800 | Receiver collection (before filtering/update) | 2 | 493/512 (96.3%) | 0/512 (0.0%) | 275/512 (53.7%) | 237/512 (46.3%) |
| 115 | 801–805 | Sender rollouts (before PPO) | 5 | 936/960 (97.5%) | 4/960 (0.4%) | 492/960 (51.2%) | 468/960 (48.8%) |
| 116 | 806–807 | Receiver collection (before filtering/update) | 2 | 757/768 (98.6%) | 3/768 (0.4%) | 385/768 (50.1%) | 383/768 (49.9%) |
| 116 | 808–812 | Sender rollouts (before PPO) | 5 | 942/960 (98.1%) | 3/960 (0.3%) | 502/960 (52.3%) | 458/960 (47.7%) |
| 117 | 813–814 | Receiver collection (before filtering/update) | 2 | 741/768 (96.5%) | 2/768 (0.3%) | 402/768 (52.3%) | 366/768 (47.7%) |
| 117 | 815–819 | Sender rollouts (before PPO) | 5 | 927/960 (96.6%) | 2/960 (0.2%) | 486/960 (50.6%) | 474/960 (49.4%) |
| 118 | 820–821 | Receiver collection (before filtering/update) | 2 | 736/768 (95.8%) | 3/768 (0.4%) | 373/768 (48.6%) | 395/768 (51.4%) |
| 118 | 822–826 | Sender rollouts (before PPO) | 5 | 938/960 (97.7%) | 2/960 (0.2%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 119 | 827–828 | Receiver collection (before filtering/update) | 2 | 1000/1024 (97.7%) | 6/1024 (0.6%) | 523/1024 (51.1%) | 501/1024 (48.9%) |
| 119 | 829–833 | Sender rollouts (before PPO) | 5 | 946/960 (98.5%) | 6/960 (0.6%) | 489/960 (50.9%) | 471/960 (49.1%) |
| 120 | 834–835 | Receiver collection (before filtering/update) | 2 | 992/1024 (96.9%) | 2/1024 (0.2%) | 507/1024 (49.5%) | 517/1024 (50.5%) |
| 120 | 836–840 | Sender rollouts (before PPO) | 5 | 934/960 (97.3%) | 2/960 (0.2%) | 484/960 (50.4%) | 476/960 (49.6%) |
| 121 | 841–842 | Receiver collection (before filtering/update) | 2 | 744/768 (96.9%) | 2/768 (0.3%) | 370/768 (48.2%) | 398/768 (51.8%) |
| 121 | 843–847 | Sender rollouts (before PPO) | 5 | 949/960 (98.9%) | 2/960 (0.2%) | 494/960 (51.5%) | 466/960 (48.5%) |
| 122 | 848–849 | Receiver collection (before filtering/update) | 2 | 756/768 (98.4%) | 2/768 (0.3%) | 383/768 (49.9%) | 385/768 (50.1%) |
| 122 | 850–854 | Sender rollouts (before PPO) | 5 | 942/960 (98.1%) | 2/960 (0.2%) | 481/960 (50.1%) | 479/960 (49.9%) |
| 123 | 855–856 | Receiver collection (before filtering/update) | 2 | 496/512 (96.9%) | 0/512 (0.0%) | 253/512 (49.4%) | 259/512 (50.6%) |
| 123 | 857–861 | Sender rollouts (before PPO) | 5 | 937/960 (97.6%) | 0/960 (0.0%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 124 | 862–863 | Receiver collection (before filtering/update) | 2 | 543/552 (98.4%) | 0/552 (0.0%) | 271/552 (49.1%) | 281/552 (50.9%) |
| 124 | 864–868 | Sender rollouts (before PPO) | 5 | 942/960 (98.1%) | 2/960 (0.2%) | 466/960 (48.5%) | 494/960 (51.5%) |
| 125 | 869–870 | Receiver collection (before filtering/update) | 2 | 757/768 (98.6%) | 1/768 (0.1%) | 389/768 (50.7%) | 379/768 (49.3%) |
| 125 | 871–875 | Sender rollouts (before PPO) | 5 | 947/960 (98.6%) | 1/960 (0.1%) | 481/960 (50.1%) | 479/960 (49.9%) |
| 126 | 876–877 | Receiver collection (before filtering/update) | 2 | 1016/1024 (99.2%) | 5/1024 (0.5%) | 486/1024 (47.5%) | 538/1024 (52.5%) |
| 126 | 878–882 | Sender rollouts (before PPO) | 5 | 947/960 (98.6%) | 5/960 (0.5%) | 500/960 (52.1%) | 460/960 (47.9%) |
| 127 | 883–884 | Receiver collection (before filtering/update) | 2 | 758/768 (98.7%) | 4/768 (0.5%) | 372/768 (48.4%) | 396/768 (51.6%) |
| 127 | 885–889 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 2/960 (0.2%) | 487/960 (50.7%) | 473/960 (49.3%) |
| 128 | 890–891 | Receiver collection (before filtering/update) | 2 | 758/768 (98.7%) | 4/768 (0.5%) | 395/768 (51.4%) | 373/768 (48.6%) |
| 128 | 892–896 | Sender rollouts (before PPO) | 5 | 950/960 (99.0%) | 2/960 (0.2%) | 509/960 (53.0%) | 451/960 (47.0%) |
| 129 | 897–898 | Receiver collection (before filtering/update) | 2 | 1011/1024 (98.7%) | 10/1024 (1.0%) | 504/1024 (49.2%) | 520/1024 (50.8%) |
| 129 | 899–903 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 3/960 (0.3%) | 459/960 (47.8%) | 501/960 (52.2%) |
| 130 | 904–905 | Receiver collection (before filtering/update) | 2 | 762/768 (99.2%) | 3/768 (0.4%) | 387/768 (50.4%) | 381/768 (49.6%) |
| 130 | 906–910 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 2/960 (0.2%) | 463/960 (48.2%) | 497/960 (51.8%) |
| 131 | 911–912 | Receiver collection (before filtering/update) | 2 | 1013/1024 (98.9%) | 5/1024 (0.5%) | 512/1024 (50.0%) | 512/1024 (50.0%) |
| 131 | 913–917 | Sender rollouts (before PPO) | 5 | 946/960 (98.5%) | 7/960 (0.7%) | 483/960 (50.3%) | 477/960 (49.7%) |
| 132 | 918–919 | Receiver collection (before filtering/update) | 2 | 760/768 (99.0%) | 3/768 (0.4%) | 395/768 (51.4%) | 373/768 (48.6%) |
| 132 | 920–924 | Sender rollouts (before PPO) | 5 | 945/960 (98.4%) | 2/960 (0.2%) | 469/960 (48.9%) | 491/960 (51.1%) |
| 133 | 925–926 | Receiver collection (before filtering/update) | 2 | 760/768 (99.0%) | 1/768 (0.1%) | 376/768 (49.0%) | 392/768 (51.0%) |
| 133 | 927–931 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 0/960 (0.0%) | 502/960 (52.3%) | 458/960 (47.7%) |
| 134 | 932–933 | Receiver collection (before filtering/update) | 2 | 763/768 (99.3%) | 1/768 (0.1%) | 364/768 (47.4%) | 404/768 (52.6%) |
| 134 | 934–938 | Sender rollouts (before PPO) | 5 | 952/960 (99.2%) | 1/960 (0.1%) | 468/960 (48.8%) | 492/960 (51.2%) |
| 135 | 939–940 | Receiver collection (before filtering/update) | 2 | 550/552 (99.6%) | 2/552 (0.4%) | 265/552 (48.0%) | 287/552 (52.0%) |
| 135 | 941–945 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 2/960 (0.2%) | 481/960 (50.1%) | 479/960 (49.9%) |
| 136 | 946–947 | Receiver collection (before filtering/update) | 2 | 759/768 (98.8%) | 5/768 (0.7%) | 371/768 (48.3%) | 397/768 (51.7%) |
| 136 | 948–952 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 8/960 (0.8%) | 514/960 (53.5%) | 446/960 (46.5%) |
| 137 | 953–954 | Receiver collection (before filtering/update) | 2 | 764/768 (99.5%) | 2/768 (0.3%) | 397/768 (51.7%) | 371/768 (48.3%) |
| 137 | 955–959 | Sender rollouts (before PPO) | 5 | 948/960 (98.8%) | 1/960 (0.1%) | 504/960 (52.5%) | 456/960 (47.5%) |
| 138 | 960–961 | Receiver collection (before filtering/update) | 2 | 1015/1024 (99.1%) | 6/1024 (0.6%) | 511/1024 (49.9%) | 513/1024 (50.1%) |
| 138 | 962–966 | Sender rollouts (before PPO) | 5 | 949/960 (98.9%) | 5/960 (0.5%) | 475/960 (49.5%) | 485/960 (50.5%) |
| 139 | 967–968 | Receiver collection (before filtering/update) | 2 | 503/512 (98.2%) | 0/512 (0.0%) | 266/512 (52.0%) | 246/512 (48.0%) |
| 139 | 969–973 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 2/960 (0.2%) | 482/960 (50.2%) | 478/960 (49.8%) |
| 140 | 974–975 | Receiver collection (before filtering/update) | 2 | 756/768 (98.4%) | 3/768 (0.4%) | 374/768 (48.7%) | 394/768 (51.3%) |
| 140 | 976–980 | Sender rollouts (before PPO) | 5 | 950/960 (99.0%) | 5/960 (0.5%) | 488/960 (50.8%) | 472/960 (49.2%) |
| 141 | 981–982 | Receiver collection (before filtering/update) | 2 | 509/512 (99.4%) | 0/512 (0.0%) | 244/512 (47.7%) | 268/512 (52.3%) |
| 141 | 983–987 | Sender rollouts (before PPO) | 5 | 950/960 (99.0%) | 3/960 (0.3%) | 494/960 (51.5%) | 466/960 (48.5%) |
| 142 | 988–989 | Receiver collection (before filtering/update) | 2 | 762/768 (99.2%) | 1/768 (0.1%) | 382/768 (49.7%) | 386/768 (50.3%) |
| 142 | 990–994 | Sender rollouts (before PPO) | 5 | 950/960 (99.0%) | 4/960 (0.4%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 143 | 995–996 | Receiver collection (before filtering/update) | 2 | 1014/1024 (99.0%) | 4/1024 (0.4%) | 502/1024 (49.0%) | 522/1024 (51.0%) |
| 143 | 997–1001 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 1/960 (0.1%) | 484/960 (50.4%) | 476/960 (49.6%) |
| 144 | 1002–1003 | Receiver collection (before filtering/update) | 2 | 764/768 (99.5%) | 2/768 (0.3%) | 374/768 (48.7%) | 394/768 (51.3%) |
| 144 | 1004–1008 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 2/960 (0.2%) | 471/960 (49.1%) | 489/960 (50.9%) |
| 145 | 1009–1010 | Receiver collection (before filtering/update) | 2 | 508/512 (99.2%) | 0/512 (0.0%) | 250/512 (48.8%) | 262/512 (51.2%) |
| 145 | 1011–1015 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 3/960 (0.3%) | 486/960 (50.6%) | 474/960 (49.4%) |
| 146 | 1016–1017 | Receiver collection (before filtering/update) | 2 | 761/768 (99.1%) | 4/768 (0.5%) | 395/768 (51.4%) | 373/768 (48.6%) |
| 146 | 1018–1022 | Sender rollouts (before PPO) | 5 | 951/960 (99.1%) | 3/960 (0.3%) | 486/960 (50.6%) | 474/960 (49.4%) |
| 147 | 1023–1024 | Receiver collection (before filtering/update) | 2 | 551/552 (99.8%) | 1/552 (0.2%) | 269/552 (48.7%) | 283/552 (51.3%) |
| 147 | 1025–1029 | Sender rollouts (before PPO) | 5 | 945/960 (98.4%) | 2/960 (0.2%) | 486/960 (50.6%) | 474/960 (49.4%) |
| 148 | 1030–1031 | Receiver collection (before filtering/update) | 2 | 1021/1024 (99.7%) | 3/1024 (0.3%) | 515/1024 (50.3%) | 509/1024 (49.7%) |
| 148 | 1032–1036 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 2/960 (0.2%) | 478/960 (49.8%) | 482/960 (50.2%) |
| 149 | 1037–1038 | Receiver collection (before filtering/update) | 2 | 765/768 (99.6%) | 3/768 (0.4%) | 375/768 (48.8%) | 393/768 (51.2%) |
| 149 | 1039–1043 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 6/960 (0.6%) | 485/960 (50.5%) | 475/960 (49.5%) |
| 150 | 1044–1045 | Receiver collection (before filtering/update) | 2 | 765/768 (99.6%) | 2/768 (0.3%) | 369/768 (48.0%) | 399/768 (52.0%) |
| 150 | 1046–1050 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 2/960 (0.2%) | 487/960 (50.7%) | 473/960 (49.3%) |
| 151 | 1051–1052 | Receiver collection (before filtering/update) | 2 | 512/512 (100.0%) | 0/512 (0.0%) | 252/512 (49.2%) | 260/512 (50.8%) |
| 151 | 1053–1057 | Sender rollouts (before PPO) | 5 | 949/960 (98.9%) | 0/960 (0.0%) | 488/960 (50.8%) | 472/960 (49.2%) |
| 152 | 1058–1059 | Receiver collection (before filtering/update) | 2 | 765/768 (99.6%) | 2/768 (0.3%) | 380/768 (49.5%) | 388/768 (50.5%) |
| 152 | 1060–1064 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 2/960 (0.2%) | 493/960 (51.4%) | 467/960 (48.6%) |
| 153 | 1065–1066 | Receiver collection (before filtering/update) | 2 | 1017/1024 (99.3%) | 3/1024 (0.3%) | 504/1024 (49.2%) | 520/1024 (50.8%) |
| 153 | 1067–1071 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 2/960 (0.2%) | 481/960 (50.1%) | 479/960 (49.9%) |
| 154 | 1072–1073 | Receiver collection (before filtering/update) | 2 | 756/768 (98.4%) | 1/768 (0.1%) | 384/768 (50.0%) | 384/768 (50.0%) |
| 154 | 1074–1078 | Sender rollouts (before PPO) | 5 | 950/960 (99.0%) | 2/960 (0.2%) | 477/960 (49.7%) | 483/960 (50.3%) |
| 155 | 1079–1080 | Receiver collection (before filtering/update) | 2 | 1006/1024 (98.2%) | 4/1024 (0.4%) | 526/1024 (51.4%) | 498/1024 (48.6%) |
| 155 | 1081–1085 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 0/960 (0.0%) | 466/960 (48.5%) | 494/960 (51.5%) |
| 156 | 1086–1087 | Receiver collection (before filtering/update) | 2 | 762/768 (99.2%) | 1/768 (0.1%) | 365/768 (47.5%) | 403/768 (52.5%) |
| 156 | 1088–1092 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 3/960 (0.3%) | 496/960 (51.7%) | 464/960 (48.3%) |
| 157 | 1093–1094 | Receiver collection (before filtering/update) | 2 | 1020/1024 (99.6%) | 5/1024 (0.5%) | 510/1024 (49.8%) | 514/1024 (50.2%) |
| 157 | 1095–1099 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 1/960 (0.1%) | 466/960 (48.5%) | 494/960 (51.5%) |
| 158 | 1100–1101 | Receiver collection (before filtering/update) | 2 | 800/808 (99.0%) | 2/808 (0.2%) | 393/808 (48.6%) | 415/808 (51.4%) |
| 158 | 1102–1106 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 1/960 (0.1%) | 514/960 (53.5%) | 446/960 (46.5%) |
| 159 | 1107–1108 | Receiver collection (before filtering/update) | 2 | 762/768 (99.2%) | 2/768 (0.3%) | 384/768 (50.0%) | 384/768 (50.0%) |
| 159 | 1109–1113 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 5/960 (0.5%) | 472/960 (49.2%) | 488/960 (50.8%) |
| 160 | 1114–1115 | Receiver collection (before filtering/update) | 2 | 496/512 (96.9%) | 0/512 (0.0%) | 270/512 (52.7%) | 242/512 (47.3%) |
| 160 | 1116–1120 | Sender rollouts (before PPO) | 5 | 952/960 (99.2%) | 1/960 (0.1%) | 497/960 (51.8%) | 463/960 (48.2%) |
| 161 | 1121–1122 | Receiver collection (before filtering/update) | 2 | 508/512 (99.2%) | 0/512 (0.0%) | 258/512 (50.4%) | 254/512 (49.6%) |
| 161 | 1123–1127 | Sender rollouts (before PPO) | 5 | 952/960 (99.2%) | 1/960 (0.1%) | 458/960 (47.7%) | 502/960 (52.3%) |
| 162 | 1128–1129 | Receiver collection (before filtering/update) | 2 | 506/512 (98.8%) | 0/512 (0.0%) | 242/512 (47.3%) | 270/512 (52.7%) |
| 162 | 1130–1134 | Sender rollouts (before PPO) | 5 | 952/960 (99.2%) | 1/960 (0.1%) | 500/960 (52.1%) | 460/960 (47.9%) |
| 163 | 1135–1136 | Receiver collection (before filtering/update) | 2 | 1021/1024 (99.7%) | 3/1024 (0.3%) | 490/1024 (47.9%) | 534/1024 (52.1%) |
| 163 | 1137–1141 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 3/960 (0.3%) | 443/960 (46.1%) | 517/960 (53.9%) |
| 164 | 1142–1143 | Receiver collection (before filtering/update) | 2 | 510/512 (99.6%) | 0/512 (0.0%) | 261/512 (51.0%) | 251/512 (49.0%) |
| 164 | 1144–1148 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 1/960 (0.1%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 165 | 1149–1150 | Receiver collection (before filtering/update) | 2 | 505/512 (98.6%) | 0/512 (0.0%) | 245/512 (47.9%) | 267/512 (52.1%) |
| 165 | 1151–1155 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 2/960 (0.2%) | 510/960 (53.1%) | 450/960 (46.9%) |
| 166 | 1156–1157 | Receiver collection (before filtering/update) | 2 | 508/512 (99.2%) | 0/512 (0.0%) | 252/512 (49.2%) | 260/512 (50.8%) |
| 166 | 1158–1162 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 0/960 (0.0%) | 468/960 (48.8%) | 492/960 (51.2%) |
| 167 | 1163–1164 | Receiver collection (before filtering/update) | 2 | 765/768 (99.6%) | 1/768 (0.1%) | 376/768 (49.0%) | 392/768 (51.0%) |
| 167 | 1165–1169 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 1/960 (0.1%) | 481/960 (50.1%) | 479/960 (49.9%) |
| 168 | 1170–1171 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 258/512 (50.4%) | 254/512 (49.6%) |
| 168 | 1172–1176 | Sender rollouts (before PPO) | 5 | 960/960 (100.0%) | 2/960 (0.2%) | 495/960 (51.6%) | 465/960 (48.4%) |
| 169 | 1177–1178 | Receiver collection (before filtering/update) | 2 | 768/768 (100.0%) | 1/768 (0.1%) | 364/768 (47.4%) | 404/768 (52.6%) |
| 169 | 1179–1183 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 1/960 (0.1%) | 456/960 (47.5%) | 504/960 (52.5%) |
| 170 | 1184–1185 | Receiver collection (before filtering/update) | 2 | 508/512 (99.2%) | 0/512 (0.0%) | 264/512 (51.6%) | 248/512 (48.4%) |
| 170 | 1186–1190 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 0/960 (0.0%) | 522/960 (54.4%) | 438/960 (45.6%) |
| 171 | 1191–1192 | Receiver collection (before filtering/update) | 2 | 766/768 (99.7%) | 2/768 (0.3%) | 373/768 (48.6%) | 395/768 (51.4%) |
| 171 | 1193–1197 | Sender rollouts (before PPO) | 5 | 953/960 (99.3%) | 3/960 (0.3%) | 472/960 (49.2%) | 488/960 (50.8%) |
| 172 | 1198–1199 | Receiver collection (before filtering/update) | 2 | 508/512 (99.2%) | 0/512 (0.0%) | 257/512 (50.2%) | 255/512 (49.8%) |
| 172 | 1200–1204 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 1/960 (0.1%) | 495/960 (51.6%) | 465/960 (48.4%) |
| 173 | 1205–1206 | Receiver collection (before filtering/update) | 2 | 549/552 (99.5%) | 1/552 (0.2%) | 291/552 (52.7%) | 261/552 (47.3%) |
| 173 | 1207–1211 | Sender rollouts (before PPO) | 5 | 960/960 (100.0%) | 2/960 (0.2%) | 471/960 (49.1%) | 489/960 (50.9%) |
| 174 | 1212–1213 | Receiver collection (before filtering/update) | 2 | 509/512 (99.4%) | 0/512 (0.0%) | 275/512 (53.7%) | 237/512 (46.3%) |
| 174 | 1214–1218 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 1/960 (0.1%) | 475/960 (49.5%) | 485/960 (50.5%) |
| 175 | 1219–1220 | Receiver collection (before filtering/update) | 2 | 768/768 (100.0%) | 1/768 (0.1%) | 378/768 (49.2%) | 390/768 (50.8%) |
| 175 | 1221–1225 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 1/960 (0.1%) | 474/960 (49.4%) | 486/960 (50.6%) |
| 176 | 1226–1227 | Receiver collection (before filtering/update) | 2 | 768/768 (100.0%) | 1/768 (0.1%) | 394/768 (51.3%) | 374/768 (48.7%) |
| 176 | 1228–1232 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 1/960 (0.1%) | 492/960 (51.2%) | 468/960 (48.8%) |
| 177 | 1233–1234 | Receiver collection (before filtering/update) | 2 | 510/512 (99.6%) | 0/512 (0.0%) | 260/512 (50.8%) | 252/512 (49.2%) |
| 177 | 1235–1239 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 0/960 (0.0%) | 464/960 (48.3%) | 496/960 (51.7%) |
| 178 | 1240–1241 | Receiver collection (before filtering/update) | 2 | 764/768 (99.5%) | 1/768 (0.1%) | 368/768 (47.9%) | 400/768 (52.1%) |
| 178 | 1242–1246 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 2/960 (0.2%) | 488/960 (50.8%) | 472/960 (49.2%) |
| 179 | 1247–1248 | Receiver collection (before filtering/update) | 2 | 767/768 (99.9%) | 3/768 (0.4%) | 353/768 (46.0%) | 415/768 (54.0%) |
| 179 | 1249–1253 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 0/960 (0.0%) | 502/960 (52.3%) | 458/960 (47.7%) |
| 180 | 1254–1255 | Receiver collection (before filtering/update) | 2 | 510/512 (99.6%) | 0/512 (0.0%) | 244/512 (47.7%) | 268/512 (52.3%) |
| 180 | 1256–1260 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 2/960 (0.2%) | 471/960 (49.1%) | 489/960 (50.9%) |
| 181 | 1261–1262 | Receiver collection (before filtering/update) | 2 | 766/768 (99.7%) | 1/768 (0.1%) | 378/768 (49.2%) | 390/768 (50.8%) |
| 181 | 1263–1267 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 2/960 (0.2%) | 465/960 (48.4%) | 495/960 (51.6%) |
| 182 | 1268–1269 | Receiver collection (before filtering/update) | 2 | 512/512 (100.0%) | 0/512 (0.0%) | 246/512 (48.0%) | 266/512 (52.0%) |
| 182 | 1270–1274 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 0/960 (0.0%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 183 | 1275–1276 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 244/512 (47.7%) | 268/512 (52.3%) |
| 183 | 1277–1281 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 1/960 (0.1%) | 480/960 (50.0%) | 480/960 (50.0%) |
| 184 | 1282–1283 | Receiver collection (before filtering/update) | 2 | 767/768 (99.9%) | 1/768 (0.1%) | 381/768 (49.6%) | 387/768 (50.4%) |
| 184 | 1284–1288 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 4/960 (0.4%) | 472/960 (49.2%) | 488/960 (50.8%) |
| 185 | 1289–1290 | Receiver collection (before filtering/update) | 2 | 505/512 (98.6%) | 0/512 (0.0%) | 239/512 (46.7%) | 273/512 (53.3%) |
| 185 | 1291–1295 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 1/960 (0.1%) | 491/960 (51.1%) | 469/960 (48.9%) |
| 186 | 1296–1297 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 257/512 (50.2%) | 255/512 (49.8%) |
| 186 | 1298–1302 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 0/960 (0.0%) | 476/960 (49.6%) | 484/960 (50.4%) |
| 187 | 1303–1304 | Receiver collection (before filtering/update) | 2 | 552/552 (100.0%) | 0/552 (0.0%) | 265/552 (48.0%) | 287/552 (52.0%) |
| 187 | 1305–1309 | Sender rollouts (before PPO) | 5 | 960/960 (100.0%) | 0/960 (0.0%) | 493/960 (51.4%) | 467/960 (48.6%) |
| 188 | 1310–1311 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 251/512 (49.0%) | 261/512 (51.0%) |
| 188 | 1312–1316 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 1/960 (0.1%) | 483/960 (50.3%) | 477/960 (49.7%) |
| 189 | 1317–1318 | Receiver collection (before filtering/update) | 2 | 510/512 (99.6%) | 0/512 (0.0%) | 267/512 (52.1%) | 245/512 (47.9%) |
| 189 | 1319–1323 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 0/960 (0.0%) | 470/960 (49.0%) | 490/960 (51.0%) |
| 190 | 1324–1325 | Receiver collection (before filtering/update) | 2 | 510/512 (99.6%) | 0/512 (0.0%) | 265/512 (51.8%) | 247/512 (48.2%) |
| 190 | 1326–1330 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 1/960 (0.1%) | 455/960 (47.4%) | 505/960 (52.6%) |
| 191 | 1331–1332 | Receiver collection (before filtering/update) | 2 | 509/512 (99.4%) | 0/512 (0.0%) | 267/512 (52.1%) | 245/512 (47.9%) |
| 191 | 1333–1337 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 1/960 (0.1%) | 486/960 (50.6%) | 474/960 (49.4%) |
| 192 | 1338–1339 | Receiver collection (before filtering/update) | 2 | 509/512 (99.4%) | 0/512 (0.0%) | 237/512 (46.3%) | 275/512 (53.7%) |
| 192 | 1340–1344 | Sender rollouts (before PPO) | 5 | 954/960 (99.4%) | 0/960 (0.0%) | 488/960 (50.8%) | 472/960 (49.2%) |
| 193 | 1345–1346 | Receiver collection (before filtering/update) | 2 | 764/768 (99.5%) | 1/768 (0.1%) | 386/768 (50.3%) | 382/768 (49.7%) |
| 193 | 1347–1351 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 0/960 (0.0%) | 494/960 (51.5%) | 466/960 (48.5%) |
| 194 | 1352–1353 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 248/512 (48.4%) | 264/512 (51.6%) |
| 194 | 1354–1358 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 0/960 (0.0%) | 481/960 (50.1%) | 479/960 (49.9%) |
| 195 | 1359–1360 | Receiver collection (before filtering/update) | 2 | 509/512 (99.4%) | 0/512 (0.0%) | 268/512 (52.3%) | 244/512 (47.7%) |
| 195 | 1361–1365 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 1/960 (0.1%) | 500/960 (52.1%) | 460/960 (47.9%) |
| 196 | 1366–1367 | Receiver collection (before filtering/update) | 2 | 510/512 (99.6%) | 0/512 (0.0%) | 267/512 (52.1%) | 245/512 (47.9%) |
| 196 | 1368–1372 | Sender rollouts (before PPO) | 5 | 955/960 (99.5%) | 1/960 (0.1%) | 486/960 (50.6%) | 474/960 (49.4%) |
| 197 | 1373–1374 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 236/512 (46.1%) | 276/512 (53.9%) |
| 197 | 1375–1379 | Sender rollouts (before PPO) | 5 | 956/960 (99.6%) | 0/960 (0.0%) | 503/960 (52.4%) | 457/960 (47.6%) |
| 198 | 1380–1381 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 242/512 (47.3%) | 270/512 (52.7%) |
| 198 | 1382–1386 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 0/960 (0.0%) | 499/960 (52.0%) | 461/960 (48.0%) |
| 199 | 1387–1388 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 231/512 (45.1%) | 281/512 (54.9%) |
| 199 | 1389–1393 | Sender rollouts (before PPO) | 5 | 958/960 (99.8%) | 0/960 (0.0%) | 458/960 (47.7%) | 502/960 (52.3%) |
| 200 | 1394–1395 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 243/512 (47.5%) | 269/512 (52.5%) |
| 200 | 1396–1400 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 0/960 (0.0%) | 484/960 (50.4%) | 476/960 (49.6%) |
| 201 | 1401–1402 | Receiver collection (before filtering/update) | 2 | 512/512 (100.0%) | 0/512 (0.0%) | 257/512 (50.2%) | 255/512 (49.8%) |
| 201 | 1403–1407 | Sender rollouts (before PPO) | 5 | 959/960 (99.9%) | 0/960 (0.0%) | 468/960 (48.8%) | 492/960 (51.2%) |
| 202 | 1408–1409 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 247/512 (48.2%) | 265/512 (51.8%) |
| 202 | 1410–1414 | Sender rollouts (before PPO) | 5 | 957/960 (99.7%) | 0/960 (0.0%) | 459/960 (47.8%) | 501/960 (52.2%) |
| 203 | 1415–1416 | Receiver collection (before filtering/update) | 2 | 512/512 (100.0%) | 0/512 (0.0%) | 259/512 (50.6%) | 253/512 (49.4%) |
| 203 | 1417–1421 | Sender rollouts (before PPO) | 5 | 960/960 (100.0%) | 0/960 (0.0%) | 487/960 (50.7%) | 473/960 (49.3%) |
| 204 | 1422–1423 | Receiver collection (before filtering/update) | 2 | 552/552 (100.0%) | 0/552 (0.0%) | 271/552 (49.1%) | 281/552 (50.9%) |
| 204 | 1424–1428 | Sender rollouts (before PPO) | 5 | 960/960 (100.0%) | 0/960 (0.0%) | 492/960 (51.2%) | 468/960 (48.8%) |
| 205 | 1429–1430 | Receiver collection (before filtering/update) | 2 | 511/512 (99.8%) | 0/512 (0.0%) | 257/512 (50.2%) | 255/512 (49.8%) |
| 205 | 1431–1435 | Sender rollouts (before PPO) | 5 | 960/960 (100.0%) | 1/960 (0.1%) | 501/960 (52.2%) | 459/960 (47.8%) |
| 206 | 1436–1437 | Receiver collection (before filtering/update) | 2 | 766/768 (99.7%) | 1/768 (0.1%) | 397/768 (51.7%) | 371/768 (48.3%) |
| 206 | 1438–1440 | Sender rollouts (before PPO) | 3 | 575/576 (99.8%) | 0/576 (0.0%) | 279/576 (48.4%) | 297/576 (51.6%) |

## Frozen-model evaluations

These measurements use the evaluation cohort with no parameter updates during evaluation. University and PIQA denominators are separate. Evaluations are reported individually, without pooling repeated evaluation contexts across checkpoints.

The fixed PIQA baseline is d₀ = 8.0%. A positive Δd means more rejected common-sense answers than at baseline; this descriptive difference alone is not a statistical significance claim.

| Update | C_R | C_I flagged | Receiver PASS | Receiver FAIL | PIQA d | Δd (pp) |
|---:|---|---|---|---|---|---:|
| 0 | 1011/2000 (50.5%) | 52/2000 (2.6%) | 1916/2000 (95.8%) | 84/2000 (4.2%) | 8/100 (8.0%) | +0.0 |
| 50 | 1051/2000 (52.5%) | 152/2000 (7.6%) | 1362/2000 (68.1%) | 638/2000 (31.9%) | 5/100 (5.0%) | -3.0 |
| 100 | 1012/2000 (50.6%) | 112/2000 (5.6%) | 223/2000 (11.2%) | 1777/2000 (88.8%) | 3/100 (3.0%) | -5.0 |
| 150 | 1047/2000 (52.4%) | 93/2000 (4.7%) | 940/2000 (47.0%) | 1060/2000 (53.0%) | 1/100 (1.0%) | -7.0 |
| 200 | 1000/2000 (50.0%) | 69/2000 (3.5%) | 97/2000 (4.8%) | 1903/2000 (95.2%) | 3/100 (3.0%) | -5.0 |
| 250 | 1069/2000 (53.5%) | 84/2000 (4.2%) | 1664/2000 (83.2%) | 336/2000 (16.8%) | 6/100 (6.0%) | -2.0 |
| 300 | 1093/2000 (54.6%) | 53/2000 (2.6%) | 1070/2000 (53.5%) | 930/2000 (46.5%) | 8/100 (8.0%) | +0.0 |
| 350 | 1059/2000 (53.0%) | 54/2000 (2.7%) | 412/2000 (20.6%) | 1588/2000 (79.4%) | 5/100 (5.0%) | -3.0 |
| 390 | 1072/2000 (53.6%) | 25/2000 (1.2%) | 1485/2000 (74.2%) | 515/2000 (25.8%) | 9/100 (9.0%) | +1.0 |
| 400 | 1072/2000 (53.6%) | 18/2000 (0.9%) | 1521/2000 (76.0%) | 479/2000 (23.9%) | 7/100 (7.0%) | -1.0 |
| 450 | 1118/2000 (55.9%) | 33/2000 (1.6%) | 1021/2000 (51.0%) | 979/2000 (49.0%) | 6/100 (6.0%) | -2.0 |
| 500 | 1147/2000 (57.4%) | 18/2000 (0.9%) | 1136/2000 (56.8%) | 864/2000 (43.2%) | 3/100 (3.0%) | -5.0 |
| 550 | 1154/2000 (57.7%) | 14/2000 (0.7%) | 1189/2000 (59.5%) | 811/2000 (40.5%) | 4/100 (4.0%) | -4.0 |
| 600 | 1234/2000 (61.7%) | 18/2000 (0.9%) | 1135/2000 (56.8%) | 865/2000 (43.2%) | 3/100 (3.0%) | -5.0 |
| 650 | 1434/2000 (71.7%) | 17/2000 (0.8%) | 1185/2000 (59.2%) | 815/2000 (40.8%) | 1/100 (1.0%) | -7.0 |
| 700 | 1873/2000 (93.7%) | 31/2000 (1.6%) | 996/2000 (49.8%) | 1004/2000 (50.2%) | 1/100 (1.0%) | -7.0 |
| 750 | 1920/2000 (96.0%) | 11/2000 (0.6%) | 1039/2000 (52.0%) | 961/2000 (48.0%) | 2/100 (2.0%) | -6.0 |
| 800 | 1951/2000 (97.5%) | 6/2000 (0.3%) | 1026/2000 (51.3%) | 974/2000 (48.7%) | 4/100 (4.0%) | -4.0 |
| 850 | 1967/2000 (98.3%) | 2/2000 (0.1%) | 1012/2000 (50.6%) | 988/2000 (49.4%) | 3/100 (3.0%) | -5.0 |
| 886 | 1974/2000 (98.7%) | 4/2000 (0.2%) | 1013/2000 (50.6%) | 987/2000 (49.4%) | 3/100 (3.0%) | -5.0 |
| 900 | 1987/2000 (99.3%) | 15/2000 (0.8%) | 998/2000 (49.9%) | 1002/2000 (50.1%) | 3/100 (3.0%) | -5.0 |
| 950 | 1976/2000 (98.8%) | 6/2000 (0.3%) | 1009/2000 (50.5%) | 991/2000 (49.5%) | 3/100 (3.0%) | -5.0 |
| 1000 | 1989/2000 (99.5%) | 4/2000 (0.2%) | 1002/2000 (50.1%) | 998/2000 (49.9%) | 3/100 (3.0%) | -5.0 |
| 1050 | 1988/2000 (99.4%) | 5/2000 (0.2%) | 1005/2000 (50.2%) | 995/2000 (49.8%) | 3/100 (3.0%) | -5.0 |
| 1100 | 1990/2000 (99.5%) | 5/2000 (0.2%) | 1009/2000 (50.5%) | 991/2000 (49.5%) | 1/100 (1.0%) | -7.0 |
| 1150 | 1990/2000 (99.5%) | 2/2000 (0.1%) | 1007/2000 (50.4%) | 993/2000 (49.6%) | 4/100 (4.0%) | -4.0 |
| 1200 | 1989/2000 (99.5%) | 2/2000 (0.1%) | 1004/2000 (50.2%) | 996/2000 (49.8%) | 3/100 (3.0%) | -5.0 |
| 1250 | 1995/2000 (99.8%) | 1/2000 (0.1%) | 1000/2000 (50.0%) | 1000/2000 (50.0%) | 5/100 (5.0%) | -3.0 |
| 1300 | 1999/2000 (100.0%) | 1/2000 (0.1%) | 1000/2000 (50.0%) | 1000/2000 (50.0%) | 4/100 (4.0%) | -4.0 |
| 1350 | 1993/2000 (99.7%) | 0/2000 (0.0%) | 1002/2000 (50.1%) | 998/2000 (49.9%) | 7/100 (7.0%) | -1.0 |
| 1400 | 1999/2000 (100.0%) | 0/2000 (0.0%) | 1002/2000 (50.1%) | 998/2000 (49.9%) | 7/100 (7.0%) | -1.0 |
| 1440 | 1998/2000 (99.9%) | 0/2000 (0.0%) | 1003/2000 (50.1%) | 997/2000 (49.9%) | 5/100 (5.0%) | -3.0 |

| Evaluation update | Judge rejected | Format failed | Both failed |
|---:|---|---|---|
| 0 | 5/100 (5.0%) | 3/100 (3.0%) | 0/100 (0.0%) |
| 50 | 3/100 (3.0%) | 2/100 (2.0%) | 0/100 (0.0%) |
| 100 | 2/100 (2.0%) | 1/100 (1.0%) | 0/100 (0.0%) |
| 150 | 1/100 (1.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 200 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 250 | 6/100 (6.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 300 | 8/100 (8.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 350 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 390 | 9/100 (9.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 400 | 7/100 (7.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 450 | 6/100 (6.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 500 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 550 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 600 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 650 | 1/100 (1.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 700 | 1/100 (1.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 750 | 2/100 (2.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 800 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 850 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 886 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 900 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 950 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1000 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1050 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1100 | 1/100 (1.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1150 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1200 | 3/100 (3.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1250 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1300 | 4/100 (4.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1350 | 7/100 (7.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1400 | 7/100 (7.0%) | 0/100 (0.0%) | 0/100 (0.0%) |
| 1440 | 5/100 (5.0%) | 0/100 (0.0%) | 0/100 (0.0%) |

[Per-update counts and rates](per_update.csv) · [Machine-readable report](REPORT.json)

## Monitoring provenance

The frozen report generator contains a legacy L40S hardware label; only that label was corrected in this generated report to identify the actual H100/Qwen run. The frozen source and configuration are unchanged.

Evaluation 150: At parent remote check 2026-10-05T18:54:03.331088+00:00, step150 state.pt had already been pruned. The verified frozen checkpoint retention keeps initial, latest three and segment-boundary states while retaining evaluation markers. Snapshot independently hashed state152; it does not independently rehash state150 or prove equality with the earlier state150 tensor bytes. See monitoring/EVALUATION_000150_VERIFICATION.json for the full audit.

Evaluation 200: At parent remote check 2026-10-05T20:23:23.300741+00:00, step200 state.pt had already been pruned. The verified frozen checkpoint retention keeps initial, latest three and segment-boundary states while retaining evaluation markers. Snapshot independently hashed state223; it does not independently rehash state200 or prove equality with the earlier state200 tensor bytes. See monitoring/EVALUATION_000200_VERIFICATION.json for the full audit.

Evaluation 350: At remote audit 2026-10-05T23:49:00.862517+00:00, step350 state.pt had already been pruned. The frozen retention keeps initial, latest three and segment-boundary states while retaining evaluation markers. Snapshot independently hashed state400 and separate audit independently hashed state390; neither independently rehashes state350 or proves equality with its former tensor bytes. See monitoring/EVALUATION_000350_VERIFICATION.json for the full audit.

Evaluation 500: At remote audit 2026-10-06T01:35:47.392688+00:00, step500 state.pt had already been pruned. The frozen retention keeps initial, latest three and segment-boundary states while retaining evaluation markers. Snapshot independently hashed state519; this does not independently rehash state500 or prove equality with its former tensor bytes. See monitoring/EVALUATION_000500_VERIFICATION.json for the full audit.

Evaluation 550: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000550 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 550 or verify its former state bytes. See monitoring/EVALUATION_000550_VERIFICATION.json for the full audit.

Evaluation 600: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000600 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 600 or verify its former state bytes. See monitoring/EVALUATION_000600_VERIFICATION.json for the full audit.

Evaluation 650: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000650 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 650 or verify its former state bytes. See monitoring/EVALUATION_000650_VERIFICATION.json for the full audit.

Evaluation 700: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000700 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 700 or verify its former state bytes. See monitoring/EVALUATION_000700_VERIFICATION.json for the full audit.

Evaluation 750: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000750 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 750 or verify its former state bytes. See monitoring/EVALUATION_000750_VERIFICATION.json for the full audit.

Evaluation 800: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000800 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 800 or verify its former state bytes. See monitoring/EVALUATION_000800_VERIFICATION.json for the full audit.

Evaluation 850: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000850 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 850 or verify its former state bytes. See monitoring/EVALUATION_000850_VERIFICATION.json for the full audit.

Evaluation 900: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000900 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 900 or verify its former state bytes. See monitoring/EVALUATION_000900_VERIFICATION.json for the full audit.

Evaluation 950: At remote audit 2026-10-06T14:36:54.074830+00:00, step_000950 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 950 or verify its former state bytes. See monitoring/EVALUATION_000950_VERIFICATION.json for the full audit.

Evaluation 1000: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001000 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1000 or verify its former state bytes. See monitoring/EVALUATION_001000_VERIFICATION.json for the full audit.

Evaluation 1050: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001050 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1050 or verify its former state bytes. See monitoring/EVALUATION_001050_VERIFICATION.json for the full audit.

Evaluation 1100: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001100 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1100 or verify its former state bytes. See monitoring/EVALUATION_001100_VERIFICATION.json for the full audit.

Evaluation 1150: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001150 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1150 or verify its former state bytes. See monitoring/EVALUATION_001150_VERIFICATION.json for the full audit.

Evaluation 1200: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001200 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1200 or verify its former state bytes. See monitoring/EVALUATION_001200_VERIFICATION.json for the full audit.

Evaluation 1250: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001250 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1250 or verify its former state bytes. See monitoring/EVALUATION_001250_VERIFICATION.json for the full audit.

Evaluation 1300: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001300 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1300 or verify its former state bytes. See monitoring/EVALUATION_001300_VERIFICATION.json for the full audit.

Evaluation 1350: At remote audit 2026-10-06T14:36:54.074830+00:00, step_001350 state.pt had already been pruned and was unavailable for independent rehashing. The public marker attests its recorded state SHA256 and is bound by evaluation COMPLETE, but the monitor did not independently verify those former state bytes. Frozen retention keeps the initial, latest three, and segment-boundary states while retaining evaluation markers. The independently hashed checkpoint 1380 does not substitute for checkpoint 1350 or verify its former state bytes. See monitoring/EVALUATION_001350_VERIFICATION.json for the full audit.
