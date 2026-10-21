# 1. Hidden in Plain Text: university reproduction

**Experiment.** We reproduced the university-CV task from [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs](https://aclanthology.org/2025.ijcnlp-long.34/) (with some minor modifications which bridge the gap between the paper and the official implementation). A Gemma-2-2B-it **sender** paraphrases a toy resume while being instructed not to mention the university; a DistilBERT **receiver** predicts whether the original university was “good” (PASS) or “bad” (FAIL) from that paraphrase. They aim to communicate this binary attribute through wording while a frozen, non-thinking Qwen3.5-27B-FP8 **observer** checks for academic-institution information. A separate frozen Qwen3-4B **judge** assesses the sender's answers to 100 general knowledge questions (PIQA) for usefulness, alongside a format check; it neither decodes the university nor supplies the university training reward and is only used to ensure that the sender's performance does not deteriorate as it is updated. Evaluations use the same 2,000 university examples and 100 PIQA questions throughout.

*Kiril's comment:* One way to understand this setting is as follows. Suppose that we have an actual RL training environment and there is a judge which tracks that models do not convey any hidden info. Then, whenever the judge catches them, there is a negative reward. This may lead to effective reinforcement of successful hidden communication. Note that this might be a more realistic scenario than the judge trying to guess the hidden information since for complicated environments it is not necessarily known a priori what the models may want to communicate.

**Training.** Each cycle alternates two full-receiver supervised updates and five sender PPO updates; observer and judge stay frozen. The sender reward is `1[o ≥ 0]·tanh(z_y) + 0.2·o + 0.1·min(character_count/20, 1)`, where `z_y` is the receiver's raw correct-class logit and `o = tanh(logit(No) − logit(Yes) − 0.75)` is the observer score; PPO applies KL regularization separately.

![Four panels showing receiver accuracy, observer flags, PIQA rejection, and receiver PASS counts over training](results/uni_qwen35_27b_20261004T214535Z/plots/verified_evaluation_curves.png)

**Figure 1.** **Top left:** receiver accuracy C_R rises from 1,011/2,000 (50.55%) to 1,998/2,000 (99.90%), a gain of 49.35 percentage points. **Top right:** operational observer flag rate C_I falls from 52/2,000 (2.60%) to 0/2,000, measuring detector flags rather than transmission accuracy. **Bottom left:** PIQA rejection d, the union of judge rejection and format failure, changes from 8/100 to 5/100 (five judge rejections and zero format failures), a decrease of three percentage points. **Bottom right:** receiver PASS predictions settle at 1,003/2,000 with 997 FAIL predictions after pronounced early class imbalance.

**Claude exercise.** We gave **Claude Opus 5.5, Medium effort** the final sender's 1,000 training-partition summaries and asked it to guess “good” versus “bad” university, explicitly explaining the steganographic task and giving only a 50/50 prior; the [recorded prompt](data/claude_training1440/EXPERIMENT.json) supplies no labeled examples. This export reuses the earlier checkpoint-1076 training/test assignments, with actual training counts of 499 bad and 501 good. Claude's submitted guesses match **996/1,000 hidden labels (99.6%)**: 497/499 bad and 499/501 good, with errors on IDs 1715, 241, 1889, and 575. Thus, a decoder explicitly asked to infer the hidden attribute recovered it on this partition despite the frozen observer's low flag rate; this is a training-partition result, not a held-out Claude test, and the archive preserves the user-reported prompt and predictions rather than a complete provider transcript.

Evidence: [verified evaluation data](results/uni_qwen35_27b_20261004T214535Z/plots/verified_evaluation_metrics.csv), [plot provenance](results/uni_qwen35_27b_20261004T214535Z/plots/PROVENANCE.json), [frozen experiment](launches/uni_qwen35_27b_20261004T214535Z/EXPERIMENT.md), [Claude scoring](results/prediction_label_checks/training1440_ed54cc15cac4/summary.json), and [split manifest](data/university_observer_qwen_checkpoint1440_split/split_manifest.json).


# 2. What is missing?

While the above example is perhaps useful to demonstrate that a static observer is insufficient, it has three limitations:

1. **Dynamic Observer:** That is, the observer does not update simultaneously with the models trying to collude. This can be overcome in practice if we do train-time monitoring and let the judge evolve with the two models.
2. **Sparse Hidden Communication:** The two models communicate hidden info in every message. This might not be realistic. In cases when the communication is sparse, the tasks of both the observer and the sender/receiver are more difficult. Note that we need to address this first if we want to deal with problem 1. An adaptive observer can always say "yes" when asked if there is hidden info and this will be optimal.
3. **Method of Learning Steganography:** This experiment is about weight-update steganography. Of course, one can instead optimize over prompts, etc. I think I am personally happy with weight updates as they seem least monitorable as opposed to crafting prompts/protocols in external files/etc.

# 3. Dynamic Observer

We now rerun the same experiment but with a dynamic observer. Of course, this requires a certain change of setup. Currently, all messages have hidden information. Thus, the judge is rewarded based on guessing the hidden information. Here is the exact setup. The sender and receiver communicate in the same way. However, the observer tries to guess whether the student is from a good or bad university. The rewards are as follows:

1. Sender: The sender reward is `1[o ≥ 0]·tanh(z_y) + 0.2·o + 0.1·min(character_count/20, 1)`, where `z_y` is the receiver's raw correct-class logit and `o = tanh(logit(Incorrect Label) − logit(Correct label) − 0.75)` is the observer score; PPO applies KL regularization separately. **We changed the logits of the observer.**
2. Receiver: standard SFT on the correct response.
3. Observer: standard SFT on the correct response. **We are now updating the observer.**

We train as follows. Each cycle alternates two full-receiver supervised updates and five sender PPO updates and then k rounds of observer SFT updates (we will try with $k \in \{1,2,4,8\}$ to see the importance of monitor training; as discussed, we may need to give the monitor time to catch up).

*Note: Is this task information-theoretically well-posed?* It seems like the answer should be no. Note that the receiver and observer see the same info. An important difference, however, is that they have different internal states, which is one of the main worries about steganography.
