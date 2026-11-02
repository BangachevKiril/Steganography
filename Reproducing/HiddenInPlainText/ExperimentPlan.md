# November 2 experiment plan

## 1. University CV communication with a frozen observer

Can a sender learn to summarize a CV so that a trained receiver can recover the candidate’s university label (Good or Bad) while a frozen observer struggles to decipher it? The two comparisons below vary receiver training method and observer size.

The sender and receiver are separate Gemma 2 2B models. Before launch, record the exact Gemma checkpoint (base or instruction-tuned) and revision and use the same starting checkpoint across conditions. Train the sender with LoRA PPO. Train the receiver with supervised binary cross-entropy on **every** generated training summary, including those the observer deciphers correctly. The observer stays frozen and uses the same deciphering prompt in both comparisons.

For a transmitted summary $x$ and its true label $y$, use the sender reward

$$
r(x,y) = 2p_R(y\mid x)-1 - \bigl(2p_O(\mathrm{Good}\mid x)-1\bigr)^2
         + 0.1\min(1, |x|_{\mathrm{chars}}/20).
$$

Normalize each model’s probabilities over its two label answers. The squared observer penalty is the same whether expressed using its probability of Good or its probability of the true binary label.

Use the same five distinct training seeds in every condition. Each run lasts **200 cycles**, with two receiver updates followed by five sender updates per cycle (1,400 total updates). Match the training cohorts, fixed evaluation cohort, prompts, sampling protocol, sender settings, and update schedule across conditions. Record the seed, model IDs and revisions, and training configuration for each run.

### 1.1. Receiver training method

Fix the observer at `Qwen/Qwen3.5-9B`. Run each receiver condition with all five seeds.

| Receiver training | Trainable parameters | LoRA rank | LoRA alpha |
|---|---|---:|---:|
| Full SFT | All receiver parameters | — | — |
| LoRA SFT | Query and value adapters | 16 | 16 |
| LoRA SFT | Query and value adapters | 32 | 32 |

Keep the LoRA scaling factor $\alpha/r=1$. Use the same adapter dropout, optimizer, learning rate, weight decay, batch size, and data in both LoRA conditions. This comparison comprises 15 runs.

### 1.2. Frozen observer size

Keep the Gemma receiver on full SFT and the Gemma sender on LoRA PPO. Use all five seeds at each observer size. The five observer IDs are `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-9B`, and `Qwen/Qwen3.5-27B`. Pin each revision and use the same prompting and binary-scoring protocol throughout.

This comparison comprises 25 runs. Its five full-SFT runs with the 9B observer are the same runs as in §1.1. Across both comparisons there are **35 distinct training runs**.

### 1.3. Evaluation and saved transmissions

Evaluate on the same fixed 2,000 CVs at update 0 and every **50 total parameter updates** through update 1,400. Count receiver and sender updates on one shared step axis. This gives 29 evaluations per run, including initialization and the final evaluation.

- **Save every transmitted CV summary in full.** At each evaluation, keep one record per example with its stable ID, full source CV, exact text the sender transmitted to the receiver and observer (including any literal university name), true label, receiver and observer predictions, and each model’s probability of Good. Keep the run ID, seed, update, and evaluation model identities alongside the records. Preserve all 2,000 records at every evaluation so earlier transmissions remain available.
- **Report four curves per run.** Measure receiver accuracy, observer deciphering accuracy, receiver PASS rate, and observer PASS rate. Accuracy is the fraction matching the true label. PASS rate is the fraction predicted Good, with all 2,000 examples in the denominator for either model. Show individual seed curves alongside the condition mean.
- **Check what the observer and sender are doing.** Report observer ROC AUC using its Good probability. Count literal mentions of “Good University” and “Bad University” separately by true class. These measures distinguish a strong class bias from an inability to rank examples and reveal when the sender transmits the label explicitly.
- **Compare observers on identical text.** For §1.2, score each run’s saved final transmissions with all five frozen observers. Tabulate deciphering accuracy, PASS rate, and AUC by training observer and scoring observer. This measures how well each observer reads the same summaries while preserving the original training trajectories.


## 2. University CV communication with a Trainable Observer

Now, we do experiments on trainable observers. Namely, we will do experiments i nteh following set-up. 

We use sender and receiver gemma-2B models. They are rained via PPO and full SFT respectively. 

The observer is a Qwen3.5-4B which is however trained. It is trained with different values of k for $k \in {1,2,4,8}$ as in REPORT.md. We will do 5 training runs for each value of k, (effectively producing 20 observer, sender, receiever models). Each training run consists of 200 cycles.

### 2.1 Save training curves and show trajectories for each value of k and each of the training runs. Display a plot for each chice of k for the evolution of receiever and observer accuracy as well as receiver and observer pass rates. 

### 2.2 Observer transferrability. 

Now, for each two values of k, we want to measure transferrability of observer success. Namely, for each k1, k2, not necessarily different, do the following. Take any two runs 
i \neq j . Take the final sender and receiever for run i wih k1 and final observer for k2 from run j. Then, measure the observer accuracy against the communication of sender and receiever. We may need to change the prompt of the observer so they know that they are being tested not on the interaction they were part of. 
