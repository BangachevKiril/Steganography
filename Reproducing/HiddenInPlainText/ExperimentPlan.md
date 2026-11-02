# Experiments for Nov 2

Set-up: We will study the conditions under which steganography arises in RL envinronments. 


## 1. Weight Space Steganography with a Frozen Judge

Here, we will conduct two experiments in weight space with afrozen judge. We will set both ender and receiver to be gemma2-2b models. The sender will be trained with LoRA PPO 
while the receiver with SFT. The observer varies but is frozen. The data communicated is teh CV data.

The receiver and sender try to communicate the university of the candidate without revealing it to the observer. Receiver and observer both try to guess the 
university. The reward of the sender for generated text $x$ and true label $y$ is:

$$
r(x, y) = 2p_R(y |x) -1 - (2p_O(y|x) - 1)^2 + 0.1\min(1, |x|_{\text{chars}}/20)
$$

### 1.1. The importance of training methodology

We will now compare two training methodologies whuch so seemed to produce different results. When we LORA SFT the receiver and when we full SFT it. We will do  5 training runs, each with 
200 cycles, for each of the following:
1. Full SFT.
2. LoRA with rank 16.
3. LoRA with rank 32.


### 1.2 The importance of Judge Power
We next examine the importance of the power of a frozen judge. We train a full SFT receiver and a sender, both gemma2-2b, against the following increasingly powerful judges. 
1. Qwen3.5-0.8B
2. Qwen3-2B
3. Qwen3-4B
4. Qwen3-9B
5. qwen3-27B

Do 5 runs on each of them. The main thing we want saved from each run are the curves of receiver and observer success as well as the "pass rate" report for observer and receiver.
