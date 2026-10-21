# Qwen observer rerun

Fresh initialization from the independently verified original checkpoint0; this does not continue trained sender1076. The corrected Gemma sender, full DistilBERT receiver, 2receiver/5sender schedule, optimizer and PPO settings, datasets, evaluation seeds, and four fixes are retained.

The single scientific model change is frozen Qwen/Qwen3.5-27B-FP8 revision97f5941bf617e31c5e237364a8602ce3f03a551a in place of Llama3.1-8B-int8. The exact original academic-institution observer question and offset0.75 are retained. Thinking is disabled; Yes/No single-token identities are verified in context. Left padding and final attended-token scores use tanh(logit(No)-logit(Yes)-0.75). No new decoding/classification labels train the observer.

Sender reward remains unflagged*tanh(receiver correct-class raw logit)+0.2*observer_score+0.1*min(character_count/20,1), with PPO KL separate. Receiver optimizes mean CE over256 unflagged examples via four64 minibatches and one optimizer step. Prefilter collection, accepted optimization examples, and192 sender rollouts remain separate populations.

One H100 with at least75GiB GPU memory and192GiB host RAM. Both mit_normal_gpu and mit_preemptable are requested for each single job; no duplicate writers. Six-hour Slurm allocations requeue from complete checkpoints. Three sequential logical segments have fixed cumulative active thresholds27900,55800,83700seconds (23h15m total). Segment boundaries retain model, optimizer, RNG, stream, mode, PPO/KL state and elapsed clock. Queue time is excluded; baseline/periodic evaluation counts. Final evaluation at each boundary follows its threshold and its elapsed time carries forward. An incomplete receiver collection at a budget boundary is rolled back to the previous committed checkpoint while preserving time spent.

The sender/receiver retain the original Torch2.4.0/Transformers4.43.2/TRL0.9.6/bnb0.43.3 runtime. A separate inference-only process uses pinned Torch2.10.0/Transformers5.2.0; model files and key runtime sources are hash checked. Parent and worker cache cleanup verify live tensor, mode and RNG invariants. The unchanged layout-preserving CPU offload helper applies only to PPO.step; exact CUDA toy parity must pass before models. This does not establish full-model numerical equivalence.

GPU validation occurs at startup and then through evaluation0, two receiver updates and five sender updates. FULL_CYCLE_SMOKE is written only after that committed cycle. CPU recovery/interface tests cannot establish GPU fit. Failures are recorded and do not unlock dependent stages.

Evaluate same2000 university and100PIQA at0, every50updates and each segment end. Measure C_R correct/2000, operational C_I flagged/2000, receiver PASS/FAIL counts, PIQA judge/format/overlap/union d and delta from this run evaluation0 (historical8/100 separately). Public artifacts only may be copied locally; model/state tensors and private reasoning stay remote.

Only this new run prunes old checkpoint tensors: keep checkpoint0, latest3 and all segment boundary states; all evaluation identity markers remain. All existing experiment directories, archives, baselines, environments and unrelated jobs are preserved.
