# Dynamic university observer: equal-cycle sweep

Submitted to ORCD on 21 October 2026. **GPU validation is pending; no training result or measured GPU fit is claimed.** This implements the new experiment in section 3 of [the report](../../REPORT.md). The report's existing results describe the earlier frozen-observer experiment.

All five settings use the same original checkpoint zero, university training/held-out datasets and sender/receiver training split. Each runs **200 cycles**, with **2 receiver updates and 5 sender updates per cycle**, followed by k observer updates, for k = 0, 1, 2, 4, 8. Thus every setting gets 400 receiver updates and 1,000 sender updates; observer updates total 0, 200, 400, 800 and 1,600. Elapsed GPU time will differ.

The sender reward is `2*p_R(correct)-1 - (2*p_O(good)-1)^2 + 0.1*min(character_count/20,1)`, with PPO's KL penalty separate. The receiver trains on all 256 unfiltered examples using mean binary cross-entropy. Sender PPO keeps the original batch of 192. The character-count bonus is unchanged.

The university observer remains Qwen3.5-27B. The pinned FP8 checkpoint is **dequantized to BF16** because the installed FP8 loader does not support training. The base is frozen; rank-16 FP32 LoRA adapters train using binary Bad/Good cross-entropy. Every setting, including k=0, uses this same precision and initialization. In k=0, the observer has no optimizer or backward passes, but its Bad/Good probabilities still enter the sender reward. The **separate Qwen3-4B PIQA usefulness judge stays frozen for every k**.

Observer training uses only summaries from the current training cycle: 512 receiver and 960 sender examples. Each observer step samples 256 from this pool using an isolated deterministic RNG. Held-out examples never enter SFT. Frozen evaluations cover the same 2,000 university examples and 100 PIQA questions at cycle 0, every 10 cycles and cycle 200, after that cycle's observer updates. They record receiver accuracy, observer accuracy/confidence and PIQA rejection.

## Jobs and validation

| Job | Purpose | Status at submission check |
| --- | --- | --- |
| 26539915 | H200 validation | Pending: priority |
| 26539917 | H100 validation | Pending: priority |
| 26539918 | H200, k=0 | Pending: successful H200 validation |
| 26539919 | H200, k=1 | Pending: successful H200 validation |
| 26539921 | H200, k=2 | Pending: successful H200 validation |
| 26539923 | H200, k=4 | Pending: successful H200 validation |
| 26539926 | H200, k=8 | Pending: successful H200 validation |

Each requests one GPU, 256 GiB host RAM and 8 CPUs, eligible for `mit_normal_gpu` and `mit_preemptable`. Validation uses a real k=1 cycle with full training batches, 64 university/4 PIQA evaluation rows, and joint checkpoint restoration. Production also checks a matching source/spec/hardware validation artifact, then performs its own full initial evaluation. Slurm has not provided a start time for these submitted jobs. Earlier test-only queue estimates are not reserved starts.

Production uses 6-hour allocations, with a signal 15 minutes before the time limit and operational requeue from the latest committed joint checkpoint. Checkpoints include sender/receiver weights, observer adapters, all applicable optimizers, RNG, sampler position, PPO/KL state and the current observer training pool. A numerical, provenance or OOM failure exits unsuccessfully; it is not blindly retried. CPU offload is used during sender PPO, including temporarily moving the observer base to host RAM. Historical experiments are untouched.

## Source and checks

The self-contained [release](release/) contains the exact submitted source and configurations. Key files:

- [EXPERIMENT.md](release/EXPERIMENT.md): detailed protocol and implementation choices.
- [dynamic_training.py](release/stego/dynamic_training.py): schedule, reward integration, unfiltered receiver training and evaluation.
- [dynamic_observer_worker.py](release/orcd/dynamic_observer_worker.py): label prompt, LoRA, binary SFT and observer state.
- [dynamic_run.py](release/orcd/dynamic_run.py) and [dynamic.slurm](release/orcd/dynamic.slurm): validation gate and resumable Slurm execution.
- [LAUNCH.json](release/LAUNCH.json): exact config digests, paths, job IDs and checks at submission.

The 10 CPU tests passed in the actual legacy training environment. Tokenizer checks verified Bad=16579 and Good=14773 at the non-thinking answer boundary. CPU model inspection counted 27,356,728,560 base parameters and 116,727,808 LoRA parameters across 496 text linear modules. Those counts do not establish GPU fit. The small LoRA forward/gradient check passed; no full-model numerical equivalence is claimed.

Run the targeted tests from `release/` in the recorded training environment:

```sh
python -m unittest discover -s tests -p 'test_dynamic*.py'
```

Every `configs/dynamic_*.json` records the 60 submitted source hashes and pinned input/model/runtime hashes. Absolute ORCD paths are preserved for provenance; model weights and training checkpoint tensors are not included. Files named `PARENT_*`, `CORRECTIONS.json`, `RECOVERY_*`, the root `VALIDATION.json`, and earlier diagnostic code are inherited historical evidence, **not validation of this new sweep**. Actual GPU validation must appear at each job's explicit `validation_path` in `LAUNCH.json`.
