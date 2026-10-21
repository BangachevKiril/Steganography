# University corrected implementation rerun

Fresh run from the original verified checkpoint zero, following the request to redo the experiment with the identified bugs fixed. The completed original experiment remains a separate control.

- Receiver: all 256 filtered examples contribute to one mean-loss optimizer step through four minibatches of 64.
- Sender: pass unpadded queries to TRL so generation and PPO receive genuine attention masks.
- Observer: score the last non-padding token of each row, retaining right padding, the prompt, Yes/No formula and threshold.
- Receiver reward scoring: temporary evaluation mode; supervised training retains dropout.

The released raw-logit reward, character-based length reward, argmax decisions, learning rates, PPO/KL settings, 2:5 update schedule, batches, data cohorts, model revisions and precision are retained. This is a corrected implementation experiment, not a claim of exact paper reproduction or an isolated single-fix ablation.

The initial models and RNG come from the original hash-verified baseline checkpoint; optimizer states start empty. A complete frozen evaluation of the new checkpoint zero measures the baseline under the corrected observer. Later d changes use that measured baseline. Historical C_I used pad-position scoring and is not directly comparable.

Runtime uses one L40S with layout-preserving saved-tensor CPU offload only around PPO.step, plus unused allocator-cache release before the unchanged PIQA judge subprocess. The previously audited helper implementations remain hash-pinned and unchanged. GPU toy parity, checkpoint/initialization identity, numerical finiteness and cleanup invariants must pass.

The budget is 27,900 active seconds, including fresh step-zero and periodic evaluations, followed by a final frozen evaluation. Six-hour, 128G host-memory allocations requeue and restore the saved elapsed budget; throughput does not change the budget. Each frozen evaluation uses the same 2,000 university rows and 100 PIQA questions. All outputs and launch records are separate. Only public artifacts and checkpoint manifests may be copied locally; weights remain remote.
