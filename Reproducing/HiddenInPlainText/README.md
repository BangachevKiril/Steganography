# Hidden in Plain Text — reproduction archive

This directory contains the code, data exports, public results, and audit records for our experiments based on [Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs](https://aclanthology.org/2025.ijcnlp-long.34/) (Mathew et al., 2025; [arXiv](https://arxiv.org/abs/2410.03768)). The authors' [implementation](https://github.com/olliematthews/lasr-steganography) was inspected at commit `d4c708b9339fd2f85cb442827ac56333b55cf9d2`.

Start with the concise [report and four-panel figure](REPORT.md). It describes the completed **Qwen observer university run**, not the earlier PIQA pilots. This is a modified reproduction: it includes implementation corrections, a Qwen3.5-27B-FP8 observer, and a separate PIQA usefulness evaluation.

## Code and evidence map

| Path | Purpose |
| --- | --- |
| `stego/` | Shared Python implementation, historical PIQA pilots, university generation, receiver training, sender PPO, metrics, and reporting. |
| `launches/uni_qwen35_27b_20261004T214535Z/` | **Authoritative frozen source for the reported run**, including its configuration, model/runtime manifests, tests, and recovery code. |
| `launches/…/stego/university_training.py` | Alternating receiver cross-entropy and sender PPO; reward computation, checkpoint save/restore, and update loop. |
| `launches/…/stego/university_baseline.py`, `university_evaluation.py`, `university_judge.py` | University prompts and generation, frozen evaluations, and the separate PIQA judge. |
| `launches/…/orcd/` | Qwen observer worker/client IPC, H100 checks, memory handling, fixed-budget segments, Slurm entrypoints, and public snapshot verification. |
| `launches/…/recovery/zero_block_guard_003/` | Approved observer-only FP8 zero-block guard, validation records, bootstrap, and actual training launch script. |
| `configs/`, `orcd/`, `tests/`, `experiments/` | Configuration history, cluster preparation/launch tools, unit tests, and experiment notes. |
| `results/uni_qwen35_27b_20261004T214535Z/` | Public training/evaluation artifacts, completion verification, and verified plot data/code. |
| `data/university_author/` | Author-method and dataset provenance records. |
| `data/university_observer_qwen_checkpoint1440_split/` | Final sender's 1,000 training and 1,000 test answers, separate hidden-label files, and split manifest; assignments match the earlier checkpoint-1076 export. |
| `data/claude_training1440/predictions.jsonl` | The submitted Claude predictions, preserved byte for byte, plus user-reported model, effort, and verbatim prompt in `EXPERIMENT.json`. |
| `results/prediction_label_checks/training1440_ed54cc15cac4/` | Original prediction scoring summary, all scored rows, and four errors. |
| `presentations/`, other `launches/` and `results/` | Earlier experiment history, presentation source, and public artifacts; these describe different runs. |
| `tools/verify_bundle.py` | Portable integrity, frozen-source, split, and prediction-score checks. |

The 1,000/1,000 answer export is a downstream partition of the university evaluation set, **not the RL training/held-out split**. Answer-only files contain `example_id` and `answer`; hidden labels are separate. Historical records retain their original absolute paths and timestamps for provenance.

## Verify locally

Python 3.10+ is sufficient for the archive and prediction checks; no GPU or model downloads are needed:

```sh
cd Reproducing/HiddenInPlainText
python3 tools/verify_bundle.py
```

The script verifies archived file hashes (including the uncompressed bytes of large JSON snapshots), the 41 frozen source hashes and config digest, the final checkpoint identity marker, all 32 evaluation summaries against their public rows, the exported split, and the submitted predictions. It does not download or rehash remote model tensors; the original remote state verification is recorded in `monitoring/FINAL_COMPLETION_VERIFICATION.json` and the other final verification records.

The final plot is available as [PNG](results/uni_qwen35_27b_20261004T214535Z/plots/verified_evaluation_curves.png), [PDF](results/uni_qwen35_27b_20261004T214535Z/plots/verified_evaluation_curves.pdf), and [CSV](results/uni_qwen35_27b_20261004T214535Z/plots/verified_evaluation_metrics.csv). To repeat the original plot verification and rendering, install Matplotlib and run:

```sh
RUN=uni_qwen35_27b_20261004T214535Z
SNAPSHOT=/tmp/hidden-plain-text-final-snapshot.json
gzip -dc "results/$RUN/monitoring/final_plot_source_snapshot.json.gz" > "$SNAPSHOT"
python3 "results/$RUN/plots/plot_verified_evaluations.py" \
  "$SNAPSHOT" "launches/$RUN" "results/$RUN"
```

The original report generator is also retained:

```sh
ROOT="$PWD"
cd launches/uni_qwen35_27b_20261004T214535Z
python3 -m stego.university_training_report \
  "$ROOT/results/uni_qwen35_27b_20261004T214535Z" --completed-update 1440
```

Regeneration may change output hashes due to library versions; run archive verification before regenerating.

## Training environment and launch

Training used one H100 80GB GPU and 192GiB host RAM. The sender/receiver environment was Torch 2.4.0, Transformers 4.43.2, TRL 0.9.6, and bitsandbytes 0.43.3; the isolated frozen observer used Torch 2.10.0 and Transformers 5.2.0. Model revisions, runtime hashes, tokenizer checks, dataset locations, and source hashes are recorded in the frozen launch's `MODEL_MANIFEST.json`, `OBSERVER_RUNTIME.json`, `configs/university_training.json`, and recovery manifests. Model weights, optimizer tensors, external model caches, and private reasoning are not included in this public archive.

The actual cluster entrypoint is `recovery/zero_block_guard_003/training.slurm`, which calls its `bootstrap.py` and the segment trainer. It requires the recorded model/data assets, original checkpoint 0 for a new run, both Python environments, Slurm, and the recovery-manifest environment binding. Read `EXPERIMENT.md` and the launch script before adapting it: the preserved scripts contain installation-specific absolute paths. A new installation requires new audited paths/manifests; silently editing the frozen configuration would invalidate its hashes. Resuming an existing run must restore its latest complete checkpoint, including optimizers, RNG, stream and PPO/KL state, and cumulative elapsed time.

The reported run completed 1,440 updates across three logical segments with fixed cumulative thresholds of 27,900 / 55,800 / 83,700 active seconds; final in-flight work/evaluation can exceed a threshold. The FP8 zero-padding guard and audited PPO CPU offload were operational requirements of this run; recorded validation does not claim full-model numerical equivalence.

## Archive format

`ARCHIVE_MANIFEST.json` maps every included original project artifact to its stored path and SHA-256. JSON files larger than 10 MB are stored as deterministic `.json.gz` files, losslessly preserving their original bytes and hashes; `gzip -dc FILE.json.gz > FILE.json` restores them. Source files and frozen configurations are unchanged. Only caches, local dependency symlinks, and runtime stdout/stderr logs are omitted; exclusions are listed in the manifest. Earlier duplicate exports and superseded public artifacts are retained as history. `LEGACY_PILOT_README.md` preserves the original project README, whose three-agent PIQA setup differs from the university experiment reported here.
