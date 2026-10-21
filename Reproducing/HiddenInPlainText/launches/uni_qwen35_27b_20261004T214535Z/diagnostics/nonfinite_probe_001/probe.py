#!/usr/bin/env python3
"""Isolated public-control diagnostic; never modifies the frozen experiment.

Outputs only STATUS.json and DIAGNOSTIC.json. No model tensors, activations,
generated text, training data, or checkpoints are serialized.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time

sys.dont_write_bytecode = True
CONFIG_SHA = "42580b347ec09edac9f497c73c30f97f0c457a7955000873b81a8d0b5494e12d"
REVISION = "97f5941bf617e31c5e237364a8602ce3f03a551a"


class BudgetExpired(BaseException):
    pass


class ProbeTerminated(BaseException):
    pass


class FirstNonfinite(RuntimeError):
    pass


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=int, default=870)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 900:
        parser.error("budget must be at most 900 seconds")
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    if output == source:
        parser.error("output must be an isolated diagnostic directory")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("STATUS.json", "DIAGNOSTIC.json")):
        parser.error("refusing to replace an existing diagnostic")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                      PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    report = {"schema_version": 1, "kind": "qwen_fp8_nonfinite_probe_001",
              "started_at": utcnow(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "probe_sha256": sha256(__file__), "config_digest": CONFIG_SHA,
              "budget_seconds": args.budget_seconds, "phase": "preflight",
              "status": "running", "quantization_probes": [], "model_probes": [],
              "production_settings_modified": False, "generated_text": False,
              "training_or_checkpoint_writes": False,
              "full_model_equivalence_claimed": False}

    def flush(phase=None):
        if phase is not None:
            report["phase"] = phase
        report["updated_at"] = utcnow()
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "DIAGNOSTIC.json", report)
        atomic_json(output / "STATUS.json", {key: report[key] for key in
                    ("kind", "status", "phase", "updated_at", "elapsed_seconds",
                     "budget_seconds", "job_id", "config_digest")})

    def on_alarm(_signum, _frame):
        raise BudgetExpired()

    def on_term(_signum, _frame):
        raise ProbeTerminated()

    signal.signal(signal.SIGALRM, on_alarm)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGUSR1, on_term)
    signal.alarm(args.budget_seconds)
    flush()
    handles = []
    exit_code = 0
    try:
        config = json.loads((source / "configs/university_training.json").read_text())
        assert hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() == CONFIG_SHA
        assert len(config["source_sha256"]) == 41
        for name, expected in config["source_sha256"].items():
            relative = Path(name)
            assert not relative.is_absolute() and ".." not in relative.parts
            assert sha256(source / relative) == expected, "Frozen source digest mismatch"
        observer = config["qwen_observer"]
        assert observer["revision"] == REVISION and observer["offset"] == .75
        assert observer["micro_batch_size"] == 4 and observer["max_input_tokens"] == 2048
        assert Path(sys.prefix).resolve() == Path(observer["python"]).parent.parent.resolve()
        assert observer["expected_versions"]["torch"] == "2.10.0"
        assert observer["expected_versions"]["transformers"] == "5.2.0"
        assert observer["expected_versions"]["triton"] == "3.6.0"
        runtime = json.loads((source / "OBSERVER_RUNTIME.json").read_text())
        assert runtime["versions"] == observer["expected_versions"]
        assert runtime["source_sha256"] == observer["runtime_source_sha256"]
        assert runtime["python"] == observer["python"]
        manifest = json.loads((source / "MODEL_MANIFEST.json").read_text())
        assert manifest["repo"] == "Qwen/Qwen3.5-27B-FP8" and manifest["revision"] == REVISION
        assert manifest["path"] == observer["model_path"]
        assert {item["path"]: item["sha256"] for item in manifest["files"]} == observer["artifact_sha256"]
        worker_path = source / "orcd/qwen_observer_worker.py"
        spec = importlib.util.spec_from_file_location("frozen_observer_probe_import", worker_path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        report["provenance"] = {"source_hashes_verified": 41,
            "source_audit_sha256": sha256(source / "SOURCE_AUDIT.json"),
            "model_manifest_sha256": sha256(source / "MODEL_MANIFEST.json"),
            "observer_runtime_manifest_sha256": sha256(source / "OBSERVER_RUNTIME.json"),
            "runtime_audit": worker.verify_runtime(observer)}
        flush("hardware_preflight")
        import numpy as np
        import torch
        import transformers
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from transformers.integrations.finegrained_fp8 import FP8Linear, act_quant
        imported_runtime = {}
        for target in (FP8Linear, Qwen3_5ForConditionalGeneration):
            imported_path = str(Path(inspect.getfile(target)).resolve())
            assert imported_path in observer["runtime_source_sha256"]
            assert sha256(imported_path) == observer["runtime_source_sha256"][imported_path]
            imported_runtime[imported_path] = sha256(imported_path)
        report["provenance"]["actually_imported_runtime_sha256"] = imported_runtime
        assert torch.cuda.device_count() == 1
        prop = torch.cuda.get_device_properties(0)
        assert "H100" in prop.name and prop.total_memory >= 75 * 1024**3
        assert torch.cuda.get_device_capability(0)[0] >= 9
        report["hardware"] = {"gpu_name": prop.name, "total_bytes": prop.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__, "transformers_version": transformers.__version__}

        def stats(tensor):
            value = tensor.detach().float()
            finite = torch.isfinite(value)
            count = int(finite.sum().item())
            return {"numel": value.numel(), "finite_count": count,
                    "nonfinite_count": value.numel() - count,
                    "zero_count": int((value == 0).sum().item()),
                    "finite_min": float(value[finite].min().item()) if count else None,
                    "finite_max": float(value[finite].max().item()) if count else None}

        flush("frozen_quantization_probes")
        generator = torch.Generator(device="cuda").manual_seed(0)
        for name in ("zeros", "ones", "seeded_random", "mixed_zero_nonzero_rows"):
            if name == "zeros":
                value = torch.zeros((4, 128), device="cuda", dtype=torch.bfloat16)
            elif name == "ones":
                value = torch.ones((4, 128), device="cuda", dtype=torch.bfloat16)
            elif name == "seeded_random":
                value = torch.randn((4, 128), generator=generator, device="cuda", dtype=torch.bfloat16)
            else:
                value = torch.ones((4, 128), device="cuda", dtype=torch.bfloat16)
                value[0::2].zero_()
            row = {"name": name, "shape": [4, 128], "dtype": "torch.bfloat16", "block_size": 128}
            report["quantization_probes"].append(row)
            try:
                with torch.inference_mode():
                    quantized, scales = act_quant(value, 128)
                    torch.cuda.synchronize()
                    row.update(status="complete", input=stats(value), quantized=stats(quantized), scales=stats(scales))
                del quantized, scales
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__)
            del value
            flush()

        flush("model_artifact_verification")
        model_path = Path(observer["model_path"])
        report["model_artifact_hashes_verified"] = 0
        for item in manifest["files"]:
            relative = Path(item["path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            artifact = model_path / relative
            assert artifact.stat().st_size == item["size"]
            assert sha256(artifact) == item["sha256"], "Pinned model digest mismatch"
            report["model_artifact_hashes_verified"] += 1
            flush()
        model_config = json.loads((model_path / "config.json").read_text())
        assert model_config["model_type"] == "qwen3_5"
        assert model_config["quantization_config"]["quant_method"] == "fp8"
        assert model_config["quantization_config"]["weight_block_size"] == [128, 128]
        random.seed(observer["worker_seed"])
        np.random.seed(observer["worker_seed"])
        torch.manual_seed(observer["worker_seed"])
        torch.cuda.manual_seed_all(observer["worker_seed"])
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True,
                        trust_remote_code=False, revision=REVISION)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        assert tokenizer.pad_token_id is not None
        prompts = [worker.format_prompt(tokenizer, observer["prompt"], text) for text in worker.SELF_TEST_TEXTS]
        answer_ids = worker.answer_token_ids(tokenizer, prompts[0])
        assert answer_ids == {"No": 2665, "Yes": 9175}
        report["public_controls"] = {"count": len(prompts), "text_sha256": worker.text_hashes(worker.SELF_TEST_TEXTS),
            "answer_token_ids": answer_ids, "non_thinking": True, "padding_side": "left"}
        flush("model_load")
        model = Qwen3_5ForConditionalGeneration.from_pretrained(str(model_path), revision=REVISION,
                    local_files_only=True, trust_remote_code=False, dtype="auto",
                    device_map={"": 0}, attn_implementation="sdpa")
        model.eval()
        model.requires_grad_(False)
        model.config.use_cache = False
        if hasattr(model.config, "text_config"):
            model.config.text_config.use_cache = False
        dtype_counts = {}
        for parameter in model.parameters():
            assert parameter.device.type == "cuda" and parameter.device.index == 0
            key = str(parameter.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
        assert any("float8" in key for key in dtype_counts)
        fp8_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, FP8Linear)]
        assert fp8_modules
        report["model"] = {"class": type(model).__name__, "parameter_dtype_numel": dtype_counts,
            "fp8_linear_count": len(fp8_modules), "fp8_linear_names": [name for name, _ in fp8_modules],
            "use_cache": False, "logits_to_keep": 1, "attn_implementation": "sdpa",
            "cuda_allocated_bytes_after_load": torch.cuda.memory_allocated(0)}
        active = {}

        def make_hook(name):
            def hook(_module, inputs, result):
                active["fp8_modules_checked"] += 1
                if bool(torch.isfinite(result).all().item()):
                    return
                value = inputs[0].detach()
                item = {"module": name, "input_shape": list(value.shape), "output_shape": list(result.shape),
                        "input": stats(value), "output": stats(result)}
                if value.shape[-1] % 128 == 0:
                    zero_blocks = (value.reshape(-1, 128) == 0).all(dim=-1)
                    blocks_per_token = value.shape[-1] // 128
                    per_token = zero_blocks.reshape(-1, blocks_per_token)
                    item["input_zero_blocks"] = int(zero_blocks.sum().item())
                    item["input_total_blocks"] = zero_blocks.numel()
                    item["input_all_zero_token_rows"] = int(per_token.all(dim=-1).sum().item())
                    mask = active["attention_mask"].reshape(-1).bool()
                    if per_token.shape[0] == mask.numel():
                        item["zero_blocks_at_padding"] = int(per_token[~mask].sum().item())
                        item["zero_blocks_at_attended_tokens"] = int(per_token[mask].sum().item())
                mask = active["attention_mask"].reshape(-1).bool()
                nonfinite_rows = ~torch.isfinite(result.detach()).reshape(-1, result.shape[-1]).all(dim=-1)
                if nonfinite_rows.numel() == mask.numel():
                    item["nonfinite_output_rows_at_padding"] = int(nonfinite_rows[~mask].sum().item())
                    item["nonfinite_output_rows_at_attended_tokens"] = int(nonfinite_rows[mask].sum().item())
                active["row"]["first_nonfinite_fp8"] = item
                raise FirstNonfinite()
            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in fp8_modules]
        for label, selected_prompts in (("padded_batch", prompts), ("single_control_0", prompts[:1]),
                                        ("single_control_1", prompts[1:2])):
            row = {"name": label, "status": "running", "batch_size": len(selected_prompts)}
            report["model_probes"].append(row)
            flush("model_probe_" + label)
            encoded = tokenizer(selected_prompts, add_special_tokens=False, padding=False, truncation=False)
            lengths = worker.check_input_lengths(encoded["input_ids"], observer["max_input_tokens"])
            batch = tokenizer.pad(encoded, padding=True, return_tensors="pt")
            assert bool(batch["attention_mask"][:, -1].all())
            batch = {key: value.to("cuda:0") for key, value in batch.items() if key in ("input_ids", "attention_mask")}
            row.update(input_token_lengths=lengths, padding_tokens=int((batch["attention_mask"] == 0).sum().item()))
            active.update(row=row, attention_mask=batch["attention_mask"], fp8_modules_checked=0)
            try:
                with torch.inference_mode():
                    result = model(**batch, use_cache=False, logits_to_keep=1, return_dict=True)
                    selected = result.logits[:, -1, [answer_ids["No"], answer_ids["Yes"]]]
                    row.update(status="complete", selected_logit_statistics=stats(selected))
                    row["all_selected_logits_finite"] = bool(torch.isfinite(selected).all().item())
                    del selected, result
            except FirstNonfinite:
                row["status"] = "aborted_at_first_nonfinite_fp8"
            except Exception as error:
                row.update(status="error", error_type=type(error).__name__)
            finally:
                row["fp8_modules_checked"] = active["fp8_modules_checked"]
                active.clear()
                del batch
                flush()
        report["status"] = "complete"
        flush("complete")
    except BudgetExpired:
        report["status"] = "budget_exhausted"
        exit_code = 2
    except ProbeTerminated:
        report["status"] = "terminated"
        exit_code = 3
    except Exception as error:
        report["status"] = "error"
        report["error_type"] = type(error).__name__
        exit_code = 1
    finally:
        signal.alarm(0)
        for handle in handles:
            handle.remove()
        report["finished_at"] = utcnow()
        flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
