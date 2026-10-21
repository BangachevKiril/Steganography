#!/usr/bin/env python3
"""Bounded isolated validation of the explicitly authorized FP8 zero-block guard.

Only scalar/public metadata is persisted as DIAGNOSTIC.json and STATUS.json.
This script never changes source, runtime, model files, training, or checkpoints.
The original quantizer is reconstructed from the verified guarded source and
imported under a separate module name from temporary storage for exact parity.
"""
from __future__ import annotations

import argparse
import ast
import copy
import datetime
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import tempfile
import time

sys.dont_write_bytecode = True
CONFIG_SHA = "42580b347ec09edac9f497c73c30f97f0c457a7955000873b81a8d0b5494e12d"
REVISION = "97f5941bf617e31c5e237364a8602ce3f03a551a"
ORIGINAL_SHA = "a486628c4f03abfe299653200ebd2cc577a9ff514fa37cc773031440702107a7"
GUARDED_SHA = "5ea1130b150506e7a7e94534ee4e9aedd6e57ef576bfc50ced6e00927d926612"
ORIGINAL_LINE = "    s = tl.max(tl.abs(x)) / 448.0\n"
GUARDED_LINES = "    absmax = tl.max(tl.abs(x))\n    s = tl.where(absmax == 0.0, 1.0, absmax / 448.0)\n"


class BudgetExpired(BaseException):
    pass


class ProbeTerminated(BaseException):
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


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original_quantizer_source(source):
    """Extract actual original kernel/wrapper text without rewriting their logic."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    names = ("act_quant_kernel", "act_quant")
    parts = ["from __future__ import annotations\nimport torch\nimport triton\nimport triton.language as tl\nfrom typing import *\n"]
    for name in names:
        matches = [node for node in ast.walk(tree)
                   if isinstance(node, ast.FunctionDef) and node.name == name]
        assert len(matches) == 1, "Original quantizer function missing or ambiguous"
        node = matches[0]
        start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
        parts.append("".join(lines[start:node.end_lineno]) + "\n\n")
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=int, default=840)
    args = parser.parse_args()
    if not 1 <= args.budget_seconds <= 850:
        parser.error("budget must be at most 850 seconds")
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    if output == source or source in output.parents:
        parser.error("output must be outside the frozen release")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("DIAGNOSTIC.json", "STATUS.json")):
        parser.error("refusing to replace an existing diagnostic")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", PYTHONDONTWRITEBYTECODE="1")
    started = time.monotonic()
    report = {"schema_version": 1, "kind": "qwen_fp8_zero_guard_validation_001",
              "started_at": utcnow(), "job_id": os.environ.get("SLURM_JOB_ID"),
              "script_sha256": sha256(__file__), "slurm_sha256": sha256(Path(__file__).with_name("validate.slurm")),
              "config_digest": CONFIG_SHA, "budget_seconds": args.budget_seconds,
              "runtime_after_sha256": GUARDED_SHA,
              "phase": "preflight", "status": "running", "validation_passed": False,
              "runtime_modified_by_this_script": False, "generated_text": False,
              "training_or_checkpoint_writes": False, "full_model_equivalence_claimed": False,
              "combined_training_gpu_fit_verified": False,
              "quantization_probes": [], "model_probes": []}

    def flush(phase=None):
        if phase is not None:
            report["phase"] = phase
        report["updated_at"] = utcnow()
        report["elapsed_seconds"] = time.monotonic() - started
        atomic_json(output / "DIAGNOSTIC.json", report)
        atomic_json(output / "STATUS.json", {key: report[key] for key in
                    ("kind", "status", "phase", "updated_at", "elapsed_seconds", "budget_seconds",
                     "job_id", "config_digest", "validation_passed")})

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
        runtime = json.loads((source / "OBSERVER_RUNTIME.json").read_text())
        assert runtime["versions"] == observer["expected_versions"]
        assert runtime["source_sha256"] == observer["runtime_source_sha256"]
        assert runtime["python"] == observer["python"]
        manifest = json.loads((source / "MODEL_MANIFEST.json").read_text())
        assert manifest["repo"] == "Qwen/Qwen3.5-27B-FP8" and manifest["revision"] == REVISION
        assert manifest["path"] == observer["model_path"]
        assert {item["path"]: item["sha256"] for item in manifest["files"]} == observer["artifact_sha256"]
        versions = {name: importlib.metadata.version(name) for name in observer["expected_versions"]}
        assert versions == observer["expected_versions"], "Pinned runtime package version changed"
        effective_hashes = {}
        for name, expected in observer["runtime_source_sha256"].items():
            path = Path(name)
            assert path.is_absolute() and ".." not in path.parts
            effective = GUARDED_SHA if path.name == "finegrained_fp8.py" else expected
            if path.name == "finegrained_fp8.py":
                assert expected == ORIGINAL_SHA
            assert sha256(path) == effective, "Unexpected runtime source digest"
            effective_hashes[name] = effective
        worker = import_file("frozen_qwen_guard_validation_worker", source / "orcd/qwen_observer_worker.py")
        report["provenance"] = {"source_hashes_verified": 41,
            "source_audit_sha256": sha256(source / "SOURCE_AUDIT.json"),
            "model_manifest_sha256": sha256(source / "MODEL_MANIFEST.json"),
            "observer_runtime_manifest_sha256": sha256(source / "OBSERVER_RUNTIME.json"),
            "expected_versions_verified": versions, "effective_runtime_source_sha256": effective_hashes,
            "original_fp8_sha256": ORIGINAL_SHA, "guarded_fp8_sha256": GUARDED_SHA}
        flush("hardware_preflight")
        import numpy as np
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
        from transformers.integrations import finegrained_fp8 as fp8
        imported = {}
        for target in (fp8.FP8Linear, Qwen3_5ForConditionalGeneration):
            path = str(Path(inspect.getfile(target)).resolve())
            assert path in effective_hashes and sha256(path) == effective_hashes[path]
            imported[path] = sha256(path)
        report["provenance"]["actually_imported_runtime_sha256"] = imported
        assert torch.cuda.device_count() == 1
        prop = torch.cuda.get_device_properties(0)
        assert "H100" in prop.name and prop.total_memory >= 75 * 1024**3
        assert torch.cuda.get_device_capability(0)[0] >= 9
        report["hardware"] = {"gpu_name": prop.name, "total_bytes": prop.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)), "torch_version": torch.__version__,
            "slurm_mem_per_node_mib": int(os.environ.get("SLURM_MEM_PER_NODE", "0"))}
        assert report["hardware"]["slurm_mem_per_node_mib"] >= 192 * 1024

        guarded_source = Path(inspect.getfile(fp8)).read_text()
        assert guarded_source.count(GUARDED_LINES) == 1
        original_source = guarded_source.replace(GUARDED_LINES, ORIGINAL_LINE)
        assert hashlib.sha256(original_source.encode()).hexdigest() == ORIGINAL_SHA
        extracted = original_quantizer_source(original_source)
        report["provenance"]["original_quantizer_extracted_sha256"] = hashlib.sha256(extracted.encode()).hexdigest()
        flush("quantizer_parity")
        generator = torch.Generator(device="cuda").manual_seed(0)
        with tempfile.TemporaryDirectory(prefix="qwen-original-quantizer-") as temporary:
            original_path = Path(temporary) / "original_quantizer.py"
            original_path.write_text(extracted)
            original = import_file("qwen_original_quantizer_validation", original_path)
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                for name, width in (("ones", 128), ("signed_ramp", 128), ("random_small", 5120),
                                    ("random_unit", 5120), ("random_large", 10240),
                                    ("zeros", 5120), ("mixed_zero_nonzero", 5120)):
                    if name in ("ones", "zeros"):
                        value = torch.full((4, width), 0 if name == "zeros" else 1, device="cuda", dtype=dtype)
                    elif name == "signed_ramp":
                        value = torch.linspace(-4, 4, width, device="cuda", dtype=dtype).repeat(4, 1)
                    else:
                        value = torch.randn((4, width), generator=generator, device="cuda", dtype=dtype)
                        if name == "random_small":
                            value.mul_(.001)
                        elif name == "random_large":
                            value.mul_(100)
                        elif name == "mixed_zero_nonzero":
                            value[0::2].zero_()
                    with torch.inference_mode():
                        old_q, old_s = original.act_quant(value, 128)
                        new_q, new_s = fp8.act_quant(value, 128)
                        torch.cuda.synchronize()
                        zero_blocks = (value.reshape(-1, 128) == 0).all(dim=1)
                        nonzero = ~zero_blocks
                        old_bits = old_q.contiguous().view(torch.uint8).reshape(-1, 128)
                        new_bits = new_q.contiguous().view(torch.uint8).reshape(-1, 128)
                        row = {"name": name, "dtype": str(dtype), "shape": [4, width], "block_size": 128,
                            "zero_block_count": int(zero_blocks.sum().item()),
                            "nonzero_block_count": int(nonzero.sum().item()),
                            "nonzero_quantized_bitwise_equal": bool(torch.equal(old_bits[nonzero], new_bits[nonzero])),
                            "nonzero_scales_exactly_equal": bool(torch.equal(old_s.reshape(-1)[nonzero], new_s.reshape(-1)[nonzero])),
                            "guarded_quantized_all_finite": bool(torch.isfinite(new_q.float()).all().item()),
                            "guarded_scales_all_finite": bool(torch.isfinite(new_s).all().item()),
                            "guarded_zero_blocks_are_zero": bool((new_q.float().reshape(-1, 128)[zero_blocks] == 0).all().item()),
                            "guarded_zero_scales_are_one": bool((new_s.reshape(-1)[zero_blocks] == 1).all().item()),
                            "original_zero_nonfinite_count": int((~torch.isfinite(old_q.float().reshape(-1, 128)[zero_blocks])).sum().item())}
                        report["quantization_probes"].append(row)
                        for key in ("nonzero_quantized_bitwise_equal", "nonzero_scales_exactly_equal",
                                    "guarded_quantized_all_finite", "guarded_scales_all_finite",
                                    "guarded_zero_blocks_are_zero", "guarded_zero_scales_are_one"):
                            assert row[key], "Quantizer validation failed: " + key
                        assert row["original_zero_nonfinite_count"] == row["zero_block_count"] * 128
                    del value, old_q, old_s, new_q, new_s, old_bits, new_bits, zero_blocks, nonzero
                    flush()
        report["quantizer_validation_passed"] = True
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
        controls = worker.SELF_TEST_TEXTS * 2
        prompts = [worker.format_prompt(tokenizer, observer["prompt"], text) for text in controls]
        answer_ids = worker.answer_token_ids(tokenizer, prompts[0])
        assert answer_ids == {"No": 2665, "Yes": 9175}
        report["public_controls"] = {"count": len(controls), "distinct_count": len(set(controls)),
            "text_sha256": worker.text_hashes(controls), "answer_token_ids": answer_ids,
            "non_thinking": all(item.endswith(worker.NON_THINKING_SUFFIX) for item in prompts),
            "assistant_prefix_sha256": hashlib.sha256(worker.NON_THINKING_SUFFIX.encode()).hexdigest(),
            "observer_prompt_sha256": hashlib.sha256(observer["prompt"].encode()).hexdigest(),
            "padding_side": "left", "micro_batch_size": 4, "offset": .75, "truncation": False}
        assert report["public_controls"]["non_thinking"]
        flush("model_load")
        # This diagnostic-only copy honestly describes the sole approved runtime
        # change. The frozen configuration and manifests are never rewritten.
        effective_observer = copy.deepcopy(observer)
        fp8_paths = [name for name in effective_observer["runtime_source_sha256"]
                     if Path(name).name == "finegrained_fp8.py"]
        assert len(fp8_paths) == 1
        effective_observer["runtime_source_sha256"][fp8_paths[0]] = GUARDED_SHA
        adapter = worker.FrozenQwenObserver(effective_observer)
        model, tokenizer = adapter.model, adapter.tokenizer
        assert adapter.answer_ids == answer_ids
        report["frozen_worker_startup"] = adapter.ready
        dtype_counts = {}
        for parameter in model.parameters():
            assert parameter.device.type == "cuda" and parameter.device.index == 0
            key = str(parameter.dtype)
            dtype_counts[key] = dtype_counts.get(key, 0) + parameter.numel()
        assert any("float8" in key for key in dtype_counts)
        fp8_modules = [(name, module) for name, module in model.named_modules() if isinstance(module, fp8.FP8Linear)]
        assert len(fp8_modules) == 400
        report["model"] = {"class": type(model).__name__, "parameter_dtype_numel": dtype_counts,
            "fp8_linear_count": len(fp8_modules), "use_cache": False, "logits_to_keep": 1,
            "attn_implementation": "sdpa", "cuda_allocated_bytes_after_load": torch.cuda.memory_allocated(0)}
        active = {}

        def make_hook(name):
            def hook(_module, _inputs, result):
                active["fp8_modules_checked"] += 1
                if not bool(torch.isfinite(result).all().item()):
                    active["row"]["first_nonfinite_fp8_module"] = name
                    raise RuntimeError("Nonfinite FP8 projection")
            return hook

        handles = [module.register_forward_hook(make_hook(name)) for name, module in fp8_modules]
        batch_rows = None
        singleton_rows = []
        for label, texts in [("padded_microbatch4", controls)] + [
                ("singleton_" + str(index), [text]) for index, text in enumerate(worker.SELF_TEST_TEXTS)]:
            row = {"name": label, "status": "running", "batch_size": len(texts)}
            report["model_probes"].append(row)
            flush("model_probe_" + label)
            rendered = [worker.format_prompt(tokenizer, observer["prompt"], text) for text in texts]
            encoded = tokenizer(rendered, add_special_tokens=False, padding=False, truncation=False)
            lengths = worker.check_input_lengths(encoded["input_ids"], 2048)
            row.update(input_token_lengths=lengths, padding_tokens=max(lengths) * len(lengths) - sum(lengths))
            if label == "padded_microbatch4":
                assert row["batch_size"] == 4 and row["padding_tokens"] > 0
            else:
                assert row["padding_tokens"] == 0
            active.update(row=row, fp8_modules_checked=0)
            rows = adapter.score(texts)
            row.update(status="complete", scalar_scores=rows, selected_logits_all_finite=True,
                       rng_unchanged=True, fp8_modules_checked=active["fp8_modules_checked"])
            assert row["fp8_modules_checked"] == 400
            active.clear()
            if label == "padded_microbatch4":
                batch_rows = rows
                assert abs(rows[0]["observer_logit_difference"] - rows[1]["observer_logit_difference"]) >= 1e-6
            else:
                singleton_rows.extend(rows)
            row["cleanup"] = adapter.cleanup()
            flush()
        differences = [abs(batch_rows[i]["observer_score"] - singleton_rows[i]["observer_score"]) for i in range(2)]
        tolerance = observer.get("self_test_score_atol")
        if tolerance is not None:
            assert max(differences) <= float(tolerance)
        report["batch_singleton_comparison"] = {"max_abs_score_difference": max(differences),
            "configured_score_tolerance": tolerance,
            "same_flags": all(batch_rows[i]["observer_flag"] == singleton_rows[i]["observer_flag"] for i in range(2)),
            "exact_full_model_equivalence_claimed": False}
        # Check the actual frozen two-control startup scoring population as well.
        active.update(row={}, fp8_modules_checked=0)
        startup = adapter.score(worker.SELF_TEST_TEXTS)
        assert active["fp8_modules_checked"] == 400
        assert abs(startup[0]["observer_logit_difference"] - startup[1]["observer_logit_difference"]) >= 1e-6
        startup_differences = [abs(startup[i]["observer_score"] - singleton_rows[i]["observer_score"]) for i in range(2)]
        if tolerance is not None:
            assert max(startup_differences) <= float(tolerance)
        report["frozen_startup_control_check"] = {"passed": True, "scalar_scores": startup,
            "max_abs_score_difference": max(startup_differences), "cleanup": adapter.cleanup()}
        active.clear()
        report["memory_at_completion"] = adapter.memory()
        report["model_validation_passed"] = True
        report["validation_passed"] = True
        report["status"] = "passed"
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
        # Exception text/tracebacks may contain model input; never serialize them.
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
