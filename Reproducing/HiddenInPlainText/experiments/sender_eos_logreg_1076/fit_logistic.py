#!/usr/bin/env python3
"""Fit a prespecified linear probe without using held-out data for fitting.

Inputs are frozen answer-only EOS embeddings. The fitted objective is
mean(binary cross-entropy) + ||w||^2 / (2 * n_train * C), with C=1 and
an unpenalized intercept. Reported losses exclude the penalty and use nats.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit


N_PER_SPLIT = 1000
FEATURES = 2304
C = 1.0
OPTIMIZER_OPTIONS = {"maxiter": 5000, "gtol": 1e-8, "ftol": 1e-12}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_verified_inputs(embeddings_dir, split_dir):
    """Check provenance and return arrays, preserving the saved split order."""
    completion_path = embeddings_dir / "EMBEDDING_COMPLETE.json"
    completion = read_json(completion_path)
    require(completion.get("status", "complete") == "complete", "Embedding extraction is not complete")
    declared = completion.get("files")
    require(isinstance(declared, dict) and bool(declared), "Embedding completion lacks file hashes")
    for name in ("training_embeddings.npz", "test_embeddings.npz"):
        require(name in declared, f"Missing completion hash: {name}")
    embedding_hashes = {}
    for name, expected in declared.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe completion path: {name}")
        path = embeddings_dir / relative
        require(path.resolve().is_relative_to(embeddings_dir.resolve()), f"Completion file escapes directory: {name}")
        require(isinstance(expected, str) and len(expected) == 64, f"Invalid SHA-256 for {name}")
        actual = sha256(path)
        require(actual == expected, f"Embedding completion hash mismatch: {name}")
        embedding_hashes[name] = actual

    manifest_path = split_dir / "split_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("source_checkpoint_update") == 1076, "Split does not come from checkpoint 1076")
    source_path = Path(manifest["source_file"])
    source_audit_path = Path(manifest["source_audit_file"])
    require(sha256(source_path) == manifest["source_sha256"], "Source answer file hash mismatch")
    require(sha256(source_audit_path) == manifest["source_audit_sha256"], "Source audit hash mismatch")
    source = read_json(source_path)
    source_audit = read_json(source_audit_path)
    require(source_audit.get("status") == "passed" and not source_audit.get("issues"), "Source evaluation audit did not pass")
    require(source_audit["public_file_sha256"]["UNIVERSITY_RESULTS.json"] == manifest["source_sha256"], "Source audit answer hash mismatch")
    require(source_audit["completed_update"] == 1076, "Wrong source audit checkpoint")
    require(source_audit["source_checkpoint"] == manifest["source_checkpoint"], "Source checkpoint provenance mismatch")
    require(source_audit["config_digest"] == manifest["config_digest"], "Source configuration mismatch")
    require(source["config_digest"] == source_audit["evaluation_config_digest"], "Source evaluation configuration mismatch")
    source_rows = source["rows"]
    source_by_id = {row["example_id"]: row for row in source_rows}
    require(len(source_rows) == len(source_by_id) == 2 * N_PER_SPLIT, "Source must have 2000 unique rows")
    require(set(source_by_id) == set(range(2 * N_PER_SPLIT)), "Unexpected source IDs")
    require(manifest["hidden_label_mapping"] == {
        "0": {"university": "The Bad University", "class": "FAIL"},
        "1": {"university": "The Good University", "class": "PASS"},
    }, "Unexpected hidden-label mapping")
    shuffled = sorted(source_by_id)
    random.Random(manifest["randomization"]["seed"]).shuffle(shuffled)

    arrays, answers, labels, ids, eos_positions = {}, {}, {}, {}, {}
    split_hashes = {}
    for split, expected_ids in (("training", shuffled[:N_PER_SPLIT]), ("test", shuffled[N_PER_SPLIT:])):
        for suffix in ("answers", "hidden_labels"):
            name = f"{split}_{suffix}.jsonl"
            path = split_dir / name
            actual_hash = sha256(path)
            require(actual_hash == manifest["files"][name]["sha256"], f"Split file hash mismatch: {name}")
            split_hashes[name] = actual_hash
            rows = read_jsonl(path)
            require(len(rows) == manifest["files"][name]["rows"] == N_PER_SPLIT, f"Wrong row count: {name}")
            require(all(type(row.get("example_id")) is int for row in rows), f"Noninteger example ID: {name}")
            require([row["example_id"] for row in rows] == expected_ids, f"Shuffle/order mismatch: {name}")
            if suffix == "answers":
                require(all(set(row) == {"example_id", "answer"} for row in rows), f"Answer schema contains extra data: {name}")
                require(all(isinstance(row["answer"], str) and row["answer"] == source_by_id[row["example_id"]]["answer"] for row in rows), f"Answer text changed: {name}")
                answers[split] = [row["answer"] for row in rows]
            else:
                require(all(set(row) == {"example_id", "hidden_label"} for row in rows), f"Unexpected label schema: {name}")
                require(all(type(row["hidden_label"]) is int and row["hidden_label"] in (0, 1) and row["hidden_label"] == source_by_id[row["example_id"]]["true_label"] for row in rows), f"Incorrect joined label: {name}")
                labels[split] = np.asarray([row["hidden_label"] for row in rows], dtype=np.int64)
                actual_counts = {str(k): int((labels[split] == k).sum()) for k in (0, 1)}
                require(actual_counts == manifest["hidden_label_counts"][split], f"Label-count mismatch: {name}")
        with np.load(embeddings_dir / f"{split}_embeddings.npz", allow_pickle=False) as archive:
            require(set(archive.files) == {"embeddings", "example_ids", "eos_positions"}, f"Unexpected embedding NPZ schema: {split}")
            x = archive["embeddings"]
            example_ids = archive["example_ids"]
            positions = archive["eos_positions"]
            require(x.dtype == np.float32 and x.shape == (N_PER_SPLIT, FEATURES), f"Wrong embedding dtype/shape: {split}")
            require(np.isfinite(x).all(), f"Nonfinite embeddings: {split}")
            require(example_ids.dtype == np.int64 and example_ids.shape == (N_PER_SPLIT,), f"Wrong ID dtype/shape: {split}")
            require(np.array_equal(example_ids, expected_ids), f"Embedding IDs do not align: {split}")
            require(positions.dtype == np.int64 and positions.shape == (N_PER_SPLIT,) and (positions >= 1).all(), f"Invalid explicit EOS positions: {split}")
            arrays[split] = x.astype(np.float64)
            ids[split] = example_ids.copy()
            eos_positions[split] = positions.copy()
    metadata = completion.get("metadata", {})
    require(metadata.get("completed_update") == 1076, "Embedding checkpoint update mismatch")
    require(metadata.get("checkpoint") == manifest["source_checkpoint"]["path"], "Embedding checkpoint path mismatch")
    require(metadata.get("checkpoint_state_sha256") == manifest["source_checkpoint"]["state_sha256_from_manifest"], "Embedding checkpoint state hash mismatch")
    require(metadata.get("checkpoint_complete_sha256") == manifest["source_checkpoint"]["complete_sha256"], "Embedding checkpoint marker hash mismatch")
    require(metadata.get("split_manifest_sha256") == sha256(manifest_path), "Embedding extraction used a different split manifest")
    require(metadata.get("answer_file_sha256") == {name: split_hashes[name] for name in ("training_answers.jsonl", "test_answers.jsonl")}, "Embedding extraction answer hashes mismatch")
    require((metadata.get("total_blocks"), metadata.get("block_number_one_based"), metadata.get("block_index_zero_based"), metadata.get("features")) == (26, 17, 16, FEATURES), "Unexpected frozen probe layer or feature count")
    require((metadata.get("bos_token_id"), metadata.get("eos_token_id"), metadata.get("pad_token_id")) == (2, 1, 0), "Unexpected BOS/EOS/padding convention")
    required_checks = ("hook_matches_post_block_hidden_state_exactly", "repeated_forward_exact",
                       "exact_checkpoint_parameter_copy", "all_parameters_frozen", "answer_only_inputs",
                       "explicit_eos_positions_and_attention_masks_verified",
                       "trained_parameter_hash_unchanged_after_inference")
    require(all(metadata.get("validation", {}).get(name) is True for name in required_checks), "Embedding extraction validation missing or failed")
    require(set(ids["training"]).isdisjoint(ids["test"]), "Train/test ID leakage")
    require(set(ids["training"]) | set(ids["test"]) == set(source_by_id), "Split is not exhaustive")
    train_texts = set(answers["training"])
    secondary_mask = np.asarray([answer not in train_texts for answer in answers["test"]], dtype=bool)
    # This sensitivity cohort is declared before fitting and never replaces the primary test set.
    require(int(secondary_mask.sum()) == 998, "Unexpected shared-text sensitivity cohort; review provenance")
    evidence = {
        "embedding_completion_sha256": sha256(completion_path),
        "embedding_file_sha256": embedding_hashes,
        "embedding_completion": completion,
        "split_manifest_sha256": sha256(manifest_path),
        "split_file_sha256": split_hashes,
        "source_sha256": manifest["source_sha256"],
        "source_audit_sha256": manifest["source_audit_sha256"],
        "source_checkpoint": manifest["source_checkpoint"],
        "split_seed": manifest["randomization"]["seed"],
        "hidden_label_mapping": manifest["hidden_label_mapping"],
        "exact_answer_whitespace_preserved": True,
        "shuffle_replayed_exactly": True,
    }
    return arrays, labels, ids, eos_positions, secondary_mask, evidence


def mean_bce(logits, labels):
    """Unpenalized mean BCE in nats, stable even for very large logits."""
    return float(np.logaddexp(0.0, (1.0 - 2.0 * labels) * logits).mean())


def objective_and_gradient(theta, x, labels, c=C):
    w, intercept = theta[:-1], theta[-1]
    logits = x @ w + intercept
    loss = mean_bce(logits, labels)
    penalty = float(w @ w) / (2.0 * len(labels) * c)
    residual = expit(logits) - labels
    gradient = np.empty_like(theta)
    gradient[:-1] = x.T @ residual / len(labels) + w / (len(labels) * c)
    gradient[-1] = residual.mean()
    return loss + penalty, gradient


def check_gradient():
    """Fixed, label-independent central-difference check of the full objective."""
    x = np.asarray([[1.1, -0.2, 0.3], [-0.5, 1.4, 0.0], [0.7, 0.9, -1.2], [0.0, -0.8, 1.5]], dtype=np.float64)
    y = np.asarray([0, 1, 1, 0], dtype=np.float64)
    theta = np.asarray([0.2, -0.4, 0.3, 0.1], dtype=np.float64)
    _, analytic = objective_and_gradient(theta, x, y)
    numerical = np.empty_like(theta)
    step = 1e-6
    for i in range(len(theta)):
        plus, minus = theta.copy(), theta.copy()
        plus[i] += step
        minus[i] -= step
        numerical[i] = (objective_and_gradient(plus, x, y)[0] - objective_and_gradient(minus, x, y)[0]) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-8)
    return {"passed": True, "central_difference_step": step, "max_absolute_error": float(np.max(np.abs(analytic - numerical)))}


def score(logits, labels):
    predictions = (logits >= 0.0).astype(np.int64)
    confusion = [[int(((labels == truth) & (predictions == prediction)).sum()) for prediction in (0, 1)] for truth in (0, 1)]
    return {
        "n": int(len(labels)),
        "mean_unpenalized_bce_nats": mean_bce(logits, labels),
        "accuracy": float((predictions == labels).mean()),
        "correct": int((predictions == labels).sum()),
        "confusion_matrix": {"rows": "true label", "columns": "predicted label", "label_order": [0, 1], "counts": confusion},
        "true_label_counts": {str(k): int((labels == k).sum()) for k in (0, 1)},
        "prediction_threshold": "logit >= 0 predicts label 1 (equivalently probability >= 0.5)",
    }


def write_trace(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run(args):
    embeddings_dir, split_dir, output = (Path(getattr(args, name)).resolve() for name in ("embeddings", "split_dir", "output"))
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Output directory must be new or empty")
    arrays, labels, ids, eos_positions, secondary_mask, evidence = load_verified_inputs(embeddings_dir, split_dir)
    gradient_check = check_gradient()

    # No test features or labels enter scaler estimation, optimization, or stopping.
    x_train, y_train = arrays["training"], labels["training"]
    require(set(y_train.tolist()) == {0, 1}, "Training set must contain both classes")
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=0)
    scale = np.where(std == 0.0, 1.0, std)
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), "Nonfinite training scaler")
    standardized_train = (x_train - mean) / scale
    prior = float(y_train.mean())
    prior_logit = float(np.log(prior) - np.log1p(-prior))
    initial = np.zeros(FEATURES + 1, dtype=np.float64)
    initial[-1] = prior_logit
    trace = []

    def record(theta):
        objective, gradient = objective_and_gradient(theta, standardized_train, y_train)
        loss = mean_bce(standardized_train @ theta[:-1] + theta[-1], y_train)
        trace.append({"iteration": len(trace), "train_mean_unpenalized_bce_nats": loss,
                      "l2_penalty": float(theta[:-1] @ theta[:-1]) / (2.0 * N_PER_SPLIT * C),
                      "train_penalized_objective": objective, "gradient_infinity_norm": float(np.max(np.abs(gradient)))})

    output.mkdir(parents=True, exist_ok=True)
    record(initial)
    result = minimize(objective_and_gradient, initial, args=(standardized_train, y_train),
                      method="L-BFGS-B", jac=True, callback=record, options=OPTIMIZER_OPTIONS)
    write_trace(output / "training_loss_trace.csv", trace)
    optimization = {"method": "L-BFGS-B", "success": bool(result.success), "status": int(result.status),
                    "message": str(result.message), "iterations": int(result.nit), "function_evaluations": int(result.nfev),
                    "options": OPTIMIZER_OPTIONS, "gradient_infinity_norm": float(np.max(np.abs(result.jac)))}
    if not result.success:
        write_json(output / "FAILURE.json", {"reason": "Logistic fit did not converge", "optimization": optimization,
                                             "test_metrics_computed": False})
        raise RuntimeError(f"Logistic fit did not converge: {result.message}")
    require(np.isfinite(result.x).all() and np.isfinite(result.fun), "Nonfinite fitted model")
    theta = result.x
    final_objective, _ = objective_and_gradient(theta, standardized_train, y_train)
    require(np.isclose(final_objective, result.fun, rtol=1e-12, atol=1e-12), "Optimizer objective verification failed")
    logits = {split: ((x - mean) / scale) @ theta[:-1] + theta[-1] for split, x in arrays.items()}
    require(all(np.isfinite(z).all() for z in logits.values()), "Nonfinite fitted logits")
    model_path = output / "logistic_model.npz"
    np.savez(model_path, coefficient=theta[:-1], intercept=np.asarray(theta[-1]),
             feature_mean=mean, feature_std_population=std, feature_scale=scale,
             zero_std_mask=(std == 0.0), classes=np.asarray([0, 1], dtype=np.int64), C=np.asarray(C))

    # Independently reconstruct predictions in raw-feature coordinates from the saved model.
    saved_check = {}
    with np.load(model_path, allow_pickle=False) as saved:
        raw_w = saved["coefficient"] / saved["feature_scale"]
        raw_intercept = float(saved["intercept"]) - float(saved["feature_mean"] @ raw_w)
        for split, x in arrays.items():
            recalculated = np.einsum("ij,j->i", x, raw_w) + raw_intercept
            np.testing.assert_allclose(recalculated, logits[split], rtol=1e-10, atol=1e-9)
            np.testing.assert_allclose(expit(recalculated), expit(logits[split]), rtol=1e-10, atol=1e-10)
            require(np.array_equal(recalculated >= 0, logits[split] >= 0), f"Saved-model classifications differ: {split}")
            saved_check[split] = {"passed": True, "maximum_logit_absolute_difference": float(np.max(np.abs(recalculated - logits[split]))),
                                  "maximum_probability_absolute_difference": float(np.max(np.abs(expit(recalculated) - expit(logits[split]))))}

    metrics = {
        "schema_version": 1, "status": "complete", "method": "Frozen sender checkpoint-1076 EOS linear probe",
        "loss_definition": "Unpenalized mean binary cross-entropy in natural-log units (nats)",
        "training_objective_definition": "mean_BCE + ||w||^2 / (2 * n_training * C); intercept unpenalized",
        "C": C, "optimization": optimization,
        "training_penalized_objective": final_objective,
        "training_l2_penalty": float(theta[:-1] @ theta[:-1]) / (2.0 * N_PER_SPLIT * C),
        "training": score(logits["training"], labels["training"]),
        "test": score(logits["test"], labels["test"]),
        "constant_training_prior_baseline": {
            "probability_label_1": prior, "fitted_on_training_only": True,
            **{split: score(np.full(N_PER_SPLIT, prior_logit), y) for split, y in labels.items()},
        },
        "secondary_test_excluding_exact_text_shared_with_training": {
            **score(logits["test"][secondary_mask], labels["test"][secondary_mask]),
            "excluded_example_ids": ids["test"][~secondary_mask].tolist(),
            "predeclared": True, "used_for_fitting_or_model_selection": False,
            "note": "Sensitivity analysis only; primary held-out result includes all 1000 assigned test examples.",
        },
        "scaler": {"fitted_on": "training only", "standard_deviation_ddof": 0,
                   "zero_standard_deviation_features": int((std == 0.0).sum()), "zero_std_replacement_scale": 1.0},
        "validation": {"analytic_gradient_finite_difference": gradient_check,
                       "saved_model_raw_coordinate_recomputation": saved_check,
                       "test_data_used_for_fitting_or_stopping_or_hyperparameter_selection": False},
    }
    predictions_path = output / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for i, example_id in enumerate(ids["test"]):
            row = {"example_id": int(example_id), "true_label": int(labels["test"][i]),
                   "logit_label_1": float(logits["test"][i]), "probability_label_1": float(expit(logits["test"][i])),
                   "predicted_label": int(logits["test"][i] >= 0),
                   "included_in_no_shared_text_sensitivity": bool(secondary_mask[i])}
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    saved_rows = read_jsonl(predictions_path)
    require([r["example_id"] for r in saved_rows] == ids["test"].tolist(), "Saved prediction ID alignment failed")
    np.testing.assert_allclose([r["probability_label_1"] for r in saved_rows], expit(logits["test"]), rtol=0, atol=0)
    require(score(np.asarray([r["logit_label_1"] for r in saved_rows]), labels["test"]) == metrics["test"], "Saved prediction metrics recomputation failed")
    write_json(output / "metrics.json", metrics)

    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(8.4, 5.0), layout="constrained")
    ax.plot([r["iteration"] for r in trace], [r["train_mean_unpenalized_bce_nats"] for r in trace], label="Training mean BCE (unpenalized)", linewidth=2)
    ax.plot([r["iteration"] for r in trace], [r["train_penalized_objective"] for r in trace], label="Training objective (BCE + L2 penalty)", linestyle="--", linewidth=1.6)
    ax.set(xlabel="L-BFGS-B iteration (0 = initialization)", ylabel="Loss / objective (natural-log units)",
           title="Checkpoint 1076 EOS logistic probe · training convergence")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output / "training_loss.png", dpi=180)
    plt.close(fig)
    provenance = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "embeddings_directory": str(embeddings_dir), "split_directory": str(split_dir),
        "script_sha256": sha256(Path(__file__)), "inputs": evidence,
        "packages": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "precision": "float64 preprocessing, optimization, and scoring; input embeddings float32",
        "embedding_shape_per_split": [N_PER_SPLIT, FEATURES],
        "eos_position_ranges": {split: {"min": int(pos.min()), "max": int(pos.max())} for split, pos in eos_positions.items()},
        "model_input_access": "Only frozen embedding matrices enter the classifier; answer text is used solely for provenance checks and predetermined duplicate sensitivity membership.",
        "model_selection": "No validation/test tuning; C, solver, tolerances, feature extraction, and duplicate sensitivity are fixed before fitting.",
        "scope": "Linear decodability of university labels from frozen sender EOS representations; not the static observer flag rate.",
    }
    write_json(output / "provenance.json", provenance)
    output_names = ["training_loss_trace.csv", "logistic_model.npz", "test_predictions.jsonl", "metrics.json", "training_loss.png", "provenance.json"]
    write_json(output / "FIT_COMPLETE.json", {"status": "complete", "source_checkpoint_update": 1076,
        "embedding_complete_sha256": evidence["embedding_completion_sha256"],
        "split_manifest_sha256": evidence["split_manifest_sha256"],
        "files": {name: sha256(output / name) for name in output_names}})
    print(json.dumps({"status": "complete", "output": str(output), "training": metrics["training"], "test": metrics["test"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, help="Directory containing NPZs and EMBEDDING_COMPLETE.json")
    parser.add_argument("--split-dir", required=True, help="Verified checkpoint-1076 1000/1000 split directory")
    parser.add_argument("--output", required=True, help="New or empty output directory")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
