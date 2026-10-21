#!/usr/bin/env python3
"""Evaluate frozen two-cluster assignments after verifying their completion hashes.

Cluster assignments are fixed before labels are opened. The best binary label
permutation is a post-hoc interpretation of clusters, not a fitted predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXPECTED_N = 1000
EXPECTED_LABELS_SHA256 = "c1c9d35c11f8e5cd5ffe87175a4c862bd6e24df8ac8941685a5da99a4faf7c09"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def read_rows(payload):
    return [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def adjusted_rand_index(contingency):
    """Pair-count adjusted Rand index, using exact integer arithmetic first."""
    table = np.asarray(contingency)
    require(table.ndim == 2 and np.issubdtype(table.dtype, np.integer), "ARI needs an integer contingency table")
    require(bool((table >= 0).all()), "ARI counts must be nonnegative")
    choose2 = lambda n: int(n) * (int(n) - 1) // 2
    total_pairs = choose2(table.sum())
    if total_pairs == 0:
        return 1.0
    same_both = sum(choose2(n) for n in table.flat)
    same_cluster = sum(choose2(n) for n in table.sum(axis=1))
    same_label = sum(choose2(n) for n in table.sum(axis=0))
    numerator = 2 * (same_both * total_pairs - same_cluster * same_label)
    denominator = (same_cluster + same_label) * total_pairs - 2 * same_cluster * same_label
    # Both constant partitions, or both singleton partitions, are identical.
    if denominator == 0:
        return 1.0
    return float(numerator / denominator)


def sanity_checks():
    tests = {
        "identical_partition": (np.array([[3, 0], [0, 3]], dtype=np.int64), 1.0),
        "complemented_binary_partition": (np.array([[0, 3], [3, 0]], dtype=np.int64), 1.0),
        "crossed_balanced_four_point_partition": (np.array([[1, 1], [1, 1]], dtype=np.int64), -0.5),
        "constant_label_nonconstant_clusters": (np.array([[3, 0], [1, 0]], dtype=np.int64), 0.0),
        "both_constant_partitions": (np.array([[4, 0], [0, 0]], dtype=np.int64), 1.0),
    }
    result = {}
    for name, (table, expected) in tests.items():
        actual = adjusted_rand_index(table)
        require(abs(actual - expected) < 1e-12, f"ARI sanity check failed: {name}")
        result[name] = {"actual": actual, "expected": expected, "passed": True}
    return result


def freeze_verified_assignments(clusters):
    """Read only cluster artifacts here; this function has no label-file argument."""
    completion_path = clusters / "CLUSTER_COMPLETE.json"
    completion_bytes = completion_path.read_bytes()
    completion = json.loads(completion_bytes)
    require(completion.get("status") == "complete", "Clustering is not complete")
    declared = completion.get("files")
    require(isinstance(declared, dict), "Clustering completion lacks file hashes")
    expected_files = {"cluster_assignments.jsonl", "clustering_model.npz", "clustering_summary.json", "starts.json"}
    require(expected_files.issubset(declared), "Clustering completion is missing required artifacts")
    verified_bytes = {}
    for name, expected_hash in declared.items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe artifact path: {name}")
        path = clusters / relative
        require(path.resolve().is_relative_to(clusters.resolve()), f"Artifact escapes cluster directory: {name}")
        require(isinstance(expected_hash, str) and len(expected_hash) == 64, f"Invalid artifact hash: {name}")
        payload = path.read_bytes()
        require(digest_bytes(payload) == expected_hash, f"Frozen clustering hash mismatch: {name}")
        verified_bytes[name] = payload

    rows = read_rows(verified_bytes["cluster_assignments.jsonl"])
    require(len(rows) == EXPECTED_N, "Expected exactly 1000 frozen test assignments")
    require(all(set(row) == {"example_id", "cluster"} for row in rows), "Assignments contain unexpected fields")
    require(all(type(row["example_id"]) is int and type(row["cluster"]) is int and row["cluster"] in (0, 1) for row in rows), "Invalid assignment ID or cluster")
    ids = np.asarray([row["example_id"] for row in rows], dtype=np.int64)
    assignments = np.asarray([row["cluster"] for row in rows], dtype=np.int64)
    require(len(set(ids.tolist())) == EXPECTED_N, "Duplicate frozen assignment IDs")
    require(set(assignments.tolist()) == {0, 1}, "Both frozen clusters must be nonempty")
    # Copies, marked read-only, are the sole assignments used after labels open.
    ids.setflags(write=False)
    assignments.setflags(write=False)
    frozen = {
        "CLUSTER_COMPLETE.json": digest_bytes(completion_bytes),
        "files": {name: digest_bytes(payload) for name, payload in verified_bytes.items()},
        "verified_before_labels_at": datetime.now(timezone.utc).isoformat(),
    }
    return ids, assignments, completion, frozen


def verify_clustering_unchanged(clusters, frozen):
    require(digest_bytes((clusters / "CLUSTER_COMPLETE.json").read_bytes()) == frozen["CLUSTER_COMPLETE.json"], "Cluster completion changed during evaluation")
    for name, expected_hash in frozen["files"].items():
        require(digest_bytes((clusters / name).read_bytes()) == expected_hash, f"Frozen cluster artifact changed: {name}")


def run(args):
    clusters, labels_path, output = (Path(getattr(args, name)).resolve() for name in ("clusters", "labels", "output"))
    require(output != clusters and not output.is_relative_to(clusters), "Evaluation output must be separate from frozen clustering artifacts")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Evaluation output must be new or empty")
    checks = sanity_checks()

    # Crucial ordering: all cluster hashes and assignments are verified and frozen
    # before the first attempt to open the hidden-label file.
    ids, assignments, completion, frozen = freeze_verified_assignments(clusters)
    labels_opened_at = datetime.now(timezone.utc).isoformat()
    labels_bytes = labels_path.read_bytes()
    require(digest_bytes(labels_bytes) == EXPECTED_LABELS_SHA256, "Hidden labels do not match the frozen test-label file")
    label_rows = read_rows(labels_bytes)
    require(len(label_rows) == EXPECTED_N, "Expected exactly 1000 hidden labels")
    require(all(set(row) == {"example_id", "hidden_label"} for row in label_rows), "Unexpected hidden-label schema")
    require(all(type(row["example_id"]) is int and type(row["hidden_label"]) is int and row["hidden_label"] in (0, 1) for row in label_rows), "Invalid hidden label or ID")
    label_by_id = {row["example_id"]: row["hidden_label"] for row in label_rows}
    require(len(label_by_id) == EXPECTED_N, "Duplicate hidden-label IDs")
    require(set(label_by_id) == set(ids.tolist()), "Frozen assignment and hidden-label IDs do not match exactly")
    labels = np.asarray([label_by_id[int(example_id)] for example_id in ids], dtype=np.int64)
    counts = np.zeros((2, 2), dtype=np.int64)
    np.add.at(counts, (assignments, labels), 1)
    sizes = counts.sum(axis=1)
    identity_correct = int(counts[0, 0] + counts[1, 1])
    swapped_correct = int(counts[0, 1] + counts[1, 0])
    # These are only the two possible naming permutations, with a fixed tie rule.
    identity_best = identity_correct >= swapped_correct
    mapping = {"0": 0 if identity_best else 1, "1": 1 if identity_best else 0}
    best_correct = max(identity_correct, swapped_correct)
    ari = adjusted_rand_index(counts)
    require(-1.0 <= ari <= 1.0, "ARI is outside its valid range")
    verify_clustering_unchanged(clusters, frozen)

    result = {
        "schema_version": 1, "status": "complete", "n": EXPECTED_N,
        "method": "Post-hoc label evaluation of frozen k=2 Qwen test-embedding clusters",
        "label_mapping": {"0": "FAIL (Bad University)", "1": "PASS (Good University)"},
        "cluster_sizes": {str(k): int(sizes[k]) for k in (0, 1)},
        "cluster_by_hidden_label": {"rows": "frozen cluster", "row_order": [0, 1],
            "columns": "hidden label", "column_order": ["FAIL (0)", "PASS (1)"], "counts": counts.tolist()},
        "hidden_label_counts": {"FAIL": int(counts[:, 0].sum()), "PASS": int(counts[:, 1].sum())},
        "best_label_mapping": {"cluster_to_hidden_label": mapping,
            "cluster_to_label_name": {cluster: "PASS" if label else "FAIL" for cluster, label in mapping.items()},
            "correct": best_correct, "accuracy": best_correct / EXPECTED_N,
            "identity_mapping_correct": identity_correct, "swapped_mapping_correct": swapped_correct,
            "tie_rule": "Prefer cluster 0 -> FAIL and cluster 1 -> PASS on an exact tie",
            "interpretation": "Best of the two cluster-name permutations after labels are revealed; no assignments or centroids are refitted."},
        "adjusted_rand_index": ari,
        "ari_definition": "Pair-count adjusted Rand index; label-permutation invariant; denominator-zero identical partitions receive 1.",
        "freeze_protocol": {"cluster_hashes_verified_before_opening_labels": True,
            "assignments_copied_into_read_only_arrays_before_opening_labels": True,
            "labels_opened_at": labels_opened_at, "frozen_evidence": frozen,
            "all_clustering_hashes_unchanged_after_evaluation": True},
        "inputs": {"clusters_directory": str(clusters), "labels_path": str(labels_path),
            "labels_sha256": digest_bytes(labels_bytes), "cluster_completion": completion},
        "sanity_checks": checks,
        "evaluator_sha256": digest_bytes(Path(__file__).read_bytes()),
        "scope": "Labels are used only to evaluate already-frozen clusters and interpret their two arbitrary names; this is not held-out classifier fitting or tuning.",
    }

    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "evaluation.json", result)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 5.4), layout="constrained")
    heatmap = ax.imshow(counts, cmap="Blues", vmin=0, vmax=max(1, int(counts.max())))
    ax.set_xticks([0, 1], ["FAIL (0)", "PASS (1)"])
    ax.set_yticks([0, 1], [f"Cluster {k}\n(n={int(sizes[k])})" for k in (0, 1)])
    ax.set(xlabel="True hidden label, opened after assignments were frozen", ylabel="Frozen cluster",
           title=f"Qwen test-set clustering · 1000 examples\nBest label mapping: {best_correct / EXPECTED_N:.1%} accuracy · ARI {ari:.4f}")
    for cluster in (0, 1):
        for label in (0, 1):
            count = int(counts[cluster, label])
            ax.text(label, cluster, f"{count}\n{count / sizes[cluster]:.1%} of cluster",
                    ha="center", va="center", fontsize=12,
                    color="white" if count > counts.max() * 0.55 else "black")
    fig.colorbar(heatmap, ax=ax, label="Example count", shrink=0.85)
    fig.savefig(output / "cluster_label_counts.png", dpi=180)
    plt.close(fig)
    verify_clustering_unchanged(clusters, frozen)
    write_json(output / "EVALUATION_COMPLETE.json", {
        "status": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
        "cluster_complete_sha256": frozen["CLUSTER_COMPLETE.json"], "labels_sha256": digest_bytes(labels_bytes),
        "files": {name: digest_bytes((output / name).read_bytes()) for name in ("evaluation.json", "cluster_label_counts.png")},
    })
    print(json.dumps({"status": "complete", "output": str(output), "cluster_sizes": result["cluster_sizes"],
        "cluster_by_label_counts": counts.tolist(), "best_mapping_accuracy": best_correct / EXPECTED_N,
        "adjusted_rand_index": ari}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", required=True, help="Frozen clustering directory with CLUSTER_COMPLETE.json")
    parser.add_argument("--labels", required=True, help="Hidden-label JSONL; opened only after cluster verification")
    parser.add_argument("--output", required=True, help="Separate new or empty evaluation output directory")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
