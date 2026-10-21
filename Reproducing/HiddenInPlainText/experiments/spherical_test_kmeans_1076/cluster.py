"""Label-free centered spherical k=2 clustering on frozen EOS test embeddings."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def unit_rows(x):
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(norms).all() or (norms <= 1e-12).any():
        raise ValueError('Cannot normalize a nonfinite or zero-length direction')
    return x / norms[:, None], norms


def fit_start(u, seed, max_iter):
    rng = np.random.default_rng(int(seed))
    first = int(rng.integers(len(u)))
    # Squared chord distance on unit vectors, equivalently 2 * (1 - cosine).
    weights = np.maximum(0.0, 2.0 * (1.0 - u @ u[first]))
    weights[first] = 0.0
    if weights.sum() <= 1e-12:
        raise ValueError('No distinct second initialization direction')
    second = int(rng.choice(len(u), p=weights / weights.sum()))
    centers = u[[first, second]].copy()
    previous, trace = None, []
    for iteration in range(1, max_iter + 1):
        labels = (u @ centers.T).argmax(axis=1)
        counts = np.bincount(labels, minlength=2)
        if (counts == 0).any():
            raise ValueError('Empty cluster; no implicit reseeding is permitted')
        centers, _ = unit_rows(np.stack([u[labels == k].mean(axis=0) for k in range(2)]))
        cosine_objective = float((1.0 - np.einsum('ij,ij->i', u, centers[labels])).sum())
        assert np.isfinite(cosine_objective)
        if trace:
            assert cosine_objective <= trace[-1] + 1e-10 * max(1.0, abs(trace[-1]))
        trace.append(cosine_objective)
        if previous is not None and np.array_equal(labels, previous):
            break
        previous = labels.copy()
    else:
        raise RuntimeError('An initialization did not reach stable assignments')
    assert np.array_equal((u @ centers.T).argmax(axis=1), labels)
    np.testing.assert_allclose(np.linalg.norm(centers, axis=1), 1.0, rtol=1e-12, atol=1e-12)
    return centers, labels, trace, (first, second), iteration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['script_sha256'] == sha(__file__)
    assert plan['k'] == 2 and plan['n_init'] == 50 and plan['master_seed'] == 42
    assert plan['max_iter'] == 1000 and plan['pca'] is False
    assert plan['preprocessing'] == 'test_mean_center_then_row_l2_normalize'
    source = Path(plan['input_embeddings'])
    assert sha(source) == plan['input_embeddings_sha256']
    completion_path = Path(plan['embedding_completion'])
    assert sha(completion_path) == plan['embedding_completion_sha256']
    completion = json.loads(completion_path.read_text())
    assert completion['status'] == 'complete'
    assert completion['files']['test_embeddings.npz'] == sha(source)
    for key, expected in plan['expected_embedding_metadata'].items():
        assert completion['metadata'][key] == expected, key
    assert completion['metadata']['validation']['all_parameters_frozen'] is True
    assert completion['metadata']['validation']['answer_only_inputs'] is True
    output = Path(plan['cluster_output'])
    output.mkdir(exist_ok=False)
    with np.load(source, allow_pickle=False) as f:
        assert set(f.files) == {'embeddings', 'example_ids', 'eos_positions'}
        x, ids = f['embeddings'].astype(np.float64), f['example_ids'].copy()
    assert x.shape == (1000, plan['features']) and np.isfinite(x).all()
    assert ids.dtype == np.int64 and len(set(ids)) == 1000
    mean = x.mean(axis=0)
    u, row_norms = unit_rows(x - mean)
    seeds = np.random.SeedSequence(plan['master_seed']).generate_state(plan['n_init'], dtype=np.uint32)
    starts, centers_all, labels_all, traces = [], [], [], []
    for start, seed in enumerate(seeds):
        centers, labels, trace, initial, iteration = fit_start(u, seed, plan['max_iter'])
        canonical = labels if labels[0] == 0 else 1 - labels
        starts.append(dict(start=start, seed=int(seed), initial_example_ids=ids[list(initial)].tolist(),
                           iterations=iteration, converged=True, cosine_objective=trace[-1],
                           cluster_sizes=np.bincount(labels, minlength=2).tolist(),
                           partition_sha256=hashlib.sha256(canonical.astype(np.int64).tobytes()).hexdigest()))
        centers_all.append(centers.copy()); labels_all.append(labels.copy()); traces.append(trace)
    selected = min(range(len(starts)), key=lambda i: (starts[i]['cosine_objective'], i))
    centers, labels = centers_all[selected], labels_all[selected]
    np.savez(output / 'clustering_model.npz', centers=centers, feature_mean=mean,
             centered_row_norms=row_norms)
    np.savez_compressed(output / 'restart_solutions.npz', centers=np.asarray(centers_all),
                        assignments=np.asarray(labels_all, dtype=np.int64), seeds=seeds)
    with (output / 'cluster_assignments.jsonl').open('w') as f:
        for example_id, cluster in zip(ids, labels):
            f.write(json.dumps({'example_id': int(example_id), 'cluster': int(cluster)}) + '\n')
    write_json(output / 'starts.json', {'starts': starts, 'cosine_objective_traces': traces})
    summary = dict(model_name=plan['model_name'], status='complete', n=1000, features=plan['features'],
                   k=2, n_init=50, master_seed=42, selected_start=selected,
                   selection='Minimum sum of (1 - cosine to assigned unit centroid); ties by start index',
                   cosine_objective=starts[selected]['cosine_objective'],
                   mean_cosine_similarity=1.0 - starts[selected]['cosine_objective'] / len(u),
                   cluster_sizes=np.bincount(labels, minlength=2).tolist(),
                   converged_starts=sum(s['converged'] for s in starts),
                   unique_partitions=len({s['partition_sha256'] for s in starts}),
                   starts_with_selected_partition=sum(s['partition_sha256'] == starts[selected]['partition_sha256'] for s in starts),
                   objective_range=[min(s['cosine_objective'] for s in starts), max(s['cosine_objective'] for s in starts)],
                   preprocessing='Mean-center raw test embeddings using only the 1000 test rows, then normalize each row to unit length',
                   feature_standardization=False, labels_loaded=False, training_embeddings_loaded=False,
                   pca_used=False, script_sha256=sha(__file__), plan_sha256=sha(args.plan),
                   input_embeddings_sha256=sha(source), embedding_completion_sha256=sha(completion_path),
                   packages={'python': platform.python_version(), 'numpy': np.__version__},
                   stopping='Exactly unchanged assignments; maximum 1000 iterations per initialization',
                   centroid_update='Normalize the mean of assigned unit input vectors after every update',
                   assignment='Maximum dot product with unit centroid; exact ties go to cluster 0',
                   initialization='Uniform first point; squared chord distance weighted second point',
                   interpretation='Exploratory reuse of an already examined test cohort; labels do not enter this fit')
    write_json(output / 'clustering_summary.json', summary)
    names=['clustering_model.npz','restart_solutions.npz','cluster_assignments.jsonl','starts.json','clustering_summary.json']
    write_json(output / 'CLUSTER_COMPLETE.json', dict(status='complete', frozen_at=datetime.now(timezone.utc).isoformat(),
               labels_loaded=False, plan_sha256=sha(args.plan), files={n:sha(output/n) for n in names}))
    print(json.dumps({key: summary[key] for key in ['model_name','selected_start','cosine_objective','cluster_sizes',
                                                 'converged_starts','unique_partitions','starts_with_selected_partition']}, indent=2))


if __name__ == '__main__':
    main()
