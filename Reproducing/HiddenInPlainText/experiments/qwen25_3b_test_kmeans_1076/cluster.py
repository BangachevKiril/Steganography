"""Label-free k=2 k-means++ on the fixed Qwen test embeddings."""
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    assert plan['script_sha256'] == sha(__file__)
    assert plan['k'] == 2 and plan['n_init'] == 50 and plan['master_seed'] == 42
    source = Path(plan['input_embeddings'])
    assert sha(source) == plan['input_embeddings_sha256']
    completion_path = Path(plan['embedding_completion'])
    assert sha(completion_path) == plan['embedding_completion_sha256']
    completion = json.loads(completion_path.read_text())
    assert completion['status'] == 'complete'
    assert completion['files']['test_embeddings.npz'] == sha(source)
    assert completion['metadata']['embedding_model_kind'] == 'pretrained_qwen25_3b_instruct'
    assert completion['metadata']['source_checkpoint_update'] == 1076
    output = Path(plan['cluster_output'])
    output.mkdir(exist_ok=False)
    with np.load(source, allow_pickle=False) as f:
        assert set(f.files) == {'embeddings', 'example_ids', 'eos_positions'}
        x = f['embeddings'].astype(np.float64)
        ids = f['example_ids'].copy()
    assert x.shape == (1000, 2048) and np.isfinite(x).all()
    assert ids.dtype == np.int64 and len(set(ids)) == 1000
    # Only this unlabeled test matrix is used to fit feature scaling.
    mean, std = x.mean(axis=0), x.std(axis=0, ddof=0)
    scale = np.where(std == 0, 1.0, std)
    z = (x - mean) / scale
    seeds = np.random.SeedSequence(plan['master_seed']).generate_state(plan['n_init'], dtype=np.uint32)
    starts, centers_all, labels_all, traces = [], [], [], []
    for start, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        # Ordinary k-means++: uniform first point, D-squared weighted second.
        first = int(rng.integers(len(z)))
        distances = cdist(z, z[[first]], metric='sqeuclidean')[:, 0]
        assert distances.sum() > 0
        second = int(rng.choice(len(z), p=distances / distances.sum()))
        centers = z[[first, second]].copy()
        previous = None
        objectives = []
        for iteration in range(1, plan['max_iter'] + 1):
            centers, labels = kmeans2(z, centers, iter=1, minit='matrix', missing='raise', check_finite=True)
            counts = np.bincount(labels, minlength=2)
            assert (counts > 0).all()
            inertia = float(np.square(z - centers[labels]).sum())
            assert np.isfinite(inertia)
            if objectives:
                assert inertia <= objectives[-1] + 1e-8 * max(1, objectives[-1])
            objectives.append(inertia)
            if previous is not None and np.array_equal(labels, previous):
                break
            previous = labels.copy()
        else:
            raise RuntimeError('An initialization failed to reach stable assignments')
        # Verify a Lloyd fixed point directly, using a separate distance routine.
        assert np.array_equal(cdist(z, centers, metric='sqeuclidean').argmin(axis=1), labels)
        np.testing.assert_allclose(centers, np.stack([z[labels == k].mean(axis=0) for k in range(2)]), rtol=1e-12, atol=1e-12)
        canonical = labels if labels[0] == 0 else 1 - labels
        starts.append(dict(start=start, seed=int(seed), initial_example_ids=ids[[first, second]].tolist(),
                           iterations=iteration, converged=True, inertia=inertia, cluster_sizes=counts.tolist(),
                           partition_sha256=hashlib.sha256(canonical.astype(np.int64).tobytes()).hexdigest()))
        centers_all.append(centers.copy()); labels_all.append(labels.copy()); traces.append(objectives)
    selected = min(range(len(starts)), key=lambda i: (starts[i]['inertia'], i))
    centers, labels = centers_all[selected], labels_all[selected]
    np.savez(output / 'clustering_model.npz', centers=centers, feature_mean=mean,
             feature_std_population=std, feature_scale=scale, zero_std_mask=(std == 0))
    np.savez_compressed(output / 'restart_solutions.npz', centers=np.asarray(centers_all),
                        assignments=np.asarray(labels_all, dtype=np.int64), seeds=seeds)
    with (output / 'cluster_assignments.jsonl').open('w') as f:
        for example_id, cluster in zip(ids, labels):
            f.write(json.dumps({'example_id': int(example_id), 'cluster': int(cluster)}) + '\n')
    write_json(output / 'starts.json', {'starts': starts, 'inertia_traces': traces})
    summary = dict(status='complete', n=1000, features=2048, k=2, n_init=50, master_seed=42,
                   selected_start=selected, selection='Minimum total within-cluster squared Euclidean distance; ties by start index',
                   inertia=starts[selected]['inertia'], mean_squared_distance=starts[selected]['inertia'] / len(z),
                   cluster_sizes=np.bincount(labels, minlength=2).tolist(),
                   converged_starts=sum(s['converged'] for s in starts),
                   unique_partitions=len({s['partition_sha256'] for s in starts}),
                   starts_with_selected_partition=sum(s['partition_sha256'] == starts[selected]['partition_sha256'] for s in starts),
                   inertia_range=[min(s['inertia'] for s in starts), max(s['inertia'] for s in starts)],
                   scaler_fitted_on='Only the1000unlabeled test embeddings; population std ddof0; zero std becomes1',
                   zero_std_features=int((std == 0).sum()), labels_loaded=False, training_embeddings_loaded=False,
                   pca_used=False, script_sha256=sha(__file__), plan_sha256=sha(args.plan),
                   input_embeddings_sha256=sha(source), embedding_completion_sha256=sha(completion_path),
                   packages={'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
                   stopping='Exactly unchanged assignments; maximum1000Lloyd iterations per initialization',
                   distance='Squared Euclidean on per-feature standardized vectors')
    write_json(output / 'clustering_summary.json', summary)
    names=['clustering_model.npz','restart_solutions.npz','cluster_assignments.jsonl','starts.json','clustering_summary.json']
    write_json(output / 'CLUSTER_COMPLETE.json', dict(status='complete', frozen_at=datetime.now(timezone.utc).isoformat(),
               labels_loaded=False, plan_sha256=sha(args.plan), files={n:sha(output/n) for n in names}))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
