#!/usr/bin/env python3
"""
Unified experiment runner for DES-kNN benchmarks.

Usage:
    # Single experiment
    python run_experiments.py --dataset mnist --method des_knn --k 10

    # Batch experiments (all combinations)
    python run_experiments.py --batch --quick

    # Batch with specific methods
    python run_experiments.py --batch --methods des_knn hnsw --datasets mnist
"""

import argparse
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.des_knn import DESKNNSearcher
from src.des_knn_guarantee import DESKNNSearcherGuarantee
from src.baselines import (
    ExactBruteForceKNN,
    KDTreeKNN,
    SklearnKDTreeKNN,
    BallTreeKNN,
    SklearnBallTreeKNN,
    LSHKNN,
    AnnoyKNN,
    HNSWKNN,
    FAISSIVFKNN
)
from src.sorting import PCASorter, ClusterSorter
from src.utils.data_loader import DataLoader
from src.utils.metrics import recall_at_k, aggregate_metrics


# =============================================================================
# Configuration Constants
# =============================================================================

DATASETS = [
    'mnist',
    'fashion_mnist',
    'synthetic_clustered',
    'synthetic_uniform',
    'sift1m',
    'gist1m',
    'glove'
]

METHODS = [
    'exact',
    'sklearn_kdtree',
    'sklearn_balltree',
    'lsh',
    'annoy',
    'hnsw',
    'des_knn',
    'des_knn_guarantee',
    'des_knn_pca',
    'des_knn_cluster'
]

K_VALUES = [1, 5, 10, 20, 50]

SEEDS = [42, 123]

# Method-specific parameter grids for batch experiments
METHOD_PARAMS = {
    'des_knn': [
        {'tolerance': 0.1, 'confidence': 0.99, 'max_cv': 0.2},   # Strict
        {'tolerance': 0.5, 'confidence': 0.99, 'max_cv': 0.3},   # Balanced
        {'tolerance': 1.0, 'confidence': 0.95, 'max_cv': None},  # Fast
    ],
    'des_knn_pca': [
        {'n_components': 16, 'tolerance': 1.0, 'confidence': 0.99, 'max_cv': 0.3},
        {'n_components': 32, 'tolerance': 1.0, 'confidence': 0.99, 'max_cv': 0.3},
        {'n_components': 64, 'tolerance': 0.5, 'confidence': 0.99, 'max_cv': 0.3},
    ],
    'des_knn_cluster': [
        {'n_clusters': 50, 'tolerance': 1.0, 'confidence': 0.99, 'max_cv': 0.3},
        {'n_clusters': 100, 'tolerance': 1.0, 'confidence': 0.99, 'max_cv': 0.3},
        {'n_clusters': 200, 'tolerance': 0.5, 'confidence': 0.99, 'max_cv': 0.3},
    ],
    'lsh': [
        {'n_tables': 10, 'n_bits': 10, 'n_probes': 2},
        {'n_tables': 20, 'n_bits': 12, 'n_probes': 3},
    ],
    'annoy': [
        {'n_trees': 10, 'search_k': None},
        {'n_trees': 50, 'search_k': None},
    ],
    'hnsw': [
        {'M': 16, 'ef_construction': 200, 'ef': 50},
        {'M': 16, 'ef_construction': 200, 'ef': 100},
    ]
}


# =============================================================================
# Method Factory
# =============================================================================

def get_method(method_name: str, X: np.ndarray, params: Dict[str, Any]):
    """Create a method instance by name with given parameters."""
    methods = {
        'exact': lambda: ExactBruteForceKNN(X),
        'kdtree': lambda: KDTreeKNN(X, leaf_size=params.get('leaf_size', 40)),
        'sklearn_kdtree': lambda: SklearnKDTreeKNN(X, leaf_size=params.get('leaf_size', 40)),
        'balltree': lambda: BallTreeKNN(X, leaf_size=params.get('leaf_size', 40)),
        'sklearn_balltree': lambda: SklearnBallTreeKNN(X, leaf_size=params.get('leaf_size', 40)),
        'lsh': lambda: LSHKNN(
            X,
            n_tables=params.get('n_tables', 10),
            n_bits=params.get('n_bits', 10),
            n_probes=params.get('n_probes', 1)
        ),
        'annoy': lambda: AnnoyKNN(
            X,
            n_trees=params.get('n_trees', 10),
            search_k=params.get('search_k', None)
        ),
        'hnsw': lambda: HNSWKNN(
            X,
            M=params.get('M', 16),
            ef_construction=params.get('ef_construction', 200),
            ef=params.get('ef', 10)
        ),
        'faiss_ivf': lambda: FAISSIVFKNN(
            X,
            nlist=params.get('nlist', 100),
            nprobe=params.get('nprobe', 10)
        ),
        'des_knn': lambda: DESKNNSearcher(
            X,
            tolerance=params.get('tolerance', 0.5),
            confidence=params.get('confidence', 0.99),
            max_cv=params.get('max_cv', None),
            min_samples=params.get('min_samples', None),
            block_size=params.get('block_size', 256)
        ),
        'des_knn_guarantee': lambda: DESKNNSearcherGuarantee(
            X,
            tolerance=params.get('tolerance', 0.5),
            confidence=params.get('confidence', 0.99),
            min_samples=params.get('min_samples', None),
            block_size=params.get('block_size', 256)
        ),
        'des_knn_pca': lambda: DESKNNSearcher(
            X,
            tolerance=params.get('tolerance', 1.0),
            confidence=params.get('confidence', 0.99),
            max_cv=params.get('max_cv', 0.3),
            min_samples=params.get('min_samples', None),
            sorter=PCASorter(n_components=params.get('n_components', 32)),
            block_size=params.get('block_size', 256)
        ),
        'des_knn_cluster': lambda: DESKNNSearcher(
            X,
            tolerance=params.get('tolerance', 1.0),
            confidence=params.get('confidence', 0.99),
            max_cv=params.get('max_cv', 0.3),
            min_samples=params.get('min_samples', None),
            sorter=ClusterSorter(n_clusters=params.get('n_clusters', 100)),
            block_size=params.get('block_size', 256)
        )
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")

    return methods[method_name]()


# =============================================================================
# Core Experiment Runner
# =============================================================================

def run_experiment(
    dataset_name: str,
    method_name: str,
    k: int,
    n_queries: int = 1000,
    method_params: Dict[str, Any] = None,
    dataset_params: Dict[str, Any] = None,
    random_seed: int = 42,
    verbose: bool = True,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """
    Run a single experiment.

    Parameters
    ----------
    dataset_name : str
        Name of dataset to use.
    method_name : str
        Name of method to evaluate.
    k : int
        Number of neighbors.
    n_queries : int
        Number of query points.
    method_params : Dict[str, Any]
        Method-specific parameters.
    dataset_params : Dict[str, Any]
        Dataset-specific parameters.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.
    n_jobs : int
        Number of parallel jobs.

    Returns
    -------
    results : Dict[str, Any]
        Experiment results.
    """
    method_params = method_params or {}
    np.random.seed(random_seed)

    # Load data
    if verbose:
        print(f"Loading dataset: {dataset_name}")

    data_loader = DataLoader(random_state=random_seed)
    dataset_params = dataset_params or {}
    X_train, X_test, y_train, y_test = data_loader.load(dataset_name, **dataset_params)

    if verbose:
        print(f"  Training set: {X_train.shape}")
        print(f"  Test set: {X_test.shape}")

    # Sample query points
    n_queries = min(n_queries, len(X_test))
    query_indices = np.random.choice(len(X_test), n_queries, replace=False)
    queries = X_test[query_indices]

    # Compute ground truth using exact search
    if verbose:
        print("Computing ground truth...")

    exact_searcher = ExactBruteForceKNN(X_train)
    exact_searcher.fit()

    ground_truth_neighbors = []
    exact_times = []

    for q in queries:
        t0 = time.perf_counter()
        neighbors, _, _ = exact_searcher.query(q, k)
        exact_times.append(time.perf_counter() - t0)
        ground_truth_neighbors.append(neighbors)

    ground_truth_neighbors = np.array(ground_truth_neighbors)
    mean_exact_time = np.mean(exact_times)

    if verbose:
        print(f"  Mean exact query time: {mean_exact_time*1000:.2f} ms")

    # Initialize and build method
    if verbose:
        print(f"Building {method_name}...")

    searcher = get_method(method_name, X_train, method_params)
    build_start = time.perf_counter()
    searcher.fit()
    build_time = time.perf_counter() - build_start

    if verbose:
        print(f"  Build time: {build_time:.2f} s")

    # Run queries
    if verbose:
        print(f"Running {n_queries} queries...")

    is_des_method = method_name.startswith('des_knn')

    all_recalls = []
    all_speedups = []
    all_dist_counts = []
    all_query_times = []
    all_scan_ratios = []
    all_expected_misses = [] if is_des_method else None

    if n_jobs == 1:
        for i, q in enumerate(queries):
            t0 = time.perf_counter()

            if is_des_method:
                neighbors, distances, dist_count, stats = searcher.query(q, k, return_stats=True)
                scan_ratio = stats.get('scan_ratio', dist_count / len(X_train))
                expected_misses = stats.get('expected_misses', None)
            else:
                neighbors, distances, dist_count = searcher.query(q, k)
                scan_ratio = dist_count / len(X_train)
                expected_misses = None

            query_time = time.perf_counter() - t0

            recall = recall_at_k(neighbors, ground_truth_neighbors[i])
            speedup = mean_exact_time / query_time if query_time > 0 else float('inf')

            all_recalls.append(recall)
            all_speedups.append(speedup)
            all_dist_counts.append(dist_count)
            all_query_times.append(query_time)
            all_scan_ratios.append(scan_ratio)
            if is_des_method:
                all_expected_misses.append(expected_misses)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_queries} queries...")
    else:
        from joblib import Parallel, delayed

        def _run_one(q, gt_neighbors):
            t0 = time.perf_counter()
            if is_des_method:
                neighbors, distances, dist_count, stats = searcher.query(q, k, return_stats=True)
                scan_ratio = stats.get('scan_ratio', dist_count / len(X_train))
                expected_misses = stats.get('expected_misses', None)
            else:
                neighbors, distances, dist_count = searcher.query(q, k)
                scan_ratio = dist_count / len(X_train)
                expected_misses = None

            query_time = time.perf_counter() - t0
            recall = recall_at_k(neighbors, gt_neighbors)
            speedup = mean_exact_time / query_time if query_time > 0 else float('inf')
            return recall, speedup, dist_count, query_time, scan_ratio, expected_misses

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_one)(q, ground_truth_neighbors[i]) for i, q in enumerate(queries)
        )

        for recall, speedup, dist_count, query_time, scan_ratio, expected_misses in results:
            all_recalls.append(recall)
            all_speedups.append(speedup)
            all_dist_counts.append(dist_count)
            all_query_times.append(query_time)
            all_scan_ratios.append(scan_ratio)
            if is_des_method:
                all_expected_misses.append(expected_misses)

    # Aggregate metrics
    metrics = aggregate_metrics(all_recalls, all_speedups, all_dist_counts, len(X_train))
    metrics['scan_ratio'] = {'mean': np.mean(all_scan_ratios), 'std': np.std(all_scan_ratios)}
    if is_des_method and all_expected_misses:
        expected_misses_arr = np.array(all_expected_misses, dtype=np.float32)
        metrics['expected_misses'] = {
            'mean': float(np.mean(expected_misses_arr)),
            'std': float(np.std(expected_misses_arr)),
            'median': float(np.median(expected_misses_arr)),
        }

    results = {
        'dataset': dataset_name,
        'method': method_name,
        'k': k,
        'n_queries': n_queries,
        'n_train': len(X_train),
        'n_dims': X_train.shape[1],
        'random_seed': random_seed,
        'method_params': method_params,
        'build_time': build_time,
        'mean_exact_time': mean_exact_time,
        'metrics': metrics,
        'per_query': {
            'recalls': all_recalls,
            'speedups': all_speedups,
            'dist_counts': all_dist_counts,
            'query_times': all_query_times,
            'scan_ratios': all_scan_ratios
        }
    }
    if is_des_method and all_expected_misses:
        results['per_query']['expected_misses'] = all_expected_misses

    if verbose:
        print("\nResults:")
        print(f"  Recall@{k}: {metrics['recall']['mean']:.4f} +/- {metrics['recall']['std']:.4f}")
        print(f"  Speedup: {metrics['speedup']['mean']:.2f}x +/- {metrics['speedup']['std']:.2f}")
        print(f"  Scan Ratio: {metrics['scan_ratio']['mean']*100:.1f}%")
        print(f"  Dist ratio: {metrics['dist_ratio']['mean']:.4f}")

    return results


# =============================================================================
# Batch Experiment Runner
# =============================================================================

def run_batch_experiments(
    datasets: List[str] = None,
    methods: List[str] = None,
    k_values: List[int] = None,
    seeds: List[int] = None,
    n_queries: int = 1000,
    output_dir: str = 'results',
    quick: bool = False,
    param_grid: str = 'single',
    n_jobs: int = 1
) -> Dict[str, Any]:
    """Run batch experiments over all combinations."""
    datasets = datasets or DATASETS
    methods = methods or METHODS
    k_values = k_values or K_VALUES
    seeds = seeds or SEEDS

    if quick:
        datasets = datasets[:2]
        methods = ['exact', 'des_knn', 'hnsw']
        k_values = [10]
        seeds = seeds[:1]
        n_queries = 100

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Count total experiments
    total = 0
    for dataset in datasets:
        for method in methods:
            if param_grid == 'full' and method in METHOD_PARAMS:
                param_configs = METHOD_PARAMS[method]
            else:
                param_configs = [{}]
            total += len(param_configs) * len(k_values) * len(seeds)

    current = 0

    for dataset in datasets:
        all_results[dataset] = {}

        for method in methods:
            all_results[dataset][method] = {}

            if param_grid == 'full' and method in METHOD_PARAMS:
                param_configs = METHOD_PARAMS[method]
            else:
                param_configs = [{}]

            for k in k_values:
                all_results[dataset][method][k] = {}

                for param_config in param_configs:
                    config_key = str(param_config) if param_config else 'default'
                    all_results[dataset][method][k][config_key] = []

                    for seed in seeds:
                        current += 1
                        print(f"\n[{current}/{total}] {dataset} / {method} / k={k} / {config_key} / seed={seed}")

                        try:
                            results = run_experiment(
                                dataset_name=dataset,
                                method_name=method,
                                k=k,
                                n_queries=n_queries,
                                method_params=param_config,
                                random_seed=seed,
                                verbose=False,
                                n_jobs=n_jobs
                            )

                            all_results[dataset][method][k][config_key].append({
                                'seed': seed,
                                'recall_mean': results['metrics']['recall']['mean'],
                                'recall_std': results['metrics']['recall']['std'],
                                'speedup_mean': results['metrics']['speedup']['mean'],
                                'speedup_std': results['metrics']['speedup']['std'],
                                'scan_ratio': results['metrics']['scan_ratio']['mean'],
                                'dist_ratio': results['metrics']['dist_ratio']['mean'],
                                'build_time': results['build_time'],
                            })

                            print(f"  Recall: {results['metrics']['recall']['mean']:.4f}, "
                                  f"Speedup: {results['metrics']['speedup']['mean']:.2f}x, "
                                  f"Scan: {results['metrics']['scan_ratio']['mean']*100:.1f}%")

                        except Exception as e:
                            print(f"  ERROR: {e}")
                            all_results[dataset][method][k][config_key].append({
                                'seed': seed,
                                'error': str(e)
                            })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f'batch_results_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    return all_results


# =============================================================================
# CLI
# =============================================================================

def convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(
        description='Run DES-kNN experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python run_experiments.py --dataset mnist --method des_knn --k 10

  # Quick batch test
  python run_experiments.py --batch --quick

  # Full batch with parameter grid
  python run_experiments.py --batch --param_grid full

  # Specific methods and datasets
  python run_experiments.py --batch --methods des_knn hnsw --datasets mnist fashion_mnist
"""
    )

    # Mode selection
    parser.add_argument('--batch', action='store_true',
                        help='Run batch experiments over all combinations')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with reduced settings')

    # Dataset and method selection
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=DATASETS + ['cifar10', 'cifar10_resnet'],
                        help='Dataset for single experiment')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Datasets for batch experiments')
    parser.add_argument('--method', type=str, default='des_knn',
                        choices=METHODS + ['kdtree', 'balltree', 'faiss_ivf', 'all'],
                        help='Method for single experiment')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods for batch experiments')

    # Experiment parameters
    parser.add_argument('--k', type=int, default=10, help='Number of neighbors')
    parser.add_argument('--k_values', nargs='+', type=int, default=None,
                        help='K values for batch experiments')
    parser.add_argument('--n_queries', type=int, default=1000, help='Number of queries')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--n_jobs', type=int, default=1, help='Parallel jobs')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--param_grid', type=str, default='single',
                        choices=['single', 'full'], help='Parameter grid mode')

    # Method-specific parameters
    parser.add_argument('--tolerance', type=float, default=0.5)
    parser.add_argument('--confidence', type=float, default=0.99)
    parser.add_argument('--max_cv', type=float, default=None)
    parser.add_argument('--block_size', type=int, default=256)
    parser.add_argument('--n_components', type=int, default=32, help='PCA components')
    parser.add_argument('--n_clusters', type=int, default=100, help='Cluster count')
    parser.add_argument('--n_tables', type=int, default=10, help='LSH tables')
    parser.add_argument('--n_trees', type=int, default=10, help='Annoy trees')
    parser.add_argument('--search_k', type=int, default=None, help='Annoy search_k')
    parser.add_argument('--ef', type=int, default=10, help='HNSW ef')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        # Batch mode
        run_batch_experiments(
            datasets=args.datasets,
            methods=args.methods,
            k_values=args.k_values,
            n_queries=args.n_queries,
            output_dir=args.output_dir,
            quick=args.quick,
            param_grid=args.param_grid,
            n_jobs=args.n_jobs
        )
    else:
        # Single experiment mode
        method_params = {
            'tolerance': args.tolerance,
            'confidence': args.confidence,
            'max_cv': args.max_cv,
            'block_size': args.block_size,
            'n_components': args.n_components,
            'n_clusters': args.n_clusters,
            'n_tables': args.n_tables,
            'n_trees': args.n_trees,
            'search_k': args.search_k,
            'ef': args.ef,
        }

        if args.method == 'all':
            methods = METHODS
        else:
            methods = [args.method]

        all_results = {}

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Running {method} on {args.dataset}")
            print('='*60)

            results = run_experiment(
                dataset_name=args.dataset,
                method_name=method,
                k=args.k,
                n_queries=args.n_queries,
                method_params=method_params,
                random_seed=args.seed,
                verbose=True,
                n_jobs=args.n_jobs
            )
            all_results[method] = results

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f'results_{args.dataset}_k{args.k}_{timestamp}.json'

        with open(output_file, 'w') as f:
            json.dump(convert_to_serializable(all_results), f, indent=2)

        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
