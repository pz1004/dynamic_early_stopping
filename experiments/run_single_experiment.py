#!/usr/bin/env python3
"""
Run a single k-NN experiment with specified parameters.

Usage:
    python run_single_experiment.py --dataset mnist --k 10 --method des_knn
    python run_single_experiment.py --dataset synthetic_clustered --k 20 --method all
"""

import argparse
import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.des_knn import DESKNNSearcher
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
from src.utils.data_loader import DataLoader
from src.utils.metrics import recall_at_k, aggregate_metrics, query_difficulty_analysis


def get_method(method_name: str, X: np.ndarray, params: Dict[str, Any]):
    """
    Get method instance by name.

    Parameters
    ----------
    method_name : str
        Name of the method.
    X : np.ndarray
        Training data.
    params : Dict[str, Any]
        Method-specific parameters.

    Returns
    -------
    searcher : BaseKNNSearcher
        Initialized searcher instance.
    """
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
            alpha=params.get('alpha', 0.01),
            window_size=params.get('window_size', 100),
            adaptive_alpha=params.get('adaptive_alpha', True),
            weibull_refresh_every=params.get('weibull_refresh_every', 10)
        ),
    }

    if method_name not in methods:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(methods.keys())}")

    return methods[method_name]()


def run_experiment(
    dataset_name: str,
    method_name: str,
    k: int,
    n_queries: int = 1000,
    method_params: Dict[str, Any] = None,
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
        Number of query points to use.
    method_params : Dict[str, Any]
        Method-specific parameters.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.

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
    X_train, X_test, y_train, y_test = data_loader.load(dataset_name)

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
    ground_truth_distances = []
    exact_times = []

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        neighbors, distances, _ = exact_searcher.query(q, k)
        exact_times.append(time.perf_counter() - t0)
        ground_truth_neighbors.append(neighbors)
        ground_truth_distances.append(distances)

    ground_truth_neighbors = np.array(ground_truth_neighbors)
    ground_truth_distances = np.array(ground_truth_distances)
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

    all_recalls = []
    all_speedups = []
    all_dist_counts = []
    all_query_times = []
    all_stopping_points = []

    stop_check_every = method_params.get('stop_check_every', 50)
    if method_name == 'des_knn' and verbose:
        print("Running DES-kNN without return_stats for speed.")

    if n_jobs == 1:
        for i, q in enumerate(queries):
            t0 = time.perf_counter()

            if method_name == 'des_knn':
                neighbors, distances, dist_count = searcher.query(
                    q,
                    k,
                    stop_check_every=stop_check_every
                )
            else:
                neighbors, distances, dist_count = searcher.query(q, k)
            all_stopping_points.append(dist_count / len(X_train))

            query_time = time.perf_counter() - t0

            # Compute metrics
            recall = recall_at_k(neighbors, ground_truth_neighbors[i])
            speedup = mean_exact_time / query_time if query_time > 0 else float('inf')

            all_recalls.append(recall)
            all_speedups.append(speedup)
            all_dist_counts.append(dist_count)
            all_query_times.append(query_time)

            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_queries} queries...")
    else:
        from joblib import Parallel, delayed

        def _run_one(q, gt_neighbors):
            t0 = time.perf_counter()
            if method_name == 'des_knn':
                neighbors, distances, dist_count = searcher.query(
                    q,
                    k,
                    stop_check_every=stop_check_every
                )
            else:
                neighbors, distances, dist_count = searcher.query(q, k)
            stopping_point = dist_count / len(X_train)

            query_time = time.perf_counter() - t0
            recall = recall_at_k(neighbors, gt_neighbors)
            speedup = mean_exact_time / query_time if query_time > 0 else float('inf')
            return recall, speedup, dist_count, query_time, stopping_point

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_one)(q, ground_truth_neighbors[i]) for i, q in enumerate(queries)
        )

        for recall, speedup, dist_count, query_time, stopping_point in results:
            all_recalls.append(recall)
            all_speedups.append(speedup)
            all_dist_counts.append(dist_count)
            all_query_times.append(query_time)
            all_stopping_points.append(stopping_point)

    # Aggregate metrics
    metrics = aggregate_metrics(
        all_recalls,
        all_speedups,
        all_dist_counts,
        len(X_train)
    )

    # Add additional info
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
            'stopping_points': all_stopping_points
        }
    }

    if verbose:
        print("\nResults:")
        print(f"  Recall@{k}: {metrics['recall']['mean']:.4f} +/- {metrics['recall']['std']:.4f}")
        print(f"  Speedup: {metrics['speedup']['mean']:.2f}x +/- {metrics['speedup']['std']:.2f}")
        print(f"  Dist ratio: {metrics['dist_ratio']['mean']:.4f}")
        print(f"  Failure rate (<99% recall): {metrics['failure_rate_99']*100:.1f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run k-NN experiment')
    parser.add_argument('--dataset', type=str, default='mnist',
                       choices=['mnist', 'fashion_mnist', 'cifar10', 'cifar10_resnet',
                               'synthetic_clustered', 'synthetic_uniform'])
    parser.add_argument('--method', type=str, default='des_knn',
                       choices=['exact', 'kdtree', 'sklearn_kdtree', 'balltree',
                               'sklearn_balltree', 'lsh', 'annoy', 'hnsw',
                               'faiss_ivf', 'des_knn', 'all'])
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--n_queries', type=int, default=1000)
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs for query evaluation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')

    # Method-specific parameters
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='DES-kNN confidence level')
    parser.add_argument('--window_size', type=int, default=100,
                       help='DES-kNN window size')
    parser.add_argument('--stop_check_every', type=int, default=50,
                       help='DES-kNN stop check cadence')
    parser.add_argument('--weibull_refresh_every', type=int, default=10,
                       help='DES-kNN Weibull refresh cadence')
    parser.add_argument('--n_tables', type=int, default=10,
                       help='LSH number of tables')
    parser.add_argument('--n_trees', type=int, default=10,
                       help='Annoy number of trees')
    parser.add_argument('--ef', type=int, default=10,
                       help='HNSW search ef parameter')

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect method parameters
    method_params = {
        'alpha': args.alpha,
        'window_size': args.window_size,
        'stop_check_every': args.stop_check_every,
        'weibull_refresh_every': args.weibull_refresh_every,
        'n_tables': args.n_tables,
        'n_trees': args.n_trees,
        'ef': args.ef
    }

    # Run experiment(s)
    if args.method == 'all':
        methods = [
            'exact',
            'kdtree',
            'sklearn_kdtree',
            'balltree',
            'sklearn_balltree',
            'lsh',
            'annoy',
            'hnsw',
            'faiss_ivf',
            'des_knn'
        ]
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

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
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

    with open(output_file, 'w') as f:
        json.dump(convert_to_serializable(all_results), f, indent=2)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
