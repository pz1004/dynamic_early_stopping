#!/usr/bin/env python3
"""
Optimal Tolerance Finder for DES-kNN.

This script finds the optimal tolerance parameter for a given dataset by:
1. Running a coarse grid search over tolerance values
2. Fine-tuning around the best candidate
3. Selecting the highest tolerance that maintains target recall

Supports both des_knn_pca and des_knn_guarantee methods.

Usage:
    python find_optimal_tolerance.py --dataset sift1m --k 10
    python find_optimal_tolerance.py --dataset mnist --k 10 --method des_knn_guarantee
    python find_optimal_tolerance.py --dataset sift1m --k 10 --target_recall 0.95
"""

import argparse
import numpy as np
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.neighbors import NearestNeighbors

from src.des_knn import DESKNNSearcher
from src.des_knn_guarantee import DESKNNSearcherGuarantee
from src.sorting import PCASorter
from src.utils.data_loader import DataLoader
from src.utils.metrics import recall_at_k


def create_searcher(X_train, tolerance, method='des_knn_pca',
                    sorter=None, n_components=32, block_size=256):
    """
    Create a DES-kNN searcher based on method type.

    Args:
        X_train: Training data
        tolerance: Tolerance parameter
        method: 'des_knn_pca' or 'des_knn_guarantee'
        sorter: Pre-fitted sorter (for des_knn_pca)
        n_components: PCA components (if sorter not provided)
        block_size: Block size for vectorized processing

    Returns:
        Configured DESKNNSearcher or DESKNNSearcherGuarantee
    """
    if method == 'des_knn_pca':
        if sorter is None:
            sorter = PCASorter(n_components=n_components)
        return DESKNNSearcher(
            X_train,
            tolerance=tolerance,
            confidence=0.99,
            sorter=sorter,
            block_size=block_size
        )
    elif method == 'des_knn_guarantee':
        return DESKNNSearcherGuarantee(
            X_train,
            tolerance=tolerance,
            confidence=0.99,
            block_size=block_size
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'des_knn_pca' or 'des_knn_guarantee'")


def evaluate_tolerance(X_train, X_test, gt_neighbors, k, tolerance,
                       method='des_knn_pca', sorter=None,
                       n_components=32, n_queries=100, block_size=256,
                       mean_exact_time=None):
    """
    Evaluate a specific tolerance value.

    Returns: (mean_recall, mean_scan_ratio, mean_speedup, std_speedup)

    Args:
        method: 'des_knn_pca' or 'des_knn_guarantee'
        sorter: Pre-fitted sorter instance (avoids re-fitting PCA each time)
        mean_exact_time: Pre-computed baseline time (avoids re-measuring)
    """
    # Create searcher
    searcher = create_searcher(
        X_train, tolerance, method=method,
        sorter=sorter, n_components=n_components,
        block_size=block_size
    )
    searcher.fit()

    # Compute exact search time baseline if not provided
    if mean_exact_time is None:
        exact_times = []
        for i in range(min(20, n_queries)):
            t0 = time.perf_counter()
            dists = np.linalg.norm(X_train - X_test[i], axis=1)
            _ = np.argpartition(dists, k)[:k]
            exact_times.append(time.perf_counter() - t0)
        mean_exact_time = np.mean(exact_times)

    # Evaluate
    recalls = []
    scan_ratios = []
    speedups = []

    for i in range(n_queries):
        t0 = time.perf_counter()
        neighbors, _, _, stats = searcher.query(X_test[i], k=k, return_stats=True)
        query_time = time.perf_counter() - t0

        recalls.append(recall_at_k(neighbors, gt_neighbors[i]))
        scan_ratios.append(stats['scan_ratio'])
        speedups.append(mean_exact_time / query_time if query_time > 0 else 0)

    return (
        np.mean(recalls),
        np.mean(scan_ratios),
        np.mean(speedups),
        np.std(speedups)
    )


def find_optimal_tolerance(
    X_train, X_test,
    k=10,
    target_recall=0.99,
    method='des_knn_pca',
    n_components=32,
    n_queries=500,
    tau_range=(1.0, 500.0),
    verbose=True
):
    """
    Find optimal tolerance using coarse grid search + fine-tuning.

    Args:
        X_train: Training data
        X_test: Test queries
        k: Number of neighbors
        target_recall: Minimum acceptable recall (default: 0.99)
        method: 'des_knn_pca' or 'des_knn_guarantee'
        n_components: PCA components (for des_knn_pca)
        n_queries: Number of queries for evaluation
        tau_range: (min, max) tolerance search range
        verbose: Print progress

    Returns:
        dict with optimal_tau, recall, scan_ratio, speedup, and sweep details
    """
    results = {
        'method': method,
        'dataset_info': {
            'n_train': len(X_train),
            'n_dims': X_train.shape[1],
            'n_queries': n_queries
        }
    }

    # Step 1: Compute ground truth and baseline time
    if verbose:
        print(f"Method: {method}")
        print(f"\nStep 1: Computing ground truth neighbors...")

    nn = NearestNeighbors(n_neighbors=k, algorithm='brute')
    nn.fit(X_train)
    _, gt_neighbors = nn.kneighbors(X_test[:n_queries])

    # Pre-compute baseline exact search time
    if verbose:
        print("  Computing baseline exact search time...")
    exact_times = []
    for i in range(min(50, n_queries)):
        t0 = time.perf_counter()
        dists = np.linalg.norm(X_train - X_test[i], axis=1)
        _ = np.argpartition(dists, k)[:k]
        exact_times.append(time.perf_counter() - t0)
    mean_exact_time = np.mean(exact_times)
    if verbose:
        print(f"  Baseline: {mean_exact_time*1000:.2f} ms per query")

    # Pre-train sorter for des_knn_pca (reuse for all evaluations)
    sorter = None
    if method == 'des_knn_pca':
        if verbose:
            print("  Pre-training PCA sorter...")
        sorter = PCASorter(n_components=n_components)
        sorter.fit(X_train)
        if verbose:
            print("  Sorter training complete.")

    # Step 2: Coarse grid search
    if verbose:
        print("\nStep 2: Coarse tolerance sweep...")
        print(f"{'τ':>6} | {'Recall':>7} | {'Scan':>6} | {'Speedup':>8}")
        print("-" * 40)

    # Generate coarse tau values with logarithmic spacing
    n_coarse = 12
    coarse_taus = np.unique(np.round(
        np.geomspace(max(1, tau_range[0]), tau_range[1], n_coarse)
    ).astype(int))
    # Ensure we include endpoints
    coarse_taus = np.unique(np.concatenate([
        [tau_range[0]] if tau_range[0] >= 1 else [],
        coarse_taus,
        [tau_range[1]]
    ]))

    coarse_results = []
    for tau in coarse_taus:
        recall, scan, speedup, speedup_std = evaluate_tolerance(
            X_train, X_test[:n_queries], gt_neighbors, k, tau,
            method=method, sorter=sorter,
            n_components=n_components, n_queries=n_queries,
            mean_exact_time=mean_exact_time
        )
        coarse_results.append({
            'tau': tau, 'recall': recall, 'scan_ratio': scan,
            'speedup': speedup, 'speedup_std': speedup_std
        })

        if verbose:
            print(f"{tau:>6.1f} | {recall*100:>6.1f}% | {scan*100:>5.1f}% | {speedup:>7.2f}x")

    results['coarse_sweep'] = coarse_results

    # Step 3: Find best tau with recall >= target
    # Strategy: Prefer HIGHEST τ (lowest scan ratio) for stability
    valid_results = [r for r in coarse_results if r['recall'] >= target_recall]

    if not valid_results:
        best_coarse = min(coarse_results, key=lambda x: x['tau'])
        if verbose:
            print(f"\n  Warning: No tolerance achieves {target_recall*100:.0f}% recall")
    else:
        best_coarse = max(valid_results, key=lambda x: x['tau'])

    # Step 4: Fine-grained search around best
    if verbose:
        print(f"\nStep 3: Fine-tuning around τ = {best_coarse['tau']:.1f}...")

    # Wider fine-tuning range: ±10% of best_coarse['tau'] or ±5
    fine_delta = max(5, best_coarse['tau'] * 0.1)
    fine_range = np.linspace(
        max(tau_range[0], best_coarse['tau'] - fine_delta),
        min(tau_range[1], best_coarse['tau'] + fine_delta),
        9
    )

    fine_results = []
    for tau in fine_range:
        recall, scan, speedup, speedup_std = evaluate_tolerance(
            X_train, X_test[:n_queries], gt_neighbors, k, tau,
            method=method, sorter=sorter,
            n_components=n_components, n_queries=n_queries,
            mean_exact_time=mean_exact_time
        )
        fine_results.append({
            'tau': tau, 'recall': recall, 'scan_ratio': scan,
            'speedup': speedup, 'speedup_std': speedup_std
        })

        if verbose:
            status = "✓" if recall >= target_recall else "✗"
            print(f"  τ={tau:>5.1f}: Recall={recall*100:.1f}%, Scan={scan*100:.1f}%, Speedup={speedup:.2f}x {status}")

    results['fine_sweep'] = fine_results

    # Step 5: Select optimal (highest τ that maintains recall)
    valid_fine = [r for r in fine_results if r['recall'] >= target_recall]

    if valid_fine:
        optimal = max(valid_fine, key=lambda x: x['tau'])
    else:
        optimal = min(fine_results, key=lambda x: x['tau'])

    results['optimal'] = {
        'tolerance': optimal['tau'],
        'recall': optimal['recall'],
        'scan_ratio': optimal['scan_ratio'],
        'speedup': optimal['speedup'],
        'speedup_std': optimal['speedup_std']
    }

    if verbose:
        print("\n" + "=" * 50)
        print("OPTIMAL TOLERANCE FOUND")
        print("=" * 50)
        print(f"  Method: {method}")
        print(f"  τ_opt = {optimal['tau']:.1f}")
        print(f"  Expected Recall: {optimal['recall']*100:.1f}%")
        print(f"  Expected Scan Ratio: {optimal['scan_ratio']*100:.1f}%")
        print(f"  Expected Speedup: {optimal['speedup']:.2f}x ± {optimal['speedup_std']:.2f}")
        print("=" * 50)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Find optimal tolerance for DES-kNN methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Find optimal tolerance for des_knn_pca on SIFT1M
  python find_optimal_tolerance.py --dataset sift1m --k 10

  # Find optimal tolerance for des_knn_guarantee on MNIST
  python find_optimal_tolerance.py --dataset mnist --k 10 --method des_knn_guarantee

  # Custom recall target and search range
  python find_optimal_tolerance.py --dataset sift1m --k 10 --target_recall 0.95 --tau_max 200
"""
    )

    parser.add_argument('--dataset', type=str, default='sift1m',
                        help='Dataset name')
    parser.add_argument('--method', type=str, default='des_knn_pca',
                        choices=['des_knn_pca', 'des_knn_guarantee'],
                        help='DES-kNN method to optimize')
    parser.add_argument('--k', type=int, default=10,
                        help='Number of neighbors')
    parser.add_argument('--target_recall', type=float, default=0.999,
                        help='Target recall (default: 0.99)')
    parser.add_argument('--n_components', type=int, default=32,
                        help='PCA components (for des_knn_pca)')
    parser.add_argument('--n_queries', type=int, default=500,
                        help='Number of queries for evaluation (higher = more stable)')
    parser.add_argument('--tau_min', type=float, default=1.0,
                        help='Minimum tolerance to search')
    parser.add_argument('--tau_max', type=float, default=500.0,
                        help='Maximum tolerance to search')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    np.random.seed(args.seed)

    # Load data
    if not args.quiet:
        print(f"Loading dataset: {args.dataset}")

    loader = DataLoader()
    X_train, X_test, _, _ = loader.load(args.dataset)

    if not args.quiet:
        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print()

    # Find optimal tolerance
    results = find_optimal_tolerance(
        X_train, X_test,
        k=args.k,
        target_recall=args.target_recall,
        method=args.method,
        n_components=args.n_components,
        n_queries=args.n_queries,
        tau_range=(args.tau_min, args.tau_max),
        verbose=not args.quiet
    )

    # Print command to use optimal
    if not args.quiet:
        print("\nTo use this tolerance:")
        print(f"  python experiments/run_experiments.py \\")
        print(f"      --dataset {args.dataset} --method {args.method} \\")
        print(f"      --k {args.k} --tolerance {results['optimal']['tolerance']:.1f}")

    return results


if __name__ == '__main__':
    main()
