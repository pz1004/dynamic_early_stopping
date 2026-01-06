#!/usr/bin/env python3
"""
Benchmark harness for DES-kNN (Gap + CV Version).
"""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.des_knn import DESKNNSearcher
from src.baselines.exact_brute_force import ExactBruteForceKNN


def summarize(values: List[float]) -> dict:
    """Return summary statistics."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DES-kNN")
    parser.add_argument("--n", type=int, default=10000, help="Number of points")
    parser.add_argument("--d", type=int, default=128, help="Dimensionality")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # New DES-kNN parameters
    parser.add_argument("--tolerance", type=float, default=0.5, 
                       help="Expected misses tolerance")
    parser.add_argument("--confidence", type=float, default=0.99, 
                       help="Statistical confidence")
    parser.add_argument("--max-cv", type=float, default=None, 
                       help="Max Coefficient of Variation (optional)")
    
    parser.add_argument("--out", type=str, default=None, help="JSON output path")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, args.d)).astype(np.float32)
    queries = rng.standard_normal((args.n_queries, args.d)).astype(np.float32)

    # 1. Ground Truth
    print("Computing exact ground truth...")
    exact = ExactBruteForceKNN(X)
    exact.fit()
    exact_neighbors = []
    for q in queries:
        neighbors, _, _ = exact.query(q, args.k)
        exact_neighbors.append(neighbors)

    # 2. Benchmark DES-kNN
    print(f"Benchmarking DES-kNN (tol={args.tolerance}, conf={args.confidence}, cv={args.max_cv})...")
    
    searcher = DESKNNSearcher(
        X,
        tolerance=args.tolerance,
        confidence=args.confidence,
        max_cv=args.max_cv,
        random_state=args.seed
    )

    times = []
    counts = []
    recalls = []
    scan_ratios = []

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        neighbors, _, count, stats = searcher.query(q, args.k, return_stats=True)
        times.append(time.perf_counter() - t0)
        counts.append(count)
        scan_ratios.append(stats['scan_ratio'])

        # Recall
        intersection = len(set(neighbors) & set(exact_neighbors[i]))
        recalls.append(intersection / args.k)

    # 3. Report
    res = {
        "recall": summarize(recalls),
        "time_ms": {k: v * 1000 for k, v in summarize(times).items()},
        "scan_ratio": summarize(scan_ratios),
        "count": summarize(counts)
    }

    print(f"\nResults:")
    print(f"  Recall Mean: {res['recall']['mean']:.4f}")
    print(f"  Speedup:     {1.0 / res['scan_ratio']['mean']:.2f}x (avg scan {res['scan_ratio']['mean']*100:.1f}%)")
    print(f"  Time (p50):  {res['time_ms']['p50']:.2f} ms")
    
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())