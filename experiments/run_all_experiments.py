#!/usr/bin/env python3
"""
Run complete experiment suite across all datasets and methods.

Usage:
    python run_all_experiments.py
    python run_all_experiments.py --quick  # Reduced settings for testing
"""

import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import sys
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_single_experiment import run_experiment


# Experiment configurations
DATASETS = [
    'mnist',
    'fashion_mnist',
    'synthetic_clustered',
    'synthetic_uniform'
]

METHODS = [
    'exact',
    'sklearn_kdtree',
    'sklearn_balltree',
    'lsh',
    'annoy',
    'hnsw',
    'des_knn'
]

K_VALUES = [1, 5, 10, 20, 50]

# SEEDS = [42, 123, 456, 789, 1024]
SEEDS = [42, 123]

# Method-specific parameter grids
METHOD_PARAMS = {
    'des_knn': [
        {'alpha': 0.01, 'window_size': 100, 'adaptive_alpha': True},
        {'alpha': 0.005, 'window_size': 100, 'adaptive_alpha': True},
        {'alpha': 0.01, 'window_size': 200, 'adaptive_alpha': True},
        {'alpha': 0.01, 'window_size': 100, 'adaptive_alpha': False},
    ],
    'lsh': [
        {'n_tables': 10, 'n_bits': 10, 'n_probes': 2},
        {'n_tables': 20, 'n_bits': 12, 'n_probes': 3},
        {'n_tables': 50, 'n_bits': 15, 'n_probes': 5},
    ],
    'annoy': [
        {'n_trees': 10, 'search_k': None},
        {'n_trees': 50, 'search_k': None},
        {'n_trees': 100, 'search_k': None},
    ],
    'hnsw': [
        {'M': 16, 'ef_construction': 200, 'ef': 50},
        {'M': 16, 'ef_construction': 200, 'ef': 100},
        {'M': 32, 'ef_construction': 200, 'ef': 100},
    ]
}


def run_all_experiments(
    datasets: List[str] = None,
    methods: List[str] = None,
    k_values: List[int] = None,
    seeds: List[int] = None,
    n_queries: int = 1000,
    output_dir: str = 'results',
    quick: bool = False
) -> Dict[str, Any]:
    """
    Run full experiment suite.
    """
    datasets = datasets or DATASETS
    methods = methods or METHODS
    k_values = k_values or K_VALUES
    seeds = seeds or SEEDS

    if quick:
        # Reduced settings for quick testing
        datasets = datasets[:2]
        methods = ['exact', 'des_knn', 'hnsw']
        k_values = [10]
        seeds = seeds[:2]
        n_queries = 100

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = {}

    total_experiments = len(datasets) * len(methods) * len(k_values) * len(seeds)
    current = 0

    for dataset in datasets:
        all_results[dataset] = {}

        for method in methods:
            all_results[dataset][method] = {}

            # Get parameter configurations for this method
            if method in METHOD_PARAMS:
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
                        print(f"\n[{current}/{total_experiments}] "
                              f"Dataset: {dataset}, Method: {method}, k: {k}, seed: {seed}")

                        try:
                            results = run_experiment(
                                dataset_name=dataset,
                                method_name=method,
                                k=k,
                                n_queries=n_queries,
                                method_params=param_config,
                                random_seed=seed,
                                verbose=False
                            )

                            all_results[dataset][method][k][config_key].append({
                                'seed': seed,
                                'recall_mean': results['metrics']['recall']['mean'],
                                'recall_std': results['metrics']['recall']['std'],
                                'speedup_mean': results['metrics']['speedup']['mean'],
                                'speedup_std': results['metrics']['speedup']['std'],
                                'dist_ratio': results['metrics']['dist_ratio']['mean'],
                                'build_time': results['build_time'],
                                'failure_rate_99': results['metrics']['failure_rate_99']
                            })

                            print(f"  Recall: {results['metrics']['recall']['mean']:.4f}, "
                                  f"Speedup: {results['metrics']['speedup']['mean']:.2f}x")

                        except Exception as e:
                            print(f"  ERROR: {e}")
                            all_results[dataset][method][k][config_key].append({
                                'seed': seed,
                                'error': str(e)
                            })

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_path / f'full_results_{timestamp}.json'

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll results saved to: {output_file}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run full experiment suite')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version with reduced settings')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Datasets to use')
    parser.add_argument('--methods', nargs='+', default=None,
                       help='Methods to evaluate')
    parser.add_argument('--k_values', nargs='+', type=int, default=None,
                       help='Values of k to test')
    parser.add_argument('--n_queries', type=int, default=1000,
                       help='Number of queries per experiment')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory')

    args = parser.parse_args()

    run_all_experiments(
        datasets=args.datasets,
        methods=args.methods,
        k_values=args.k_values,
        n_queries=args.n_queries,
        output_dir=args.output_dir,
        quick=args.quick
    )


if __name__ == '__main__':
    main()
