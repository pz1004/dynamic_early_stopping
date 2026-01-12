"""
experiments/run_paper_experiments.py

Dedicated runner for generating PRL results.
Runs parameter sweeps with per-seed logging for matched-recall frontiers
and DES transparency figures.
"""

import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_experiments import run_experiment

# Configuration for the Paper
# Include synthetic datasets for transparency plots, and SIFT/GloVe for scale.
DATASETS = [
    'mnist',
    'fashion_mnist',
    'synthetic_clustered',
    'synthetic_uniform',
    'sift1m',
    'glove'
]
METHODS = ['exact', 'des_knn_pca', 'des_knn_guarantee', 'hnsw', 'annoy']
K_VALUES = [10]
SEEDS = [0, 1, 2]  # 3 seeds for error bars

# Dataset-specific loader parameters
DATASET_PARAMS = {
    'synthetic_clustered': {
        'n': 100000,
        'd': 128,
        'n_clusters': 50,
        'cluster_std': 1.0,
    },
    'synthetic_uniform': {
        'n': 100000,
        'd': 128,
    },
    'glove': {
        'dim': 300,
        'max_words': 400000,
        'allow_synthetic': False,
    },
}

# Dense sweeps for Pareto curves (dataset-specific where needed)
# Middle value targets ~0.99 recall based on empirical analysis
DES_KNN_PCA_TAUS = {
    # mnist/fashion_mnist: All taus achieve ~1.0 recall (easy dataset)
    'mnist': [12800, 25600, 51200],
    'fashion_mnist': [100, 200, 400, 800, 1600, 3200, 6400],
    # synthetic_clustered: 0.99 at tau~500
    'synthetic_clustered': [50, 100, 200, 500, 1000, 2000, 4000],
    # synthetic_uniform: 0.99 at tau~10 (hardest for PCA sorter)
    'synthetic_uniform': [2, 5, 10, 20, 50, 100, 200],
    # sift1m: All taus achieve 0.9994 recall
    'sift1m': [100, 200, 400, 800, 1600, 3200, 6400],
    'glove': [10, 20, 30, 40, 50],
}

DES_KNN_GUARANTEE_TAUS = {
    # mnist/fashion_mnist: 0.99 at tau~20
    'mnist': [10, 20, 30, 40, 50],
    'fashion_mnist': [1, 5, 10, 20, 50, 100, 200],
    # synthetic_clustered/uniform: 0.99 at tau~20
    'synthetic_clustered': [2, 5, 10, 20, 50, 100, 200],
    'synthetic_uniform': [2, 5, 10, 20, 50, 100, 200],
    # sift1m: Does NOT achieve 0.99 (max ~0.4), include for completeness
    'sift1m': [1, 5, 10, 50, 100, 500, 1000],
    'glove': [10, 20, 30, 40, 50],
}

HNSW_EF_SWEEPS = {
    # mnist/fashion_mnist: 0.99 at ef~32
    'mnist': [8, 16, 32, 64, 128, 256],
    'fashion_mnist': [8, 16, 32, 64, 128, 256],
    # synthetic_clustered: 0.99 at ef~100
    'synthetic_clustered': [20, 50, 100, 200, 400, 800],
    # synthetic_uniform: Never reaches 0.99 (max 0.93 at ef=800)
    'synthetic_uniform': [100, 200, 400, 800, 1600, 3200],
    # sift1m: 0.99 at ef~200
    'sift1m': [50, 100, 200, 400, 600, 800, 1200],
    'glove': [50, 100, 200, 400, 600, 800, 1200],
}

ANNOY_SEARCHK_SWEEPS = {
    # mnist/fashion_mnist: 0.99 at search_k~5000-10000
    'mnist': [1000, 2000, 5000, 10000, 20000, 40000],
    'fashion_mnist': [1000, 2000, 5000, 10000, 20000, 40000],
    # synthetic_clustered: 0.99 at search_k~5000
    'synthetic_clustered': [1000, 2000, 5000, 10000, 20000, 40000],
    # synthetic_uniform: Never reaches 0.99 (max 0.96 at 80000)
    'synthetic_uniform': [10000, 20000, 40000, 80000, 160000, 320000],
    # sift1m: 0.99 at search_k~40000
    'sift1m': [10000, 20000, 40000, 80000, 120000, 200000],
    'glove': [10000, 20000, 40000, 80000, 120000, 200000],
}


def get_method_configs(method: str, dataset: str):
    if method == 'des_knn_pca':
        taus = DES_KNN_PCA_TAUS.get(dataset, [100, 200, 400, 800, 1200])
        return [
            {'tolerance': t, 'confidence': 0.99, 'max_cv': 0.3, 'n_components': 32}
            for t in taus
        ]
    if method == 'des_knn_guarantee':
        taus = DES_KNN_GUARANTEE_TAUS.get(dataset, [1, 2, 5, 10, 20])
        return [{'tolerance': t, 'confidence': 0.99} for t in taus]
    if method == 'hnsw':
        efs = HNSW_EF_SWEEPS.get(dataset, [10, 20, 50, 100, 200])
        return [
            {'M': 16, 'ef_construction': 200, 'ef': ef}
            for ef in efs
        ]
    if method == 'annoy':
        search_ks = ANNOY_SEARCHK_SWEEPS.get(dataset, [500, 1000, 2000, 5000])
        return [
            {'n_trees': 50, 'search_k': search_k}
            for search_k in search_ks
        ]
    return [{}]

def summarize_seed_result(res, method, log_per_query):
    metrics = res['metrics']
    per_query = res['per_query']
    query_times = np.array(per_query['query_times'], dtype=np.float32)

    expected_misses = per_query.get('expected_misses')
    expected_misses_mean = None
    expected_misses_median = None
    if expected_misses:
        expected_misses_arr = np.array(expected_misses, dtype=np.float32)
        expected_misses_mean = float(np.mean(expected_misses_arr))
        expected_misses_median = float(np.median(expected_misses_arr))

    summary = {
        'seed': res['random_seed'],
        'recall_mean': metrics['recall']['mean'],
        'recall_std': metrics['recall']['std'],
        'speedup_mean': metrics['speedup']['mean'],
        'speedup_std': metrics['speedup']['std'],
        'scan_ratio_mean': metrics['scan_ratio']['mean'],
        'scan_ratio_std': metrics['scan_ratio']['std'],
        'dist_ratio_mean': metrics['dist_ratio']['mean'],
        'dist_ratio_std': metrics['dist_ratio']['std'],
        'build_time': res['build_time'],
        'mean_exact_time_ms': res['mean_exact_time'] * 1000.0,
        'query_time_ms_mean': float(np.mean(query_times) * 1000.0),
        'query_time_ms_std': float(np.std(query_times) * 1000.0),
        'expected_misses_mean': expected_misses_mean,
        'expected_misses_median': expected_misses_median,
    }

    if log_per_query:
        summary['per_query'] = {
            'recalls': per_query['recalls'],
            'query_times_ms': (query_times * 1000.0).tolist(),
            'scan_ratios': per_query['scan_ratios'],
        }
        if expected_misses:
            summary['per_query']['expected_misses'] = expected_misses

    return summary


def aggregate_seed_summaries(seed_summaries):
    def _mean_std(key):
        values = [s[key] for s in seed_summaries if s[key] is not None]
        if not values:
            return None, None
        return float(np.mean(values)), float(np.std(values))

    recall_mean, recall_std = _mean_std('recall_mean')
    speedup_mean, speedup_std = _mean_std('speedup_mean')
    query_time_ms_mean, query_time_ms_std = _mean_std('query_time_ms_mean')
    scan_ratio_mean, scan_ratio_std = _mean_std('scan_ratio_mean')
    dist_ratio_mean, dist_ratio_std = _mean_std('dist_ratio_mean')
    build_time_mean, build_time_std = _mean_std('build_time')

    expected_misses_vals = [
        s['expected_misses_mean']
        for s in seed_summaries
        if s['expected_misses_mean'] is not None
    ]
    expected_misses_mean = float(np.mean(expected_misses_vals)) if expected_misses_vals else None
    expected_misses_median = float(np.median(expected_misses_vals)) if expected_misses_vals else None

    return {
        'recall_mean': recall_mean,
        'recall_std': recall_std,
        'speedup_mean': speedup_mean,
        'speedup_std': speedup_std,
        'query_time_ms_mean': query_time_ms_mean,
        'query_time_ms_std': query_time_ms_std,
        'scan_ratio_mean': scan_ratio_mean,
        'scan_ratio_std': scan_ratio_std,
        'dist_ratio_mean': dist_ratio_mean,
        'dist_ratio_std': dist_ratio_std,
        'build_time_mean': build_time_mean,
        'build_time_std': build_time_std,
        'expected_misses_mean': expected_misses_mean,
        'expected_misses_median': expected_misses_median,
        'n_seeds': len(seed_summaries),
    }


def run_paper_suite(
    output_dir='results/paper',
    n_queries=1000,
    log_per_query=True,
    seeds=None,
    datasets=None
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS if seeds is None else seeds
    datasets = DATASETS if datasets is None else datasets

    all_results = {
        'meta': {
            'n_queries': n_queries,
            'seeds': seeds,
            'k_values': K_VALUES,
            'datasets': datasets,
            'methods': METHODS,
        },
        'results': {}
    }
    
    for dataset in datasets:
        dataset_params = DATASET_PARAMS.get(dataset, {})
        # Skip if dataset not found (simple check)
        try:
            from src.utils.data_loader import DataLoader
            DataLoader().load(dataset, **dataset_params)
        except Exception:
            print(f"Skipping {dataset} (not found)")
            continue

        all_results['results'][dataset] = {}
        
        for method in METHODS:
            all_results['results'][dataset][method] = {}
            configs = get_method_configs(method, dataset)
            
            for k in K_VALUES:
                all_results['results'][dataset][method][k] = []
                
                for config in configs:
                    print(f"Running {dataset} | {method} | k={k} | {config}")
                    
                    # Aggregate across seeds
                    seed_summaries = []
                    for seed in seeds:
                        try:
                            res = run_experiment(
                                dataset_name=dataset,
                                method_name=method,
                                k=k,
                                n_queries=n_queries,
                                method_params=config,
                                dataset_params=dataset_params,
                                random_seed=seed,
                                verbose=False
                            )
                            summary = summarize_seed_result(
                                res,
                                method=method,
                                log_per_query=log_per_query and method.startswith('des_knn')
                            )
                            seed_summaries.append(summary)
                        except Exception as e:
                            print(f"  Error: {e}")
                    
                    if not seed_summaries:
                        continue

                    aggregate = aggregate_seed_summaries(seed_summaries)
                    entry = {
                        'params': config,
                        'aggregate': aggregate,
                        'seeds': seed_summaries
                    }
                    all_results['results'][dataset][method][k].append(entry)
                    print(
                        f"  -> Recall: {aggregate['recall_mean']:.4f}, "
                        f"Time: {aggregate['query_time_ms_mean']:.2f} ms"
                    )

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(output_path / f'paper_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nSaved to {output_path / f'paper_results_{timestamp}.json'}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run paper experiment sweeps')
    parser.add_argument('--output_dir', type=str, default='results/paper',
                        help='Directory to save results')
    parser.add_argument('--n_queries', type=int, default=1000,
                        help='Number of queries per run')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        choices=DATASETS, metavar='DATASET',
                        help='Datasets to run (default: all)')
    parser.set_defaults(log_per_query=True)
    parser.add_argument('--no_log_per_query', dest='log_per_query', action='store_false',
                        help='Disable per-query logging')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seeds (e.g., "0,1,2")')
    args = parser.parse_args()

    seeds = None
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]

    run_paper_suite(
        output_dir=args.output_dir,
        n_queries=args.n_queries,
        log_per_query=args.log_per_query,
        seeds=seeds,
        datasets=args.datasets
    )
