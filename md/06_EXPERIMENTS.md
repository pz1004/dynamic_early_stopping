# Experiment Runner and Analysis

## Overview

This document specifies the experiment running scripts and analysis pipeline.

---

## 1. Single Experiment Runner

### File: `experiments/run_single_experiment.py`

```python
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
        'kdtree': lambda: KDTreeKNN(X, leaf_size=params.get('leaf_size', 30)),
        'sklearn_kdtree': lambda: SklearnKDTreeKNN(X, leaf_size=params.get('leaf_size', 30)),
        'balltree': lambda: BallTreeKNN(X, leaf_size=params.get('leaf_size', 30)),
        'sklearn_balltree': lambda: SklearnBallTreeKNN(X, leaf_size=params.get('leaf_size', 30)),
        'lsh': lambda: LSHKNN(
            X, 
            n_tables=params.get('n_tables', 20),
            n_bits=params.get('n_bits', 12),
            n_probes=params.get('n_probes', 3)
        ),
        'annoy': lambda: AnnoyKNN(
            X,
            n_trees=params.get('n_trees', 50),
            search_k=params.get('search_k', None)
        ),
        'hnsw': lambda: HNSWKNN(
            X,
            M=params.get('M', 16),
            ef_construction=params.get('ef_construction', 200),
            ef=params.get('ef', 50)
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
            adaptive_alpha=params.get('adaptive_alpha', True)
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
    verbose: bool = True
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
    
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        
        if method_name == 'des_knn':
            neighbors, distances, dist_count, stats = searcher.query(q, k, return_stats=True)
            all_stopping_points.append(stats['stopping_point'])
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
        print(f"  Recall@{k}: {metrics['recall']['mean']:.4f} ± {metrics['recall']['std']:.4f}")
        print(f"  Speedup: {metrics['speedup']['mean']:.2f}× ± {metrics['speedup']['std']:.2f}")
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
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    
    # Method-specific parameters
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='DES-kNN confidence level')
    parser.add_argument('--window_size', type=int, default=100,
                       help='DES-kNN window size')
    parser.add_argument('--n_tables', type=int, default=20,
                       help='LSH number of tables')
    parser.add_argument('--n_trees', type=int, default=50,
                       help='Annoy number of trees')
    parser.add_argument('--ef', type=int, default=50,
                       help='HNSW search ef parameter')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect method parameters
    method_params = {
        'alpha': args.alpha,
        'window_size': args.window_size,
        'n_tables': args.n_tables,
        'n_trees': args.n_trees,
        'ef': args.ef
    }
    
    # Run experiment(s)
    if args.method == 'all':
        methods = ['exact', 'sklearn_kdtree', 'sklearn_balltree', 
                  'lsh', 'annoy', 'hnsw', 'des_knn']
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
            verbose=True
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
```

---

## 2. Full Experiment Suite

### File: `experiments/run_all_experiments.py`

```python
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

SEEDS = [42, 123, 456, 789, 1024]

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
                                  f"Speedup: {results['metrics']['speedup']['mean']:.2f}×")
                            
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
```

---

## 3. Results Analysis

### File: `experiments/analyze_results.py`

```python
#!/usr/bin/env python3
"""
Analyze experiment results and generate tables/figures.

Usage:
    python analyze_results.py --results_dir results/
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    plot_recall_vs_speedup,
    plot_stopping_point_distribution,
    plot_query_difficulty_analysis,
    create_results_table
)


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all result files from directory."""
    results_path = Path(results_dir)
    all_results = {}
    
    for json_file in results_path.glob('*.json'):
        with open(json_file) as f:
            data = json.load(f)
            all_results[json_file.stem] = data
    
    return all_results


def aggregate_across_seeds(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across multiple seeds."""
    if not results or all('error' in r for r in results):
        return None
    
    valid_results = [r for r in results if 'error' not in r]
    
    return {
        'recall_mean': np.mean([r['recall_mean'] for r in valid_results]),
        'recall_std': np.std([r['recall_mean'] for r in valid_results]),
        'speedup_mean': np.mean([r['speedup_mean'] for r in valid_results]),
        'speedup_std': np.std([r['speedup_mean'] for r in valid_results]),
        'dist_ratio_mean': np.mean([r['dist_ratio'] for r in valid_results]),
        'build_time_mean': np.mean([r['build_time'] for r in valid_results]),
        'failure_rate_mean': np.mean([r['failure_rate_99'] for r in valid_results]),
        'n_seeds': len(valid_results)
    }


def create_summary_table(results: Dict[str, Any], k: int = 10) -> pd.DataFrame:
    """
    Create summary DataFrame for a specific k value.
    """
    rows = []
    
    for dataset in results:
        for method in results[dataset]:
            if str(k) not in results[dataset][method]:
                continue
            
            for config, seed_results in results[dataset][method][str(k)].items():
                agg = aggregate_across_seeds(seed_results)
                if agg is None:
                    continue
                
                rows.append({
                    'Dataset': dataset,
                    'Method': method,
                    'Config': config if config != 'default' else '-',
                    'Recall': f"{agg['recall_mean']:.4f}±{agg['recall_std']:.4f}",
                    'Speedup': f"{agg['speedup_mean']:.2f}×",
                    'Dist Ratio': f"{agg['dist_ratio_mean']:.4f}",
                    'Build Time': f"{agg['build_time_mean']:.2f}s",
                    'Failure Rate': f"{agg['failure_rate_mean']*100:.1f}%"
                })
    
    return pd.DataFrame(rows)


def generate_latex_table(df: pd.DataFrame, caption: str = "") -> str:
    """Generate LaTeX table from DataFrame."""
    latex = df.to_latex(index=False, escape=False)
    
    if caption:
        latex = latex.replace(
            r'\begin{tabular}',
            f'\\caption{{{caption}}}\n\\begin{{tabular}}'
        )
    
    return latex


def compare_des_knn_configs(results: Dict[str, Any], dataset: str, k: int = 10) -> pd.DataFrame:
    """Compare different DES-kNN configurations."""
    if 'des_knn' not in results.get(dataset, {}):
        return None
    
    rows = []
    for config, seed_results in results[dataset]['des_knn'].get(str(k), {}).items():
        agg = aggregate_across_seeds(seed_results)
        if agg:
            rows.append({
                'Configuration': config,
                'Recall': agg['recall_mean'],
                'Speedup': agg['speedup_mean'],
                'Failure Rate': agg['failure_rate_mean']
            })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Directory for output figures')
    parser.add_argument('--k', type=int, default=10,
                       help='Value of k for analysis')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    all_results = load_results(args.results_dir)
    
    if not all_results:
        print("No results found!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each result file
    for result_name, results in all_results.items():
        print(f"\nAnalyzing: {result_name}")
        
        # Create summary table
        summary_df = create_summary_table(results, k=args.k)
        if not summary_df.empty:
            print("\nSummary Table:")
            print(summary_df.to_string())
            
            # Save to CSV
            summary_df.to_csv(output_dir / f'{result_name}_summary_k{args.k}.csv', index=False)
            
            # Generate LaTeX
            latex = generate_latex_table(summary_df, f"Results for k={args.k}")
            with open(output_dir / f'{result_name}_summary_k{args.k}.tex', 'w') as f:
                f.write(latex)
        
        # DES-kNN configuration comparison
        for dataset in results:
            config_df = compare_des_knn_configs(results, dataset, k=args.k)
            if config_df is not None and not config_df.empty:
                print(f"\nDES-kNN Configurations for {dataset}:")
                print(config_df.to_string())
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
```

---

## 4. Configuration Files

### File: `experiments/configs/default_config.yaml`

```yaml
# Default experiment configuration

# Datasets to evaluate
datasets:
  - mnist
  - fashion_mnist
  - synthetic_clustered
  - synthetic_uniform

# Methods to compare
methods:
  - exact
  - sklearn_kdtree
  - sklearn_balltree
  - lsh
  - annoy
  - hnsw
  - des_knn

# k values to test
k_values:
  - 1
  - 5
  - 10
  - 20
  - 50

# Random seeds for multiple runs
seeds:
  - 42
  - 123
  - 456
  - 789
  - 1024

# Number of queries per experiment
n_queries: 1000

# DES-kNN parameters
des_knn:
  alpha: 0.01
  window_size: 100
  min_samples: null  # Uses max(2k, 50) if null
  adaptive_alpha: true
  use_weibull: true

# LSH parameters
lsh:
  n_tables: 20
  n_bits: 12
  n_probes: 3

# Annoy parameters
annoy:
  n_trees: 50
  search_k: null  # Uses n_trees * k * 10 if null

# HNSW parameters
hnsw:
  M: 16
  ef_construction: 200
  ef: 50

# FAISS IVF parameters
faiss_ivf:
  nlist: 100
  nprobe: 10

# Tree parameters
tree:
  leaf_size: 30

# Output settings
output:
  dir: results
  save_per_query: true
  save_figures: true
```

### File: `experiments/configs/quick_test.yaml`

```yaml
# Quick test configuration for debugging

datasets:
  - synthetic_uniform

methods:
  - exact
  - des_knn

k_values:
  - 10

seeds:
  - 42

n_queries: 100

des_knn:
  alpha: 0.01
  window_size: 100
  adaptive_alpha: true

output:
  dir: results/quick_test
  save_per_query: false
  save_figures: false
```

---

## 5. Requirements File

### File: `requirements.txt`

```
# Core dependencies
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0

# Data handling
pandas>=1.3.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0

# Configuration
pyyaml>=6.0

# Progress bars
tqdm>=4.62.0

# Deep learning (for CIFAR features)
torch>=1.9.0
torchvision>=0.10.0

# Optional: ANN libraries (install if available)
# annoy>=1.17.0
# hnswlib>=0.6.0
# faiss-cpu>=1.7.0

# Testing
pytest>=6.0.0

# Parallel processing
joblib>=1.0.0
```

---

## 6. Setup Script

### File: `setup.py`

```python
from setuptools import setup, find_packages

setup(
    name='des_knn',
    version='0.1.0',
    description='Dynamic Early Stopping for k-Nearest Neighbors Search',
    author='Your Name',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'pandas>=1.3.0',
        'matplotlib>=3.4.0',
        'pyyaml>=6.0',
        'tqdm>=4.62.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
        ],
        'deep': [
            'torch>=1.9.0',
            'torchvision>=0.10.0',
        ],
        'ann': [
            'annoy>=1.17.0',
            'hnswlib>=0.6.0',
        ]
    }
)
```
