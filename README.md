# DES-kNN: Dynamic Early Stopping for k-Nearest Neighbors Search

This project implements a query-adaptive early stopping criterion for k-NN search that monitors the convergence of the k-th nearest distance during search and terminates early when high confidence is reached that the true k-NN set has been found.

## Key Innovation

Uses online statistical bounds to estimate the probability that remaining unsearched points could enter the k-NN set, enabling query-dependent computation allocation.

## Project Structure

```
des_knn/
├── src/
│   ├── __init__.py
│   ├── des_knn.py              # Main DES-kNN implementation
│   ├── statistics.py           # Online statistical estimators
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── exact_brute_force.py
│   │   ├── kdtree.py
│   │   ├── balltree.py
│   │   ├── lsh.py
│   │   ├── annoy_wrapper.py
│   │   ├── hnsw.py
│   │   └── faiss_ivf.py
│   └── utils/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── metrics.py
│       ├── visualization.py
│       └── heap.py
├── experiments/
│   ├── run_all_experiments.py
│   ├── run_single_experiment.py
│   ├── analyze_results.py
│   └── configs/
│       ├── default_config.yaml
│       ├── quick_test.yaml
│       ├── mnist.yaml
│       ├── fashion_mnist.yaml
│       ├── cifar10.yaml
│       └── synthetic.yaml
├── results/
├── figures/
├── tests/
│   ├── test_des_knn.py
│   ├── test_baselines.py
│   └── test_statistics.py
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

```python
import numpy as np
from src.des_knn import DESKNNSearcher

# Create sample data
np.random.seed(42)
X = np.random.randn(10000, 128).astype(np.float32)
q = np.random.randn(128).astype(np.float32)

# Initialize searcher
searcher = DESKNNSearcher(
    X,
    alpha=0.01,
    window_size=100,
    adaptive_alpha=True,
    distance_metric='euclidean',
    random_state=42
)

# Query
k = 10
neighbors, distances, dist_count = searcher.query(q, k)

print(f"Found {k} neighbors")
print(f"Distance computations: {dist_count} / {len(X)} ({100*dist_count/len(X):.1f}%)")
print(f"Speedup: {len(X) / dist_count:.2f}x")
```

## Running Experiments

```bash
# Run single experiment
python experiments/run_single_experiment.py --dataset mnist --k 10

# Run all experiments
python experiments/run_all_experiments.py

# Quick test
python experiments/run_all_experiments.py --quick

# Analyze results
python experiments/analyze_results.py --results_dir results/
```

## Expected Results

- **Recall@k**: >= 99% with default alpha=0.01
- **Speedup**: 2-5x over exact search
- **Adaptive behavior**: Easy queries stop early, hard queries continue longer

## Algorithm Overview

DES-kNN uses three complementary stopping criteria:

1. **Exceedance Probability**: Estimates the probability that remaining unsearched points could enter the k-NN set using Weibull distribution fitting and Hoeffding bounds.

2. **Update Pattern Confidence**: Models inter-update gaps as geometric random variables to estimate confidence that k-NN set is complete.

3. **Gap Ratio**: Tracks the fraction of search completed since the last k-NN update.

The algorithm terminates when at least two of these criteria are satisfied.

## Baselines

The project includes implementations of:

- **Exact Methods**: Brute Force
- **Tree-Based**: KD-Tree, Ball Tree (with sklearn wrappers)
- **Approximate NN**: LSH, Annoy, HNSW, FAISS IVF

## License

MIT
