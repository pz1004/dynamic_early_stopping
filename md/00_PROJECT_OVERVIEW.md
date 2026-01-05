# Dynamic Early Stopping for k-Nearest Neighbors Search (DES-kNN)

## Project Overview

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
│   └── .gitkeep
├── figures/
│   └── .gitkeep
├── tests/
│   ├── test_des_knn.py
│   ├── test_baselines.py
│   └── test_statistics.py
├── requirements.txt
├── setup.py
└── README.md
```

## Implementation Order

1. **Phase 1: Core Utilities** (30 min)
   - `src/utils/heap.py` - Max heap implementation
   - `src/utils/data_loader.py` - Dataset loading
   - `src/utils/metrics.py` - Evaluation metrics

2. **Phase 2: Statistics Module** (30 min)
   - `src/statistics.py` - Online statistical estimators

3. **Phase 3: Main Algorithm** (1 hour)
   - `src/des_knn.py` - Core DES-kNN implementation

4. **Phase 4: Baselines** (1.5 hours)
   - All baseline implementations

5. **Phase 5: Experiments** (1 hour)
   - Experiment runners and analysis scripts

## Dependencies

```
numpy>=1.21.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
pandas>=1.3.0
pyyaml>=6.0
tqdm>=4.62.0
torch>=1.9.0  # For CIFAR-10 feature extraction
torchvision>=0.10.0
```

## Expected Results

- **Recall@k**: ≥ 99% with default α=0.01
- **Speedup**: 2-5× over exact search
- **Adaptive behavior**: Easy queries stop early, hard queries continue longer

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run single experiment
python experiments/run_single_experiment.py --dataset mnist --k 10

# Run all experiments
python experiments/run_all_experiments.py

# Analyze results
python experiments/analyze_results.py --results_dir results/
```
