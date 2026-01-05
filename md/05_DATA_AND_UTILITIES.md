# Data Loading and Utility Modules

## Overview

This document specifies the data loading utilities and helper functions for the DES-kNN experiments.

---

## 1. Data Loader

### File: `src/utils/data_loader.py`

```python
"""
Dataset loading utilities for k-NN experiments.

Provides unified interface for loading:
- MNIST
- Fashion-MNIST  
- CIFAR-10 (with feature extraction)
- Synthetic datasets
- GloVe word vectors
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import gzip
import struct
import os


class DataLoader:
    """
    Unified data loader for all experiment datasets.
    
    Parameters
    ----------
    data_dir : str, default='./data'
        Directory to store/load datasets.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(self, data_dir: str = './data', random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
    
    def load(
        self,
        dataset_name: str,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load a dataset.
        
        Parameters
        ----------
        dataset_name : str
            Name of dataset: 'mnist', 'fashion_mnist', 'cifar10', 
            'synthetic_clustered', 'synthetic_uniform'.
        **kwargs : dict
            Dataset-specific parameters.
        
        Returns
        -------
        X_train : np.ndarray
            Training data (database to search).
        X_test : np.ndarray
            Test data (query points).
        y_train : np.ndarray
            Training labels (if available).
        y_test : np.ndarray
            Test labels (if available).
        """
        loaders = {
            'mnist': self._load_mnist,
            'fashion_mnist': self._load_fashion_mnist,
            'cifar10': self._load_cifar10,
            'cifar10_resnet': self._load_cifar10_features,
            'synthetic_clustered': self._load_synthetic_clustered,
            'synthetic_uniform': self._load_synthetic_uniform,
            'glove': self._load_glove,
        }
        
        if dataset_name not in loaders:
            raise ValueError(f"Unknown dataset: {dataset_name}. "
                           f"Available: {list(loaders.keys())}")
        
        return loaders[dataset_name](**kwargs)
    
    def _load_mnist(
        self,
        normalize: bool = True,
        flatten: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load MNIST dataset.
        
        MNIST: 70,000 handwritten digits (28x28 grayscale images)
        - 60,000 training, 10,000 test
        - 784 dimensions when flattened
        
        Downloads from: http://yann.lecun.com/exdb/mnist/
        But since no internet, assumes data is pre-downloaded or uses
        sklearn/torchvision fallback.
        """
        try:
            # Try sklearn first (easiest)
            from sklearn.datasets import fetch_openml
            
            mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
            X, y = mnist.data, mnist.target.astype(np.int32)
            
        except Exception:
            try:
                # Try torchvision
                import torchvision
                import torchvision.transforms as transforms
                
                transform = transforms.ToTensor()
                
                train_dataset = torchvision.datasets.MNIST(
                    root=str(self.data_dir), train=True, download=False, transform=transform
                )
                test_dataset = torchvision.datasets.MNIST(
                    root=str(self.data_dir), train=False, download=False, transform=transform
                )
                
                X_train = train_dataset.data.numpy().reshape(-1, 784)
                X_test = test_dataset.data.numpy().reshape(-1, 784)
                y_train = train_dataset.targets.numpy()
                y_test = test_dataset.targets.numpy()
                
                X = np.vstack([X_train, X_test])
                y = np.concatenate([y_train, y_test])
                
            except Exception:
                # Generate synthetic MNIST-like data for testing
                print("Warning: Could not load MNIST. Using synthetic data.")
                return self._load_synthetic_uniform(n=70000, d=784)
        
        # Split
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        # Convert to float32
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        
        return X_train, X_test, y_train, y_test
    
    def _load_fashion_mnist(
        self,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load Fashion-MNIST dataset.
        
        Fashion-MNIST: 70,000 fashion items (28x28 grayscale images)
        - Same structure as MNIST
        - 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, 
          Sandal, Shirt, Sneaker, Bag, Ankle boot
        """
        try:
            from sklearn.datasets import fetch_openml
            
            fmnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
            X, y = fmnist.data, fmnist.target.astype(np.int32)
            
        except Exception:
            try:
                import torchvision
                import torchvision.transforms as transforms
                
                transform = transforms.ToTensor()
                
                train_dataset = torchvision.datasets.FashionMNIST(
                    root=str(self.data_dir), train=True, download=False, transform=transform
                )
                test_dataset = torchvision.datasets.FashionMNIST(
                    root=str(self.data_dir), train=False, download=False, transform=transform
                )
                
                X_train = train_dataset.data.numpy().reshape(-1, 784)
                X_test = test_dataset.data.numpy().reshape(-1, 784)
                y_train = train_dataset.targets.numpy()
                y_test = test_dataset.targets.numpy()
                
                X = np.vstack([X_train, X_test])
                y = np.concatenate([y_train, y_test])
                
            except Exception:
                print("Warning: Could not load Fashion-MNIST. Using synthetic data.")
                return self._load_synthetic_uniform(n=70000, d=784)
        
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        
        return X_train, X_test, y_train, y_test
    
    def _load_cifar10(
        self,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CIFAR-10 raw pixels.
        
        CIFAR-10: 60,000 color images (32x32x3)
        - 50,000 training, 10,000 test
        - 3072 dimensions when flattened
        """
        try:
            import torchvision
            import torchvision.transforms as transforms
            
            transform = transforms.ToTensor()
            
            train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=True, download=False, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=False, download=False, transform=transform
            )
            
            X_train = np.array([img.numpy().flatten() for img, _ in train_dataset])
            X_test = np.array([img.numpy().flatten() for img, _ in test_dataset])
            y_train = np.array(train_dataset.targets)
            y_test = np.array(test_dataset.targets)
            
        except Exception:
            print("Warning: Could not load CIFAR-10. Using synthetic data.")
            return self._load_synthetic_uniform(n=60000, d=3072)
        
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        
        if normalize:
            X_train = X_train / 255.0
            X_test = X_test / 255.0
        
        return X_train, X_test, y_train, y_test
    
    def _load_cifar10_features(
        self,
        model_name: str = 'resnet18'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CIFAR-10 with features extracted from pretrained ResNet.
        
        Parameters
        ----------
        model_name : str
            Model to use: 'resnet18' (512 dims) or 'resnet50' (2048 dims).
        """
        cache_path = self.data_dir / f'cifar10_{model_name}_features.npz'
        
        if cache_path.exists():
            # Load from cache
            data = np.load(cache_path)
            return data['X_train'], data['X_test'], data['y_train'], data['y_test']
        
        try:
            import torch
            import torchvision
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader
            
            # Setup transforms (ResNet expects 224x224 but we'll resize)
            transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            
            train_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=True, download=False, transform=transform
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=str(self.data_dir), train=False, download=False, transform=transform
            )
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            # Load pretrained model
            if model_name == 'resnet18':
                model = torchvision.models.resnet18(pretrained=True)
                feature_dim = 512
            else:
                model = torchvision.models.resnet50(pretrained=True)
                feature_dim = 2048
            
            # Remove final classification layer
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            # Extract features
            def extract_features(loader):
                features = []
                labels = []
                with torch.no_grad():
                    for images, targets in loader:
                        images = images.to(device)
                        feats = model(images).squeeze()
                        features.append(feats.cpu().numpy())
                        labels.append(targets.numpy())
                return np.vstack(features), np.concatenate(labels)
            
            X_train, y_train = extract_features(train_loader)
            X_test, y_test = extract_features(test_loader)
            
            # Cache for future use
            np.savez(cache_path, X_train=X_train, X_test=X_test, 
                    y_train=y_train, y_test=y_test)
            
        except Exception as e:
            print(f"Warning: Could not extract CIFAR-10 features: {e}")
            print("Using synthetic data.")
            feature_dim = 512 if model_name == 'resnet18' else 2048
            return self._load_synthetic_uniform(n=60000, d=feature_dim)
        
        return X_train.astype(np.float32), X_test.astype(np.float32), y_train, y_test
    
    def _load_synthetic_clustered(
        self,
        n: int = 100000,
        d: int = 128,
        n_clusters: int = 50,
        cluster_std: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic clustered data using Gaussian mixture.
        
        Parameters
        ----------
        n : int
            Total number of samples.
        d : int
            Dimensionality.
        n_clusters : int
            Number of Gaussian clusters.
        cluster_std : float
            Standard deviation within clusters.
        """
        # Generate cluster centers
        centers = self.rng.standard_normal((n_clusters, d)) * 10
        
        # Assign points to clusters
        labels = self.rng.integers(0, n_clusters, n)
        
        # Generate points around cluster centers
        X = np.zeros((n, d), dtype=np.float32)
        for i in range(n):
            X[i] = centers[labels[i]] + self.rng.standard_normal(d) * cluster_std
        
        # Split 90-10
        n_train = int(0.9 * n)
        perm = self.rng.permutation(n)
        
        X_train = X[perm[:n_train]]
        X_test = X[perm[n_train:]]
        y_train = labels[perm[:n_train]]
        y_test = labels[perm[n_train:]]
        
        return X_train, X_test, y_train, y_test
    
    def _load_synthetic_uniform(
        self,
        n: int = 100000,
        d: int = 128
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic uniformly distributed data.
        
        This represents "hard" case where there's no clustering structure.
        """
        X = self.rng.standard_normal((n, d)).astype(np.float32)
        y = np.zeros(n, dtype=np.int32)  # No meaningful labels
        
        n_train = int(0.9 * n)
        
        return X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    
    def _load_glove(
        self,
        dim: int = 50,
        max_words: int = 400000
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load GloVe word vectors.
        
        Parameters
        ----------
        dim : int
            Embedding dimension (50, 100, 200, or 300).
        max_words : int
            Maximum number of words to load.
        """
        glove_path = self.data_dir / f'glove.6B.{dim}d.txt'
        
        if not glove_path.exists():
            print(f"Warning: GloVe file not found at {glove_path}")
            print("Using synthetic data.")
            return self._load_synthetic_uniform(n=max_words, d=dim)
        
        vectors = []
        words = []
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= max_words:
                    break
                parts = line.strip().split()
                word = parts[0]
                vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                if len(vector) == dim:
                    words.append(word)
                    vectors.append(vector)
        
        X = np.array(vectors)
        y = np.arange(len(vectors))  # Word indices as labels
        
        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (norms + 1e-10)
        
        n_train = int(0.9 * len(X))
        
        return X[:n_train], X[n_train:], y[:n_train], y[n_train:]
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get metadata about a dataset."""
        info = {
            'mnist': {
                'n_train': 60000,
                'n_test': 10000,
                'dims': 784,
                'n_classes': 10,
                'description': 'Handwritten digits 0-9'
            },
            'fashion_mnist': {
                'n_train': 60000,
                'n_test': 10000,
                'dims': 784,
                'n_classes': 10,
                'description': 'Fashion items (clothing, shoes, bags)'
            },
            'cifar10': {
                'n_train': 50000,
                'n_test': 10000,
                'dims': 3072,
                'n_classes': 10,
                'description': 'Color images of objects'
            },
            'cifar10_resnet': {
                'n_train': 50000,
                'n_test': 10000,
                'dims': 512,
                'n_classes': 10,
                'description': 'CIFAR-10 with ResNet18 features'
            },
            'synthetic_clustered': {
                'n_train': 90000,
                'n_test': 10000,
                'dims': 128,
                'n_classes': 50,
                'description': 'Gaussian mixture clusters'
            },
            'synthetic_uniform': {
                'n_train': 90000,
                'n_test': 10000,
                'dims': 128,
                'n_classes': 0,
                'description': 'Uniform random data (hard case)'
            }
        }
        return info.get(dataset_name, {})
```

---

## 2. Metrics Module

### File: `src/utils/metrics.py`

```python
"""
Evaluation metrics for k-NN experiments.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from scipy import stats


def recall_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Recall@k for a single query.
    
    Parameters
    ----------
    retrieved : np.ndarray of shape (k,)
        Indices of retrieved neighbors.
    ground_truth : np.ndarray of shape (k,)
        Indices of true k nearest neighbors.
    
    Returns
    -------
    recall : float
        Proportion of true neighbors that were retrieved.
    """
    retrieved_set = set(retrieved)
    truth_set = set(ground_truth)
    
    if len(truth_set) == 0:
        return 1.0
    
    return len(retrieved_set & truth_set) / len(truth_set)


def precision_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Precision@k for a single query.
    """
    retrieved_set = set(retrieved)
    truth_set = set(ground_truth)
    
    if len(retrieved_set) == 0:
        return 0.0
    
    return len(retrieved_set & truth_set) / len(retrieved_set)


def average_precision(
    retrieved: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Average Precision for a single query.
    
    AP = (1/R) * Σ P(k) * rel(k)
    where R is total relevant items, P(k) is precision at rank k,
    and rel(k) is 1 if item at rank k is relevant.
    """
    truth_set = set(ground_truth)
    
    if len(truth_set) == 0:
        return 1.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, idx in enumerate(retrieved):
        if idx in truth_set:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    
    return precision_sum / len(truth_set)


def compute_speedup(
    exact_time: float,
    method_time: float
) -> float:
    """Compute speedup factor."""
    if method_time == 0:
        return float('inf')
    return exact_time / method_time


def compute_distance_ratio(
    dist_count: int,
    total_points: int
) -> float:
    """Compute fraction of distance computations."""
    return dist_count / total_points


def aggregate_metrics(
    all_recalls: List[float],
    all_speedups: List[float],
    all_dist_counts: List[int],
    total_points: int,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple queries.
    
    Parameters
    ----------
    all_recalls : List[float]
        Recall values for each query.
    all_speedups : List[float]
        Speedup values for each query.
    all_dist_counts : List[int]
        Distance computation counts for each query.
    total_points : int
        Total points in database (n).
    confidence : float
        Confidence level for intervals.
    
    Returns
    -------
    metrics : Dict[str, Any]
        Aggregated metrics with statistics.
    """
    all_recalls = np.array(all_recalls)
    all_speedups = np.array(all_speedups)
    all_dist_counts = np.array(all_dist_counts)
    
    # Basic statistics
    metrics = {
        'recall': {
            'mean': np.mean(all_recalls),
            'std': np.std(all_recalls),
            'min': np.min(all_recalls),
            'max': np.max(all_recalls),
            'median': np.median(all_recalls),
        },
        'speedup': {
            'mean': np.mean(all_speedups),
            'std': np.std(all_speedups),
            'min': np.min(all_speedups),
            'max': np.max(all_speedups),
            'median': np.median(all_speedups),
        },
        'dist_ratio': {
            'mean': np.mean(all_dist_counts) / total_points,
            'std': np.std(all_dist_counts) / total_points,
        }
    }
    
    # Percentiles
    for p in [10, 25, 75, 90]:
        metrics['recall'][f'p{p}'] = np.percentile(all_recalls, p)
        metrics['speedup'][f'p{p}'] = np.percentile(all_speedups, p)
    
    # Confidence intervals (using t-distribution for small samples)
    n = len(all_recalls)
    if n > 1:
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        recall_se = np.std(all_recalls, ddof=1) / np.sqrt(n)
        metrics['recall']['ci_lower'] = np.mean(all_recalls) - t_value * recall_se
        metrics['recall']['ci_upper'] = np.mean(all_recalls) + t_value * recall_se
        
        speedup_se = np.std(all_speedups, ddof=1) / np.sqrt(n)
        metrics['speedup']['ci_lower'] = np.mean(all_speedups) - t_value * speedup_se
        metrics['speedup']['ci_upper'] = np.mean(all_speedups) + t_value * speedup_se
    
    # Count failures (recall < threshold)
    metrics['failure_rate_95'] = np.mean(all_recalls < 0.95)
    metrics['failure_rate_99'] = np.mean(all_recalls < 0.99)
    
    return metrics


def statistical_test(
    method1_recalls: np.ndarray,
    method2_recalls: np.ndarray,
    test_type: str = 'paired_t'
) -> Dict[str, float]:
    """
    Perform statistical test comparing two methods.
    
    Parameters
    ----------
    method1_recalls : np.ndarray
        Recalls from method 1.
    method2_recalls : np.ndarray
        Recalls from method 2 (same queries).
    test_type : str
        'paired_t' for paired t-test, 'wilcoxon' for Wilcoxon signed-rank.
    
    Returns
    -------
    result : Dict[str, float]
        Test statistic and p-value.
    """
    if test_type == 'paired_t':
        stat, pvalue = stats.ttest_rel(method1_recalls, method2_recalls)
    elif test_type == 'wilcoxon':
        stat, pvalue = stats.wilcoxon(method1_recalls, method2_recalls)
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    return {
        'statistic': stat,
        'pvalue': pvalue,
        'significant_05': pvalue < 0.05,
        'significant_01': pvalue < 0.01
    }


def query_difficulty_analysis(
    queries: np.ndarray,
    database: np.ndarray,
    ground_truth_dists: np.ndarray,
    method_recalls: np.ndarray,
    method_stopping_points: np.ndarray,
    k: int
) -> Dict[str, Any]:
    """
    Analyze how method performance varies with query difficulty.
    
    Parameters
    ----------
    queries : np.ndarray of shape (n_queries, d)
        Query points.
    database : np.ndarray of shape (n, d)
        Database points.
    ground_truth_dists : np.ndarray of shape (n_queries, k)
        Distances to true k-NN.
    method_recalls : np.ndarray of shape (n_queries,)
        Recall values per query.
    method_stopping_points : np.ndarray of shape (n_queries,)
        Fraction of database searched per query.
    k : int
        Number of neighbors.
    
    Returns
    -------
    analysis : Dict[str, Any]
        Analysis results.
    """
    n_queries = len(queries)
    n = len(database)
    
    # Compute query difficulty metrics
    # 1. Average distance to k-th neighbor (lower = easier)
    kth_distances = ground_truth_dists[:, -1]
    
    # 2. Ratio of k-th distance to mean distance in database
    # This normalizes for different distance scales
    sample_indices = np.random.choice(n, min(1000, n), replace=False)
    mean_dists = []
    for q in queries:
        dists = np.sqrt(np.sum((database[sample_indices] - q) ** 2, axis=1))
        mean_dists.append(np.mean(dists))
    mean_dists = np.array(mean_dists)
    
    difficulty_ratio = kth_distances / (mean_dists + 1e-10)
    
    # Partition queries into difficulty bins
    p33 = np.percentile(difficulty_ratio, 33)
    p67 = np.percentile(difficulty_ratio, 67)
    
    easy_mask = difficulty_ratio <= p33
    medium_mask = (difficulty_ratio > p33) & (difficulty_ratio <= p67)
    hard_mask = difficulty_ratio > p67
    
    analysis = {
        'easy_queries': {
            'count': np.sum(easy_mask),
            'avg_recall': np.mean(method_recalls[easy_mask]),
            'avg_stopping_point': np.mean(method_stopping_points[easy_mask]),
        },
        'medium_queries': {
            'count': np.sum(medium_mask),
            'avg_recall': np.mean(method_recalls[medium_mask]),
            'avg_stopping_point': np.mean(method_stopping_points[medium_mask]),
        },
        'hard_queries': {
            'count': np.sum(hard_mask),
            'avg_recall': np.mean(method_recalls[hard_mask]),
            'avg_stopping_point': np.mean(method_stopping_points[hard_mask]),
        },
        'correlation_difficulty_stopping': np.corrcoef(
            difficulty_ratio, method_stopping_points
        )[0, 1],
        'correlation_difficulty_recall': np.corrcoef(
            difficulty_ratio, method_recalls
        )[0, 1]
    }
    
    return analysis
```

---

## 3. Heap Utility

### File: `src/utils/heap.py`

```python
"""
Heap implementations for k-NN search.
"""

import numpy as np
from typing import Tuple, List


class MaxHeap:
    """
    Max heap for maintaining k nearest neighbors.
    
    Uses a max heap so we can quickly check if a new point
    is closer than the current k-th nearest.
    
    Parameters
    ----------
    capacity : int
        Maximum number of elements (k).
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.heap: List[Tuple[float, int]] = []  # (distance, index)
    
    def push(self, distance: float, index: int) -> bool:
        """
        Try to add a new element.
        
        Returns True if element was added, False if rejected.
        """
        if len(self.heap) < self.capacity:
            self._heap_push((distance, index))
            return True
        elif distance < self.heap[0][0]:
            self._heap_replace((distance, index))
            return True
        return False
    
    def peek_max(self) -> Tuple[float, int]:
        """Return the maximum element (k-th nearest)."""
        if not self.heap:
            return (float('inf'), -1)
        return self.heap[0]
    
    def get_sorted(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return sorted arrays of indices and distances."""
        if not self.heap:
            return np.array([]), np.array([])
        
        sorted_items = sorted(self.heap)
        distances = np.array([d for d, _ in sorted_items])
        indices = np.array([i for _, i in sorted_items])
        return indices, distances
    
    def _heap_push(self, item: Tuple[float, int]):
        """Push item onto heap."""
        self.heap.append(item)
        self._sift_up(len(self.heap) - 1)
    
    def _heap_replace(self, item: Tuple[float, int]):
        """Replace root with new item and re-heapify."""
        self.heap[0] = item
        self._sift_down(0)
    
    def _sift_up(self, pos: int):
        """Move item at pos up to maintain heap property."""
        item = self.heap[pos]
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent = self.heap[parent_pos]
            if item[0] > parent[0]:  # Max heap: larger goes up
                self.heap[pos] = parent
                pos = parent_pos
            else:
                break
        self.heap[pos] = item
    
    def _sift_down(self, pos: int):
        """Move item at pos down to maintain heap property."""
        n = len(self.heap)
        item = self.heap[pos]
        child_pos = 2 * pos + 1
        
        while child_pos < n:
            right_pos = child_pos + 1
            if right_pos < n and self.heap[right_pos][0] > self.heap[child_pos][0]:
                child_pos = right_pos
            
            if item[0] < self.heap[child_pos][0]:
                self.heap[pos] = self.heap[child_pos]
                pos = child_pos
                child_pos = 2 * pos + 1
            else:
                break
        
        self.heap[pos] = item
    
    def __len__(self) -> int:
        return len(self.heap)


class MinHeap:
    """Min heap implementation (for priority queues)."""
    
    def __init__(self):
        self.heap: List[Tuple[float, any]] = []
    
    def push(self, priority: float, item: any):
        """Push item with given priority."""
        self.heap.append((priority, item))
        self._sift_up(len(self.heap) - 1)
    
    def pop(self) -> Tuple[float, any]:
        """Pop item with minimum priority."""
        if not self.heap:
            raise IndexError("pop from empty heap")
        
        min_item = self.heap[0]
        last = self.heap.pop()
        
        if self.heap:
            self.heap[0] = last
            self._sift_down(0)
        
        return min_item
    
    def _sift_up(self, pos: int):
        item = self.heap[pos]
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            if item[0] < self.heap[parent_pos][0]:
                self.heap[pos] = self.heap[parent_pos]
                pos = parent_pos
            else:
                break
        self.heap[pos] = item
    
    def _sift_down(self, pos: int):
        n = len(self.heap)
        item = self.heap[pos]
        child_pos = 2 * pos + 1
        
        while child_pos < n:
            right_pos = child_pos + 1
            if right_pos < n and self.heap[right_pos][0] < self.heap[child_pos][0]:
                child_pos = right_pos
            
            if item[0] > self.heap[child_pos][0]:
                self.heap[pos] = self.heap[child_pos]
                pos = child_pos
                child_pos = 2 * pos + 1
            else:
                break
        
        self.heap[pos] = item
    
    def __len__(self) -> int:
        return len(self.heap)
    
    def __bool__(self) -> bool:
        return bool(self.heap)
```

---

## 4. Visualization Module

### File: `src/utils/visualization.py`

```python
"""
Visualization utilities for experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path


def plot_recall_vs_speedup(
    results: Dict[str, Dict[str, Any]],
    title: str = "Recall vs Speedup Trade-off",
    save_path: Optional[str] = None
):
    """
    Plot recall vs speedup for multiple methods.
    
    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary: method_name -> metrics dict.
    title : str
        Plot title.
    save_path : str or None
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for method_name, metrics in results.items():
        recall = metrics['recall']['mean']
        speedup = metrics['speedup']['mean']
        
        # Error bars if available
        recall_err = metrics['recall'].get('std', 0)
        speedup_err = metrics['speedup'].get('std', 0)
        
        ax.errorbar(
            speedup, recall,
            xerr=speedup_err, yerr=recall_err,
            fmt='o', markersize=8, capsize=5,
            label=method_name
        )
    
    ax.set_xlabel('Speedup (×)', fontsize=12)
    ax.set_ylabel('Recall@k', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 99% recall
    ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='99% recall')
    
    ax.set_ylim([0.8, 1.02])
    ax.set_xlim([0, None])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_stopping_point_distribution(
    stopping_points: np.ndarray,
    method_name: str = "DES-kNN",
    save_path: Optional[str] = None
):
    """
    Plot distribution of stopping points (fraction of data searched).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(stopping_points * 100, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(stopping_points) * 100, color='r', linestyle='--',
               label=f'Mean: {np.mean(stopping_points)*100:.1f}%')
    ax.axvline(np.median(stopping_points) * 100, color='g', linestyle=':',
               label=f'Median: {np.median(stopping_points)*100:.1f}%')
    
    ax.set_xlabel('Stopping Point (% of data searched)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{method_name} - Stopping Point Distribution', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_query_difficulty_analysis(
    difficulty_scores: np.ndarray,
    recalls: np.ndarray,
    stopping_points: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot performance vs query difficulty.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Recall vs difficulty
    ax1 = axes[0]
    ax1.scatter(difficulty_scores, recalls, alpha=0.5, s=20)
    z = np.polyfit(difficulty_scores, recalls, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(difficulty_scores), max(difficulty_scores), 100)
    ax1.plot(x_line, p(x_line), 'r--', label='Trend')
    ax1.set_xlabel('Query Difficulty Score', fontsize=12)
    ax1.set_ylabel('Recall@k', fontsize=12)
    ax1.set_title('Recall vs Query Difficulty', fontsize=14)
    ax1.legend()
    
    # Stopping point vs difficulty
    ax2 = axes[1]
    ax2.scatter(difficulty_scores, stopping_points * 100, alpha=0.5, s=20)
    z = np.polyfit(difficulty_scores, stopping_points, 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line) * 100, 'r--', label='Trend')
    ax2.set_xlabel('Query Difficulty Score', fontsize=12)
    ax2.set_ylabel('Stopping Point (%)', fontsize=12)
    ax2.set_title('Adaptive Behavior: Harder Queries → Later Stopping', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, axes


def create_results_table(
    results: Dict[str, Dict[str, Any]],
    dataset_name: str,
    k: int
) -> str:
    """
    Create formatted results table as string.
    """
    lines = []
    lines.append(f"\nResults for {dataset_name}, k={k}")
    lines.append("=" * 70)
    lines.append(f"{'Method':<20} {'Recall':>10} {'Speedup':>10} {'Dist Ratio':>12}")
    lines.append("-" * 70)
    
    for method_name, metrics in sorted(results.items()):
        recall = metrics['recall']['mean']
        speedup = metrics['speedup']['mean']
        dist_ratio = metrics['dist_ratio']['mean']
        
        lines.append(
            f"{method_name:<20} {recall:>10.4f} {speedup:>10.2f}× {dist_ratio:>12.4f}"
        )
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
```
