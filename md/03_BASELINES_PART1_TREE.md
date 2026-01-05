# Baseline Implementations - Part 1: Exact and Tree-Based Methods

## Overview

This document provides self-contained specifications for implementing baseline k-NN methods. Each baseline should follow a common interface for easy comparison.

## Common Interface

All baselines must implement this interface:

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Optional

class BaseKNNSearcher(ABC):
    """Abstract base class for all k-NN search methods."""
    
    def __init__(self, X: np.ndarray, **kwargs):
        """
        Initialize the searcher with dataset.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The dataset to search.
        **kwargs : dict
            Method-specific parameters.
        """
        self.X = np.asarray(X, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.is_fitted = False
        self._build_time = 0.0
    
    @abstractmethod
    def fit(self) -> 'BaseKNNSearcher':
        """Build any required index structures."""
        pass
    
    @abstractmethod
    def query(
        self, 
        q: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Find k nearest neighbors.
        
        Parameters
        ----------
        q : np.ndarray of shape (n_features,)
            Query point.
        k : int
            Number of neighbors.
        
        Returns
        -------
        neighbors : np.ndarray of shape (k,)
            Indices of k nearest neighbors.
        distances : np.ndarray of shape (k,)
            Distances to neighbors.
        dist_count : int
            Number of distance computations (for fair comparison).
        """
        pass
    
    def query_batch(
        self, 
        queries: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Query multiple points."""
        n_queries = len(queries)
        all_neighbors = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        all_dist_counts = np.zeros(n_queries, dtype=np.int64)
        
        for i, q in enumerate(queries):
            neighbors, distances, dist_count = self.query(q, k)
            all_neighbors[i] = neighbors
            all_distances[i] = distances
            all_dist_counts[i] = dist_count
        
        return all_neighbors, all_distances, all_dist_counts
    
    @property
    def build_time(self) -> float:
        """Return index build time in seconds."""
        return self._build_time
```

---

## 1. Exact Brute Force

### File: `src/baselines/exact_brute_force.py`

```python
"""
Brute Force Exact k-NN Search

The simplest baseline - computes distance to all points.
Always returns exact k-NN but O(nd) per query.
"""

import numpy as np
import time
from typing import Tuple
import heapq

class ExactBruteForceKNN(BaseKNNSearcher):
    """
    Exact k-NN using brute force distance computation.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    distance_metric : str, default='euclidean'
        Distance metric: 'euclidean', 'cosine', 'manhattan'.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        distance_metric: str = 'euclidean'
    ):
        super().__init__(X)
        self.distance_metric = distance_metric
        
        # Precompute for cosine
        if distance_metric == 'cosine':
            self.norms = np.linalg.norm(X, axis=1, keepdims=True)
            self.X_normalized = X / (self.norms + 1e-10)
    
    def fit(self) -> 'ExactBruteForceKNN':
        """No index to build for brute force."""
        t0 = time.perf_counter()
        # Just mark as fitted
        self.is_fitted = True
        self._build_time = time.perf_counter() - t0
        return self
    
    def query(
        self, 
        q: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Find k nearest neighbors by computing all distances.
        
        Uses vectorized operations for efficiency.
        """
        q = np.asarray(q, dtype=np.float32)
        
        # Compute all distances (vectorized)
        if self.distance_metric == 'euclidean':
            # ||q - x||^2 = ||q||^2 + ||x||^2 - 2*q·x
            # More numerically stable than direct subtraction
            diff = self.X - q
            distances = np.sqrt(np.sum(diff * diff, axis=1))
        
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.X - q), axis=1)
        
        elif self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q)
            q_normalized = q / (q_norm + 1e-10)
            similarities = np.dot(self.X_normalized, q_normalized)
            distances = 1.0 - similarities
        
        else:
            raise ValueError(f"Unknown metric: {self.distance_metric}")
        
        # Find k smallest
        if k >= self.n:
            indices = np.argsort(distances)
            return indices, distances[indices], self.n
        
        # Use argpartition for efficiency (O(n) instead of O(n log n))
        indices = np.argpartition(distances, k)[:k]
        # Sort the k smallest
        sorted_order = np.argsort(distances[indices])
        indices = indices[sorted_order]
        
        return indices, distances[indices], self.n
    
    def query_batch_vectorized(
        self, 
        queries: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorized batch query (more efficient for many queries).
        
        Note: Does not return dist_count per query.
        """
        queries = np.asarray(queries, dtype=np.float32)
        n_queries = len(queries)
        
        if self.distance_metric == 'euclidean':
            # Compute pairwise distances using broadcasting
            # queries: (n_queries, d), X: (n, d)
            # Result: (n_queries, n)
            q_sq = np.sum(queries ** 2, axis=1, keepdims=True)  # (n_queries, 1)
            x_sq = np.sum(self.X ** 2, axis=1)  # (n,)
            cross = np.dot(queries, self.X.T)  # (n_queries, n)
            distances = np.sqrt(np.maximum(q_sq + x_sq - 2 * cross, 0))
        
        elif self.distance_metric == 'cosine':
            q_norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries_normalized = queries / (q_norms + 1e-10)
            similarities = np.dot(queries_normalized, self.X_normalized.T)
            distances = 1.0 - similarities
        
        else:
            # Fall back to loop for manhattan
            distances = np.zeros((n_queries, self.n), dtype=np.float32)
            for i, q in enumerate(queries):
                distances[i] = np.sum(np.abs(self.X - q), axis=1)
        
        # Find k nearest for each query
        all_neighbors = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        
        for i in range(n_queries):
            indices = np.argpartition(distances[i], k)[:k]
            sorted_order = np.argsort(distances[i, indices])
            all_neighbors[i] = indices[sorted_order]
            all_distances[i] = distances[i, all_neighbors[i]]
        
        return all_neighbors, all_distances
```

---

## 2. KD-Tree

### File: `src/baselines/kdtree.py`

```python
"""
KD-Tree for k-NN Search

A space-partitioning data structure that recursively divides space
along coordinate axes. Efficient for low dimensions (d < 20).

Implementation notes:
- Uses median split for balanced trees
- Supports backtracking limit for approximate search
- Performance degrades in high dimensions ("curse of dimensionality")
"""

import numpy as np
import time
from typing import Tuple, Optional, List
import heapq

class KDTreeNode:
    """Node in the KD-Tree."""
    
    __slots__ = ['point', 'index', 'left', 'right', 'split_dim', 'split_val']
    
    def __init__(
        self, 
        point: np.ndarray, 
        index: int, 
        split_dim: int,
        split_val: float
    ):
        self.point = point
        self.index = index
        self.split_dim = split_dim
        self.split_val = split_val
        self.left: Optional[KDTreeNode] = None
        self.right: Optional[KDTreeNode] = None


class KDTreeKNN(BaseKNNSearcher):
    """
    KD-Tree based k-NN search.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    leaf_size : int, default=30
        Number of points at which to switch to brute force.
    backtrack_limit : int or None, default=None
        Maximum backtracking steps. None for exact search.
    distance_metric : str, default='euclidean'
        Only 'euclidean' supported for KD-Tree.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        leaf_size: int = 30,
        backtrack_limit: Optional[int] = None,
        distance_metric: str = 'euclidean'
    ):
        super().__init__(X)
        self.leaf_size = leaf_size
        self.backtrack_limit = backtrack_limit
        
        if distance_metric != 'euclidean':
            raise ValueError("KD-Tree only supports Euclidean distance")
        
        self.root: Optional[KDTreeNode] = None
        self._dist_count = 0
    
    def fit(self) -> 'KDTreeKNN':
        """Build the KD-Tree."""
        t0 = time.perf_counter()
        
        indices = np.arange(self.n)
        self.root = self._build_tree(indices, depth=0)
        
        self._build_time = time.perf_counter() - t0
        self.is_fitted = True
        return self
    
    def _build_tree(
        self, 
        indices: np.ndarray, 
        depth: int
    ) -> Optional[KDTreeNode]:
        """Recursively build the tree."""
        if len(indices) == 0:
            return None
        
        if len(indices) <= self.leaf_size:
            # Create leaf: just return node with first point
            # Store all indices for leaf search
            split_dim = depth % self.d
            median_idx = len(indices) // 2
            sorted_indices = indices[np.argsort(self.X[indices, split_dim])]
            median_point_idx = sorted_indices[median_idx]
            
            node = KDTreeNode(
                point=self.X[median_point_idx],
                index=median_point_idx,
                split_dim=split_dim,
                split_val=self.X[median_point_idx, split_dim]
            )
            # For leaf nodes, we'll handle differently in search
            node._leaf_indices = sorted_indices
            return node
        
        # Choose split dimension (cycle through dimensions)
        split_dim = depth % self.d
        
        # Find median along split dimension
        values = self.X[indices, split_dim]
        median_idx = len(indices) // 2
        
        # Partition around median
        sorted_order = np.argsort(values)
        sorted_indices = indices[sorted_order]
        
        median_point_idx = sorted_indices[median_idx]
        
        node = KDTreeNode(
            point=self.X[median_point_idx],
            index=median_point_idx,
            split_dim=split_dim,
            split_val=self.X[median_point_idx, split_dim]
        )
        
        # Recursively build subtrees
        node.left = self._build_tree(sorted_indices[:median_idx], depth + 1)
        node.right = self._build_tree(sorted_indices[median_idx + 1:], depth + 1)
        
        return node
    
    def query(
        self, 
        q: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Find k nearest neighbors using the KD-Tree."""
        q = np.asarray(q, dtype=np.float32)
        self._dist_count = 0
        self._backtrack_count = 0
        
        # Max heap for k nearest (use negative distances)
        heap: List[Tuple[float, int]] = []
        
        self._search(self.root, q, k, heap)
        
        # Extract results
        heap.sort()
        neighbors = np.array([idx for _, idx in heap])
        distances = np.array([-d for d, _ in heap])
        
        return neighbors, distances, self._dist_count
    
    def _search(
        self, 
        node: Optional[KDTreeNode],
        q: np.ndarray,
        k: int,
        heap: List[Tuple[float, int]]
    ):
        """Recursive k-NN search."""
        if node is None:
            return
        
        # Check backtrack limit
        if self.backtrack_limit is not None:
            if self._backtrack_count >= self.backtrack_limit:
                return
        
        # Handle leaf nodes with multiple points
        if hasattr(node, '_leaf_indices'):
            for idx in node._leaf_indices:
                dist = self._distance(q, self.X[idx])
                self._dist_count += 1
                
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, idx))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, idx))
            return
        
        # Compute distance to current node
        dist = self._distance(q, node.point)
        self._dist_count += 1
        
        # Update heap
        if len(heap) < k:
            heapq.heappush(heap, (-dist, node.index))
        elif dist < -heap[0][0]:
            heapq.heapreplace(heap, (-dist, node.index))
        
        # Determine which subtree to search first
        if q[node.split_dim] < node.split_val:
            first, second = node.left, node.right
        else:
            first, second = node.right, node.left
        
        # Search closer subtree
        self._search(first, q, k, heap)
        
        # Check if we need to search the other subtree
        # (if the splitting hyperplane is closer than current k-th distance)
        split_dist = abs(q[node.split_dim] - node.split_val)
        
        if len(heap) < k or split_dist < -heap[0][0]:
            self._backtrack_count += 1
            self._search(second, q, k, heap)
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute Euclidean distance."""
        diff = a - b
        return np.sqrt(np.dot(diff, diff))


class SklearnKDTreeKNN(BaseKNNSearcher):
    """
    Wrapper around sklearn's KDTree for comparison.
    
    Uses scipy's cKDTree for efficient implementation.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        leaf_size: int = 30
    ):
        super().__init__(X)
        self.leaf_size = leaf_size
        self._tree = None
    
    def fit(self) -> 'SklearnKDTreeKNN':
        """Build the tree using sklearn."""
        from sklearn.neighbors import KDTree
        
        t0 = time.perf_counter()
        self._tree = KDTree(self.X, leaf_size=self.leaf_size)
        self._build_time = time.perf_counter() - t0
        self.is_fitted = True
        return self
    
    def query(
        self, 
        q: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Query using sklearn's implementation."""
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        distances, indices = self._tree.query(q, k=k)
        
        # Estimate distance computations (sklearn doesn't expose this)
        # Use heuristic based on tree depth
        est_dist_count = int(self.n ** 0.5) * k
        
        return indices[0], distances[0], est_dist_count
```

---

## 3. Ball Tree

### File: `src/baselines/balltree.py`

```python
"""
Ball Tree for k-NN Search

A tree where each node represents a ball (hypersphere) containing
a subset of points. More robust than KD-Tree in higher dimensions.

Key advantages:
- Works better in higher dimensions than KD-Tree
- Can use any metric (not just Euclidean)
- Better pruning when data is clustered
"""

import numpy as np
import time
from typing import Tuple, Optional, List
import heapq


class BallTreeNode:
    """Node in the Ball Tree."""
    
    __slots__ = ['center', 'radius', 'indices', 'left', 'right', 'is_leaf']
    
    def __init__(self):
        self.center: Optional[np.ndarray] = None
        self.radius: float = 0.0
        self.indices: Optional[np.ndarray] = None  # For leaf nodes
        self.left: Optional[BallTreeNode] = None
        self.right: Optional[BallTreeNode] = None
        self.is_leaf: bool = False


class BallTreeKNN(BaseKNNSearcher):
    """
    Ball Tree based k-NN search.
    
    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    leaf_size : int, default=30
        Maximum number of points in a leaf node.
    distance_metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan'.
    """
    
    def __init__(
        self, 
        X: np.ndarray, 
        leaf_size: int = 30,
        distance_metric: str = 'euclidean'
    ):
        super().__init__(X)
        self.leaf_size = leaf_size
        self.distance_metric = distance_metric
        self.root: Optional[BallTreeNode] = None
        self._dist_count = 0
    
    def fit(self) -> 'BallTreeKNN':
        """Build the Ball Tree."""
        t0 = time.perf_counter()
        
        indices = np.arange(self.n)
        self.root = self._build_tree(indices)
        
        self._build_time = time.perf_counter() - t0
        self.is_fitted = True
        return self
    
    def _build_tree(self, indices: np.ndarray) -> BallTreeNode:
        """Recursively build the tree."""
        node = BallTreeNode()
        points = self.X[indices]
        
        # Compute bounding ball
        node.center = np.mean(points, axis=0)
        distances_to_center = np.linalg.norm(points - node.center, axis=1)
        node.radius = np.max(distances_to_center)
        
        if len(indices) <= self.leaf_size:
            # Create leaf node
            node.is_leaf = True
            node.indices = indices
            return node
        
        # Split: find dimension with maximum spread
        spread = np.max(points, axis=0) - np.min(points, axis=0)
        split_dim = np.argmax(spread)
        
        # Split at median
        values = points[:, split_dim]
        median = np.median(values)
        
        left_mask = values <= median
        # Handle edge case where all values are equal
        if np.all(left_mask) or np.all(~left_mask):
            left_mask = np.zeros(len(indices), dtype=bool)
            left_mask[:len(indices)//2] = True
        
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            # Can't split further, make leaf
            node.is_leaf = True
            node.indices = indices
            return node
        
        node.left = self._build_tree(left_indices)
        node.right = self._build_tree(right_indices)
        
        return node
    
    def query(
        self, 
        q: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Find k nearest neighbors using the Ball Tree."""
        q = np.asarray(q, dtype=np.float32)
        self._dist_count = 0
        
        # Max heap for k nearest
        heap: List[Tuple[float, int]] = []
        
        self._search(self.root, q, k, heap)
        
        # Extract results
        heap.sort()
        neighbors = np.array([idx for _, idx in heap])
        distances = np.array([-d for d, _ in heap])
        
        return neighbors, distances, self._dist_count
    
    def _search(
        self, 
        node: BallTreeNode,
        q: np.ndarray,
        k: int,
        heap: List[Tuple[float, int]]
    ):
        """Recursive k-NN search with ball pruning."""
        # Compute distance to ball center
        dist_to_center = self._distance(q, node.center)
        
        # Pruning: if the closest possible point in this ball
        # is farther than our current k-th nearest, skip
        if len(heap) >= k:
            current_kth = -heap[0][0]
            if dist_to_center - node.radius > current_kth:
                return
        
        if node.is_leaf:
            # Search all points in leaf
            for idx in node.indices:
                dist = self._distance(q, self.X[idx])
                self._dist_count += 1
                
                if len(heap) < k:
                    heapq.heappush(heap, (-dist, idx))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-dist, idx))
        else:
            # Determine which child to search first
            # (the one whose center is closer)
            dist_left = self._distance(q, node.left.center)
            dist_right = self._distance(q, node.right.center)
            
            if dist_left < dist_right:
                self._search(node.left, q, k, heap)
                self._search(node.right, q, k, heap)
            else:
                self._search(node.right, q, k, heap)
                self._search(node.left, q, k, heap)
    
    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute distance between two points."""
        if self.distance_metric == 'euclidean':
            diff = a - b
            return np.sqrt(np.dot(diff, diff))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(a - b))
        else:
            raise ValueError(f"Unknown metric: {self.distance_metric}")


class SklearnBallTreeKNN(BaseKNNSearcher):
    """Wrapper around sklearn's BallTree."""
    
    def __init__(
        self, 
        X: np.ndarray, 
        leaf_size: int = 30,
        metric: str = 'euclidean'
    ):
        super().__init__(X)
        self.leaf_size = leaf_size
        self.metric = metric
        self._tree = None
    
    def fit(self) -> 'SklearnBallTreeKNN':
        """Build the tree using sklearn."""
        from sklearn.neighbors import BallTree
        
        t0 = time.perf_counter()
        self._tree = BallTree(self.X, leaf_size=self.leaf_size, metric=self.metric)
        self._build_time = time.perf_counter() - t0
        self.is_fitted = True
        return self
    
    def query(
        self, 
        q: np.ndarray, 
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Query using sklearn's implementation."""
        q = np.asarray(q, dtype=np.float32).reshape(1, -1)
        distances, indices = self._tree.query(q, k=k)
        
        # Estimate distance computations
        est_dist_count = int(self.n ** 0.5) * k
        
        return indices[0], distances[0], est_dist_count
```

---

## Usage Example

```python
from src.baselines.exact_brute_force import ExactBruteForceKNN
from src.baselines.kdtree import KDTreeKNN, SklearnKDTreeKNN
from src.baselines.balltree import BallTreeKNN, SklearnBallTreeKNN
import numpy as np

# Create sample data
np.random.seed(42)
X = np.random.randn(10000, 50).astype(np.float32)
q = np.random.randn(50).astype(np.float32)
k = 10

# Test each method
methods = [
    ("Brute Force", ExactBruteForceKNN(X)),
    ("KD-Tree", KDTreeKNN(X, leaf_size=30)),
    ("Ball-Tree", BallTreeKNN(X, leaf_size=30)),
    ("Sklearn KD-Tree", SklearnKDTreeKNN(X)),
    ("Sklearn Ball-Tree", SklearnBallTreeKNN(X)),
]

for name, searcher in methods:
    searcher.fit()
    neighbors, distances, dist_count = searcher.query(q, k)
    print(f"{name}: {dist_count} distance computations, build time: {searcher.build_time:.4f}s")
```
