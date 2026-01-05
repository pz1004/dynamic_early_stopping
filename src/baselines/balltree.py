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

from .exact_brute_force import BaseKNNSearcher


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
