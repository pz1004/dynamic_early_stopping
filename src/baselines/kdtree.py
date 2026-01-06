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

from .exact_brute_force import BaseKNNSearcher


class KDTreeNode:
    """Node in the KD-Tree."""

    __slots__ = ['point', 'index', 'left', 'right', 'split_dim', 'split_val', '_leaf_indices']

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
        self._leaf_indices: Optional[np.ndarray] = None


class KDTreeKNN(BaseKNNSearcher):
    """
    KD-Tree based k-NN search.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    leaf_size : int, default=40
        Number of points at which to switch to brute force.
    backtrack_limit : int or None, default=None
        Maximum backtracking steps. None for exact search.
    distance_metric : str, default='euclidean'
        Only 'euclidean' supported for KD-Tree.
    """

    def __init__(
        self,
        X: np.ndarray,
        leaf_size: int = 40,
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
        if node._leaf_indices is not None:
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
        leaf_size: int = 40
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
