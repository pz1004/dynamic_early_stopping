"""
Annoy - Approximate Nearest Neighbors Oh Yeah

Annoy builds a forest of random projection trees. Each tree recursively
splits the space using random hyperplanes until leaf nodes contain few points.

Key features:
- Memory-mapped: index can be larger than RAM
- Fast build and query times
- Good for serving in production

Trade-offs:
- More trees = higher recall, more memory and slower queries
- search_k parameter controls query-time accuracy/speed tradeoff

NOTE: This file provides both a pure Python implementation for understanding
and a wrapper for the actual annoy library when available.
"""

import numpy as np
import time
from typing import Tuple, Optional, List, Set
from collections import defaultdict
import heapq

from .exact_brute_force import BaseKNNSearcher


class AnnoyNode:
    """Node in an Annoy tree."""

    __slots__ = ['hyperplane', 'offset', 'left', 'right', 'indices', 'is_leaf']

    def __init__(self):
        self.hyperplane: Optional[np.ndarray] = None
        self.offset: float = 0.0
        self.left: Optional[AnnoyNode] = None
        self.right: Optional[AnnoyNode] = None
        self.indices: Optional[np.ndarray] = None
        self.is_leaf: bool = False


class AnnoyTree:
    """Single random projection tree for Annoy."""

    def __init__(self, max_leaf_size: int = 100, random_state=None):
        self.max_leaf_size = max_leaf_size
        self.rng = np.random.default_rng(random_state)
        self.root: Optional[AnnoyNode] = None

    def fit(self, X: np.ndarray):
        """Build the tree."""
        self.root = self._build_node(X, np.arange(len(X)))

    def _build_node(self, X: np.ndarray, indices: np.ndarray) -> AnnoyNode:
        """Recursively build tree nodes."""
        node = AnnoyNode()

        if len(indices) <= self.max_leaf_size:
            node.is_leaf = True
            node.indices = indices
            return node

        # Pick two random points and use their difference as the hyperplane
        if len(indices) >= 2:
            idx1, idx2 = self.rng.choice(len(indices), 2, replace=False)
            p1, p2 = X[indices[idx1]], X[indices[idx2]]
            node.hyperplane = p2 - p1
            norm = np.linalg.norm(node.hyperplane)
            if norm > 1e-10:
                node.hyperplane = node.hyperplane / norm
            node.offset = -np.dot(node.hyperplane, (p1 + p2) / 2)
        else:
            # Only one point, make it a leaf
            node.is_leaf = True
            node.indices = indices
            return node

        # Split points based on which side of hyperplane
        projections = np.dot(X[indices], node.hyperplane) + node.offset
        left_mask = projections <= 0

        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        # Handle edge cases
        if len(left_indices) == 0 or len(right_indices) == 0:
            node.is_leaf = True
            node.indices = indices
            return node

        node.left = self._build_node(X, left_indices)
        node.right = self._build_node(X, right_indices)

        return node

    def get_candidates(
        self,
        q: np.ndarray,
        search_k: int
    ) -> Set[int]:
        """
        Get candidate points by traversing the tree.

        Parameters
        ----------
        q : np.ndarray
            Query point.
        search_k : int
            Target number of candidates to retrieve.

        Returns
        -------
        candidates : Set[int]
            Set of candidate point indices.
        """
        candidates = set()

        # Priority queue: (priority, node)
        # Use negative distance to hyperplane as priority (closer = higher priority)
        queue = [(0.0, self.root)]

        while queue and len(candidates) < search_k:
            _, node = heapq.heappop(queue)

            if node.is_leaf:
                candidates.update(node.indices.tolist())
            else:
                # Compute distance to hyperplane
                margin = np.dot(q, node.hyperplane) + node.offset

                # Primary child (where query falls)
                if margin <= 0:
                    primary, secondary = node.left, node.right
                else:
                    primary, secondary = node.right, node.left

                # Always explore primary child
                heapq.heappush(queue, (0.0, primary))

                # Maybe explore secondary child (closer hyperplanes = higher priority)
                heapq.heappush(queue, (abs(margin), secondary))

        return candidates


class AnnoyIndexPure:
    """
    Pure Python implementation of Annoy index (for understanding).

    For production use, use the actual annoy library wrapper below.
    """

    def __init__(
        self,
        n_trees: int = 50,
        max_leaf_size: int = 100,
        random_state: Optional[int] = None
    ):
        self.n_trees = n_trees
        self.max_leaf_size = max_leaf_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.trees: List[AnnoyTree] = []

    def fit(self, X: np.ndarray) -> 'AnnoyIndexPure':
        """Build the forest of trees."""
        self.trees = []
        for i in range(self.n_trees):
            tree = AnnoyTree(
                max_leaf_size=self.max_leaf_size,
                random_state=self.rng.integers(1000000)
            )
            tree.fit(X)
            self.trees.append(tree)
        return self

    def query_candidates(
        self,
        q: np.ndarray,
        search_k: int
    ) -> Set[int]:
        """Get candidates from all trees."""
        candidates = set()
        per_tree = search_k // self.n_trees + 1

        for tree in self.trees:
            candidates.update(tree.get_candidates(q, per_tree))

        return candidates


class AnnoyKNN(BaseKNNSearcher):
    """
    Annoy-based approximate k-NN search.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    n_trees : int, default=50
        Number of trees in the forest. More trees = higher recall.
    search_k : int or None, default=None
        Number of nodes to inspect during query. If None, uses n_trees * k * 10.
    distance_metric : str, default='euclidean'
        Distance metric: 'euclidean', 'angular' (cosine), 'manhattan'.
    use_library : bool, default=False
        Whether to use the actual annoy library (if installed).
    random_state : int or None, default=None
        Random seed.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_trees: int = 50,
        search_k: Optional[int] = None,
        distance_metric: str = 'euclidean',
        use_library: bool = False,
        random_state: Optional[int] = None
    ):
        super().__init__(X)
        self.n_trees = n_trees
        self.search_k = search_k
        self.distance_metric = distance_metric
        self.use_library = use_library
        self.random_state = random_state

        self.index = None
        self._using_library = False

    def fit(self) -> 'AnnoyKNN':
        """Build the Annoy index."""
        t0 = time.perf_counter()

        if self.use_library:
            try:
                from annoy import AnnoyIndex

                metric_map = {
                    'euclidean': 'euclidean',
                    'angular': 'angular',
                    'cosine': 'angular',
                    'manhattan': 'manhattan'
                }

                self.index = AnnoyIndex(self.d, metric_map.get(self.distance_metric, 'euclidean'))

                for i in range(self.n):
                    self.index.add_item(i, self.X[i])

                self.index.build(self.n_trees)
                self._using_library = True

            except ImportError:
                # Fall back to pure Python
                self.index = AnnoyIndexPure(
                    n_trees=self.n_trees,
                    random_state=self.random_state
                )
                self.index.fit(self.X)
                self._using_library = False
        else:
            self.index = AnnoyIndexPure(
                n_trees=self.n_trees,
                random_state=self.random_state
            )
            self.index.fit(self.X)
            self._using_library = False

        self._build_time = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def query(
        self,
        q: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Find approximate k nearest neighbors."""
        q = np.asarray(q, dtype=np.float32)

        search_k = self.search_k or (self.n_trees * k * 10)

        if self._using_library:
            neighbors, distances = self.index.get_nns_by_vector(
                q, k, search_k=search_k, include_distances=True
            )
            return np.array(neighbors), np.array(distances), search_k
        else:
            # Pure Python implementation
            candidates = self.index.query_candidates(q, search_k)
            candidates = list(candidates)
            dist_count = len(candidates)

            if len(candidates) == 0:
                return np.zeros(k, dtype=np.int64), np.full(k, np.inf), 0

            # Compute distances
            if self.distance_metric == 'euclidean':
                distances = np.sqrt(np.sum((self.X[candidates] - q) ** 2, axis=1))
            elif self.distance_metric in ['angular', 'cosine']:
                q_norm = np.linalg.norm(q)
                dots = np.dot(self.X[candidates], q)
                norms = np.linalg.norm(self.X[candidates], axis=1)
                distances = 1 - dots / (norms * q_norm + 1e-10)
            else:
                distances = np.sum(np.abs(self.X[candidates] - q), axis=1)

            if len(candidates) <= k:
                sorted_order = np.argsort(distances)
                neighbors = np.array(candidates)[sorted_order]
                distances = distances[sorted_order]
            else:
                top_k = np.argpartition(distances, k)[:k]
                sorted_order = np.argsort(distances[top_k])
                neighbors = np.array(candidates)[top_k[sorted_order]]
                distances = distances[top_k[sorted_order]]

            return neighbors, distances, dist_count
