"""
Brute Force Exact k-NN Search

The simplest baseline - computes distance to all points.
Always returns exact k-NN but O(nd) per query.
"""

import numpy as np
import time
from typing import Tuple, Optional
from abc import ABC, abstractmethod
import heapq


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
        else:
            self.norms = None
            self.X_normalized = None

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
            # ||q - x||^2 = ||q||^2 + ||x||^2 - 2*q.x
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
