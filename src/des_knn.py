"""
Dynamic Early Stopping k-Nearest Neighbors (DES-kNN) Implementation.

This module implements the core DES-kNN algorithm that adaptively determines
when to stop searching based on statistical confidence bounds.
"""

import numpy as np
from collections import deque
from typing import Tuple, List, Optional, Dict, Any
import heapq
from .statistics import (
    estimate_exceedance_probability,
    compute_confidence_bound,
    adaptive_alpha,
    fit_weibull
)


class DESKNNSearcher:
    """
    Dynamic Early Stopping k-Nearest Neighbors Searcher.

    Uses online statistical bounds to estimate when the true k-NN set
    has been found with high probability, enabling early termination.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The dataset to search.
    alpha : float, default=0.01
        Significance level for stopping criterion (1-alpha = confidence).
    window_size : int, default=100
        Size of sliding window for distance statistics.
    min_samples : int or None, default=None
        Minimum samples to compute before considering early stop.
        If None, uses max(2*k, 50).
    adaptive_alpha : bool, default=True
        Whether to adaptively adjust alpha based on query difficulty.
    use_weibull : bool, default=True
        Whether to use Weibull distribution fit for probability estimation.
    distance_metric : str, default='euclidean'
        Distance metric: 'euclidean', 'cosine', or 'manhattan'.
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        X: np.ndarray,
        alpha: float = 0.01,
        window_size: int = 100,
        min_samples: Optional[int] = None,
        adaptive_alpha: bool = True,
        use_weibull: bool = True,
        distance_metric: str = 'euclidean',
        random_state: Optional[int] = None
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.alpha = alpha
        self.window_size = window_size
        self.min_samples = min_samples
        self.use_adaptive_alpha = adaptive_alpha
        self.use_weibull = use_weibull
        self.distance_metric = distance_metric
        self.random_state = random_state

        # Precompute for efficiency
        if distance_metric == 'cosine':
            self.norms = np.linalg.norm(X, axis=1, keepdims=True)
            self.X_normalized = X / (self.norms + 1e-10)
        else:
            self.norms = None
            self.X_normalized = None

        # Set random generator
        self.rng = np.random.default_rng(random_state)

    def fit(self) -> 'DESKNNSearcher':
        """No-op for API compatibility with baselines."""
        return self

    def query(
        self,
        q: np.ndarray,
        k: int,
        return_distances: bool = True,
        return_stats: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int] | Tuple[np.ndarray, int] | Tuple[np.ndarray, np.ndarray, int, Dict] | Tuple[np.ndarray, int, Dict]:
        """
        Find k nearest neighbors with dynamic early stopping.

        Parameters
        ----------
        q : np.ndarray of shape (n_features,)
            Query point.
        k : int
            Number of neighbors to find.
        return_distances : bool, default=True
            Whether to return distances.
        return_stats : bool, default=False
            Whether to return detailed statistics.

        Returns
        -------
        neighbors : np.ndarray of shape (k,)
            Indices of k nearest neighbors.
        distances : np.ndarray of shape (k,), optional
            Distances to k nearest neighbors (only if return_distances=True).
        dist_count : int
            Number of distance computations performed.
        stats : dict (optional)
            Detailed statistics if return_stats=True.
        """
        q = np.asarray(q, dtype=np.float32)

        # Determine minimum samples
        min_samples = self.min_samples if self.min_samples is not None else max(2 * k, 50)

        # Initialize data structures
        # Using list as max heap (negate distances for max heap behavior)
        heap = []  # (neg_distance, index)
        d_k = np.inf

        # Statistics tracking
        distances_window = deque(maxlen=self.window_size)
        update_history = []
        dist_count = 0
        last_update = 0

        # For detailed stats
        all_computed_distances = [] if return_stats else None
        stopping_criteria_history = [] if return_stats else None

        # Random permutation for unbiased statistics
        indices = self.rng.permutation(self.n)

        # Precompute query-specific values for distance computation
        if self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q)
            q_normalized = q / (q_norm + 1e-10)

        # Main search loop
        stopped_early = False
        for i, idx in enumerate(indices):
            # Compute distance
            d = self._compute_distance(q, idx)
            dist_count += 1

            if return_stats:
                all_computed_distances.append(d)

            # Update k-NN heap
            if len(heap) < k:
                heapq.heappush(heap, (-d, idx))
                if len(heap) == k:
                    d_k = -heap[0][0]
            elif d < d_k:
                heapq.heapreplace(heap, (-d, idx))
                d_k = -heap[0][0]
                last_update = i
                update_history.append(i)

            # Track distance statistics
            distances_window.append(d)

            # Check early stopping criterion
            if i >= min_samples:
                should_stop, criteria = self._should_stop(
                    distances_window=list(distances_window),
                    d_k=d_k,
                    update_history=update_history,
                    current_idx=i,
                    last_update=last_update,
                    k=k,
                    remaining=self.n - i - 1
                )

                if return_stats:
                    stopping_criteria_history.append(criteria)

                if should_stop:
                    stopped_early = True
                    break

        # Extract results (sort by distance)
        heap_items = [(-d, idx) for d, idx in heap]
        heap_items.sort()

        neighbors = np.array([idx for d, idx in heap_items])
        distances = np.array([d for d, idx in heap_items])

        stats = None
        if return_stats:
            stats = {
                'stopped_early': stopped_early,
                'stopping_point': dist_count / self.n,
                'last_update': last_update,
                'num_updates': len(update_history),
                'update_history': update_history,
                'final_d_k': d_k,
                'all_distances': all_computed_distances,
                'criteria_history': stopping_criteria_history
            }

        if return_distances and return_stats:
            return neighbors, distances, dist_count, stats
        if return_distances:
            return neighbors, distances, dist_count
        if return_stats:
            return neighbors, dist_count, stats
        return neighbors, dist_count

    def _compute_distance(self, q: np.ndarray, idx: int) -> float:
        """
        Compute distance between query and dataset point.

        Parameters
        ----------
        q : np.ndarray
            Query point.
        idx : int
            Index of dataset point.

        Returns
        -------
        distance : float
        """
        x = self.X[idx]

        if self.distance_metric == 'euclidean':
            diff = q - x
            return np.sqrt(np.dot(diff, diff))

        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(q - x))

        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - cosine_similarity
            if self.X_normalized is not None:
                q_norm = np.linalg.norm(q)
                q_normalized = q / (q_norm + 1e-10)
                similarity = np.dot(q_normalized, self.X_normalized[idx])
            else:
                dot_product = np.dot(q, x)
                norm_q = np.linalg.norm(q)
                norm_x = np.linalg.norm(x)
                similarity = dot_product / (norm_q * norm_x + 1e-10)
            return 1.0 - similarity

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    def _should_stop(
        self,
        distances_window: List[float],
        d_k: float,
        update_history: List[int],
        current_idx: int,
        last_update: int,
        k: int,
        remaining: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine whether to stop searching based on statistical criteria.

        Returns
        -------
        should_stop : bool
        criteria : dict
            Dictionary with criterion values for analysis.
        """
        # Get current alpha (possibly adaptive)
        if self.use_adaptive_alpha:
            current_alpha = adaptive_alpha(
                distances_window, d_k, self.alpha
            )
        else:
            current_alpha = self.alpha

        # Criterion 1: Estimate probability remaining points enter k-NN
        p_remain, expected_entrants = estimate_exceedance_probability(
            distances=distances_window,
            d_k=d_k,
            remaining=remaining,
            alpha=current_alpha,
            use_weibull=self.use_weibull
        )

        # Criterion 2: Confidence from update pattern
        conf = compute_confidence_bound(
            update_history=update_history,
            current_idx=current_idx,
            k=k
        )

        # Criterion 3: Gap ratio
        gap = current_idx - last_update
        gap_ratio = gap / (current_idx + 1)

        # Evaluate criteria
        crit1 = p_remain < current_alpha / k
        crit2 = conf > 1 - current_alpha
        crit3 = (gap_ratio > 0.5) and (gap > 10 * k)

        # Combined decision: require at least 2 criteria
        should_stop = (crit1 and crit2) or (crit1 and crit3) or (crit2 and crit3)

        criteria = {
            'p_remain': p_remain,
            'expected_entrants': expected_entrants,
            'confidence': conf,
            'gap_ratio': gap_ratio,
            'gap': gap,
            'alpha': current_alpha,
            'crit1': crit1,
            'crit2': crit2,
            'crit3': crit3
        }

        return should_stop, criteria

    def query_batch(
        self,
        queries: np.ndarray,
        k: int,
        n_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Query multiple points.

        Parameters
        ----------
        queries : np.ndarray of shape (n_queries, n_features)
            Query points.
        k : int
            Number of neighbors.
        n_jobs : int, default=1
            Number of parallel jobs (1 = sequential).

        Returns
        -------
        all_neighbors : np.ndarray of shape (n_queries, k)
        all_distances : np.ndarray of shape (n_queries, k)
        all_dist_counts : np.ndarray of shape (n_queries,)
        """
        n_queries = len(queries)
        all_neighbors = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        all_dist_counts = np.zeros(n_queries, dtype=np.int64)

        if n_jobs == 1:
            for i, q in enumerate(queries):
                neighbors, distances, dist_count = self.query(q, k)
                all_neighbors[i] = neighbors
                all_distances[i] = distances
                all_dist_counts[i] = dist_count
        else:
            # Parallel implementation using joblib
            from joblib import Parallel, delayed

            def _single_query(q):
                return self.query(q, k)

            results = Parallel(n_jobs=n_jobs)(
                delayed(_single_query)(q) for q in queries
            )

            for i, (neighbors, distances, dist_count) in enumerate(results):
                all_neighbors[i] = neighbors
                all_distances[i] = distances
                all_dist_counts[i] = dist_count

        return all_neighbors, all_distances, all_dist_counts

    def _compute_distances_batch(self, q: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """
        Compute distances to multiple points at once (for potential optimization).

        Parameters
        ----------
        q : np.ndarray of shape (n_features,)
            Query point.
        indices : np.ndarray of shape (batch_size,)
            Indices of dataset points.

        Returns
        -------
        distances : np.ndarray of shape (batch_size,)
        """
        X_batch = self.X[indices]

        if self.distance_metric == 'euclidean':
            diff = X_batch - q
            return np.sqrt(np.sum(diff * diff, axis=1))

        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(X_batch - q), axis=1)

        elif self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q)
            q_normalized = q / (q_norm + 1e-10)
            similarities = np.dot(self.X_normalized[indices], q_normalized)
            return 1.0 - similarities

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
