"""
Dynamic Early Stopping k-Nearest Neighbors (DES-kNN) Implementation.

This module implements the core DES-kNN algorithm that adaptively determines
when to stop searching based on statistical confidence bounds.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import heapq
from .statistics import (
    estimate_exceedance_probability,
    compute_confidence_bound,
    adaptive_alpha,
    CircularBuffer,
    WeibullCache
)
from .utils.profiling import Profiler


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
    batch_size : int, default=512
        Number of distance computations to perform per vectorized batch.
    crit1_mode : str, default="alpha_over_k"
        Criterion-1 threshold mode: "alpha_over_k" or "alpha".
    crit3_gap_ratio : float, default=0.5
        Criterion-3 gap ratio threshold.
    crit3_gap_mult : float, default=10.0
        Criterion-3 gap multiplier applied to k (gap > crit3_gap_mult * k).
    weibull_refresh_every : int, default=1
        Refresh Weibull fit every N stop checks.
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
        random_state: Optional[int] = None,
        batch_size: int = 512,
        crit1_mode: str = "alpha_over_k",
        crit3_gap_ratio: float = 0.5,
        crit3_gap_mult: float = 10.0,
        weibull_refresh_every: int = 1
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
        self.batch_size = max(1, int(batch_size))
        self.crit1_mode = crit1_mode
        self.crit3_gap_ratio = crit3_gap_ratio
        self.crit3_gap_mult = crit3_gap_mult
        self.weibull_refresh_every = max(1, int(weibull_refresh_every))

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
        return_stats: bool = False,
        stop_check_every: int = 1,
        crit1_mode: Optional[str] = None,
        crit3_gap_ratio: Optional[float] = None,
        crit3_gap_mult: Optional[float] = None,
        weibull_refresh_every: Optional[int] = None
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
        stop_check_every : int, default=1
            Only evaluate early-stopping criteria every N distance computations.
        crit1_mode : str, optional
            Criterion-1 threshold mode: "alpha_over_k" or "alpha".
            Defaults to the instance setting.
        crit3_gap_ratio : float, optional
            Criterion-3 gap ratio threshold. Defaults to the instance setting.
        crit3_gap_mult : float, optional
            Criterion-3 gap multiplier applied to k (gap > crit3_gap_mult * k).
            Defaults to the instance setting.
        weibull_refresh_every : int, optional
            Refresh Weibull fit every N stop checks. Defaults to the instance setting.

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
        stop_check_every = int(stop_check_every)
        if stop_check_every < 1:
            raise ValueError("stop_check_every must be >= 1")
        if crit1_mode is None:
            crit1_mode = self.crit1_mode
        if crit3_gap_ratio is None:
            crit3_gap_ratio = self.crit3_gap_ratio
        if crit3_gap_mult is None:
            crit3_gap_mult = self.crit3_gap_mult
        if weibull_refresh_every is None:
            weibull_refresh_every = self.weibull_refresh_every
        weibull_refresh_every = int(weibull_refresh_every)
        if crit1_mode not in {"alpha_over_k", "alpha"}:
            raise ValueError("crit1_mode must be 'alpha_over_k' or 'alpha'")
        if crit3_gap_ratio <= 0:
            raise ValueError("crit3_gap_ratio must be > 0")
        if crit3_gap_mult <= 0:
            raise ValueError("crit3_gap_mult must be > 0")
        if weibull_refresh_every < 1:
            raise ValueError("weibull_refresh_every must be >= 1")

        profiler = Profiler.from_env()
        profiling_enabled = profiler.enabled

        # Determine minimum samples
        min_samples = self.min_samples if self.min_samples is not None else max(2 * k, 50)

        # Initialize data structures
        # Using list as max heap (negate distances for max heap behavior)
        heap = []  # (neg_distance, index)
        d_k = np.inf

        # Statistics tracking
        distances_window = CircularBuffer(self.window_size)
        update_history = []
        dist_count = 0
        last_update = 0
        stop_check_count = 0
        weibull_cache = None
        if self.use_weibull and weibull_refresh_every > 1:
            weibull_cache = WeibullCache()

        # For detailed stats
        all_computed_distances = [] if return_stats else None
        stopping_criteria_history = [] if return_stats else None

        # Random permutation for unbiased statistics
        indices = self.rng.permutation(self.n)

        # Precompute query-specific values for distance computation
        q_normalized = None
        if self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q)
            q_normalized = q / (q_norm + 1e-10)

        # Main search loop
        stopped_early = False
        n = self.n
        batch_size = min(self.batch_size, n)
        heap_push = heapq.heappush
        heap_replace = heapq.heapreplace
        window_add = distances_window.add
        for batch_start in range(0, n, batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            if profiling_enabled:
                with profiler.time("distance_compute"):
                    batch_distances = self._compute_distances_batch(
                        q,
                        batch_indices,
                        q_normalized=q_normalized
                    )
            else:
                batch_distances = self._compute_distances_batch(
                    q,
                    batch_indices,
                    q_normalized=q_normalized
                )

            for offset, idx in enumerate(batch_indices):
                d = batch_distances[offset]
                i = batch_start + offset
                dist_count += 1

                if return_stats:
                    all_computed_distances.append(d)

                # Update k-NN heap
                if len(heap) < k:
                    if profiling_enabled:
                        with profiler.time("heap_update"):
                            heap_push(heap, (-d, idx))
                            if len(heap) == k:
                                d_k = -heap[0][0]
                    else:
                        heap_push(heap, (-d, idx))
                        if len(heap) == k:
                            d_k = -heap[0][0]
                elif d < d_k:
                    if profiling_enabled:
                        with profiler.time("heap_update"):
                            heap_replace(heap, (-d, idx))
                            d_k = -heap[0][0]
                            last_update = i
                            update_history.append(i)
                    else:
                        heap_replace(heap, (-d, idx))
                        d_k = -heap[0][0]
                        last_update = i
                        update_history.append(i)

                # Track distance statistics
                window_add(d)

                # Check early stopping criterion
                if i >= min_samples and (dist_count % stop_check_every == 0):
                    stop_check_count += 1
                    if profiling_enabled:
                        with profiler.time("window_to_list"):
                            window_values = distances_window.get_all()
                        with profiler.time("stop_criteria"):
                            should_stop, criteria = self._should_stop(
                                distances_window=window_values,
                                d_k=d_k,
                                update_history=update_history,
                                current_idx=i,
                                last_update=last_update,
                                k=k,
                                remaining=n - i - 1,
                                profiler=profiler,
                                crit1_mode=crit1_mode,
                                crit3_gap_ratio=crit3_gap_ratio,
                                crit3_gap_mult=crit3_gap_mult,
                                weibull_cache=weibull_cache,
                                weibull_refresh_every=weibull_refresh_every,
                                stop_check_count=stop_check_count
                            )
                    else:
                        should_stop, criteria = self._should_stop(
                            distances_window=distances_window.get_all(),
                            d_k=d_k,
                            update_history=update_history,
                            current_idx=i,
                            last_update=last_update,
                            k=k,
                            remaining=n - i - 1,
                            profiler=None,
                            crit1_mode=crit1_mode,
                            crit3_gap_ratio=crit3_gap_ratio,
                            crit3_gap_mult=crit3_gap_mult,
                            weibull_cache=weibull_cache,
                            weibull_refresh_every=weibull_refresh_every,
                            stop_check_count=stop_check_count
                        )

                    if return_stats:
                        stopping_criteria_history.append(criteria)

                    if should_stop:
                        stopped_early = True
                        break

            if stopped_early:
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
            if profiling_enabled:
                stats['profile'] = profiler.summary()

        if profiling_enabled:
            self.last_profile = profiler.summary()
            self.last_profile_text = profiler.format_summary()
        else:
            self.last_profile = None
            self.last_profile_text = None

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
        remaining: int,
        profiler: Optional[Profiler] = None,
        crit1_mode: str = "alpha_over_k",
        crit3_gap_ratio: float = 0.5,
        crit3_gap_mult: float = 10.0,
        weibull_cache: Optional[WeibullCache] = None,
        weibull_refresh_every: int = 1,
        stop_check_count: int = -1
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
        distances = np.asarray(distances_window, dtype=np.float64)
        if self.use_adaptive_alpha:
            current_alpha = adaptive_alpha(
                distances, d_k, self.alpha
            )
        else:
            current_alpha = self.alpha

        # Criterion 1: Estimate probability remaining points enter k-NN
        p_remain, expected_entrants = estimate_exceedance_probability(
            distances=distances,
            d_k=d_k,
            remaining=remaining,
            alpha=current_alpha,
            use_weibull=self.use_weibull,
            profiler=profiler,
            weibull_cache=weibull_cache,
            weibull_refresh_every=weibull_refresh_every,
            check_step=stop_check_count
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
        if crit1_mode == "alpha":
            crit1_threshold = current_alpha
        else:
            crit1_threshold = current_alpha / k
        crit1 = p_remain < crit1_threshold
        crit2 = conf > 1 - current_alpha
        crit3 = (gap_ratio > crit3_gap_ratio) and (gap > crit3_gap_mult * k)

        # Combined decision: require at least 2 criteria
        should_stop = (crit1 and crit2) or (crit1 and crit3) or (crit2 and crit3)

        criteria = {
            'p_remain': p_remain,
            'expected_entrants': expected_entrants,
            'confidence': conf,
            'gap_ratio': gap_ratio,
            'gap': gap,
            'alpha': current_alpha,
            'crit1_threshold': crit1_threshold,
            'crit3_gap_ratio': crit3_gap_ratio,
            'crit3_gap_threshold': crit3_gap_mult * k,
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

            results = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_single_query)(q) for q in queries
            )

            for i, (neighbors, distances, dist_count) in enumerate(results):
                all_neighbors[i] = neighbors
                all_distances[i] = distances
                all_dist_counts[i] = dist_count

        return all_neighbors, all_distances, all_dist_counts

    def _compute_distances_batch(
        self,
        q: np.ndarray,
        indices: np.ndarray,
        q_normalized: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute distances to multiple points at once (for potential optimization).

        Parameters
        ----------
        q : np.ndarray of shape (n_features,)
            Query point.
        indices : np.ndarray of shape (batch_size,)
            Indices of dataset points.
        q_normalized : np.ndarray or None
            Pre-normalized query vector for cosine metric.

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
            if q_normalized is None:
                q_norm = np.linalg.norm(q)
                q_normalized = q / (q_norm + 1e-10)
            similarities = np.dot(self.X_normalized[indices], q_normalized)
            return 1.0 - similarities

        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
