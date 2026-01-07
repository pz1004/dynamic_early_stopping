"""
Dynamic Early Stopping k-Nearest Neighbors (DES-kNN) Implementation.

This module implements the DES-kNN algorithm with the Beta-Geometric "Gap" 
stopping criterion. It uses the inter-arrival times of neighbor updates to 
statistically bound the expected number of missed neighbors.
"""

import numpy as np
import heapq
from typing import Tuple, List, Optional, Dict, Any
from .statistics import estimate_future_matches
from .utils.profiling import Profiler
from joblib import Parallel, delayed
from .sorting import BaseSorter, PCASorter


class DESKNNSearcher:
    """
    Dynamic Early Stopping k-Nearest Neighbors Searcher.

    Uses a Beta-Geometric statistical model to estimate the expected number
    of missed neighbors based on the "gap" (number of samples) since the 
    last improvement to the k-NN set.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The dataset to search.
    tolerance : float, default=0.5
        The acceptable expected number of missed neighbors.
        Lower values (e.g., 0.1) are stricter (higher recall, slower).
        Higher values (e.g., 1.0) are faster (lower recall).
    confidence : float, default=0.99
        Confidence level for the statistical bound (0.0 to 1.0).
    min_samples : int or None, default=None
        Minimum samples to scan before attempting to stop.
        If None, uses max(k + 50, 1% of n).
    distance_metric : str, default='euclidean'
        Distance metric: 'euclidean', 'cosine', or 'manhattan'.
    random_state : int or None, default=None
        Random seed for reproducibility.
    block_size : int, default=256
        Number of points to process in each vectorized block.
        Larger blocks amortize Python overhead but may compute extra distances.
        The stopping criterion is only checked once per block.
        Default of 256 balances vectorization benefits with stopping granularity.
    """

    def __init__(
        self,
        X: np.ndarray,
        tolerance: float = 0.5,
        confidence: float = 0.99,
        min_samples: Optional[int] = None,
        max_cv: Optional[float] = None,
        sorter: Optional[BaseSorter] = None,
        distance_metric: str = 'euclidean',
        random_state: Optional[int] = None,
        block_size: int = 256,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.tolerance = tolerance
        self.confidence = confidence
        self.min_samples = min_samples
        self.max_cv = max_cv
        self.distance_metric = distance_metric
        self.random_state = random_state
        self.block_size = block_size

        # Precompute for cosine efficiency
        if distance_metric == 'cosine':
            self.norms = np.linalg.norm(X, axis=1, keepdims=True)
            self.X_normalized = X / (self.norms + 1e-10)
        else:
            self.norms = None
            self.X_normalized = None

        # Set random generator
        self.rng = np.random.default_rng(random_state)

        # Initialize Sorter
        if sorter is None:
            # Default to random behavior if not specified
            # Pass the RNG to ensure reproducibility
            self.sorter = BaseSorter(rng=self.rng)
        else:
            self.sorter = sorter
            # If the sorter needs training and hasn't been fit, fit it now
            # (Note: PCASorter needs X to fit)
            if hasattr(self.sorter, 'fit') and getattr(self.sorter, 'X_reduced', None) is None:
                self.sorter.fit(self.X)

    def fit(self) -> 'DESKNNSearcher':
        """No-op for API compatibility with baselines."""
        return self

    def query(
        self,
        q: np.ndarray,
        k: int,
        return_distances: bool = True,
        return_stats: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int] | Tuple[np.ndarray, np.ndarray, int, Dict]:
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
            Distances to k nearest neighbors.
        dist_count : int
            Number of distance computations performed.
        stats : dict, optional
            Detailed statistics (if return_stats=True).
        """
        q = np.asarray(q, dtype=np.float32)
        n = self.n

        # Validate parameters
        if k > n:
            k = n

        # Determine minimum samples to scan
        if self.min_samples is not None:
            min_scan = self.min_samples
        else:
            # Default: at least k+50 or 1% of data, whichever is reasonable
            min_scan = max(k + 50, int(0.01 * n))
        
        # Ensure we don't scan more than N
        min_scan = min(min_scan, n)

        # Initialize Heap
        # Python's heapq is a min-heap. We store (-dist, idx) to simulate max-heap.
        heap = [] 
        d_k = np.inf
        
        # Statistics Tracking
        dist_count = 0
        last_update_idx = 0
        
        # Precompute query-specific values for distance
        q_normalized = None
        if self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q)
            q_normalized = q / (q_norm + 1e-10)

        # Generate random search order
        indices = self.sorter.get_sorted_indices(q, self.X)

        # Optimization: Localize variables for the hot loop
        compute_distances_batch = self._compute_distances_batch
        estimate_matches = estimate_future_matches
        tolerance = self.tolerance
        confidence = self.confidence
        block_size = self.block_size

        # Output stats collection
        stopped_early = False
        final_expected_misses = 0.0

        # Block-Based Main Search Loop
        # Process points in blocks to amortize Python overhead
        for block_start in range(0, n, block_size):
            block_end = min(block_start + block_size, n)
            block_indices = indices[block_start:block_end]

            # 1. Vectorized Distance Computation (fast BLAS operation)
            block_dists = compute_distances_batch(q, block_indices, q_normalized)
            dist_count += len(block_indices)

            # 2. Batch Heap Update
            # Process all distances in this block
            for j, (d, idx) in enumerate(zip(block_dists, block_indices)):
                global_idx = block_start + j  # Position in overall search order

                if len(heap) < k:
                    heapq.heappush(heap, (-d, idx))
                    if len(heap) == k:
                        d_k = -heap[0][0]
                        last_update_idx = global_idx
                elif d < d_k:
                    heapq.heapreplace(heap, (-d, idx))
                    d_k = -heap[0][0]
                    last_update_idx = global_idx

            # 3. Check Early Stopping (once per block, not per element)
            # Only check after processing minimum samples
            current_position = block_end - 1
            if current_position >= min_scan:
                current_gap = current_position - last_update_idx

                # Check 1: Large enough gap?
                if current_gap > k:
                    remaining = n - block_end
                    expected_misses = estimate_matches(current_gap, remaining, confidence)

                    # Check 2: Statistical Convergence
                    if expected_misses < tolerance:

                        # Check 3: Dispersion / Variance (Optional Safety)
                        should_stop = True
                        if self.max_cv is not None:
                            current_dists = np.array([-h[0] for h in heap])
                            mean_d = np.mean(current_dists)

                            if mean_d > 1e-9:
                                std_d = np.std(current_dists)
                                cv = std_d / mean_d
                                if cv > self.max_cv:
                                    should_stop = False

                        if should_stop:
                            stopped_early = True
                            final_expected_misses = expected_misses
                            break

        # Extract results
        # Sort heap by distance (ascending)
        heap.sort(key=lambda x: -x[0])
        
        neighbors = np.array([idx for _, idx in heap], dtype=np.int64)
        distances = np.array([-d for d, _ in heap], dtype=np.float32)

        if return_stats:
            stats = {
                'stopped_early': stopped_early,
                'scan_ratio': dist_count / n,
                'dist_count': dist_count,
                'last_update_index': last_update_idx,
                'final_gap': dist_count - 1 - last_update_idx,
                'expected_misses': final_expected_misses if stopped_early else 0.0
            }
            if return_distances:
                return neighbors, distances, dist_count, stats
            return neighbors, dist_count, stats

        if return_distances:
            return neighbors, distances, dist_count
        return neighbors, dist_count

    def _compute_distance_fast(
        self, 
        q: np.ndarray, 
        idx: int, 
        q_normalized: Optional[np.ndarray] = None
    ) -> float:
        """
        Optimized internal distance computation.
        Assumes q and idx are valid.
        """
        # Note: In a production C++ extension, this would be the critical path.
        # In Python, function call overhead is significant, but we keep it 
        # distinct for clarity and support for different metrics.
        
        if self.distance_metric == 'euclidean':
            # Direct numpy subtraction is faster than generic linalg.norm
            x = self.X[idx]
            d = q - x
            return np.sqrt(d.dot(d))
            
        elif self.distance_metric == 'manhattan':
            x = self.X[idx]
            return np.sum(np.abs(q - x))
            
        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - dot(q_norm, x_norm)
            if self.X_normalized is not None:
                # Use precomputed normalized X
                # q_normalized is passed from query()
                similarity = np.dot(q_normalized, self.X_normalized[idx])
                return 1.0 - similarity
            else:
                # Fallback (slow)
                x = self.X[idx]
                return 1.0 - (np.dot(q, x) / (np.linalg.norm(q) * np.linalg.norm(x) + 1e-10))
        
        return 0.0

    def _compute_distances_batch(
        self,
        q: np.ndarray,
        indices: np.ndarray,
        q_normalized: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Vectorized batch distance computation using BLAS operations.

        Computes distances from query q to all points at the given indices
        in a single vectorized operation.

        Parameters
        ----------
        q : np.ndarray of shape (n_features,)
            Query point.
        indices : np.ndarray of shape (batch_size,)
            Indices of database points to compute distances to.
        q_normalized : np.ndarray or None
            Pre-normalized query for cosine metric.

        Returns
        -------
        distances : np.ndarray of shape (batch_size,)
            Distances from q to each point in X[indices].
        """
        # Fetch batch of points: shape (batch_size, n_features)
        batch = self.X[indices]

        if self.distance_metric == 'euclidean':
            # Vectorized Euclidean: ||q - x||^2 = ||q||^2 + ||x||^2 - 2*q.x
            # Using the expansion for efficiency with BLAS
            diff = batch - q  # (batch_size, n_features)
            # Sum of squares along feature axis
            distances = np.sqrt(np.einsum('ij,ij->i', diff, diff))

        elif self.distance_metric == 'manhattan':
            # Vectorized Manhattan distance
            distances = np.sum(np.abs(batch - q), axis=1)

        elif self.distance_metric == 'cosine':
            # Cosine distance = 1 - dot(q_norm, x_norm)
            if self.X_normalized is not None and q_normalized is not None:
                batch_normalized = self.X_normalized[indices]
                # Batch dot product: (batch_size, d) @ (d,) -> (batch_size,)
                similarities = batch_normalized @ q_normalized
                distances = 1.0 - similarities
            else:
                # Fallback (should not happen in normal usage)
                norms = np.linalg.norm(batch, axis=1)
                q_norm = np.linalg.norm(q)
                similarities = (batch @ q) / (norms * q_norm + 1e-10)
                distances = 1.0 - similarities
        else:
            distances = np.zeros(len(indices), dtype=np.float32)

        return distances.astype(np.float32)

    def query_batch(
        self,
        queries: np.ndarray,
        k: int,
        n_jobs: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Query multiple points in parallel.

        Parameters
        ----------
        queries : np.ndarray of shape (n_queries, n_features)
            Query points.
        k : int
            Number of neighbors.
        n_jobs : int, default=1
            Number of parallel jobs.

        Returns
        -------
        all_neighbors : np.ndarray
        all_distances : np.ndarray
        all_dist_counts : np.ndarray
        """
        n_queries = len(queries)
        all_neighbors = np.zeros((n_queries, k), dtype=np.int64)
        all_distances = np.zeros((n_queries, k), dtype=np.float32)
        all_dist_counts = np.zeros(n_queries, dtype=np.int64)

        if n_jobs == 1:
            for i, q in enumerate(queries):
                n_idxs, dists, count = self.query(q, k, return_distances=True)
                all_neighbors[i] = n_idxs
                all_distances[i] = dists
                all_dist_counts[i] = count
        else:
            def _job(i, q):
                return i, self.query(q, k, return_distances=True)

            results = Parallel(n_jobs=n_jobs)(
                delayed(_job)(i, q) for i, q in enumerate(queries)
            )

            for i, (n_idxs, dists, count) in results:
                all_neighbors[i] = n_idxs
                all_distances[i] = dists
                all_dist_counts[i] = count

        return all_neighbors, all_distances, all_dist_counts

    def _compute_distance(self, q: np.ndarray, idx: int) -> float:
        """Legacy wrapper for single distance computation."""
        return self._compute_distance_fast(q, idx)