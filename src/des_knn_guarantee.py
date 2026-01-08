"""
DES-kNN (Guarantee Variant)

This variant enforces the Beta-Geometric assumptions by:
1) Using a random-order scan (no PCA/cluster ordering).
2) Freezing a reference distance once min_scan is reached so the "success"
   probability is stationary for the bound.

In this file, `expected_misses` refers to the expected number of remaining points
that would beat the fixed `reference_distance` (not the continuously-updating
current k-th distance), at the configured confidence level.
"""

import numpy as np
import heapq
from typing import Tuple, Optional, Dict

from .statistics import estimate_future_matches
from .sorting import BaseSorter, RandomOrderSorter


class DESKNNSearcherGuarantee:
    """
    Dynamic Early Stopping k-NN with Beta-Geometric guarantee assumptions.

    This enforces a random scan order and uses a fixed reference threshold
    for the gap model after min_scan to make the success probability stationary.
    """

    def __init__(
        self,
        X: np.ndarray,
        tolerance: float = 0.5,
        confidence: float = 0.99,
        min_samples: Optional[int] = None,
        distance_metric: str = 'euclidean',
        random_state: Optional[int] = None,
        block_size: int = 256,
        sorter: Optional[BaseSorter] = None,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.n, self.d = self.X.shape
        self.tolerance = tolerance
        self.confidence = confidence
        self.min_samples = min_samples
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

        self.rng = np.random.default_rng(random_state)

        if sorter is None:
            self.sorter = RandomOrderSorter(rng=self.rng)
        else:
            if not isinstance(sorter, RandomOrderSorter):
                raise ValueError(
                    "DESKNNSearcherGuarantee requires RandomOrderSorter to keep "
                    "the Beta-Geometric assumptions valid."
                )
            self.sorter = sorter

    def fit(self) -> 'DESKNNSearcherGuarantee':
        """No-op for API compatibility with baselines."""
        return self

    def query(
        self,
        q: np.ndarray,
        k: int,
        return_distances: bool = True,
        return_stats: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, int] | Tuple[np.ndarray, np.ndarray, int, Dict]:
        """Find k nearest neighbors with a conservative Beta-Geometric stop."""
        q = np.asarray(q, dtype=np.float32)
        n = self.n

        if k > n:
            k = n

        if self.min_samples is not None:
            min_scan = self.min_samples
        else:
            min_scan = max(k + 50, int(0.01 * n))
        min_scan = min(min_scan, n)

        heap = []
        d_k = np.inf

        dist_count = 0

        q_normalized = None
        if self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q)
            q_normalized = q / (q_norm + 1e-10)

        indices = self.sorter.get_sorted_indices(q, self.X)

        compute_distances_batch = self._compute_distances_batch
        estimate_matches = estimate_future_matches
        tolerance = self.tolerance
        confidence = self.confidence
        block_size = self.block_size

        stopped_early = False
        final_expected_misses = 0.0

        ref_distance = None
        ref_set_at = None
        last_success_idx = None

        for block_start in range(0, n, block_size):
            block_end = min(block_start + block_size, n)
            block_indices = indices[block_start:block_end]

            block_dists = compute_distances_batch(q, block_indices, q_normalized)
            dist_count += len(block_indices)

            for j, (d, idx) in enumerate(zip(block_dists, block_indices)):
                global_idx = block_start + j

                if len(heap) < k:
                    heapq.heappush(heap, (-d, idx))
                    if len(heap) == k:
                        d_k = -heap[0][0]
                elif d < d_k:
                    heapq.heapreplace(heap, (-d, idx))
                    d_k = -heap[0][0]

                if ref_distance is not None and d < ref_distance:
                    last_success_idx = global_idx

            current_position = block_end - 1

            if ref_distance is None:
                if current_position >= min_scan and len(heap) == k:
                    ref_distance = d_k
                    ref_set_at = current_position
                    last_success_idx = current_position
            else:
                gap = current_position - last_success_idx
                if gap > k:
                    remaining = n - block_end
                    expected_misses = estimate_matches(gap, remaining, confidence)
                    if expected_misses < tolerance:
                        stopped_early = True
                        final_expected_misses = expected_misses
                        break

        heap.sort(key=lambda x: -x[0])
        neighbors = np.array([idx for _, idx in heap], dtype=np.int64)
        distances = np.array([-d for d, _ in heap], dtype=np.float32)

        if return_stats:
            expected_misses_reference = final_expected_misses if stopped_early else 0.0
            stats = {
                'stopped_early': stopped_early,
                'scan_ratio': dist_count / n,
                'dist_count': dist_count,
                'reference_distance': ref_distance if ref_distance is not None else 0.0,
                'reference_set_at': ref_set_at if ref_set_at is not None else -1,
                'final_gap': (dist_count - 1 - last_success_idx) if last_success_idx is not None else 0,
                'expected_misses': expected_misses_reference,
                'expected_misses_reference': expected_misses_reference,
                'expected_misses_definition': (
                    "Expected number of remaining points with dist(q, x) < reference_distance "
                    "at the configured confidence level."
                )
            }
            if return_distances:
                return neighbors, distances, dist_count, stats
            return neighbors, dist_count, stats

        if return_distances:
            return neighbors, distances, dist_count
        return neighbors, dist_count

    def _compute_distances_batch(
        self,
        q: np.ndarray,
        indices: np.ndarray,
        q_normalized: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Vectorized batch distance computation."""
        batch = self.X[indices]

        if self.distance_metric == 'euclidean':
            diff = batch - q
            distances = np.sqrt(np.einsum('ij,ij->i', diff, diff))

        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(batch - q), axis=1)

        elif self.distance_metric == 'cosine':
            if self.X_normalized is not None and q_normalized is not None:
                batch_normalized = self.X_normalized[indices]
                similarities = batch_normalized @ q_normalized
                distances = 1.0 - similarities
            else:
                norms = np.linalg.norm(batch, axis=1)
                q_norm = np.linalg.norm(q)
                similarities = (batch @ q) / (norms * q_norm + 1e-10)
                distances = 1.0 - similarities
        else:
            distances = np.zeros(len(indices), dtype=np.float32)

        return distances.astype(np.float32)
