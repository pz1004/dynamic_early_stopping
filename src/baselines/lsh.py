"""
Locality Sensitive Hashing (LSH) for Approximate k-NN

LSH uses hash functions that map similar items to the same bucket
with high probability. For Euclidean distance, we use random hyperplane LSH.

Key concepts:
- Hash functions: random projections that bucket similar points together
- Multiple hash tables: increase recall by using multiple independent hash functions
- AND/OR amplification: combine hash functions to tune precision/recall tradeoff

Parameters trade-offs:
- More hash tables -> higher recall, more memory and build time
- More hash bits per table -> higher precision, lower recall per table
- More probes -> higher recall, slower query time
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Set, Optional
from collections import defaultdict
import heapq

from .exact_brute_force import BaseKNNSearcher


class LSHIndex:
    """
    Random Hyperplane LSH for Euclidean/Cosine similarity.

    Parameters
    ----------
    n_tables : int, default=10
        Number of hash tables.
    n_bits : int, default=10
        Number of bits per hash (hash functions per table).
    random_state : int or None, default=None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_tables: int = 10,
        n_bits: int = 10,
        random_state: Optional[int] = None
    ):
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.random_state = random_state

        self.rng = np.random.default_rng(random_state)

        # Will be set during fit
        self.d = None
        self.hyperplanes = None  # Shape: (n_tables, n_bits, d)
        self.hash_tables: List[Dict[int, List[int]]] = None

    def _init_hyperplanes(self, d: int):
        """Initialize random hyperplanes for hashing."""
        self.d = d
        # Each hyperplane is a random unit vector
        self.hyperplanes = self.rng.standard_normal((self.n_tables, self.n_bits, d))
        # Normalize each hyperplane
        norms = np.linalg.norm(self.hyperplanes, axis=2, keepdims=True)
        self.hyperplanes = self.hyperplanes / norms

    def _hash_point(self, point: np.ndarray, table_idx: int) -> int:
        """
        Compute hash value for a point in a specific table.

        Uses sign of dot product with each hyperplane as hash bit.
        """
        projections = np.dot(self.hyperplanes[table_idx], point)
        bits = (projections >= 0).astype(np.int32)
        # Convert binary to integer
        hash_value = 0
        for bit in bits:
            hash_value = (hash_value << 1) | bit
        return hash_value

    def fit(self, X: np.ndarray) -> 'LSHIndex':
        """Build the LSH index."""
        X = np.asarray(X, dtype=np.float32)
        n, d = X.shape

        self._init_hyperplanes(d)

        # Initialize hash tables
        self.hash_tables = [defaultdict(list) for _ in range(self.n_tables)]

        # Hash all points
        for idx in range(n):
            for t in range(self.n_tables):
                h = self._hash_point(X[idx], t)
                self.hash_tables[t][h].append(idx)

        return self

    def query_candidates(
        self,
        point: np.ndarray,
        n_probes: int = 1
    ) -> Set[int]:
        """
        Get candidate neighbors from hash tables.

        Parameters
        ----------
        point : np.ndarray
            Query point.
        n_probes : int, default=1
            Number of buckets to probe per table (multi-probe LSH).

        Returns
        -------
        candidates : Set[int]
            Set of candidate point indices.
        """
        candidates = set()

        for t in range(self.n_tables):
            h = self._hash_point(point, t)

            # Primary bucket
            candidates.update(self.hash_tables[t].get(h, []))

            # Multi-probe: check nearby buckets (flip individual bits)
            if n_probes > 1:
                for bit in range(min(n_probes - 1, self.n_bits)):
                    h_flipped = h ^ (1 << bit)
                    candidates.update(self.hash_tables[t].get(h_flipped, []))

        return candidates


class LSHKNN(BaseKNNSearcher):
    """
    LSH-based approximate k-NN search.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    n_tables : int, default=10
        Number of hash tables. More tables = higher recall.
    n_bits : int, default=10
        Bits per hash. More bits = more selective hashing.
    n_probes : int, default=1
        Number of buckets to probe per table during query.
    distance_metric : str, default='euclidean'
        Distance metric for candidate verification.
    random_state : int or None, default=None
        Random seed.
    """

    def __init__(
        self,
        X: np.ndarray,
        n_tables: int = 10,
        n_bits: int = 10,
        n_probes: int = 1,
        distance_metric: str = 'euclidean',
        random_state: Optional[int] = None
    ):
        super().__init__(X)
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.n_probes = n_probes
        self.distance_metric = distance_metric
        self.random_state = random_state

        self.index = LSHIndex(n_tables, n_bits, random_state)

    def fit(self) -> 'LSHKNN':
        """Build the LSH index."""
        t0 = time.perf_counter()
        self.index.fit(self.X)
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

        # Get candidates from LSH
        candidates = self.index.query_candidates(q, self.n_probes)

        if len(candidates) == 0:
            # No candidates found, fall back to random sample
            candidates = set(np.random.choice(self.n, min(k * 10, self.n), replace=False))

        candidates = list(candidates)
        dist_count = len(candidates)

        # Compute actual distances to candidates
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X[candidates] - q) ** 2, axis=1))
        elif self.distance_metric == 'cosine':
            q_norm = np.linalg.norm(q)
            dots = np.dot(self.X[candidates], q)
            norms = np.linalg.norm(self.X[candidates], axis=1)
            distances = 1 - dots / (norms * q_norm + 1e-10)
        else:
            distances = np.sum(np.abs(self.X[candidates] - q), axis=1)

        # Find k nearest among candidates
        if len(candidates) <= k:
            sorted_order = np.argsort(distances)
            neighbors = np.array(candidates)[sorted_order]
            distances = distances[sorted_order]
            # Pad if necessary
            if len(neighbors) < k:
                neighbors = np.pad(neighbors, (0, k - len(neighbors)), constant_values=-1)
                distances = np.pad(distances, (0, k - len(distances)), constant_values=np.inf)
        else:
            top_k_indices = np.argpartition(distances, k)[:k]
            sorted_order = np.argsort(distances[top_k_indices])
            neighbors = np.array(candidates)[top_k_indices[sorted_order]]
            distances = distances[top_k_indices[sorted_order]]

        return neighbors, distances, dist_count
