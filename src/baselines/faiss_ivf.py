"""
FAISS IVF - Inverted File Index

IVF partitions the dataset using k-means clustering, then only searches
within nearby clusters during query time.

Key parameters:
- nlist: number of clusters (more = finer partitioning)
- nprobe: number of clusters to search (more = higher recall)

NOTE: Provides wrapper for FAISS library and pure Python fallback.
"""

import numpy as np
import time
from typing import Tuple, Optional, Dict, List
from sklearn.cluster import KMeans
import heapq

from .exact_brute_force import BaseKNNSearcher


class IVFIndexPure:
    """
    Pure Python implementation of IVF index.

    Parameters
    ----------
    d : int
        Dimensionality.
    nlist : int
        Number of clusters.
    """

    def __init__(self, d: int, nlist: int = 100):
        self.d = d
        self.nlist = nlist
        self.centroids: Optional[np.ndarray] = None
        self.inverted_lists: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    def fit(self, X: np.ndarray) -> 'IVFIndexPure':
        """Build the IVF index using k-means."""
        n = len(X)

        # Train k-means for clustering
        kmeans = KMeans(
            n_clusters=min(self.nlist, n),
            random_state=42,
            n_init=1,
            max_iter=20
        )
        labels = kmeans.fit_predict(X)
        self.centroids = kmeans.cluster_centers_

        # Build inverted lists
        self.inverted_lists = {i: [] for i in range(len(self.centroids))}
        for idx, label in enumerate(labels):
            self.inverted_lists[label].append((idx, X[idx]))

        return self

    def search(
        self,
        q: np.ndarray,
        k: int,
        nprobe: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors."""
        # Find nprobe nearest centroids
        centroid_dists = np.sqrt(np.sum((self.centroids - q) ** 2, axis=1))
        probe_clusters = np.argpartition(centroid_dists, min(nprobe, len(centroid_dists) - 1))[:nprobe]

        # Search within those clusters
        candidates = []
        for cluster_id in probe_clusters:
            for idx, point in self.inverted_lists[cluster_id]:
                dist = np.sqrt(np.sum((point - q) ** 2))
                candidates.append((dist, idx))

        # Find k nearest
        if len(candidates) == 0:
            return np.array([]), np.array([])

        if len(candidates) <= k:
            candidates.sort()
            distances = np.array([d for d, _ in candidates])
            neighbors = np.array([idx for _, idx in candidates])
        else:
            candidates.sort()
            candidates = candidates[:k]
            distances = np.array([d for d, _ in candidates])
            neighbors = np.array([idx for _, idx in candidates])

        return neighbors, distances


class FAISSIVFKNN(BaseKNNSearcher):
    """
    FAISS IVF-based approximate k-NN search.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    nlist : int, default=100
        Number of clusters.
    nprobe : int, default=10
        Number of clusters to search during query.
    use_library : bool, default=False
        Whether to use FAISS library (if installed).
    """

    def __init__(
        self,
        X: np.ndarray,
        nlist: int = 100,
        nprobe: int = 10,
        use_library: bool = False
    ):
        super().__init__(X)
        self.nlist = nlist
        self.nprobe = nprobe
        self.use_library = use_library

        self.index = None
        self._using_library = False

    def fit(self) -> 'FAISSIVFKNN':
        """Build the IVF index."""
        t0 = time.perf_counter()

        if self.use_library:
            try:
                import faiss

                quantizer = faiss.IndexFlatL2(self.d)
                self.index = faiss.IndexIVFFlat(quantizer, self.d, self.nlist)
                self.index.train(self.X)
                self.index.add(self.X)
                self.index.nprobe = self.nprobe
                self._using_library = True

            except ImportError:
                self.index = IVFIndexPure(d=self.d, nlist=self.nlist)
                self.index.fit(self.X)
                self._using_library = False
        else:
            self.index = IVFIndexPure(d=self.d, nlist=self.nlist)
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

        if self._using_library:
            distances, neighbors = self.index.search(q.reshape(1, -1), k)
            return neighbors[0], distances[0], self.nprobe * (self.n // self.nlist)
        else:
            neighbors, distances = self.index.search(q, k, self.nprobe)
            return neighbors, distances, self.nprobe * (self.n // self.nlist)
