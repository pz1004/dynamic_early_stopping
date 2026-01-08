"""
HNSW - Hierarchical Navigable Small World Graph

HNSW builds a multi-layer graph where:
- Bottom layer contains all points
- Higher layers contain a subset (exponentially fewer points)
- Search starts from top layer and greedily descends

This is currently the state-of-the-art for high-dimensional ANN.

Key parameters:
- M: number of connections per node (default 16)
- ef_construction: beam width during construction (default 200)
- ef: beam width during search (controls recall/speed tradeoff)
"""

import numpy as np
import time
from typing import Tuple, Optional, List, Dict, Set
import heapq
from collections import defaultdict

from .exact_brute_force import BaseKNNSearcher


class HNSWLayer:
    """Single layer of the HNSW graph."""

    def __init__(self):
        # Adjacency list: node_id -> list of neighbor node_ids
        self.neighbors: Dict[int, List[int]] = defaultdict(list)


class HNSWIndexPure:
    """
    Pure Python implementation of HNSW (for understanding).

    Parameters
    ----------
    d : int
        Dimensionality of vectors.
    M : int, default=16
        Number of bi-directional links created for each element.
    ef_construction : int, default=200
        Size of dynamic candidate list during construction.
    ml : float, default=1/ln(M)
        Level multiplier (probability of appearing in higher layer).
    """

    def __init__(
        self,
        d: int,
        M: int = 16,
        ef_construction: int = 200,
        random_state: Optional[int] = None
    ):
        self.d = d
        self.M = M
        self.M0 = 2 * M  # Max connections at layer 0
        self.ef_construction = ef_construction
        self.ml = 1.0 / np.log(M)

        self.rng = np.random.default_rng(random_state)

        self.layers: List[HNSWLayer] = []
        self.entry_point: Optional[int] = None
        self.max_level = -1
        self.point_levels: Dict[int, int] = {}  # node_id -> max layer

        self.data: Optional[np.ndarray] = None

    def _random_level(self) -> int:
        """Generate random level for a new node."""
        level = 0
        while self.rng.random() < np.exp(-level / self.ml) and level < 32:
            level += 1
        return level

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Euclidean distance."""
        diff = a - b
        return np.sqrt(np.dot(diff, diff))

    def _search_layer(
        self,
        q: np.ndarray,
        entry_points: List[int],
        ef: int,
        layer: int
    ) -> List[Tuple[float, int]]:
        """
        Search a single layer of the graph.

        Returns list of (distance, node_id) sorted by distance.
        """
        visited = set(entry_points)

        # Priority queues
        # candidates: min-heap of (distance, node_id) - nodes to explore
        # results: max-heap of (-distance, node_id) - best ef nodes found

        candidates = []
        results = []

        for ep in entry_points:
            d = self._distance(q, self.data[ep])
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            d_c, c = heapq.heappop(candidates)

            # Check if we can stop
            if len(results) >= ef and d_c > -results[0][0]:
                break

            # Explore neighbors
            if c in self.layers[layer].neighbors:
                for neighbor in self.layers[layer].neighbors[c]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        d_n = self._distance(q, self.data[neighbor])

                        if len(results) < ef or d_n < -results[0][0]:
                            heapq.heappush(candidates, (d_n, neighbor))
                            heapq.heappush(results, (-d_n, neighbor))

                            if len(results) > ef:
                                heapq.heappop(results)

        # Convert results to sorted list
        results = [(-d, idx) for d, idx in results]
        results.sort()
        return results

    def _select_neighbors(
        self,
        q: np.ndarray,
        candidates: List[Tuple[float, int]],
        M: int
    ) -> List[int]:
        """Select M best neighbors from candidates using simple selection."""
        return [idx for _, idx in candidates[:M]]

    def fit(self, X: np.ndarray) -> 'HNSWIndexPure':
        """Build the HNSW index."""
        self.data = np.asarray(X, dtype=np.float32)
        n = len(X)

        for i in range(n):
            self._insert(i)

        return self

    def _insert(self, node_id: int):
        """Insert a single node into the index."""
        q = self.data[node_id]
        level = self._random_level()
        self.point_levels[node_id] = level

        # Ensure we have enough layers
        while len(self.layers) <= level:
            self.layers.append(HNSWLayer())

        if self.entry_point is None:
            # First node
            self.entry_point = node_id
            self.max_level = level
            return

        # Find entry point at each layer
        curr_entry = [self.entry_point]

        # Traverse from top layer down to level + 1
        for lc in range(self.max_level, level, -1):
            if lc < len(self.layers):
                results = self._search_layer(q, curr_entry, 1, lc)
                if results:
                    curr_entry = [results[0][1]]

        # Insert at layers level down to 0
        for lc in range(min(level, self.max_level), -1, -1):
            M_layer = self.M0 if lc == 0 else self.M

            results = self._search_layer(q, curr_entry, self.ef_construction, lc)
            neighbors = self._select_neighbors(q, results, M_layer)

            # Add bidirectional edges
            self.layers[lc].neighbors[node_id] = neighbors

            for neighbor in neighbors:
                neighbor_connections = self.layers[lc].neighbors[neighbor]
                if len(neighbor_connections) < M_layer:
                    neighbor_connections.append(node_id)
                else:
                    # Keep only M closest neighbors
                    candidates = [(self._distance(self.data[neighbor], self.data[n]), n)
                                 for n in neighbor_connections + [node_id]]
                    candidates.sort()
                    self.layers[lc].neighbors[neighbor] = [n for _, n in candidates[:M_layer]]

            curr_entry = [r[1] for r in results[:1]]

        if level > self.max_level:
            self.max_level = level
            self.entry_point = node_id

    def search(
        self,
        q: np.ndarray,
        k: int,
        ef: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Parameters
        ----------
        q : np.ndarray
            Query vector.
        k : int
            Number of neighbors.
        ef : int
            Search beam width (ef >= k).
        """
        if self.entry_point is None:
            return np.array([]), np.array([])

        q = np.asarray(q, dtype=np.float32)
        ef = max(ef, k)

        curr_entry = [self.entry_point]

        # Traverse from top to layer 1
        for lc in range(self.max_level, 0, -1):
            results = self._search_layer(q, curr_entry, 1, lc)
            if results:
                curr_entry = [results[0][1]]

        # Search bottom layer
        results = self._search_layer(q, curr_entry, ef, 0)

        # Return top k
        results = results[:k]
        distances = np.array([d for d, _ in results])
        neighbors = np.array([idx for _, idx in results])

        return neighbors, distances


class HNSWKNN(BaseKNNSearcher):
    """
    HNSW-based approximate k-NN search.

    Parameters
    ----------
    X : np.ndarray
        Dataset of shape (n_samples, n_features).
    M : int, default=16
        Number of connections per node.
    ef_construction : int, default=200
        Beam width during construction.
    ef : int, default=10
        Beam width during search.
    use_library : bool, default=True
        Whether to use hnswlib (fast baseline). Falls back to pure Python if False.
    random_state : int or None, default=None
        Random seed.
    """

    def __init__(
        self,
        X: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 10,
        use_library: bool = True,
        random_state: Optional[int] = None
    ):
        super().__init__(X)
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.use_library = use_library
        self.random_state = random_state

        self.index = None
        self._using_library = False

    def fit(self) -> 'HNSWKNN':
        """Build the HNSW index."""
        t0 = time.perf_counter()

        if self.use_library:
            try:
                import hnswlib
            except ImportError as exc:
                raise ImportError(
                    "hnswlib is required for the fast HNSW baseline. "
                    "Install it or set use_library=False to use the pure Python reference."
                ) from exc

            self.index = hnswlib.Index(space='l2', dim=self.d)
            self.index.init_index(
                max_elements=self.n,
                ef_construction=self.ef_construction,
                M=self.M
            )
            self.index.add_items(self.X, np.arange(self.n))
            self.index.set_ef(self.ef)
            self._using_library = True
        else:
            self.index = HNSWIndexPure(
                d=self.d,
                M=self.M,
                ef_construction=self.ef_construction,
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

        if self._using_library:
            neighbors, distances = self.index.knn_query(q.reshape(1, -1), k=k)
            return neighbors[0], np.sqrt(distances[0]), self.ef  # hnswlib returns squared distances
        else:
            neighbors, distances = self.index.search(q, k, self.ef)
            return neighbors, distances, self.ef
