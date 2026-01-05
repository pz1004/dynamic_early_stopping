# Baseline Implementations - Part 2: Approximate Nearest Neighbor Methods

## Overview

This document covers approximate nearest neighbor (ANN) methods that trade accuracy for speed. These are important baselines as they represent the state-of-the-art for large-scale nearest neighbor search.

---

## 4. Locality Sensitive Hashing (LSH)

### File: `src/baselines/lsh.py`

```python
"""
Locality Sensitive Hashing (LSH) for Approximate k-NN

LSH uses hash functions that map similar items to the same bucket
with high probability. For Euclidean distance, we use random hyperplane LSH.

Key concepts:
- Hash functions: random projections that bucket similar points together
- Multiple hash tables: increase recall by using multiple independent hash functions
- AND/OR amplification: combine hash functions to tune precision/recall tradeoff

Parameters trade-offs:
- More hash tables → higher recall, more memory and build time
- More hash bits per table → higher precision, lower recall per table
- More probes → higher recall, slower query time
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Set
from collections import defaultdict
import heapq


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
    n_tables : int, default=20
        Number of hash tables. More tables = higher recall.
    n_bits : int, default=12
        Bits per hash. More bits = more selective hashing.
    n_probes : int, default=3
        Number of buckets to probe per table during query.
    distance_metric : str, default='euclidean'
        Distance metric for candidate verification.
    random_state : int or None, default=None
        Random seed.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        n_tables: int = 20,
        n_bits: int = 12,
        n_probes: int = 3,
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
```

---

## 5. Annoy (Approximate Nearest Neighbors Oh Yeah)

### File: `src/baselines/annoy_wrapper.py`

```python
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
            node.hyperplane = node.hyperplane / (np.linalg.norm(node.hyperplane) + 1e-10)
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
```

---

## 6. HNSW (Hierarchical Navigable Small World)

### File: `src/baselines/hnsw.py`

```python
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
    ef : int, default=50
        Beam width during search.
    use_library : bool, default=False
        Whether to use hnswlib (if installed).
    random_state : int or None, default=None
        Random seed.
    """
    
    def __init__(
        self,
        X: np.ndarray,
        M: int = 16,
        ef_construction: int = 200,
        ef: int = 50,
        use_library: bool = False,
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
                
                self.index = hnswlib.Index(space='l2', dim=self.d)
                self.index.init_index(
                    max_elements=self.n,
                    ef_construction=self.ef_construction,
                    M=self.M
                )
                self.index.add_items(self.X, np.arange(self.n))
                self.index.set_ef(self.ef)
                self._using_library = True
                
            except ImportError:
                self.index = HNSWIndexPure(
                    d=self.d,
                    M=self.M,
                    ef_construction=self.ef_construction,
                    random_state=self.random_state
                )
                self.index.fit(self.X)
                self._using_library = False
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
```

---

## 7. FAISS IVF (Inverted File Index)

### File: `src/baselines/faiss_ivf.py`

```python
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
from typing import Tuple, Optional
from sklearn.cluster import KMeans
import heapq


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
        probe_clusters = np.argpartition(centroid_dists, nprobe)[:nprobe]
        
        # Search within those clusters
        candidates = []
        for cluster_id in probe_clusters:
            for idx, point in self.inverted_lists[cluster_id]:
                dist = np.sqrt(np.sum((point - q) ** 2))
                candidates.append((dist, idx))
        
        # Find k nearest
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
```

---

## Summary: All Baselines

```python
# src/baselines/__init__.py

from .exact_brute_force import ExactBruteForceKNN
from .kdtree import KDTreeKNN, SklearnKDTreeKNN
from .balltree import BallTreeKNN, SklearnBallTreeKNN
from .lsh import LSHKNN
from .annoy_wrapper import AnnoyKNN
from .hnsw import HNSWKNN
from .faiss_ivf import FAISSIVFKNN

__all__ = [
    'ExactBruteForceKNN',
    'KDTreeKNN',
    'SklearnKDTreeKNN',
    'BallTreeKNN',
    'SklearnBallTreeKNN',
    'LSHKNN',
    'AnnoyKNN',
    'HNSWKNN',
    'FAISSIVFKNN',
]
```
