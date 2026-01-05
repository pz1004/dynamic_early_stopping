"""
Tests for baseline implementations.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.baselines import (
    ExactBruteForceKNN,
    KDTreeKNN,
    SklearnKDTreeKNN,
    BallTreeKNN,
    SklearnBallTreeKNN,
    LSHKNN,
    AnnoyKNN,
    HNSWKNN,
    FAISSIVFKNN
)


class TestExactBruteForce:
    """Test exact brute force implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        return np.random.randn(500, 20).astype(np.float32)

    @pytest.fixture
    def query_point(self, sample_data):
        """Create query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_basic_query(self, sample_data, query_point):
        """Test basic query functionality."""
        k = 10
        searcher = ExactBruteForceKNN(sample_data)
        searcher.fit()
        neighbors, distances, dist_count = searcher.query(query_point, k)

        assert len(neighbors) == k
        assert len(distances) == k
        assert dist_count == len(sample_data)  # Brute force computes all distances

    def test_distances_sorted(self, sample_data, query_point):
        """Test that distances are returned sorted."""
        k = 10
        searcher = ExactBruteForceKNN(sample_data)
        searcher.fit()
        _, distances, _ = searcher.query(query_point, k)

        # Distances should be non-decreasing
        for i in range(len(distances) - 1):
            assert distances[i] <= distances[i + 1]

    def test_distance_metrics(self, sample_data, query_point):
        """Test different distance metrics."""
        k = 10

        for metric in ['euclidean', 'manhattan', 'cosine']:
            searcher = ExactBruteForceKNN(sample_data, distance_metric=metric)
            searcher.fit()
            neighbors, distances, _ = searcher.query(query_point, k)

            assert len(neighbors) == k
            assert all(d >= 0 for d in distances)


class TestKDTree:
    """Test KD-Tree implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset (low dimensional for KD-Tree)."""
        np.random.seed(42)
        return np.random.randn(500, 10).astype(np.float32)

    @pytest.fixture
    def query_point(self, sample_data):
        """Create query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_kdtree_exact(self, sample_data, query_point):
        """Test that KD-Tree gives exact results (no backtrack limit)."""
        k = 10

        # Exact brute force
        exact = ExactBruteForceKNN(sample_data)
        exact.fit()
        exact_neighbors, _, _ = exact.query(query_point, k)

        # KD-Tree
        kdtree = KDTreeKNN(sample_data)
        kdtree.fit()
        tree_neighbors, _, _ = kdtree.query(query_point, k)

        assert set(tree_neighbors) == set(exact_neighbors)

    def test_kdtree_build_time(self, sample_data):
        """Test that build time is recorded."""
        searcher = KDTreeKNN(sample_data)
        searcher.fit()
        assert searcher.build_time > 0


class TestBallTree:
    """Test Ball Tree implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        return np.random.randn(500, 20).astype(np.float32)

    @pytest.fixture
    def query_point(self, sample_data):
        """Create query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_balltree_exact(self, sample_data, query_point):
        """Test that Ball Tree gives exact results."""
        k = 10

        # Exact brute force
        exact = ExactBruteForceKNN(sample_data)
        exact.fit()
        exact_neighbors, _, _ = exact.query(query_point, k)

        # Ball Tree
        balltree = BallTreeKNN(sample_data)
        balltree.fit()
        tree_neighbors, _, _ = balltree.query(query_point, k)

        assert set(tree_neighbors) == set(exact_neighbors)


class TestLSH:
    """Test LSH implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        return np.random.randn(1000, 50).astype(np.float32)

    @pytest.fixture
    def query_point(self, sample_data):
        """Create query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_lsh_query(self, sample_data, query_point):
        """Test LSH query returns results."""
        k = 10
        searcher = LSHKNN(sample_data, n_tables=10, n_bits=8, random_state=42)
        searcher.fit()
        neighbors, distances, dist_count = searcher.query(query_point, k)

        assert len(neighbors) == k
        assert len(distances) == k
        assert dist_count < len(sample_data)  # Should be approximate


class TestAnnoy:
    """Test Annoy implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        return np.random.randn(1000, 50).astype(np.float32)

    @pytest.fixture
    def query_point(self, sample_data):
        """Create query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_annoy_query(self, sample_data, query_point):
        """Test Annoy query returns results."""
        k = 10
        searcher = AnnoyKNN(sample_data, n_trees=10, random_state=42)
        searcher.fit()
        neighbors, distances, dist_count = searcher.query(query_point, k)

        assert len(neighbors) == k
        assert len(distances) == k


class TestHNSW:
    """Test HNSW implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        return np.random.randn(500, 30).astype(np.float32)

    @pytest.fixture
    def query_point(self, sample_data):
        """Create query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_hnsw_query(self, sample_data, query_point):
        """Test HNSW query returns results."""
        k = 10
        searcher = HNSWKNN(sample_data, M=8, ef_construction=100, ef=30, random_state=42)
        searcher.fit()
        neighbors, distances, _ = searcher.query(query_point, k)

        assert len(neighbors) == k
        assert len(distances) == k


class TestFAISSIVF:
    """Test FAISS IVF implementation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        return np.random.randn(1000, 50).astype(np.float32)

    @pytest.fixture
    def query_point(self, sample_data):
        """Create query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_faiss_ivf_query(self, sample_data, query_point):
        """Test FAISS IVF query returns results."""
        k = 10
        searcher = FAISSIVFKNN(sample_data, nlist=20, nprobe=5)
        searcher.fit()
        neighbors, distances, dist_count = searcher.query(query_point, k)

        assert len(neighbors) == k
        assert len(distances) == k


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
