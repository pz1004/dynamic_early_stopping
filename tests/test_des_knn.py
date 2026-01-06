"""
Tests for DES-kNN implementation (Gap + CV Version).
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.des_knn import DESKNNSearcher
from src.baselines.exact_brute_force import ExactBruteForceKNN


class TestDESKNN:
    """Test suite for DES-kNN."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        # Create clustered data to test dispersion check
        X1 = np.random.randn(500, 50).astype(np.float32)
        X2 = np.random.randn(500, 50).astype(np.float32) + 10.0
        return np.vstack([X1, X2])

    @pytest.fixture
    def query_point(self, sample_data):
        """Create sample query point."""
        np.random.seed(123)
        return np.random.randn(sample_data.shape[1]).astype(np.float32)

    def test_des_knn_basic(self, sample_data, query_point):
        """Test basic functionality returns correct number of neighbors."""
        k = 10
        searcher = DESKNNSearcher(sample_data, random_state=42)
        neighbors, distances, dist_count = searcher.query(query_point, k)

        assert len(neighbors) == k
        assert len(distances) == k
        assert dist_count <= len(sample_data)
        assert dist_count > 0

    def test_des_knn_exact_small_dataset(self):
        """On tiny dataset, should return exact k-NN."""
        np.random.seed(42)
        X = np.random.randn(50, 10).astype(np.float32)
        q = np.random.randn(10).astype(np.float32)
        k = 5

        # DES-kNN with strict tolerance
        des_searcher = DESKNNSearcher(X, tolerance=0.001, random_state=42)
        des_neighbors, _, _ = des_searcher.query(q, k)

        # Exact
        exact_searcher = ExactBruteForceKNN(X)
        exact_searcher.fit()
        exact_neighbors, _, _ = exact_searcher.query(q, k)

        assert set(des_neighbors) == set(exact_neighbors)

    def test_des_knn_high_recall(self, sample_data):
        """Test recall meets threshold."""
        k = 10
        n_queries = 20
        exact_searcher = ExactBruteForceKNN(sample_data)
        exact_searcher.fit()
        
        # Use reasonable tolerance
        des_searcher = DESKNNSearcher(sample_data, tolerance=0.1, random_state=42)

        recalls = []
        for seed in range(n_queries):
            np.random.seed(seed)
            q = np.random.randn(sample_data.shape[1]).astype(np.float32)

            exact_neighbors, _, _ = exact_searcher.query(q, k)
            des_neighbors, _, _ = des_searcher.query(q, k)

            recall = len(set(des_neighbors) & set(exact_neighbors)) / k
            recalls.append(recall)

        assert np.mean(recalls) >= 0.95

    def test_des_knn_speedup(self):
        """Test that early stopping provides speedup on clustered data."""
        # Use clustered data where early stopping is more effective.
        # Uniform random data in high dimensions has similar distances everywhere,
        # making early stopping hard. Clustered data has clear "easy" queries.
        np.random.seed(42)
        n = 5000
        d = 50

        # Create clustered data: 10 clusters
        n_clusters = 10
        points_per_cluster = n // n_clusters
        X = []
        for i in range(n_clusters):
            center = np.random.randn(d) * 10  # Spread clusters apart
            cluster = center + np.random.randn(points_per_cluster, d) * 0.5
            X.append(cluster)
        X = np.vstack(X).astype(np.float32)

        # Query near one cluster center (easy query)
        q = X[0] + np.random.randn(d).astype(np.float32) * 0.1
        k = 10

        # Reasonable tolerance and confidence
        searcher = DESKNNSearcher(X, tolerance=0.5, confidence=0.99, random_state=42)

        _, _, dist_count = searcher.query(q, k)

        # Should check fewer than all N points for an easy query
        assert dist_count < n, f"Expected some speedup but checked all {dist_count}/{n} points"

    def test_des_knn_return_stats(self, sample_data, query_point):
        """Test return_stats option."""
        k = 10
        searcher = DESKNNSearcher(sample_data, random_state=42)
        neighbors, distances, dist_count, stats = searcher.query(
            query_point, k, return_stats=True
        )

        assert 'stopped_early' in stats
        assert 'scan_ratio' in stats
        assert 'expected_misses' in stats
        assert len(neighbors) == k

    def test_des_knn_cv_check(self, sample_data):
        """Test that max_cv forces more computation on dispersed neighbors."""
        k = 10
        q = np.random.randn(sample_data.shape[1]).astype(np.float32)
        
        # 1. Without CV check
        searcher_lax = DESKNNSearcher(sample_data, tolerance=0.5, max_cv=None, random_state=42)
        _, _, count_lax = searcher_lax.query(q, k)

        # 2. With strict CV check
        # This forces the algorithm to continue if variance is high, even if gap is satisfied
        searcher_strict = DESKNNSearcher(sample_data, tolerance=0.5, max_cv=0.01, random_state=42)
        _, _, count_strict = searcher_strict.query(q, k)

        assert count_strict >= count_lax

if __name__ == '__main__':
    pytest.main([__file__, '-v'])