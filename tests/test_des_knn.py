"""
Tests for DES-kNN implementation.
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
        X = np.random.randn(1000, 50).astype(np.float32)
        return X

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

        # DES-kNN
        des_searcher = DESKNNSearcher(X, alpha=0.01, random_state=42)
        des_neighbors, des_distances, _ = des_searcher.query(q, k)

        # Exact
        exact_searcher = ExactBruteForceKNN(X)
        exact_searcher.fit()
        exact_neighbors, exact_distances, _ = exact_searcher.query(q, k)

        # On small dataset, should be exact
        assert set(des_neighbors) == set(exact_neighbors)

    def test_des_knn_recall(self, sample_data, query_point):
        """Test recall meets threshold on synthetic data."""
        k = 10
        n_queries = 20

        # Get ground truth
        exact_searcher = ExactBruteForceKNN(sample_data)
        exact_searcher.fit()

        # DES-kNN
        des_searcher = DESKNNSearcher(sample_data, alpha=0.01, random_state=42)

        recalls = []
        for seed in range(n_queries):
            np.random.seed(seed)
            q = np.random.randn(sample_data.shape[1]).astype(np.float32)

            exact_neighbors, _, _ = exact_searcher.query(q, k)
            des_neighbors, _, _ = des_searcher.query(q, k)

            recall = len(set(des_neighbors) & set(exact_neighbors)) / k
            recalls.append(recall)

        mean_recall = np.mean(recalls)
        assert mean_recall >= 0.9, f"Mean recall {mean_recall} is below threshold"

    def test_des_knn_speedup(self, sample_data, query_point):
        """Test that early stopping provides speedup."""
        k = 10
        searcher = DESKNNSearcher(sample_data, alpha=0.01, random_state=42)

        counts = []
        for seed in range(5):
            np.random.seed(seed)
            q = np.random.randn(sample_data.shape[1]).astype(np.float32)
            _, _, dist_count = searcher.query(q, k)
            counts.append(dist_count)

        # At least one query should stop early
        assert min(counts) < len(sample_data)

    def test_des_knn_distance_metrics(self, sample_data, query_point):
        """Test all distance metrics work correctly."""
        k = 10

        for metric in ['euclidean', 'cosine', 'manhattan']:
            searcher = DESKNNSearcher(
                sample_data,
                distance_metric=metric,
                random_state=42
            )
            neighbors, distances, dist_count = searcher.query(query_point, k)

            assert len(neighbors) == k
            assert len(distances) == k
            assert all(d >= 0 for d in distances)

    def test_des_knn_reproducibility(self, sample_data, query_point):
        """Test random_state provides reproducible results."""
        k = 10

        searcher1 = DESKNNSearcher(sample_data, random_state=42)
        neighbors1, distances1, count1 = searcher1.query(query_point, k)

        searcher2 = DESKNNSearcher(sample_data, random_state=42)
        neighbors2, distances2, count2 = searcher2.query(query_point, k)

        np.testing.assert_array_equal(neighbors1, neighbors2)
        np.testing.assert_array_almost_equal(distances1, distances2)
        assert count1 == count2

    def test_des_knn_batch_query(self, sample_data):
        """Test batch query returns correct shapes."""
        k = 10
        n_queries = 5

        np.random.seed(42)
        queries = np.random.randn(n_queries, sample_data.shape[1]).astype(np.float32)

        searcher = DESKNNSearcher(sample_data, random_state=42)
        all_neighbors, all_distances, all_counts = searcher.query_batch(queries, k)

        assert all_neighbors.shape == (n_queries, k)
        assert all_distances.shape == (n_queries, k)
        assert len(all_counts) == n_queries

    def test_des_knn_return_stats(self, sample_data, query_point):
        """Test return_stats option."""
        k = 10
        searcher = DESKNNSearcher(sample_data, random_state=42)
        neighbors, distances, dist_count, stats = searcher.query(
            query_point, k, return_stats=True
        )

        assert 'stopped_early' in stats
        assert 'stopping_point' in stats
        assert 'last_update' in stats
        assert 'num_updates' in stats
        assert 0 <= stats['stopping_point'] <= 1
        assert len(neighbors) == k
        assert len(distances) == k

    def test_des_knn_return_distances_toggle(self, sample_data, query_point):
        """Test that return_distances flag controls outputs."""
        k = 5
        searcher = DESKNNSearcher(sample_data, random_state=0)

        neighbors_only, dist_count_only = searcher.query(
            query_point, k, return_distances=False
        )
        assert len(neighbors_only) == k
        assert isinstance(dist_count_only, int)

        neighbors_stats, dist_count_stats, stats = searcher.query(
            query_point, k, return_distances=False, return_stats=True
        )
        assert len(neighbors_stats) == k
        assert isinstance(dist_count_stats, int)
        assert isinstance(stats, dict)

    def test_des_knn_adaptive_alpha(self, sample_data, query_point):
        """Test adaptive alpha adjustment."""
        k = 10

        # With adaptive alpha
        searcher_adaptive = DESKNNSearcher(
            sample_data, adaptive_alpha=True, random_state=42
        )
        _, _, count_adaptive = searcher_adaptive.query(query_point, k)

        # Without adaptive alpha
        searcher_fixed = DESKNNSearcher(
            sample_data, adaptive_alpha=False, random_state=42
        )
        _, _, count_fixed = searcher_fixed.query(query_point, k)

        # Both should work, potentially with different stopping points
        assert count_adaptive > 0
        assert count_fixed > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
