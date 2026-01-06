"""
Correctness tests for DES-kNN edge cases.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.des_knn import DESKNNSearcher
from src.baselines.exact_brute_force import ExactBruteForceKNN


def test_deterministic_small_dataset_exact_match():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 2.0],
            [3.0, 1.0],
            [4.0, 3.0],
        ],
        dtype=np.float32,
    )
    q = np.array([0.2, 0.1], dtype=np.float32)
    k = 2

    searcher = DESKNNSearcher(X, random_state=0)
    neighbors, distances, dist_count = searcher.query(q, k)

    exact = ExactBruteForceKNN(X)
    exact.fit()
    exact_neighbors, exact_distances, _ = exact.query(q, k)

    np.testing.assert_array_equal(neighbors, exact_neighbors)
    np.testing.assert_allclose(distances, exact_distances, rtol=1e-6, atol=1e-6)
    assert dist_count == len(X)


def test_k_equals_one_matches_exact():
    X = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    q = np.array([1.1, 0.1, 0.1], dtype=np.float32)
    k = 1

    searcher = DESKNNSearcher(X, random_state=1)
    neighbors, distances, dist_count = searcher.query(q, k)

    exact = ExactBruteForceKNN(X)
    exact.fit()
    exact_neighbors, exact_distances, _ = exact.query(q, k)

    np.testing.assert_array_equal(neighbors, exact_neighbors)
    np.testing.assert_allclose(distances, exact_distances, rtol=1e-6, atol=1e-6)
    assert dist_count == len(X)


def test_k_greater_than_n_returns_all_points():
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=np.float32,
    )
    q = np.array([0.9, 0.0], dtype=np.float32)
    k = 10

    searcher = DESKNNSearcher(X, random_state=2)
    neighbors, distances, dist_count = searcher.query(q, k)

    exact = ExactBruteForceKNN(X)
    exact.fit()
    exact_neighbors, exact_distances, _ = exact.query(q, k)

    assert len(neighbors) == len(X)
    np.testing.assert_array_equal(neighbors, exact_neighbors)
    np.testing.assert_allclose(distances, exact_distances, rtol=1e-6, atol=1e-6)
    assert dist_count == len(X)


def test_duplicates_are_included():
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
        ],
        dtype=np.float32,
    )
    q = np.array([0.0, 0.0], dtype=np.float32)
    k = 2

    searcher = DESKNNSearcher(X, random_state=3)
    neighbors, distances, _ = searcher.query(q, k)

    assert set(neighbors) == {0, 1}
    assert np.allclose(distances, 0.0)


def test_zero_vector_query():
    X = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    q = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    k = 1

    searcher = DESKNNSearcher(X, random_state=4)
    neighbors, distances, _ = searcher.query(q, k)

    assert neighbors[0] == 0
    assert distances[0] == pytest.approx(0.0)


def test_identical_vectors_deterministic_selection():
    """Test that with identical vectors, we get the first k from permutation order."""
    n = 10
    d = 4
    k = 3
    seed = 123
    X = np.ones((n, d), dtype=np.float32)
    q = np.ones(d, dtype=np.float32)

    searcher = DESKNNSearcher(X, random_state=seed)
    neighbors, distances, _ = searcher.query(q, k)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    expected_set = set(perm[:k])

    # When all distances are equal, the neighbors are the first k in permutation order.
    # We compare sets since ordering with equal distances is implementation-dependent.
    assert set(neighbors) == expected_set
    assert np.allclose(distances, 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
