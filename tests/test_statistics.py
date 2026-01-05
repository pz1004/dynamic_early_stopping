"""
Tests for statistics module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.statistics import (
    fit_weibull,
    weibull_cdf,
    estimate_exceedance_probability,
    compute_confidence_bound,
    adaptive_alpha,
    CircularBuffer
)


class TestWeibull:
    """Test Weibull distribution functions."""

    def test_weibull_fit(self):
        """Test Weibull fitting on known distribution."""
        np.random.seed(42)
        # Generate Weibull samples with known parameters
        shape_true, scale_true = 2.0, 5.0
        samples = np.random.weibull(shape_true, 1000) * scale_true

        shape_est, scale_est = fit_weibull(samples)
        assert abs(shape_est - shape_true) < 0.5
        assert abs(scale_est - scale_true) < 1.0

    def test_weibull_cdf(self):
        """Test Weibull CDF computation."""
        # F(x) = 1 - exp(-(x/scale)^shape)
        shape, scale = 2.0, 1.0

        # At x=0, CDF should be 0
        assert weibull_cdf(0, shape, scale) == 0.0

        # CDF should be monotonically increasing
        prev = 0.0
        for x in np.linspace(0.1, 5.0, 20):
            curr = weibull_cdf(x, shape, scale)
            assert curr > prev
            prev = curr

        # CDF should approach 1
        assert weibull_cdf(10.0, shape, scale) > 0.99

    def test_weibull_cdf_invalid_params(self):
        """Test Weibull CDF with invalid parameters."""
        # Should return 0.5 for invalid params
        assert weibull_cdf(1.0, 0, 1.0) == 0.5
        assert weibull_cdf(1.0, 1.0, 0) == 0.5
        assert weibull_cdf(1.0, -1, 1.0) == 0.5


class TestExceedanceProbability:
    """Test exceedance probability estimation."""

    def test_exceedance_probability(self):
        """Test exceedance probability estimation."""
        distances = [1.0, 2.0, 3.0, 4.0, 5.0] * 20  # 100 samples
        d_k = 2.5
        p, expected = estimate_exceedance_probability(distances, d_k, remaining=100)

        # About 40% of samples are below 2.5
        assert 0.3 < p < 0.7
        assert expected > 0

    def test_exceedance_probability_empty(self):
        """Test with empty distances."""
        p, expected = estimate_exceedance_probability([], 1.0, 100)
        assert p == 1.0
        assert expected == 100

    def test_exceedance_probability_zero_dk(self):
        """Test with zero d_k."""
        p, expected = estimate_exceedance_probability([1.0, 2.0], 0, 100)
        assert p == 1.0


class TestConfidenceBound:
    """Test confidence bound computation."""

    def test_confidence_bound(self):
        """Test confidence bound computation."""
        # Long gap since last update should give high confidence
        update_history = [10, 20, 30, 40, 50]
        conf = compute_confidence_bound(update_history, current_idx=200, k=5)
        assert conf > 0.8

    def test_confidence_bound_few_updates(self):
        """Test with few updates."""
        update_history = [5]
        conf = compute_confidence_bound(update_history, current_idx=10, k=5)
        assert conf == 0.0  # Not enough updates

    def test_confidence_bound_recent_update(self):
        """Test with recent update (low confidence)."""
        update_history = [10, 20, 30, 40, 50]
        conf = compute_confidence_bound(update_history, current_idx=51, k=5)
        assert conf < 0.5  # Recent update, low confidence


class TestAdaptiveAlpha:
    """Test adaptive alpha adjustment."""

    def test_adaptive_alpha(self):
        """Test adaptive alpha adjustment."""
        np.random.seed(42)
        distances = list(np.random.randn(100) + 5)  # Mean around 5

        # Easy query: d_k well below mean
        alpha_easy = adaptive_alpha(distances, d_k=2.0, initial_alpha=0.01)

        # Hard query: d_k near mean
        alpha_hard = adaptive_alpha(distances, d_k=5.0, initial_alpha=0.01)

        assert alpha_easy < 0.01  # More aggressive
        assert alpha_hard >= 0.01  # More conservative

    def test_adaptive_alpha_few_samples(self):
        """Test with few samples (returns initial)."""
        alpha = adaptive_alpha([1.0, 2.0], 1.5, 0.01)
        assert alpha == 0.01

    def test_adaptive_alpha_bounds(self):
        """Test alpha stays within bounds."""
        distances = list(np.random.randn(100) + 5)

        for d_k in [0.0, 2.0, 5.0, 10.0]:
            alpha = adaptive_alpha(distances, d_k, 0.01)
            assert 0.001 <= alpha <= 0.1


class TestCircularBuffer:
    """Test circular buffer operations."""

    def test_circular_buffer(self):
        """Test circular buffer operations."""
        buf = CircularBuffer(5)
        for i in range(10):
            buf.add(float(i))

        assert len(buf) == 5
        assert list(buf.get_all()) == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_circular_buffer_mean(self):
        """Test buffer mean computation."""
        buf = CircularBuffer(5)
        for i in range(5):
            buf.add(float(i))

        assert buf.mean() == 2.0  # mean of 0,1,2,3,4

    def test_circular_buffer_std(self):
        """Test buffer std computation."""
        buf = CircularBuffer(5)
        for i in range(5):
            buf.add(float(i))

        expected_std = np.std([0, 1, 2, 3, 4])
        assert abs(buf.std() - expected_std) < 1e-10

    def test_circular_buffer_empty(self):
        """Test empty buffer."""
        buf = CircularBuffer(5)
        assert len(buf) == 0
        assert buf.mean() == 0.0
        assert buf.std() == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
