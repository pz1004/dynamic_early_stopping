"""
Tests for the new Beta-Geometric statistics module.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.statistics import estimate_future_matches, compute_required_gap

class TestGapStatistics:
    """Test Beta-Geometric estimation functions."""

    def test_estimate_future_matches_basic(self):
        """Test basic behavior of expected matches estimation."""
        # If we have 1000 points remaining and a gap of 0, 
        # we can't be sure of anything, so expected misses should be high.
        # (Technically bounded by confidence, but close to remaining)
        misses_g0 = estimate_future_matches(gap=0, remaining=1000, confidence_level=0.99)
        assert misses_g0 > 900 

    def test_estimate_future_matches_monotonicity(self):
        """As gap increases, expected misses should decrease."""
        remaining = 1000
        misses_g10 = estimate_future_matches(gap=10, remaining=remaining)
        misses_g50 = estimate_future_matches(gap=50, remaining=remaining)
        misses_g100 = estimate_future_matches(gap=100, remaining=remaining)
        
        assert misses_g10 > misses_g50
        assert misses_g50 > misses_g100
        assert misses_g100 < remaining * 0.1  # Should be fairly low by now

    def test_confidence_impact(self):
        """Higher confidence requirements should result in higher expected misses (more conservative)."""
        gap = 20
        rem = 1000
        
        misses_c90 = estimate_future_matches(gap, rem, confidence_level=0.90)
        misses_c99 = estimate_future_matches(gap, rem, confidence_level=0.99)
        
        assert misses_c99 > misses_c90

    def test_compute_required_gap(self):
        """Test inverse calculation of gap."""
        rem = 1000
        tol = 0.5
        conf = 0.99
        
        req_gap = compute_required_gap(remaining=rem, tolerance=tol, confidence_level=conf)
        
        # Verify that this gap actually satisfies the tolerance
        est_misses = estimate_future_matches(req_gap, rem, conf)
        est_misses_prev = estimate_future_matches(req_gap - 1, rem, conf)
        
        assert est_misses <= tol
        assert est_misses_prev > tol  # The gap should be tight (minimal required)

    def test_edge_cases(self):
        """Test boundary conditions."""
        # No remaining items
        assert estimate_future_matches(10, 0) == 0.0
        assert compute_required_gap(0, 0.5) == 0
        
        # Zero tolerance (requires checking everything)
        # In practice, compute_required_gap returns 'remaining' for strictly 0 tolerance,
        # or a very high number. Our implementation clamps or calculates it.
        # If tolerance is 0, we can never be statistically sure unless we check all.
        req_gap = compute_required_gap(100, 0.0, 0.99)
        assert req_gap >= 100 # Must check everything

if __name__ == '__main__':
    pytest.main([__file__, '-v'])