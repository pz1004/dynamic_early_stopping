# Statistical Estimation Module

## File: `src/statistics.py`

## Overview

Implements lightweight, O(1) statistical estimators for the DES-kNN algorithm based on the "Gap" model (Beta-Geometric process). This module replaces the computationally expensive Weibull fitting from the previous version.

## Key Innovation

Instead of modeling the shape of the distance distribution (Weibull), we model the **probability of the next update** based on how long it has been since the last update. This is:
- **O(1)** per computation
- **Algebraically closed** (no iterative fitting)
- **Directly interpretable** as "Expected Missed Neighbors"

## Required Imports

```python
import numpy as np
from typing import Tuple
```

## Function: estimate_future_matches

```python
def estimate_future_matches(
    gap: int,
    remaining: int,
    confidence_level: float = 0.99
) -> float:
    """
    Estimate the expected number of relevant points remaining in the dataset.

    This uses a Bayesian approach (Beta Conjugate) to bound the probability
    that a random unseen point is better than the current k-th neighbor,
    given that we have seen 'gap' consecutive failures (non-updates).

    Parameters
    ----------
    gap : int
        Number of points checked since the last update to the k-NN set.
    remaining : int
        Total number of points left to search.
    confidence_level : float, default=0.99
        The confidence level for the probability upper bound.
        (e.g., we are 99% sure the true rate of finding neighbors is below p_max)

    Returns
    -------
    expected_matches : float
        The expected number of k-NN neighbors we might miss if we stop now.
        Calculated as: remaining * p_upper_bound
    """
    if gap < 0:
        return float(remaining)
    if remaining <= 0:
        return 0.0

    # Model:
    # Let p be the probability that a random point x satisfies dist(x, q) < d_k.
    # We assume the search order is random.
    # Prior on p: Beta(1, 1) (Uniform/Uninformative)
    # Likelihood: 'gap' consecutive failures (Bernoulli trials).
    # Posterior on p: Beta(1, 1 + gap).
    #
    # We calculate p_max such that P(p < p_max) = confidence_level.
    # The CDF of Beta(1, beta) is 1 - (1 - x)^beta.
    #
    # Solving for p_max:
    # confidence = 1 - (1 - p_max)^(1 + gap)
    # (1 - p_max)^(1 + gap) = 1 - confidence
    # 1 - p_max = (1 - confidence)^(1 / (1 + gap))
    # p_max = 1 - (1 - confidence)^(1 / (1 + gap))

    delta = 1.0 - confidence_level

    # Handle edge case where confidence is 1.0
    if delta <= 1e-12:
        return float(remaining)  # Cannot be 100% sure unless gap is infinite

    exponent = 1.0 / (gap + 1.0)
    p_max = 1.0 - np.power(delta, exponent)

    return remaining * p_max
```

### Mathematical Derivation

Given:
- `p` = probability that a random unseen point is a true neighbor
- `gap` = number of consecutive non-updates observed
- `confidence` = desired confidence level (e.g., 0.99)

**Bayesian Model:**
1. Prior: `p ~ Beta(1, 1)` (Uniform on [0, 1])
2. Observation: `gap` consecutive failures
3. Posterior: `p ~ Beta(1, 1 + gap)`

**Upper Bound Calculation:**

The CDF of Beta(1, β) is: `F(x) = 1 - (1 - x)^β`

We want `p_max` such that `P(p < p_max) = confidence`:
```
confidence = 1 - (1 - p_max)^(1 + gap)
p_max = 1 - (1 - confidence)^(1/(1 + gap))
```

**Expected Misses:**
```
E[Missed] = remaining * p_max
```

### Complexity

- **Time**: O(1) - single power operation
- **Space**: O(1) - no allocations

## Function: compute_required_gap

```python
def compute_required_gap(
    remaining: int,
    tolerance: float = 0.5,
    confidence_level: float = 0.99
) -> int:
    """
    Calculate the minimum gap needed to stop safely.

    This acts as an inverse to estimate_future_matches. It calculates how many
    consecutive non-updates we need to see before the expected number of
    missed neighbors drops below the tolerance.

    Used to optimize the search loop (don't check stopping criteria until
    gap > required_gap).

    Parameters
    ----------
    remaining : int
        Number of points left to search.
    tolerance : float, default=0.5
        Maximum acceptable expected misses (e.g., 0.5 means we expect to miss
        less than half a neighbor).
    confidence_level : float, default=0.99
        Confidence level for the bound.

    Returns
    -------
    required_gap : int
        The minimum gap size required to satisfy the condition.
    """
    if remaining <= 0:
        return 0
    if tolerance <= 0:
        return remaining  # If zero tolerance, must search everything

    # We want: remaining * p_max < tolerance
    # p_max < tolerance / remaining
    target_p = tolerance / remaining

    if target_p >= 1.0:
        return 0  # Expected misses already low enough naturally

    delta = 1.0 - confidence_level

    # From estimate_future_matches logic:
    # target_p = 1 - delta^(1/(gap+1))
    # 1 - target_p = delta^(1/(gap+1))
    # ln(1 - target_p) = ln(delta) / (gap + 1)
    # gap + 1 = ln(delta) / ln(1 - target_p)
    # gap = (ln(delta) / ln(1 - target_p)) - 1

    # Safety for very small target_p
    if target_p < 1e-9:
        # Approximation: ln(1-x) ~ -x for small x
        denom = -target_p
    else:
        denom = np.log(1.0 - target_p)

    numerator = np.log(delta)

    # Use ceil to ensure we strictly satisfy the inequality
    required_gap = int(np.ceil(numerator / denom) - 1)

    return max(0, required_gap)
```

### Use Case: Loop Optimization

The `compute_required_gap` function can be used to skip early stopping checks until the gap is large enough:

```python
# Pre-compute minimum gap needed
min_gap = compute_required_gap(remaining=n, tolerance=tolerance, confidence=confidence)

# In the search loop:
if current_gap >= min_gap:
    # Only then check the full stopping criterion
    expected_misses = estimate_future_matches(current_gap, remaining, confidence)
    if expected_misses < tolerance:
        break
```

## Comparison: Old vs New Approach

| Aspect | Old (Weibull) | New (Beta-Geometric) |
|--------|---------------|----------------------|
| **Model** | Distance distribution shape | Gap between updates |
| **Complexity** | O(N * iterations) per fit | O(1) per check |
| **Parameters** | shape, scale (2 params) | gap, remaining (observed) |
| **Fitting** | MLE/Method of moments | Closed-form Bayesian |
| **Interpretability** | Abstract probability | "Expected missed neighbors" |
| **Calibration** | Hard to tune | tolerance directly controls recall |

## Theoretical Foundation

### Geometric Distribution Assumption

Under random search order, the number of samples until the next k-NN update follows a Geometric distribution with parameter `p`, where `p` is the proportion of remaining points that would improve the k-NN set.

**Key insight**: As we observe longer gaps without updates, we have increasing evidence that `p` is small (i.e., most true neighbors have been found).

### Rule of Succession Connection

The Beta(1, 1+G) posterior is equivalent to the "Rule of Succession" from probability theory. If we've seen G failures, the expected value of p is:

```
E[p] = 1 / (G + 2)
```

But we use an upper bound (quantile) rather than the expectation for safety.

### "Rule of Three" Approximation

For quick mental calculation with 95% confidence:
```
p_max ≈ 3 / G
```

This is a commonly used approximation in statistics for estimating rare event probabilities.

## Unit Tests

```python
# tests/test_statistics.py

class TestGapStatistics:
    """Test Beta-Geometric estimation functions."""

    def test_estimate_future_matches_basic(self):
        """Test basic behavior of expected matches estimation."""
        # Gap of 0 with 1000 remaining: should be close to remaining
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
        assert misses_g100 < remaining * 0.1

    def test_confidence_impact(self):
        """Higher confidence = more conservative = higher expected misses."""
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

        # Verify this gap satisfies the tolerance
        est_misses = estimate_future_matches(req_gap, rem, conf)
        est_misses_prev = estimate_future_matches(req_gap - 1, rem, conf)

        assert est_misses <= tol
        assert est_misses_prev > tol  # Gap should be minimal

    def test_edge_cases(self):
        """Test boundary conditions."""
        # No remaining items
        assert estimate_future_matches(10, 0) == 0.0
        assert compute_required_gap(0, 0.5) == 0

        # Zero tolerance requires checking everything
        req_gap = compute_required_gap(100, 0.0, 0.99)
        assert req_gap >= 100
```

## Performance Characteristics

### Gap Required vs Remaining Points

For `tolerance=0.5` and `confidence=0.99`:

| Remaining | Required Gap | Interpretation |
|-----------|--------------|----------------|
| 10,000 | ~900 | Need ~9% of remaining as gap |
| 1,000 | ~450 | Need ~45% of remaining as gap |
| 100 | ~90 | Need ~90% of remaining as gap |
| 10 | ~9 | Nearly need to check all |

**Insight**: The algorithm becomes more effective with larger datasets, where the relative gap required is smaller.

### Practical Implications

1. **Large datasets**: Can achieve significant speedup (2-5x typical)
2. **Small datasets**: May not stop early (but overhead is minimal)
3. **Clustered data**: Better speedup (tight clusters = long gaps)
4. **Uniform data**: Less speedup (frequent updates = short gaps)
