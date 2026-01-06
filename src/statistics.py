"""
Statistical Estimation Module for DES-kNN.

Implements lightweight, O(1) statistical estimators for the DES-kNN algorithm
based on the "Gap" model (Beta-Geometric process). This replaces the computationally
expensive Weibull fitting from the previous version.
"""

import numpy as np
from typing import Tuple

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

    # Optimization: If confidence is high and gap is small, p_max is high.
    # If gap is very large, p_max approaches 0.
    
    delta = 1.0 - confidence_level
    
    # Handle edge case where confidence is 1.0
    if delta <= 1e-12:
        return float(remaining) # Cannot be 100% sure unless gap is infinite

    exponent = 1.0 / (gap + 1.0)
    p_max = 1.0 - np.power(delta, exponent)

    return remaining * p_max


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
        return 0  # Expected misses already low enough naturally (e.g. remaining is small)

    delta = 1.0 - confidence_level

    # From estimate_future_matches logic:
    # target_p = 1 - delta^(1/(gap+1))
    # 1 - target_p = delta^(1/(gap+1))
    # ln(1 - target_p) = ln(delta) / (gap + 1)
    # gap + 1 = ln(delta) / ln(1 - target_p)
    # gap = (ln(delta) / ln(1 - target_p)) - 1

    # Safety for very small target_p (avoid division by zero or precision issues)
    if target_p < 1e-9:
        # Approximation: ln(1-x) ~ -x for small x
        # gap ~ ln(delta) / (-target_p) - 1
        denom = -target_p
    else:
        denom = np.log(1.0 - target_p)

    numerator = np.log(delta)
    
    # required_gap = numerator / denominator - 1
    # We use ceil to ensure we strictly satisfy the inequality
    required_gap = int(np.ceil(numerator / denominator) - 1)

    return max(0, required_gap)