"""
Statistical Estimation Module for DES-kNN.

Implements online statistical estimators for the DES-kNN algorithm,
including probability estimation for remaining points entering the
k-NN set and confidence bounds based on update patterns.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy import stats
from scipy.special import gamma


def estimate_exceedance_probability(
    distances: List[float],
    d_k: float,
    remaining: int,
    alpha: float = 0.01,
    use_weibull: bool = True
) -> Tuple[float, float]:
    """
    Estimate probability that remaining points could enter the k-NN set.

    Uses both empirical and parametric (Weibull) estimates, taking the
    more conservative (higher) value to ensure safety.

    Parameters
    ----------
    distances : List[float]
        Observed distances in the sliding window.
    d_k : float
        Current k-th nearest distance (threshold).
    remaining : int
        Number of remaining unsearched points.
    alpha : float, default=0.01
        Significance level for confidence bounds.
    use_weibull : bool, default=True
        Whether to use Weibull distribution fitting.

    Returns
    -------
    p_below_upper : float
        Upper bound on probability that a random point has distance < d_k.
    expected_entrants : float
        Expected number of remaining points that could enter k-NN set.

    Notes
    -----
    The Weibull distribution is commonly used for distance distributions
    in high-dimensional spaces because:
    1. Distances are non-negative
    2. The distribution often has a long right tail
    3. It can model various shapes depending on parameters
    """
    distances = np.array(distances, dtype=np.float64)
    m = len(distances)

    if m == 0 or d_k <= 0:
        return 1.0, remaining

    # Method 1: Empirical estimate with continuity correction
    # This is robust but may underestimate for small samples
    count_below = np.sum(distances < d_k)
    p_below_empirical = (count_below + 0.5) / (m + 1)

    # Method 2: Parametric estimate using Weibull distribution
    # Weibull CDF: F(x) = 1 - exp(-(x/scale)^shape)
    if use_weibull and m >= 10:
        try:
            shape, scale = fit_weibull(distances)
            if shape > 0 and scale > 0:
                p_below_parametric = weibull_cdf(d_k, shape, scale)
            else:
                p_below_parametric = p_below_empirical
        except Exception:
            p_below_parametric = p_below_empirical
    else:
        p_below_parametric = p_below_empirical

    # Use conservative (higher) estimate
    p_below = max(p_below_empirical, p_below_parametric)

    # Add Hoeffding bound for uncertainty
    # P(|p_hat - p| > epsilon) <= 2 * exp(-2 * m * epsilon^2)
    # Solving for epsilon at confidence level alpha:
    # epsilon = sqrt(log(2/alpha) / (2*m))
    epsilon = np.sqrt(np.log(2.0 / alpha) / (2.0 * m))
    p_below_upper = min(1.0, p_below + epsilon)

    # Expected number of points that could enter k-NN set
    expected_entrants = remaining * p_below_upper

    return p_below_upper, expected_entrants


def fit_weibull(distances: np.ndarray) -> Tuple[float, float]:
    """
    Fit Weibull distribution to observed distances using MLE.

    The Weibull distribution has PDF:
        f(x; k, lambda) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)

    where k = shape, lambda = scale.

    Parameters
    ----------
    distances : np.ndarray
        Observed positive distances.

    Returns
    -------
    shape : float
        Weibull shape parameter (k).
    scale : float
        Weibull scale parameter (lambda).

    Notes
    -----
    Uses scipy.stats.weibull_min.fit() for robust estimation.
    Falls back to method of moments if MLE fails.
    """
    distances = np.asarray(distances, dtype=np.float64)
    distances = distances[distances > 0]  # Remove zeros

    if len(distances) < 5:
        # Not enough data, return defaults
        return 1.0, np.mean(distances) if len(distances) > 0 else 1.0

    try:
        # scipy.stats.weibull_min uses different parameterization
        # weibull_min.fit returns (c, loc, scale) where c is shape
        # loc should be fixed at 0 for distance distributions
        shape, loc, scale = stats.weibull_min.fit(distances, floc=0)

        if shape <= 0 or scale <= 0:
            raise ValueError("Invalid parameters")

        return shape, scale

    except Exception:
        # Fallback: Method of moments approximation
        mean_d = np.mean(distances)
        std_d = np.std(distances)

        if std_d == 0:
            return 1.0, mean_d

        cv = std_d / mean_d  # Coefficient of variation

        # Approximate shape from CV (inverse relationship)
        # For Weibull: CV ~ 1.0/k for moderate k
        shape = 1.0 / max(cv, 0.1)
        shape = np.clip(shape, 0.5, 10.0)

        # Scale from mean: E[X] = scale * Gamma(1 + 1/shape)
        scale = mean_d / gamma(1 + 1/shape)

        return shape, scale


def weibull_cdf(x: float, shape: float, scale: float) -> float:
    """
    Compute Weibull cumulative distribution function.

    F(x) = 1 - exp(-(x/scale)^shape)

    Parameters
    ----------
    x : float
        Value at which to evaluate CDF.
    shape : float
        Shape parameter (k > 0).
    scale : float
        Scale parameter (lambda > 0).

    Returns
    -------
    probability : float
        P(X <= x) for Weibull distributed X.
    """
    if x <= 0:
        return 0.0
    if shape <= 0 or scale <= 0:
        return 0.5  # Return uninformative value

    return 1.0 - np.exp(-np.power(x / scale, shape))


def compute_confidence_bound(
    update_history: List[int],
    current_idx: int,
    k: int
) -> float:
    """
    Compute confidence that k-NN set is complete based on update pattern.

    Models the inter-update gaps as geometric random variables (memoryless
    property). If we've gone a long time without an update, we have
    increasing confidence that the k-NN set is complete.

    Parameters
    ----------
    update_history : List[int]
        Indices where d_k was updated (when a new point entered k-NN).
    current_idx : int
        Current search position.
    k : int
        Number of neighbors.

    Returns
    -------
    confidence : float
        Estimated probability in [0, 1] that k-NN set is complete.

    Notes
    -----
    The confidence is based on the observation that if updates follow
    a geometric distribution with parameter p (probability of update
    at each step), then seeing a long gap suggests p is very small,
    meaning most relevant points have been found.
    """
    if len(update_history) < k:
        # Haven't even found k points yet
        return 0.0

    if len(update_history) < 2:
        return 0.0

    # Compute inter-update gaps
    gaps = []
    for i in range(len(update_history) - 1):
        gaps.append(update_history[i + 1] - update_history[i])

    # Current gap since last update
    last_update_idx = update_history[-1]
    current_gap = current_idx - last_update_idx

    if len(gaps) < 3:
        # Not enough gaps for reliable estimation
        # Use simple heuristic based on current gap
        expected_gap = current_idx / (len(update_history) + 1)
        if current_gap > 3 * expected_gap:
            return 0.8
        elif current_gap > 2 * expected_gap:
            return 0.5
        else:
            return 0.0

    # Model gaps as geometric distribution
    # MLE for geometric parameter p: p_hat = 1 / mean_gap
    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    if mean_gap == 0:
        return 0.0

    p_update = 1.0 / mean_gap  # Estimated probability of update per step

    # Probability of seeing no update for current_gap steps
    # if the true update probability is p_update
    # This is (1 - p)^current_gap
    if current_gap == 0:
        return 0.0

    p_no_update_if_active = np.power(1 - p_update, current_gap)

    # The confidence that we're "done" increases as this probability decreases
    # Using Bayesian interpretation:
    # P(complete | no update for gap) ~ P(no update | complete) * P(complete)
    # We approximate this as 1 - p_no_update_if_active
    confidence = 1.0 - p_no_update_if_active

    # Additional adjustment based on gap relative to observed distribution
    # If current gap is much larger than typical gaps, increase confidence
    if std_gap > 0:
        z_score = (current_gap - mean_gap) / std_gap
        if z_score > 2.0:
            confidence = min(1.0, confidence + 0.1 * (z_score - 2.0))

    return np.clip(confidence, 0.0, 1.0)


def adaptive_alpha(
    distances: List[float],
    d_k: float,
    initial_alpha: float
) -> float:
    """
    Dynamically adjust significance level based on query difficulty.

    Easy queries (well-separated neighbors) can use lower alpha for more
    aggressive early stopping. Hard queries (dense regions) need higher
    alpha for safety.

    Parameters
    ----------
    distances : List[float]
        Observed distances in sliding window.
    d_k : float
        Current k-th nearest distance.
    initial_alpha : float
        Base significance level.

    Returns
    -------
    alpha_adjusted : float
        Adjusted significance level in [0.001, 0.1].

    Notes
    -----
    Query difficulty is measured by how far d_k is from the mean distance
    in terms of standard deviations. Well-separated queries have d_k
    far below the mean.
    """
    distances = np.array(distances, dtype=np.float64)

    if len(distances) < 10:
        return initial_alpha

    mu = np.mean(distances)
    sigma = np.std(distances)

    if sigma == 0 or mu == 0:
        return initial_alpha

    # Distance margin: how far d_k is from mean (in std devs)
    # Positive margin means d_k is below mean (easier query)
    margin = (mu - d_k) / sigma

    # Adjust alpha based on margin
    if margin > 2.0:
        # Easy query: can be more aggressive
        alpha_adjusted = initial_alpha / 2.0
    elif margin > 1.0:
        # Moderately easy
        alpha_adjusted = initial_alpha * 0.75
    elif margin < 0.5:
        # Hard query: be conservative
        alpha_adjusted = initial_alpha * 2.0
    elif margin < 0.0:
        # Very hard query: d_k above mean
        alpha_adjusted = initial_alpha * 3.0
    else:
        alpha_adjusted = initial_alpha

    # Clip to reasonable range
    return np.clip(alpha_adjusted, 0.001, 0.1)


def compute_stopping_probability_bayesian(
    distances: List[float],
    d_k: float,
    remaining: int,
    k: int,
    prior_complete: float = 0.5
) -> float:
    """
    Alternative Bayesian approach to estimate probability k-NN is complete.

    Uses Beta-Binomial conjugate model for the proportion of points
    with distance < d_k.

    Parameters
    ----------
    distances : List[float]
        Observed distances.
    d_k : float
        Current k-th nearest distance.
    remaining : int
        Number of remaining points.
    k : int
        Number of neighbors.
    prior_complete : float, default=0.5
        Prior probability that k-NN set is complete.

    Returns
    -------
    posterior_complete : float
        Posterior probability that k-NN set is complete.
    """
    distances = np.array(distances)
    m = len(distances)

    if m == 0:
        return prior_complete

    # Count successes (distances < d_k)
    successes = np.sum(distances < d_k)
    failures = m - successes

    # Beta posterior for proportion p with uniform prior
    # p ~ Beta(1 + successes, 1 + failures)
    alpha_post = 1 + successes
    beta_post = 1 + failures

    # Expected value of p
    p_expected = alpha_post / (alpha_post + beta_post)

    # Expected number of remaining points in k-NN
    expected_entrants = remaining * p_expected

    # Probability that 0 additional points enter k-NN
    # Using Beta-Binomial distribution
    # Approximate with Poisson for large remaining
    if expected_entrants > 0:
        p_zero_entrants = np.exp(-expected_entrants)
    else:
        p_zero_entrants = 1.0

    # Update prior with this likelihood
    posterior_complete = (p_zero_entrants * prior_complete) / (
        p_zero_entrants * prior_complete + (1 - prior_complete) * (1 - p_zero_entrants)
    )

    return posterior_complete


class CircularBuffer:
    """
    Fixed-size circular buffer for streaming statistics.

    More memory-efficient than deque for fixed-size windows.
    """

    def __init__(self, size: int):
        self.size = size
        self.buffer = np.zeros(size, dtype=np.float64)
        self.count = 0
        self.index = 0

    def add(self, value: float):
        """Add a value to the buffer."""
        self.buffer[self.index] = value
        self.index = (self.index + 1) % self.size
        if self.count < self.size:
            self.count += 1

    def get_all(self) -> np.ndarray:
        """Get all values in insertion order."""
        if self.count < self.size:
            return self.buffer[:self.count].copy()
        else:
            # Reconstruct in correct order
            return np.concatenate([
                self.buffer[self.index:],
                self.buffer[:self.index]
            ])

    def mean(self) -> float:
        """Compute mean of buffer values."""
        if self.count == 0:
            return 0.0
        return np.mean(self.buffer[:self.count] if self.count < self.size else self.buffer)

    def std(self) -> float:
        """Compute standard deviation of buffer values."""
        if self.count < 2:
            return 0.0
        data = self.buffer[:self.count] if self.count < self.size else self.buffer
        return np.std(data)

    def __len__(self) -> int:
        return self.count
