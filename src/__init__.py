"""
DES-kNN: Dynamic Early Stopping for k-Nearest Neighbors Search
"""

from .des_knn import DESKNNSearcher
from .statistics import (
    estimate_exceedance_probability,
    compute_confidence_bound,
    adaptive_alpha,
    fit_weibull,
    weibull_cdf
)

__version__ = '0.1.0'

__all__ = [
    'DESKNNSearcher',
    'estimate_exceedance_probability',
    'compute_confidence_bound',
    'adaptive_alpha',
    'fit_weibull',
    'weibull_cdf'
]
