"""
DES-kNN: Dynamic Early Stopping for k-Nearest Neighbors Search

Uses the Beta-Geometric "Gap" model for O(1) stopping criterion computation.
"""

from .des_knn import DESKNNSearcher
from .statistics import (
    estimate_future_matches,
    compute_required_gap
)

__version__ = '0.2.0'

__all__ = [
    'DESKNNSearcher',
    'estimate_future_matches',
    'compute_required_gap'
]
