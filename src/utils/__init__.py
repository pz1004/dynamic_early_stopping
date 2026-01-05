"""
Utility modules for DES-kNN.
"""

from .heap import MaxHeap, MinHeap
from .data_loader import DataLoader
from .metrics import (
    recall_at_k,
    precision_at_k,
    average_precision,
    compute_speedup,
    compute_distance_ratio,
    aggregate_metrics
)

__all__ = [
    'MaxHeap',
    'MinHeap',
    'DataLoader',
    'recall_at_k',
    'precision_at_k',
    'average_precision',
    'compute_speedup',
    'compute_distance_ratio',
    'aggregate_metrics'
]
