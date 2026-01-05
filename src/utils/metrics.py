"""
Evaluation metrics for k-NN experiments.
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from scipy import stats


def recall_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Recall@k for a single query.

    Parameters
    ----------
    retrieved : np.ndarray of shape (k,)
        Indices of retrieved neighbors.
    ground_truth : np.ndarray of shape (k,)
        Indices of true k nearest neighbors.

    Returns
    -------
    recall : float
        Proportion of true neighbors that were retrieved.
    """
    retrieved_set = set(retrieved)
    truth_set = set(ground_truth)

    if len(truth_set) == 0:
        return 1.0

    return len(retrieved_set & truth_set) / len(truth_set)


def precision_at_k(
    retrieved: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Precision@k for a single query.
    """
    retrieved_set = set(retrieved)
    truth_set = set(ground_truth)

    if len(retrieved_set) == 0:
        return 0.0

    return len(retrieved_set & truth_set) / len(retrieved_set)


def average_precision(
    retrieved: np.ndarray,
    ground_truth: np.ndarray
) -> float:
    """
    Compute Average Precision for a single query.

    AP = (1/R) * Σ P(k) * rel(k)
    where R is total relevant items, P(k) is precision at rank k,
    and rel(k) is 1 if item at rank k is relevant.
    """
    truth_set = set(ground_truth)

    if len(truth_set) == 0:
        return 1.0

    relevant_count = 0
    precision_sum = 0.0

    for i, idx in enumerate(retrieved):
        if idx in truth_set:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)

    return precision_sum / len(truth_set)


def compute_speedup(
    exact_time: float,
    method_time: float
) -> float:
    """Compute speedup factor."""
    if method_time == 0:
        return float('inf')
    return exact_time / method_time


def compute_distance_ratio(
    dist_count: int,
    total_points: int
) -> float:
    """Compute fraction of distance computations."""
    return dist_count / total_points


def aggregate_metrics(
    all_recalls: List[float],
    all_speedups: List[float],
    all_dist_counts: List[int],
    total_points: int,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple queries.

    Parameters
    ----------
    all_recalls : List[float]
        Recall values for each query.
    all_speedups : List[float]
        Speedup values for each query.
    all_dist_counts : List[int]
        Distance computation counts for each query.
    total_points : int
        Total points in database (n).
    confidence : float
        Confidence level for intervals.

    Returns
    -------
    metrics : Dict[str, Any]
        Aggregated metrics with statistics.
    """
    all_recalls = np.array(all_recalls)
    all_speedups = np.array(all_speedups)
    all_dist_counts = np.array(all_dist_counts)

    # Basic statistics
    metrics = {
        'recall': {
            'mean': np.mean(all_recalls),
            'std': np.std(all_recalls),
            'min': np.min(all_recalls),
            'max': np.max(all_recalls),
            'median': np.median(all_recalls),
        },
        'speedup': {
            'mean': np.mean(all_speedups),
            'std': np.std(all_speedups),
            'min': np.min(all_speedups),
            'max': np.max(all_speedups),
            'median': np.median(all_speedups),
        },
        'dist_ratio': {
            'mean': np.mean(all_dist_counts) / total_points,
            'std': np.std(all_dist_counts) / total_points,
        }
    }

    # Percentiles
    for p in [10, 25, 75, 90]:
        metrics['recall'][f'p{p}'] = np.percentile(all_recalls, p)
        metrics['speedup'][f'p{p}'] = np.percentile(all_speedups, p)

    # Confidence intervals (using t-distribution for small samples)
    n = len(all_recalls)
    if n > 1:
        t_value = stats.t.ppf((1 + confidence) / 2, n - 1)

        recall_se = np.std(all_recalls, ddof=1) / np.sqrt(n)
        metrics['recall']['ci_lower'] = np.mean(all_recalls) - t_value * recall_se
        metrics['recall']['ci_upper'] = np.mean(all_recalls) + t_value * recall_se

        speedup_se = np.std(all_speedups, ddof=1) / np.sqrt(n)
        metrics['speedup']['ci_lower'] = np.mean(all_speedups) - t_value * speedup_se
        metrics['speedup']['ci_upper'] = np.mean(all_speedups) + t_value * speedup_se

    # Count failures (recall < threshold)
    metrics['failure_rate_95'] = np.mean(all_recalls < 0.95)
    metrics['failure_rate_99'] = np.mean(all_recalls < 0.99)

    return metrics


def statistical_test(
    method1_recalls: np.ndarray,
    method2_recalls: np.ndarray,
    test_type: str = 'paired_t'
) -> Dict[str, float]:
    """
    Perform statistical test comparing two methods.

    Parameters
    ----------
    method1_recalls : np.ndarray
        Recalls from method 1.
    method2_recalls : np.ndarray
        Recalls from method 2 (same queries).
    test_type : str
        'paired_t' for paired t-test, 'wilcoxon' for Wilcoxon signed-rank.

    Returns
    -------
    result : Dict[str, float]
        Test statistic and p-value.
    """
    if test_type == 'paired_t':
        stat, pvalue = stats.ttest_rel(method1_recalls, method2_recalls)
    elif test_type == 'wilcoxon':
        stat, pvalue = stats.wilcoxon(method1_recalls, method2_recalls)
    else:
        raise ValueError(f"Unknown test type: {test_type}")

    return {
        'statistic': stat,
        'pvalue': pvalue,
        'significant_05': pvalue < 0.05,
        'significant_01': pvalue < 0.01
    }


def query_difficulty_analysis(
    queries: np.ndarray,
    database: np.ndarray,
    ground_truth_dists: np.ndarray,
    method_recalls: np.ndarray,
    method_stopping_points: np.ndarray,
    k: int
) -> Dict[str, Any]:
    """
    Analyze how method performance varies with query difficulty.

    Parameters
    ----------
    queries : np.ndarray of shape (n_queries, d)
        Query points.
    database : np.ndarray of shape (n, d)
        Database points.
    ground_truth_dists : np.ndarray of shape (n_queries, k)
        Distances to true k-NN.
    method_recalls : np.ndarray of shape (n_queries,)
        Recall values per query.
    method_stopping_points : np.ndarray of shape (n_queries,)
        Fraction of database searched per query.
    k : int
        Number of neighbors.

    Returns
    -------
    analysis : Dict[str, Any]
        Analysis results.
    """
    n_queries = len(queries)
    n = len(database)

    # Compute query difficulty metrics
    # 1. Average distance to k-th neighbor (lower = easier)
    kth_distances = ground_truth_dists[:, -1]

    # 2. Ratio of k-th distance to mean distance in database
    # This normalizes for different distance scales
    sample_indices = np.random.choice(n, min(1000, n), replace=False)
    mean_dists = []
    for q in queries:
        dists = np.sqrt(np.sum((database[sample_indices] - q) ** 2, axis=1))
        mean_dists.append(np.mean(dists))
    mean_dists = np.array(mean_dists)

    difficulty_ratio = kth_distances / (mean_dists + 1e-10)

    # Partition queries into difficulty bins
    p33 = np.percentile(difficulty_ratio, 33)
    p67 = np.percentile(difficulty_ratio, 67)

    easy_mask = difficulty_ratio <= p33
    medium_mask = (difficulty_ratio > p33) & (difficulty_ratio <= p67)
    hard_mask = difficulty_ratio > p67

    analysis = {
        'easy_queries': {
            'count': np.sum(easy_mask),
            'avg_recall': np.mean(method_recalls[easy_mask]) if np.any(easy_mask) else 0.0,
            'avg_stopping_point': np.mean(method_stopping_points[easy_mask]) if np.any(easy_mask) else 0.0,
        },
        'medium_queries': {
            'count': np.sum(medium_mask),
            'avg_recall': np.mean(method_recalls[medium_mask]) if np.any(medium_mask) else 0.0,
            'avg_stopping_point': np.mean(method_stopping_points[medium_mask]) if np.any(medium_mask) else 0.0,
        },
        'hard_queries': {
            'count': np.sum(hard_mask),
            'avg_recall': np.mean(method_recalls[hard_mask]) if np.any(hard_mask) else 0.0,
            'avg_stopping_point': np.mean(method_stopping_points[hard_mask]) if np.any(hard_mask) else 0.0,
        },
        'correlation_difficulty_stopping': np.corrcoef(
            difficulty_ratio, method_stopping_points
        )[0, 1],
        'correlation_difficulty_recall': np.corrcoef(
            difficulty_ratio, method_recalls
        )[0, 1]
    }

    return analysis
