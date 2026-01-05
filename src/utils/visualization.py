"""
Visualization utilities for experiment results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from pathlib import Path


def plot_recall_vs_speedup(
    results: Dict[str, Dict[str, Any]],
    title: str = "Recall vs Speedup Trade-off",
    save_path: Optional[str] = None
):
    """
    Plot recall vs speedup for multiple methods.

    Parameters
    ----------
    results : Dict[str, Dict[str, Any]]
        Results dictionary: method_name -> metrics dict.
    title : str
        Plot title.
    save_path : str or None
        Path to save figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for method_name, metrics in results.items():
        recall = metrics['recall']['mean']
        speedup = metrics['speedup']['mean']

        # Error bars if available
        recall_err = metrics['recall'].get('std', 0)
        speedup_err = metrics['speedup'].get('std', 0)

        ax.errorbar(
            speedup, recall,
            xerr=speedup_err, yerr=recall_err,
            fmt='o', markersize=8, capsize=5,
            label=method_name
        )

    ax.set_xlabel('Speedup (x)', fontsize=12)
    ax.set_ylabel('Recall@k', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add horizontal line at 99% recall
    ax.axhline(y=0.99, color='r', linestyle='--', alpha=0.5, label='99% recall')

    ax.set_ylim([0.8, 1.02])
    ax.set_xlim([0, None])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_stopping_point_distribution(
    stopping_points: np.ndarray,
    method_name: str = "DES-kNN",
    save_path: Optional[str] = None
):
    """
    Plot distribution of stopping points (fraction of data searched).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(stopping_points * 100, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(stopping_points) * 100, color='r', linestyle='--',
               label=f'Mean: {np.mean(stopping_points)*100:.1f}%')
    ax.axvline(np.median(stopping_points) * 100, color='g', linestyle=':',
               label=f'Median: {np.median(stopping_points)*100:.1f}%')

    ax.set_xlabel('Stopping Point (% of data searched)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{method_name} - Stopping Point Distribution', fontsize=14)
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


def plot_query_difficulty_analysis(
    difficulty_scores: np.ndarray,
    recalls: np.ndarray,
    stopping_points: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot performance vs query difficulty.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Recall vs difficulty
    ax1 = axes[0]
    ax1.scatter(difficulty_scores, recalls, alpha=0.5, s=20)
    z = np.polyfit(difficulty_scores, recalls, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(difficulty_scores), max(difficulty_scores), 100)
    ax1.plot(x_line, p(x_line), 'r--', label='Trend')
    ax1.set_xlabel('Query Difficulty Score', fontsize=12)
    ax1.set_ylabel('Recall@k', fontsize=12)
    ax1.set_title('Recall vs Query Difficulty', fontsize=14)
    ax1.legend()

    # Stopping point vs difficulty
    ax2 = axes[1]
    ax2.scatter(difficulty_scores, stopping_points * 100, alpha=0.5, s=20)
    z = np.polyfit(difficulty_scores, stopping_points, 1)
    p = np.poly1d(z)
    ax2.plot(x_line, p(x_line) * 100, 'r--', label='Trend')
    ax2.set_xlabel('Query Difficulty Score', fontsize=12)
    ax2.set_ylabel('Stopping Point (%)', fontsize=12)
    ax2.set_title('Adaptive Behavior: Harder Queries -> Later Stopping', fontsize=14)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def create_results_table(
    results: Dict[str, Dict[str, Any]],
    dataset_name: str,
    k: int
) -> str:
    """
    Create formatted results table as string.
    """
    lines = []
    lines.append(f"\nResults for {dataset_name}, k={k}")
    lines.append("=" * 70)
    lines.append(f"{'Method':<20} {'Recall':>10} {'Speedup':>10} {'Dist Ratio':>12}")
    lines.append("-" * 70)

    for method_name, metrics in sorted(results.items()):
        recall = metrics['recall']['mean']
        speedup = metrics['speedup']['mean']
        dist_ratio = metrics['dist_ratio']['mean']

        lines.append(
            f"{method_name:<20} {recall:>10.4f} {speedup:>10.2f}x {dist_ratio:>12.4f}"
        )

    lines.append("=" * 70)

    return "\n".join(lines)
