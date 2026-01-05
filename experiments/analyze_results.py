#!/usr/bin/env python3
"""
Analyze experiment results and generate tables/figures.

Usage:
    python analyze_results.py --results_dir results/
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.visualization import (
    plot_recall_vs_speedup,
    plot_stopping_point_distribution,
    plot_query_difficulty_analysis,
    create_results_table
)


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all result files from directory."""
    results_path = Path(results_dir)
    all_results = {}

    for json_file in results_path.glob('*.json'):
        with open(json_file) as f:
            data = json.load(f)
            all_results[json_file.stem] = data

    return all_results


def aggregate_across_seeds(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across multiple seeds."""
    if not results or all('error' in r for r in results):
        return None

    valid_results = [r for r in results if 'error' not in r]

    return {
        'recall_mean': np.mean([r['recall_mean'] for r in valid_results]),
        'recall_std': np.std([r['recall_mean'] for r in valid_results]),
        'speedup_mean': np.mean([r['speedup_mean'] for r in valid_results]),
        'speedup_std': np.std([r['speedup_mean'] for r in valid_results]),
        'dist_ratio_mean': np.mean([r['dist_ratio'] for r in valid_results]),
        'build_time_mean': np.mean([r['build_time'] for r in valid_results]),
        'failure_rate_mean': np.mean([r['failure_rate_99'] for r in valid_results]),
        'n_seeds': len(valid_results)
    }


def create_summary_table(results: Dict[str, Any], k: int = 10) -> pd.DataFrame:
    """
    Create summary DataFrame for a specific k value.
    """
    rows = []

    for dataset in results:
        for method in results[dataset]:
            if str(k) not in results[dataset][method]:
                continue

            for config, seed_results in results[dataset][method][str(k)].items():
                agg = aggregate_across_seeds(seed_results)
                if agg is None:
                    continue

                rows.append({
                    'Dataset': dataset,
                    'Method': method,
                    'Config': config if config != 'default' else '-',
                    'Recall': f"{agg['recall_mean']:.4f}+/-{agg['recall_std']:.4f}",
                    'Speedup': f"{agg['speedup_mean']:.2f}x",
                    'Dist Ratio': f"{agg['dist_ratio_mean']:.4f}",
                    'Build Time': f"{agg['build_time_mean']:.2f}s",
                    'Failure Rate': f"{agg['failure_rate_mean']*100:.1f}%"
                })

    return pd.DataFrame(rows)


def generate_latex_table(df: pd.DataFrame, caption: str = "") -> str:
    """Generate LaTeX table from DataFrame."""
    latex = df.to_latex(index=False, escape=False)

    if caption:
        latex = latex.replace(
            r'\begin{tabular}',
            f'\\caption{{{caption}}}\n\\begin{{tabular}}'
        )

    return latex


def compare_des_knn_configs(results: Dict[str, Any], dataset: str, k: int = 10) -> pd.DataFrame:
    """Compare different DES-kNN configurations."""
    if 'des_knn' not in results.get(dataset, {}):
        return None

    rows = []
    for config, seed_results in results[dataset]['des_knn'].get(str(k), {}).items():
        agg = aggregate_across_seeds(seed_results)
        if agg:
            rows.append({
                'Configuration': config,
                'Recall': agg['recall_mean'],
                'Speedup': agg['speedup_mean'],
                'Failure Rate': agg['failure_rate_mean']
            })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing result files')
    parser.add_argument('--output_dir', type=str, default='figures',
                       help='Directory for output figures')
    parser.add_argument('--k', type=int, default=10,
                       help='Value of k for analysis')

    args = parser.parse_args()

    # Load results
    print(f"Loading results from {args.results_dir}...")
    all_results = load_results(args.results_dir)

    if not all_results:
        print("No results found!")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each result file
    for result_name, results in all_results.items():
        print(f"\nAnalyzing: {result_name}")

        # Create summary table
        summary_df = create_summary_table(results, k=args.k)
        if not summary_df.empty:
            print("\nSummary Table:")
            print(summary_df.to_string())

            # Save to CSV
            summary_df.to_csv(output_dir / f'{result_name}_summary_k{args.k}.csv', index=False)

            # Generate LaTeX
            latex = generate_latex_table(summary_df, f"Results for k={args.k}")
            with open(output_dir / f'{result_name}_summary_k{args.k}.tex', 'w') as f:
                f.write(latex)

        # DES-kNN configuration comparison
        for dataset in results:
            config_df = compare_des_knn_configs(results, dataset, k=args.k)
            if config_df is not None and not config_df.empty:
                print(f"\nDES-kNN Configurations for {dataset}:")
                print(config_df.to_string())

    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
