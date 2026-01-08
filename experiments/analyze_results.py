#!/usr/bin/env python3
"""
Analyze paper experiment results and generate plots/tables.

Usage:
    python analyze_results.py --results_dir results/paper --output_dir figures
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))


def select_results_file(results_dir: str) -> Path:
    results_path = Path(results_dir)
    candidates = sorted(results_path.glob('paper_results_*.json'))
    if candidates:
        return candidates[-1]
    all_json = sorted(results_path.glob('*.json'))
    if not all_json:
        raise FileNotFoundError(f"No JSON results found in {results_dir}")
    return all_json[-1]


def load_paper_results(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if 'results' in data:
        return data['results'], data.get('meta', {})
    return data, {}


def get_entries(results: Dict[str, Any], dataset: str, method: str, k: int) -> List[Dict[str, Any]]:
    k_key = str(k)
    return results.get(dataset, {}).get(method, {}).get(k_key, [])


def method_label(method: str) -> str:
    labels = {
        'des_knn_pca': 'DES-kNN (Heuristic)',
        'des_knn_guarantee': 'DES-kNN (Guarantee)',
        'hnsw': 'HNSW',
        'annoy': 'Annoy',
    }
    return labels.get(method, method)


def build_frontier_df(results: Dict[str, Any], dataset: str, k: int) -> pd.DataFrame:
    rows = []
    for method in results.get(dataset, {}):
        for entry in get_entries(results, dataset, method, k):
            agg = entry.get('aggregate', {})
            rows.append({
                'dataset': dataset,
                'method': method,
                'label': method_label(method),
                'recall_mean': agg.get('recall_mean'),
                'recall_std': agg.get('recall_std'),
                'query_time_ms_mean': agg.get('query_time_ms_mean'),
                'query_time_ms_std': agg.get('query_time_ms_std'),
                'speedup_mean': agg.get('speedup_mean'),
                'speedup_std': agg.get('speedup_std'),
                'params': json.dumps(entry.get('params', {}), sort_keys=True),
            })
    df = pd.DataFrame(rows)
    return df.dropna(subset=['recall_mean', 'query_time_ms_mean'])


def plot_frontier(df: pd.DataFrame, output_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    styles = {
        'des_knn_pca': {'linestyle': '-', 'marker': 'o'},
        'des_knn_guarantee': {'linestyle': '--', 'marker': 'o'},
        'hnsw': {'linestyle': '-', 'marker': 's'},
        'annoy': {'linestyle': '-', 'marker': '^'},
    }

    for method in df['method'].unique():
        sub = df[df['method'] == method].sort_values('recall_mean')
        style = styles.get(method, {})
        ax.plot(
            sub['recall_mean'],
            sub['query_time_ms_mean'],
            label=method_label(method),
            **style
        )

    ax.set_xlabel('Recall@10')
    ax.set_ylabel('Query Time (ms)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def select_operating_point(entries: List[Dict[str, Any]], recall_target: float) -> Dict[str, Any]:
    valid = []
    for entry in entries:
        agg = entry.get('aggregate', {})
        recall = agg.get('recall_mean')
        time_ms = agg.get('query_time_ms_mean')
        if recall is None or time_ms is None:
            continue
        valid.append((entry, recall, time_ms))
    if not valid:
        return None
    meets = [v for v in valid if v[1] >= recall_target]
    if meets:
        return min(meets, key=lambda v: v[2])[0]
    return max(valid, key=lambda v: v[1])[0]


def build_table(results: Dict[str, Any], dataset: str, k: int, recall_targets: List[float]) -> pd.DataFrame:
    rows = []
    for recall_target in recall_targets:
        for method in results.get(dataset, {}):
            entries = get_entries(results, dataset, method, k)
            selected = select_operating_point(entries, recall_target)
            if not selected:
                continue
            agg = selected.get('aggregate', {})
            mode = '-' if not method.startswith('des_knn') else (
                'guarantee' if 'guarantee' in method else 'heuristic'
            )
            row = {
                'Dataset': dataset,
                'TargetRecall': recall_target,
                'Method': method_label(method),
                'Mode': mode,
                'Recall@10': f"{agg.get('recall_mean', 0):.4f}±{agg.get('recall_std', 0):.4f}",
                'Time(ms)': f"{agg.get('query_time_ms_mean', 0):.2f}±{agg.get('query_time_ms_std', 0):.2f}",
            }
            if method.startswith('des_knn'):
                row['ScanRatio(%)'] = f"{(agg.get('scan_ratio_mean', 0) * 100):.1f}"
                exp_misses = agg.get('expected_misses_median')
                row['ExpectedMisses'] = f"{exp_misses:.2f}" if exp_misses is not None else '-'
            else:
                row['ScanRatio(%)'] = '-'
                row['ExpectedMisses'] = '-'
            if method in ['hnsw', 'annoy']:
                row['BuildTime(s)'] = f"{agg.get('build_time_mean', 0):.2f}"
            else:
                row['BuildTime(s)'] = '-'
            rows.append(row)
    return pd.DataFrame(rows)


def collect_per_query(entry: Dict[str, Any], key: str) -> np.ndarray:
    values = []
    for seed in entry.get('seeds', []):
        per_query = seed.get('per_query', {})
        chunk = per_query.get(key, [])
        values.extend(chunk)
    return np.array(values, dtype=np.float32)


def plot_transparency(
    results: Dict[str, Any],
    datasets: List[str],
    methods: List[str],
    k: int,
    recall_target: float,
    output_path: Path
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_cdf, ax_hist = axes

    for dataset in datasets:
        for method in methods:
            entries = get_entries(results, dataset, method, k)
            selected = select_operating_point(entries, recall_target)
            if not selected:
                continue
            label = f"{dataset}-{method_label(method)}"
            scan_ratios = collect_per_query(selected, 'scan_ratios')
            expected_misses = collect_per_query(selected, 'expected_misses')

            if scan_ratios.size > 0:
                xs = np.sort(scan_ratios)
                ys = np.arange(1, len(xs) + 1) / len(xs)
                ax_cdf.plot(xs, ys, label=label)

            if expected_misses.size > 0:
                ax_hist.hist(expected_misses, bins=30, density=True, alpha=0.4, label=label)

    ax_cdf.set_title('CDF of Scan Ratio per Query')
    ax_cdf.set_xlabel('Scan Ratio')
    ax_cdf.set_ylabel('CDF')
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.legend(fontsize=8)

    ax_hist.set_title('Expected Misses at Stop')
    ax_hist.set_xlabel('Expected Misses')
    ax_hist.set_ylabel('Density')
    ax_hist.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Analyze paper experiment results')
    parser.add_argument('--results_dir', type=str, default='results/paper',
                        help='Directory containing result files')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Specific results JSON (overrides results_dir)')
    parser.add_argument('--output_dir', type=str, default='figures',
                        help='Directory for output figures/tables')
    parser.add_argument('--k', type=int, default=10,
                        help='Value of k for analysis')
    parser.add_argument('--recall_targets', type=str, default='0.95,0.99',
                        help='Comma-separated recall targets for Table 1')
    parser.add_argument('--transparency_recall', type=float, default=0.99,
                        help='Recall target for transparency plots')

    args = parser.parse_args()

    results_path = Path(args.results_file) if args.results_file else select_results_file(args.results_dir)
    results, meta = load_paper_results(results_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recall_targets = [float(x) for x in args.recall_targets.split(',')]

    # Figure 1: matched-recall frontier per dataset
    for dataset in results:
        frontier_df = build_frontier_df(results, dataset, args.k)
        if frontier_df.empty:
            continue
        frontier_csv = output_dir / f'frontier_{dataset}_k{args.k}.csv'
        frontier_df.to_csv(frontier_csv, index=False)
        plot_frontier(
            frontier_df,
            output_dir / f'fig1_frontier_{dataset}_k{args.k}.png',
            title=f"{dataset} (k={args.k})"
        )

        table_df = build_table(results, dataset, args.k, recall_targets)
        if not table_df.empty:
            table_df.to_csv(output_dir / f'table1_{dataset}_k{args.k}.csv', index=False)
            table_df.to_latex(output_dir / f'table1_{dataset}_k{args.k}.tex', index=False, escape=False)

    # Figure 2: transparency (clustered vs uniform)
    plot_transparency(
        results,
        datasets=['synthetic_clustered', 'synthetic_uniform'],
        methods=['des_knn_guarantee', 'des_knn_pca'],
        k=args.k,
        recall_target=args.transparency_recall,
        output_path=output_dir / f'fig2_transparency_k{args.k}.png'
    )

    print(f"Analysis complete. Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
