#!/usr/bin/env python3
"""
Analyze paper experiment results and generate plots/tables.

Usage:
    python analyze_results.py --results_dir results/paper --output_dir figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_ORDER = [
    'synthetic_uniform',
    'synthetic_clustered',
    'mnist',
    'fashion_mnist',
    'sift1m',
]


def order_datasets(datasets: List[str]) -> List[str]:
    ordered = [d for d in DATASET_ORDER if d in datasets]
    remainder = [d for d in datasets if d not in DATASET_ORDER]
    return ordered + remainder


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
        'exact': 'Brute-force',
        'des_knn_pca': 'DES-kNN (heuristic)',
        'des_knn_guarantee': 'DES-kNN (guarantee)',
        'hnsw': 'HNSW',
        'annoy': 'Annoy',
    }
    return labels.get(method, method)

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


def select_best_speedup(
    entries: List[Dict[str, Any]],
    exact_time: Optional[float],
    min_recall: float
) -> Optional[Dict[str, Any]]:
    if exact_time is None:
        return None
    candidates = []
    for entry in entries:
        agg = entry.get('aggregate', {})
        recall = agg.get('recall_mean')
        time_ms = agg.get('query_time_ms_mean')
        if recall is None or time_ms is None:
            continue
        if recall <= min_recall:
            continue
        speedup = exact_time / time_ms if time_ms else None
        if speedup is None:
            continue
        candidates.append((speedup, recall, entry))
    if not candidates:
        return None
    return max(candidates, key=lambda v: (v[0], v[1]))[2]


def get_exact_time(results: Dict[str, Any], dataset: str, k: int) -> Optional[float]:
    entries = get_entries(results, dataset, 'exact', k)
    if not entries:
        return None
    best = max(entries, key=lambda e: e.get('aggregate', {}).get('recall_mean') or -1)
    return best.get('aggregate', {}).get('query_time_ms_mean')


def build_table_rows(
    results: Dict[str, Any],
    datasets: List[str],
    k: int,
    recall_targets: List[float]
) -> List[Dict[str, Any]]:
    methods_order = ['exact', 'hnsw', 'annoy', 'des_knn_guarantee', 'des_knn_pca']
    rows = []
    for dataset in datasets:
        exact_time = get_exact_time(results, dataset, k)
        if exact_time is None:
            continue
        for recall_target in recall_targets:
            for method in methods_order:
                entries = get_entries(results, dataset, method, k)
                if method in ['hnsw', 'annoy']:
                    selected = select_best_speedup(entries, exact_time, min_recall=0.99)
                    if selected is None:
                        selected = select_operating_point(entries, recall_target)
                else:
                    selected = select_operating_point(entries, recall_target)
                if not selected:
                    continue
                agg = selected.get('aggregate', {})
                time_ms = agg.get('query_time_ms_mean')
                time_ms_std = agg.get('query_time_ms_std')
                recall_mean = agg.get('recall_mean')
                recall_std = agg.get('recall_std')
                scan_ratio = agg.get('scan_ratio_mean')
                scan_ratio_std = agg.get('scan_ratio_std')
                speedup = (exact_time / time_ms) if time_ms else None
                speedup_std = agg.get('speedup_std')
                row_target = recall_target
                if method in ['hnsw', 'annoy'] and row_target < 0.99:
                    row_target = 0.99
                rows.append({
                    'dataset': dataset,
                    'method': method_label(method),
                    'target_recall': row_target,
                    'time_ms': time_ms,
                    'time_ms_std': time_ms_std,
                    'speedup': speedup,
                    'speedup_std': speedup_std,
                    'scan_ratio': scan_ratio,
                    'scan_ratio_std': scan_ratio_std,
                    'recall_mean': recall_mean,
                    'recall_std': recall_std,
                    'is_des': method.startswith('des_knn'),
                    'is_exact': method == 'exact',
                })
    return rows


def format_decimal(value: Optional[float], std: Optional[float] = None) -> str:
    if value is None:
        return '-'
    if std is not None:
        return f"{value:.4f} $\\pm$ {std:.4f}"
    return f"{value:.4f}"


def format_speedup(speedup: Optional[float], std: Optional[float] = None) -> str:
    if speedup is None:
        return '-'
    if std is not None:
        return f"{speedup:.4f} $\\pm$ {std:.4f}$\\times$"
    return f"{speedup:.4f}$\\times$"


def format_scan_ratio(scan_ratio: Optional[float], std: Optional[float], is_des: bool, is_exact: bool) -> str:
    if is_exact:
        return "1.0000"
    if not is_des or scan_ratio is None:
        return "--"
    if std is not None:
        return f"{scan_ratio:.4f} $\\pm$ {std:.4f}"
    return f"{scan_ratio:.4f}"


def write_table_latex(
    output_path: Path,
    caption: str,
    rows: List[Dict[str, Any]]
) -> None:
    lines = [
        "\\begin{table*}[tb]\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:results}",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Dataset & Method & Recall@10 & Time (ms) & Speedup & Scan Ratio \\\\",
        "\\hline",
    ]

    grouped = {}
    for row in rows:
        grouped.setdefault(row['dataset'], []).append(row)

    for dataset, dataset_rows in grouped.items():
        for i, row in enumerate(dataset_rows):
            dataset_cell = f"\\multirow{{{len(dataset_rows)}}}{{*}}{{{dataset}}}" if i == 0 else ""
            recall_mean = format_decimal(row['recall_mean'], row.get('recall_std'))
            time_ms = format_decimal(row['time_ms'], row.get('time_ms_std'))
            speedup = format_speedup(1.0 if row['is_exact'] else row['speedup'],
                                     None if row['is_exact'] else row.get('speedup_std'))
            scan_ratio = format_scan_ratio(row['scan_ratio'], row.get('scan_ratio_std'),
                                           row['is_des'], row['is_exact'])
            prefix = f"{dataset_cell} &" if dataset_cell else " &"
            lines.append(
                f"{prefix} {row['method']} & {recall_mean} & {time_ms} & "
                f"{speedup} & {scan_ratio} \\\\"
            )
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    output_path.write_text("\n".join(lines))


def build_table2_rows(
    results: Dict[str, Any],
    datasets: List[str],
    k: int
) -> List[Dict[str, Any]]:
    rows = []
    for dataset in datasets:
        exact_time = get_exact_time(results, dataset, k)
        if exact_time is None:
            continue
        entries = get_entries(results, dataset, 'des_knn_guarantee', k)
        for entry in entries:
            agg = entry.get('aggregate', {})
            params = entry.get('params', {})
            tolerance = params.get('tolerance')
            time_ms = agg.get('query_time_ms_mean')
            recall_mean = agg.get('recall_mean')
            recall_std = agg.get('recall_std')
            speedup_std = agg.get('speedup_std')
            if tolerance is None or time_ms is None or recall_mean is None:
                continue
            speedup = exact_time / time_ms if time_ms else None
            rows.append({
                'dataset': dataset,
                'tolerance': tolerance,
                'recall_mean': recall_mean,
                'recall_std': recall_std,
                'speedup': speedup,
                'speedup_std': speedup_std,
            })
    rows.sort(key=lambda r: (r['dataset'], r['tolerance']))
    return rows


def write_table2_latex(
    output_path: Path,
    caption: str,
    rows: List[Dict[str, Any]]
) -> None:
    lines = [
        "\\begin{table*}[tb]\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:results_tolerance}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Dataset & Tolerance ($\\tau$) & Recall@10 & Speedup \\\\",
        "\\hline",
    ]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row['dataset'], []).append(row)

    for dataset, dataset_rows in grouped.items():
        for i, row in enumerate(dataset_rows):
            dataset_cell = f"\\multirow{{{len(dataset_rows)}}}{{*}}{{{dataset}}}" if i == 0 else ""
            tol = format_decimal(float(row['tolerance']))
            recall = format_decimal(row['recall_mean'], row.get('recall_std'))
            speedup = format_speedup(row['speedup'], row.get('speedup_std'))
            prefix = f"{dataset_cell} &" if dataset_cell else " &"
            lines.append(f"{prefix} {tol} & {recall} & {speedup} \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    output_path.write_text("\n".join(lines))


def build_table3_rows(
    results: Dict[str, Any],
    datasets: List[str],
    k: int
) -> List[Dict[str, Any]]:
    rows = []
    for dataset in datasets:
        exact_time = get_exact_time(results, dataset, k)
        if exact_time is None:
            continue
        entries = get_entries(results, dataset, 'des_knn_pca', k)
        for entry in entries:
            agg = entry.get('aggregate', {})
            params = entry.get('params', {})
            tolerance = params.get('tolerance')
            time_ms = agg.get('query_time_ms_mean')
            recall_mean = agg.get('recall_mean')
            recall_std = agg.get('recall_std')
            speedup_std = agg.get('speedup_std')
            if tolerance is None or time_ms is None or recall_mean is None:
                continue
            speedup = exact_time / time_ms if time_ms else None
            rows.append({
                'dataset': dataset,
                'tolerance': tolerance,
                'recall_mean': recall_mean,
                'recall_std': recall_std,
                'speedup': speedup,
                'speedup_std': speedup_std,
            })
    rows.sort(key=lambda r: (r['dataset'], r['tolerance']))
    return rows


def write_table3_latex(
    output_path: Path,
    caption: str,
    rows: List[Dict[str, Any]]
) -> None:
    lines = [
        "\\begin{table*}[tb]\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:results_tolerance_pca}",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Dataset & Tolerance ($\\tau$) & Recall@10 & Speedup \\\\",
        "\\hline",
    ]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row['dataset'], []).append(row)

    for dataset, dataset_rows in grouped.items():
        for i, row in enumerate(dataset_rows):
            dataset_cell = f"\\multirow{{{len(dataset_rows)}}}{{*}}{{{dataset}}}" if i == 0 else ""
            tol = format_decimal(float(row['tolerance']))
            recall = format_decimal(row['recall_mean'], row.get('recall_std'))
            speedup = format_speedup(row['speedup'], row.get('speedup_std'))
            prefix = f"{dataset_cell} &" if dataset_cell else " &"
            lines.append(f"{prefix} {tol} & {recall} & {speedup} \\\\")
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    output_path.write_text("\n".join(lines))


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
    parser.add_argument('--recall_targets', type=str, default='0.99',
                        help='Comma-separated recall targets for Table 1')

    args = parser.parse_args()

    results_path = Path(args.results_file) if args.results_file else select_results_file(args.results_dir)
    results, meta = load_paper_results(results_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    recall_targets = [float(x) for x in args.recall_targets.split(',')]

    datasets = meta.get('datasets') or list(results.keys())
    datasets = order_datasets(datasets)

    table_rows = build_table_rows(results, datasets, args.k, recall_targets)
    if table_rows:
        table_caption = (
            "Performance of DES-kNN and baselines at fixed recall targets. "
            "Speedup is relative to brute-force time. "
            "Scan ratio is the fraction of points scanned (DES-kNN only)."
        )
        write_table_latex(output_dir / f'table1_k{args.k}.tex', caption=table_caption, rows=table_rows)

    table2_rows = build_table2_rows(results, datasets, args.k)
    if table2_rows:
        table2_caption = (
            "DES-kNN (guarantee) sensitivity to tolerance $\\tau$ across datasets. "
            "Speedup is relative to brute-force time."
        )
        write_table2_latex(output_dir / f'table2_k{args.k}.tex', caption=table2_caption, rows=table2_rows)

    table3_rows = build_table3_rows(results, datasets, args.k)
    if table3_rows:
        table3_caption = (
            "DES-kNN (heuristic) sensitivity to tolerance $\\tau$ across datasets. "
            "Speedup is relative to brute-force time."
        )
        write_table3_latex(output_dir / f'table3_k{args.k}.tex', caption=table3_caption, rows=table3_rows)

    print(f"Analysis complete. Outputs saved to {output_dir}")


if __name__ == '__main__':
    main()
