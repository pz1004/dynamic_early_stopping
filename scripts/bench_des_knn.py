#!/usr/bin/env python3
"""
Benchmark harness for DES-kNN.

Measures recall@k against an exact brute-force oracle, wall time per query,
and distance computation counts.
"""

import argparse
import json
import time
from pathlib import Path
import sys

import numpy as np
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.des_knn import DESKNNSearcher
from src.baselines.exact_brute_force import ExactBruteForceKNN


def parse_bool(value: str) -> bool:
    """Parse flexible boolean values from CLI."""
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "yes", "y", "t"}:
        return True
    if value in {"0", "false", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def summarize(values: List[float]) -> dict:
    """Return summary statistics for a list of numeric values."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def aggregate_criteria(criteria_histories: List[List[Dict]]) -> Dict:
    """Aggregate stop criteria stats across queries."""
    flat = [item for history in criteria_histories for item in history]
    if not flat:
        return {}

    def extract(key: str) -> List[float]:
        return [float(item[key]) for item in flat]

    crit1_vals = [bool(item["crit1"]) for item in flat]
    crit2_vals = [bool(item["crit2"]) for item in flat]
    crit3_vals = [bool(item["crit3"]) for item in flat]
    two_of_three = [
        (int(c1) + int(c2) + int(c3)) >= 2
        for c1, c2, c3 in zip(crit1_vals, crit2_vals, crit3_vals)
    ]

    return {
        "evaluations": len(flat),
        "p_remain": summarize(extract("p_remain")),
        "expected_entrants": summarize(extract("expected_entrants")),
        "confidence": summarize(extract("confidence")),
        "gap_ratio": summarize(extract("gap_ratio")),
        "gap": summarize(extract("gap")),
        "alpha": summarize(extract("alpha")),
        "crit1_threshold": summarize(extract("crit1_threshold")),
        "crit3_gap_ratio": summarize(extract("crit3_gap_ratio")),
        "crit3_gap_threshold": summarize(extract("crit3_gap_threshold")),
        "crit1_rate": float(np.mean(crit1_vals)),
        "crit2_rate": float(np.mean(crit2_vals)),
        "crit3_rate": float(np.mean(crit3_vals)),
        "two_of_three_rate": float(np.mean(two_of_three)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark DES-kNN vs exact k-NN")
    parser.add_argument("--n", type=int, default=10000, help="Number of points")
    parser.add_argument("--d", type=int, default=128, help="Dimensionality")
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors")
    parser.add_argument("--n-queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repeats")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON output path")
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["default", "aggressive"],
        help="Alpha preset (default=0.01, aggressive=0.05)",
    )
    parser.add_argument(
        "--stop-check-every",
        type=int,
        default=1,
        help="Only evaluate early stopping every N distance computations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for distance computations",
    )
    parser.add_argument(
        "--use-weibull",
        type=parse_bool,
        default=True,
        help="Use Weibull fit for exceedance probability (true/false)",
    )
    parser.add_argument(
        "--adaptive-alpha",
        type=parse_bool,
        default=True,
        help="Use adaptive alpha (true/false)",
    )
    parser.add_argument(
        "--weibull-refresh-every",
        type=int,
        default=1,
        help="Refresh Weibull fit every N stop checks (default 1)",
    )
    parser.add_argument(
        "--crit1-mode",
        type=str,
        default="alpha_over_k",
        choices=["alpha_over_k", "alpha"],
        help="Criterion-1 threshold mode (default alpha_over_k)",
    )
    parser.add_argument(
        "--crit3-gap-ratio",
        type=float,
        default=0.5,
        help="Criterion-3 gap ratio threshold",
    )
    parser.add_argument(
        "--crit3-gap-mult",
        type=float,
        default=10.0,
        help="Criterion-3 gap multiplier applied to k",
    )
    parser.add_argument(
        "--criteria-queries",
        type=int,
        default=0,
        help="Collect stop-criteria stats for first N queries in the first repeat",
    )

    args = parser.parse_args()

    if args.n < 1 or args.d < 1 or args.n_queries < 1:
        raise ValueError("--n, --d, and --n-queries must be positive")

    alpha = 0.01 if args.mode == "default" else 0.05

    rng = np.random.default_rng(args.seed)
    X = rng.standard_normal((args.n, args.d)).astype(np.float32)
    queries = rng.standard_normal((args.n_queries, args.d)).astype(np.float32)

    exact = ExactBruteForceKNN(X)
    exact.fit()

    exact_neighbors = []
    for q in queries:
        neighbors, _, _ = exact.query(q, args.k)
        exact_neighbors.append(neighbors)

    k_eff = min(args.k, args.n)

    all_times = []
    all_dist_counts = []
    all_recalls = []
    criteria_histories = []

    for repeat in range(args.repeats):
        searcher = DESKNNSearcher(
            X,
            alpha=alpha,
            batch_size=args.batch_size,
            use_weibull=args.use_weibull,
            adaptive_alpha=args.adaptive_alpha,
            weibull_refresh_every=args.weibull_refresh_every,
            random_state=args.seed + repeat,
        )

        for i, q in enumerate(queries):
            collect_criteria = args.criteria_queries > 0 and repeat == 0 and i < args.criteria_queries
            t0 = time.perf_counter()
            if collect_criteria:
                neighbors, _, dist_count, stats = searcher.query(
                    q,
                    args.k,
                    stop_check_every=args.stop_check_every,
                    crit1_mode=args.crit1_mode,
                    crit3_gap_ratio=args.crit3_gap_ratio,
                    crit3_gap_mult=args.crit3_gap_mult,
                    return_stats=True,
                )
                criteria_histories.append(stats.get("criteria_history", []))
            else:
                neighbors, _, dist_count = searcher.query(
                    q,
                    args.k,
                    stop_check_every=args.stop_check_every,
                    crit1_mode=args.crit1_mode,
                    crit3_gap_ratio=args.crit3_gap_ratio,
                    crit3_gap_mult=args.crit3_gap_mult,
                )
            t1 = time.perf_counter()

            all_times.append(t1 - t0)
            all_dist_counts.append(dist_count)

            if k_eff > 0:
                recall = (
                    len(set(neighbors[:k_eff]) & set(exact_neighbors[i][:k_eff]))
                    / k_eff
                )
            else:
                recall = 1.0
            all_recalls.append(recall)

    time_summary = summarize(all_times)
    dist_summary = summarize(all_dist_counts)
    recall_summary = summarize(all_recalls)
    speedup = args.n / dist_summary["mean"] if dist_summary["mean"] > 0 else 0.0

    criteria_summary = aggregate_criteria(criteria_histories)
    result = {
        "config": {
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "n_queries": args.n_queries,
            "seed": args.seed,
            "repeats": args.repeats,
            "mode": args.mode,
            "alpha": alpha,
            "stop_check_every": args.stop_check_every,
            "batch_size": args.batch_size,
            "use_weibull": args.use_weibull,
            "adaptive_alpha": args.adaptive_alpha,
            "weibull_refresh_every": args.weibull_refresh_every,
            "crit1_mode": args.crit1_mode,
            "crit3_gap_ratio": args.crit3_gap_ratio,
            "crit3_gap_mult": args.crit3_gap_mult,
            "criteria_queries": args.criteria_queries,
        },
        "metrics": {
            "recall": recall_summary,
            "time_seconds": time_summary,
            "dist_count": dist_summary,
            "speedup_estimate": speedup,
        },
    }
    if criteria_summary:
        result["criteria_summary"] = criteria_summary

    print("DES-kNN benchmark")
    print(
        f"n={args.n} d={args.d} k={args.k} queries={args.n_queries} repeats={args.repeats}"
    )
    print(
        f"mode={args.mode} alpha={alpha} batch_size={args.batch_size} "
        f"stop_check_every={args.stop_check_every} weibull_refresh_every={args.weibull_refresh_every}"
    )
    print(
        f"recall@{k_eff}: mean={recall_summary['mean']:.4f} "
        f"min={recall_summary['min']:.4f}"
    )
    print(
        "time_per_query_s: mean={mean:.6f} p50={p50:.6f} p95={p95:.6f}".format(
            **time_summary
        )
    )
    print(
        "dist_count: mean={mean:.1f} min={min:.0f} max={max:.0f} "
        "speedup~{speedup:.2f}x".format(
            speedup=speedup,
            **dist_summary,
        )
    )
    if criteria_summary:
        print(
            "criteria: evals={evaluations} crit1={crit1_rate:.2f} "
            "crit2={crit2_rate:.2f} crit3={crit3_rate:.2f} "
            "two_of_three={two_of_three_rate:.2f}".format(**criteria_summary)
        )
        print(
            "p_remain: mean={mean:.4f} p50={p50:.4f} p95={p95:.4f}".format(
                **criteria_summary["p_remain"]
            )
        )
        print(
            "confidence: mean={mean:.4f} p50={p50:.4f} p95={p95:.4f}".format(
                **criteria_summary["confidence"]
            )
        )
        print(
            "gap_ratio: mean={mean:.4f} p50={p50:.4f} p95={p95:.4f}".format(
                **criteria_summary["gap_ratio"]
            )
        )
        print(
            "gap: mean={mean:.2f} p50={p50:.2f} p95={p95:.2f}".format(
                **criteria_summary["gap"]
            )
        )
        print(
            "crit1_threshold: mean={mean:.6f} p50={p50:.6f} p95={p95:.6f}".format(
                **criteria_summary["crit1_threshold"]
            )
        )
        print(
            "crit3_gap_ratio: mean={mean:.4f} p50={p50:.4f} p95={p95:.4f}".format(
                **criteria_summary["crit3_gap_ratio"]
            )
        )
        print(
            "crit3_gap_threshold: mean={mean:.2f} p50={p50:.2f} p95={p95:.2f}".format(
                **criteria_summary["crit3_gap_threshold"]
            )
        )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote results to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
