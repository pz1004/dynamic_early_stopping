# DES-kNN Core Algorithm Implementation

## File: `src/des_knn.py`

## Overview

`DESKNNSearcher` implements **Dynamic Early Stopping k-NN (DES‑kNN)** using a lightweight **Beta‑Geometric “gap” model**. It scans points in some order, maintains a best‑k heap, and checks a confidence‑calibrated stopping criterion that estimates how many better neighbors might remain unseen.

The current implementation is optimized for Python by:

- Scanning in **blocks** (`block_size`) and computing distances with vectorized NumPy operations.
- Separating **ordering** (via a `sorter`) from **stopping** (via `estimate_future_matches()`).

## Key Modules Used

```python
import numpy as np
import heapq
from typing import Tuple, Optional, Dict

from .statistics import estimate_future_matches
from .sorting import BaseSorter, PCASorter  # other sorters (e.g., ClusterSorter) live in sorting.py
from joblib import Parallel, delayed
```

## Class: `DESKNNSearcher`

### Constructor (matches `src/des_knn.py`)

Parameters of note:

- `tolerance`: acceptable expected number of missed neighbors (higher stops earlier).
- `confidence`: confidence level used to upper bound the success probability.
- `min_samples`: don’t stop before scanning at least this many points.
- `max_cv`: optional safety check to avoid stopping when the current k distances are highly dispersed.
- `sorter`: controls the scan order (random by default via `BaseSorter`).
- `block_size`: vectorization granularity; stopping is checked once per block.

### Fit

`fit()` is a no‑op for API consistency. Some sorters (e.g., PCA) are fit during `__init__` so `query()` can assume ordering is ready.

## Core Method: `query()`

Signature:

```python
neighbors, distances, dist_count = searcher.query(q, k)
neighbors, distances, dist_count, stats = searcher.query(q, k, return_stats=True)
```

What it does:

1) Compute a scan order:
- Default `BaseSorter` returns a random permutation (assumption‑aligned).
- PCA/cluster sorters return a heuristic order (often faster in practice).

2) Process points in blocks:
- For each block: compute all distances in a vectorized batch.
- Update a max‑heap (stored as negative distances) to keep best k.

3) After `min_samples`, check the stopping rule once per block:

- `gap = (current_position - last_update_index)`
- If `gap > k` then compute `expected_misses = estimate_future_matches(gap, remaining, confidence)`
- Stop when `expected_misses < tolerance` and (optionally) the CV check passes.

Returned statistics (`return_stats=True`) include:

- `scan_ratio`: `dist_count / n` (true fraction of distances computed by DES‑kNN)
- `expected_misses`: the final bound value at the stop decision
- `stopped_early`, `last_update_index`, `final_gap`

## Distance Computation

The main query path uses `_compute_distances_batch()` for speed:

- Euclidean: vectorized squared distance then square root
- Manhattan: vectorized absolute deviation
- Cosine: dot product on pre‑normalized data (when enabled)

`_compute_distance_fast()` remains as a single‑point helper (and is used by a legacy wrapper), but is not the hot path for `query()`.

## Batch Queries: `query_batch()`

`query_batch()` runs `query()` repeatedly:

- `n_jobs=1`: simple Python loop
- `n_jobs>1`: uses `joblib.Parallel` to parallelize across queries

It returns:

- `all_neighbors`: `(n_queries, k)`
- `all_distances`: `(n_queries, k)`
- `all_dist_counts`: `(n_queries,)`

## Algorithm: Beta‑Geometric Gap Model (as implemented)

The statistics module models “success” events via a simple Bayesian bound:

- Let a success be “the next scanned point improves the current k‑NN threshold”.
- After observing a gap of `G` consecutive failures, compute an upper bound `p_max` on the success probability at confidence `c`.
- Estimate expected remaining successes as `remaining * p_max`.

DES‑kNN stops when:

```
expected_misses < tolerance
```

Implementation details live in:
- `src/statistics.py` (`estimate_future_matches`)

Important note for interpretation:
- The Beta‑Geometric story is most assumption‑aligned when the scan order is random (BaseSorter). With heuristic sorters (PCA/cluster), treat the bound as an empirically effective stopping heuristic unless you enforce guarantee-mode assumptions separately.

## Pseudocode

### DES‑kNN with PCA ordering (`des_knn_pca`, heuristic mode)

```
Inputs:
  X (n×d), query q, k
  tolerance τ, confidence c
  min_samples m, block_size B

Preprocess (once):
  Fit PCA on X to obtain reduced vectors X_r

Per-query:
  q_r ← PCA.transform(q)
  order ← argsort_i ||X_r[i] - q_r||^2            # heuristic scan order

  heap ← empty max-heap of size ≤ k storing (-dist, idx)
  d_k ← +∞
  last_update ← 0
  dist_count ← 0

  for block_start in {0, B, 2B, ...}:
    block ← order[block_start : min(block_start+B, n)]
    D ← dist(q, X[block])                        # vectorized true distances
    dist_count += |block|

    for (j, (d, idx)) in enumerate(zip(D, block)):
      pos ← block_start + j
      if heap.size < k:
        heap.push((-d, idx))
        if heap.size == k:
          d_k ← -heap.top().dist
          last_update ← pos
      else if d < d_k:
        heap.replace_top((-d, idx))
        d_k ← -heap.top().dist
        last_update ← pos

    pos_end ← min(block_start+B, n) - 1
    if pos_end ≥ m:
      gap ← pos_end - last_update
      if gap > k:
        remaining ← n - (pos_end + 1)
        expected_misses ← estimate_future_matches(gap, remaining, c)
        if expected_misses < τ and (optional CV-check passes):
          break

  return heap.sorted_by_distance(), dist_count, (optional stats)
```

### DES‑kNN Guarantee Variant (`des_knn_guarantee`, assumption‑aligned)

This matches `src/des_knn_guarantee.py`. The key differences are:

- The scan order is enforced to be random (`RandomOrderSorter`).
- The stopping model uses a fixed `reference_distance` so the success event is stationary:
  “success ⇔ dist(q, x) < reference_distance”.

```
Inputs:
  X (n×d), query q, k
  tolerance τ, confidence c
  min_samples m, block_size B

Per-query:
  order ← RandomPermutation(0..n-1)
  heap ← empty max-heap of size ≤ k
  d_k ← +∞
  dist_count ← 0

  reference_distance ← None
  reference_set_at ← None
  last_success ← None

  for block_start in {0, B, 2B, ...}:
    block ← order[block_start : min(block_start+B, n)]
    D ← dist(q, X[block])                        # vectorized true distances
    dist_count += |block|

    for (j, (d, idx)) in enumerate(zip(D, block)):
      pos ← block_start + j
      update heap / d_k using d                  # same as above
      if reference_distance is not None and d < reference_distance:
        last_success ← pos

    pos_end ← min(block_start+B, n) - 1

    if reference_distance is None and pos_end ≥ m and heap.size == k:
      reference_distance ← d_k
      reference_set_at ← pos_end
      last_success ← pos_end                     # start counting the gap after reference is fixed
    else if reference_distance is not None:
      gap ← pos_end - last_success
      if gap > k:
        remaining ← n - (pos_end + 1)
        expected_misses ← estimate_future_matches(gap, remaining, c)
        if expected_misses < τ:
          break

  return heap.sorted_by_distance(), dist_count, stats where:
    expected_misses is defined relative to reference_distance
```

## Guarantee Variant: `DESKNNSearcherGuarantee`

This repo also includes a conservative, assumption-aligned variant in `src/des_knn_guarantee.py`:

- Enforces **random scan order** (requires `RandomOrderSorter` from `src/sorting.py`).
- Freezes a fixed `reference_distance` once `min_samples` is reached so the “success” event is stationary for the bound.

In this guarantee variant, `expected_misses` is explicitly defined relative to `reference_distance` (see `expected_misses_definition` in the returned stats).

Usage:

```python
from src.des_knn_guarantee import DESKNNSearcherGuarantee

searcher = DESKNNSearcherGuarantee(X, tolerance=0.5, confidence=0.99, random_state=42)
neighbors, distances, dist_count, stats = searcher.query(q, k=10, return_stats=True)
print(stats["reference_distance"], stats["expected_misses"])
```

## Usage Example

```python
import numpy as np

from src.des_knn import DESKNNSearcher
from src.sorting import BaseSorter, PCASorter

np.random.seed(42)
X = np.random.randn(10_000, 128).astype(np.float32)
q = np.random.randn(128).astype(np.float32)
k = 10

# Assumption-aligned (random order)
searcher = DESKNNSearcher(X, tolerance=0.5, confidence=0.99, sorter=BaseSorter(), block_size=256, random_state=42)
neighbors, distances, dist_count, stats = searcher.query(q, k, return_stats=True)

# Heuristic mode (often faster; no strict random-order assumption)
searcher_pca = DESKNNSearcher(X, tolerance=1.0, confidence=0.99, sorter=PCASorter(n_components=32), block_size=256, random_state=42)
neighbors, distances, dist_count = searcher_pca.query(q, k)
```

## Parameter Tuning (practical guidance)

| Parameter | Effect | Notes |
|---|---|---|
| `tolerance` | Higher = earlier stop | Can be dataset-dependent; values can be much larger than 1.0 in practice. |
| `confidence` | Higher = more conservative | Typical: 0.95–0.999. |
| `min_samples` | Minimum scan before stopping | Default is `max(k+50, 1% of n)`. |
| `max_cv` | Safety against dispersed k-NN set | Smaller = more conservative; set `None` to disable. |
| `block_size` | Vectorization granularity | 64–1024; larger reduces Python overhead but checks stopping less frequently. |
| `sorter` | Scan ordering | `BaseSorter` (random), `PCASorter`, `ClusterSorter`. |

## Tests

Implemented tests live in:
- `tests/test_des_knn.py`
- `tests/test_des_knn_correctness.py`
- `tests/test_statistics.py`
