# Positioning DES‑kNN: Transparent Early‑Stopping for Scan‑Based kNN

## What DES‑kNN is (and is not)

DES‑kNN is best framed as an **early‑termination layer for scan‑based kNN** (exact or near‑exact), not as a replacement for specialized ANN indices such as **HNSW**, **Annoy**, or **FAISS**.

- **ANN indices** (HNSW/Annoy/FAISS) win on large benchmarks because they invest in an index structure and optimized native code to avoid most distance evaluations.
- **DES‑kNN** keeps the simplicity and transparency of scanning but **stops early** using a statistically interpretable rule, returning a k‑NN set together with evidence about *how safe it was to stop*.

This positioning avoids an unfair “head‑to‑head” claim against HNSW and instead highlights DES‑kNN’s niche: **controllable risk + minimal infrastructure**.

## Problem setting: when scan‑based kNN is still relevant

Even when ANN indices exist, scan‑based kNN is common in:

- **No/low indexing budget**: ephemeral workloads, rapidly changing datasets, or one‑off experiments.
- **Transparency / auditability**: need to explain *why* a query stopped early (how much was scanned, what bound was satisfied).
- **Index‑free deployment constraints**: memory limits, strict dependency policies, or environments where building/serving indices is undesirable.
- **Two‑stage retrieval**: scan (or partial scan) as a second‑stage re‑ranker after a coarse retrieval step.

In these settings, the key question becomes:
> “How far do I need to scan before I can stop with controlled risk?”

## Core contribution: a gap‑based, confidence‑calibrated stopping rule

DES‑kNN tracks the **gap**: the number of consecutive scanned points since the last “success” (a point that improves the current k‑NN set). It then applies a lightweight **Beta‑Geometric** model to upper bound the probability of future successes and estimate the expected number of missed neighbors if stopping now.

Concrete, transparent outputs that are directly useful in papers and systems:

- **`scan_ratio`**: fraction of the dataset actually evaluated (true distance computations).
- **`expected_misses`**: confidence‑calibrated estimate of how many better neighbors remain.
- **`confidence` / `tolerance`**: user‑controlled accuracy–speed knob with a statistical interpretation.

Code references:
- Main implementation: `src/des_knn.py`
- Bound computation: `src/statistics.py`

## “Guarantee mode” vs “heuristic mode”

It is useful (and honest) to present DES‑kNN as having two operating modes:

### 1) Guarantee mode (assumption‑aligned)

Goal: make the Beta‑Geometric assumptions defensible.

Two issues to address:
1) **Random order**: the bound assumes i.i.d. trials under random scan order.
2) **Stationarity**: the “success” event should be defined against a fixed threshold, otherwise the success probability changes as the k‑NN radius tightens.

This repo includes a conservative variant that enforces both:
- Random order via `RandomOrderSorter` in `src/sorting.py`
- Fixed reference threshold after `min_scan` in `src/des_knn_guarantee.py` (`DESKNNSearcherGuarantee`)

In a paper, this mode supports language like:
> “Under random order and a fixed success threshold, we provide a confidence‑calibrated early‑stopping criterion that bounds the expected number of missed improvements.”

### 2) Heuristic mode (performance‑oriented)

Goal: improve practical speed by scanning likely‑near points early.

Sorters such as PCA/cluster ordering can massively improve performance on clustered data, but they **break the random‑order assumption**. In this mode the bound becomes a **stopping heuristic** (often still useful, but no longer a strict guarantee).

Code references:
- `src/sorting.py` (`PCASorter`, `ClusterSorter`)
- Method entry points in experiments: `experiments/run_experiments.py`

Empirical note (MNIST, PCA sorter): with a higher tolerance (e.g., 146.0), we observed Recall@10 = 1.0 with ~6x speedup and ~3.4% scan ratio over 1000 queries. This highlights that tolerance tuning can materially change the speed/recall tradeoff, even on datasets where default settings underperform.

## How to compare fairly against ANN libraries

For a scientific evaluation, treat ANN libraries as *strong baselines*, but compare at **matched recall**:

- Sweep HNSW (`ef`) and Annoy (`search_k`) until Recall@k meets a target (e.g., 0.95 / 0.99 / 1.00).
- Then compare wall‑clock query time and memory; avoid mixing “scan ratio” with ANN backends because they do not expose true distance counts.

Suggested reporting format:

- **Primary**: Recall@k vs Query Time (or Speedup), with confidence intervals.
- **Secondary (DES‑kNN only)**: Scan ratio and expected_misses distributions (transparency story).
- **Build cost**: include sorter/index training consistently (important for PCA/cluster sorters).

## Clear, defensible paper claims

Examples of claims that are typically easier to defend than “beating HNSW”:

- **Transparent early stopping**: DES‑kNN returns not just neighbors, but also a calibrated stopping certificate (`expected_misses`, scan ratio).
- **Index‑free acceleration**: meaningful speedups over brute force on large/clustered data without building a complex ANN index.
- **Controllable risk**: a user‑facing tolerance/confidence knob with a statistical interpretation (strongest in guarantee mode).

## Known limitations (good to state explicitly)

- DES‑kNN is not expected to outperform state‑of‑the‑art ANN indices on large ANN benchmarks when those indices are properly tuned.
- Heuristic ordering improves speed but weakens theoretical guarantees; in practice, treat it as “empirically calibrated stopping.”
- Sorting can dominate runtime on smaller datasets; report preprocessing/build time honestly.
