# DSPG Synthesis and Analysis — Empirical Evaluation

This repository provides the implementation and evaluation code for **Section 4.7.2** of the chapter:

> *"Optimization of DSPG Synthesis and Trust Network Analysis with Subjective Logic"*

It empirically compares **two DSPG synthesis strategies** — Jøsang's original criterion and a proposed optimized criterion — across randomly generated trust networks, measuring information retention, computational cost, and the uncertainty of the derived opinion.

---

## Background

### Subjective Logic and Trust Networks

**Subjective Logic (SL)** is a framework for reasoning under uncertainty, where an agent's belief about a proposition is expressed as a *binomial opinion* — a 4-tuple (b, d, u, a) encoding belief, disbelief, uncertainty, and base rate. Trust networks model chains and webs of trust relationships, each edge carrying an opinion.

### Dispersed Series-Parallel Graphs (DSPGs)

A **Dispersed Series-Parallel Graph (DSPG)** is a directed acyclic graph that can be reduced to a single edge through repeated application of two reductions:

- **Series reduction**: replace a chain `A → B → C` with `A → C`.
- **Parallel reduction**: collapse a set of parallel paths between two nodes into a single edge by fusing their opinions.

Only graphs with this structure admit a closed-form, lossless reduction to a source-to-sink opinion — a property essential for trust propagation in SL.

### The Synthesis Problem

Given an arbitrary trust network (a DAG), the **synthesis problem** asks: construct the largest DSPG sub-graph `G'` of `G` that preserves as many trust edges as possible, while remaining reducible.

The key structural constraint is enforced through the **Node Nesting Level (NNL)**: for each node `v`, `NNL(v)` counts how many Parallel-Path Structures (PPSs) contain `v` as an interior node.

### Two Synthesis Criteria

A candidate branch (a path segment whose endpoints are already in `G'`) is admitted only if it satisfies the chosen criterion:

| Criterion | A branch `A → ... → B` is admitted when … |
|-----------|-------------------------------------------|
| **Original** (Jøsang) | `NNL(A) = NNL(B)` and every intermediate node `v` satisfies `NNL(v) ≥ NNL(A)` |
| **Optimized** (this work) | `|NNL(A) − NNL(B)| ≤ 1` and every intermediate node `v` satisfies `NNL(v) ≥ max(NNL(A), NNL(B))` |

The optimized criterion relaxes the equal-NNL requirement, admitting branches between nodes at adjacent nesting levels. This retains more edges from the original trust network while preserving DSPG validity.

---

## Repository Structure

```
.
├── sl_operators.py              # Subjective Logic: Opinion class, discounting, fusion
├── dspg.py                      # PPS/NNL primitives, DSPG check, synthesis, analysis
├── random_dag.py                # Random layered DAG generator with SL edge opinions
├── evaluation.py                # Full benchmark sweep and comparison plots
├── reproduce_chapter_eval.py    # Reproduction of Fig 4.9 (Section 4.7.1 sanity check)
├── dspg_benchmark_results.json  # Pre-computed benchmark results (re-plottable without re-running)
├── requirements.txt             # Python dependencies
├── CITATION.cff                 # Citation metadata
└── LICENSE                      # MIT License
```

---

## Installation

Python **≥ 3.10** is required.

```bash
pip install -r requirements.txt
```

Or install dependencies directly:

```bash
pip install networkx matplotlib numpy
```

---

## Reproducing the Results

```bash
# Full benchmark sweep (~2 minutes) and generate all plots
python evaluation.py

# Reproduce chapter Figure 4.9 (Section 4.7.1 sanity check)
python reproduce_chapter_eval.py
```

### Output Files

| File | Description |
|------|-------------|
| `dspg_benchmark_results.json` | Raw per-seed measurements (can be re-plotted without re-running the sweep) |
| `time_vs_size.{pdf,png}` | Total computation time (synthesis + analysis) vs. input graph size |
| `time_breakdown.{pdf,png}` | Synthesis time and analysis time plotted separately |
| `retention_vs_size.{pdf,png}` | Fraction of input edges retained in `G'` |
| `uncertainty_vs_size.{pdf,png}` | Mean uncertainty of the derived source-to-sink opinion |
| `uncertainty_diff_scatter.{pdf,png}` | Per-seed uncertainty difference (original − optimized) |
| `chapter_fig4_4_reproduction.{pdf,png}` | Reproduction of Fig 4.9 |

---

## Default Sweep Parameters

| Parameter | Value |
|-----------|-------|
| Internal layers | 2, 3, 4, 5, 6, 7 |
| Nodes per internal layer | up to 3 |
| Edge probability | 0.55 |
| Maximum layer skip | 2 |
| Random seeds per graph size | 20 |
| Fusion operator | Cumulative |

---

## Results (Cumulative Fusion)

| Layers | \|V\| | \|E\| | Retain (orig) | Retain (opt) | u (orig) | u (opt) | t orig (ms) | t opt (ms) |
|-------:|------:|------:|-------------:|-------------:|---------:|--------:|------------:|-----------:|
| 2 | 6.9 | 11.2 | 0.613 | **0.828** | 0.398 | 0.417 | 0.77 | 1.02 |
| 3 | 9.4 | 18.7 | 0.508 | **0.588** | 0.483 | 0.523 | 1.62 | 2.08 |
| 4 | 12.0 | 25.7 | 0.465 | **0.533** | 0.443 | 0.484 | 4.02 | 4.44 |
| 5 | 14.6 | 33.0 | 0.457 | **0.469** | 0.467 | 0.532 | 14.03 | 23.92 |
| 6 | 17.1 | 41.0 | 0.403 | **0.477** | 0.493 | 0.479 | 43.73 | 58.09 |
| 7 | 19.6 | 47.1 | 0.368 | **0.432** | 0.511 | 0.521 | 115.89 | 141.03 |

### Key Findings

1. **Information retention is consistently higher for the optimized criterion** across every graph size. The gap is largest for small graphs (~22 percentage points at 2 layers) and remains visible up to 47 edges (~6 percentage points at 7 layers).

2. **Computation cost is in the same order of magnitude.** The optimized criterion admits more edges, producing a slightly larger DSPG whose analysis costs a little more. The synthesis stage itself has nearly identical cost for both criteria — both evaluate the same candidate branches; the difference is only in the admission threshold.

3. **Final-opinion uncertainty depends on the retained structure.** On the chapter's Fig 4.4a example (one extra parallel edge, cumulative fusion), the optimized version strictly reduces uncertainty — verified independently by `reproduce_chapter_eval.py`. On random graphs, where the optimized criterion tends to keep additional *paths* (not just single edges), discount accumulation along longer paths can offset the cumulative-fusion gain. This is an honest empirical finding discussed in the chapter.

4. **DSPG validity is preserved by both methods on every seed.** `is_dspg(G')` returns `True` in 100% of runs, confirming the correctness of both synthesis procedures on randomly generated non-DSPG inputs.

---

## Module Reference

### `sl_operators.py` — Subjective Logic Primitives

| Name | Description |
|------|-------------|
| `Opinion(b, d, u, a)` | Immutable binomial opinion; enforces `b + d + u = 1` and `a ∈ [0, 1]` |
| `Opinion.P` | Projected probability `b + a·u` |
| `random_opinion(rng, base_rate)` | Sample a random valid binomial opinion |
| `discount_TE(ω_ref, ω_fun)` | Two-Edge trust discounting (referral → functional) |
| `discount_RE(ω_AB, ω_BC)` | Referral-Edge trust discounting (both referral opinions) |
| `discount_path(opinions)` | Discount a chain: RE for referral hops, TE for the final step |
| `fuse_cumulative(op1, op2)` | Cumulative belief fusion |
| `fuse_average(op1, op2)` | Averaging belief fusion |
| `fuse_many(opinions, operator)` | Fuse a list of opinions sequentially |

### `dspg.py` — DSPG Algorithms

| Name | Description |
|------|-------------|
| `find_pps(G)` | Enumerate all Parallel-Path Structures (A, B) in the DAG |
| `compute_nnl(G)` | Compute Node Nesting Level for every node |
| `is_dspg(G)` | Verify that `G` is a valid DSPG via iterative series reductions |
| `synthesize_dspg(G, source, sink, strategy)` | Build the DSPG sub-graph using `"original"` or `"optimized"` criterion |
| `analyse_dspg(G, source, sink, fusion)` | Reduce the DSPG to a single source-to-sink opinion |

### `random_dag.py` — Test Graph Generation

| Name | Description |
|------|-------------|
| `random_dag(n_layers, nodes_per_layer, edge_prob, max_skip, seed)` | Layered random DAG with random SL opinions on every edge |

### `evaluation.py` — Benchmarking

| Name | Description |
|------|-------------|
| `sweep(...)` | Run the full parameter sweep; returns a raw results dict |
| `plot_results(results, out_dir)` | Generate all comparison plots from a results dict |
| `save_results(results, path)` | Persist raw results as JSON for later re-plotting |
| `print_summary_table(results)` | Print a human-readable summary table to stdout |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@incollection{ouattara2025dspg,
  author    = {Ouattara, Isma\"{e}l},
  title     = {Optimization of {DSPG} Synthesis and Trust Network Analysis
               with Subjective Logic},
  booktitle = {[Book / Thesis title]},
  year      = {2025},
}
```

---

## License

This project is released under the [MIT License](LICENSE).
