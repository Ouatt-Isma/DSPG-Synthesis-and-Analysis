# DSPG Synthesis & Analysis — Empirical Evaluation

This is the missing evaluation code for Section 4.7.2 of the chapter
("Optimization of DSPG Synthesis and Trust Network Analysis with
Subjective Logic"). It implements the original Jøsang criteria, the
optimized criteria from the chapter, and benchmarks both on randomly
generated DAGs.

## Files

| file                          | purpose                                                                     |
|-------------------------------|-----------------------------------------------------------------------------|
| `sl_operators.py`             | Binomial-opinion class + TE / RE discounting + cumulative / averaging fusion |
| `dspg.py`                     | PPS / NNL / DSPG check, **synthesis** (both criteria), **analysis** (PPS reduction) |
| `random_dag.py`               | Layered random-DAG generator with single source / sink and random opinions  |
| `evaluation.py`               | Full sweep across graph sizes; produces all benchmark plots                  |
| `reproduce_chapter_eval.py`   | Re-derives the chapter's Fig 4.9 (Section 4.7.1) for sanity checking         |

## Reproducing the results

```bash
pip install networkx matplotlib
python evaluation.py            # full benchmark + plots
python reproduce_chapter_eval.py  # chapter Fig 4.9 reproduction
```

`evaluation.py` writes the following into `/mnt/user-data/outputs/`:

* `dspg_benchmark_results.json` — raw per-seed numbers (re-plottable)
* `time_vs_size.{pdf,png}` — total time (synthesis + analysis) vs |E|
* `time_breakdown.{pdf,png}` — synthesis time and analysis time separately
* `retention_vs_size.{pdf,png}` — fraction of input edges retained
* `uncertainty_vs_size.{pdf,png}` — mean uncertainty of the derived opinion
* `uncertainty_diff_scatter.{pdf,png}` — per-seed uncertainty difference

## Default sweep

* `n_layers ∈ {2, 3, 4, 5, 6, 7}` (1 source + N internal layers + 1 sink)
* up to 3 nodes per internal layer
* edge probability 0.55, max layer-skip 2
* 20 random seeds per size
* cumulative fusion in the analysis stage

## Numbers (cumulative fusion)

| layers | \|V\| | \|E\| | retain orig | retain opt | u orig | u opt | t orig (ms) | t opt (ms) |
|-------:|-----:|-----:|-----------:|----------:|------:|------:|-----------:|----------:|
| 2 | 6.9 | 11.2 | 0.613 | **0.828** | 0.398 | 0.417 | 0.77 | 1.02 |
| 3 | 9.4 | 18.7 | 0.508 | **0.588** | 0.483 | 0.523 | 1.62 | 2.08 |
| 4 | 12.0 | 25.7 | 0.465 | **0.533** | 0.443 | 0.484 | 4.02 | 4.44 |
| 5 | 14.6 | 33.0 | 0.457 | **0.469** | 0.467 | 0.532 | 14.03 | 23.92 |
| 6 | 17.1 | 41.0 | 0.403 | **0.477** | 0.493 | 0.479 | 43.73 | 58.09 |
| 7 | 19.6 | 47.1 | 0.368 | **0.432** | 0.511 | 0.521 | 115.89 | 141.03 |

## Take-aways for the chapter text

1. **Information retention is consistently higher for the optimized
   criteria across every graph size.** The gap is largest for small
   graphs (≈ 22 percentage points) and remains visible up to 47 edges
   (≈ 6 percentage points).

2. **Computation cost is in the same order of magnitude.** The optimized
   criteria admit more edges, so the resulting DSPG is larger and its
   subsequent analysis is slightly more expensive (right panel of
   `time_breakdown.png`). The synthesis stage itself has nearly
   identical cost — both criteria evaluate the same number of
   candidates; the difference is only in the threshold.

3. **Final-opinion uncertainty depends on the structure that ends up
   being retained.** On the chapter's Fig-4.4 example (single extra
   parallel edge added, cumulative fusion) the optimized version
   strictly reduces uncertainty — verified independently by
   `reproduce_chapter_eval.py`. On random graphs, where the optimized
   criteria tend to keep additional **paths** (not just additional
   single edges), the discount-blow-up along the longer path can offset
   the cumulative-fusion gain. This is an honest finding worth
   discussing in the chapter — the *information* is retained, but its
   downstream effect on uncertainty is structure-dependent.

4. **DSPG validity** is preserved by both methods on every seed
   (`is_dspg(G')` returns `True` in 100 % of runs), confirming the
   correctness of both syntheses on randomly generated non-DSPG inputs.
