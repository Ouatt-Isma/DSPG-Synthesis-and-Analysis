"""
Evaluation: original criteria vs. optimized criteria for DSPG synthesis.

Metrics:
  * synthesis time
  * analysis time
  * total time (synthesis + analysis)
  * information retention  : |E(G')| / |E(G)|
  * final opinion uncertainty difference

We sweep graph size (number of nodes / number of edges in the input DAG) and
average across multiple random seeds.
"""

from __future__ import annotations
import time
import statistics
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from random_dag import random_dag
from dspg import synthesize_dspg, analyse_dspg, is_dspg


# --------------------------------------------------------------------------
# Single-run measurement
# --------------------------------------------------------------------------

def measure_one(G: nx.DiGraph, source, sink, strategy: str,
                fusion: str = "cumulative") -> dict:
    """Run synthesis + analysis once and record metrics."""
    t0 = time.perf_counter()
    Gp = synthesize_dspg(G, source, sink, strategy=strategy)
    t1 = time.perf_counter()
    if Gp.number_of_edges() == 0:
        return {
            "synthesis_time": t1 - t0,
            "analysis_time": float("nan"),
            "total_time": float("nan"),
            "retained_edges": 0,
            "retention_ratio": 0.0,
            "final_uncertainty": float("nan"),
            "final_belief": float("nan"),
            "final_disbelief": float("nan"),
            "is_dspg": False,
        }
    t2 = time.perf_counter()
    op = analyse_dspg(Gp, source, sink, fusion=fusion)
    t3 = time.perf_counter()
    return {
        "synthesis_time": t1 - t0,
        "analysis_time": t3 - t2,
        "total_time": (t1 - t0) + (t3 - t2),
        "retained_edges": Gp.number_of_edges(),
        "retention_ratio": Gp.number_of_edges() / G.number_of_edges(),
        "final_uncertainty": op.u,
        "final_belief": op.b,
        "final_disbelief": op.d,
        "is_dspg": is_dspg(Gp),
    }


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

def sweep(layer_counts=(2, 3, 4, 5, 6),
          nodes_per_layer=3,
          edge_prob=0.5,
          max_skip=2,
          n_seeds=20,
          fusion="cumulative",
          base_seed=42) -> dict:
    """Run the full benchmark sweep."""
    results = {
        "layer_counts": list(layer_counts),
        "n_seeds": n_seeds,
        "edge_prob": edge_prob,
        "nodes_per_layer": nodes_per_layer,
        "max_skip": max_skip,
        "fusion": fusion,
        "by_size": [],
    }

    for nL in layer_counts:
        bucket = {
            "n_layers": nL,
            "input_n_nodes": [],
            "input_n_edges": [],
            "input_was_dspg": [],
            "original": [],
            "optimized": [],
        }
        for s in range(n_seeds):
            G, src, snk = random_dag(
                n_layers=nL, nodes_per_layer=nodes_per_layer,
                edge_prob=edge_prob, max_skip=max_skip,
                seed=base_seed + s,
            )
            bucket["input_n_nodes"].append(G.number_of_nodes())
            bucket["input_n_edges"].append(G.number_of_edges())
            bucket["input_was_dspg"].append(is_dspg(G))
            bucket["original"].append(
                measure_one(G, src, snk, "original", fusion=fusion))
            bucket["optimized"].append(
                measure_one(G, src, snk, "optimized", fusion=fusion))
        results["by_size"].append(bucket)
        # print quick progress
        n_nodes = statistics.mean(bucket["input_n_nodes"])
        n_edges = statistics.mean(bucket["input_n_edges"])
        ret_o = statistics.mean(r["retention_ratio"] for r in bucket["original"])
        ret_n = statistics.mean(r["retention_ratio"] for r in bucket["optimized"])
        t_o = statistics.mean(r["total_time"] for r in bucket["original"])
        t_n = statistics.mean(r["total_time"] for r in bucket["optimized"])
        print(f"layers={nL:2d}  |V|≈{n_nodes:4.1f}  |E|≈{n_edges:4.1f}  "
              f"retention orig={ret_o:.3f} opt={ret_n:.3f}  "
              f"time orig={t_o*1e3:.2f}ms opt={t_n*1e3:.2f}ms")
    return results


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _summary(bucket, key, method):
    vals = [r[key] for r in bucket[method] if not _isnan(r[key])]
    if not vals:
        return float("nan"), float("nan")
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _isnan(x):
    try:
        return x != x
    except TypeError:
        return False


def plot_results(results: dict, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes_x = [statistics.mean(b["input_n_edges"]) for b in results["by_size"]]

    # ------------------------------------------------------------------
    # 1) Computation time vs graph size
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color in [("original", "#1f77b4"), ("optimized", "#d62728")]:
        means, stds = [], []
        for b in results["by_size"]:
            m, s = _summary(b, "total_time", method)
            means.append(m * 1e3)
            stds.append(s * 1e3)
        ax.errorbar(sizes_x, means, yerr=stds, marker="o", capsize=3,
                    color=color, label=method)
    ax.set_xlabel("input graph size (avg. number of edges)")
    ax.set_ylabel("synthesis + analysis time [ms]")
    ax.set_title("Computation time: original vs. optimized")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_vs_size.pdf")
    fig.savefig(out_dir / "time_vs_size.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2) Synthesis vs analysis time, side-by-side
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, key, title in [
        (axes[0], "synthesis_time", "DSPG Synthesis time"),
        (axes[1], "analysis_time", "DSPG Analysis time"),
    ]:
        for method, color in [("original", "#1f77b4"), ("optimized", "#d62728")]:
            means, stds = [], []
            for b in results["by_size"]:
                m, s = _summary(b, key, method)
                means.append(m * 1e3)
                stds.append(s * 1e3)
            ax.errorbar(sizes_x, means, yerr=stds, marker="o", capsize=3,
                        color=color, label=method)
        ax.set_xlabel("input graph size (avg. number of edges)")
        ax.set_ylabel("time [ms]")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_breakdown.pdf")
    fig.savefig(out_dir / "time_breakdown.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3) Information retention (edges kept / total edges)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color in [("original", "#1f77b4"), ("optimized", "#d62728")]:
        means, stds = [], []
        for b in results["by_size"]:
            m, s = _summary(b, "retention_ratio", method)
            means.append(m)
            stds.append(s)
        ax.errorbar(sizes_x, means, yerr=stds, marker="o", capsize=3,
                    color=color, label=method)
    ax.set_xlabel("input graph size (avg. number of edges)")
    ax.set_ylabel("retained edge ratio  |E(G')| / |E(G)|")
    ax.set_title("Information retention: original vs. optimized")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "retention_vs_size.pdf")
    fig.savefig(out_dir / "retention_vs_size.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4) Final-opinion uncertainty
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for method, color in [("original", "#1f77b4"), ("optimized", "#d62728")]:
        means, stds = [], []
        for b in results["by_size"]:
            m, s = _summary(b, "final_uncertainty", method)
            means.append(m)
            stds.append(s)
        ax.errorbar(sizes_x, means, yerr=stds, marker="o", capsize=3,
                    color=color, label=method)
    ax.set_xlabel("input graph size (avg. number of edges)")
    ax.set_ylabel("uncertainty of final opinion  $u_{X}^{A}$")
    ax.set_title("Derived-opinion uncertainty: original vs. optimized")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_vs_size.pdf")
    fig.savefig(out_dir / "uncertainty_vs_size.png", dpi=160)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5) Per-seed scatter of uncertainty difference
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    diffs_all = []
    sizes_per_diff = []
    for b in results["by_size"]:
        sz = statistics.mean(b["input_n_edges"])
        for ro, rn in zip(b["original"], b["optimized"]):
            uo = ro["final_uncertainty"]
            un = rn["final_uncertainty"]
            if not (_isnan(uo) or _isnan(un)):
                diffs_all.append(uo - un)
                sizes_per_diff.append(sz)
    ax.scatter(sizes_per_diff, diffs_all, alpha=0.5, s=18, color="#2ca02c")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.set_xlabel("input graph size (avg. number of edges)")
    ax.set_ylabel(r"$u_{\mathrm{original}} - u_{\mathrm{optimized}}$")
    ax.set_title("Per-seed: uncertainty reduction from optimized synthesis")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "uncertainty_diff_scatter.pdf")
    fig.savefig(out_dir / "uncertainty_diff_scatter.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def save_results(results: dict, path: Path):
    """JSON dump of the raw results so they can be re-plotted later."""
    Path(path).write_text(json.dumps(results, indent=2))


def print_summary_table(results: dict):
    print()
    print(f"{'layers':>6} {'|V|':>5} {'|E|':>5} "
          f"{'orig retain':>11} {'opt retain':>10} "
          f"{'orig u':>8} {'opt u':>8} "
          f"{'orig t (ms)':>11} {'opt t (ms)':>10}")
    print("-" * 90)
    for b in results["by_size"]:
        n = statistics.mean(b["input_n_nodes"])
        e = statistics.mean(b["input_n_edges"])
        ro, _ = _summary(b, "retention_ratio", "original")
        rn, _ = _summary(b, "retention_ratio", "optimized")
        uo, _ = _summary(b, "final_uncertainty", "original")
        un, _ = _summary(b, "final_uncertainty", "optimized")
        to, _ = _summary(b, "total_time", "original")
        tn, _ = _summary(b, "total_time", "optimized")
        print(f"{b['n_layers']:>6} {n:>5.1f} {e:>5.1f} "
              f"{ro:>11.3f} {rn:>10.3f} "
              f"{uo:>8.3f} {un:>8.3f} "
              f"{to*1e3:>11.2f} {tn*1e3:>10.2f}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running DSPG synthesis + analysis benchmark...")
    print("(non-DSPG random DAGs, original vs. optimized criteria)\n")

    results = sweep(
        layer_counts=(2, 3, 4, 5, 6, 7),
        nodes_per_layer=3,
        edge_prob=0.55,
        max_skip=2,
        n_seeds=20,
        fusion="cumulative",
        base_seed=42,
    )
    print_summary_table(results)
    save_results(results, "./dspg_benchmark_results.json")
    plot_results(results, ".")
    print("\nPlots written to /mnt/user-data/outputs/")
