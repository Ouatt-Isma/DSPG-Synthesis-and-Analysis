"""
Reproduce the evaluation of Section 4.7.1 of the chapter
(Fig 4.9 in the chapter / Fig 6 in the SL-DSPG paper).

We use the trust graph of Fig 4.4a:

         A --> B --> C
         |     |     |
         v     v     v
         D --> E <-- E
              |
              v
              E

with edges  A->B, A->D, B->C, B->E, C->E, D->E.

* Original criteria discard the extra edge  B -> E.
* Optimized criteria keep the extra edge.

We sweep the uncertainty of  ω^B_E  and plot the final-derived-opinion
uncertainty for both fusion operators (cumulative and averaging).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from sl_operators import Opinion, fuse_cumulative, fuse_average, discount_TE


# Default opinion (used for every edge except  B->E):
DEFAULT_OP = Opinion(1/3, 1/3, 1/3, 0.5)


def _final_opinion(omega_BE: Opinion, fusion: str = "cumulative") -> Opinion:
    """
    Compute  ω^E_A  using the chapter's Fig-4.4a graph.

    With the optimized criteria (edge B->E retained):
        ω^E_A = [ω^A_B  ⊗  ((ω^B_C ⊗ ω^C_E) ⊕ ω^B_E)]
                 ⊕  [ω^A_D ⊗ ω^D_E]
    """
    fuse = fuse_cumulative if fusion == "cumulative" else fuse_average

    # discount along  B -> C -> E
    BCE = discount_TE(DEFAULT_OP, DEFAULT_OP)
    # parallel fusion of  BCE  and  B->E
    inner = fuse(BCE, omega_BE)
    # discount along  A -> B -> "inner"
    left = discount_TE(DEFAULT_OP, inner)
    # discount along  A -> D -> E
    right = discount_TE(DEFAULT_OP, DEFAULT_OP)
    # final fusion
    return fuse(left, right)


def _final_opinion_no_BE(fusion: str = "cumulative") -> Opinion:
    """Same graph but with ω^B_E discarded (original criteria)."""
    fuse = fuse_cumulative if fusion == "cumulative" else fuse_average

    BCE = discount_TE(DEFAULT_OP, DEFAULT_OP)
    left = discount_TE(DEFAULT_OP, BCE)
    right = discount_TE(DEFAULT_OP, DEFAULT_OP)
    return fuse(left, right)


def reproduce_figure(out_path: str):
    """Plot uncertainty of the derived opinion as u(ω^B_E) varies."""
    us = np.linspace(0.0, 1.0, 51)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, fusion, title in [
        (axes[0], "cumulative",
         r"Impact of edge $B\to E$ using cumulative fusion"),
        (axes[1], "average",
         r"Impact of edge $B\to E$ using averaging fusion"),
    ]:
        # uncertainty of final opinion when extra edge is included
        with_edge = []
        for u in us:
            b = (1 - u) / 2
            omega_BE = Opinion(b, b, u, 0.5)
            with_edge.append(_final_opinion(omega_BE, fusion=fusion).u)

        # uncertainty when extra edge is dropped (constant)
        without_edge = _final_opinion_no_BE(fusion=fusion).u

        ax.plot(us, with_edge,    label="with $B \\to E$ (optimized)", color="#1f77b4", lw=2)
        ax.axhline(without_edge,  label="without $B \\to E$ (original)", color="#d62728", lw=2)
        ax.set_xlabel(r"uncertainty of $\omega^B_E$")
        ax.set_ylabel(r"uncertainty of derived $\omega^E_A$")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".png", dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}.png / .pdf")


if __name__ == "__main__":
    reproduce_figure("./chapter_fig4_4_reproduction")
