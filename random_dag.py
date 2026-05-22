"""
Random DAG generator that produces graphs likely to require DSPG synthesis.

Strategy: a layered DAG with a single source, a single sink, and a configurable
number of internal layers.  Each internal node connects to a random number
of nodes in *strictly later* layers (not necessarily the immediately next
layer), which gives plenty of overlapping paths and therefore plenty of
non-DSPG structure.
"""

import random
import networkx as nx
from sl_operators import Opinion, random_opinion


def random_dag(n_layers: int = 4,
               nodes_per_layer: int = 3,
               edge_prob: float = 0.5,
               max_skip: int = 2,
               base_rate: float = 0.5,
               seed: int | None = None) -> tuple[nx.DiGraph, str, str]:
    """
    Build a random DAG.

    Parameters
    ----------
    n_layers          : number of intermediate layers
    nodes_per_layer   : (max) nodes in each intermediate layer
    edge_prob         : probability that any candidate forward edge exists
    max_skip          : maximum number of layers an edge may skip
                        (1 = only adjacent layers; 2 = adjacent or one-skip)
    base_rate         : base rate to use for every opinion
    seed              : RNG seed

    Returns
    -------
    G       : the DAG (nx.DiGraph) with edge attribute 'opinion'
    source  : the unique source node
    sink    : the unique sink node
    """
    rng = random.Random(seed)

    G = nx.DiGraph()
    layers = []
    # build the layers
    layers.append(["S"])
    for li in range(n_layers):
        # each layer has between 2 and nodes_per_layer nodes
        size = rng.randint(2, max(2, nodes_per_layer))
        layers.append([f"L{li}_{i}" for i in range(size)])
    layers.append(["T"])

    for layer in layers:
        for n in layer:
            G.add_node(n)

    # add edges
    for li, layer in enumerate(layers[:-1]):
        for u in layer:
            # choose how many forward layers this node may reach
            for skip in range(1, max_skip + 1):
                tgt_idx = li + skip
                if tgt_idx >= len(layers):
                    break
                for v in layers[tgt_idx]:
                    if rng.random() < edge_prob:
                        G.add_edge(u, v, opinion=random_opinion(rng, base_rate))

    # ensure connectivity: every node must lie on some path source -> sink
    source, sink = layers[0][0], layers[-1][0]

    # ensure each layer-i node has at least one incoming and one outgoing edge
    # within the DAG structure
    for li in range(1, len(layers)):
        for u in layers[li]:
            if G.in_degree(u) == 0:
                # connect from a random node in an earlier layer
                src_layer = layers[rng.randint(max(0, li - max_skip), li - 1)]
                p = rng.choice(src_layer)
                G.add_edge(p, u, opinion=random_opinion(rng, base_rate))
    for li in range(len(layers) - 1):
        for u in layers[li]:
            if G.out_degree(u) == 0:
                tgt_layer = layers[rng.randint(li + 1,
                                               min(len(layers) - 1,
                                                   li + max_skip))]
                p = rng.choice(tgt_layer)
                G.add_edge(u, p, opinion=random_opinion(rng, base_rate))

    # remove any node not on any source->sink path (rare with our procedure)
    reachable_from_source = nx.descendants(G, source) | {source}
    can_reach_sink = nx.ancestors(G, sink) | {sink}
    keep = reachable_from_source & can_reach_sink
    drop = set(G.nodes()) - keep
    G.remove_nodes_from(drop)

    return G, source, sink
