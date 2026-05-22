"""
DSPG synthesis and analysis with two synthesis strategies:

  * "original" : Jøsang's original criteria
        1. path A->B exists in current graph
        2. NNL(A) == NNL(B)
        3. every intermediate node C has NNL(C) >= NNL(A)

  * "optimized" : the criteria proposed in the chapter
        1. path A->B exists in current graph
        2. |NNL(A) - NNL(B)| <= 1
        3. every intermediate node C has NNL(C) >= max(NNL(A), NNL(B))

Both strategies share the same incremental skeleton:

    1. Enumerate all source-to-sink paths in the input DAG.
    2. Initialize G' with one valid path (we pick the shortest).
    3. For every remaining path, walk along it edge-by-edge and try to
       add each edge that is not yet in G'.  An edge is added iff it
       satisfies the chosen criteria; otherwise it is dropped.

NNL is recomputed from G' on demand using a brute-force PPS enumeration.
The graphs we use in the evaluation are small enough for this to be cheap;
the dominant cost when comparing the two methods is the *number* of NNL
recomputations and the *number* of admitted edges, both of which differ
between the two strategies.

DSPG analysis is done by repeated PPS reduction.  For each PPS at the
maximum nesting level, we replace its sub-network with a single edge
whose opinion is the cumulative-fusion of all path opinions inside the
PPS (each path opinion being the discount-chain of its edges).
"""

from __future__ import annotations
import networkx as nx
from itertools import combinations
from typing import Iterable

from sl_operators import (
    Opinion,
    discount_path,
    discount_TE,
    fuse_cumulative,
    fuse_average,
    fuse_many,
)


# ------------------------------------------------------------------
# PPS / NNL primitives
# ------------------------------------------------------------------

def all_simple_paths(G: nx.DiGraph, src, tgt) -> list[list]:
    """Return all simple paths from src to tgt as node lists."""
    if src not in G or tgt not in G:
        return []
    return list(nx.all_simple_paths(G, src, tgt))


def find_pps(G: nx.DiGraph) -> list[tuple]:
    """
    Enumerate every PPS in the DAG G.

    A pair (A, B) is a PPS iff there exist at least two distinct simple
    paths from A to B.  This implies out-degree(A) >= 2 and
    in-degree(B) >= 2 along those paths.
    """
    pps_list = []
    nodes = list(G.nodes())
    for A in nodes:
        if G.out_degree(A) < 2:
            continue
        # we only need to look at descendants reachable through >=2 edges
        descendants = nx.descendants(G, A)
        for B in descendants:
            if G.in_degree(B) < 2:
                continue
            # count simple paths until we know there are >=2
            cnt = 0
            for _ in nx.all_simple_paths(G, A, B):
                cnt += 1
                if cnt >= 2:
                    break
            if cnt >= 2:
                pps_list.append((A, B))
    return pps_list


def pps_intermediate_nodes(G: nx.DiGraph, A, B) -> set:
    """
    Return the set of nodes that are intermediate in the PPS (A, B):
    every node that appears on some simple A->B path other than A and B.
    """
    inter = set()
    for path in nx.all_simple_paths(G, A, B):
        for n in path[1:-1]:
            inter.add(n)
    return inter


def compute_nnl(G: nx.DiGraph) -> dict:
    """Node Nesting Level: NNL(v) = number of PPSs of which v is intermediate."""
    nnl = {n: 0 for n in G.nodes()}
    for A, B in find_pps(G):
        for v in pps_intermediate_nodes(G, A, B):
            nnl[v] += 1
    return nnl


# ------------------------------------------------------------------
# DSPG check
# ------------------------------------------------------------------

def is_dspg(G: nx.DiGraph) -> bool:
    """
    A directed acyclic graph is a DSPG iff it can be reduced to a single
    edge by series and parallel reductions.  We verify this constructively.
    """
    H = G.copy()
    if not nx.is_directed_acyclic_graph(H):
        return False
    sources = [n for n in H.nodes() if H.in_degree(n) == 0]
    sinks = [n for n in H.nodes() if H.out_degree(n) == 0]
    if len(sources) != 1 or len(sinks) != 1:
        return False
    source, sink = sources[0], sinks[0]

    # repeated series + parallel reductions until nothing changes
    changed = True
    while changed:
        changed = False
        # parallel reduction: collapse multi-edges (here each edge is unique;
        # we only get parallel after series reductions).  Simulate using
        # a multigraph-aware structure.  Since we use a simple DiGraph,
        # parallel edges manifest as multiple paths of length 1; combine
        # them into a single edge by removing duplicates (already done).
        # series reduction
        for n in list(H.nodes()):
            if n == source or n == sink:
                continue
            if H.in_degree(n) == 1 and H.out_degree(n) == 1:
                pred = next(iter(H.predecessors(n)))
                succ = next(iter(H.successors(n)))
                if pred == succ:
                    return False
                H.remove_node(n)
                if not H.has_edge(pred, succ):
                    H.add_edge(pred, succ)
                changed = True
                break
        if changed:
            continue
        # parallel reduction: if two nodes have multiple paths of length 1
        # between them - in a simple DiGraph that can't happen, but if a
        # series reduction tried to add an existing edge, we already merged.
        # So instead look for "pure parallel" pairs (A,B) where every path
        # from A to B is a direct edge - already collapsed.
        # Try detecting parallel structure that can be merged:
        # if a node n has predecessors p1, p2 with both also being only
        # connected by the path through n, we still rely on series.
    # success iff only the single edge source->sink is left
    return H.number_of_nodes() == 2 and H.number_of_edges() == 1 \
        and H.has_edge(source, sink)


# ------------------------------------------------------------------
# Synthesis criteria
# ------------------------------------------------------------------

def _criterion_original(G_prime: nx.DiGraph, A, B) -> bool:
    """Jøsang's original criteria."""
    if A not in G_prime or B not in G_prime:
        return False
    if not nx.has_path(G_prime, A, B):
        return False
    nnl = compute_nnl(G_prime)
    if nnl[A] != nnl[B]:
        return False
    threshold = nnl[A]
    # every intermediate node on every simple A->B path must have NNL >= threshold
    for path in nx.all_simple_paths(G_prime, A, B):
        for v in path[1:-1]:
            if nnl[v] < threshold:
                return False
    return True


def _criterion_optimized(G_prime: nx.DiGraph, A, B) -> bool:
    """Optimized criteria from the chapter."""
    if A not in G_prime or B not in G_prime:
        return False
    if not nx.has_path(G_prime, A, B):
        return False
    nnl = compute_nnl(G_prime)
    if abs(nnl[A] - nnl[B]) > 1:
        return False
    threshold = max(nnl[A], nnl[B])
    for path in nx.all_simple_paths(G_prime, A, B):
        for v in path[1:-1]:
            if nnl[v] < threshold:
                return False
    return True


_CRITERIA = {
    "original": _criterion_original,
    "optimized": _criterion_optimized,
}


# ------------------------------------------------------------------
# Synthesis
# ------------------------------------------------------------------

def _path_uncertainty(G: nx.DiGraph, path: list) -> float:
    """A simple priority key: uncertainty accumulated along the path
    (using TE discounting where applicable).  Falls back to length."""
    if len(path) < 2:
        return 0.0
    op = G[path[0]][path[1]]["opinion"]
    for i in range(1, len(path) - 1):
        op = discount_TE(op, G[path[i]][path[i + 1]]["opinion"])
    return op.u


def _branches_of_path(path: list, G_prime: nx.DiGraph) -> list[tuple]:
    """
    Decompose a path into maximal branches.

    A "branch" is a maximal contiguous sub-path P[i..j] such that:
      * P[i] and P[j] are already in G_prime,
      * none of the edges (P[k], P[k+1]) for i <= k < j are in G_prime.

    If every edge along the path is already in G_prime, no branches are
    returned.  We return the list of (i, j) index pairs.
    """
    n = len(path)
    branches = []
    i = 0
    while i < n - 1:
        # advance i to the first index where path[i] is in G_prime
        while i < n - 1 and path[i] not in G_prime:
            i += 1
        if i >= n - 1:
            break
        # if the edge (path[i], path[i+1]) is already in G_prime, skip
        if G_prime.has_edge(path[i], path[i + 1]):
            i += 1
            continue
        # walk forward until we reach another node in G_prime
        j = i + 1
        while j < n and (path[j] not in G_prime
                         or not _segment_fully_external(
                             path, i, j, G_prime)):
            # we keep walking while we haven't hit an existing node, or
            # while none of the segment's edges are already in G_prime
            # (the second condition is automatically true here).
            j += 1
            # safety bail if we run past the end
            if j == n:
                break
        if j < n:
            branches.append((i, j))
            i = j  # continue from this in-G' node
        else:
            break
    return branches


def _segment_fully_external(path, i, j, G_prime):
    """Helper: are all edges in path[i..j] not in G_prime?"""
    for k in range(i, j):
        if G_prime.has_edge(path[k], path[k + 1]):
            return False
    return True


def synthesize_dspg(G: nx.DiGraph, source, sink, strategy: str = "optimized",
                    sort_by: str = "length") -> nx.DiGraph:
    """
    Build a DSPG sub-graph G' of G that contains source and sink.

    Algorithm (matching Algorithm 1 of the chapter):
      1. Enumerate all simple source -> sink paths in G.
      2. Sort paths (shortest first by default).
      3. Initialise G' with the first path entirely.
      4. For every remaining path: find each maximal branch (segment whose
         endpoints are in G' but whose interior edges are not), in left-to-
         right order, and add the branch to G' if the chosen criterion
         accepts it.
    """
    check = _CRITERIA[strategy]
    paths = list(nx.all_simple_paths(G, source, sink))
    if not paths:
        return nx.DiGraph()

    if sort_by == "uncertainty":
        paths.sort(key=lambda p: _path_uncertainty(G, p))
    else:
        paths.sort(key=len)

    G_prime = nx.DiGraph()
    first = paths[0]
    for u, v in zip(first[:-1], first[1:]):
        G_prime.add_edge(u, v, opinion=G[u][v]["opinion"])

    for path in paths[1:]:
        # repeatedly find the *first* branch (whose endpoints are already
        # in G_prime) and try to add it.  After adding, the formerly-
        # interior nodes of the branch become part of G_prime, which may
        # uncover further branches in the same path.
        while True:
            branches = _branches_of_path(path, G_prime)
            if not branches:
                break
            i, j = branches[0]
            A, B = path[i], path[j]
            # candidate branch is path[i..j]; both endpoints in G_prime
            if check(G_prime, A, B):
                # admissible: add all interior edges of the branch
                for k in range(i, j):
                    G_prime.add_edge(
                        path[k], path[k + 1],
                        opinion=G[path[k]][path[k + 1]]["opinion"],
                    )
                # cycle safety check (should not happen for DAGs but guard)
                if not nx.is_directed_acyclic_graph(G_prime):
                    for k in range(i, j):
                        if G_prime.has_edge(path[k], path[k + 1]):
                            G_prime.remove_edge(path[k], path[k + 1])
                    break
            else:
                break  # rejection: skip the rest of this path
    return G_prime


# ------------------------------------------------------------------
# DSPG analysis (reduction)
# ------------------------------------------------------------------

def _pick_max_nl_pps(G: nx.DiGraph) -> tuple | None:
    """Return the PPS with maximal nesting level (min over its edges of NL)."""
    pps_list = find_pps(G)
    if not pps_list:
        return None
    # NL of edge = number of PPSs containing that edge
    edge_to_pps = {(u, v): set() for u, v in G.edges()}
    pps_edges_map = {}
    for A, B in pps_list:
        es = set()
        for path in nx.all_simple_paths(G, A, B):
            for u, v in zip(path[:-1], path[1:]):
                es.add((u, v))
                edge_to_pps[(u, v)].add((A, B))
        pps_edges_map[(A, B)] = es
    edge_nl = {e: len(s) for e, s in edge_to_pps.items()}

    def pps_nl(p):
        return min(edge_nl[e] for e in pps_edges_map[p])

    # most nested = largest NL
    return max(pps_list, key=pps_nl)


def analyse_dspg(G: nx.DiGraph, source, sink,
                 fusion: str = "cumulative") -> Opinion:
    """
    Reduce the DSPG to a single edge and return the source->sink opinion.
    Uses iterative PPS reduction.  Each reduction step:
      - pick the PPS with maximum nesting level
      - for each simple path inside the PPS, discount its opinions
      - fuse the resulting edge-opinions into one
      - replace the PPS sub-graph with a single edge carrying that opinion
    Finally a series reduction of the remaining linear chain yields the
    answer.
    """
    fuse = fuse_cumulative if fusion == "cumulative" else fuse_average
    H = G.copy()
    if source not in H or sink not in H or not nx.has_path(H, source, sink):
        raise ValueError("DSPG analysis: no path from source to sink")

    # iterate PPS reductions
    safety_steps = 10_000
    while True:
        safety_steps -= 1
        if safety_steps == 0:
            raise RuntimeError("DSPG analysis did not terminate")

        pps = _pick_max_nl_pps(H)
        if pps is None:
            break
        A, B = pps
        # collect every simple A->B path; discount each into a single
        # edge-opinion; then fuse them all into one
        path_opinions = []
        # we also need to know which nodes/edges belong to this PPS so we
        # can remove them
        pps_nodes = set([A, B])
        for path in nx.all_simple_paths(H, A, B):
            ops = [H[u][v]["opinion"] for u, v in zip(path[:-1], path[1:])]
            # PPS reduction inside the chapter framework:
            # treat the chain as discount of multiple opinions
            path_op = _reduce_chain(ops)
            path_opinions.append(path_op)
            for n in path[1:-1]:
                pps_nodes.add(n)
        if not path_opinions:
            break
        merged = fuse_many(path_opinions, operator=fusion)

        # remove all internal nodes (and their incident edges) of this PPS
        internals = pps_nodes - {A, B}
        for n in internals:
            H.remove_node(n)
        # remove any direct A->B edge that may have existed
        if H.has_edge(A, B):
            H.remove_edge(A, B)
        H.add_edge(A, B, opinion=merged)

    # what remains should be a single linear chain from source to sink
    paths = list(nx.all_simple_paths(H, source, sink))
    if len(paths) != 1:
        # fallback: fuse remaining paths
        fused = fuse_many(
            [_reduce_chain([H[u][v]["opinion"] for u, v in zip(p[:-1], p[1:])])
             for p in paths],
            operator=fusion,
        )
        return fused
    chain = paths[0]
    ops = [H[u][v]["opinion"] for u, v in zip(chain[:-1], chain[1:])]
    return _reduce_chain(ops)


def _reduce_chain(ops: list[Opinion]) -> Opinion:
    """Discount a list of opinions along a chain.  We default to TE-only
    chains because random graphs do not distinguish referral vs functional
    edges; this matches the classical behaviour."""
    if len(ops) == 1:
        return ops[0]
    # left-to-right TE
    cur = ops[0]
    for nxt in ops[1:]:
        cur = discount_TE(cur, nxt)
    return cur
