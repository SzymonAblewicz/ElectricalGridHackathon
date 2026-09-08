# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib"]
# ///
"""Node betweenness centrality of one PyPSA case, on the 1/capacity metric.

Betweenness counts, for every node, the share of shortest paths between other
pairs of nodes that run through it. On a grid it reads as "how much of the
network's through-traffic has no choice but to pass here" - a node with a high
score is one whose loss splits the most routes.

The graph and its weights come from `get_weighted_graph.py` rather than being
rebuilt here, so there is one definition of a corridor and one definition of its
weight. The metric is always the reciprocal one: a shortest path needs its
weight to read as a *distance*, and `s_nom` read directly says the opposite -
under a capacity weighting the 3,846 MVA backbone would be 116 times more
expensive to cross than a 33 MVA spur, so paths would route around the
transmission system and the corridors that matter would score lowest. Hence
`reciprocal()`, on the summed corridor capacity: a double-circuit corridor is
half as far as a single one.

Scores are Brandes' algorithm, computed exactly rather than sampled - these
graphs run to a few thousand nodes, which is small enough not to need pivots.
Two conventions follow networkx so the two can be compared directly:
`betweenness_raw` is halved (an undirected sweep counts each pair from both
ends) and `betweenness` is divided by (n-1)(n-2), which puts it on 0-1 and makes
cases of different sizes comparable.

One caveat worth knowing. Ties between equal-cost paths are decided by exact
float equality, which is what Brandes assumes and what networkx does: two
genuinely equal-length but distinct routes are counted as tied only if their
floating-point sums come out bitwise equal. With real-valued 1/capacity weights
an exact tie is rare, but a perfectly symmetric pair of corridors is exactly the
place it could happen, and where it does the paths through one of the two are
dropped rather than split.

To run it: set CASE and GENERATORS in the "what to run" block below, then run
the file. It reads the PyPSA case straight from the kit, so it does not matter
whether get_weighted_graph.py has been run first.
"""

from __future__ import annotations

import time
from heapq import heappop, heappush
from math import inf

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph

import get_weighted_graph as gwg


# ---- what to run ---- #
#
# Same cases as get_weighted_graph.py: directory names under
# grid_TF_Wind/data/pypsa, of the shape
# TYTFS2024_{WP,SV}{2024,2033}_V35_{transmission,full}.
#
# The metric is always the reciprocal one - see the note above on why a
# capacity weighting would rank the backbone lowest - so there is no switch
# for it here.

CASE = "TYTFS2024_WP2024_V35_transmission"
GENERATORS = True    # include the generators as nodes of their own


# ---- the algorithm ---- #


def betweenness(W: sparse.csr_array) -> np.ndarray:
    """Brandes' algorithm on a weighted undirected graph.

    Returns the unscaled accumulator - one Dijkstra per source, then one pass
    back down the shortest-path DAG adding each node's dependency. Both halves
    are O(m + n log n) per source, so the whole thing is O(nm + n^2 log n).

    Requires strictly positive weights, which is what lets a finalised node be
    skipped for good: any path found later is at least as long, so it can add
    neither a shorter route nor a tie.
    """
    if W.nnz and W.data.min() <= 0:
        raise ValueError("betweenness needs strictly positive weights")

    n = W.shape[0]
    # The inner loop is scalar Python, where indexing a numpy array costs
    # several times what indexing a list does, so the CSR arrays are unpacked
    # once here rather than paying that back on every edge of every sweep.
    indptr = W.indptr.tolist()
    indices = W.indices.tolist()
    data = W.data.tolist()

    bc = [0.0] * n
    dist = [inf] * n
    sigma = [0.0] * n       # number of shortest paths from the source
    delta = [0.0] * n       # dependency of the source on each node
    done = [False] * n
    preds: list[list[int]] = [[] for _ in range(n)]

    for s in range(n):
        stack: list[int] = []       # nodes in the order they were finalised
        heap = [(0.0, s)]
        dist[s] = 0.0
        sigma[s] = 1.0

        while heap:
            d, v = heappop(heap)
            if done[v]:
                continue            # a stale heap entry for an improved node
            done[v] = True
            stack.append(v)
            sigma_v = sigma[v]

            for k in range(indptr[v], indptr[v + 1]):
                w = indices[k]
                if done[w]:
                    continue
                alt = d + data[k]
                if alt < dist[w]:
                    dist[w] = alt
                    sigma[w] = sigma_v
                    into = preds[w]
                    into.clear()
                    into.append(v)
                    heappush(heap, (alt, w))
                elif alt == dist[w]:
                    sigma[w] += sigma_v
                    preds[w].append(v)

        # Back down the DAG, furthest first, so a node's dependency is complete
        # before it is handed to its predecessors.
        for w in reversed(stack):
            coeff = (1.0 + delta[w]) / sigma[w]
            for v in preds[w]:
                delta[v] += sigma[v] * coeff
            if w != s:
                bc[w] += delta[w]

        # Reset only what this source reached, so an unreachable node costs
        # nothing and a small component does not pay for the whole graph.
        for v in stack:
            dist[v] = inf
            sigma[v] = 0.0
            delta[v] = 0.0
            done[v] = False
            preds[v].clear()

    return np.array(bc)


def scored(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The raw accumulator as (undirected, normalised) scores, networkx-style.

    An undirected sweep walks every pair from both ends, so the raw count is
    halved. The normalised score divides by the (n-1)(n-2) ordered pairs that
    could have passed through a node, which puts it on 0-1 and makes cases of
    different sizes comparable.
    """
    n = len(raw)
    normalised = raw / ((n - 1) * (n - 2)) if n > 2 else np.zeros(n)
    return raw * 0.5, normalised


# ---- entry point ---- #


def main() -> None:
    case_dir = gwg.PYPSA_DIR / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available: {', '.join(sorted(p.name for p in gwg.PYPSA_DIR.iterdir() if p.is_dir()))}")

    buses, branches = gwg.read_case(case_dir)
    generators = gwg.read_generators(case_dir) if GENERATORS else None
    A, node_index = gwg.adjacency(buses, branches, generators)
    W = gwg.reciprocal(A)

    components = csgraph.connected_components(W, directed=False)[0]
    started = time.perf_counter()
    raw, normalised = scored(betweenness(W))
    elapsed = time.perf_counter() - started

    frame = node_index.assign(betweenness=normalised, betweenness_raw=raw)
    frame = frame.sort_values("betweenness", ascending=False)
    frame.insert(len(frame.columns), "rank", np.arange(1, len(frame) + 1))

    out = gwg.out_dir(CASE, GENERATORS, "betweenness")
    tag = "_gen" if GENERATORS else ""
    path = out / f"betweenness{tag}.csv"
    frame.to_csv(path)

    label = "station" if "station" in frame else "name"
    top = frame.head(10)
    listing = "\n".join(
        f"    {r.rank:>2}. {r.name_:<10} {str(getattr(r, label, '') or ''):<22}"
        f" {r.betweenness:.4f}"
        for r in top.rename(columns={"name": "name_"}).itertuples()
    )
    print(
        f"{CASE}{' +generators' if GENERATORS else ''}\n"
        f"  nodes      {W.shape[0]} in {components} component"
        f"{'s' if components != 1 else ''}\n"
        f"  edges      {W.nnz // 2}\n"
        f"  computed   exact Brandes in {elapsed:.1f}s\n"
        f"  top nodes  by normalised betweenness\n{listing}\n"
        f"  wrote      {path}"
    )


if __name__ == "__main__":
    main()
