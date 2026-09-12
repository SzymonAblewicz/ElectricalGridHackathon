# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa"]
# ///
"""The grid cut into k zones by max-flow / min-cut, instead of by a Laplacian.

`cluster.py` (used by both `get_susceptance_graph.py` and `get_flow_graph.py`) is spectral: it
takes the k smallest Laplacian eigenvectors as coordinates and runs k-means on the resulting
cloud, which is a *relaxation* of finding the cheapest k-way cut subject to the pieces staying
roughly balanced. This file finds k-way cuts a different way - by literally computing min
s-t cuts with max-flow - which drops the balance requirement entirely. A corridor only has to
be a genuine bottleneck to be picked, whatever size pocket that isolates; that is precisely
the property the participant kit's shift-factor heatmap showed spectral clustering missing on
`WP2033_all-island` - six of its ten congested branches sit inside well-meshed parts of the
network that no balanced k-way partition will ever propose as a boundary.

**The algorithm: Gomory-Hu tree, built Gusfield's way (1990).** A Gomory-Hu tree is a
spanning tree on the same n nodes such that the min s-t cut value between ANY pair of nodes
equals the minimum edge weight on the tree path between them - one object standing in for all
n(n-1)/2 pairwise min-cuts at once. Gusfield's construction gets it with exactly n-1 max-flow
computations, all run on the *original* graph (no contraction, unlike Gomory and Hu's own
1961 algorithm):

    parent = [0, 0, ..., 0]              # every node initially points at an arbitrary root
    for i in 1..n-1:
        t = parent[i]
        f, S = maxflow_mincut(i, t)      # S = the min-cut side containing i
        weight[i] = f
        for j in i+1..n-1:
            if j in S and parent[j] == t:
                parent[j] = i
        if parent[t] in S:                # re-root: i sits between t and t's own parent
            parent[i], parent[t] = parent[t], i
            weight[i], weight[t] = weight[t], f

Tree edge i is (i, parent[i]) weighted `weight[i]`. **k clusters, deterministically**: drop
the k-1 lightest tree edges. A tree has exactly n-1 edges and no cycles, so removing exactly
k-1 of them always yields exactly k connected components - no k-means, no restarts, no local
minima to get unlucky with. The price is that ties in tree-edge weight are broken by array
order rather than any judgement call; two equally-thin corridors near the cut threshold can
have their fates decided by iteration order.

**Weighting** follows `get_flow_graph.py`'s own three readings of a corridor as *affinity*
(WEIGHTING below): `headroom` (s_nom - |F|, spare MW - the default, and the one the participant
kit's dispatch actually saturates) or `loading` (|F| / s_nom, scale-free "how full") or
`inverse_flow`. Whichever is chosen is fed to max-flow as a literal edge *capacity* - which is
exactly how spectral clustering already reads it too (a Laplacian entry is also read as
affinity, never distance, the same requirement `spectrum.py` documents when it refuses
`reciprocal()`). Only `betweenness.py`'s node-centrality read needs the reciprocal instead,
because a shortest path needs a true distance; a max-flow capacity does not.

This script reuses `get_flow_graph.py`'s dispatch machinery directly - `kit_dispatch`,
`rebalance`, `branch_flow`, `branch_weights`, `generator_weights` - rather than re-deriving it,
so a corridor's weight here and its weight in the spectral run can never quietly disagree.
Those functions read their inputs off `get_flow_graph`'s own module-level constants rather
than taking every argument, so this file's `main()` sets `gfg.DATASET`, `gfg.WEIGHTING`,
`gfg.BALANCE`, `gfg.DISPATCH` and `gfg.SNAPSHOT` from the constants below before calling them,
rather than trusting whatever `get_flow_graph.py` was last edited to run.

**Integers, and why.** `scipy.sparse.csgraph.maximum_flow` requires integer capacities. A
headroom of 1e-7 MVA (`get_flow_graph.HEADROOM_FLOOR`) is what a saturated corridor is read
as, so weights are scaled by SCALE before rounding rather than truncated - at SCALE = 1000 a
saturated corridor still rounds to 0 (correctly: no spare MW to speak of), and a corridor with
a 0.5 MW difference in headroom from its neighbour keeps that difference rather than being
rounded into it.

**Cost, honestly.** n-1 max-flow computations on a ~1,600-node case is not free - this takes
minutes, not seconds, unlike a single Laplacian eigendecomposition. A progress line prints
every few hundred nodes so a long run does not look hung.

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # graph_lib, cluster live one level up

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402
from scipy.sparse.csgraph import maximum_flow  # noqa: E402

import cluster as clu  # noqa: E402
import cluster_maps  # noqa: E402
import team6.Oleksandr.Final_scripts.graph_lib as gwg  # noqa: E402
import team6.Oleksandr.Final_scripts.get_flow_graph as gfg  # noqa: E402


# ---- what to run ---- #
#
# Same meaning as the matching constants in get_flow_graph.py - see that file's own
# docstring for DATASET/CASE pairing, what SNAPSHOT = 155 is ("worst" hour, 3 branches over
# rating), and why BALANCE = "scale_loads" is the one to leave alone.

DATASET = "kit"               # "tytfs" | "kit"
CASE = "WP2033_all-island"    # a directory name under whichever DATASET selects
WEIGHTING = "loading"        # "headroom" | "loading" | "inverse_flow" - fed to max-flow as
                              # a literal edge capacity, same affinity reading cluster.py gives it
BALANCE = "scale_loads"       # "scale_loads" | "none"
DISPATCH = "prorata"          # "prorata" | "lopf" - kit only
SNAPSHOT = 155                # "anchor" | "peak" | 0..167 - kit only

K = 9                         # clusters wanted: the k-1 lightest Gomory-Hu tree edges are cut

SCALE = 1000                  # headroom/loading -> integer capacity units for maximum_flow


# ---- Gomory-Hu tree (Gusfield 1990) ---- #


def min_cut_side(capacity: sparse.csr_matrix, flow: sparse.csr_matrix, source: int,
                 n: int) -> np.ndarray:
    """Boolean mask of the nodes on `source`'s side of the min cut just computed.

    The residual capacity from i to j is `capacity[i, j] - flow[i, j]`; `source` can still
    reach every node the max-flow has spare residual capacity toward. Nodes reachable from
    `source` in that residual graph are exactly the min-cut side containing it - the textbook
    read-off from a finished max-flow computation, needed here because scipy's
    `maximum_flow` returns the flow value and the flow itself but not which side is which.
    """
    residual = (capacity - flow).tocsr()
    residual.data[residual.data <= 0] = 0
    residual.eliminate_zeros()

    reachable = np.zeros(n, dtype=bool)
    reachable[source] = True
    indptr, indices = residual.indptr, residual.indices
    stack = [source]
    while stack:
        u = stack.pop()
        for idx in range(indptr[u], indptr[u + 1]):
            v = int(indices[idx])
            if not reachable[v]:
                reachable[v] = True
                stack.append(v)
    return reachable


def gomory_hu_tree(capacity: sparse.csr_matrix, progress_every: int = 200) -> tuple[
        np.ndarray, np.ndarray]:
    """Gusfield's n-1-maxflow construction. Returns (parent, weight), each length n.

    `parent[0]` and `weight[0]` are meaningless - node 0 is the arbitrary root every other
    node's tree edge eventually chains back to. Tree edge i is (i, parent[i]), weighted
    `weight[i]`; the min s-t cut between any pair of nodes is the lightest edge on the tree
    path between them.
    """
    n = capacity.shape[0]
    parent = np.zeros(n, dtype=np.int64)
    weight = np.zeros(n, dtype=np.float64)
    started = time.perf_counter()

    for i in range(1, n):
        t = int(parent[i])
        result = maximum_flow(capacity, i, t)
        side = min_cut_side(capacity, result.flow.tocsr(), i, n)
        weight[i] = float(result.flow_value)

        for j in range(i + 1, n):
            if side[j] and parent[j] == t:
                parent[j] = i
        if side[int(parent[t])]:
            parent[i], parent[t] = parent[t], i
            weight[i], weight[t] = weight[t], weight[i]

        if i % progress_every == 0 or i == n - 1:
            elapsed = time.perf_counter() - started
            print(f"    gomory-hu  {i}/{n - 1} min-cuts done [{elapsed:.0f}s]")
    return parent, weight


def tree_to_labels(parent: np.ndarray, weight: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """k clusters from the tree: drop its k-1 lightest edges, label the surviving pieces.

    A spanning tree has exactly n-1 edges and no cycles, so removing exactly k-1 of them
    always yields exactly k connected components - unlike k-means, there is no restart to
    get unlucky and no cluster that can come out empty. Returns the labels and the k-1
    cut weights (the min-cut value each severed tree edge represents), largest first.
    """
    n = len(parent)
    if not 2 <= k <= n:
        raise ValueError(f"K = {k} must be between 2 and the node count {n}")

    edge_weight = weight[1:]                       # edge i lives at index i-1 here
    order = np.argsort(edge_weight, kind="stable")
    cut_nodes = order[:k - 1] + 1                   # back to node indices 1..n-1
    severed = np.zeros(n, dtype=bool)
    severed[cut_nodes] = True

    kept = np.arange(1, n)[~severed[1:]]
    tree = sparse.coo_matrix(
        (np.ones(len(kept)), (kept, parent[kept])), shape=(n, n))
    n_comp, labels = csgraph.connected_components(tree, directed=False)
    assert n_comp == k, f"expected {k} pieces from cutting {k - 1} tree edges, got {n_comp}"
    return labels, weight[cut_nodes][::-1]


# ---- entry point ---- #


def main() -> None:
    if WEIGHTING not in gfg.WEIGHTINGS:
        raise ValueError(f"WEIGHTING must be one of {gfg.WEIGHTINGS}, not {WEIGHTING!r}")
    if BALANCE not in gfg.BALANCES:
        raise ValueError(f"BALANCE must be one of {gfg.BALANCES}, not {BALANCE!r}")
    if DISPATCH not in gfg.DISPATCHES:
        raise ValueError(f"DISPATCH must be one of {gfg.DISPATCHES}, not {DISPATCH!r}")

    gfg.quiet()
    # get_flow_graph's dispatch helpers read these off the module rather than taking every
    # argument - set explicitly so this run never silently depends on whatever
    # get_flow_graph.py's own constants were last left at.
    gfg.DATASET = DATASET
    gfg.WEIGHTING = WEIGHTING
    gfg.BALANCE = BALANCE
    gfg.DISPATCH = DISPATCH
    gfg.SNAPSHOT = SNAPSHOT

    case_dir = gfg.case_root() / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(f"no such case: {case_dir}")

    network = pypsa.Network(str(case_dir))
    stamp, fraction = ("now", float("nan"))
    if DATASET == "kit":
        stamp, fraction = gfg.kit_dispatch(network)

    if BALANCE == "scale_loads":
        gfg.rebalance(network)
    network.lpf()

    flow = gfg.branch_flow(network)
    buses, branches = gwg.read_case(case_dir)
    generators = gwg.read_generators(case_dir)

    rated = branches                # s_nom still in MVA - the cut-corridor table reads this
    branches = branches.assign(s_nom=gfg.branch_weights(branches, flow))
    generators = generators.assign(
        p_nom=gfg.generator_weights(generators, network.generators["p_set"]))

    A, node_index = gwg.adjacency(buses, branches, generators)
    n_nodes = A.shape[0]
    if not 2 <= K < n_nodes:
        raise ValueError(f"K = {K} must be between 2 and {n_nodes - 1} for a {n_nodes}-node graph")

    scaled = np.rint(A.data * SCALE).astype(np.int64)
    capacity = sparse.csr_matrix((scaled, A.indices, A.indptr), shape=A.shape)
    capacity.eliminate_zeros()

    print(f"{CASE} +generators - max-flow/min-cut ({WEIGHTING})  [{DATASET}]\n"
          f"  graph      {n_nodes} nodes, {capacity.nnz // 2} edges - "
          f"{n_nodes - 1} max-flow computations ahead")
    parent, weight = gomory_hu_tree(capacity)
    labels, cut_values = tree_to_labels(parent, weight, K)
    labels = labels.astype(np.int64)

    crossing, total = clu.cut(A, labels)
    sizes = np.bincount(labels, minlength=K)

    links = clu.corridors(rated, node_index, labels)
    rings = gfg.cut_transformer_rings(links, node_index)
    placed = gwg.geocode(node_index, branches)

    run = f"gomoryhu_{WEIGHTING}_k{K}"
    if DATASET == "kit":
        run += f"_{pd.Timestamp(stamp):%Y%m%d_%H%M}"
    out = gfg.out_dir(CASE, "gomory_hu")

    table = node_index.assign(cluster=labels)
    table["gh_tree_weight"] = weight / SCALE        # this node's own Gomory-Hu tree edge, MW
    csv_path = out / f"clusters_{run}.csv"
    table.to_csv(csv_path)

    corridors_path = out / f"corridors_{run}.csv"
    links[links["cut"]].drop(columns="cut").to_csv(corridors_path, index=False)

    pdf_path = out / f"clusters_{run}.pdf"
    title = (f"{CASE} +generators - max-flow/min-cut ({WEIGHTING}), k = {K}"
             + (f", {stamp}" if DATASET == "kit" else ""))
    clu.plot(A, placed, labels, pdf_path, title, rings=rings)

    maps_dir = out / f"clusters_{run}"
    maps = cluster_maps.plot_per_cluster(
        lambda ax: clu.draw(ax, A, placed, labels, rings), placed, labels, maps_dir, title)

    listing = "\n".join(f"    {c:>2}. {sizes[c]:>5} nodes" for c in range(K))
    cuts = "\n".join(f"    {v / SCALE:>13,.4g}" for v in cut_values)
    print(
        f"  clusters\n{listing}\n"
        f"  cut        {100 * crossing / total:.2f}% of the graph's {WEIGHTING} weight crosses "
        f"a boundary\n"
        f"             {int(links['cut'].sum())} of {len(links)} corridors cut, "
        f"{links.loc[links['cut'], 's_nom'].sum():,.0f} MVA of rating\n"
        f"  severed    the {K - 1} tree edges cut, largest min-cut value first:\n{cuts}\n"
        f"  wrote      {csv_path}\n"
        f"             {corridors_path}\n"
        f"             {pdf_path}\n"
        f"             {maps_dir}/   ({len(maps)} per-cluster zooms)"
    )


if __name__ == "__main__":
    main()
