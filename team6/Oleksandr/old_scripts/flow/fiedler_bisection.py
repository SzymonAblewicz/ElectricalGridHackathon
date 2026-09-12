# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa"]
# ///
"""The flow-weighted grid cut into k zones by recursive Fiedler bisection.

`get_flow_graph.py` takes k Laplacian eigenvectors at once and hands them to k-means. This
takes one eigenvector at a time and cuts with it, k-1 times:

    one cluster, the whole graph
    repeat k-1 times:
        lambda_2 of every current cluster       (its Fiedler value, on L_sym of that cluster)
        take the cluster with the smallest one  (the one most weakly held together)
        sweep along its Fiedler vector and cut at the best-scoring prefix

Only the two pieces of the last cut need a new eigenvector - every other cluster is unchanged,
so its lambda_2 is cached. By Cheeger's inequality a small lambda_2 of the normalised Laplacian
means a cluster with a cheap cut somewhere inside it, which is why the smallest is the one cut.

The sweep orders the cluster's nodes by the random-walk Fiedler vector u_2 / sqrt(d) (the
Shi-Malik one, which is what L_sym's eigenvector becomes once the degree skew is divided back
out) and scores every prefix S under one of two cut functions (CUTS below):

    ncut      cut(S)/vol(S) + cut(S)/vol(S')    balance counted in weight
    ratiocut  cut(S)/|S|   + cut(S)/|S'|        balance counted in nodes - generators included,
                                                since they are nodes of this graph

Both use the same eigenvector; only the prefix each picks differs. A plain cut(S) is not
offered: along a sweep it all but always picks a one-node prefix, the lightest leaf.

A cluster can come out of a sweep in more than one piece. Its lambda_2 is then exactly 0 and
its Fiedler vector is any mix of component indicators, so it is cut along its components -
largest component against the rest - a zero-weight cut. The steps file marks such a step
"components" rather than "fiedler".

Both weightings are run, `headroom` and `loading`, as get_flow_graph.py defines them, and both
cut functions on each - four partitions from one power flow. Written to
`data/graphs/<case>/flow/fiedler/`:

  graph_<w>_<stamp>.pdf, {A,D,L,Lsym}_<w>_<stamp>.npz   the full weighted graph, graph_lib's own
                                                         plot and save
  clusters_<run>.csv / .pdf / folder, corridors_<run>.csv  as the other flow scripts write them
  fiedler_<run>.csv    node_index plus one column per cut: the Fiedler vector that cut was made
                       along, NaN outside the cluster it split
  steps_<run>.csv      one row per cut: every candidate's lambda_2, which was cut, how, and at
                       what weight and score

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # graph_lib, cluster, cluster_maps, spectrum live one level up

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402

import cluster as clu  # noqa: E402
import cluster_maps  # noqa: E402
import team6.Oleksandr.Final_scripts.get_flow_graph as gfg  # noqa: E402
import team6.Oleksandr.Final_scripts.graph_lib as gwg  # noqa: E402
import spectrum as spc  # noqa: E402


# ---- what to run ---- #
#
# The first four mean exactly what they mean in get_flow_graph.py, and are pushed into it
# before any of its helpers run - so the graph cut here is the graph it clusters.

DATASET = "kit"               # "tytfs" | "kit"
CASE = "WP2024_all-island"    # a directory name under whichever DATASET selects
BALANCE = "scale_loads"       # "scale_loads" | "none"
SNAPSHOT = 155                # kit only: "anchor" | "peak" | 0..167
DISPATCH = "prorata"          # kit only: "prorata" | "lopf"

K = 6                       # clusters wanted: K-1 cuts

WEIGHTINGS = ("headroom", "loading")   # each run separately
CUTS = ("ncut", "ratiocut")            # each run separately, on each weighting

# Clusters smaller than this are solved densely: eigsh cannot return k >= n eigenpairs, and
# on a few dozen nodes a dense solve costs nothing anyway.
DENSE_BELOW = 50

CUT_FUNCTIONS = ("ncut", "ratiocut")


# ---- one cut ---- #


def fiedler(A: sparse.csr_array) -> tuple[float, np.ndarray | None]:
    """lambda_2 of L_sym and the random-walk Fiedler vector, for one connected cluster.

    A disconnected cluster returns (0, None): its lambda_2 is exactly 0, and its eigenvector
    is an arbitrary mix of component indicators, so bisect() cuts it along its components.
    """
    if csgraph.connected_components(A, directed=False)[0] > 1:
        return 0.0, None
    _, _, L_sym = gwg.matrices(A)
    if A.shape[0] < DENSE_BELOW:
        values, vectors = np.linalg.eigh(L_sym.toarray())
    else:
        values, vectors = spc.spectrum(L_sym, 2)
    d = np.asarray(A.sum(axis=1)).ravel()
    return float(values[1]), vectors[:, 1] / np.sqrt(d)


def sweep(A: sparse.csr_array, order: np.ndarray, cut_fn: str) -> int:
    """Index m of the best-scoring prefix order[:m+1], under `cut_fn`.

    Vectorised the way local_conductance.sweep() is: an edge becomes internal at the step its
    later endpoint joins, so every prefix's internal weight is one bincount and a cumsum, and
    its cut is vol - 2 * internal. The full prefix has an empty complement and scores inf.
    """
    n = len(order)
    d = np.asarray(A.sum(axis=1)).ravel()
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    upper = sparse.triu(A, k=1).tocoo()
    internal = np.bincount(np.maximum(rank[upper.row], rank[upper.col]),
                           weights=upper.data, minlength=n).cumsum()
    vol = d[order].cumsum()
    # Clamped: vol - 2*internal cancels to a few ulps of the volume, which a cut at the 1e-7
    # headroom floor is small enough to be pushed negative by.
    cut = np.maximum(vol - 2 * internal, 0.0)
    if cut_fn == "ncut":
        inside, outside = vol, d.sum() - vol
    else:
        inside = np.arange(1, n + 1, dtype=float)
        outside = n - inside
    with np.errstate(divide="ignore", invalid="ignore"):
        score = cut / inside + cut / outside
    score[-1] = np.inf
    return int(np.argmin(score))


def cut_score(A: sparse.csr_array, side: np.ndarray, cut_fn: str) -> tuple[float, float]:
    """Cut weight and score of one split, recomputed from the edges rather than the sweep."""
    d = np.asarray(A.sum(axis=1)).ravel()
    x = side.astype(float)
    cut = float(x @ (A @ (1 - x)))
    if cut_fn == "ncut":
        inside, outside = d[side].sum(), d[~side].sum()
    else:
        inside, outside = side.sum(), (~side).sum()
    return cut, cut / inside + cut / outside


# ---- the partition ---- #


def bisect(A: sparse.csr_array, k: int, cut_fn: str) -> tuple[np.ndarray, pd.DataFrame, list]:
    """K-1 Fiedler cuts, each on the current cluster with the smallest lambda_2.

    Returns the labels, one row per cut, and per cut the Fiedler vector it was made along as a
    full-length column (NaN outside the cluster it split; all NaN for a component split).
    """
    n = A.shape[0]
    labels = np.zeros(n, dtype=np.int64)
    cache: dict[int, tuple[float, np.ndarray | None]] = {}
    steps, columns = [], []

    for step in range(1, k):
        for c in range(step):
            if c not in cache:
                members = np.flatnonzero(labels == c)
                # A single node has nothing to cut, so it is never a candidate.
                cache[c] = (fiedler(A[members][:, members]) if len(members) > 1
                            else (np.inf, None))
        c = min(cache, key=lambda c: cache[c][0])
        if not np.isfinite(cache[c][0]):
            raise ValueError(f"every cluster is a single node after {step - 1} cuts; K = {k} "
                             f"is more than this graph has nodes to give")
        candidates = " ".join(f"{i}:{cache[i][0]:.4g}" for i in sorted(cache))
        lam2, vector = cache.pop(c)   # c changes below, so its entry is stale either way

        members = np.flatnonzero(labels == c)
        sub = A[members][:, members]
        column = np.full(n, np.nan)
        if vector is None:
            _, comp = csgraph.connected_components(sub, directed=False)
            side = comp != np.argmax(np.bincount(comp))
            how = "components"
        else:
            order = np.argsort(vector, kind="stable")
            side = np.zeros(len(members), dtype=bool)
            side[order[:sweep(sub, order, cut_fn) + 1]] = True
            column[members] = vector
            how = "fiedler"

        cut, score = cut_score(sub, side, cut_fn)
        labels[members[side]] = step
        columns.append(column)
        steps.append({"step": step, "cluster": c, "new_cluster": step, "how": how,
                      "lambda2": lam2, "candidates": candidates, "nodes": len(members),
                      "kept": int((~side).sum()), "split_off": int(side.sum()),
                      "cut": cut, cut_fn: score})
    return labels, pd.DataFrame(steps), columns


# ---- the graph ---- #


def build_flow():
    """get_flow_graph.main()'s power flow, once - it does not depend on the weighting.

    Returns the case's buses, branches (s_nom still in MVA) and generators, the |MW| on every
    branch, the generator dispatch and the snapshot.
    """
    gfg.DATASET, gfg.SNAPSHOT, gfg.DISPATCH = DATASET, SNAPSHOT, DISPATCH

    root = gfg.case_root()
    case_dir = root / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available under DATASET={DATASET!r}: "
            f"{', '.join(sorted(p.name for p in root.iterdir() if p.is_dir()))}")

    gfg.quiet()
    network = pypsa.Network(str(case_dir))
    stamp = "now"
    if DATASET == "kit":
        stamp, _ = gfg.kit_dispatch(network)
    if BALANCE == "scale_loads":
        gfg.rebalance(network)
    network.lpf()

    flow = gfg.branch_flow(network)
    buses, branches = gwg.read_case(case_dir)
    generators = gwg.read_generators(case_dir)
    return buses, branches, generators, flow, network.generators["p_set"], stamp


def weighted_graph(weighting: str, buses, branches, generators, flow, dispatch):
    """The flow graph under one weighting, through get_flow_graph's own helpers."""
    gfg.WEIGHTING = weighting
    weighted = branches.assign(s_nom=gfg.branch_weights(branches, flow))
    generators = generators.assign(p_nom=gfg.generator_weights(generators, dispatch))
    return gwg.adjacency(buses, weighted, generators)


# ---- entry point ---- #


def main() -> None:
    for name, value, allowed in (("DATASET", DATASET, gfg.DATASETS),
                                 ("BALANCE", BALANCE, gfg.BALANCES),
                                 ("DISPATCH", DISPATCH, gfg.DISPATCHES)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
    for name, values, allowed in (("WEIGHTINGS", WEIGHTINGS, gfg.WEIGHTINGS),
                                  ("CUTS", CUTS, CUT_FUNCTIONS)):
        if not set(values) <= set(allowed):
            raise ValueError(f"{name} must be drawn from {allowed}, not {values!r}")

    buses, rated, generators, flow, dispatch, stamp = build_flow()
    folder = gfg.out_dir(CASE, "fiedler")
    when = f"_{pd.Timestamp(stamp):%Y%m%d_%H%M}" if DATASET == "kit" else ""

    for weighting in WEIGHTINGS:
        A, node_index = weighted_graph(weighting, buses, rated, generators, flow, dispatch)
        if not 2 <= K <= A.shape[0]:
            raise ValueError(f"K = {K} must be between 2 and the node count {A.shape[0]}")
        placed = gwg.geocode(node_index, rated)

        graph = f"{weighting}{when}"
        gwg.save(A, folder, graph)
        gwg.plot(A, placed, folder / f"graph_{graph}.pdf",
                 note=f"{CASE} +generators, weighted by {weighting}"
                      + (f", {stamp}" if DATASET == "kit" else ""))

        for cut_fn in CUTS:
            labels, steps, columns = bisect(A, K, cut_fn)
            crossing, total = clu.cut(A, labels)
            sizes = np.bincount(labels, minlength=K)

            run = f"{weighting}_{cut_fn}_k{K}{when}"
            title = (f"{CASE} +generators — Fiedler bisection, {weighting}, {cut_fn}, k = {K}"
                     + (f", {stamp}" if DATASET == "kit" else ""))

            csv_path = folder / f"clusters_{run}.csv"
            node_index.assign(cluster=labels).to_csv(csv_path)
            fiedler_path = folder / f"fiedler_{run}.csv"
            node_index.assign(**{f"cut{s + 1}": col for s, col in enumerate(columns)}).to_csv(
                fiedler_path)
            steps_path = folder / f"steps_{run}.csv"
            steps.to_csv(steps_path, index=False)
            links = clu.corridors(rated, node_index, labels)
            corridors_path = folder / f"corridors_{run}.csv"
            links[links["cut"]].drop(columns="cut").to_csv(corridors_path, index=False)

            rings = gfg.cut_transformer_rings(links, node_index)
            pdf_path = folder / f"clusters_{run}.pdf"
            clu.plot(A, placed, labels, pdf_path, title, rings=rings)
            maps_dir = folder / f"clusters_{run}"
            maps = cluster_maps.plot_per_cluster(
                lambda ax: clu.draw(ax, A, placed, labels, rings), placed, labels, maps_dir,
                title)

            listing = "\n".join(
                f"    {row.step:>2}. cut {row.cluster} at lambda_2 {row.lambda2:.4g} "
                f"({row.how}): {row.nodes} -> {row.kept} + {row.split_off}, "
                f"{cut_fn} {getattr(row, cut_fn):.4g}\n"
                f"        candidates {row.candidates}"
                for row in steps.itertuples())
            print(
                f"{CASE} +generators - Fiedler bisection, {weighting}, {cut_fn}  "
                f"[{DATASET}, {stamp}]\n"
                f"  graph      {A.shape[0]} nodes, {A.nnz // 2} edges\n"
                f"  cuts\n{listing}\n"
                f"  clusters   sizes {sizes.tolist()}\n"
                f"  cut        {100 * crossing / total:.2f}% of the graph's {weighting} weight "
                f"crosses a boundary; {int(links['cut'].sum())} of {len(links)} corridors cut,\n"
                f"             {links.loc[links['cut'], 's_nom'].sum():,.0f} MVA of rating\n"
                f"  wrote      {folder}\n"
                f"               {pdf_path.name}   <- the clustered map\n"
                f"               {csv_path.name}, {fiedler_path.name}, {steps_path.name}, "
                f"{corridors_path.name}\n"
                f"               {maps_dir.name}/   ({len(maps)} per-cluster zooms)\n"
                f"               graph_{graph}.pdf, {{A,D,L,Lsym}}_{graph}.npz\n"
            )


if __name__ == "__main__":
    main()
