# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa"]
# ///
"""Local conductance clustering of the flow-weighted grid: personalised PageRank + sweep cut.

`get_flow_graph.py` clusters the flow graph *globally*: k eigenvectors of the whole
Laplacian, then k-means. This clusters the same graph *locally*, one cluster at a time,
each one grown outward from a seed bus and stopped where it is cheapest to cut off.

What "cheapest to cut off" means is conductance,

    phi(S) = cut(S) / min(vol S, vol V\\S),

the affinity crossing the boundary of S divided by the affinity inside the smaller side. A
set with phi near 0 is one the rest of the grid barely holds on to. Under `headroom` that
is a zone walled in by saturated corridors; under `loading`, a zone walled in by idle ones.

Finding one such set is the Andersen-Chung-Lang recipe:

  1. personalised PageRank from a seed s - a random walk that teleports back to s with
     probability ALPHA each step. p(v) is how much of the walk's time is spent at v.
  2. order the nodes by p(v)/d(v), which undoes the walk's pull towards high-degree nodes.
  3. sweep: of the prefixes {v1}, {v1, v2}, ... of that order, keep the lowest-phi one.

The PageRank vector is solved exactly - one sparse LU per round, shared by every seed -
rather than approximated by ACL's push. On ~1,300 nodes that is faster than push and leaves
no tolerance to tune. The cluster is still local in the sense that matters: it is whatever
the walk from one seed finds, not a slice of a global embedding.

One local cluster is not a partition, so the partition is built by *peeling*. Each round
runs the sweep from every still-unassigned bus on the still-unassigned subgraph, keeps the
lowest-phi set of all of them, labels it and removes it. After K-1 rounds whatever is left
is cluster K. Clusters are therefore numbered in the order they were peeled: cluster 0 is
the most weakly attached region of the grid, and the phi column of conductance_<run>.csv
says by how much.

Two guards stop that from returning nonsense:

  MIN_BUSES             a set must hold this many buses. Without it round 1 returns a single
                        generator whose leaf edge sits at the headroom floor - phi ~ 0,
                        true, and useless as a zone.
  MAX_VOLUME_FRACTION   a set may hold at most this share of the remaining volume. At 0.5 the
                        set is always the smaller side of its cut, which is the side
                        conductance measures anyway.

Leaves orphaned by a peel - a generator whose bus just went, say - go with the set that
orphaned them. That can only lower the set's cut, and it stops isolated nodes collecting in
the remainder.

Written to `data/graphs/<case>/flow/local/`, beside `clustering/` rather than inside it, so
a peeled partition is never mistaken for a spectral one.

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # graph_lib, cluster, cluster_maps live one level up

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

import cluster as clu  # noqa: E402
import cluster_maps  # noqa: E402
import team6.Oleksandr.Final_scripts.get_flow_graph as gfg  # noqa: E402
import team6.Oleksandr.Final_scripts.graph_lib as gwg  # noqa: E402


# ---- what to run ---- #
#
# The first five mean exactly what they mean in get_flow_graph.py, and are pushed into it
# before any of its helpers run - so the graph clustered here is the graph it clusters.

DATASET = "kit"               # "tytfs" | "kit"
CASE = "WP2033_all-island"    # a directory name under whichever DATASET selects
WEIGHTING = "headroom"        # "headroom" | "loading" | "inverse_flow"
BALANCE = "scale_loads"       # "scale_loads" | "none"
SNAPSHOT = 155                # kit only: "anchor" | "peak" | 0..167
DISPATCH = "prorata"          # kit only: "prorata" | "lopf"

K = 9                         # clusters wanted: K-1 peeled, plus the remainder

# Teleport probability. The walk takes 1/ALPHA steps on average before returning to its
# seed, so this sets how far a local cluster can reach: 0.05 is ~20 hops, a sizeable
# region of an all-island graph; 0.2 is ~5 hops, a few substations.
ALPHA = 0.05
MIN_BUSES = 5                 # smallest set a round may return, counted in buses
MAX_VOLUME_FRACTION = 0.5     # largest set a round may return, as a share of remaining volume
# Stop peeling early once the best set left has phi above this - "nothing weakly attached
# remains". None peels all K-1 rounds whatever their phi.
PHI_MAX: float | None = None


# ---- one local cluster ---- #


def pagerank(A: sparse.csr_array, d: np.ndarray, seeds: np.ndarray,
             alpha: float) -> np.ndarray:
    """Personalised PageRank from each seed, one column per seed.

    Solves p = alpha e_s + (1 - alpha) A D^-1 p for every seed at once. The matrix is the
    same for all of them, so it is factorised once and only the right-hand sides differ.
    It is nonsingular for any alpha > 0: A D^-1 is column-stochastic, so its spectral
    radius is 1 and (1 - alpha) pulls it strictly inside.
    """
    n = A.shape[0]
    M = sparse.eye_array(n, format="csc") - (1 - alpha) * (A @ sparse.diags_array(1.0 / d))
    rhs = np.zeros((n, len(seeds)))
    rhs[seeds, np.arange(len(seeds))] = alpha
    return splu(sparse.csc_array(M)).solve(rhs)


def sweep(order: np.ndarray, d: np.ndarray,
          edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Conductance and volume of every prefix of `order`.

    Vectorised rather than grown a node at a time: an edge becomes internal at the step
    its later endpoint joins, so the internal weight of every prefix is one bincount and a
    cumsum, and the cut is vol - 2 * internal. The full prefix has an empty complement and
    so no conductance; it comes back as inf.
    """
    rank = np.empty(len(d), dtype=int)
    rank[order] = np.arange(len(order))
    i, j, w = edges
    internal = np.bincount(np.maximum(rank[i], rank[j]), weights=w,
                           minlength=len(order)).cumsum()
    vol = d[order].cumsum()
    # Clamped: vol - 2*internal cancels to a few ulps of the volume, and a saturated cut
    # at the 1e-7 headroom floor is small enough for that to push it negative.
    cut = np.maximum(vol - 2 * internal, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        phi = cut / np.minimum(vol, d.sum() - vol)
    phi[-1] = np.inf
    return phi, vol


def best_local_cluster(A: sparse.csr_array, is_bus: np.ndarray, alpha: float,
                       min_buses: int, max_fraction: float) -> tuple[float, int, np.ndarray] | None:
    """The lowest-phi sweep set over every bus seed of A, as (phi, seed, members).

    Generators are never seeds: a generator is a leaf on its bus, so a walk from it is a
    walk from its bus with one extra step. None if no seed yields a set inside the guards.
    """
    d = np.asarray(A.sum(axis=1)).ravel()
    seeds = np.flatnonzero(is_bus)
    P = pagerank(A, d, seeds, alpha)
    upper = sparse.triu(A, k=1).tocoo()
    edges = (upper.row, upper.col, upper.data)
    cap = max_fraction * d.sum()

    best = None
    for c, s in enumerate(seeds):
        # Stable sort, so ties - p = 0 beyond the seed's component - break by index and
        # a rerun returns the same partition.
        order = np.argsort(-P[:, c] / d, kind="stable")
        phi, vol = sweep(order, d, edges)
        ok = (np.cumsum(is_bus[order]) >= min_buses) & (vol <= cap)
        if not ok.any():
            continue
        m = np.flatnonzero(ok)[np.argmin(phi[ok])]
        if best is None or phi[m] < best[0]:
            best = (float(phi[m]), int(s), order[:m + 1])
    return best


# ---- the partition ---- #


def peel(A: sparse.csr_array, node_index: pd.DataFrame, k: int) -> tuple[np.ndarray, pd.DataFrame]:
    """K-1 rounds of best_local_cluster on what is left, then the remainder.

    Returns the labels and one row per round. `phi_round` is the set's conductance in the
    graph it was peeled from - the remaining subgraph - which is what chose it; the caller
    adds its conductance in the full graph beside it.
    """
    is_bus = (node_index["node_type"] == "bus").to_numpy()
    labels = np.full(A.shape[0], -1)
    rounds = []
    for r in range(k - 1):
        free = np.flatnonzero(labels < 0)
        sub = A[free][:, free]
        d = np.asarray(sub.sum(axis=1)).ravel()
        # Only a node with nothing left to connect to has zero degree, and orphan absorption
        # below should have taken it already - but 1/d would be a division by zero, not a
        # wrong answer, so it is excluded rather than trusted.
        live = d > 0
        free, sub = free[live], sub[live][:, live]

        found = best_local_cluster(sub, is_bus[free], ALPHA, MIN_BUSES, MAX_VOLUME_FRACTION)
        if found is None or (PHI_MAX is not None and found[0] > PHI_MAX):
            break
        _, seed, members = found

        x = np.zeros(len(free), dtype=bool)
        x[members] = True
        # Orphans: nodes outside the set whose every neighbour is inside it.
        pattern = sub.copy()
        pattern.data[:] = 1.0
        orphans = ~x & (pattern @ (~x).astype(float) == 0)
        x |= orphans

        # Recomputed from the edges rather than read off the sweep: exact, not a
        # difference of two volumes, and it includes the orphans.
        d = np.asarray(sub.sum(axis=1)).ravel()
        cut = float(x.astype(float) @ (sub @ (~x).astype(float)))
        vol = float(d[x].sum())
        labels[free[x]] = r
        rounds.append({
            "cluster": r, "seed": node_index["name"].iat[free[seed]],
            "seed_station": node_index["station"].iat[free[seed]],
            "buses": int(is_bus[free[x]].sum()), "nodes": int(x.sum()),
            "orphans": int(orphans.sum()), "volume_round": vol, "cut_round": cut,
            "phi_round": cut / min(vol, d.sum() - vol),
        })

    rest = labels < 0
    labels[rest] = len(rounds)
    rounds.append({"cluster": len(rounds), "seed": "(remainder)", "seed_station": "",
                   "buses": int(is_bus[rest].sum()), "nodes": int(rest.sum()), "orphans": 0,
                   "volume_round": np.nan, "cut_round": np.nan, "phi_round": np.nan})
    return labels, pd.DataFrame(rounds)


def conductance(A: sparse.csr_array, labels: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
    """Volume, cut and conductance of every cluster in the full graph."""
    d = np.asarray(A.sum(axis=1)).ravel()
    vol = np.bincount(labels, weights=d, minlength=k)
    upper = sparse.triu(A, k=1).tocoo()
    a, b = labels[upper.row], labels[upper.col]
    x = a != b
    cut = (np.bincount(a[x], weights=upper.data[x], minlength=k)
           + np.bincount(b[x], weights=upper.data[x], minlength=k))
    return vol, cut, cut / np.minimum(vol, vol.sum() - vol)


# ---- the graph ---- #


def build_graph():
    """get_flow_graph.main()'s graph, step for step, through its own helpers.

    Returns the flow-weighted adjacency, its node index, the geocoded copy for plotting,
    the branch table still rated in MVA, the snapshot and the per-branch loading.
    """
    gfg.DATASET, gfg.WEIGHTING = DATASET, WEIGHTING
    gfg.SNAPSHOT, gfg.DISPATCH = SNAPSHOT, DISPATCH

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
    s_nom = branches["s_nom"].to_numpy(float)
    loading = flow.reindex(branches["name"]).to_numpy(float) / np.where(s_nom > 0, s_nom, np.nan)

    rated = branches              # s_nom still in MVA - the cut-corridor table reads this
    branches = branches.assign(s_nom=gfg.branch_weights(branches, flow))
    generators = generators.assign(
        p_nom=gfg.generator_weights(generators, network.generators["p_set"]))
    A, node_index = gwg.adjacency(buses, branches, generators)
    placed = gwg.geocode(node_index, branches)
    return A, node_index, placed, rated, stamp, loading


# ---- entry point ---- #


def main() -> None:
    for name, value, allowed in (("DATASET", DATASET, gfg.DATASETS),
                                 ("WEIGHTING", WEIGHTING, gfg.WEIGHTINGS),
                                 ("BALANCE", BALANCE, gfg.BALANCES),
                                 ("DISPATCH", DISPATCH, gfg.DISPATCHES)):
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
    if not 0 < ALPHA < 1:
        raise ValueError(f"ALPHA must lie in (0, 1), not {ALPHA}")
    if not 0 < MAX_VOLUME_FRACTION <= 0.5:
        raise ValueError(f"MAX_VOLUME_FRACTION must lie in (0, 0.5], not {MAX_VOLUME_FRACTION}")

    A, node_index, placed, rated, stamp, loading = build_graph()
    labels, rounds = peel(A, node_index, K)
    k = int(labels.max()) + 1     # fewer than K if PHI_MAX or the guards stopped the peel

    vol, cut, phi = conductance(A, labels, k)
    rounds = rounds.assign(volume=vol, cut=cut, phi=phi)
    crossing, total = clu.cut(A, labels)

    folder = gfg.out_dir(CASE, "local")
    run = f"{WEIGHTING}_a{ALPHA:g}_k{k}"
    if DATASET == "kit":
        run += f"_{pd.Timestamp(stamp):%Y%m%d_%H%M}"
    title = (f"{CASE} +generators — local conductance, {WEIGHTING}, alpha = {ALPHA:g}, k = {k}"
             + (f", {stamp}" if DATASET == "kit" else ""))

    csv_path = folder / f"clusters_{run}.csv"
    node_index.assign(cluster=labels).to_csv(csv_path)
    phi_path = folder / f"conductance_{run}.csv"
    rounds.to_csv(phi_path, index=False)
    links = clu.corridors(rated, node_index, labels)
    corridors_path = folder / f"corridors_{run}.csv"
    links[links["cut"]].drop(columns="cut").to_csv(corridors_path, index=False)

    rings = gfg.cut_transformer_rings(links, node_index)
    pdf_path = folder / f"clusters_{run}.pdf"
    clu.plot(A, placed, labels, pdf_path, title, rings=rings)
    maps_dir = folder / f"clusters_{run}"
    maps = cluster_maps.plot_per_cluster(
        lambda ax: clu.draw(ax, A, placed, labels, rings), placed, labels, maps_dir, title)

    listing = "\n".join(
        f"    {row.cluster:>2}. {row.buses:>4} buses {row.nodes:>5} nodes   "
        f"phi {row.phi:>10.3g}   "
        + (f"(peeled at {row.phi_round:.3g}, seed {row.seed} {row.seed_station})"
           if row.seed != "(remainder)" else "(remainder)")
        for row in rounds.itertuples())
    print(
        f"{CASE} +generators - local conductance, {WEIGHTING}  [{DATASET}, {stamp}]\n"
        f"  flows      max loading {np.nanmax(loading):.1%}, "
        f"{int(np.nansum(loading > 1))} over rating\n"
        f"  graph      {A.shape[0]} nodes, {A.nnz // 2} edges\n"
        f"  peeling    alpha {ALPHA:g}, >= {MIN_BUSES} buses, <= {MAX_VOLUME_FRACTION:g} of "
        f"remaining volume, {k - 1} of {K - 1} rounds run\n"
        f"  clusters   (phi in the full graph; peeled-at is phi in the subgraph it came from)\n"
        f"{listing}\n"
        f"  cut        {100 * crossing / total:.2f}% of the graph's weight crosses a boundary; "
        f"{int(links['cut'].sum())} of {len(links)} corridors cut,\n"
        f"             {links.loc[links['cut'], 's_nom'].sum():,.0f} MVA of rating\n"
        f"  wrote      {folder}\n"
        f"               {pdf_path.name}   <- the clustered map\n"
        f"               {csv_path.name}, {phi_path.name}, {corridors_path.name}\n"
        f"               {maps_dir.name}/   ({len(maps)} per-cluster zooms)"
    )


if __name__ == "__main__":
    main()
