# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa"]
# ///
"""The flow-weighted grid cut into k zones by higher-order Cheeger rounding.

`fiedler_bisection.py` uses one eigenvector per cut, k-1 times. This uses the bottom k
eigenvectors at once, the way Lee, Oveis Gharan & Trevisan's higher-order Cheeger inequality
does (JACM 2014, arXiv:1111.1055, Theorem 1.1):

    lambda_k / 2  <=  rho(k)  <=  O(k^2) sqrt(lambda_k)

where lambda_k is the k-th smallest eigenvalue of L_sym (lambda_1 = 0) and rho(k) is the best
achievable worst conductance over k disjoint sets, phi(S) = cut(S) / vol(S). The left half is
a certificate every run prints: no k disjoint sets, however chosen, can all have conductance
below lambda_k / 2 (the Rayleigh quotient of a set's indicator is its phi; k disjoint
indicators span a k-dim space whose quotient is at most 2 max phi; Courant-Fischer). The right
half's constant is not given by the paper, so no upper bound is printed.

The rounding, following the paper's shape:

  1. embed        F(v) = D^-1/2 (u_1(v), ..., u_k(v)), the bottom k eigenvectors of L_sym.
                  A node's mass d(v) |F(v)|^2 is its share of the embedding; masses sum to k.
  2. project      F^(v) = F(v) / |F(v)|, compared by d_F(u, v) = |F^(u) - F^(v)|. Nodes
                  pointing the same way belong together, however long their vectors.
  3. regions      k centres by mass-weighted farthest point in d_F, from the heaviest node,
                  and every node to its nearest centre. The paper partitions the sphere at
                  random; this is a deterministic stand-in so a rerun returns the same answer.
  4. localise     h_i(v) = |F(v)| min(1, margin(v) / EPS) inside cell i, 0 outside, where
                  margin is how much nearer v is to its own centre than to the next. h_i falls
                  to 0 at the cell's edge - the paper's theta = max(0, 1 - dist / eps) taper -
                  so the k supports are disjoint by construction. EPS -> 0 is hard cells.
  5. sweep        each cell's nodes by h_i, descending; the prefix of lowest cut/vol in the
                  full graph is that cell's core set.

That gives k disjoint cores which need not cover the grid. The rest are absorbed a layer at a
time: each unassigned node joins the core it has the most edge weight to, until none is left.
The clusters CSV keeps which nodes were core; the sets CSV gives each set's conductance before
and after absorption, and the certificate is checked against both - both are k disjoint sets.

Same graph as fiedler_bisection.py: one power flow at SNAPSHOT, weighted by headroom and by
loading, each run separately. Written to `data/graphs/<case>/flow/cheeger/`:

  graph_<w>_<stamp>.pdf, {A,D,L,Lsym}_<w>_<stamp>.npz   the full weighted graph (graph_lib)
  clusters_<run>.csv / .pdf / folder, corridors_<run>.csv  as the other flow scripts write them
  sets_<run>.csv       per set: centre, sizes, volume, cut and conductance, core and final
  embedding_<run>.npz  eigenvalues, F, mass, centres

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # graph_lib, cluster, cluster_maps, spectrum live one level up

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

import cluster as clu  # noqa: E402
import cluster_maps  # noqa: E402
import team6.Oleksandr.old_scripts.flow.fiedler_bisection as fb  # noqa: E402
import team6.Oleksandr.Final_scripts.get_flow_graph as gfg  # noqa: E402
import team6.Oleksandr.Final_scripts.graph_lib as gwg  # noqa: E402
import spectrum as spc  # noqa: E402


# ---- what to run ---- #
#
# The first five mean exactly what they mean in get_flow_graph.py, and are pushed through
# fiedler_bisection's graph builder into it - so the graph cut here is the graph cut there.

DATASET = "kit"               # "tytfs" | "kit"
CASE = "WP2033_all-island"    # a directory name under whichever DATASET selects
BALANCE = "scale_loads"       # "scale_loads" | "none"
SNAPSHOT = 155                # kit only: "anchor" | "peak" | 0..167
DISPATCH = "prorata"          # kit only: "prorata" | "lopf"

K = 9                         # clusters wanted = eigenvectors used

WEIGHTINGS = ("headroom", "loading")   # each run separately

# Width of the taper at a cell's edge, in d_F (which runs 0..2). A node whose own centre is
# nearer than the next by EPS or more keeps its full |F|; one on the boundary gets 0.
EPS = 0.1


# ---- the rounding ---- #


def embedding(A: sparse.csr_array, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The k smallest eigenvalues of L_sym, the embedding F = D^-1/2 U, and the degrees."""
    _, _, L_sym = gwg.matrices(A)
    if A.shape[0] < fb.DENSE_BELOW:
        values, U = np.linalg.eigh(L_sym.toarray())
        values, U = values[:k], U[:, :k]
    else:
        values, U = spc.spectrum(L_sym, k)
    d = np.asarray(A.sum(axis=1)).ravel()
    return values, U / np.sqrt(d)[:, None], d


def centres(Fhat: np.ndarray, mass: np.ndarray, k: int) -> np.ndarray:
    """k nodes, far apart in d_F and heavy: mass-weighted farthest-point, heaviest first."""
    chosen = [int(np.argmax(mass))]
    dist = np.linalg.norm(Fhat - Fhat[chosen[0]], axis=1)
    for _ in range(k - 1):
        nxt = int(np.argmax(mass * dist))
        if nxt in chosen:
            raise ValueError(f"only {len(chosen)} distinct directions in the embedding; "
                             f"K = {k} asks for more regions than it separates")
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(Fhat - Fhat[nxt], axis=1))
    return np.array(chosen)


def best_prefix(upper: tuple[np.ndarray, np.ndarray, np.ndarray], d: np.ndarray,
                order: np.ndarray) -> int:
    """Index m of the prefix order[:m+1] of lowest cut/vol, cut counted in the full graph.

    As fiedler_bisection.sweep(), with one change: `order` covers only part of the graph, so
    an edge to a node outside it never becomes internal and stays in every prefix's cut.
    """
    n, m = len(d), len(order)
    rank = np.full(n, m)
    rank[order] = np.arange(m)
    i, j, w = upper
    joins = np.maximum(rank[i], rank[j])
    inside = joins < m
    internal = np.bincount(joins[inside], weights=w[inside], minlength=m).cumsum()
    vol = d[order].cumsum()
    cut = np.maximum(vol - 2 * internal, 0.0)   # clamped: cancellation, as in fb.sweep
    return int(np.argmin(cut / vol))


def cheeger_sets(A: sparse.csr_array, F: np.ndarray, d: np.ndarray, k: int,
                 eps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """k disjoint core sets. Returns labels (-1 off every core), the centres and h."""
    norms = np.linalg.norm(F, axis=1)
    Fhat = F / norms[:, None]
    mass = d * norms ** 2
    centre = centres(Fhat, mass, k)

    to_centre = np.linalg.norm(Fhat[:, None, :] - Fhat[centre][None, :, :], axis=2)
    cell = np.argmin(to_centre, axis=1)
    two = np.sort(to_centre, axis=1)[:, :2]
    margin = two[:, 1] - two[:, 0]
    h = norms * (np.minimum(1.0, margin / eps) if eps > 0 else 1.0)

    upper = sparse.triu(A, k=1).tocoo()
    edges = (upper.row, upper.col, upper.data)
    labels = np.full(len(d), -1)
    for i in range(k):
        support = np.flatnonzero((cell == i) & (h > 0))
        # Stable, so ties in h break by index and a rerun returns the same sets.
        order = support[np.argsort(-h[support], kind="stable")]
        labels[order[:best_prefix(edges, d, order) + 1]] = i
    return labels, centre, h


def absorb(A: sparse.csr_array, core: np.ndarray, k: int) -> np.ndarray:
    """Every non-core node into the set it has most edge weight to, one layer at a time."""
    labels = core.copy()
    while (labels < 0).any():
        free = labels < 0
        onehot = np.zeros((len(labels), k))
        onehot[np.flatnonzero(~free), labels[~free]] = 1.0
        pull = A @ onehot
        take = free & (pull.max(axis=1) > 0)
        if not take.any():
            raise ValueError(f"{int(free.sum())} nodes have no path to any core set")
        labels[take] = pull[take].argmax(axis=1)
    return labels


def expansion(A: sparse.csr_array, labels: np.ndarray, k: int) -> tuple[np.ndarray, ...]:
    """Volume, cut and cut/vol of every set; label -1 is outside every set."""
    d = np.asarray(A.sum(axis=1)).ravel()
    on = labels >= 0
    vol = np.bincount(labels[on], weights=d[on], minlength=k)
    upper = sparse.triu(A, k=1).tocoo()
    a, b, w = labels[upper.row], labels[upper.col], upper.data
    x = a != b
    cut = (np.bincount(a[x & (a >= 0)], weights=w[x & (a >= 0)], minlength=k)
           + np.bincount(b[x & (b >= 0)], weights=w[x & (b >= 0)], minlength=k))
    return vol, cut, cut / vol


# ---- entry point ---- #


def main() -> None:
    for name, values, allowed in (("WEIGHTINGS", WEIGHTINGS, gfg.WEIGHTINGS),):
        if not set(values) <= set(allowed):
            raise ValueError(f"{name} must be drawn from {allowed}, not {values!r}")
    if EPS < 0:
        raise ValueError(f"EPS must be >= 0, not {EPS}")

    fb.DATASET, fb.CASE, fb.BALANCE = DATASET, CASE, BALANCE
    fb.SNAPSHOT, fb.DISPATCH = SNAPSHOT, DISPATCH
    buses, rated, generators, flow, dispatch, stamp = fb.build_flow()
    folder = gfg.out_dir(CASE, "cheeger")
    when = f"_{pd.Timestamp(stamp):%Y%m%d_%H%M}" if DATASET == "kit" else ""

    for weighting in WEIGHTINGS:
        A, node_index = fb.weighted_graph(weighting, buses, rated, generators, flow, dispatch)
        if not 2 <= K < A.shape[0]:
            raise ValueError(f"K = {K} must be between 2 and {A.shape[0] - 1}")
        placed = gwg.geocode(node_index, rated)

        graph = f"{weighting}{when}"
        gwg.save(A, folder, graph)
        gwg.plot(A, placed, folder / f"graph_{graph}.pdf",
                 note=f"{CASE} +generators, weighted by {weighting}"
                      + (f", {stamp}" if DATASET == "kit" else ""))

        values, F, d = embedding(A, K)
        core, centre, h = cheeger_sets(A, F, d, K, EPS)
        labels = absorb(A, core, K)
        vol_c, cut_c, phi_c = expansion(A, core, K)
        vol_f, cut_f, phi_f = expansion(A, labels, K)
        bound = values[K - 1] / 2
        is_bus = (node_index["node_type"] == "bus").to_numpy()

        sets = pd.DataFrame({
            "cluster": np.arange(K),
            "centre": node_index["name"].to_numpy()[centre],
            "centre_station": node_index["station"].to_numpy()[centre],
            "core_buses": np.bincount(core[(core >= 0) & is_bus], minlength=K),
            "core_nodes": np.bincount(core[core >= 0], minlength=K),
            "core_volume": vol_c, "core_cut": cut_c, "core_phi": phi_c,
            "buses": np.bincount(labels[is_bus], minlength=K),
            "nodes": np.bincount(labels, minlength=K),
            "volume": vol_f, "cut": cut_f, "phi": phi_f,
        })

        run = f"{weighting}_k{K}_eps{EPS:g}{when}"
        title = (f"{CASE} +generators — higher-order Cheeger, {weighting}, k = {K}"
                 + (f", {stamp}" if DATASET == "kit" else ""))
        csv_path = folder / f"clusters_{run}.csv"
        node_index.assign(cluster=labels, core=core >= 0, h=h).to_csv(csv_path)
        sets_path = folder / f"sets_{run}.csv"
        sets.to_csv(sets_path, index=False)
        emb_path = folder / f"embedding_{run}.npz"
        np.savez(emb_path, eigenvalues=values, F=F, mass=d * (F ** 2).sum(axis=1),
                 centres=centre, eps=np.array(EPS))
        links = clu.corridors(rated, node_index, labels)
        corridors_path = folder / f"corridors_{run}.csv"
        links[links["cut"]].drop(columns="cut").to_csv(corridors_path, index=False)

        rings = gfg.cut_transformer_rings(links, node_index)
        pdf_path = folder / f"clusters_{run}.pdf"
        clu.plot(A, placed, labels, pdf_path, title, rings=rings)
        maps_dir = folder / f"clusters_{run}"
        maps = cluster_maps.plot_per_cluster(
            lambda ax: clu.draw(ax, A, placed, labels, rings), placed, labels, maps_dir, title)

        crossing, total = clu.cut(A, labels)
        listing = "\n".join(
            f"    {r.cluster:>2}. centre {r.centre} {r.centre_station}: core {r.core_buses} "
            f"buses/{r.core_nodes} nodes phi {r.core_phi:.4g} -> final {r.buses} buses/"
            f"{r.nodes} nodes phi {r.phi:.4g}"
            for r in sets.itertuples())

        def verdict(phi: np.ndarray) -> str:
            # A violation is not a bad clustering, it is a bug: the bound holds for any k
            # disjoint sets whatsoever.
            return "holds" if phi.max() >= bound * (1 - 1e-9) else "VIOLATED - a bug"

        print(
            f"{CASE} +generators - higher-order Cheeger, {weighting}  [{DATASET}, {stamp}]\n"
            f"  graph      {A.shape[0]} nodes, {A.nnz // 2} edges\n"
            f"  spectrum   lambda_1..lambda_{K} = {np.array2string(values, precision=4)}\n"
            f"  sets       EPS {EPS:g}; {int((core >= 0).sum())} of {len(core)} nodes in a "
            f"core, the rest absorbed\n{listing}\n"
            f"  cheeger    lambda_{K}/2 = {bound:.4g} <= max phi: cores {phi_c.max():.4g} "
            f"({verdict(phi_c)}), final {phi_f.max():.4g} ({verdict(phi_f)})\n"
            f"  cut        {100 * crossing / total:.2f}% of the graph's {weighting} weight "
            f"crosses a boundary; {int(links['cut'].sum())} of {len(links)} corridors cut,\n"
            f"             {links.loc[links['cut'], 's_nom'].sum():,.0f} MVA of rating\n"
            f"  wrote      {folder}\n"
            f"               {pdf_path.name}   <- the clustered map\n"
            f"               {csv_path.name}, {sets_path.name}, {corridors_path.name}, "
            f"{emb_path.name}\n"
            f"               {maps_dir.name}/   ({len(maps)} per-cluster zooms)\n"
            f"               graph_{graph}.pdf, {{A,D,L,Lsym}}_{graph}.npz\n"
        )


if __name__ == "__main__":
    main()
