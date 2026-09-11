# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib"]
# ///
"""k-means on the Laplacian eigenvectors: the grid cut into k cohesive zones.

The second half of spectral clustering. `spectrum.py` writes the eigenvectors
belonging to the smallest Laplacian eigenvalues; this reads them, treats the k
of them as k coordinates per node, and runs k-means on the resulting cloud of
one point per bus. Buses that k-means puts together are buses the graph struggles
to separate without cutting a lot of capacity, which is what makes the output a
partition of the grid rather than a ranking of it.

Worth being clear about what is clustered, because the phrasing invites the other
reading: k-means runs on the n-by-k matrix of eigen*vectors*, one row per node,
not on the k scalar eigenvalues. The eigenvalues say how many clusters to look
for (that is the eigengap `spectrum.py` prints); the eigenvectors say which node
goes where.

The three embeddings differ only in how the eigenvector matrix is rescaled
before k-means sees it:

    sym     rows scaled to unit length          (Ng-Jordan-Weiss)
    rw      columns scaled by D^-1/2            (Shi-Malik)
    unnorm  used as-is

The first two are the two standard ways of undoing the degree skew, and on these
cases they agree closely. "unnorm" is there to be looked at rather than used: on
a graph whose weighted degrees span 0.02 MW to 7,910 MVA it spends every
eigenvector isolating one near-dangling generator, and the six clusters come out
as one blob and five singletons.

k-means is `scipy.cluster.vq.kmeans2`, which has no `n_init`, so the restarts are
done here: RESTARTS seeds, keep whichever lands the lowest within-cluster sum of
squares. Spectral embeddings do have bad local minima, and a single k-means++
draw finds one often enough to matter.

Set the constants below - the same four as in `spectrum.py`, which must have been
run first with matching values - and run the file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.vq import ClusterError, kmeans2

matplotlib.use("Agg")  # written to a file, never shown
import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set above)
from matplotlib.collections import LineCollection  # noqa: E402

import cluster_maps  # noqa: E402
import graph_lib as gwg  # noqa: E402


# ---- what to run ---- #
#
# These four have to match the run of spectrum.py that produced the eigenvectors:
# CASE and GENERATORS pick the directory and the node set, EMBEDDING picks the
# file, and K has to be no larger than the K it was solved for. A mismatch shows
# up as a missing file rather than as a wrong answer, and the error says which
# constant to change.

CASE = "TYTFS2024_WP2024_V35_transmission"
K = 8                # clusters wanted
GENERATORS = True    # include the generators as nodes of their own
EMBEDDING = "sym"    # "sym" | "rw" | "unnorm"


# k-means++ draws to try. Ten is enough that the best of them is stable across
# runs on these cases, and the whole loop still costs a few milliseconds.
RESTARTS = 10


# ---- the algorithm ---- #


def embed(vectors: np.ndarray, degrees: np.ndarray, embedding: str) -> np.ndarray:
    """The k eigenvectors as one point per node, rescaled for the embedding.

    "sym" divides each row by its own length, which puts every node on the unit
    sphere so that k-means compares directions rather than magnitudes - the
    Ng-Jordan-Weiss step, and the reason the sym embedding tolerates a degree
    skew that the raw eigenvectors do not. The floor on the norm is for a node
    whose row is all zeros, which is not expected here but is a division by zero
    rather than a wrong answer if it happens.

    "rw" recovers the random-walk eigenvectors, which are exactly D^-1/2 times
    the symmetric ones - so Shi-Malik needs no second decomposition, only this.
    """
    if embedding == "sym":
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
    if embedding == "rw":
        return vectors / np.sqrt(degrees)[:, None]
    return vectors


def kmeans(E: np.ndarray, k: int, restarts: int) -> tuple[np.ndarray, float]:
    """Best of `restarts` k-means++ runs, by within-cluster sum of squares.

    `missing="raise"` rather than the default warn: a seed that empties a cluster
    has not produced a k-way partition, so it is discarded and the next seed
    tried, instead of being returned with a warning and k-1 real clusters.
    """
    best_labels, best_inertia = None, np.inf
    for seed in range(restarts):
        try:
            centroids, labels = kmeans2(
                E, k, minit="++", seed=seed, missing="raise")
        except ClusterError:
            continue
        inertia = float(((E - centroids[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_labels, best_inertia = labels, inertia

    if best_labels is None:
        raise RuntimeError(
            f"every one of {restarts} k-means restarts emptied a cluster; "
            f"K = {k} is probably larger than the number of groups the "
            f"embedding actually separates - try the eigengap spectrum.py printed")
    return best_labels, best_inertia


def cut(A: sparse.csr_array, labels: np.ndarray) -> tuple[float, float]:
    """Capacity crossing cluster boundaries, and capacity in total.

    The upper triangle only, so a corridor is counted once rather than from both
    ends. This is the quantity the partition is trying to make small: spectral
    clustering is a relaxation of minimising exactly this, subject to the
    clusters not being trivially unbalanced.

    That second clause is why this is not a score to rank the embeddings by, and
    reading it as "lower is better" gets the answer backwards. The unnormalised
    embedding cuts 3.4% against the sym embedding's 4.8% on the bus-only case,
    and 0.0% against 5.3% once generators are in - because a cut of nothing is
    exactly what you get for lopping five dangling generators off a graph and
    calling the remainder a cluster. Balance is the half of the objective this
    number does not show, so read it beside the cluster sizes, not instead.
    """
    upper = sparse.triu(A, k=1).tocoo()
    crossing = labels[upper.row] != labels[upper.col]
    return float(upper.data[crossing].sum()), float(upper.data.sum())


def corridors(
    branches: pd.DataFrame, node_index: pd.DataFrame, labels: np.ndarray
) -> pd.DataFrame:
    """One row per corridor, with the cluster at each end and whether it is cut.

    A corridor is a bus pair joined by one or more branches, lines and
    transformers alike, rated at their summed `s_nom` - the same merge
    `adjacency()` does, so each row here is one bus-bus entry of A. Generator
    leaf edges are left out: they are not corridors, and a generator never lands
    in a different cluster from its own bus anyway.

    Each row is oriented so `cluster0 <= cluster1`, which makes the cut rows
    group by cluster pair when sorted. Cut corridors come first, largest first.

    A cut transformer leg often joins two buses of the same substation - a 220 kV
    bus and its 3-winding star point, say - so its two ends share a coordinate
    and the map draws it with zero length. This file is where those show up.
    """
    ends = np.sort(branches[["bus0", "bus1"]].to_numpy(str), axis=1)
    grouped = (
        branches.assign(bus0=ends[:, 0], bus1=ends[:, 1])
        .groupby(["bus0", "bus1"], as_index=False)
        .agg(s_nom=("s_nom", "sum"),
             circuits=("name", "size"),
             kind=("kind", lambda s: "+".join(sorted(set(s)))),
             branches=("name", lambda s: " ".join(map(str, s))))
    )
    position = pd.Series(np.arange(len(node_index)), index=node_index["name"])
    i = position[grouped["bus0"]].to_numpy()
    j = position[grouped["bus1"]].to_numpy()
    flip = labels[i] > labels[j]
    i, j = np.where(flip, j, i), np.where(flip, i, j)

    name = node_index["name"].to_numpy()
    station = node_index["station"].to_numpy()
    v_nom = node_index["v_nom"].to_numpy()
    frame = pd.DataFrame({
        "bus0": name[i], "station0": station[i], "v_nom0": v_nom[i], "cluster0": labels[i],
        "bus1": name[j], "station1": station[j], "v_nom1": v_nom[j], "cluster1": labels[j],
        "s_nom": grouped["s_nom"].to_numpy(),
        "circuits": grouped["circuits"].to_numpy(),
        "kind": grouped["kind"].to_numpy(),
        "branches": grouped["branches"].to_numpy(),
        "cut": labels[i] != labels[j],
    })
    return frame.sort_values(["cut", "cluster0", "cluster1", "s_nom"],
                             ascending=[False, True, True, False], ignore_index=True)


# ---- output ---- #


def draw(
    ax: plt.Axes,
    A: sparse.csr_array,
    node_index: pd.DataFrame,
    labels: np.ndarray,
    rings: np.ndarray | None = None,
) -> str:
    """The geographic view onto `ax`, nodes coloured by cluster.

    Returns the "buses placed, corridors cut" line that plot() puts under its
    title. Kept apart from plot() so cluster_maps.py can draw this same map onto
    axes of its own and zoom it to one cluster.

    `rings`, when given, are node indices to circle in red - main() passes the
    substations whose transformer is cut, since that cut joins two co-located
    buses and would otherwise draw as a red line of zero length.

    Adapted from `get_weighted_graph.plot` rather than sharing it: that one
    draws capacity and a sparsity pattern, this one draws a partition, and the
    two want different colours, a legend and a different second panel. Corridors
    are drawn flat grey here so that the colour in the picture is carrying the
    cluster and nothing else.

    Call `gwg.geocode()` on `node_index` before passing it here - this
    function itself never estimates a missing position, it only draws
    whatever `x`/`y` it is given.
    """
    x = node_index["x"].to_numpy(float)
    y = node_index["y"].to_numpy(float)
    located = np.isfinite(x) & np.isfinite(y)

    upper = sparse.triu(A, k=1).tocoo()
    drawable = located[upper.row] & located[upper.col]
    i, j = upper.row[drawable], upper.col[drawable]
    segments = np.stack([np.c_[x[i], y[i]], np.c_[x[j], y[j]]], axis=1)
    # A corridor between two clusters is the thing the partition chose to cut,
    # so it is worth being able to see which ones those are.
    crossing = labels[i] != labels[j]

    k = int(labels.max()) + 1
    palette = plt.get_cmap("tab20" if k > 10 else "tab10")
    colours = np.array([palette(c % palette.N) for c in range(k)])

    ax.add_collection(LineCollection(
        segments[~crossing], linewidths=0.5, colors="#b8bec9", alpha=0.8, zorder=1))
    ax.add_collection(LineCollection(
        segments[crossing], linewidths=1.1, colors="#d1495b", alpha=0.9, zorder=2))
    # Generators take the shared generator marker rather than a bus dot, so a
    # unit on its bus reads as a unit; colour still says which cluster it is in.
    is_gen = (node_index["node_type"] == "generator").to_numpy() \
        if "node_type" in node_index else np.zeros(len(node_index), dtype=bool)
    bus_dot, gen_dot = located & ~is_gen, located & is_gen
    ax.scatter(x[bus_dot], y[bus_dot], s=14, c=colours[labels[bus_dot]],
               zorder=3, linewidths=0)
    if gen_dot.any():
        ax.scatter(x[gen_dot], y[gen_dot], s=24, marker=gwg.GENERATOR_MARKER,
                   c=colours[labels[gen_dot]], zorder=3, linewidths=0)
    ringed = rings[located[rings]] if rings is not None else np.array([], dtype=int)
    if len(ringed):
        ax.scatter(x[ringed], y[ringed], s=160, facecolors="none",
                   edgecolors="#d1495b", linewidths=1.6, zorder=4)

    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(y))))  # rough WGS84 fix
    ax.autoscale_view()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # No per-cluster entries: the colours already tell the clusters apart, and a
    # list of k of them covers the map - worst on the zoomed per-cluster views.
    handles = []
    if gen_dot.any():
        handles.append(plt.Line2D([], [], marker=gwg.GENERATOR_MARKER, linestyle="",
                                  color="#5b6472", label="generator (cluster colour)"))
    handles.append(plt.Line2D([], [], color="#d1495b", label="corridor between clusters"))
    if len(ringed):
        handles.append(plt.Line2D(
            [], [], marker="o", linestyle="", markersize=11, markerfacecolor="none",
            markeredgecolor="#d1495b", markeredgewidth=1.6,
            label=f"cut transformer — {len(ringed)} substations"))
    ax.legend(handles=handles, loc="best", fontsize=9, frameon=False)
    # A generator row may or may not carry a coordinate - gwg.geocode() gives
    # it one, plain node_index never does - but either way it is not a bus,
    # so it is excluded from both sides of the fraction rather than
    # inflating the numerator against a bus-only denominator.
    is_bus = (node_index["node_type"] == "bus").to_numpy() if "node_type" in node_index \
        else np.ones(len(node_index), dtype=bool)
    drawn = int(is_bus.sum())
    bus_located = int((located & is_bus).sum())
    return (f"{bus_located}/{drawn} buses placed, "
            f"{crossing.sum()}/{drawable.sum()} drawn corridors cut")


def plot(
    A: sparse.csr_array,
    node_index: pd.DataFrame,
    labels: np.ndarray,
    path: Path,
    title: str,
    rings: np.ndarray | None = None,
) -> None:
    """The overview map: draw() on a figure of its own, titled and saved."""
    fig, ax = plt.subplots(figsize=(8.5, 9))
    stats = draw(ax, A, node_index, labels, rings)
    ax.set_title(f"{title}\n{stats}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- entry point ---- #


def main() -> None:
    # The same stage folder spectrum.py wrote to, since the eigenvectors read
    # here and the clusters written below are two halves of one thing.
    out = gwg.out_dir(CASE, GENERATORS, "clustering")
    tag = "_gen" if GENERATORS else ""
    stem = f"spectrum_{EMBEDDING}{tag}"
    path = out / f"{stem}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no eigenvectors at {path}\n"
            f"run spectrum.py first with CASE = {CASE!r}, "
            f"EMBEDDING = {EMBEDDING!r}, GENERATORS = {GENERATORS}, K >= {K}")

    stored = np.load(path, allow_pickle=False)
    values, vectors = stored["eigenvalues"], stored["eigenvectors"]
    degrees, embedding = stored["degrees"], str(stored["embedding"])
    # The filename says which embedding this is and so does the payload; if they
    # disagree the file has been renamed, and the payload is the honest one.
    if embedding != EMBEDDING:
        raise ValueError(
            f"{path.name} holds a {embedding!r} decomposition, not {EMBEDDING!r}")
    if vectors.shape[1] < K:
        raise ValueError(
            f"{path.name} holds {vectors.shape[1]} eigenvectors, K = {K} needs "
            f"that many; re-run spectrum.py with K >= {K}")

    E = embed(vectors[:, :K], degrees, embedding)
    labels, inertia = kmeans(E, K, RESTARTS)

    # Rebuilt rather than read back from disk, so this runs on a case whose
    # matrices were never written out - the same call betweenness.py makes.
    buses, branches = gwg.read_case(gwg.PYPSA_DIR / CASE)
    generators = gwg.read_generators(gwg.PYPSA_DIR / CASE) if GENERATORS else None
    A, node_index = gwg.adjacency(buses, branches, generators)

    crossing, total = cut(A, labels)
    sizes = np.bincount(labels, minlength=K)

    frame = node_index.assign(cluster=labels)
    for c in range(K):
        frame[f"e{c + 1}"] = E[:, c]
    csv_path = out / f"clusters_{EMBEDDING}{tag}_k{K}.csv"
    frame.to_csv(csv_path)

    # Only the cut corridors go to the file - the boundary is what it is for. The
    # full frame is kept here so the summary can still say how many of how many.
    links = corridors(branches, node_index, labels)
    corridors_path = out / f"corridors_{EMBEDDING}{tag}_k{K}.csv"
    links[links["cut"]].drop(columns="cut").to_csv(corridors_path, index=False)

    # Estimated for the plot only, same as get_weighted_graph.py's own
    # construction graph - the CSV above keeps the real, ungeocoded x/y.
    pdf_path = out / f"clusters_{EMBEDDING}{tag}_k{K}.pdf"
    # A cut transformer joins two buses of one substation - often a bus and its
    # fictitious 3-winding star point - so its red line usually has zero length
    # and the cut is invisible. It is ringed instead, on its real bus end: a star
    # point's position is only an average of its neighbours, the bus is the site.
    cut_tx = links[links["cut"] & links["kind"].str.contains("transformer")]
    real_end = np.where(cut_tx["bus0"].str.startswith("star:"),
                        cut_tx["bus1"], cut_tx["bus0"])
    position = pd.Series(np.arange(len(node_index)), index=node_index["name"])
    rings = np.unique(position[real_end].to_numpy())
    placed = gwg.geocode(node_index, branches)
    title = f"{CASE}{' +generators' if GENERATORS else ''} — {EMBEDDING}, k = {K}"
    plot(A, placed, labels, pdf_path, title, rings=rings)

    # The same map again, zoomed to one cluster at a time, in a subfolder of its
    # own named after the run so the overview is not buried among them.
    maps_dir = out / f"clusters_{EMBEDDING}{tag}_k{K}"
    maps = cluster_maps.plot_per_cluster(
        lambda ax: draw(ax, A, placed, labels, rings),
        placed, labels, maps_dir, title)

    listing = "\n".join(
        f"    {c:>2}. {sizes[c]:>5} nodes" for c in range(K))
    units = "MVA/MW" if GENERATORS else "MVA"
    print(
        f"{CASE}{' +generators' if GENERATORS else ''}\n"
        f"  nodes      {A.shape[0]}\n"
        f"  embedding  {EMBEDDING!r} on {K} eigenvectors, "
        f"lambda_2..lambda_{K} = {np.array2string(values[1:K], precision=5)}\n"
        f"  k-means    best of {RESTARTS} restarts, inertia {inertia:.4g}\n"
        f"  clusters\n{listing}\n"
        f"  cut        {crossing:,.0f} of {total:,.0f} {units} crosses a "
        f"boundary ({100 * crossing / total:.2f}%)\n"
        f"  corridors  {links['cut'].sum()} of {len(links)} cut, "
        f"{links.loc[links['cut'], 's_nom'].sum():,.0f} MVA\n"
        f"  wrote      {csv_path}\n"
        f"             {corridors_path}\n"
        f"             {pdf_path}\n"
        f"             {maps_dir}  ({len(maps)} cluster maps)"
    )


if __name__ == "__main__":
    main()
