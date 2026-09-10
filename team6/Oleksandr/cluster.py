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


# ---- output ---- #


def plot(
    A: sparse.csr_array,
    node_index: pd.DataFrame,
    labels: np.ndarray,
    path: Path,
    title: str,
) -> None:
    """The geographic view, nodes coloured by cluster.

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

    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.add_collection(LineCollection(
        segments[~crossing], linewidths=0.5, colors="#b8bec9", alpha=0.8, zorder=1))
    ax.add_collection(LineCollection(
        segments[crossing], linewidths=1.1, colors="#d1495b", alpha=0.9, zorder=2))
    ax.scatter(x[located], y[located], s=14, c=colours[labels[located]],
               zorder=3, linewidths=0)

    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(y))))  # rough WGS84 fix
    ax.autoscale_view()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    sizes = np.bincount(labels, minlength=k)
    ax.legend(
        handles=[plt.Line2D([], [], marker="o", linestyle="", color=colours[c],
                            label=f"cluster {c} — {sizes[c]} nodes")
                 for c in range(k)]
        + [plt.Line2D([], [], color="#d1495b", label="corridor between clusters")],
        loc="best", fontsize=9, frameon=False)
    # A generator row may or may not carry a coordinate - gwg.geocode() gives
    # it one, plain node_index never does - but either way it is not a bus,
    # so it is excluded from both sides of the fraction rather than
    # inflating the numerator against a bus-only denominator.
    is_bus = (node_index["node_type"] == "bus").to_numpy() if "node_type" in node_index \
        else np.ones(len(node_index), dtype=bool)
    drawn = int(is_bus.sum())
    bus_located = int((located & is_bus).sum())
    ax.set_title(f"{title}\n{bus_located}/{drawn} buses placed, "
                 f"{crossing.sum()}/{drawable.sum()} drawn corridors cut")

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

    # Estimated for the plot only, same as get_weighted_graph.py's own
    # construction graph - the CSV above keeps the real, ungeocoded x/y.
    pdf_path = out / f"clusters_{EMBEDDING}{tag}_k{K}.pdf"
    plot(A, gwg.geocode(node_index, branches), labels, pdf_path,
         f"{CASE}{' +generators' if GENERATORS else ''} — {EMBEDDING}, k = {K}")

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
        f"  wrote      {csv_path}\n"
        f"             {pdf_path}"
    )


if __name__ == "__main__":
    main()
