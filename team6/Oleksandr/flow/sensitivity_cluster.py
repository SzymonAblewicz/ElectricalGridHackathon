# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib"]
# ///
"""k-means on the generators' sensitivity vectors: generators that move the same branches together.

Reads `data/generator_sensitivity_WP2033.csv` (written by generator_sensitivity.py): one row per
generator, one column per branch (980 lines and transformers), each entry the signed PTDF - MW on
that branch per MW of this generator. Each row is taken as-is as a point in 980 dimensions and
k-means groups the points. No rescaling: every column is already in the same unit, MW per MW.

A generator has no sensitivity of its own, it inherits its bus's, so generators on one bus have
identical rows and always land in the same cluster. 716 generators give only 263 distinct points.

The import/export units on the AC-isolated buses are left out: Szymon's ptdf README says their
columns are near-degenerate and should be ignored.

k-means is `cluster.kmeans` - the same best-of-RESTARTS k-means++ that clusters the Laplacian
eigenvectors.

How many clusters. get_flow_graph.py computes K + EXTRA eigenpairs and lets the eigengap suggest
k. There is no Laplacian here, so no eigenvalues to take a gap of; the counterpart is to run
k-means at every k from 2 to K + EXTRA and score each partition by its mean silhouette -
(b - a) / max(a, b) per generator, a its mean distance to its own cluster and b to the nearest
other one. The highest mean silhouette is the suggested k. As in get_flow_graph.py it is only
reported: the CSV is written at the configured K.
"""

from __future__ import annotations

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # cluster.py lives one level up

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist, squareform  # noqa: E402

matplotlib.use("Agg")  # written to a file, never shown
import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set above)
from matplotlib.collections import LineCollection  # noqa: E402
from scipy import sparse  # noqa: E402

import graph_lib as gwg  # noqa: E402
from cluster import kmeans  # noqa: E402

DATA = FOLDER.parent / "data"
SOURCE = DATA / "generator_sensitivity_WP2033.csv"
# The network the sensitivities were computed on (generator_sensitivity.py), drawn for the map.
NETWORK = FOLDER.parents[1] / "Szymon" / "participant-kit new" / "networks" / "WP2033_all-island"

K = 9           # clusters wanted
EXTRA = 0      # k is also tried up to K + EXTRA, so the best k past K is visible
RESTARTS = 10   # k-means++ draws; the lowest within-cluster sum of squares is kept

# AC-isolated behind DC links, same set as generator_sensitivity.py.
ISOLATED = {"86221", "GB_EWIC", "GB_GREENLINK"}


def silhouette(D: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette over all points, from the full distance matrix D.

    A point alone in its cluster scores 0, the usual convention: it has no a to compare with.
    """
    k = int(labels.max()) + 1
    onehot = np.eye(k)[labels]                       # points x clusters
    sizes = onehot.sum(axis=0)
    mean_to = D @ onehot                              # summed distance to each cluster
    own = sizes[labels]
    a = mean_to[np.arange(len(labels)), labels] / np.maximum(own - 1, 1)
    mean_to = mean_to / sizes
    mean_to[np.arange(len(labels)), labels] = np.inf
    b = mean_to.min(axis=1)
    s = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return float(s.mean())


def plot_scan(ks: np.ndarray, scores: np.ndarray, inertias: np.ndarray, best: int,
              path: Path, title: str) -> None:
    """Silhouette and inertia against k, with the best k and the configured K marked."""
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    top.plot(ks, scores, "-o", color="#2c6fbb", markersize=5)
    top.set_ylabel("mean silhouette")
    bottom.plot(ks, inertias, "-o", color="#5b6472", markersize=5)
    bottom.set_ylabel("inertia")
    bottom.set_xlabel("k")
    for ax in (top, bottom):
        ax.axvline(best, color="#d1495b", alpha=0.6, linewidth=6,
                   label=f"best silhouette — suggests k = {best}")
        ax.axvline(K, color="#3f8f4f", linestyle="--", linewidth=1.2, label=f"K = {K} (configured)")
    top.legend(loc="best", fontsize=9, frameon=False)
    top.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_clusters(clusters: pd.DataFrame, path: Path, title: str) -> None:
    """The grid from graph_lib, corridors and buses in grey, each generator in its cluster's colour.

    The graph and every position come from graph_lib - `adjacency()` with the generators as
    nodes, `geocode()` to place them beside their bus - so the map reads like the other maps in
    this folder. Only the colouring is new: the clusters are of generators, so buses carry none.
    """
    buses, branches = gwg.read_case(NETWORK)
    generators = gwg.read_generators(NETWORK)
    generators = generators[generators["name"].isin(clusters["generator"])]
    A, node_index = gwg.adjacency(buses, branches, generators)
    placed = gwg.geocode(node_index, branches)

    x = placed["x"].to_numpy(float)
    y = placed["y"].to_numpy(float)
    located = np.isfinite(x) & np.isfinite(y)
    is_gen = (placed["node_type"] == "generator").to_numpy()
    label = placed["name"].map(clusters.set_index("generator")["cluster"]).to_numpy()

    # Corridors only: the generator leaf edges would draw as a tangle of short spokes.
    upper = sparse.triu(A, k=1).tocoo()
    keep = located[upper.row] & located[upper.col] & ~is_gen[upper.row] & ~is_gen[upper.col]
    i, j = upper.row[keep], upper.col[keep]
    segments = np.stack([np.c_[x[i], y[i]], np.c_[x[j], y[j]]], axis=1)

    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.add_collection(LineCollection(segments, linewidths=0.5, colors="#b8bec9", alpha=0.8, zorder=1))
    ax.scatter(x[located & ~is_gen], y[located & ~is_gen], s=6, c="#8a919c", zorder=2, linewidths=0)
    # Grey is reserved for the buses, which are not clustered, so tab10's grey (index 7) is
    # skipped and tab20b's colours follow once tab10 runs out.
    palette = [c for i, c in enumerate(plt.get_cmap("tab10").colors) if i != 7] \
        + list(plt.get_cmap("tab20b").colors)
    for c in range(K):
        mask = located & is_gen & (label == c)
        ax.scatter(x[mask], y[mask], s=26, marker=gwg.GENERATOR_MARKER, color=palette[c],
                   zorder=3, linewidths=0)
    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(y))))  # rough WGS84 fix, as graph_lib.plot
    ax.autoscale_view()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{title}\n{int((located & is_gen).sum())}/{int(is_gen.sum())} generators placed")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    table = pd.read_csv(SOURCE, dtype={"bus": str})
    table = table[~table["bus"].isin(ISOLATED)].reset_index(drop=True)
    meta = table[["generator", "bus", "carrier", "p_nom_MW"]]
    X = table.drop(columns=meta.columns).to_numpy(float)

    D = squareform(pdist(X))
    ks = np.arange(2, K + EXTRA + 1)
    runs = {k: kmeans(X, k, RESTARTS) for k in ks}
    scores = np.array([silhouette(D, runs[k][0]) for k in ks])
    inertias = np.array([runs[k][1] for k in ks])
    best = int(ks[np.argmax(scores)])
    scan_path = DATA / "generator_clusters_WP2033_kscan.pdf"
    plot_scan(ks, scores, inertias, best, scan_path,
              f"WP2033 generator sensitivity — k-means, k = 2..{K + EXTRA}")

    labels, inertia = runs[K]
    out = meta.assign(cluster=labels)
    path = DATA / f"generator_clusters_WP2033_k{K}.csv"
    out.to_csv(path, index=False)
    map_path = DATA / f"generator_clusters_WP2033_k{K}.pdf"
    draw_clusters(out, map_path, f"WP2033 generators by sensitivity — k-means, k = {K}")

    summary = out.groupby("cluster").agg(
        generators=("generator", "size"), buses=("bus", "nunique"), MW=("p_nom_MW", "sum"),
        carriers=("carrier", lambda s: ", ".join(f"{c} {n}" for c, n in s.value_counts().items())))
    listing = "\n".join(
        f"    k = {k:>2}   silhouette {s:.4f}   inertia {i:>9.4g}"
        + ("  <- best" if k == best else "") + ("  <- K" if k == K else "")
        for k, s, i in zip(ks, scores, inertias))
    print(f"{len(out)} generators ({len(ISOLATED)} isolated buses dropped) x {X.shape[1]} branches\n"
          f"scan     k = 2..{K + EXTRA}, silhouette suggests k = {best}\n{listing}\n"
          f"k-means  K = {K}, best of {RESTARTS} restarts, inertia {inertia:.4g}\n")
    print(summary.to_string(float_format="%.0f"))
    print(f"\nwrote {path}\n      {map_path}\n      {scan_path}")


if __name__ == "__main__":
    main()
