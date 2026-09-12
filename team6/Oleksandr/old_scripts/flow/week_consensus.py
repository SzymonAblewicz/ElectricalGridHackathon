# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa"]
# ///
"""One clustering for the whole week, and the nodes that would not stay in it.

`week_spectrum.py` clusters the flow-weighted graph once per hour, so a week of the synthetic
kit profiles is 168 partitions of the same nodes. This reads that run back and reduces it to
one: the consensus partition, plus, for every node, how much of the week it actually spent in
its consensus zone.

**Why a co-association matrix and not a vote on the labels.** The per-hour labels are aligned
hour to hour by the Hungarian algorithm, and that alignment can swap two zones' names after
they merge and split again - under inverse_flow on WP2033 the Dublin region and the North-West
share a cluster in 47 of the 168 hours and come back out with their colours exchanged. A
per-node majority vote would read that swap as hundreds of nodes changing zone. So nothing
here reads a label *name*. The co-association matrix

    C[i, j] = fraction of the hours in which nodes i and j are in the same cluster

depends only on which nodes are grouped together, so renaming a cluster in any hour leaves it
exactly unchanged. Its heatmap is the consensus picture itself - solid blocks are zones that
hold together all week, mass off the blocks is nodes that move - and the consensus partition
is spectral clustering of C, by the same Laplacian, eigensolver and k-means as the rest of
the pipeline.

**Stability, also without label names.** Each hour's clusters are mapped onto the consensus
zones by majority overlap: an hour-cluster counts as zone z if most of its members are zone-z
nodes. A node's share is the fraction of hours it sits in its own consensus zone under that
mapping, and the rest of its week is counted against the zones it joined. A merge hour
therefore reads the way it happened - "these nodes were with zone X that hour" - not as noise.

Outputs, to data/graphs/<case>/flow/week/, stem consensus_<embedding>_<weighting>_k<k>:

    _matrix.pdf       the heatmap: C ordered by zone, most stable node first within each zone
    _map.pdf          the zones on the map with movers ringed (ring size = share of the week
                      away), beside the same map coloured by stability
    _map/             one zoomed map per zone
    _nodes.csv        every node: zone, share, hours spent in each zone
    _changing.csv     the nodes with share < 1, least stable first
    _corridors.csv    corridors the consensus partition cuts, rated in MVA

Run week_spectrum.py first with the same CASE, WEIGHTING and EMBEDDING, then set the
constants below and run this file.
"""

from __future__ import annotations

import sys
import time
from math import comb
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # graph_lib, spectrum and cluster live one level up

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import cluster as clu  # noqa: E402
import cluster_maps  # noqa: E402
import team6.Oleksandr.Final_scripts.get_flow_graph as gfg  # noqa: E402
import spectrum as spec  # noqa: E402

gwg = gfg.gwg  # whatever graph module get_flow_graph imports, so the two never diverge


# ---- what to run ---- #
#
# The first three name the week_spectrum.py run to read back, and must match one that exists.

CASE = "WP2033_all-island"    # kit cases only: {WP,SV}{2024,2033}_{all-island,north-west}
WEIGHTING = "headroom"        # "headroom" | "loading" | "inverse_flow"
EMBEDDING = "sym"             # "sym" | "rw" | "unnorm"

K = None                      # consensus zones; None = the k the week itself was clustered at
RESTARTS = 30                 # k-means++ draws. 10 found the best partition of the susceptance
                              # graph only ~92% of the time; 30 makes it near-certain
EXTRA = 5                     # eigenpairs beyond K, so the consensus eigengap can be reported

#: The colours cluster.draw gives clusters 0-9, so the heatmap's zone strip matches the map.
ZONE_COLOURS = plt.get_cmap("tab10")


# ---- reading ---- #


def load_week(out: Path) -> dict:
    """The week_spectrum.py run this summarises, checked against the constants above."""
    path = out / f"week_{EMBEDDING}_{WEIGHTING}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"no week run at {path}\nrun week_spectrum.py first with CASE = {CASE!r}, "
            f"WEIGHTING = {WEIGHTING!r}, EMBEDDING = {EMBEDDING!r}")
    week = np.load(path, allow_pickle=False)
    for key, want in (("case", CASE), ("weighting", WEIGHTING), ("embedding", EMBEDDING)):
        if str(week[key]) != want:
            raise ValueError(f"{path.name} holds {key} = {str(week[key])!r}, not {want!r}")
    return {"labels": week["labels"].astype(int), "nodes": week["nodes"].astype(str),
            "snapshots": week["snapshots"].astype(str), "eigenvalues": week["eigenvalues"],
            "k": int(week["k"]), "path": path}


# ---- consensus ---- #


def coassociation(labels: np.ndarray) -> np.ndarray:
    """C[i, j] = share of the hours nodes i and j sit in one cluster; label names never enter.

    Summed as one-hot outer products, H_t H_t^T per hour. Before the final division every
    entry is a whole number of hours, and whole numbers below 2**24 are exact in float32, so
    C does not depend on summation order - renaming the clusters of any hour reproduces it
    bit for bit.
    """
    hours, n = labels.shape
    k = int(labels.max()) + 1
    C = np.zeros((n, n), dtype=np.float32)
    rows = np.arange(n)
    for t in range(hours):
        H = np.zeros((n, k), dtype=np.float32)
        H[rows, labels[t]] = 1.0
        C += H @ H.T
    return C / hours


def consensus_partition(C: np.ndarray, k: int, restarts: int):
    """Spectral clustering of C as an affinity, with the pipeline's own Laplacian and k-means.

    The diagonal is zeroed first: every node shares a cluster with itself every hour, and that
    self-loop only inflates the degree without saying anything about where the node belongs.
    The normalised Laplacian is used whatever EMBEDDING the week ran under - it is the one that
    tolerates the degree skew a consensus matrix has, for the reason spectrum.py gives.
    """
    W = C.astype(float)
    np.fill_diagonal(W, 0.0)
    L, degrees = csgraph.laplacian(W, normed=True, return_diag=True)
    L = (L + L.T) / 2                      # eigsh assumes exact symmetry
    values, vectors = spec.spectrum(sparse.csr_array(L), k + EXTRA)
    labels, inertia = clu.kmeans(clu.embed(vectors[:, :k], degrees, "sym"), k, restarts)
    return labels, values, inertia


def zones_by_hour(labels: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """Each hour's clusters, renamed to the consensus zone most of their members belong to."""
    hours = labels.shape[0]
    k_hour, k_zone = int(labels.max()) + 1, int(zones.max()) + 1
    out = np.empty_like(labels)
    for t in range(hours):
        overlap = np.zeros((k_hour, k_zone), dtype=np.int64)
        np.add.at(overlap, (labels[t], zones), 1)
        out[t] = overlap.argmax(axis=1)[labels[t]]
    return out


def ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand index - agreement of two partitions, 1.0 identical, ~0 unrelated."""
    table = pd.crosstab(a, b).to_numpy()
    pairs = lambda m: sum(comb(int(v), 2) for v in np.ravel(m))  # noqa: E731
    together, rows, cols = pairs(table), pairs(table.sum(1)), pairs(table.sum(0))
    expected = rows * cols / comb(len(a), 2)
    return (together - expected) / ((rows + cols) / 2 - expected)


# ---- output ---- #


def plot_matrix(C: np.ndarray, zones: np.ndarray, share: np.ndarray, path: Path,
                title: str) -> None:
    """The heatmap of C, rows and columns in the same order: zone, then most stable first.

    `interpolation="none"` embeds every node pair as its own pixel, so zooming into the PDF
    recovers each one - the same choice as the week's membership plot.
    """
    order = np.lexsort((-share, zones))
    strip = zones[order]
    k = int(zones.max()) + 1
    sizes = np.bincount(zones, minlength=k)

    fig = plt.figure(figsize=(11.5, 10.8))
    grid = fig.add_gridspec(2, 3, width_ratios=(0.03, 1.0, 0.035), height_ratios=(0.03, 1.0),
                            wspace=0.015, hspace=0.015)
    ax = fig.add_subplot(grid[1, 1])
    top, side, bar = (fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 0]),
                      fig.add_subplot(grid[1, 2]))

    image = ax.imshow(C[np.ix_(order, order)], interpolation="none", cmap="magma",
                      vmin=0.0, vmax=1.0, aspect="auto")
    zone_style = dict(cmap=ZONE_COLOURS, vmin=0, vmax=9, interpolation="none", aspect="auto")
    top.imshow(strip[None, :], **zone_style)
    side.imshow(strip[:, None], **zone_style)
    for edge in np.flatnonzero(np.diff(strip)) + 0.5:
        ax.axhline(edge, color="white", linewidth=0.7)
        ax.axvline(edge, color="white", linewidth=0.7)

    centres = [float(np.flatnonzero(strip == z).mean()) for z in range(k)]
    ax.set_xticks(centres, [f"zone {z}\n{sizes[z]} nodes" for z in range(k)], fontsize=8)
    ax.set_yticks([])
    for strip_ax in (top, side):
        strip_ax.set_xticks([])
        strip_ax.set_yticks([])
    top.set_title(f"{title}\nnodes ordered by consensus zone, most stable first within each "
                  f"(zoom in: one pixel per node pair)", fontsize=10)
    fig.colorbar(image, cax=bar, label="fraction of the week the two nodes share a cluster")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def mark_movers(ax: plt.Axes, placed: pd.DataFrame, share: np.ndarray) -> int:
    """Ring every node that left its zone at some point, sized by how much of the week."""
    x = placed["x"].to_numpy(float)
    y = placed["y"].to_numpy(float)
    moved = (share < 1.0) & np.isfinite(x) & np.isfinite(y)
    # A dark halo drawn *under* the zone dots (cluster.draw puts those at zorder 3), so a zone
    # whose nodes mostly move still shows its colour. Drawn on top instead, the headroom
    # week's 549 rings buried the whole of zone 5 in black.
    ax.scatter(x[moved], y[moved], s=24 + 260 * (1.0 - share[moved]), facecolors="none",
               edgecolors="#333333", linewidths=0.6, zorder=2.5)
    return int(moved.sum())


def plot_map(A, placed: pd.DataFrame, zones: np.ndarray, share: np.ndarray, rings: np.ndarray,
             path: Path, title: str) -> None:
    """Two panels: the consensus zones with movers ringed, and the same map as a heatmap."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(17.5, 9.5))

    stats = clu.draw(left, A, placed, zones, rings)
    ringed = mark_movers(left, placed, share)
    left.add_artist(left.get_legend())               # keep draw()'s zone legend...
    left.legend(handles=[plt.Line2D(                 # ...and add the movers' one beside it
        [], [], marker="o", linestyle="", markersize=9, markerfacecolor="none",
        markeredgecolor="#333333",
        label=f"changed zone at some point - {ringed} nodes\n(ring size = share of week away)")],
        loc="lower left", fontsize=8, frameon=False)
    left.set_title(f"consensus zones\n{stats}", fontsize=10)

    x = placed["x"].to_numpy(float)
    y = placed["y"].to_numpy(float)
    located = np.isfinite(x) & np.isfinite(y)
    upper = sparse.triu(A, k=1).tocoo()
    drawable = located[upper.row] & located[upper.col]
    i, j = upper.row[drawable], upper.col[drawable]
    right.add_collection(LineCollection(np.stack([np.c_[x[i], y[i]], np.c_[x[j], y[j]]], axis=1),
                                        linewidths=0.4, colors="#c9ced6", zorder=1))
    order = np.argsort(-share, kind="stable")        # stable first, so movers are drawn on top
    order = order[located[order]]
    low = min(np.floor(share.min() * 10) / 10, 0.9)
    dots = right.scatter(x[order], y[order], c=share[order], cmap="YlOrRd_r", vmin=low, vmax=1.0,
                         s=10 + 60 * (1.0 - share[order]), linewidths=0, zorder=2)
    fig.colorbar(dots, ax=right, fraction=0.035, pad=0.02,
                 label="share of the week the node spends in its consensus zone")
    right.set_aspect(left.get_aspect())
    right.set_xlim(left.get_xlim())
    right.set_ylim(left.get_ylim())
    right.set_xlabel("longitude")
    right.set_ylabel("latitude")
    right.set_title("stability heatmap - pale: never left its zone,\n"
                    "red: spent much of the week in another zone", fontsize=10)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- entry point ---- #


def main() -> None:
    started = time.perf_counter()
    gfg.DATASET = "kit"                     # only the kit networks have a week in them
    case_dir = gfg.case_root() / CASE
    out = gfg.out_dir(CASE, "week")
    week = load_week(out)
    labels = week["labels"]
    hours = labels.shape[0]
    k = week["k"] if K is None else int(K)

    # The graph is rebuilt, not read back, so the map has its topology and the corridor list
    # its MVA ratings. Its node order has to be the week's, or every label lands on the wrong
    # node - so that is checked rather than assumed.
    buses, branches = gwg.read_case(case_dir)
    generators = gwg.read_generators(case_dir)
    A, node_index = gwg.adjacency(buses, branches, generators)
    names = node_index["name"].to_numpy().astype(str)
    if len(names) != len(week["nodes"]) or (names != week["nodes"]).any():
        raise ValueError(
            f"the nodes rebuilt from {CASE} are not in the order {week['path'].name} was "
            f"written in - the case files changed since week_spectrum.py ran; re-run it first")

    C = coassociation(labels)
    zones, values, inertia = consensus_partition(C, k, RESTARTS)
    _, suggested = spec.eigengap(values)
    by_hour = zones_by_hour(labels, zones)
    share = (by_hour == zones[None, :]).mean(axis=0)
    hours_in = np.stack([(by_hour == z).sum(axis=0) for z in range(k)], axis=1)
    away = hours_in.copy()
    away[np.arange(len(zones)), zones] = -1
    elsewhere = np.where(share < 1.0, away.argmax(axis=1), -1)

    links = clu.corridors(branches, node_index, zones)
    rings = gfg.cut_transformer_rings(links, node_index)
    crossing, total = clu.cut(A, zones)
    placed = gwg.geocode(node_index, branches)

    stem = f"consensus_{EMBEDDING}_{WEIGHTING}_k{k}"
    title = f"{CASE} +generators - consensus of {hours} hourly {WEIGHTING} clusterings, k = {k}"
    plot_matrix(C, zones, share, out / f"{stem}_matrix.pdf", title)
    plot_map(A, placed, zones, share, rings, out / f"{stem}_map.pdf", title)

    def draw_zoomed(ax: plt.Axes) -> str:
        stats = clu.draw(ax, A, placed, zones, rings)
        mark_movers(ax, placed, share)
        return stats

    maps = cluster_maps.plot_per_cluster(draw_zoomed, placed, zones, out / f"{stem}_map", title)

    table = node_index.assign(zone=zones, share=share, most_visited_other=elsewhere)
    for z in range(k):
        table[f"hours_zone_{z}"] = hours_in[:, z]
    table.to_csv(out / f"{stem}_nodes.csv")
    table[table["share"] < 1.0].sort_values("share").to_csv(out / f"{stem}_changing.csv")
    links[links["cut"]].drop(columns="cut").to_csv(out / f"{stem}_corridors.csv", index=False)

    robust = int(np.argmax(week["eigenvalues"][:, 1]))
    sizes = np.bincount(zones, minlength=k)
    # Real substations only: a 3-winding star point is a modelling device between a bus and
    # its windings, and it leaves its zone whenever either side does. It stays in the CSVs.
    is_bus = ((node_index["node_type"] == "bus")
              & ~node_index["name"].astype(str).str.startswith("star:")).to_numpy()
    least = table[is_bus & (share < 1.0)].sort_values("share").head(8)
    listing = "\n".join(
        f"               {str(row['station']):<26} bus {row['name']:<12} zone {row['zone']}: "
        f"{row['share']:.0%} of the week, else mostly zone {row['most_visited_other']}"
        for _, row in least.iterrows()) or "               (none)"
    elapsed = time.perf_counter() - started
    print(
        f"{CASE} +generators - week consensus of {WEIGHTING}  [{elapsed:.1f}s]\n"
        f"  read       {week['path'].name}: {hours} hours x {len(zones)} nodes, "
        f"clustered at k = {week['k']}\n"
        f"  consensus  k = {k}, zone sizes {sizes.tolist()}, best of {RESTARTS} k-means "
        f"restarts;\n"
        f"             the consensus matrix's own eigengap suggests k = {suggested}\n"
        f"  stability  {int((share == 1).sum())} nodes ({(share == 1).mean():.1%}) never leave "
        f"their zone; {int((share < 1).sum())} leave it at least once,\n"
        f"             {int((share < 0.9).sum())} spend over 10% of the week elsewhere, "
        f"{int((share < 0.5).sum())} over half of it\n"
        f"  least stable buses\n{listing}\n"
        f"  vs hourly  ARI {ari(zones, labels[robust]):.3f} against the week's most robust hour "
        f"({week['snapshots'][robust]}) - 1.0 would mean the\n"
        f"             consensus is just that hour\n"
        f"  cut        {int(links['cut'].sum())} of {len(links)} corridors, "
        f"{links.loc[links['cut'], 's_nom'].sum():,.0f} MVA "
        f"({100 * crossing / total:.2f}% of capacity crosses a boundary)\n"
        f"  wrote      {out}\n"
        f"               {stem}_matrix.pdf   <- the heatmap\n"
        f"               {stem}_map.pdf      <- zones and stability on the map\n"
        f"               {stem}_nodes.csv, {stem}_changing.csv, {stem}_corridors.csv\n"
        f"               {stem}_map/   ({len(maps)} per-zone zooms)"
    )


if __name__ == "__main__":
    main()
