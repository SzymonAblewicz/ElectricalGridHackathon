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
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

matplotlib.use("Agg")  # written to a file, never shown
import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set above)
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

import graph_lib as gwg  # noqa: E402


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
COLOUR_GAMMA = 0.4   # map shading: <1 stretches the crowded low end, see plot()


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


# ---- output ---- #


def plot(
    A: sparse.csr_array,
    node_index: pd.DataFrame,
    score: np.ndarray,
    path: Path,
    title: str,
    gamma: float = COLOUR_GAMMA,
) -> None:
    """The geographic view, every node shaded by its betweenness.

    Adapted from `get_weighted_graph.plot` rather than sharing it, the same way
    `cluster.py`'s is: that one draws capacity and a sparsity pattern, this one
    draws a ranking. Corridors go flat grey so the only colour in the picture is
    the score, and buses stay circles while generators stay triangles, so which
    node set a marker belongs to is carried by its shape and the colour is left
    to mean exactly one thing. Brighter is higher throughout.

    The shading is deliberately *not* linear in the score, and that is the one
    thing to know before reading the picture. Betweenness on a grid is extremely
    top-heavy - on the WP2024 transmission case 91% of buses sit below a tenth
    of the maximum - so a linear ramp puts nearly every node inside the darkest
    few percent of the colormap and produces one bright dot on a black field.
    `PowerNorm(gamma)` with gamma below 1 stretches that crowded low end while
    staying strictly monotonic in the score, so a brighter node is still always
    a higher-scoring node; only the spacing changes, never the order. The
    colourbar carries the real values at their true positions on that scale, so
    the compression is on show rather than hidden. Set `COLOUR_GAMMA` to 1 to
    see the honest linear version.

    Every generator comes out at the very bottom of the scale, and that is
    arithmetic rather than an artefact: a generator is a degree-1 leaf, no
    shortest path between two other nodes can route through it, so its
    betweenness is exactly zero by construction.

    Call `gwg.geocode()` on `node_index` before passing it here - this function
    never estimates a missing position, it only draws the x/y it is given.
    """
    x = node_index["x"].to_numpy(float)
    y = node_index["y"].to_numpy(float)
    located = np.isfinite(x) & np.isfinite(y)

    # Upper triangle only: one segment per corridor, not two.
    upper = sparse.triu(A, k=1).tocoo()
    drawable = located[upper.row] & located[upper.col]
    i, j = upper.row[drawable], upper.col[drawable]
    segments = np.stack([np.c_[x[i], y[i]], np.c_[x[j], y[j]]], axis=1)

    is_bus = (node_index["node_type"] == "bus").to_numpy() if "node_type" in node_index \
        else np.ones(len(node_index), dtype=bool)
    bus_drawn = located & is_bus
    gen_drawn = located & ~is_bus

    # An all-zero score would make vmax 0 and the norm degenerate; 1.0 keeps the
    # colourbar meaningful and every marker at the bottom, which is the truth.
    norm = PowerNorm(gamma, vmin=0.0, vmax=float(score.max()) or 1.0)

    fig, ax = plt.subplots(figsize=(8.5, 9))
    ax.add_collection(LineCollection(
        segments, linewidths=0.5, colors="#b8bec9", alpha=0.8, zorder=1))

    # A thin dark edge stops the brightest markers - the whole point of the
    # picture - from washing out against the white page.
    edge = dict(edgecolors="#333333", linewidths=0.25)
    dots = ax.scatter(x[bus_drawn], y[bus_drawn], s=26, c=score[bus_drawn],
                      cmap="viridis", norm=norm, marker="o", zorder=3, **edge)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color="#5b6472",
                          label=f"bus — {int(bus_drawn.sum())} drawn")]
    if gen_drawn.any():
        ax.scatter(x[gen_drawn], y[gen_drawn], s=24, c=score[gen_drawn],
                   cmap="viridis", norm=norm, marker=gwg.GENERATOR_MARKER, zorder=2,
                   **edge)
        handles.append(plt.Line2D([], [], marker=gwg.GENERATOR_MARKER, linestyle="",
                                  color="#5b6472",
                                  label=f"generator — {int(gen_drawn.sum())} drawn, "
                                        "all exactly 0"))
    ax.legend(handles=handles, loc="best", fontsize=9, frameon=False)

    bar = fig.colorbar(dots, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("normalised betweenness")

    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(y))))  # rough WGS84 fix
    ax.autoscale_view()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    buses = int(is_bus.sum())
    ax.set_title(
        f"{title}\n{int(bus_drawn.sum())}/{buses} buses placed, "
        f"{drawable.sum()}/{upper.nnz} corridors drawn — "
        f"colour ∝ score^{gamma:g}, not linear")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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

    # geocode() fills in the positions the plot needs - imputed for a bus with
    # no x/y, jittered beside its bus for a generator - and returns a copy, so
    # the CSV written above keeps the real ungeocoded coordinates. `normalised`
    # is still in node_index row order here; `frame` above was sorted by rank,
    # which is why the scores are passed straight rather than off the frame.
    plot_index = gwg.geocode(node_index, branches)
    graph_path = out / f"graph_betweenness{tag}.pdf"
    plot(A, plot_index, normalised, graph_path,
         f"{CASE}{' +generators' if GENERATORS else ''} — betweenness on 1/capacity")

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
        f"  wrote      {path}\n"
        f"             {graph_path}"
    )


if __name__ == "__main__":
    main()
