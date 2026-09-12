# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib"]
# ///
"""One zoomed map per cluster, cut from the overview `cluster.py` draws.

The overview puts all k clusters on one map of the island, which is the right
picture for where the boundaries run and the wrong one for what is inside a
cluster: Dublin's two clusters are a knot of well over a hundred nodes within
a few kilometres of each other. This writes the same map once per cluster, each
with its limits set to that cluster's extent, into a subfolder of its own so the
overview is not buried among them.

The drawing itself is not here. `cluster.py` passes its `draw(ax)` in, so every
zoomed map is the overview with different limits - same colours, same cut
corridors, same rings on cut transformers - and there is one implementation of
the picture. Passing it in rather than importing it also keeps the imports
one-way: `cluster.py` imports this module, this module imports nothing from it.

The limits are the cluster's own minimum and maximum longitude and latitude,
over its nodes that have a position, widened on each side by a fraction of that
span - an eighth by default, set per axis by MARGIN_LON and MARGIN_LAT below.
Neighbouring clusters are still drawn, so a corridor leaving this one can be
seen going somewhere.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # written to a file, never shown
import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set above)

# Margin added on each side of a cluster's extent, as a fraction of the span
# on that axis: 1/8 of the longitude span left and right, 1/8 of the latitude
# span above and below.
MARGIN_LON = 1 / 8
MARGIN_LAT = 1 / 8


def bounds(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """(lon_min, lon_max, lat_min, lat_max) of the points, widened by the margins."""
    dx = (x.max() - x.min()) * MARGIN_LON
    dy = (y.max() - y.min()) * MARGIN_LAT
    return x.min() - dx, x.max() + dx, y.min() - dy, y.max() + dy


def plot_per_cluster(
    draw: Callable[[plt.Axes], object],
    node_index: pd.DataFrame,
    labels: np.ndarray,
    folder: Path,
    title: str,
) -> list[Path]:
    """Write `folder/cluster_<c>.pdf` for every cluster; return the paths.

    `draw` puts the whole map onto the axes it is given; this only frames it.
    `node_index` must be the same geocoded frame `draw` plots from, so the
    limits are taken from the positions actually drawn.
    """
    folder.mkdir(parents=True, exist_ok=True)
    x = node_index["x"].to_numpy(float)
    y = node_index["y"].to_numpy(float)
    located = np.isfinite(x) & np.isfinite(y)

    paths = []
    for c in range(int(labels.max()) + 1):
        member = located & (labels == c)
        if not member.any():
            continue  # nothing in this cluster has a position to frame
        lon_min, lon_max, lat_min, lat_max = bounds(x[member], y[member])

        fig, ax = plt.subplots(figsize=(8.5, 9))
        draw(ax)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_title(f"{title}\ncluster {c} — {int((labels == c).sum())} nodes")

        path = folder / f"cluster_{c}.pdf"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths
