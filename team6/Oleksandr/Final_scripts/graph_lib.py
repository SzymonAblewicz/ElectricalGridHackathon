# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib"]
# ///
"""Shared graph construction, geocoding and plotting for one PyPSA case.

Everything here is used by more than one script in this folder -
`get_weighted_graph.py`, `betweenness.py`, `spectrum.py`, `cluster.py` and
the two graph scripts in `flow/` - so it lives in its own module rather than
inside any one of them. Each of those keeps its own "what to run" constants
and `main()`; this file has neither, it is a library, not a driver.

Reading: `read_case()` and `read_generators()` load one case's CSVs.
Construction: `adjacency()` builds the sparse capacity-weighted graph,
`matrices()` its Laplacians, `reciprocal()` the distance-reading version.
Geocoding: `geocode()` is what every script calls before drawing
`node_index` on a map - see its docstring for what it does and why.
Output: `out_dir()` and `save()` write the pipeline's folder layout;
`plot()` draws the geographic view beside the sparsity pattern of `A`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csgraph

matplotlib.use("Agg")  # written to a file, never shown
import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set above)
from matplotlib.collections import LineCollection  # noqa: E402

FOLDER = Path(__file__).resolve().parent
# FOLDER is <repo>/team6/Oleksandr/Final_scripts, so the repo root is three levels up.
PYPSA_DIR = FOLDER.parent.parent.parent / "grid_TF_Wind" / "data" / "pypsa"
#: Every output this folder writes lives under here - graphs, congestion, sensitivity.
DATA_DIR = FOLDER / "data"
OUT_DIR = DATA_DIR / "graphs"

BUS_COLUMNS = ["name", "v_nom", "x", "y", "jurisdiction", "station"]
RANDOM_RADIUS = 0.03

# The shape every map in this folder draws a generator with, so a generator
# reads as a generator on the construction graph, the betweenness map and the
# cluster maps alike, whatever colour each of them gives it. Any matplotlib
# marker code works: "^" triangle, "s" square, "D" diamond, "*" star.
GENERATOR_MARKER = "^"

# ---- reading ---- #


def read_case(case_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the buses and the branches that join them.

    Bus names are read as strings on purpose: most are numeric PSS/E ids, but
    the 3-winding transformers contribute `star:...` fictitious mid-points, and
    letting pandas infer the dtype per file would key the branch tables on ints
    and the bus table on objects.
    """
    buses = pd.read_csv(case_dir / "buses.csv", dtype={"name": str})

    ends = {"bus0": str, "bus1": str}
    lines = pd.read_csv(case_dir / "lines.csv", dtype=ends).assign(kind="line")
    transformers = pd.read_csv(
        case_dir / "transformers.csv", dtype=ends
    ).assign(kind="transformer")

    columns = ["name", "bus0", "bus1", "s_nom", "kind"]
    branches = pd.concat(
        [lines[columns], transformers[columns]], ignore_index=True
    )
    return buses, branches


def read_generators(case_dir: Path) -> pd.DataFrame:
    """The generators and the bus each one sits on.

    Names are read as strings for the same reason bus names are: a generator id
    is the PSS/E bus number with a unit suffix (`10271-1`), which is not
    numeric, while the `bus` column it points at is.
    """
    generators = pd.read_csv(
        case_dir / "generators.csv", dtype={"name": str, "bus": str}
    )
    return generators[["name", "bus", "p_nom", "carrier"]]


# ---- graph construction ---- #


def adjacency(
    buses: pd.DataFrame,
    branches: pd.DataFrame,
    generators: pd.DataFrame | None = None,
) -> tuple[sparse.csr_array, pd.DataFrame]:
    """Symmetric capacity-weighted adjacency over the buses that carry a branch.

    With `generators`, each generator becomes a node of its own, joined to its
    bus by a single edge rated `p_nom`. Generator rows are appended *after* the
    buses, so a bus keeps the index it had without the flag and the leading
    block of the matrix is exactly the bus-only graph.

    Returns the matrix and the index frame that names its rows.
    """
    connected = set(branches["bus0"]) | set(branches["bus1"])
    kept = buses[buses["name"].isin(connected)]

    node_index = kept[BUS_COLUMNS].reset_index(names="csv_row")
    node_index.index.name = "index"

    names = pd.Index(node_index["name"])
    i = names.get_indexer(branches["bus0"])
    j = names.get_indexer(branches["bus1"])
    if (i < 0).any() or (j < 0).any():
        missing = set(branches["bus0"][i < 0]) | set(branches["bus1"][j < 0])
        raise ValueError(f"branch endpoints absent from buses.csv: {sorted(missing)}")

    w = branches["s_nom"].to_numpy(float)

    if generators is not None:
        # A generator id is its bus number plus a unit suffix, so a collision
        # with a bus name should be impossible - but it would silently fold a
        # generator into a bus rather than fail, so it is worth refusing.
        clash = set(generators["name"]) & set(names)
        if clash:
            raise ValueError(
                f"generator names collide with bus names: {sorted(clash)}")

        # A generator on a bus that was dropped as branchless goes with it. In
        # these cases those buses are exactly the DC interconnector terminals,
        # and the units on them are the import/export pair - keeping them would
        # add an isolated two-node island per interconnector, on the far side of
        # a link this graph does not model.
        host = names.get_indexer(generators["bus"])
        generators = generators[host >= 0].reset_index(drop=True)
        host = host[host >= 0]

        node_index = pd.concat(
            [node_index.assign(node_type="bus"),
             generators.reset_index(names="csv_row").assign(node_type="generator")],
            ignore_index=True)
        node_index.index.name = "index"

        # Each generator is its own node, appended after the last bus, so two
        # generators on one bus stay two edges rather than summing the way
        # parallel circuits on one corridor do.
        i = np.r_[i, len(names) + np.arange(len(generators))]
        j = np.r_[j, host]
        w = np.r_[w, generators["p_nom"].to_numpy(float)]

    n = len(node_index)
    # Both triangles, so A comes out symmetric. tocsr() sums duplicate (i, j)
    # entries, which is exactly "parallel circuits on one corridor add their
    # MVA" - no groupby needed.
    A = sparse.coo_array(
        (np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)
    ).tocsr()
    return A, node_index


# ---- geocoding for plots ---- #


def impute_coordinates(buses: pd.DataFrame, branches: pd.DataFrame) -> pd.DataFrame:
    """Fill a bus's missing x/y with the average position of its neighbours.

    One pass, over the lines and transformers that actually join buses -
    generators are never a source since they carry no coordinate of their
    own. A bus is only moved if at least one directly connected bus already
    has a coordinate; a bus whose neighbours are themselves all uncoordinated
    (e.g. two adjacent transformer star points) is left exactly as it was,
    same as plot() already leaves any NaN-coordinate bus undrawn.
    """
    buses = buses.copy()
    x = buses.set_index("name")["x"]
    y = buses.set_index("name")["y"]
    missing = buses.loc[buses["x"].isna() | buses["y"].isna(), "name"]

    neighbors: dict[str, list[str]] = {name: [] for name in missing}
    for bus0, bus1 in zip(branches["bus0"], branches["bus1"]):
        if bus0 in neighbors:
            neighbors[bus0].append(bus1)
        if bus1 in neighbors:
            neighbors[bus1].append(bus0)

    imputed_x, imputed_y = {}, {}
    for name, neighs in neighbors.items():
        nx = x.reindex(neighs).dropna()
        ny = y.reindex(neighs).dropna()
        if len(nx) and len(ny):
            imputed_x[name] = nx.mean()
            imputed_y[name] = ny.mean()

    buses["x"] = buses["x"].fillna(buses["name"].map(imputed_x))
    buses["y"] = buses["y"].fillna(buses["name"].map(imputed_y))
    print(f"imputed coordinates for {len(imputed_x)}/{len(missing)} buses missing x/y, "
          f"from directly connected neighbours; {len(missing) - len(imputed_x)} left "
          f"unplaced (no neighbour has a coordinate)")
    return buses


def jitter_generator_coordinates(
    node_index: pd.DataFrame, radius: float = RANDOM_RADIUS, seed: int = 0
) -> pd.DataFrame:
    """Place each generator near the bus it sits on.

    A generator is a leaf with exactly one neighbour, so the bus-to-bus
    averaging impute_coordinates() does makes no sense here - there is
    nothing to average. Its bus's own point is the reasonable estimate, but
    several generators commonly share one bus, so drawing them all at that
    exact point would stack them on one pixel: each instead gets a small
    random offset, uniform in a disk of `radius` degrees (~1.5 km at Irish
    latitudes by default - visibly beside the bus, not somewhere else on the
    map). The seed is fixed so the plot is reproducible run to run.
    """
    node_index = node_index.copy()
    is_gen = node_index["node_type"] == "generator"
    bus_xy = node_index.loc[~is_gen].set_index("name")[["x", "y"]]

    host = node_index.loc[is_gen, "bus"]
    host_xy = bus_xy.reindex(host.to_numpy())
    located = host_xy["x"].notna().to_numpy() & host_xy["y"].notna().to_numpy()

    rng = np.random.default_rng(seed)
    n = int(located.sum())
    angle = rng.uniform(0, 2 * np.pi, n)
    r = radius * np.sqrt(rng.uniform(0, 1, n))  # uniform in area, not just angle

    idx = node_index.index[is_gen][located]
    node_index.loc[idx, "x"] = host_xy["x"].to_numpy()[located] + r * np.cos(angle)
    node_index.loc[idx, "y"] = host_xy["y"].to_numpy()[located] + r * np.sin(angle)

    n_gen = int(is_gen.sum())
    print(f"placed {n}/{n_gen} generators beside their bus (within {radius:g} deg); "
          f"{n_gen - n} left uncoordinated (host bus itself has no position)")
    return node_index


def geocode(node_index: pd.DataFrame, branches: pd.DataFrame) -> pd.DataFrame:
    """Every node's best available position for plotting a map.

    Real where `buses.csv` has it; else the average of directly connected
    neighbours (`impute_coordinates()`, run twice so a chain of uncoordinated
    buses resolves one hop at a time); else, for a generator, jittered beside
    its bus (`jitter_generator_coordinates()`), since a generator has no
    coordinate of its own and only one neighbour to average.

    This is the one place every script that draws `node_index` on a map
    should turn to first, so a bus's estimated position - and a generator's
    placed one - reads the same on every PDF in this pipeline: the
    construction graph, the flow and susceptance graphs, and the cluster
    maps. It returns a copy for plotting only; `node_index` itself, and
    whatever a script writes to CSV from it, keep the real, ungeocoded x/y
    (NaN where `buses.csv` doesn't have one).
    """
    is_bus = (node_index["node_type"] == "bus") if "node_type" in node_index \
        else pd.Series(True, index=node_index.index)
    buses = node_index.loc[is_bus, ["name", "x", "y"]]
    buses = impute_coordinates(buses, branches)
    buses = impute_coordinates(buses, branches)

    node_index = node_index.drop(columns=["x", "y"]).merge(
        buses[["name", "x", "y"]], on="name", how="left")
    if "node_type" in node_index and (node_index["node_type"] == "generator").any():
        node_index = jitter_generator_coordinates(node_index)
    return node_index


def matrices(
    A: sparse.csr_array,
) -> tuple[sparse.csr_array, sparse.csr_array, sparse.csr_array]:
    """Degree, combinatorial Laplacian and symmetric normalised Laplacian."""
    L, d = csgraph.laplacian(A, normed=False, return_diag=True)
    L_sym, _ = csgraph.laplacian(A, normed=True, return_diag=True)
    # a_ij/sqrt(d_i d_j) and its transpose multiply in a different order, so
    # L_sym comes back one machine epsilon off symmetric. eigsh assumes exact
    # symmetry, so average the two triangles rather than hand it the noise.
    L_sym = (L_sym + L_sym.T) / 2
    D = sparse.diags_array(d).tocsr()
    return D, sparse.csr_array(L), sparse.csr_array(L_sym)


def reciprocal(A: sparse.csr_array) -> sparse.csr_array:
    """The same graph weighted by 1/capacity, so the weight reads as a distance.

    The reciprocal is taken of the *summed* corridor capacity, not of each
    circuit separately, so a double-circuit corridor is half as far as a single
    one rather than twice as far. Only the stored non-zeros are inverted: an
    absent edge stays 0 rather than becoming inf.
    """
    R = A.copy()
    R.data = 1.0 / R.data
    return R


# ---- output ---- #


def out_dir(case: str, generators: bool, stage: str) -> Path:
    """The folder one stage of the pipeline writes into, created if missing.

    Results are keyed by case, then by node set, then by the script that made
    them: `data/graphs/<case>/{gen,buses}/<stage>/`, where stage is
    "construction" in get_weighted_graph.py, "betweenness" in betweenness.py,
    and "clustering" for both halves of the spectral pipeline.

    The node set comes above the stage because it is the thing that has to match
    across stages: a betweenness ranking and a clustering describe the same grid
    only if both were run over the same nodes, and a run with generators shares
    nothing with a run without them beyond the case name. Keeping the two node
    sets in sibling trees means a whole run can be read - or deleted - without
    picking files apart by suffix.
    """
    path = OUT_DIR / case / ("gen" if generators else "buses") / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def save(A: sparse.csr_array, out: Path, suffix: str) -> None:
    D, L, L_sym = matrices(A)
    for name, matrix in (("A", A), ("D", D), ("L", L), ("Lsym", L_sym)):
        sparse.save_npz(out / f"{name}_{suffix}.npz", sparse.csr_matrix(matrix))


def plot(A: sparse.csr_array, bus_index: pd.DataFrame, path: Path, note: str = "") -> None:
    """A geographic view of the grid beside the sparsity pattern of A.

    Call `geocode()` on `bus_index` first if it might have missing x/y - this
    function itself never estimates a position, it only draws whatever it is
    given. `note`, when given, is appended to the title as a second line -
    the caller's summary of how many positions are estimated rather than
    read straight from `buses.csv` (see impute_coordinates() and
    jitter_generator_coordinates()).
    """
    x = bus_index["x"].to_numpy(float)
    y = bus_index["y"].to_numpy(float)
    located = np.isfinite(x) & np.isfinite(y)

    # Upper triangle only: one segment per corridor, not two.
    upper = sparse.triu(A, k=1).tocoo()
    drawable = located[upper.row] & located[upper.col]
    i, j, capacity = upper.row[drawable], upper.col[drawable], upper.data[drawable]
    segments = np.stack([np.c_[x[i], y[i]], np.c_[x[j], y[j]]], axis=1)

    fig, (left, right) = plt.subplots(1, 2, figsize=(15, 7.5))

    left.add_collection(
        LineCollection(
            segments,
            linewidths=0.4 + 2.6 * capacity / capacity.max(),
            colors="#2c6fbb",
            alpha=0.65,
            zorder=1,
        )
    )
    # A generator row may or may not carry a coordinate (jitter_generator_
    # coordinates() gives it one, plain node_index never does) - either way
    # it is not a bus, so it is drawn separately, in its own colour and
    # marker, rather than folded into the bus scatter as one more red dot.
    is_bus = (bus_index["node_type"] == "bus").to_numpy() if "node_type" in bus_index \
        else np.ones(len(bus_index), dtype=bool)
    bus_drawn = located & is_bus
    gen_drawn = located & ~is_bus

    left.scatter(
        x[bus_drawn], y[bus_drawn],
        s=2 + bus_index["v_nom"].to_numpy(float)[bus_drawn] / 12,
        c="#d1495b", zorder=2, linewidths=0, label="bus",
    )
    if gen_drawn.any():
        left.scatter(
            x[gen_drawn], y[gen_drawn],
            s=14, marker=GENERATOR_MARKER,
            c="#2a9d8f", zorder=3, linewidths=0, label="generator",
        )
        left.legend(loc="best", fontsize=8, frameon=False)
    left.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(y))))  # rough WGS84 fix
    left.autoscale_view()
    left.set_xlabel("longitude")
    left.set_ylabel("latitude")
    buses = int(is_bus.sum())
    bus_located = int((located & is_bus).sum())
    title = (
        f"{bus_located}/{buses} buses placed, "
        f"{drawable.sum()}/{upper.nnz} corridors drawable\n"
        "line width ∝ corridor capacity, marker size ∝ v_nom"
    )
    if note:
        title += f"\n{note}"
    left.set_title(title)

    right.spy(sparse.csr_matrix(A), markersize=0.4, color="#2c6fbb")
    right.set_title(f"sparsity of A — {A.shape[0]} nodes, {A.nnz} non-zeros")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
