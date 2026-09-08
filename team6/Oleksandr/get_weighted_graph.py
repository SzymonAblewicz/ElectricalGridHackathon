# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib"]
# ///
"""Turn one PyPSA case from the participant kit into sparse graph matrices.

Takes the buses (of any kind) and the branches that join them - lines and
transformers - and writes the degree, adjacency and Laplacian matrices, both
combinatorial (L = D - A) and symmetric normalised (L_sym = I - D^-1/2 A D^-1/2).
Edge weight is `s_nom`, the branch's maximum apparent power in MVA; where a bus
pair carries several circuits their capacities are added, since that is what a
corridor rating means.

Node i of every matrix is the i-th bus of `buses.csv`, so A[0, 100] is the
corridor between the first and the hundred-and-first bus in the file. The one
wrinkle is that buses with no branch at all are dropped, which shifts the index
past the first of them - `bus_index.csv` carries the original `csv_row` for
each node so the mapping stays invertible. In the TYTFS transmission cases the
dropped buses are exactly the GB-side terminals of the DC interconnectors
(Moyle, EWIC, Greenlink), isolated here because `links.csv` is not part of the
graph: it rates its elements in `p_nom` on a DC carrier, not `s_nom`.

`--reciprocal` additionally writes the same four matrices under a 1/s_nom
weighting, which reads as a distance rather than a capacity: a double-circuit
corridor comes out closer than a single one.

`--generators` adds the generators as nodes in their own right, each joined to
its bus by one edge rated `p_nom` in MW - the same capacity reading as `s_nom`
on a branch. They are leaves, so they change no path between two buses; what
they change is which buses are endpoints, and how many. Since the node set is
then a different shape, the matrices are written under a `_gen` suffix and the
index under `node_index.csv`, so a run with the flag never overwrites the
bus-only matrices a run without it produced.

Output goes under `data/graphs/<case>/{gen,buses}/<stage>/`, which `out_dir()`
below defines for the whole pipeline - this script writes the "construction"
stage, `betweenness.py` writes "betweenness", and `spectrum.py` and
`cluster.py` share "clustering". So a case holds two parallel trees, one per
node set, and every script that reads one of these matrices back knows where to
look without being told.

To run it: set CASE, RECIPROCAL and GENERATORS in the "what to run" block
below, then run the file.
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
PYPSA_DIR = FOLDER.parent.parent / "grid_TF_Wind" / "data" / "pypsa"
OUT_DIR = FOLDER / "data" / "graphs"


# ---- what to run ---- #
#
# A case is a directory name under grid_TF_Wind/data/pypsa, of the shape
# TYTFS2024_{WP,SV}{2024,2033}_V35_{transmission,full} - winter peak or summer
# valley, 2024 or 2033, the 110 kV-and-above transmission network or the full
# one with the distribution buses in it. The northwest_* cases there do not
# work: their buses.csv carries only name, v_nom and control, so it has none of
# the BUS_COLUMNS this reads.

CASE = "TYTFS2024_WP2024_V35_transmission"
RECIPROCAL = True    # also write the 1/s_nom matrices, weighted as a distance
GENERATORS = True    # add the generators as nodes of their own


BUS_COLUMNS = ["name", "v_nom", "x", "y", "jurisdiction", "station"]


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
    "construction" here, "betweenness" in betweenness.py, and "clustering" for
    both halves of the spectral pipeline.

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


def plot(A: sparse.csr_array, bus_index: pd.DataFrame, path: Path) -> None:
    """A geographic view of the grid beside the sparsity pattern of A."""
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
    left.scatter(
        x[located], y[located],
        s=2 + bus_index["v_nom"].to_numpy(float)[located] / 12,
        c="#d1495b", zorder=2, linewidths=0,
    )
    left.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(y))))  # rough WGS84 fix
    left.autoscale_view()
    left.set_xlabel("longitude")
    left.set_ylabel("latitude")
    # Generator rows, when present, carry no coordinate and are never drawn, so
    # they would only deflate the "geocoded" fraction if counted.
    buses = ((bus_index["node_type"] == "bus").sum()
             if "node_type" in bus_index else len(bus_index))
    left.set_title(
        f"{located.sum()}/{buses} buses geocoded, "
        f"{drawable.sum()}/{upper.nnz} corridors drawable\n"
        "line width ∝ corridor capacity, marker size ∝ v_nom"
    )

    right.spy(sparse.csr_matrix(A), markersize=0.4, color="#2c6fbb")
    right.set_title(f"sparsity of A — {A.shape[0]} nodes, {A.nnz} non-zeros")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---- entry point ---- #


def main() -> None:
    case_dir = PYPSA_DIR / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available: {', '.join(sorted(p.name for p in PYPSA_DIR.iterdir() if p.is_dir()))}")

    buses, branches = read_case(case_dir)
    generators = read_generators(case_dir) if GENERATORS else None
    A, node_index = adjacency(buses, branches, generators)

    out = out_dir(CASE, GENERATORS, "construction")
    # The gen/buses split already separates the two node sets, but the suffix
    # stays on the filenames so a matrix still says which graph it belongs to
    # once it has been copied out of the tree - and so a bus-only bus_index.csv
    # can never end up describing the rows of a matrix with generators in it.
    tag = "_gen" if GENERATORS else ""
    node_index.to_csv(out / ("node_index.csv" if GENERATORS else "bus_index.csv"))
    save(A, out, f"capacity{tag}")
    if RECIPROCAL:
        save(reciprocal(A), out, f"reciprocal{tag}")
    plot(A, node_index, out / f"graph{tag}.pdf")

    dropped = sorted(set(buses["name"]) - set(node_index["name"]))
    components = csgraph.connected_components(A, directed=False)[0]
    # Buses lead the index, so the bus-only block is the leading square of A -
    # the capacity line then reads in MVA per corridor either way, rather than
    # mixing branch MVA with generator MW.
    # adjacency() drops generators whose bus went with the branchless ones, so
    # the kept set is read back off the index rather than off the input frame.
    kept_gen = (node_index[node_index["node_type"] == "generator"]
                if GENERATORS else node_index.iloc[:0])
    n_bus = len(node_index) - len(kept_gen)
    corridors = A[:n_bus, :n_bus]
    summary = (
        f"{CASE}\n"
        f"  buses      {n_bus} kept, {len(dropped)} dropped as branchless"
        f"{': ' + ', '.join(dropped) if dropped else ''}\n"
        f"  branches   {len(branches)} on {corridors.nnz // 2} corridors\n"
    )
    if GENERATORS:
        lost = len(generators) - len(kept_gen)
        summary += (
            f"  generators {len(kept_gen)} on "
            f"{kept_gen['bus'].nunique()} buses, "
            f"{kept_gen['p_nom'].sum():,.0f} MW installed"
            f"{f', {lost} dropped with their bus' if lost else ''}\n"
        )
    summary += (
        f"  capacity   {corridors.sum() / 2:,.0f} MVA total, "
        f"{corridors.data.min():,.0f}-{corridors.data.max():,.0f} MVA per corridor\n"
        f"  components {components}\n"
        f"  wrote      {out}"
    )
    print(summary)


if __name__ == "__main__":
    main()
