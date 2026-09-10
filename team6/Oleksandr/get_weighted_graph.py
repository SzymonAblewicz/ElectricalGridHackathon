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

The graph construction, geocoding and plotting code this script calls lives in
`graph_lib.py`, shared with `betweenness.py`, `spectrum.py`, `cluster.py` and
the two graph scripts in `flow/` - see that file's docstring for what each
piece does. This file is only the "what to run" constants below and `main()`.

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

Output goes under `data/graphs/<case>/{gen,buses}/<stage>/`, which
`graph_lib.out_dir()` defines for the whole pipeline - this script writes the
"construction" stage, `betweenness.py` writes "betweenness", and `spectrum.py`
and `cluster.py` share "clustering". So a case holds two parallel trees, one
per node set, and every script that reads one of these matrices back knows
where to look without being told.

`graph{_gen}.pdf`, the geographic half of that stage, does not plot the raw
`buses.csv` coordinates as-is - it plots `graph_lib.geocode(node_index,
branches)`. A bus missing x/y is placed at the average position of its
directly connected neighbours, run twice so a chain of uncoordinated buses
resolves one hop at a time - a bus stays unplaced only if no neighbour, even
after the first pass, has a coordinate. A generator has no coordinate of its
own and only one neighbour to inherit from, so it is placed beside its bus
instead, with a small random offset so several generators on one bus don't
land on the same point, and drawn in its own colour and marker so it reads as
a generator rather than one more bus. The matrices and CSVs are unaffected -
only a plot's node positions are ever estimated. `cluster.py` and the two
scripts in `flow/` call the same `geocode()` before drawing their own maps, so
a bus's estimated position - and a generator's placed one - reads the same
everywhere in this pipeline.

To run it: set CASE, RECIPROCAL and GENERATORS in the "what to run" block
below, then run the file.
"""

from __future__ import annotations

import pandas as pd
from scipy.sparse import csgraph

import graph_lib as gl

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


# ---- entry point ---- #


def main() -> None:
    case_dir = gl.PYPSA_DIR / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available: {', '.join(sorted(p.name for p in gl.PYPSA_DIR.iterdir() if p.is_dir()))}")

    buses, branches = gl.read_case(case_dir)
    generators = gl.read_generators(case_dir) if GENERATORS else None
    A, node_index = gl.adjacency(buses, branches, generators)

    out = gl.out_dir(CASE, GENERATORS, "construction")
    # The gen/buses split already separates the two node sets, but the suffix
    # stays on the filenames so a matrix still says which graph it belongs to
    # once it has been copied out of the tree - and so a bus-only bus_index.csv
    # can never end up describing the rows of a matrix with generators in it.
    tag = "_gen" if GENERATORS else ""
    node_index.to_csv(out / ("node_index.csv" if GENERATORS else "bus_index.csv"))
    gl.save(A, out, f"capacity{tag}")
    if RECIPROCAL:
        gl.save(gl.reciprocal(A), out, f"reciprocal{tag}")

    # geocode() fills gaps for the plot only - the matrices and CSVs above
    # are written from the real, ungeocoded buses. See its docstring for what
    # it does and why; this is the same call every other script in the
    # pipeline makes before drawing node_index on a map.
    plot_index = gl.geocode(node_index, branches)
    was_located = node_index["x"].notna() & node_index["y"].notna()
    now_located = plot_index["x"].notna() & plot_index["y"].notna()
    is_bus = (node_index["node_type"] == "bus") if "node_type" in node_index \
        else pd.Series(True, index=node_index.index)
    n_bus_estimated = int((is_bus & ~was_located & now_located).sum())
    note = f"{n_bus_estimated} bus position{'s' if n_bus_estimated != 1 else ''} " \
           f"estimated from neighbours"
    if GENERATORS:
        n_gen_placed = int((~is_bus & now_located).sum())
        note += f", {n_gen_placed} generators placed beside their bus"
    gl.plot(A, plot_index, out / f"graph{tag}.pdf", note=note)

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
