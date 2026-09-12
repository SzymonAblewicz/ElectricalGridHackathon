# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa"]
# ///
"""The Laplacian the power flow itself runs on: L = K B K^T, weighted by susceptance.

`get_weighted_graph.py` weights a corridor by `s_nom` and `get_flow_graph.py` weights it by
what a DC power flow puts on it. This weights it by `1/x` - the susceptance - which is
neither a rating nor a result but the physical coupling between two buses.

That makes it the only one of the three whose Laplacian is not an analogy. A DC power flow
*is* `L theta = p` with exactly this matrix: angles play the role of voltages, power the role
of current, and susceptance the role of conductance. Its eigenvectors are the network's real
electrical modes rather than a graph-theoretic stand-in for them, and its Fiedler vector
separates the buses that are hardest to hold in synchronism - which is the classical basis
for islanding and coherency studies.

It is also the only one of the three that does not depend on a dispatch. There is no `lpf`
call here and no `p_set` is read: change the generation and this matrix does not move.

Susceptances come from `flowmath.branches()` in the participant kit rather than from
`lines.csv`, because `x` in the CSV is in ohms while the flow is governed by `x_pu_eff`,
PyPSA's per-unit value with transformer tap ratios already folded in. Using the raw ohms
would silently misweight all 225 transformers.

Two wrinkles, both real and both handled rather than hidden:

  * 23 of the 980 branches have a **negative** susceptance. They are the star-point
    equivalents of three-winding transformers, where one winding legitimately carries a
    negative equivalent reactance. A negative edge weight is meaningless to a Laplacian
    eigenvector - and would make `gwg.plot()` scale its line widths off a negative maximum -
    so magnitudes are taken and the count is reported.

  * DC links are **not** branches. `flowmath.branches()` excludes them by design: a link is a
    controllable injection, not part of the linear network. That is precisely why the case
    has four synchronous areas, three of them a single GB-side terminal. `gwg.adjacency()`
    drops those as branchless, so the graph that comes out is the one connected AC island.

Set the constants below and run the file.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))  # graph_lib and spectrum sit beside this one

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402

import graph_lib as gwg  # noqa: E402
import spectrum as spec  # noqa: E402
from get_flow_graph import cluster_and_map, cluster_report, out_dir, quiet  # noqa: E402

# flowmath.py ships with the participant kit, which is not on the path by default.
# gwg.PYPSA_DIR is <repo>/grid_TF_Wind/data/pypsa, so its grandparent is grid_TF_Wind.
KIT = gwg.PYPSA_DIR.parent.parent / "participant-kit"
sys.path.insert(0, str(KIT))

import flowmath  # noqa: E402


# ---- what to run ---- #

CASE = "TYTFS2024_WP2024_V35_transmission"
K = 33                         # clusters wanted
EMBEDDING = "sym"             # "sym" | "rw" | "unnorm"

EXTRA = 30                     # eigenpairs beyond K, so the gap after the K-th is visible
RESTARTS = 10                 # k-means++ draws; best of them is kept

WEIGHTING = "susceptance"     # fixed; the name the outputs are filed under


# ---- entry point ---- #


def main() -> None:
    if EMBEDDING not in spec.EMBEDDINGS:
        raise ValueError(f"EMBEDDING must be one of {spec.EMBEDDINGS}, not {EMBEDDING!r}")

    case_dir = gwg.PYPSA_DIR / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available: {', '.join(sorted(p.name for p in gwg.PYPSA_DIR.iterdir() if p.is_dir()))}")

    quiet()
    network = pypsa.Network(str(case_dir))

    # branches() calls calculate_dependent_values() itself, which is what populates
    # x_pu_eff - so this must not be read off n.lines before the call.
    frame = flowmath.branches(network)
    negative = int((frame["susceptance"] < 0).sum())
    susceptance = frame["susceptance"].abs()

    buses, branches = gwg.read_case(case_dir)
    # Same overwrite trick get_flow_graph.py uses: put the weight in the column
    # gwg.adjacency() reads, and every downstream behaviour - parallel circuits summing,
    # branchless buses dropping, the generator block - is inherited rather than rewritten.
    weights = susceptance.reindex(branches["name"])
    if weights.isna().any():
        missing = branches["name"][weights.isna().to_numpy()].tolist()
        raise ValueError(f"no susceptance for {len(missing)} branches: {missing[:5]}")
    rated = branches              # s_nom still in MVA - the cut-corridor table reads from this
    branches = branches.assign(s_nom=weights.to_numpy(float))

    # A generator edge has no susceptance - a machine is not a branch of the linear network.
    # Weighting it by the median corridor susceptance attaches each unit to its bus about as
    # firmly as a typical circuit does, so the generators ride with their bus in the
    # embedding instead of dangling off it as their own near-zero eigenvector.
    generators = gwg.read_generators(case_dir)
    tie = float(susceptance.median())
    generators = generators.assign(p_nom=tie)

    A, node_index = gwg.adjacency(buses, branches, generators)
    D, L, L_sym = gwg.matrices(A)

    construction = out_dir(CASE, "construction")
    node_index.to_csv(construction / "node_index.csv")
    gwg.save(A, construction, WEIGHTING)
    # Estimated for the plot only, same as get_weighted_graph.py's own
    # construction graph - the CSV above keeps the real, ungeocoded x/y.
    placed = gwg.geocode(node_index, branches)
    gwg.plot(A, placed, construction / f"graph_{WEIGHTING}.pdf")

    M = L if EMBEDDING == "unnorm" else L_sym
    n_nodes = A.shape[0]
    want = K + EXTRA
    if want >= n_nodes:
        raise ValueError(f"K + EXTRA = {want} eigenpairs wanted from a {n_nodes}-node graph")

    values, vectors = spec.spectrum(M, want)
    gaps, suggested = spec.eigengap(values)
    components = csgraph.connected_components(A, directed=False)[0]

    clustering = out_dir(CASE, "clustering")
    stem = f"spectrum_{EMBEDDING}_{WEIGHTING}"
    np.savez(
        clustering / f"{stem}.npz",
        eigenvalues=values, eigenvectors=vectors, degrees=D.diagonal(),
        embedding=np.array(EMBEDDING), case=np.array(CASE),
        generators=np.array(True), weighting=np.array(WEIGHTING),
    )
    spec.K = K
    spec.plot(values, suggested, clustering / f"{stem}.pdf",
              f"{CASE} +generators — {EMBEDDING}, {WEIGHTING}")

    # The partition and its map, by the flow scripts' shared helper - cluster.py's own steps -
    # so a map drawn here and one drawn by cluster.py mean the same thing. This graph does not
    # depend on a dispatch, so there is exactly one of it per case: run it once and that is it.
    res = cluster_and_map(
        A, node_index, placed, rated, vectors, D.diagonal(), k=K, restarts=RESTARTS,
        embedding=EMBEDDING, folder=clustering, run=f"{EMBEDDING}_{WEIGHTING}_k{K}",
        title=f"{CASE} +generators — {EMBEDDING}, {WEIGHTING}, k = {K}")
    stats, files = cluster_report(res, k=K, restarts=RESTARTS, weight="susceptance")

    dropped = sorted(set(buses["name"]) - set(node_index["name"]))
    listing = "\n".join(
        f"    {i + 1:>2}. {v:>13.6g}"
        + (f"   gap {gaps[i]:>11.6g}" if i < len(gaps) else "")
        + ("  <- largest" if i + 1 == suggested else "")
        for i, v in enumerate(values))
    print(
        f"{CASE} +generators - {WEIGHTING}\n"
        f"  branches   {len(frame)} ({frame['kind'].value_counts().to_dict()}), "
        f"links excluded as controllable injections\n"
        f"  negative   {negative} branches had x_pu_eff < 0 (3-winding star points); "
        f"magnitudes taken\n"
        f"  weights    {susceptance.min():,.1f} to {susceptance.max():,.1f} MW/rad, "
        f"median {tie:,.1f} (also the generator tie)\n"
        f"  buses      {len(node_index) - len(generators)} kept, {len(dropped)} dropped as "
        f"branchless{': ' + ', '.join(dropped) if dropped else ''}\n"
        f"  graph      {n_nodes} nodes in {components} component"
        f"{'s' if components != 1 else ''}, {A.nnz // 2} edges\n"
        f"  spectrum   K = {K} configured, eigengap suggests k = {suggested}\n"
        f"{listing}\n"
        f"             (the gap after the 1st is structural on a connected graph, so it is\n"
        f"              excluded from the suggestion)\n"
        + stats
        + f"  wrote      {construction}\n"
        f"             {clustering}\n"
        + files
    )


if __name__ == "__main__":
    main()
