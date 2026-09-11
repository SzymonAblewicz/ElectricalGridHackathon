# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa", "highspy>=1.7"]
# ///
"""The flow-weighted spectrum and clustering for every hour of the synthetic week.

`get_flow_graph.py` builds one graph from one hour. This builds 168 of them, clusters each,
and reports how the partition moves - which is the question the synthetic profiles exist to
answer and the tytfs cases cannot, having a single snapshot called "now".

**The dispatch is capped by default.** DISPATCH = "capacity_constrained" runs PyPSA's LOPF
(HiGHS) over the whole week, the solve congested_lines.py uses: every branch's s_nom is a hard
constraint, so generation is redistributed round the system until nothing is over its rating,
with load shedding at the value of lost load only where no redispatch can avoid it. The
older "prorata" dispatch - every unit at the same share of its availability, blind to the
ratings - is kept as the alternative; its flows can and do sit above 100%. Capped outputs
carry a `_capped` suffix so the two never overwrite each other.

Past the dispatch, the run takes a few seconds, for two reasons worth knowing:

  * **The incidence structure is constant.** Only the edge weights change between hours, so
    the (i, j) index is built once and the data vector swapped per hour. Calling
    `adjacency()` 168 times would pay its `iterrows()` loop 168 times over.

  * **All 168 flow solutions come from one pseudoinverse.** `theta = L+ (P + K B phi)` with
    P as a 754 x 168 matrix, not 168 separate power flows.

Clustering is `cluster.py`'s, imported rather than reimplemented: `embed()` for the
Ng-Jordan-Weiss row normalisation, `kmeans()` for best-of-RESTARTS k-means++, `cut()` for the
capacity crossing a boundary. So a single-hour run of get_flow_graph.py plus cluster.py, and
one column of this, agree by construction.

**On labels.** k-means numbers its clusters by whichever centroid its seeding happened to
reach first, so the label a node carries in hour t has no relation to its label in hour t+1,
and a naive time series of them is noise. Each hour is therefore relabelled to agree as
closely as possible with the hour before, by maximising overlap with the Hungarian algorithm.
What survives that is real movement; what it removes was never information. The churn figure
is measured after alignment.

**Read the floor before reading lambda_2.** Under `headroom` every corridor at or above 100%
of its rating clamps to HEADROOM_FLOOR whatever its overload, so lambda_2 plateaus as soon as
anything saturates. It separates congested hours from clear ones cleanly, and does not rank
the congested ones against each other. `loading` has no plateau at the top - an overloaded
corridor simply scores above 1 - but clamps the other end: every element carrying under
FLOW_FLOOR reads as FLOW_FLOOR, and those idle elements are its weakest ties.

Only the kit dataset has a week in it; a one-snapshot case raises rather than producing a
series of length one.

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import cluster as clu  # noqa: E402
import get_flow_graph as gfg  # noqa: E402
import spectrum as spec  # noqa: E402

gwg = gfg.gwg  # whatever graph module get_flow_graph imports, so the two never diverge

# flowmath.py ships with the participant kit, not with this folder, so it needs the path.
# gwg.PYPSA_DIR is <repo>/grid_TF_Wind/data/pypsa, so its grandparent is grid_TF_Wind.
sys.path.insert(0, str(gwg.PYPSA_DIR.parent.parent / "participant-kit"))


# ---- what to run ---- #

CASE = "WP2033_all-island"    # kit cases only: {WP,SV}{2024,2033}_{all-island,north-west}
WEIGHTING = "loading"        # "headroom" | "loading" | "inverse_flow"
K = 6                         # clusters wanted
EMBEDDING = "sym"             # "sym" | "rw" | "unnorm"

EXTRA = 10                     # eigenpairs beyond K, so the gap after the K-th is visible
RESTARTS = 10                 # k-means++ draws per hour; best of them is kept

# "capacity_constrained": PyPSA's LOPF, s_nom a hard limit, power redistributed round the
#     system (the congested_lines.py solve). "prorata": per-area pro-rata, ratings ignored.
DISPATCH = "capacity_constrained"   # "capacity_constrained" | "prorata"
DISPATCHES = ("capacity_constrained", "prorata")

#: Eigenvalues below this count as "nearly zero" - one nearly-severed piece each.
NEAR_ZERO = 1e-4


# ---- dispatch and flows, all hours at once ---- #


def prorata_injections(network) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-area pro-rata dispatch, and the bus injections it produces, for every snapshot.

    Pro-rata is struck per synchronous area, not system-wide: three of the four areas are a
    single GB interconnector terminal carrying one `import` unit and no load, and no schedule
    is chosen for the DC links, so a system-wide fraction strands that output the instant it
    is generated. `get_flow_graph.kit_dispatch` makes the same choice for one hour; this is
    the same thing vectorised over 168.
    """
    network.determine_network_topology()
    snaps, area = network.snapshots, network.buses["sub_network"]

    available = pd.DataFrame(
        np.repeat(network.generators["p_nom"].to_numpy(float)[None, :], len(snaps), 0),
        index=snaps, columns=network.generators.index)
    profiled = network.generators_t.p_max_pu.columns
    available[profiled] = (network.generators_t.p_max_pu[profiled]
                           * network.generators.loc[profiled, "p_nom"])

    plant = ~(network.generators["carrier"].isin(gfg.ARTIFICIAL_CARRIERS)
              | (network.generators["sign"] < 0))
    # .mul(axis=1), never .where(plant, 0.0): `available` is indexed by snapshot and `plant`
    # by generator, and DataFrame.where aligns a Series on the INDEX - so where() matches
    # nothing and silently zeroes the entire frame.
    offered = available.mul(plant.astype(float), axis=1)

    generator_area = area.reindex(network.generators["bus"]).to_numpy()
    load_area = area.reindex(network.loads["bus"]).to_numpy()
    share = (network.loads_t.p_set.T.groupby(load_area).sum().T
             / offered.T.groupby(generator_area).sum().T)
    share = share.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    dispatch = offered * share.reindex(columns=generator_area).to_numpy()

    signed = dispatch * network.generators["sign"].to_numpy()
    p = signed.T.groupby(network.generators["bus"].to_numpy()).sum().T
    p = p.sub(network.loads_t.p_set.T.groupby(network.loads["bus"].to_numpy()).sum().T,
              fill_value=0.0)
    p = p.reindex(columns=network.buses.index).fillna(0.0)

    residual = float(p.sum(axis=1).abs().max())
    if residual > 1e-6:
        raise RuntimeError(
            f"injections do not balance: worst hour off by {residual:,.3f} MW. L theta = P "
            f"has no solution unless they sum to zero, so this must be fixed, not flowed.")
    return dispatch, p


def capped_injections(network) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dispatch and bus injections from PyPSA's capacity-constrained LOPF, every snapshot.

    The solve congested_lines.capped_injections makes - network.optimize with HiGHS, each
    branch's s_nom a hard constraint - returned in the shapes prorata_injections returns, so
    the flow maths and the graph downstream are untouched; only the dispatch feeding them
    differs. It cannot simply be imported from there: congested_lines imports this module.

    The solve schedules the DC links too, so their injections are added at both ends. The
    system as a whole then does not sum to zero - Moyle loses ~0.8% in conversion - but each
    synchronous area does, and each area is what the pseudoinverse solves on its own, so the
    balance is checked per area.
    """
    network.optimize(network.snapshots, solver_name="highs",
                     solver_options={"output_flag": False}, progress=False)
    dispatch = network.generators_t.p.reindex(columns=network.generators.index).fillna(0.0)

    signed = dispatch * network.generators["sign"].to_numpy()
    p = signed.T.groupby(network.generators["bus"].to_numpy()).sum().T
    p = p.sub(network.loads_t.p_set.T.groupby(network.loads["bus"].to_numpy()).sum().T,
              fill_value=0.0)
    p = p.reindex(columns=network.buses.index).fillna(0.0)
    for name in network.links.index:
        p0 = network.links_t.p0[name]
        p[network.links.at[name, "bus0"]] -= p0
        p[network.links.at[name, "bus1"]] += p0 * network.links.at[name, "efficiency"]

    network.determine_network_topology()
    residual = float(p.T.groupby(network.buses["sub_network"]).sum().abs().max().max())
    if residual > 1e-2:
        raise RuntimeError(f"LOPF injections do not balance within a synchronous area: "
                           f"worst area-hour off by {residual:,.4f} MW")
    return dispatch, p


def branch_flows(network, injections: pd.DataFrame) -> pd.DataFrame:
    """|MW| on every branch for every snapshot, from one pseudoinverse and one matmul.

    The phase-shifter term is not optional: two of the all-island transformers impose an
    angle of their own, and a Laplacian that ignores them is out by tens of MW on the
    circuits around them.
    """
    import flowmath

    frame = flowmath.branches(network)
    incidence, buses, edges = flowmath.incidence(network, frame)
    b = frame["susceptance"].to_numpy(float)
    phi = frame["phase_shift"].to_numpy(float)
    inverse = flowmath.pseudoinverse((incidence * b) @ incidence.T)
    p = injections.reindex(columns=buses).fillna(0.0).to_numpy().T
    theta = inverse @ (p + (incidence @ (b * phi))[:, None])
    flow = np.abs(b[:, None] * (incidence.T @ theta - phi[:, None]))
    return pd.DataFrame(flow, index=edges, columns=network.snapshots)


# ---- the graph, built once ---- #


def structure(case_dir: Path):
    """The (i, j) edge index, node table and nominal weights - everything time-invariant.

    Mirrors `adjacency()`: buses carrying no branch are dropped, generators are appended
    after them, and duplicate (i, j) pairs are left to sum in tocsr(), which is what makes
    parallel circuits on one corridor add their rating.
    """
    buses, branches = gwg.read_case(case_dir)
    generators = gwg.read_generators(case_dir)
    _, node_index = gwg.adjacency(buses, branches, generators)

    kept = pd.Index(node_index[node_index["node_type"] == "bus"]["name"])
    host = kept.get_indexer(generators["bus"])
    on_grid = host >= 0

    rows = np.r_[kept.get_indexer(branches["bus0"]),
                 len(kept) + np.arange(int(on_grid.sum()))]
    cols = np.r_[kept.get_indexer(branches["bus1"]), host[on_grid]]
    return (node_index, rows, cols,
            branches["name"].to_numpy(), branches["s_nom"].to_numpy(float),
            generators["name"].to_numpy()[on_grid],
            generators["p_nom"].to_numpy(float)[on_grid])


def weights(nominal: np.ndarray, used: np.ndarray) -> np.ndarray:
    """One hour of edge affinities, under WEIGHTING - the formulas get_flow_graph uses."""
    if WEIGHTING == "headroom":
        return np.maximum(nominal - used, gfg.HEADROOM_FLOOR)
    if WEIGHTING == "loading":
        return gfg.rating_share(nominal, used)
    return 1.0 / np.maximum(used, gfg.FLOW_FLOOR)


# ---- labels that mean the same thing from one hour to the next ---- #


def align(previous: np.ndarray, current: np.ndarray, k: int) -> np.ndarray:
    """Relabel `current` so its cluster numbers agree as far as possible with `previous`.

    Maximising the overlap between two labellings is an assignment problem, and the Hungarian
    algorithm solves it exactly - so the churn reported afterwards is nodes that genuinely
    moved between zones, not the solver renumbering itself.
    """
    overlap = np.zeros((k, k), dtype=np.int64)
    np.add.at(overlap, (previous, current), 1)
    _, mapping = linear_sum_assignment(-overlap)      # mapping: current label -> previous
    relabel = np.empty(k, dtype=int)
    relabel[mapping] = np.arange(k)
    return relabel[current]


# ---- output ---- #


def _day_ticks(axis, snaps) -> None:
    """One tick per midnight - 168 hourly labels are unreadable and only days are needed."""
    marks = [i for i, s in enumerate(snaps) if pd.Timestamp(s).hour == 0]
    axis.set_xticks(marks, [pd.Timestamp(snaps[i]).strftime("%a %d") for i in marks])


def plot_spectrum(series: pd.DataFrame, snaps, path: Path) -> None:
    """lambda_2 against max branch loading, over the week."""
    hours = np.arange(len(snaps))
    fig, ax = plt.subplots(figsize=(10.0, 4.4))
    ax.set_axisbelow(True)
    ax.semilogy(hours, series["lambda_2"].to_numpy(), color="#2c6fbb", linewidth=1.5,
                label=r"$\lambda_2$ (algebraic connectivity)")
    ax.set_ylabel(r"$\lambda_2$", color="#2c6fbb")
    ax.set_xlim(0, len(snaps) - 1)
    _day_ticks(ax, snaps)

    twin = ax.twinx()
    twin.plot(hours, 100 * series["max_loading"].to_numpy(), color="#d1495b",
              linewidth=1.2, linestyle=(0, (4, 2)), label="max branch loading")
    twin.axhline(100, color="#d1495b", linewidth=0.8, alpha=0.35)
    twin.set_ylabel("max loading (% of rating)", color="#d1495b")
    twin.grid(False)

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    ax.legend(handles, [h.get_label() for h in handles], loc="lower left", fontsize=9,
              frameon=False)
    ax.set_title(f"{CASE} - {WEIGHTING}, {DISPATCH}: how close the grid is to splitting, "
                 f"hour by hour")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_membership(series: pd.DataFrame, labels: np.ndarray, snaps, node_index: pd.DataFrame,
                    path: Path) -> None:
    """Who sits in which cluster, when - one cell per node per hour, recoverable by zooming.

    **Resolution.** `interpolation="none"` is the setting that matters. With "nearest", as an
    earlier version used, matplotlib resamples the 1674 x 168 grid to the output dpi before
    writing the PDF: ~500 pixel rows for 1674 nodes, so three nodes blend into each pixel and
    a single node's excursion vanishes. With "none" the PDF backends embed the raw array and
    leave scaling to the viewer, so every node-hour cell survives and zooming in shows it.

    **Row order**, most significant first:
      1. the cluster held in the *most robust* hour (highest lambda_2) - the bands;
      2. within a band, how many hours the node spends outside it - loyal nodes on top,
         restless boundary nodes at the bottom edge of each band;
      3. the node's whole label sequence - so nodes that move *together* sit together, and a
         group defection reads as a block rather than as scattered hairlines.

    Each band is fenced by a rule and labelled on the right with its size and where its buses
    sit (mean lat/lon), so a band can be read as a place, not only as a colour.
    """
    n_hours, n_nodes = labels.shape
    k = int(labels.max()) + 1
    reference = int(np.asarray(series["lambda_2"]).argmax())
    band = labels[reference]
    away = (labels != band[None, :]).sum(axis=0)
    # np.lexsort sorts on its LAST key first, so the list runs least- to most-significant.
    order = np.lexsort(tuple(labels[t] for t in range(n_hours - 1, -1, -1)) + (away, band))
    grid = labels[:, order].T                                    # nodes x hours

    height = max(6.0, n_nodes / 140)                             # ~12 in for 1674 nodes
    fig, ax = plt.subplots(figsize=(15.0, height))
    cmap = plt.get_cmap("tab10")
    ax.imshow(grid, aspect="auto", interpolation="none", cmap=cmap, vmin=0, vmax=9)

    # Time axis: a label every six hours, a tick every hour, a rule at every midnight.
    ax.set_xticks(np.arange(n_hours), minor=True)
    major = [i for i, s in enumerate(snaps) if pd.Timestamp(s).hour % 6 == 0]
    ax.set_xticks(major, [pd.Timestamp(snaps[i]).strftime("%a %d\n%H:%M")
                          if pd.Timestamp(snaps[i]).hour == 0
                          else pd.Timestamp(snaps[i]).strftime("%H:%M") for i in major],
                  fontsize=7)
    for i, s in enumerate(snaps):
        if pd.Timestamp(s).hour == 0:
            ax.axvline(i - 0.5, color="white", linewidth=0.8, alpha=0.9)
    ax.axvline(reference, color="black", linewidth=0.6, linestyle=(0, (2, 2)))

    # Band fences and labels. Coordinates come from the bus rows only - generator rows carry
    # none - and the kit's 0 N 0 E placeholder is dropped, as graph_lib.placed_buses would.
    sorted_band = band[order]
    x = node_index["x"].to_numpy(float)[order]
    y = node_index["y"].to_numpy(float)[order]
    placed = np.isfinite(x) & np.isfinite(y) & ~((np.abs(x) < 1e-9) & (np.abs(y) < 1e-9))
    for c in range(k):
        rows = np.where(sorted_band == c)[0]
        if not len(rows):
            continue
        top, bottom = rows.min() - 0.5, rows.max() + 0.5
        ax.axhline(bottom, color="black", linewidth=0.7)
        where = ""
        if placed[rows].any():
            where = (f"\n{np.mean(y[rows][placed[rows]]):.2f}N "
                     f"{-np.mean(x[rows][placed[rows]]):.2f}W")
        ax.text(n_hours - 0.5 + 1.2, (top + bottom) / 2,
                f"cluster {c}\n{len(rows)} nodes{where}",
                va="center", ha="left", fontsize=7, color=cmap(c), fontweight="bold",
                clip_on=False)

    ax.set_ylabel(f"node - grouped by cluster at {pd.Timestamp(snaps[reference]):%a %H:%M} "
                  f"(dashed), loyal nodes first within each group")
    ax.set_title(f"{CASE} - {WEIGHTING}, {DISPATCH}: cluster membership, k = {k}, aligned "
                 f"hour to hour"
                 f"   (zoom in: one cell per node per hour)")
    ax.grid(False)
    fig.tight_layout()
    fig.subplots_adjust(right=0.90)                              # room for the band labels
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---- entry point ---- #


def main() -> None:
    if WEIGHTING not in gfg.WEIGHTINGS:
        raise ValueError(f"WEIGHTING must be one of {gfg.WEIGHTINGS}, not {WEIGHTING!r}")
    if EMBEDDING not in spec.EMBEDDINGS:
        raise ValueError(f"EMBEDDING must be one of {spec.EMBEDDINGS}, not {EMBEDDING!r}")
    if DISPATCH not in DISPATCHES:
        raise ValueError(f"DISPATCH must be one of {DISPATCHES}, not {DISPATCH!r}")

    # Borrow get_flow_graph's own root resolution rather than recomputing the path, so the
    # two cannot disagree about where the kit lives.
    gfg.DATASET = "kit"
    root = gfg.case_root()
    case_dir = root / CASE
    if not case_dir.is_dir():
        available = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise FileNotFoundError(
            f"no such kit case: {case_dir}\navailable: {', '.join(available)}")

    started = time.perf_counter()
    gfg.quiet()
    network = pypsa.Network(str(case_dir))
    snaps = network.snapshots
    if len(snaps) < 2:
        raise ValueError(
            f"{CASE} has {len(snaps)} snapshot(s), so there is no week to analyse. The tytfs "
            f"cases carry a single snapshot called 'now' - use get_flow_graph.py for those.")

    if DISPATCH == "capacity_constrained":
        dispatch, injections = capped_injections(network)
    else:
        dispatch, injections = prorata_injections(network)
    flows = branch_flows(network, injections)
    solved = time.perf_counter()
    shed = network.generators.index[network.generators["carrier"].isin(gfg.ARTIFICIAL_CARRIERS)]
    shed_mwh = float(dispatch.reindex(columns=shed).fillna(0.0).clip(lower=0.0).sum().sum())

    (node_index, rows, cols, branch_names, s_nom,
     generator_names, p_nom) = structure(case_dir)
    n_nodes = len(node_index)
    flow_by_branch = flows.reindex(branch_names).to_numpy()
    used_by_generator = dispatch.T.reindex(generator_names).abs().fillna(0.0).to_numpy()
    loading = flow_by_branch / np.where(s_nom > 0, s_nom, np.nan)[:, None]

    want = K + EXTRA
    if want >= n_nodes:
        raise ValueError(f"K + EXTRA = {want} eigenpairs wanted from a {n_nodes}-node graph")

    eigenvalues = np.zeros((len(snaps), want))
    labels = np.zeros((len(snaps), n_nodes), dtype=int)
    components = np.zeros(len(snaps), dtype=int)
    pinned = np.zeros(len(snaps), dtype=int)
    cut_share = np.zeros(len(snaps))
    inertia = np.zeros(len(snaps))

    for t in range(len(snaps)):
        data = np.r_[weights(s_nom, flow_by_branch[:, t]),
                     weights(p_nom, used_by_generator[:, t])]
        if WEIGHTING == "headroom":
            # Branches only: under the capped dispatch a unit running flat out has zero
            # headroom too, but a generator leaf at the floor cuts nothing off, and counting it
            # would make "corridors at or over rating" untrue.
            pinned[t] = int((data[:len(s_nom)] <= gfg.HEADROOM_FLOOR * (1 + 1e-9)).sum())
        else:   # loading and inverse_flow both clamp the same idle elements at FLOW_FLOOR
            used = np.r_[flow_by_branch[:, t], used_by_generator[:, t]]
            pinned[t] = int((used < gfg.FLOW_FLOOR).sum())
        A = sparse.csr_array(
            sparse.coo_array((np.r_[data, data], (np.r_[rows, cols], np.r_[cols, rows])),
                             shape=(n_nodes, n_nodes)).tocsr())
        components[t] = csgraph.connected_components(A, directed=False)[0]

        L, degrees = csgraph.laplacian(A, normed=EMBEDDING != "unnorm", return_diag=True)
        if EMBEDDING != "unnorm":
            L = (L + L.T) / 2                    # eigsh assumes exact symmetry
        values, vectors = spec.spectrum(sparse.csr_array(L), want)
        eigenvalues[t] = values

        raw, inertia[t] = clu.kmeans(clu.embed(vectors[:, :K], degrees, EMBEDDING),
                                     K, RESTARTS)
        labels[t] = raw if t == 0 else align(labels[t - 1], raw, K)
        crossing, total = clu.cut(A, labels[t])
        cut_share[t] = crossing / total if total > 0 else np.nan

    churn = np.r_[np.nan, (labels[1:] != labels[:-1]).mean(axis=1)]
    series = pd.DataFrame({
        "lambda_2": eigenvalues[:, 1],
        "max_loading": np.nanmax(loading, axis=0),
        # A tolerance, because the capped solve holds a binding branch at its rating to solver
        # precision, and 1.0000001 x rating is a binding constraint, not an overload.
        "branches_over": np.nansum(loading > 1.0 + 1e-6, axis=0).astype(int),
        "near_zero": (eigenvalues < NEAR_ZERO).sum(axis=1),
        "pinned_edges": pinned,
        "components": components,
        "cut_share": cut_share,
        "kmeans_inertia": inertia,
        "churn_vs_previous": churn,
    }, index=snaps)
    for i in range(want):
        series[f"eig_{i + 1}"] = eigenvalues[:, i]

    out = gfg.out_dir(CASE, "week")
    stem = f"{EMBEDDING}_{WEIGHTING}" + ("_capped" if DISPATCH == "capacity_constrained" else "")
    series.to_csv(out / f"series_{stem}.csv")
    pd.DataFrame(labels.T, index=pd.Index(node_index["name"], name="node"),
                 columns=snaps).to_csv(out / f"clusters_{stem}_k{K}.csv")
    np.savez(out / f"week_{stem}.npz",
             eigenvalues=eigenvalues, labels=labels,
             nodes=node_index["name"].to_numpy().astype(str),
             snapshots=np.array([str(s) for s in snaps]),
             max_loading=series["max_loading"].to_numpy(),
             case=np.array(CASE), weighting=np.array(WEIGHTING),
             embedding=np.array(EMBEDDING), k=np.array(K), dispatch=np.array(DISPATCH))
    plot_spectrum(series, snaps, out / f"lambda2_{stem}.pdf")
    plot_membership(series, labels, snaps, node_index, out / f"membership_{stem}_k{K}.pdf")

    elapsed = time.perf_counter() - started
    l2 = series["lambda_2"]
    congested = series["branches_over"] > 0

    # The two weightings clamp at opposite ends and mean opposite things by it, so the
    # summary cannot be written once. Under `headroom` a clamped edge is a cut and lambda_2
    # is expected to fall as loading rises. Under `inverse_flow` every weight falls as
    # loading rises and L_sym divides the common scale back out, so the sign of the
    # correlation is not fixed by any physics - printing "negative is the physics" beside a
    # positive number, as an earlier version did, is worse than printing nothing.
    if WEIGHTING == "headroom":
        sign_note = "negative is the physics: more loaded is closer to splitting"
        clamp_note = (
            f"  floor      {gfg.HEADROOM_FLOOR:g}; {pinned.min()}-{pinned.max()} edges at it "
            f"(corridors at or over rating).\n"
            f"             lambda_2 plateaus once anything saturates, so it separates\n"
            f"             congested hours from clear ones and not the congested ones from\n"
            f"             each other. Moving the floor moves the plateau, nothing else.\n")
    elif WEIGHTING == "loading":
        # Loading rises on the busy corridors and binds them tighter, while the idle ones stay
        # at the floor - so as with inverse_flow, no physics fixes the sign of the correlation.
        sign_note = "sign is not fixed by physics under this weighting - see the code comment"
        clamp_note = (
            f"  floor      {gfg.FLOW_FLOOR:g} MW; {pinned.min()}-{pinned.max()} edges carrying "
            f"under it, read at {gfg.FLOW_FLOOR:g} MW.\n"
            f"             They are the weakest ties, so under loading the cuts fall on the\n"
            f"             corridors power is not using; a full corridor is a strong tie.\n")
    else:
        sign_note = "sign is not fixed by physics under this weighting - see the code comment"
        clamp_note = (
            f"  ceiling    {1.0 / gfg.FLOW_FLOOR:g}; {pinned.min()}-{pinned.max()} edges at "
            f"it (carrying under {gfg.FLOW_FLOOR:g} MW),\n"
            f"             which reads as maximally bound rather than as a cut. There is no\n"
            f"             lambda_2 plateau here; the cuts fall on the busiest corridors.\n")
    print(
        f"{CASE} +generators - {WEIGHTING}, {len(snaps)} snapshots  [{elapsed:.1f}s]\n"
        f"  dispatch   {DISPATCH} ({solved - started:.1f}s to dispatch and flow); "
        f"{shed_mwh:,.0f} MWh of load shed over the week\n"
        f"  graph      {n_nodes} nodes, structure built once; "
        f"components always {set(components.tolist())}\n"
        f"  loading    {series['max_loading'].min():.1%} to "
        f"{series['max_loading'].max():.1%} across the week; {int(congested.sum())} of "
        f"{len(snaps)} hours have a branch over rating\n"
        f"  lambda_2   {l2.min():.3e} at {l2.idxmin()}  (most nearly split)\n"
        f"             {l2.max():.3e} at {l2.idxmax()}  (most robust)"
        f"   - a {l2.max() / l2.min():,.0f}x swing\n"
        f"  corr       {np.corrcoef(l2, series['max_loading'])[0, 1]:+.3f} between lambda_2 "
        f"and max loading\n"
        f"             ({sign_note})\n"
        f"  clusters   k = {K}, cut {series['cut_share'].min():.1%}-"
        f"{series['cut_share'].max():.1%} of capacity; "
        f"{np.nanmean(churn):.2%} of nodes change zone per hour after alignment\n"
        + clamp_note
        + f"  wrote      {out}"
    )


if __name__ == "__main__":
    main()
