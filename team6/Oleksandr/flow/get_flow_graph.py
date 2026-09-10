# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa"]
# ///
"""Weight the grid graph by what a DC power flow actually carries, then take its spectrum.

`get_weighted_graph.py` weights every corridor by `s_nom`, which is a *structural* reading:
it says where the grid is thin, and it says the same thing whatever the system is doing. This
weights the same corridors by the flow a DC power flow puts on them, which is an
*operational* reading: it says which corridors are full **in this dispatch**.

Two datasets, chosen by DATASET, and they need different amounts of work to reach a flow:

  "tytfs"  `generators.csv`, `loads.csv` and `links.csv` all ship a populated `p_set` -
           EirGrid's own study dispatch - so `n.lpf()` runs straight off the file. One
           snapshot, real planning data, no time axis.

  "kit"    the participant-kit networks: 168 hourly snapshots, but `p_set` shipped empty, so
           a dispatch has to be constructed before anything can flow. `kit_dispatch` picks
           one hour and fills it, pro-rata per synchronous area by default. The profiles are
           synthetic - generated wind, solar and demand, seed 42, no hour in them ever
           happened - so this is the dataset for asking "what would a windy Tuesday do",
           not for quoting a number.

Either way no optimisation and no cost assumption enters, unless DISPATCH is set to "lopf".

Two weightings, both read by the Laplacian as *affinity* ("a big number means these two
belong together"), which is the same requirement `spectrum.py` documents when it refuses
`reciprocal()`:

    headroom       s_nom - |F|     spare capacity. A corridor at its rating scores ~0 and
                                   becomes a spectral cut, so the clusters are zones power
                                   moves freely inside and the cut edges are the constraints.

    inverse_flow   1 / |F|         inverse throughput. A quiet corridor scores high, so the
                                   cuts land on the *busiest* corridors instead. The opposite
                                   reading from headroom, and a different hypothesis about
                                   where the zone boundaries are - worth running both.

On BALANCE, which is not a detail. The TYTFS dispatch does not balance: on WP2033 the main
synchronous area is +877 MW long, because the original PSS/E case was an AC solution whose
losses the lossless DC approximation does not reproduce, and because the interconnectors are
scheduled in. A power flow cannot leave that unbalanced - `L theta = P` has no solution
unless `sum(P) = 0` - so PyPSA hands the whole 877 MW to its slack bus and returns flows
anyway, silently.

That is not a rounding error in the result, it is the result. Left raw, the single most
loaded branch in the network is `2202-5202-1` at **188% of rating**, terminating at slack bus
5202, carrying power nobody generated. Under `headroom` that corridor scores ~0 and the
spectral cut falls on it - so the clustering would be reporting the location of PyPSA's slack
bus. Scaling the loads by 1.0998 to absorb the 877 MW drops that branch to 26%, a 162
percentage-point correction, and reorders the top five entirely. Hence "scale_loads" is the
default and "none" is kept only so the difference can be seen.

Written to `data/graphs/<case>/flow/`, a sibling of the `gen` and `buses` trees
`get_weighted_graph.py` writes, so a flow weighting can never be mistaken for a capacity one.
Generators are always nodes here; there is no buses-only variant.

Set the constants below and run the file.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # get_weighted_graph and spectrum live one level up

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402
from scipy.sparse import csgraph  # noqa: E402

import graph_lib as gwg  # noqa: E402
import spectrum as spec  # noqa: E402


def quiet() -> None:
    """Turn down PyPSA's logging, which prints a line per sub-network per solve.

    Four AC areas means four "Performing linear load-flow" lines before any result appears,
    plus a pandas 3.0 dtype warning nothing here can act on. Nothing below changes a number;
    it only changes what reaches the terminal.
    """
    for name in ("pypsa", "pypsa.io", "pypsa.network.io", "pypsa.network.power_flow",
                 "pypsa.consistency"):
        logging.getLogger(name).setLevel(logging.ERROR)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)


# ---- what to run ---- #
#
# DATASET picks which collection CASE names, and they are not interchangeable:
#
#   "tytfs"  grid_TF_Wind/data/pypsa/TYTFS2024_{WP,SV}{2024,2033}_V35_{transmission,full}
#            EirGrid's study files. One snapshot, "now", and a dispatch already in p_set.
#            Real planning data, no time axis. The northwest_* cases do not work, for the
#            reason get_weighted_graph.py gives: their buses.csv lacks the BUS_COLUMNS.
#
#   "kit"    grid_TF_Wind/participant-kit/networks/{WP,SV}{2024,2033}_{all-island,north-west}
#            168 hourly snapshots, and p_set shipped EMPTY - so a dispatch has to be made
#            before anything can flow (see kit_dispatch). The profiles are *synthetic*:
#            generated wind, solar and demand shapes, seed 42. No hour in them ever
#            happened. Read the kit README's LIMITATIONS before quoting a number.
#
# K is the number of clusters wanted. It does not have to be right first time - the run
# reports the eigengap, which is the spectrum's own opinion on how many clusters there are.

DATASET = "kit"               # "tytfs" | "kit"
CASE = "WP2033_all-island"    # a directory name under whichever DATASET selects
WEIGHTING = "headroom"        # "headroom" | "inverse_flow"
BALANCE = "scale_loads"       # "scale_loads" | "none" - read the docstring before changing
K = 6                         # clusters wanted
EMBEDDING = "sym"             # "sym" | "rw" | "unnorm"

# --- kit only; ignored when DATASET == "tytfs" --- #
#
# Which of the 168 hours to build the graph from. An integer indexes n.snapshots directly.
#
# "anchor" is the hour the kit anchored to its TYTFS state, so it is the one directly
# comparable with the tytfs run - on WP2033 that is 2030-01-17 18:00, also the peak-demand
# hour at 8,792 MW, matching the TYTFS case's load to within a megawatt.
#
# But **the anchor is not a stressed hour**, and picking it by default hides the point of
# having a week at all. Measured across all 168 hours of WP2033_all-island under "prorata":
#
#     anchor  (2030-01-17 18:00, idx  84)   max loading  92.9%   0 branches over rating
#     worst   (2030-01-20 17:00, idx 155)   max loading 127.5%   3 branches over rating
#     calmest (2030-01-16 03:00, idx  27)   max loading  63.6%   0 branches over rating
#     56 of the 168 hours have at least one branch over its rating.
#
# That matters for the spectrum, not just for the loading table. At the anchor hour nothing
# is pinned to the floor, so lambda_2 is 3.8e-03 - indistinguishable from the capacity-
# weighted graph's 4.3e-03, i.e. the flows have told you nothing. At idx 155 three corridors
# saturate, lambda_2 falls to 6.0e-10 (at HEADROOM_FLOOR=1e-7), the eigengap drops to k=2: the graph has nearly
# split in two. Peak demand is not peak congestion; a windy shoulder hour is.
#
# The three that bind are always the same, and they are the same chain that tops the TYTFS
# case at 100%: line 3581-89516-1 (123 MVA) with transformers T89510-89515-1 and
# T89515-89516-1 (125 MVA each) behind it. Two independent datasets, one bottleneck.
SNAPSHOT = 155                # "anchor" | "peak" | 0..167

# How to turn 168 hours of *availability* into one hour of dispatch:
#
#   "prorata"  every unit at the same fraction of what it could produce, the fraction set so
#              the system balances. No costs, no merit order, no optimiser - which is the
#              point. Available plant runs 12.7-22.6 GW against 5.6-8.8 GW of demand, so the
#              fraction is 0.25-0.69, and forcing every unit to contribute regardless of
#              where it sits is what makes the constrained corridors show up.
#
#   "lopf"     least-cost dispatch via gridkit.solve. Realistic, but it brings the kit's
#              placeholder costs with it (wind bids -1 EUR/MWh) and it respects every line
#              rating, so nothing can exceed 100% and no corridor is ever driven to a cut.
DISPATCH = "prorata"          # "prorata" | "lopf"


# Eigenpairs computed beyond K, so the gap *after* the K-th is visible.
EXTRA = 5

# A corridor at exactly its rating would otherwise leave the sparse matrix altogether,
# changing the node set and the component count. The floor keeps the edge present and
# negligible, which is what "this is a cut" should mean.
#
# At 1e-7 against a 3,846 MVA backbone the ratio is 4e-11, so a saturated corridor is a
# decisive cut rather than a weak edge. What it does NOT buy is severity ranking: every
# corridor at or above 100% clamps to the same value whatever its overload, so lambda_2
# plateaus once anything saturates. Lowering the floor moves that plateau down, it does not
# remove it - a weight that keeps responding past 100% (signed headroom, or s_nom/|F|) would.
HEADROOM_FLOOR = 1e-7         # MVA

# On the tytfs cases 110 of the 980 branches carry no flow at all, and 1/0 is not a weight.
FLOW_FLOOR = 1.0              # MW

DATASETS = ("tytfs", "kit")
WEIGHTINGS = ("headroom", "inverse_flow")
BALANCES = ("scale_loads", "none")
DISPATCHES = ("prorata", "lopf")

#: Generators that are modelling devices rather than plant, and must not be given a
#: pro-rata share of the load: the per-bus "shed <bus>" units priced at the value of lost
#: load, and the sign=-1 export sinks. Left at zero output.
ARTIFICIAL_CARRIERS = ("load shedding",)


# ---- output location ---- #


def out_dir(case: str, stage: str) -> Path:
    """`data/graphs/<case>/flow/<stage>/`, created if missing.

    `gwg.out_dir()` keys on the node set - `gen` or `buses` - because that is what has to
    match between its stages. Here the node set is always "generators included", and what
    has to be kept apart is the *weighting*: a capacity graph and a flow graph of the same
    case describe the same wires under different questions, and silently mixing their
    matrices would be much worse than mixing two node sets. So `flow` sits where `gen` and
    `buses` sit, rather than underneath one of them.
    """
    path = gwg.OUT_DIR / case / "flow" / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_root() -> Path:
    """The directory CASE names, per DATASET."""
    if DATASET == "kit":
        # gwg.PYPSA_DIR is <repo>/grid_TF_Wind/data/pypsa, so its grandparent is grid_TF_Wind.
        return gwg.PYPSA_DIR.parent.parent / "participant-kit" / "networks"
    return gwg.PYPSA_DIR


# ---- turning 168 synthetic hours into one dispatch ---- #


def kit_dispatch(network) -> tuple[str, float]:
    """Reduce a participant-kit network to one snapshot with a dispatch in its static p_set.

    The kit ships `p_set` empty on purpose - in PyPSA 1.x a populated `p_set` is an equality
    constraint that would pin the dispatch and make the optimisation infeasible - so nothing
    can flow until something fills it. That is the trap `gridkit.freeze_dispatch` exists for.

    What this does is narrow the network until it looks exactly like a tytfs case: one
    snapshot, and every injection in a static column. The time series are then cleared, so
    there is one source of truth rather than a static value and a series that disagree.
    Everything downstream - area_injection, rebalance, branch_flow - reads static p_set and
    snapshots[0], and so needs no knowledge of which dataset it was handed.

    Returns the snapshot chosen and, for "prorata", the fraction of availability dispatched.
    """
    if SNAPSHOT == "anchor":
        stamp = pd.Timestamp(network.meta["anchor_snapshot"])
    elif SNAPSHOT == "peak":
        stamp = network.loads_t.p_set.sum(axis=1).idxmax()
    else:
        stamp = network.snapshots[int(SNAPSHOT)]
    if stamp not in network.snapshots:
        raise ValueError(f"{stamp} is not one of this network's snapshots")

    load_at = network.loads_t.p_set.loc[stamp].reindex(network.loads.index).fillna(0.0)

    # Availability: p_nom scaled by the weather profile where there is one, p_nom flat where
    # there is not. Only wind and solar carry a p_max_pu series in these networks.
    available = network.generators["p_nom"].astype(float).copy()
    profiled = network.generators_t.p_max_pu.columns
    available[profiled] = (network.generators_t.p_max_pu.loc[stamp, profiled]
                           * network.generators.loc[profiled, "p_nom"])

    plant = ~(network.generators["carrier"].isin(ARTIFICIAL_CARRIERS)
              | (network.generators["sign"] < 0))
    offered = available.where(plant, 0.0)

    fraction = float("nan")
    if DISPATCH == "prorata":
        # Pro-rata has to be struck **per synchronous area**, not system-wide. Three of the
        # four areas are a single GB-side interconnector terminal carrying one `import` unit
        # and no load, and no dispatch has been chosen for the links, so nothing can cross
        # out of them. A system-wide fraction hands those three units 1,082 MW that is
        # stranded the instant it is generated, leaving the island 1,082 MW short and
        # rebalance() cutting real demand by 20% to hide it.
        network.determine_network_topology()
        area = network.buses["sub_network"]
        by_area_load = load_at.groupby(area.reindex(network.loads["bus"]).to_numpy()).sum()
        generator_area = area.reindex(network.generators["bus"]).to_numpy()
        by_area_offered = offered.groupby(generator_area).sum()
        share = (by_area_load / by_area_offered).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        if not (share > 0).any():
            raise ValueError(f"no area has both load and available plant at {stamp}")
        dispatch = offered * pd.Series(generator_area,
                                      index=network.generators.index).map(share).fillna(0.0)
        fraction = float(share.reindex([by_area_load.idxmax()]).iloc[0])
    else:
        import gridkit  # only the lopf path needs the kit's solver wrapper
        gridkit.solve(network, [stamp])
        if not len(network.generators_t.p.columns):
            raise RuntimeError("the lopf produced no dispatch")
        dispatch = network.generators_t.p.loc[stamp].reindex(
            network.generators.index).fillna(0.0)

    network.set_snapshots([stamp])
    network.generators["p_set"] = dispatch.astype(float)
    network.loads["p_set"] = load_at.astype(float)
    # The kit ships link p_set as NaN. Under "prorata" no schedule was chosen for them, so
    # they are pinned at zero; under "lopf" the solver picked one, so that is copied in.
    # Either way the static column has to say what the flow will be, because that is what
    # area_injection reads.
    if len(network.links):
        network.links["p_set"] = (
            network.links_t.p0.loc[stamp].reindex(network.links.index).fillna(0.0)
            if DISPATCH == "lopf" and len(network.links_t.p0.columns) else 0.0)

    # Drop every time series now that the statics carry the answer. Left in place, lpf would
    # prefer the series and the printed p_set figures would describe a different dispatch
    # from the one that flowed.
    for frame, attr in ((network.generators_t, "p_set"), (network.generators_t, "p_max_pu"),
                        (network.loads_t, "p_set")):
        setattr(frame, attr, getattr(frame, attr).iloc[:, :0])

    return str(stamp), fraction


# ---- the power flow ---- #


def area_injection(network) -> tuple[pd.Series, pd.Series]:
    """Net MW injected per bus and per synchronous area, from the `p_set` columns.

    Read before `lpf` rather than after, because after is too late: `lpf` will already have
    balanced the system by whatever means, and the imbalance this is looking for will have
    been absorbed into a slack bus and become invisible.

    Links are injections, not branches - a DC link takes `p_set` out at `bus0` and puts
    `p_set * efficiency` in at `bus1`, which is exactly why the GB-side terminals form
    synchronous areas of their own.
    """
    network.determine_network_topology()
    p = pd.Series(0.0, index=network.buses.index)
    signed = network.generators["p_set"] * network.generators["sign"]
    p = p.add(signed.groupby(network.generators["bus"]).sum(), fill_value=0.0)
    p = p.sub(network.loads["p_set"].groupby(network.loads["bus"]).sum(), fill_value=0.0)
    if len(network.links):
        # fillna, and vectorised rather than a row loop, for one reason: the participant-kit
        # networks ship link p_set as NaN rather than 0. Subtracting that turns the link's
        # two end buses into NaN, and the groupby below then drops them from their area's
        # total *silently* - which on WP2033 hid 696 MW of real injection at the Irish ends
        # of Moyle, EWIC and Greenlink and made the imbalance look 1.6x its true size.
        schedule = network.links["p_set"].astype(float).fillna(0.0)
        efficiency = network.links["efficiency"].astype(float).fillna(1.0)
        p = p.sub(schedule.groupby(network.links["bus0"]).sum(), fill_value=0.0)
        p = p.add((schedule * efficiency).groupby(network.links["bus1"]).sum(),
                  fill_value=0.0)
    return p, p.groupby(network.buses["sub_network"]).sum()


def rebalance(network) -> tuple[float, float, str]:
    """Scale the loads of the main AC area so its injections sum to zero.

    A distributed slack, in the only form a power flow will accept. The alternative - the
    single-bus slack PyPSA falls back on - concentrates the entire mismatch at one bus and
    writes it into every branch around it.

    Only the largest area is touched. The other three are the single GB-side terminals of
    Moyle, EWIC and Greenlink; they carry no load to scale and no branch to flow on, and
    `gwg.adjacency()` drops them as branchless before the graph is built.

    Returns the scale factor, the MW it absorbed, and which area was scaled.
    """
    _, per_area = area_injection(network)
    main = per_area.abs().idxmax()
    area_of_load = network.buses["sub_network"].reindex(network.loads["bus"]).to_numpy()
    inside = area_of_load == main
    total = float(network.loads.loc[inside, "p_set"].sum())
    excess = float(per_area[main])
    if total <= 0:
        raise ValueError(f"area {main} carries {excess:,.1f} MW of imbalance and no load "
                         f"to spread it over")
    factor = (total + excess) / total
    network.loads.loc[inside, "p_set"] = network.loads.loc[inside, "p_set"] * factor
    return factor, excess, str(main)


def branch_flow(network) -> pd.Series:
    """|MW| on every line and transformer, indexed by branch name.

    `p0` is the flow into the branch at `bus0`; its sign only records which end `bus0`
    happens to be in the CSV, so it is dropped here. The two frames are concatenated in the
    same order `gwg.read_case()` concatenates lines and transformers, so the result lines up
    with the branch table without a reindex - but one is done anyway, because "lines up"
    is not something to leave to ordering.
    """
    snapshot = network.snapshots[0]
    return pd.concat([network.lines_t.p0.loc[snapshot],
                      network.transformers_t.p0.loc[snapshot]]).abs()


# ---- weighting ---- #


def branch_weights(branches: pd.DataFrame, flow: pd.Series) -> np.ndarray:
    """The affinity of each corridor, under WEIGHTING."""
    f = flow.reindex(branches["name"]).to_numpy(float)
    if np.isnan(f).any():
        missing = branches["name"][np.isnan(f)].tolist()
        raise ValueError(f"no flow computed for {len(missing)} branches: {missing[:5]}")
    if WEIGHTING == "headroom":
        return np.maximum(branches["s_nom"].to_numpy(float) - f, HEADROOM_FLOOR)
    return 1.0 / np.maximum(f, FLOW_FLOOR)


def generator_weights(generators: pd.DataFrame, dispatch: pd.Series) -> np.ndarray:
    """The affinity of each generator-to-bus edge, under the same reading.

    `gwg.adjacency()` rates this edge at `p_nom`, the same capacity reading it gives a
    branch's `s_nom`. Keeping the analogy: under `headroom` the edge is worth the output the
    unit is *not* producing, and under `inverse_flow` the inverse of what it is.
    """
    p = dispatch.reindex(generators["name"]).fillna(0.0).abs().to_numpy(float)
    if WEIGHTING == "headroom":
        return np.maximum(generators["p_nom"].to_numpy(float) - p, HEADROOM_FLOOR)
    return 1.0 / np.maximum(p, FLOW_FLOOR)


# ---- entry point ---- #


def main() -> None:
    if DATASET not in DATASETS:
        raise ValueError(f"DATASET must be one of {DATASETS}, not {DATASET!r}")
    if WEIGHTING not in WEIGHTINGS:
        raise ValueError(f"WEIGHTING must be one of {WEIGHTINGS}, not {WEIGHTING!r}")
    if BALANCE not in BALANCES:
        raise ValueError(f"BALANCE must be one of {BALANCES}, not {BALANCE!r}")
    if DISPATCH not in DISPATCHES:
        raise ValueError(f"DISPATCH must be one of {DISPATCHES}, not {DISPATCH!r}")
    if EMBEDDING not in spec.EMBEDDINGS:
        raise ValueError(f"EMBEDDING must be one of {spec.EMBEDDINGS}, not {EMBEDDING!r}")

    root = case_root()
    case_dir = root / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available under DATASET={DATASET!r}: "
            f"{', '.join(sorted(p.name for p in root.iterdir() if p.is_dir()))}")

    quiet()
    network = pypsa.Network(str(case_dir))

    stamp, fraction = ("now", float("nan"))
    if DATASET == "kit":
        stamp, fraction = kit_dispatch(network)

    _, before = area_injection(network)
    # Captured before rebalance(), which rewrites loads.p_set in place - reported afterwards
    # it would be the scaled figure, which is not the dispatch the case ships.
    shipped_gen = float((network.generators["p_set"] * network.generators["sign"]).sum())
    shipped_load = float(network.loads["p_set"].sum())

    factor, excess, area = (1.0, 0.0, "")
    if BALANCE == "scale_loads":
        factor, excess, area = rebalance(network)

    network.lpf()

    # If the injections balanced, lpf had nothing to invent and p is p_set. If they did not,
    # this is the size of what it made up, and every flow below carries a share of it.
    invented = float((network.generators_t.p.iloc[0]
                      - network.generators["p_set"].reindex(network.generators_t.p.columns)
                      ).abs().sum())

    flow = branch_flow(network)
    buses, branches = gwg.read_case(case_dir)
    generators = gwg.read_generators(case_dir)

    s_nom = branches["s_nom"].to_numpy(float)
    loading = flow.reindex(branches["name"]).to_numpy(float) / np.where(s_nom > 0, s_nom, np.nan)

    # The weight column is overwritten rather than a new adjacency written, so corridor
    # aggregation, the generator-node block and the branchless-bus drop all stay in
    # get_weighted_graph.py and there is one implementation of each.
    branches = branches.assign(s_nom=branch_weights(branches, flow))
    generators = generators.assign(
        p_nom=generator_weights(generators, network.generators["p_set"]))

    A, node_index = gwg.adjacency(buses, branches, generators)
    D, L, L_sym = gwg.matrices(A)

    construction = out_dir(CASE, "construction")
    node_index.to_csv(construction / "node_index.csv")
    gwg.save(A, construction, WEIGHTING)
    # Estimated for the plot only, same as get_weighted_graph.py's own
    # construction graph - the CSV above keeps the real, ungeocoded x/y.
    gwg.plot(A, gwg.geocode(node_index, branches), construction / f"graph_{WEIGHTING}.pdf")

    # "sym" and "rw" differ only in a rescaling cluster.py applies afterwards, so both
    # decompose the same matrix - the same choice spectrum.py makes.
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
    spec.K = K  # spec.plot marks the configured K from its own module global
    spec.plot(values, suggested, clustering / f"{stem}.pdf",
              f"{CASE} +generators — {EMBEDDING}, {WEIGHTING}")

    weights = branches["s_nom"]
    # The two weightings pin at opposite ends. Under `headroom` the clamp is a floor and a
    # pinned corridor is a cut; under `inverse_flow` it is a ceiling, and a pinned corridor
    # is one carrying nothing - the *most* strongly bound edge in the graph, not the least.
    if WEIGHTING == "headroom":
        pinned = int((weights <= HEADROOM_FLOOR * (1 + 1e-9)).sum())
        pinned_note = (f"{pinned} at the floor ({HEADROOM_FLOOR:g}), read as cuts")
    else:
        pinned = int((weights >= (1.0 / FLOW_FLOOR) * (1 - 1e-9)).sum())
        pinned_note = (f"{pinned} at the ceiling ({1.0 / FLOW_FLOOR:g}), carrying under "
                       f"{FLOW_FLOOR:g} MW and so read as maximally bound")

    # Counted, not predicted. A near-cut per saturated corridor would suggest
    # pinned + 1 pieces, but two cuts on one boundary, or two in parallel, split the graph
    # no further than one does - so the spectrum is asked rather than told.
    near_zero = int((values < 1e-4).sum())
    listing = "\n".join(
        f"    {i + 1:>2}. {v:>13.6g}"
        + (f"   gap {gaps[i]:>11.6g}" if i < len(gaps) else "")
        + ("  <- largest" if i + 1 == suggested else "")
        for i, v in enumerate(values))
    if DATASET == "kit":
        source = (f"synthetic hour {stamp}"
                  + (f", every unit at {fraction:.1%} of availability"
                     if DISPATCH == "prorata" else ", least-cost lopf"))
    else:
        source = "the case's own p_set"
    print(
        f"{CASE} +generators - {WEIGHTING}  [{DATASET}]\n"
        f"  dispatch   {shipped_gen:,.0f} MW generation, {shipped_load:,.0f} MW load, "
        f"from {source}\n"
        f"  imbalance  {before.abs().max():,.1f} MW in the largest area before balancing\n"
        + (f"  balanced   loads in area {area} scaled by {factor:.6f} to absorb "
           f"{excess:+,.1f} MW\n"
           if BALANCE == "scale_loads" else
           "  balanced   NOT balanced (BALANCE='none'); the slack bus absorbed the lot\n")
        + f"  invented   {invented:,.1f} MW written into p by lpf that was not in p_set"
          f"{'  <- should be ~0' if BALANCE == 'scale_loads' else '  <- this is in every flow below'}\n"
        f"  flows      {len(flow)} branches, max loading {np.nanmax(loading):.1%}, "
        f"{int(np.nansum(loading > 1))} over rating, {int((flow < 1e-9).sum())} carrying nothing\n"
        f"  weights    {weights.min():,.4g} to {weights.max():,.4g} per corridor, "
        f"{pinned_note}\n"
        f"  graph      {n_nodes} nodes in {components} component"
        f"{'s' if components != 1 else ''}, {A.nnz // 2} edges\n"
        f"  spectrum   K = {K} configured, eigengap suggests k = {suggested}\n"
        f"{listing}\n"
        f"             ({near_zero} eigenvalue{'s' if near_zero != 1 else ''} below 1e-4, so the graph is\n"
        f"              connected but nearly in {near_zero} piece{'s' if near_zero != 1 else ''}"
        f" - and where it is nearly cut\n"
        f"              is the finding. Expect fewer of these than pinned corridors:\n"
        f"              two cuts on one boundary split the graph no further than one.)\n"
        f"  wrote      {construction}\n"
        f"             {clustering}"
    )


if __name__ == "__main__":
    main()
