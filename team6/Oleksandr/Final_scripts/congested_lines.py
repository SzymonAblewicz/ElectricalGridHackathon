# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy>=1.14", "matplotlib", "pypsa", "highspy>=1.7"]
# ///
"""Which branches are congested, and how often, over the synthetic week.

`week_spectrum.py` flows all 168 hours of a kit case and keeps only the *worst* branch per
hour. This keeps every branch: an hour counts against a line or transformer when its |F| is
at or above LOADING_FRACTION of its rating, and the count over the week is drawn as a heat
map on the grid - the darker the red, the more hours that branch spent congested.

The data and the flow are week_spectrum's, imported rather than copied: the loads are the
case's own `loads_t.p_set`, the dispatch is its per-area pro-rata, and the flows come from
one pseudoinverse with the phase shifters included. So a count here and a `max_loading` or
`branches_over` there can never disagree about what flowed.

Writes, into `data/congestion/<CASE>/`:

  congested_lines_<CASE>_lf<frac>.csv   every branch, most-congested first: its summary
                                        columns, then a 0/1 mark for each hour
  congested_lines_<CASE>_lf<frac>.pdf   the map

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))  # week_spectrum, get_flow_graph, graph_lib all sit beside this one

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402

import week_spectrum as ws  # noqa: E402  (also puts the kit's flowmath on sys.path)

import matplotlib.pyplot as plt  # noqa: E402  (after week_spectrum, which sets Agg)
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import ListedColormap, Normalize  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

gfg, gwg = ws.gfg, ws.gwg


# ---- what to run ---- #

CASE = "WP2033_all-island"    # kit cases only: {WP,SV}{2024,2033}_{all-island,north-west}

# A branch is congested in an hour when |F| / s_nom >= this. 1.0 is "at or over rating";
# lower it to also catch branches that are merely highly loaded.
LOADING_FRACTION = 0.7

# "prorata": week_spectrum's own per-area pro-rata dispatch. It has no notion of a branch
#     rating, so the flow it produces can and does sit above 100% - this is "what would
#     physically flow if generation were shared out naively, network limits or not".
# "capacity_constrained": PyPSA's own LOPF (HiGHS solver), the same one the participant
#     kit's `gridkit.solve` wraps. s_nom is a hard constraint of the optimisation, so loading
#     tops out at ~100% wherever a redispatch can avoid the overload; load shedding (marginal
#     cost 10,000) is the last resort where it cannot. Costs more to run (168 LP solves).
DISPATCH = "capacity_constrained"          # "prorata" | "capacity_constrained"

# Reds from a quarter of the way in, so a branch congested for a single hour is still a
# visible pink rather than near-white.
HEAT = ListedColormap(plt.get_cmap("Reds")(np.linspace(0.25, 1.0, 256)))


# ---- dispatch ---- #


def capped_injections(network) -> pd.DataFrame:
    """Bus injections from a capacity-constrained economic redispatch, snapshot x bus.

    Mirrors the injections `ptdf_all.py` derives from its own LOPF solve, just kept in the
    (snapshot, bus) shape `week_spectrum.branch_flows` expects (the same shape
    `prorata_injections` returns), so it drops straight into the existing pseudoinverse flow
    math unchanged - only the dispatch feeding it differs.
    """
    network.optimize(network.snapshots, solver_name="highs",
                     solver_options={"output_flag": False}, progress=False)

    snaps = network.snapshots
    gp = network.generators_t.p.loc[snaps]
    gsign = network.generators["sign"].reindex(gp.columns).fillna(1.0)
    contrib = (gp * gsign).T.groupby(network.generators["bus"].reindex(gp.columns).to_numpy()).sum()
    p = pd.DataFrame(0.0, index=network.buses.index, columns=snaps)
    p.loc[contrib.index] += contrib

    ld = network.loads_t.p_set.loc[snaps]
    lcontrib = ld.T.groupby(network.loads["bus"].reindex(ld.columns).to_numpy()).sum()
    p.loc[lcontrib.index] -= lcontrib

    if len(network.links):
        p0 = network.links_t.p0.loc[snaps]
        for name in p0.columns:
            p.loc[network.links.at[name, "bus0"]] -= p0[name].to_numpy()
            p.loc[network.links.at[name, "bus1"]] += (
                p0[name].to_numpy() * network.links.at[name, "efficiency"])
    return p.T


# ---- output ---- #


def plot_map(case_dir: Path, table: pd.DataFrame, n_hours: int, path: Path,
            dispatch: str = DISPATCH) -> int:
    """Every branch on the map, congested ones coloured by how many hours they were.

    Positions are read and imputed the way `upgrade_lines.plot_upgrades` does it: from
    `buses.csv`, which keeps missing x/y as NaN (PyPSA would read them as 0 N 0 E), filled
    from directly connected neighbours twice, one hop per pass. Returns how many congested
    branches still could not be placed.
    """
    buses, branches = gwg.read_case(case_dir)
    buses = gwg.impute_coordinates(buses, branches)
    buses = gwg.impute_coordinates(buses, branches)
    xy = buses.set_index("name")[["x", "y"]]

    a = xy.reindex(table["bus0"]).to_numpy(float)
    b = xy.reindex(table["bus1"]).to_numpy(float)
    drawable = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    segments = np.stack([a, b], axis=1)
    count = table["hours_congested"].to_numpy()
    hot = count > 0

    fig, ax = plt.subplots(figsize=(8.5, 9.5))
    calm = drawable & ~hot
    ax.add_collection(LineCollection(segments[calm], colors="#cfcfcf", linewidths=0.4,
                                     zorder=1))
    ax.scatter(xy["x"], xy["y"], s=1.2, c="#8a8a8a", linewidths=0, zorder=2)

    norm = Normalize(1, max(int(count.max()), 2))
    sel = np.where(drawable & hot)[0]
    sel = sel[np.argsort(count[sel], kind="stable")]         # darkest drawn last, on top
    if len(sel):
        colours = HEAT(norm(count[sel]))
        size = 10 + 30 * count[sel] / count.max()
        ax.add_collection(LineCollection(
            segments[sel], colors=colours,
            linewidths=1.0 + 2.5 * count[sel] / count.max(), zorder=3))
        # A line also gets a dot at its midpoint: a Dublin circuit is a few km, so without it
        # it vanishes at island scale.
        is_tx = table["kind"].eq("Transformer").to_numpy()[sel]
        mid = segments[sel].mean(axis=1)
        ax.scatter(mid[~is_tx, 0], mid[~is_tx, 1], s=size[~is_tx], c=colours[~is_tx],
                   edgecolors="black", linewidths=0.3, zorder=4)
        # A transformer gets a ring on its real bus end instead, as cut_transformer_rings does
        # on the cluster maps: one to its own star point has ~zero length, and the star
        # point's position is only an average of its neighbours - the bus is the site.
        ends = table.iloc[sel]
        real = np.where(ends["bus0"].str.startswith("star:"), ends["bus1"], ends["bus0"])
        ring = xy.reindex(real).to_numpy(float)
        ax.scatter(ring[is_tx, 0], ring[is_tx, 1], s=2 * size[is_tx] + 20,
                   facecolors="none", edgecolors=colours[is_tx], linewidths=1.6, zorder=5)
        ax.legend(handles=[
            Line2D([], [], ls="", marker="o", mfc="#d1495b", mec="black", mew=0.3,
                   label=f"line ({int((~is_tx).sum())})"),
            Line2D([], [], ls="", marker="o", mfc="none", mec="#d1495b", mew=1.6,
                   label=f"transformer, ring on its bus ({int(is_tx.sum())})"),
        ], loc="upper left", fontsize=8, frameon=False)

    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=HEAT), ax=ax, shrink=0.6)
    bar.set_label(f"hours congested (of {n_hours})")

    ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(xy["y"]))))  # rough WGS84 fix
    ax.autoscale_view()
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    off_map = int((hot & ~drawable).sum())
    ax.set_title(f"{CASE} ({dispatch}) - branches at >= {LOADING_FRACTION:.0%} of rating, "
                 f"over {n_hours} hours\n{int(hot.sum())} of {len(table)} branches congested "
                 f"at least once; {off_map} of them not placeable", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return off_map


# ---- entry point ---- #


def main() -> None:
    if not LOADING_FRACTION > 0:
        raise ValueError(f"LOADING_FRACTION must be positive, not {LOADING_FRACTION}")
    if DISPATCH not in ("prorata", "capacity_constrained"):
        raise ValueError(f"DISPATCH must be 'prorata' or 'capacity_constrained', not "
                         f"{DISPATCH!r}")

    gfg.DATASET = "kit"
    root = gfg.case_root()
    case_dir = root / CASE
    if not case_dir.is_dir():
        available = sorted(p.name for p in root.iterdir() if p.is_dir())
        raise FileNotFoundError(f"no such kit case: {case_dir}\navailable: {', '.join(available)}")

    gfg.quiet()
    network = pypsa.Network(str(case_dir))
    snaps = network.snapshots

    if DISPATCH == "capacity_constrained":
        injections = capped_injections(network)
    else:
        _, injections = ws.prorata_injections(network)
    flows = ws.branch_flows(network, injections)            # |MW|, branch x snapshot

    import flowmath
    frame = flowmath.branches(network).reindex(flows.index)
    s_nom = frame["s_nom"].to_numpy(float)
    loading = flows.div(np.where(s_nom > 0, s_nom, np.nan), axis=0)
    congested = loading >= LOADING_FRACTION                 # NaN compares False

    count = congested.sum(axis=1)
    worst = loading.max(axis=1)
    table = pd.DataFrame({
        "kind": frame["kind"],
        "bus0": frame["bus0"].astype(str),
        "bus1": frame["bus1"].astype(str),
        "s_nom": s_nom,
        "congested": count > 0,
        "hours_congested": count,
        "share_of_hours": count / len(snaps),
        "max_loading": worst,
        "mean_loading": loading.mean(axis=1),
        "hour_of_max": loading.fillna(-np.inf).idxmax(axis=1).where(worst.notna()),
    })
    table.index.name = "branch"
    marks = congested.astype(int)
    marks.columns = [str(s) for s in snaps]
    table = pd.concat([table, marks], axis=1).sort_values(
        ["hours_congested", "max_loading"], ascending=False)

    out = gwg.DATA_DIR / "congestion" / CASE
    out.mkdir(parents=True, exist_ok=True)
    stem = f"congested_lines_{CASE}_lf{LOADING_FRACTION:g}"
    if DISPATCH == "capacity_constrained":
        stem += "_capped"
    table.to_csv(out / f"{stem}.csv")
    off_map = plot_map(case_dir, table, len(snaps), out / f"{stem}.pdf", DISPATCH)

    hot = table[table["congested"]]
    top = "\n".join(
        f"    {name:<22} {row['kind']:<11} {row['hours_congested']:>4} h   "
        f"max {row['max_loading']:.1%}"
        for name, row in hot.head(10).iterrows())
    print(
        f"{CASE} [{DISPATCH}] - {len(snaps)} snapshots, congested at >= "
        f"{LOADING_FRACTION:.0%} of rating\n"
        f"  branches   {len(hot)} of {len(table)} congested at least once "
        f"({off_map} not placeable on the map)\n"
        f"  hours      {int(congested.any(axis=0).sum())} of {len(snaps)} have at least one "
        f"congested branch\n"
        f"  most often\n{top}\n"
        f"  wrote      {out / (stem + '.csv')}\n"
        f"             {out / (stem + '.pdf')}"
    )


if __name__ == "__main__":
    main()
