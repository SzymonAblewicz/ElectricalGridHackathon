# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "pypsa"]
# ///
"""Reinforce the most loaded line, re-run the power flow, repeat until nothing is overloaded.

A DC power flow on a dispatch you choose, then a greedy reinforcement loop: find the line
carrying the largest fraction of its rating, move it one rung up the voltage ladder, run
`n.lpf()` again, and keep going until no branch exceeds THRESHOLD.

    110 -> 220 -> 275 -> 380 kV

Those are the voltages that exist in the TYTFS transmission cases. There is no 400 kV level
in this data - the Irish EHV level is 380 kV - and no 38 kV either, which appears only in the
`_full` cases with the distribution buses in them. The three interconnector artefacts (150,
260 and 365 kV) are not rungs; a line sitting on one snaps up to the next real rung, because
the ladder step is "the next voltage strictly above this one".

On what an upgrade changes. A line in these cases has no voltage of its own: `lines.csv` is
`name,bus0,bus1,x,r,b,s_nom,length,...` and PyPSA derives `v_nom` from `bus0` alone. So a
voltage upgrade has to be expressed as a change to something the file actually carries, and
the thing it carries is `s_nom` - the line's maximum power. Rebuilding at the higher voltage
carries the same current up a higher potential, so

    s_nom_new = s_nom_old * V_new / V_old

and 110 -> 220 doubles the rating. The line's own voltage class is tracked in a column beside
the network rather than by retagging its end buses, because retagging bus0 silently
reinterprets every *other* line on that bus at the new voltage and leaves the transformers
feeding it at the wrong ratio, with no PyPSA consistency check to catch it.

**Read this before reading the results.** `x` and `r` are left alone, so the impedances do not
move, and a DC power flow routes power by impedance alone. The consequence is that the flows
are identical in every iteration: only the denominator of the loading grows. The loop is
therefore not a redistribution study - it does not and cannot show power moving onto a
reinforced corridor, and it cannot exhibit Braess's paradox. What it does show is the cheapest
honest thing: given this dispatch and these impedances, how far up the ladder each corridor
has to go before it is inside its rating. Scaling `x` by (V_old/V_new)^2 alongside `s_nom`
would make the flows move; that is a different model and this script does not make it.

On BALANCE, which is not a detail - the same warning `get_flow_graph.py` carries. The TYTFS
dispatch does not balance, and scaling generation by DISPATCH makes the mismatch far bigger:
at DISPATCH_BASE = "p_nom" with every unit at 1.0 the case offers 26,266 MW of generation
against 8,792 MW of load. A power flow cannot leave that unbalanced - `L theta = P` has no
solution unless `sum(P) = 0` - so PyPSA hands the lot to its slack bus and returns flows
anyway, silently, which puts a fictitious 17 GW through the corridors around one bus and
makes that bus's neighbourhood win "most loaded line" every iteration. "scale_loads" spreads
it over the loads of the main synchronous area instead. `rebalance()` is imported from
`get_flow_graph.py` rather than reimplemented, so there is one of it.

The run reports the leftover "invented" MW in two parts, and only the first is a warning. The
main area's figure is the one that has to be ~0. The other is the three GB-side interconnector
terminals - Moyle, EWIC and Greenlink - which are single-bus synchronous areas of their own,
each carrying a link into Ireland and an import pseudo-generator whose `p_set` the case ships
blank. `rebalance()` cannot scale them because they have no load, so their local slack
generators supply their links: 80 + 300 + 300 = 680 MW on WP2033. That is the GB side of the
interconnector, not an artefact, and being one bus each it never crosses an AC branch on this
island - so it cannot reach any loading in the table.

DISPATCH_BASE is worth a moment too. "p_nom" reads your fractions as a share of nameplate, so
1.0 is every unit flat out - a stress case, and the reason the loads have to triple to absorb
it. "p_set" reads them as a share of EirGrid's own study dispatch, so 1.0 reproduces the case
exactly as shipped and the rebalance is the 877 MW correction rather than a 17 GW one.

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER.parent))  # get_weighted_graph lives one level up

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pypsa  # noqa: E402

import get_flow_graph as gfg  # noqa: E402
import graph_lib as gwg  # noqa: E402


# ---- what to run ---- #
#
# A case is a directory name under grid_TF_Wind/data/pypsa, of the shape
# TYTFS2024_{WP,SV}{2024,2033}_V35_{transmission,full}. The northwest_* cases do not work,
# for the reason get_weighted_graph.py gives: their buses.csv lacks the BUS_COLUMNS.

CASE = "TYTFS2024_WP2033_V35_transmission"

# Generator name -> the fraction of its power it runs at, 0 to 1. Anything not named here
# runs at DEFAULT_DISPATCH. Names are the PSS/E ids in generators.csv - "10271-1", "33174-1",
# "90540-LH" - and an unknown one is an error rather than a silent no-op, because a typo in a
# generator id is otherwise indistinguishable from a unit that simply does not move.
DISPATCH: dict[str, float] = {}
DEFAULT_DISPATCH = 1.0

# What the fraction is a fraction *of*.
#   "p_nom"  nameplate. 1.0 is every unit flat out; on WP2033 that is 26,266 MW against
#            8,792 MW of load, so "scale_loads" nearly triples the loads to absorb it.
#   "p_set"  the case's own TYTFS study dispatch. 1.0 reproduces the case as shipped.
DISPATCH_BASE = "p_nom"       # "p_nom" | "p_set"

# A branch above this fraction of its rating is overloaded. The loop runs until nothing is.
THRESHOLD = 0.75

BALANCE = "scale_loads"       # "scale_loads" | "none" - read the docstring before changing

# A cap, not a target. Every upgrade strictly raises one line's rating and the flows never
# move, so the loop cannot cycle - but an unreachable threshold would otherwise be found only
# by waiting for every line in the network to reach 380 kV.
MAX_UPGRADES = 1000


# The voltages that exist in these cases, in order. A line steps to the next one strictly
# above its current voltage, which both walks the ladder and snaps the 150 / 260 / 365 kV
# interconnector artefacts onto it.
LADDER = (110.0, 220.0, 275.0, 380.0)

BASES = ("p_nom", "p_set")


# ---- output location ---- #


def out_dir(case: str) -> Path:
    """`data/graphs/<case>/upgrade/`, created if missing.

    A sibling of the `gen`, `buses` and `flow` trees. This script writes a reinforced network
    rather than a graph, and filing it under one of those would invite reading its CSVs as if
    they described the case as shipped. They do not: they describe the case after the loop has
    rewritten some of its ratings.
    """
    path = gwg.OUT_DIR / case / "upgrade"
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- dispatch ---- #


def apply_dispatch(network, dispatch: dict[str, float], base: str) -> tuple[float, float]:
    """Set every generator's `p_set` to its fraction of `base`.

    Returns the signed MW before and after, `sign` included - it is -1 on the six
    interconnector export pseudo-generators, so an unsigned sum would count an export as
    generation. Those six ship a blank `p_set`; under "p_set" they stay at zero, and under
    "p_nom" each import/export pair takes the same p_nom with opposite signs and nets to
    nothing at its terminal, which is the neutral reading of "no interconnector schedule".
    """
    if base not in BASES:
        raise ValueError(f"DISPATCH_BASE must be one of {BASES}, not {base!r}")

    unknown = sorted(set(dispatch) - set(network.generators.index))
    if unknown:
        raise KeyError(
            f"{len(unknown)} generator name(s) in DISPATCH are not in this case: "
            f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''}\n"
            f"names look like {list(network.generators.index[:3])}")

    bad = {k: v for k, v in dispatch.items() if not 0.0 <= float(v) <= 1.0}
    if bad:
        raise ValueError(f"DISPATCH fractions must be between 0 and 1: {bad}")

    reference = network.generators[base].fillna(0.0)
    fraction = pd.Series(DEFAULT_DISPATCH, index=network.generators.index, dtype=float)
    fraction.update(pd.Series(dispatch, dtype=float))

    before = float((network.generators["p_set"].fillna(0.0) * network.generators["sign"]).sum())
    network.generators["p_set"] = reference * fraction
    after = float((network.generators["p_set"] * network.generators["sign"]).sum())
    return before, after


# ---- the ladder ---- #


def next_voltage(v: float) -> float | None:
    """The next rung strictly above `v`, or None if there is none.

    "Strictly above" is doing two jobs. On a line already at a rung it is the step; on one of
    the 150 / 260 / 365 kV interconnector buses, which are star points and far terminals
    rather than a level anyone builds at, it snaps to the rung above.
    """
    return next((rung for rung in LADDER if rung > v), None)


# ---- loading ---- #


def branch_table(network) -> pd.DataFrame:
    """One row per line and transformer: kind, rating, and the voltage a line sits at.

    Built once. `s_nom` is mutated in place by the loop and the flow is refreshed each
    iteration; everything else here is fixed for the run.

    A line's voltage is read from `bus0`, which is where PyPSA reads it from too, and the two
    ends agree on every line in these cases - verified, not assumed: `bus0` and `bus1` carry
    the same `v_nom` for all 755. Transformers get NaN, which is the point of them.
    """
    lines = pd.DataFrame({
        "kind": "line",
        "s_nom": network.lines["s_nom"].astype(float),
        "voltage": network.lines["bus0"].map(network.buses["v_nom"]).astype(float),
        "s_nom_source": network.lines.get("s_nom_source", pd.Series("", index=network.lines.index)),
    })
    transformers = pd.DataFrame({
        "kind": "transformer",
        "s_nom": network.transformers["s_nom"].astype(float),
        "voltage": np.nan,
        "s_nom_source": network.transformers.get(
            "s_nom_source", pd.Series("", index=network.transformers.index)),
    })
    table = pd.concat([lines, transformers])
    table.index.name = "branch"
    return table


def loading(network, table: pd.DataFrame) -> pd.Series:
    """|MW| carried as a fraction of rating, for every line and transformer.

    `gfg.branch_flow()` is the flow half of this and already concatenates the two frames in
    this order; the reindex is kept anyway, because "lines up" is not something to leave to
    ordering. A rating of zero would divide to infinity and win every iteration, so it reads
    as NaN and is ranked out.
    """
    flow = gfg.branch_flow(network).reindex(table.index)
    if flow.isna().any():
        missing = table.index[flow.isna()].tolist()
        raise ValueError(f"no flow computed for {len(missing)} branches: {missing[:5]}")
    s_nom = table["s_nom"].to_numpy(float)
    return pd.Series(flow.to_numpy(float) / np.where(s_nom > 0, s_nom, np.nan), index=table.index)


def worst_upgradable(table: pd.DataFrame, load: pd.Series) -> str | None:
    """The most loaded line that is over THRESHOLD and still has a rung above it.

    Not simply "the most loaded branch": a transformer has no voltage class to step and a line
    already at 380 kV has nowhere to go, so ranking those in would stall the loop while lines
    below them were still upgradable.
    """
    lines = table["kind"].eq("line") & table["voltage"].map(next_voltage).notna()
    candidates = load.where(lines & load.gt(THRESHOLD)).dropna()
    return None if candidates.empty else str(candidates.idxmax())


# ---- entry point ---- #


def main() -> None:
    if BALANCE not in gfg.BALANCES:
        raise ValueError(f"BALANCE must be one of {gfg.BALANCES}, not {BALANCE!r}")
    if not 0.0 < THRESHOLD:
        raise ValueError(f"THRESHOLD must be positive, not {THRESHOLD}")

    case_dir = gwg.PYPSA_DIR / CASE
    if not case_dir.is_dir():
        raise FileNotFoundError(
            f"no such case: {case_dir}\n"
            f"available: {', '.join(sorted(p.name for p in gwg.PYPSA_DIR.iterdir() if p.is_dir()))}")

    gfg.quiet()
    network = pypsa.Network(str(case_dir))

    shipped, dispatched = apply_dispatch(network, DISPATCH, DISPATCH_BASE)
    shipped_load = float(network.loads["p_set"].sum())

    # Read before rebalance(), which rewrites loads.p_set in place, and before lpf, which
    # would have absorbed the imbalance into a slack bus and made it invisible.
    _, before = gfg.area_injection(network)
    main_area = str(before.abs().idxmax())  # the area rebalance() will scale

    factor, excess, area = (1.0, 0.0, "")
    if BALANCE == "scale_loads":
        factor, excess, area = gfg.rebalance(network)

    table = branch_table(network)
    original = table["s_nom"].copy()
    start_voltage = table["voltage"].copy()
    first: pd.Series | None = None
    log: list[dict] = []

    # The dispatch does not change inside the loop and a rating does not enter `L theta = P`,
    # so the injections are balanced once, out here, rather than every iteration.
    for step in range(MAX_UPGRADES + 1):
        network.lpf()
        load = loading(network, table)
        # Captured rather than recomputed at the end as flow/original. That would give the
        # same answer only because nothing here moves a flow, so it would become quietly
        # wrong the day someone rescales `x` alongside `s_nom`.
        if first is None:
            first = load.copy()

        over = load.gt(THRESHOLD)
        if not over.any():
            outcome = "converged"
            break

        branch = worst_upgradable(table, load)
        if branch is None:
            outcome = "stalled"
            break
        if step == MAX_UPGRADES:
            outcome = "capped"
            break

        v_now = float(table.at[branch, "voltage"])
        v_next = next_voltage(v_now)
        scale = v_next / v_now
        s_now = float(table.at[branch, "s_nom"])

        log.append({
            "step": len(log) + 1, "branch": branch, "loading": float(load[branch]),
            "v_from": v_now, "v_to": v_next, "s_nom_from": s_now, "s_nom_to": s_now * scale,
            "overloaded_before": int(over.sum()), "max_loading_before": float(load.max()),
        })

        table.loc[branch, ["voltage", "s_nom"]] = [v_next, s_now * scale]
        network.lines.loc[branch, "s_nom"] = s_now * scale

    # ---- what happened ---- #

    # MW that lpf wrote into `p` which was not in `p_set`, split by where it landed.
    #
    # The split is the whole point. `rebalance()` scales the loads of the main synchronous
    # area only, because it is the only area with any load to scale - the other three are the
    # single GB-side terminals of Moyle, EWIC and Greenlink, one bus each, carrying a link and
    # an import pseudo-generator whose `p_set` the case ships blank. Their local slack
    # generators therefore supply their links, which is not an artefact but the GB side of the
    # interconnector doing its job, and none of it crosses an AC branch on this island. So
    # `main` is the number that has to be ~0 for the loadings below to mean anything, and
    # `far` is a schedule.
    delta = (network.generators_t.p.iloc[0]
             - network.generators["p_set"].reindex(network.generators_t.p.columns)).abs()
    in_main = (network.buses["sub_network"]
               .reindex(network.generators["bus"]).to_numpy() == main_area)
    invented_main = float(delta.reindex(network.generators.index)[in_main].sum())
    invented_far = float(delta.sum()) - invented_main

    over = load.gt(THRESHOLD)
    upgrades = pd.DataFrame(log)
    moved = table["voltage"].ne(start_voltage) & table["kind"].eq("line")

    # The base and the threshold go in the filename for the reason get_flow_graph.py puts
    # WEIGHTING in its own: a run from "p_nom" and a run from "p_set" describe the same case
    # under different questions, and the second would otherwise silently overwrite the first.
    out = out_dir(CASE)
    stem = f"{DISPATCH_BASE}_{THRESHOLD * 100:.0f}"
    final = table.assign(
        s_nom_original=original, voltage_original=start_voltage,
        flow=gfg.branch_flow(network).reindex(table.index),
        loading_start=first, loading=load,
        overloaded=over, upgraded=moved)
    final.sort_values("loading", ascending=False).to_csv(out / f"branch_loading_{stem}.csv")
    upgrades.to_csv(out / f"upgrades_{stem}.csv", index=False)

    remaining = load.where(over).dropna().sort_values(ascending=False)
    blocked = final.loc[remaining.index]
    at_ceiling = int((blocked["kind"].eq("line") & blocked["voltage"].map(next_voltage).isna()).sum())
    transformers_over = int(blocked["kind"].eq("transformer").sum())

    verdict = {
        "converged": f"every branch at or below {THRESHOLD:.0%} of rating",
        "stalled": (f"{len(remaining)} branch{'es' if len(remaining) != 1 else ''} still over "
                    f"{THRESHOLD:.0%} and none of them upgradable\n"
                    f"             {at_ceiling} line{'s' if at_ceiling != 1 else ''} already at "
                    f"{LADDER[-1]:.0f} kV, {transformers_over} transformer"
                    f"{'s' if transformers_over != 1 else ''} (no voltage class to step)"),
        "capped": f"MAX_UPGRADES = {MAX_UPGRADES} reached with {len(remaining)} still over",
    }[outcome]

    listing = "\n".join(
        f"    {r.step:>3}. {r.branch:<22} {r.loading:>7.1%} of {r.s_nom_from:>7,.0f} MVA"
        f"   {r.v_from:>5.0f} -> {r.v_to:>3.0f} kV   {r.s_nom_to:>7,.0f} MVA"
        for r in upgrades.head(20).itertuples()) or "    (none)"
    tail = (f"\n    ... {len(upgrades) - 20} more, all of them in upgrades.csv"
            if len(upgrades) > 20 else "")

    blocked_listing = "\n".join(
        f"    {name:<22} {remaining[name]:>7.1%} of {blocked.at[name, 's_nom']:>7,.0f} MVA"
        f"   {blocked.at[name, 'kind']}"
        + (f" at {blocked.at[name, 'voltage']:.0f} kV" if blocked.at[name, "kind"] == "line" else "")
        for name in remaining.head(10).index)

    by_rung = (upgrades.groupby("v_from").size().sort_index() if len(upgrades)
               else pd.Series(dtype=int))
    rungs = ", ".join(f"{int(v)}->{int(next_voltage(v))} kV x{n}" for v, n in by_rung.items()) or "none"

    print(
        f"{CASE} - reinforce to {THRESHOLD:.0%}\n"
        f"  dispatch   {len(DISPATCH)} generator{'s' if len(DISPATCH) != 1 else ''} named, the "
        f"other {len(network.generators) - len(DISPATCH)} at {DEFAULT_DISPATCH:g} of "
        f"{DISPATCH_BASE}\n"
        f"             {dispatched:,.0f} MW against {shipped_load:,.0f} MW of load "
        f"({shipped:,.0f} MW was the dispatch the case ships)\n"
        f"  imbalance  {before.abs().max():,.1f} MW in the largest area before balancing\n"
        + (f"  balanced   loads in area {area} scaled by {factor:.6f} to absorb "
           f"{excess:+,.1f} MW\n"
           if BALANCE == "scale_loads" else
           "  balanced   NOT balanced (BALANCE='none'); the slack bus absorbed the lot\n")
        + f"  invented   {invented_main:,.1f} MW written into p by lpf in area {main_area}, "
          f"the island itself"
          f"{'  <- should be ~0' if BALANCE == 'scale_loads' else '  <- this is in every flow below'}\n"
          f"             {invented_far:,.1f} MW at the interconnector terminals, which is "
          f"their link schedule\n"
          f"             and never crosses an AC branch here - see the docstring\n"
        f"  start      {len(table)} branches, max loading {first.max():.1%}, "
        f"{int(first.gt(THRESHOLD).sum())} over {THRESHOLD:.0%} "
        f"({int((first.gt(THRESHOLD) & table['kind'].eq('line')).sum())} lines, "
        f"{int((first.gt(THRESHOLD) & table['kind'].eq('transformer')).sum())} transformers)\n"
        f"  upgraded   {len(upgrades)} upgrade{'s' if len(upgrades) != 1 else ''} to "
        f"{int(moved.sum())} line{'s' if int(moved.sum()) != 1 else ''}: {rungs}\n"
        f"             {original.sum():,.0f} -> {table['s_nom'].sum():,.0f} MVA of rating "
        f"({table['s_nom'].sum() / original.sum() - 1:+.1%})\n"
        f"{listing}{tail}\n"
        f"  {outcome:<10} {verdict}\n"
        + (f"{blocked_listing}\n" if len(remaining) else "")
        + f"  final      max loading {load.max():.1%}, {int(over.sum())} over {THRESHOLD:.0%}\n"
        f"\n"
        f"  NOTE       only s_nom moves, so x and r are the impedances the case shipped and\n"
        f"             every lpf above returned the same flows. The loading falls because the\n"
        f"             rating grows, never because power moved. See the docstring.\n"
        f"  wrote      {out / f'upgrades_{stem}.csv'}\n"
        f"             {out / f'branch_loading_{stem}.csv'}"
    )


if __name__ == "__main__":
    main()
