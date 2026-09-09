"""Readable views of ptdf_wind.csv - plots plus an interactive HTML table.

    python explore_ptdf.py

Everything it draws is *signed* sensitivity, so the output lands under
``results/sensitivity/`` rather than beside the raw PTDF:

    sensitivity/figures/01_heatmap.png               the whole matrix at once
    sensitivity/figures/02_concentration.png         how few nodes carry the relief
    sensitivity/figures/03_group_overlap_upset.png   what the groups share
    sensitivity/viewers/ptdf_viewer.html             the table, sortable
    sensitivity/viewers/ptdf_viewer_withOverlaps.html  the table plus live scoring

No CSV is modified.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
RES = os.path.join(KIT, "results")
PTDF = os.path.join(RES, "ptdf")
NETWORK = os.path.join(RES, "network")
FIGS = os.path.join(RES, "sensitivity", "figures")
VIEWS = os.path.join(RES, "sensitivity", "viewers")
sys.path.insert(0, KIT)

import plotstyle  # noqa: E402

DPI = 190
THRESHOLD = 0.05          # illustrative constraint-group cut, not a standard


def load():
    ptdf = pd.read_csv(os.path.join(PTDF, "ptdf_wind.csv"), index_col=0)
    br = pd.read_csv(os.path.join(NETWORK, "branches.csv"),
                     dtype={"bus0": str, "bus1": str}).set_index("branch")
    bus = pd.read_csv(os.path.join(NETWORK, "buses.csv"),
                      dtype={"bus": str}).set_index("bus")
    return ptdf, br, bus


def signed(ptdf, br):
    """PTDF re-pointed along each branch's actual flow (Stage 4)."""
    return ptdf.mul(br["sign"].reindex(ptdf.index), axis=0)


def congested(br):
    return br[br["hours_at_rating"] > 0].sort_values(
        "hours_at_rating", ascending=False)


def label(bus, n):
    st = str(bus.at[n, "station"]).title() if n in bus.index else n
    return f"{st[:16]}"


def header(fig, title, subtitle, top):
    fig.suptitle(title, fontsize=15.5, x=0.018, ha="left", y=0.985,
                 fontweight="semibold")
    fig.text(0.018, 0.952, subtitle, fontsize=9, color=plotstyle.INK_SOFT,
             ha="left", va="top", linespacing=1.55)
    fig.subplots_adjust(top=top)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight", dpi=DPI, facecolor=plotstyle.SURFACE)
    plt.close(fig)
    print("->", os.path.relpath(path, HERE))


# --------------------------------------------------------------------------- #
# 1. the matrix as a heatmap
# --------------------------------------------------------------------------- #

def plot_heatmap(ptdf, br, bus):
    cong = congested(br)
    M = signed(ptdf, br).loc[cong.index]

    # Order nodes so structure is visible: by which branch feels them most,
    # then by strength within that branch.  Nothing is clustered or fitted.
    owner = M.abs().to_numpy().argmax(axis=0)
    strength = M.abs().to_numpy().max(axis=0)
    order = np.lexsort((-strength, owner))
    M = M.iloc[:, order]

    fig = plt.figure(figsize=(16.0, 7.0))
    grid = fig.add_gridspec(1, 1, left=0.145, right=0.965, top=0.80, bottom=0.20)
    header(fig, "Every congested branch against every wind node",
           f"All {M.shape[0]} x {M.shape[1]} signed shift factors at once.  Blue "
           "means curtailing there unloads the branch, red means it loads it "
           "further.\nNodes are ordered by which branch feels them most "
           "strongly, so each block is roughly one constraint group.",
           top=0.80)
    ax = fig.add_subplot(grid[0])

    limit = float(np.nanpercentile(np.abs(M.to_numpy()), 99)) or 1e-6
    im = ax.imshow(M.to_numpy(), aspect="auto", interpolation="nearest",
                   cmap=plotstyle.diverging_cmap.reversed(),
                   vmin=-limit, vmax=limit)
    ax.set_yticks(range(len(M)),
                  [f"{i}  {k[0]}  {int(h):>3} h" for i, k, h in
                   zip(M.index, cong["kind"], cong["hours_at_rating"])],
                  fontsize=8.2, family="monospace")
    step = 6
    ax.set_xticks(range(0, M.shape[1], step),
                  [label(bus, n) for n in M.columns[::step]],
                  rotation=90, fontsize=6.4)
    ax.set_xlabel("wind node", labelpad=8)
    ax.grid(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.008)
    bar.set_label("MW off the branch per MW curtailed", fontsize=9,
                  color=plotstyle.INK_SOFT)
    bar.outline.set_visible(False)
    save(fig, "01_heatmap.png")


# --------------------------------------------------------------------------- #
# 2. group overlap
# --------------------------------------------------------------------------- #

def plot_overlap(ptdf, br, bus):
    """UpSet, not Venn.

    A Venn diagram cannot honestly draw more than three sets; there are eight
    non-empty constraint groups here.  An UpSet plot shows the same thing -
    which sets a node belongs to - and stays readable at any number of sets.
    """
    cong = congested(br)
    S = signed(ptdf, br).loc[cong.index]
    groups = {b: set(S.columns[S.loc[b].abs() >= THRESHOLD]) for b in S.index}
    groups = {b: g for b, g in groups.items() if g}
    names = sorted(groups, key=lambda b: -len(groups[b]))

    members = {}
    for n in set().union(*groups.values()):
        members[n] = tuple(b in groups[b_] for b, b_ in zip(names, names)) \
            if False else tuple(n in groups[b] for b in names)
    combos = pd.Series(list(members.values())).value_counts()
    combos = combos.head(14)

    fig = plt.figure(figsize=(15.0, 8.6))
    grid = fig.add_gridspec(2, 2, width_ratios=(0.52, 3.0),
                            height_ratios=(1.0, 1.05),
                            wspace=0.40, hspace=0.05,
                            left=0.045, right=0.975, top=0.80, bottom=0.08)
    n_multi = sum(1 for v in members.values() if sum(v) > 1)
    header(fig, "How the constraint groups overlap",
           f"{len(members)} of {ptdf.shape[1]} wind nodes clear a "
           f"{THRESHOLD:.0%} threshold on at least one congested branch, and "
           f"{n_multi} of them clear it on more than one.\nA farm in several "
           "groups takes the tightest instruction, so overlap is what makes "
           "curtailment allocation hard.  UpSet, not Venn — a Venn cannot "
           "honestly draw eight sets.",
           top=0.80)

    bars = fig.add_subplot(grid[0, 1])
    bars.bar(range(len(combos)), combos.to_numpy(), width=0.62,
             color=plotstyle.CATEGORICAL[0])
    bars.bar_label(bars.containers[0], fmt="%d", padding=3, fontsize=8.5,
                   color=plotstyle.INK_SOFT)
    bars.set_ylabel("wind nodes")
    bars.set_xticks([])
    bars.set_xlim(-0.7, len(combos) - 0.3)
    bars.grid(axis="x", visible=False)
    for s in ("top", "right", "bottom"):
        bars.spines[s].set_visible(False)

    dots = fig.add_subplot(grid[1, 1], sharex=bars)
    for j, combo in enumerate(combos.index):
        rows = [i for i, on in enumerate(combo) if on]
        dots.scatter([j] * len(names), range(len(names)), s=42,
                     color="#e2e1dd", zorder=1)
        if rows:
            dots.scatter([j] * len(rows), rows, s=48,
                         color=plotstyle.INK, zorder=2)
            if len(rows) > 1:
                dots.plot([j, j], [min(rows), max(rows)],
                          color=plotstyle.INK, linewidth=1.6, zorder=2)
    dots.set_yticks(range(len(names)), names, fontsize=8.6,
                    family="monospace")
    dots.tick_params(axis="y", pad=7, length=0, labelleft=True)
    dots.set_ylim(len(names) - 0.5, -0.5)
    dots.set_xticks([])
    dots.grid(False)
    for s in dots.spines.values():
        s.set_visible(False)
    dots.set_xlabel("each column is one combination of groups", labelpad=8)

    # No sharey: the set names live on `dots`, and a shared axis would let
    # this panel's tick settings wipe them.  Same ylim keeps rows aligned.
    sizes = fig.add_subplot(grid[1, 0])
    sizes.barh(range(len(names)), [len(groups[b]) for b in names], height=0.55,
               color=plotstyle.INK_MUTED)
    sizes.bar_label(sizes.containers[0], fmt="%d", padding=3, fontsize=8.2,
                    color=plotstyle.INK_SOFT)
    sizes.invert_xaxis()
    sizes.set_xlabel("group size")
    sizes.set_ylim(len(names) - 0.5, -0.5)
    sizes.set_yticks([])
    sizes.grid(axis="y", visible=False)
    for s in ("top", "left", "right"):
        sizes.spines[s].set_visible(False)
    save(fig, "03_group_overlap_upset.png")


# --------------------------------------------------------------------------- #
# 3. concentration
# --------------------------------------------------------------------------- #

def plot_concentration(ptdf, br, bus):
    cong = congested(br)
    S = signed(ptdf, br).loc[cong.index]

    fig = plt.figure(figsize=(15.0, 6.4))
    grid = fig.add_gridspec(1, 2, wspace=0.20, left=0.055, right=0.975,
                            top=0.775, bottom=0.115)
    header(fig, "How concentrated is each branch's sensitivity",
           "Left: every congested branch's wind nodes ranked by |shift factor|.  "
           "Right: how many nodes it takes to reach a given share of that "
           "branch's total.\nA steep curve means a small, well-defined "
           "constraint group; a flat one means the branch feels the whole "
           "island weakly and curtailment is a blunt tool.",
           top=0.775)

    left = fig.add_subplot(grid[0])
    right = fig.add_subplot(grid[1])
    colours = list(plotstyle.CATEGORICAL) * 2
    for i, b in enumerate(S.index):
        v = S.loc[b].abs().sort_values(ascending=False).to_numpy()
        xs = np.arange(1, len(v) + 1)
        left.plot(xs, v, linewidth=1.5, color=colours[i], label=b)
        share = np.cumsum(v) / v.sum()
        right.plot(xs, 100 * share, linewidth=1.5, color=colours[i], label=b)

    left.axhline(THRESHOLD, color=plotstyle.INK_MUTED, linewidth=0.9,
                 linestyle=(0, (4.5, 2.2)))
    left.annotate(f"{THRESHOLD:.0%} threshold", (ptdf.shape[1], THRESHOLD),
                  xytext=(-6, 5), textcoords="offset points", ha="right",
                  fontsize=8.5, color=plotstyle.INK_MUTED)
    left.set_xlabel("wind nodes, ranked")
    left.set_ylabel("|shift factor|,  MW per MW")
    left.set_xlim(1, ptdf.shape[1])
    left.set_ylim(0, None)
    left.legend(frameon=False, fontsize=7.6, ncol=2)

    for pct in (50, 80):
        right.axhline(pct, color=plotstyle.INK_MUTED, linewidth=0.8,
                      linestyle=(0, (1.6, 2.2)))
        right.annotate(f"{pct}%", (1, pct), xytext=(4, 4),
                       textcoords="offset points", fontsize=8.5,
                       color=plotstyle.INK_MUTED)
    right.set_xlabel("wind nodes, ranked")
    right.set_ylabel("cumulative share of the branch's total |shift factor|, %")
    right.set_xlim(1, ptdf.shape[1])
    right.set_ylim(0, 101)
    save(fig, "02_concentration.png")


# --------------------------------------------------------------------------- #
# 4. the interactive table
# --------------------------------------------------------------------------- #

def build_viewer(ptdf, br, bus):
    cols = list(ptdf.columns)
    rows = list(ptdf.index)
    data = np.round(ptdf.to_numpy(), 5)

    payload = {
        "branches": rows,
        "nodes": cols,
        "values": [[None if not np.isfinite(v) else float(v) for v in r]
                   for r in data],
        "sign": [int(br.at[b, "sign"]) for b in rows],
        "kind": [str(br.at[b, "kind"])[0] for b in rows],
        "s_nom": [float(br.at[b, "s_nom"]) for b in rows],
        "hours": [int(br.at[b, "hours_at_rating"]) for b in rows],
        "maxload": [round(float(br.at[b, "max_loading_pct"]), 1) for b in rows],
        "station": [str(bus.at[n, "station"]) if n in bus.index else n
                    for n in cols],
        "windMW": [round(float(bus.at[n, "wind_MW"]), 1) if n in bus.index else 0
                   for n in cols],
        "farms": [int(bus.at[n, "wind_farms"]) if n in bus.index else 0
                  for n in cols],
        "kV": [float(bus.at[n, "v_nom_kV"]) if n in bus.index else 0
               for n in cols],
    }
    os.makedirs(VIEWS, exist_ok=True)
    blob = json.dumps(payload, separators=(",", ":"))
    for template, out in (("viewer_template.html", "ptdf_viewer.html"),
                          ("viewer_overlap_template.html",
                           "ptdf_viewer_withOverlaps.html")):
        with open(os.path.join(HERE, template), encoding="utf-8") as f:
            html = f.read()
        path = os.path.join(VIEWS, out)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html.replace("/*__DATA__*/null", blob))
        print("->", os.path.relpath(path, HERE),
              f"({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    plotstyle.use()
    ptdf, br, bus = load()
    plot_heatmap(ptdf, br, bus)
    plot_overlap(ptdf, br, bus)
    plot_concentration(ptdf, br, bus)
    build_viewer(ptdf, br, bus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
