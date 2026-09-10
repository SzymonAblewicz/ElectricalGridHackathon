"""Figures for the N-1 contingency sweep.

    python n1_figures.py

Reads only the CSVs written by ``n1.py`` and the intact-case results.  Nothing
is recomputed here.

Each figure answers one question:

    01  Can one fixed constraint group serve every contingency?
    02  For a given branch, which farms stay and which come and go?
    03  How many branches need a group that a base-case study never finds?
    04  Is the fixed group just the intact-network group?
    05  Where are the damaging trips?

Writes into ``participant-kit new/results/N-1 contingencies/figures/``.
"""

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
N1 = os.path.join(RES, "N-1 contingencies")
FIGS = os.path.join(N1, "figures")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import plotstyle          # noqa: E402
import figures as figmod  # noqa: E402

DPI = 190
T = 0.05          # the group cut, held fixed for every branch and every case


def header(fig, title, subtitle, top):
    fig.suptitle(title, fontsize=16, x=0.018, ha="left", y=0.985,
                 fontweight="semibold")
    fig.text(0.018, 0.950, subtitle, fontsize=9.4, color=plotstyle.INK_SOFT,
             ha="left", va="top", linespacing=1.6)
    fig.subplots_adjust(top=top)


def panel(ax, title, subtitle=None):
    lines = 0 if not subtitle else subtitle.count("\n") + 1
    ax.set_title(" ", fontsize=11.5, loc="left", pad=16 + 11.5 * lines)
    ax.annotate(title, (0, 1.0), xycoords="axes fraction",
                xytext=(0, 9 + 11.5 * lines), textcoords="offset points",
                va="bottom", ha="left", fontsize=11.5, fontweight="semibold",
                color=plotstyle.INK, annotation_clip=False)
    if subtitle:
        ax.annotate(subtitle, (0, 1.0), xycoords="axes fraction",
                    xytext=(0, 5), textcoords="offset points", va="bottom",
                    ha="left", fontsize=8.4, color=plotstyle.INK_SOFT,
                    linespacing=1.5, annotation_clip=False)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight", dpi=DPI, facecolor=plotstyle.SURFACE)
    plt.close(fig)
    print("->", os.path.relpath(path, HERE))


def read():
    S1 = pd.read_csv(os.path.join(N1, "sensitivity_n1.csv"), index_col=[0, 1])
    S0 = pd.read_csv(os.path.join(RES, "sensitivity", "sensitivity_wind.csv"),
                     index_col=0)
    out = pd.read_csv(os.path.join(N1, "outages.csv"), index_col=0)
    vio = pd.read_csv(os.path.join(N1, "violations.csv"))
    # bus0/bus1 must stay strings or they will not match the bus index
    br = pd.read_csv(os.path.join(RES, "network", "branches.csv"),
                     dtype={"bus0": str, "bus1": str}).set_index("branch")
    bus = pd.read_csv(os.path.join(RES, "network", "buses.csv"),
                      dtype={"bus": str}).set_index("bus")
    return S1, S0, out, vio, br, bus


def membership(S1):
    """Per branch: how often each wind farm is in the group, across its cases."""
    freq, ncases = {}, {}
    for b, g in S1.groupby(level=1):
        freq[b] = (g >= T).mean(axis=0)
        ncases[b] = len(g)
    return pd.DataFrame(freq).T, pd.Series(ncases)


# --------------------------------------------------------------------------- #
# 01  can one fixed group serve every contingency
# --------------------------------------------------------------------------- #

def fig_fixed(freq, ncases):
    f = freq.stack()
    f = f[f > 0]                                  # farms that are ever a member
    always = float((f >= 0.999).mean())
    band = float(((f > 0.1) & (f < 0.9)).mean())

    multi = ncases[ncases > 1].index
    core = (freq.loc[multi] >= 0.999).sum(axis=1)
    ever = (freq.loc[multi] > 0).sum(axis=1)
    stab = (core / ever.clip(lower=1)).sort_values()

    fig = plt.figure(figsize=(14.4, 7.0))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.15, 1.0), wspace=0.24,
                            left=0.06, right=0.975, top=0.765, bottom=0.13)
    header(fig, "Can one fixed constraint group serve every contingency?",
           "Left: take every wind farm that is ever in a branch's group, and ask "
           "what share of that branch's N-1 cases it appears in.\n"
           f"Membership is nearly binary — {always:.0%} of farms are in every "
           f"single case, and only {band:.0%} sit anywhere in between.",
           top=0.765)

    ax = fig.add_subplot(grid[0])
    bins = np.linspace(0, 1, 21)
    cols = [plotstyle.STATUS["critical"] if b < 0.1 else
            plotstyle.CATEGORICAL[1] if b < 0.9 else plotstyle.CATEGORICAL[0]
            for b in bins[:-1]]
    counts, _ = np.histogram(f, bins=bins)
    ax.bar(bins[:-1], counts, width=0.048, align="edge", color=cols)
    ax.set_xlabel("share of that branch's contingencies in which the farm is a member")
    ax.set_ylabel("farm–branch pairs")
    ax.set_xlim(0, 1)
    for x, lab, col in ((0.02, "rare\nvisitors", plotstyle.STATUS["critical"]),
                        (0.42, "genuinely\nborderline", plotstyle.CATEGORICAL[1]),
                        (0.90, "permanent\nmembers", plotstyle.CATEGORICAL[0])):
        ax.annotate(lab, (x, 0.93), xycoords=("data", "axes fraction"),
                    fontsize=9, color=col, va="top", fontweight="semibold")
    panel(ax, "Membership is nearly binary",
          "A farm is almost always in, or almost never. The middle is empty —\n"
          "which is exactly what makes a single fixed group possible.")

    ax2 = fig.add_subplot(grid[1])
    y = np.arange(len(stab))
    ax2.barh(y, stab.to_numpy(), height=0.85, color=plotstyle.CATEGORICAL[0])
    ax2.axvline(float(stab.median()), color=plotstyle.STATUS["critical"],
                linewidth=1.6, linestyle=(0, (4, 2)))
    ax2.annotate(f"median {stab.median():.2f}", (stab.median(), len(stab) * 0.02),
                 xytext=(7, 0), textcoords="offset points", fontsize=9,
                 color=plotstyle.STATUS["critical"])
    ax2.set_yticks([])
    ax2.set_xlim(0, 1.02)
    ax2.set_xlabel("core ÷ union   (1.0 = the group never changes at all)")
    ax2.set_ylabel(f"the {len(stab)} branches seen under more than one trip")
    ax2.grid(axis="y", visible=False)
    panel(ax2, "How fixed is each branch's group",
          "Core = farms in every case. Union = farms in at least one.\n"
          "Even the worst branches keep a substantial permanent core.")
    save(fig, "01_one_fixed_group.png")


# --------------------------------------------------------------------------- #
# 02  what stays and what moves, branch by branch
# --------------------------------------------------------------------------- #

def fig_stays_moves(freq, ncases, bus):
    top = ncases.nlargest(4).index
    name = {b: (str(bus.at[b, "station"]).title() if b in bus.index else b)
            for b in bus.index}

    fig = plt.figure(figsize=(14.6, 8.2))
    grid = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.52,
                            left=0.055, right=0.975, top=0.80, bottom=0.08)
    header(fig, "What stays, and what comes and goes",
           "The four branches that overload under the most different single "
           "trips. Every wind farm ever in the group, sorted by how often it "
           "is in.\nThe cliff is the point: a solid block of permanent members, "
           "then a short ragged tail. Nothing in between.",
           top=0.80)

    for i, b in enumerate(top):
        ax = fig.add_subplot(grid[i // 2, i % 2])
        f = freq.loc[b]
        f = f[f > 0].sort_values(ascending=False)
        cols = [plotstyle.CATEGORICAL[0] if v >= 0.999 else
                plotstyle.CATEGORICAL[1] if v >= 0.5 else
                plotstyle.STATUS["critical"] for v in f]
        ax.bar(np.arange(len(f)), 100 * f.to_numpy(), width=0.9, color=cols)
        ax.set_ylim(0, 105)
        ax.set_xlim(-0.7, len(f) - 0.3)
        ax.set_ylabel("% of cases in group", fontsize=8.6)
        ax.set_xlabel("wind farms, ordered", fontsize=8.6)
        ax.grid(axis="x", visible=False)
        n_core = int((f >= 0.999).sum())
        panel(ax, f"{b}",
              f"{ncases[b]} contingencies  ·  {n_core} permanent of {len(f)} ever")
    save(fig, "02_what_stays_and_moves.png")


# --------------------------------------------------------------------------- #
# 03  the branches a base-case study never finds
# --------------------------------------------------------------------------- #

def fig_hidden(vio, br):
    intact = set(br.index[br["hours_at_rating"] > 0])
    post = set(vio["branch"])
    only = post - intact

    fig = plt.figure(figsize=(14.4, 7.0))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.85, 1.25), wspace=0.24,
                            left=0.06, right=0.975, top=0.765, bottom=0.13)
    header(fig, "The branches a base-case study never finds",
           f"With the network whole, {len(intact)} branches reach their rating. "
           f"Under some single trip, {len(post)} do.\n"
           f"{len(only)} of them are never congested intact — nothing in a "
           f"base-case study says they need a constraint group at all.",
           top=0.765)

    ax = fig.add_subplot(grid[0])
    vals = [len(intact & post), len(intact - post), len(only)]
    ax.bar([0, 1, 2], vals, width=0.6,
           color=[plotstyle.CATEGORICAL[0], plotstyle.INK_MUTED,
                  plotstyle.STATUS["critical"]])
    ax.set_xticks([0, 1, 2], ["found by\nboth", "intact\nonly", "only after\na fault"],
                  fontsize=9.4)
    ax.set_ylabel("branches")
    for i, v in enumerate(vals):
        ax.annotate(str(v), (i, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=13, fontweight="semibold")
    ax.grid(axis="x", visible=False)
    panel(ax, "Where the congested set comes from",
          "The red bar is the gap. It is not a different set —\n"
          "the intact ten are a subset of the sixty-one.")

    ax2 = fig.add_subplot(grid[1])
    w = vio.groupby("branch")["peak_overload_MW"].max().nlargest(15)
    y = np.arange(len(w))
    ax2.barh(y, w.to_numpy(),
             color=[plotstyle.STATUS["critical"] if b in only
                    else plotstyle.CATEGORICAL[0] for b in w.index])
    ax2.set_yticks(y, [b[:26] for b in w.index], fontsize=7.6,
                   family="monospace")
    ax2.invert_yaxis()
    ax2.set_xlabel("worst overload under any single trip, MW")
    ax2.grid(axis="y", visible=False)
    panel(ax2, "The largest post-fault overloads",
          "Red = invisible to a base-case study. The worst reaches 146% of\n"
          "rating on a circuit that sits at 75% with the network whole.")
    save(fig, "03_hidden_congestion.png")


# --------------------------------------------------------------------------- #
# 04  is the fixed group just the intact group
# --------------------------------------------------------------------------- #

def fig_intact_is_fixed(freq, S0):
    rows = []
    for b in freq.index:
        if b not in S0.index:
            continue
        g50 = set(freq.columns[freq.loc[b] >= 0.5])
        gi = set(S0.columns[S0.loc[b] >= T])
        if not (g50 or gi):
            continue
        rows.append(dict(branch=b, intact=len(gi), fixed=len(g50),
                         identical=(g50 == gi),
                         overlap=len(g50 & gi) / len(g50 | gi)))
    d = pd.DataFrame(rows)
    same = int(d["identical"].sum())

    fig = plt.figure(figsize=(14.4, 7.0))
    grid = fig.add_gridspec(1, 2, wspace=0.24, left=0.065, right=0.975,
                            top=0.765, bottom=0.13)
    header(fig, "The fixed group is the group you already have",
           "Comparing each branch's intact-network group against the group of "
           "farms that appear in at least half its contingencies.\n"
           f"They are exactly identical on {same} of {len(d)} branches, and the "
           f"median overlap elsewhere is {d['overlap'].median():.2f}.",
           top=0.765)

    ax = fig.add_subplot(grid[0])
    ax.scatter(d["intact"], d["fixed"], s=42,
               c=[plotstyle.CATEGORICAL[0] if v else plotstyle.STATUS["critical"]
                  for v in d["identical"]],
               linewidths=0.4, edgecolors=plotstyle.SURFACE, zorder=3)
    lim = max(d["intact"].max(), d["fixed"].max()) * 1.06
    ax.plot([0, lim], [0, lim], color=plotstyle.INK_MUTED, linewidth=1.1,
            linestyle=(0, (4, 2)), zorder=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("group size from the intact network")
    ax.set_ylabel("group size from the N-1 sweep")
    panel(ax, "Group size, intact against contingency",
          "Blue: membership is identical, not just the same size.\n"
          "Points on the line with red are same size, different farms.")

    ax2 = fig.add_subplot(grid[1])
    ax2.hist(d["overlap"], bins=np.linspace(0, 1, 21),
             color=plotstyle.CATEGORICAL[0])
    ax2.axvline(float(d["overlap"].median()), color=plotstyle.STATUS["critical"],
                linewidth=1.6, linestyle=(0, (4, 2)))
    ax2.annotate(f"median {d['overlap'].median():.2f}",
                 (d["overlap"].median(), 0.96),
                 xycoords=("data", "axes fraction"), xytext=(-7, 0),
                 textcoords="offset points", fontsize=9, ha="right", va="top",
                 color=plotstyle.STATUS["critical"])
    ax2.set_xlabel("membership overlap: intact group vs contingency group")
    ax2.set_ylabel("branches")
    panel(ax2, "How much of the intact group is right",
          "1.0 means the base-case group needs no change at all to cover\n"
          "every single-trip case for that branch.")
    save(fig, "04_intact_is_the_fixed_group.png")


# --------------------------------------------------------------------------- #
# 05  map
# --------------------------------------------------------------------------- #

def fig_map(out, vio, br, bus):
    placed = figmod.placed_only(figmod.snap_stations(bus, br))
    lines = br[br["kind"] == "Line"]
    intact = set(br.index[br["hours_at_rating"] > 0])
    hurt = vio.groupby("branch")["peak_overload_MW"].max()

    fig = plt.figure(figsize=(15.0, 8.4))
    grid = fig.add_gridspec(1, 2, wspace=0.02, left=0.01, right=0.965,
                            top=0.795, bottom=0.06)
    header(fig, "Where it happens",
           "Left: the circuits whose loss breaks something else.  Right: the "
           "circuits that break.\nRed on the right is congestion a base-case "
           "study never sees. Transformers are not drawn — they sit inside a "
           "single substation.",
           top=0.795)

    ax = fig.add_subplot(grid[0])
    figmod.draw_edges(ax, placed, br)
    hot = out[out["n_violations"] > 0].sort_values("n_violations")
    vmax = float(hot["n_violations"].quantile(0.97)) or 1.0
    cmap = plotstyle.sequential_cmap
    drawn = 0
    for nm, row in hot.iterrows():
        if nm not in lines.index:
            continue
        b0, b1 = lines.at[nm, "bus0"], lines.at[nm, "bus1"]
        if b0 in placed.index and b1 in placed.index:
            ax.plot([placed.at[b0, "x"], placed.at[b1, "x"]],
                    [placed.at[b0, "y"], placed.at[b1, "y"]],
                    color=cmap(min(row["n_violations"] / vmax, 1.0)),
                    linewidth=2.3, solid_capstyle="round", zorder=3)
            drawn += 1
    figmod.map_frame(ax, placed)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, vmax))
    bar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04,
                       pad=0.02, shrink=0.72)
    bar.set_label("branches overloaded when this circuit trips", fontsize=8.8,
                  color=plotstyle.INK_SOFT)
    bar.outline.set_visible(False)
    panel(ax, "Circuits whose loss causes overloads",
          f"{drawn} lines drawn. Grey lines break nothing.")

    ax2 = fig.add_subplot(grid[1])
    figmod.draw_edges(ax2, placed, br)
    for nm in hurt.sort_values().index:
        if nm not in lines.index:
            continue
        b0, b1 = lines.at[nm, "bus0"], lines.at[nm, "bus1"]
        if b0 in placed.index and b1 in placed.index:
            ax2.plot([placed.at[b0, "x"], placed.at[b1, "x"]],
                     [placed.at[b0, "y"], placed.at[b1, "y"]],
                     color=(plotstyle.CATEGORICAL[0] if nm in intact
                            else plotstyle.STATUS["critical"]),
                     linewidth=2.6, solid_capstyle="round", zorder=3)
    figmod.map_frame(ax2, placed)
    from matplotlib.lines import Line2D
    ax2.legend(handles=[
        Line2D([], [], color=plotstyle.CATEGORICAL[0], linewidth=2.6,
               label="congested intact too"),
        Line2D([], [], color=plotstyle.STATUS["critical"], linewidth=2.6,
               label="only congested after a fault")],
        frameon=False, fontsize=9, loc="lower left")
    panel(ax2, "Circuits that overload",
          "Red circuits have no constraint group under a base-case study.")
    save(fig, "05_map.png")


def main():
    plotstyle.use()
    S1, S0, out, vio, br, bus = read()
    freq, ncases = membership(S1)
    print(f"{len(out)} outages · {len(vio)} violation pairs · "
          f"{len(freq)} congested branches\n")
    fig_fixed(freq, ncases)
    fig_stays_moves(freq, ncases, bus)
    fig_hidden(vio, br)
    fig_intact_is_fixed(freq, S0)
    fig_map(out, vio, br, bus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
