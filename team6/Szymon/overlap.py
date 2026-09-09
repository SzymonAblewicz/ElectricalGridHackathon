"""Overlap: which wind nodes are useful across MANY congested branches.

    python overlap.py

The question this answers.  A node with a large shift factor on one circuit
relieves that circuit and nothing else.  A node with a moderate shift factor on
five congested circuits may be worth more system-wide.  Ranking by
``max |sensitivity|`` cannot see the difference; every view here can.

**The catch, and it is the whole reason these plots score rather than count.**
Curtailing a node relieves every branch where its sensitivity is positive and
*worsens* every branch where it is negative.  A node that touches six branches
with three signs each way is broad and nearly worthless.  So breadth is counted,
but the headline number is net signed relief.

Writes into ``participant-kit new/results/sensitivity/``:

    04_breadth_vs_depth.png   the core plot: reach against strength
    05_systemwide_score.png   ranked nodes, showing relief and harm separately
    06_branch_similarity.png    which congested branches share their groups
    07_beyond_congested.png          the same question over all 980 branches
    08_node_matrix.png        nodes x branches, ordered by reach
    09_map_score.png          system-wide score on the map

Nothing is recomputed - every number comes from the results CSVs.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
RES = os.path.join(KIT, "results")
OUT = os.path.join(RES, "sensitivity")
FIGS = os.path.join(OUT, "figures")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import plotstyle  # noqa: E402
import figures as figmod  # reuse placed_only / snap_stations / draw_edges  # noqa: E402

DPI = 190
THRESHOLD = 0.05     # illustrative group cut, not a standard
STRESSED = 80.0      # per cent of rating


# --------------------------------------------------------------------------- #
# data and scoring
# --------------------------------------------------------------------------- #

def read():
    S = pd.read_csv(os.path.join(OUT, "sensitivity_wind.csv"), index_col=0)
    br = pd.read_csv(os.path.join(RES, "network", "branches.csv"),
                     dtype={"bus0": str, "bus1": str}).set_index("branch")
    bus = pd.read_csv(os.path.join(RES, "network", "buses.csv"),
                      dtype={"bus": str}).set_index("bus")
    return S, br, bus


def score(S, br, subset, weight="hours"):
    """Per-node reach and net relief over a subset of branches.

    ``weight="hours"`` weights each branch by how many hours it actually spends
    at its rating, so a circuit that binds for 118 hours counts for more than
    one that binds for 2.  A branch nobody is trying to unload should not pull
    a node up the ranking.

    Returns a frame indexed by wind node with:

        reach      branches where |sensitivity| >= THRESHOLD
        helps      of those, how many it relieves
        hurts      of those, how many it worsens
        depth      max |sensitivity| on any branch in the subset
        relief     weighted sum of the positive sensitivities
        harm       weighted sum of the negative sensitivities, as a positive number
        net        relief - harm.  The headline.
    """
    M = S.loc[subset]
    if weight == "hours":
        w = br.loc[subset, "hours_at_rating"].astype(float)
        w = w / w.sum() if w.sum() > 0 else pd.Series(1.0 / len(subset), index=subset)
    else:
        w = pd.Series(1.0 / len(subset), index=subset)

    big = M.abs() >= THRESHOLD
    pos = M.clip(lower=0)
    neg = (-M).clip(lower=0)
    out = pd.DataFrame({
        "reach": big.sum(),
        "helps": (big & (M > 0)).sum(),
        "hurts": (big & (M < 0)).sum(),
        "depth": M.abs().max(),
        "relief": pos.mul(w, axis=0).sum(),
        "harm": neg.mul(w, axis=0).sum(),
    })
    out["net"] = out["relief"] - out["harm"]
    # Same thing with every branch weighted equally.  Hours-weighting lets one
    # circuit that binds for 118 of 168 hours dominate, which is right for
    # today but hides whether breadth would pay off under different weather.
    flat = pd.Series(1.0 / len(subset), index=subset)
    out["net_flat"] = (pos.mul(flat, axis=0).sum()
                       - neg.mul(flat, axis=0).sum())
    return out


def header(fig, title, subtitle, top):
    fig.suptitle(title, fontsize=15.5, x=0.018, ha="left", y=0.985,
                 fontweight="semibold")
    fig.text(0.018, 0.952, subtitle, fontsize=9, color=plotstyle.INK_SOFT,
             ha="left", va="top", linespacing=1.55)
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


def name_of(bus, n):
    return str(bus.at[n, "station"]).title() if n in bus.index else str(n)


# --------------------------------------------------------------------------- #
# o1  breadth against depth
# --------------------------------------------------------------------------- #

def fig_breadth_depth(S, br, bus, cong):
    sc = score(S, br, cong)
    fig = plt.figure(figsize=(14.6, 8.4))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.5, 1.0), wspace=0.22,
                            left=0.055, right=0.975, top=0.80, bottom=0.09)
    header(fig, "Reach against strength",
           "Every wind node placed by how strongly the worst-affected congested "
           "branch feels it (across) and how many congested branches it reaches "
           "at all (up).\nThe top-left region is the case worth caring about: "
           "moderate on several branches rather than large on one.",
           top=0.80)

    ax = fig.add_subplot(grid[0])
    jitter = (np.random.default_rng(0).random(len(sc)) - 0.5) * 0.30
    lim = float(np.nanpercentile(sc["net"].abs(), 96)) or 1e-6
    d = ax.scatter(sc["depth"], sc["reach"] + jitter, c=sc["net"],
                   cmap=plotstyle.diverging_cmap.reversed(), vmin=-lim, vmax=lim,
                   s=18 + 150 * np.sqrt(bus.loc[sc.index, "wind_MW"]
                                        / bus.loc[sc.index, "wind_MW"].max()),
                   linewidths=0.4, edgecolors=plotstyle.SURFACE, zorder=3)
    # Label sparingly and alternate the offset, or the tied clusters overprint.
    broad = sc[sc["reach"] == sc["reach"].max()].nlargest(2, "net")
    deep = sc.nlargest(2, "depth")
    offsets = [(11, 13), (11, -17), (11, 13), (11, -17)]
    for k, n in enumerate(sorted(set(broad.index) | set(deep.index),
                                 key=lambda m: -sc.at[m, "depth"])):
        dx, dy = offsets[k % len(offsets)]
        ax.annotate(name_of(bus, n), (sc.at[n, "depth"], sc.at[n, "reach"]),
                    xytext=(dx, dy), textcoords="offset points", fontsize=8,
                    color=plotstyle.INK_SOFT,
                    arrowprops=dict(arrowstyle="-", linewidth=0.6,
                                    color=plotstyle.INK_MUTED))
    ax.axvline(THRESHOLD, color=plotstyle.INK_MUTED, linewidth=0.9,
               linestyle=(0, (4.5, 2.2)))
    ax.annotate(f"{THRESHOLD:.0%}", (THRESHOLD, -0.42), xytext=(5, 0),
                textcoords="offset points", fontsize=8.5,
                color=plotstyle.INK_MUTED)
    ax.set_xlabel("depth — max |sensitivity| on any congested branch")
    ax.set_ylabel(f"reach — congested branches above {THRESHOLD:.0%}")
    ax.set_yticks(range(int(sc["reach"].max()) + 1))
    ax.set_ylim(-0.6, sc["reach"].max() + 0.6)
    bar = fig.colorbar(d, ax=ax, orientation="horizontal", fraction=0.045,
                       pad=0.11, shrink=0.7)
    bar.set_label("net weighted relief  (blue helps overall, red harms overall)",
                  fontsize=8.6, color=plotstyle.INK_SOFT)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.2)
    panel(ax, "Every wind node",
          "Dot area is installed wind.  Vertical jitter separates ties only.")

    # Slopegraph: does counting breadth actually change the ranking?
    right = fig.add_subplot(grid[1])
    N = 15
    by_depth = sc.nlargest(N, "depth").index
    by_net = sc.nlargest(N, "net_flat").index
    shown = list(dict.fromkeys(list(by_depth) + list(by_net)))
    rank_d = sc["depth"].rank(ascending=False)
    rank_n = sc["net_flat"].rank(ascending=False)

    for n in shown:
        a, b = rank_d[n], rank_n[n]
        inL, inR = a <= N, b <= N
        if not (inL or inR):
            continue
        move = a - b
        col = (plotstyle.CATEGORICAL[0] if move > 2 else
               plotstyle.STATUS["critical"] if move < -2 else "#c9c8c3")
        ya = min(a, N + 1.6); yb = min(b, N + 1.6)
        right.plot([0, 1], [ya, yb], color=col, linewidth=1.9 if abs(move) > 2 else 1.0,
                   alpha=0.95 if abs(move) > 2 else 0.5, zorder=2 if abs(move) > 2 else 1)
        if inL:
            right.annotate(f"{name_of(bus, n)}", (0, ya), xytext=(-6, 0),
                           textcoords="offset points", ha="right", va="center",
                           fontsize=8.2, color=plotstyle.INK_SOFT)
        if inR:
            right.annotate(f"{name_of(bus, n)}  ({int(sc.at[n,'reach'])})",
                           (1, yb), xytext=(6, 0), textcoords="offset points",
                           ha="left", va="center", fontsize=8.2,
                           color=plotstyle.INK if move > 2 else plotstyle.INK_SOFT,
                           fontweight="semibold" if move > 2 else "normal")
    right.set_xlim(-0.62, 1.62)
    right.set_ylim(N + 2.2, 0.3)
    right.set_xticks([0, 1], ["ranked by\ndepth alone", "ranked by\nbreadth-aware net"],
                     fontsize=9)
    right.set_yticks([])
    right.grid(False)
    for s in right.spines.values():
        s.set_visible(False)
    panel(right, "Does breadth change the ranking",
          "Top 15 under each rule.  Blue rises when breadth is counted,\n"
          "red falls.  Bracketed number is congested branches reached.")
    save(fig, "04_breadth_vs_depth.png")
    return sc


# --------------------------------------------------------------------------- #
# o2  where each node's score comes from
# --------------------------------------------------------------------------- #

def fig_systemwide(S, br, bus, cong, sc):
    top = sc.nlargest(22, "net").index
    M = S.loc[cong, top]
    w = br.loc[cong, "hours_at_rating"].astype(float)
    w = w / w.sum()
    contrib = M.mul(w, axis=0)

    fig = plt.figure(figsize=(15.2, 8.6))
    grid = fig.add_gridspec(1, 1, left=0.075, right=0.80, top=0.80, bottom=0.10)
    header(fig, "Where each node's system-wide value comes from",
           "The 22 best nodes by net relief, split by which congested branch "
           "each contribution is on.  Bars right of zero relieve, left of zero "
           "worsen.\nA tall single-colour bar is a one-branch specialist.  A "
           "stack of several colours is the node that earns its place by "
           "breadth — the case that a max-only ranking misses entirely.",
           top=0.80)
    ax = fig.add_subplot(grid[0])

    colours = list(plotstyle.CATEGORICAL) + ["#7a6f9b", "#3f7f6f"]
    y = np.arange(len(top))
    posbase = np.zeros(len(top))
    negbase = np.zeros(len(top))
    for i, b in enumerate(cong):
        v = contrib.loc[b].to_numpy()
        p = np.clip(v, 0, None)
        n = np.clip(v, None, 0)
        ax.barh(y, p, left=posbase, height=0.66, color=colours[i % len(colours)],
                label=f"{b}  ({int(br.at[b, 'hours_at_rating'])} h)")
        ax.barh(y, n, left=negbase, height=0.66, color=colours[i % len(colours)],
                alpha=0.42)
        posbase += p
        negbase += n
    ax.plot(posbase + negbase, y, "o", color=plotstyle.INK, markersize=4.5,
            zorder=5)
    ax.axvline(0, color=plotstyle.INK, linewidth=1.0)
    ax.set_yticks(y, [f"{name_of(bus, n)}   ({int(sc.at[n, 'reach'])})"
                      for n in top], fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlabel("weighted sensitivity contribution, MW per MW   "
                  "(black dot is the net)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=8, loc="center left",
              bbox_to_anchor=(1.01, 0.5), title="congested branch",
              title_fontsize=8.5)
    save(fig, "05_systemwide_score.png")


# --------------------------------------------------------------------------- #
# o3  which branches share their groups
# --------------------------------------------------------------------------- #

def fig_similarity(S, br, bus, cong):
    groups = {b: set(S.columns[S.loc[b].abs() >= THRESHOLD]) for b in cong}
    keep = [b for b in cong if groups[b]]
    n = len(keep)
    J = np.zeros((n, n))
    A = np.zeros((n, n), dtype=int)
    for i, a in enumerate(keep):
        for j, b in enumerate(keep):
            inter = groups[a] & groups[b]
            union = groups[a] | groups[b]
            J[i, j] = len(inter) / len(union) if union else 0.0
            A[i, j] = len(inter)

    corr = S.loc[keep].T.corr().to_numpy()

    fig = plt.figure(figsize=(15.0, 7.4))
    grid = fig.add_gridspec(1, 2, wspace=0.30, left=0.135, right=0.965,
                            top=0.775, bottom=0.10)
    header(fig, "Which congested branches share the same wind",
           "Left: how much two branches' constraint groups overlap (Jaccard, "
           "with the shared node count printed).  Right: correlation of the two "
           "branches' full sensitivity rows.\nTwo branches that overlap heavily "
           "can be relieved by the same instruction.  Two that anti-correlate "
           "cannot — relieving one loads the other.",
           top=0.775)

    labels = [f"{b}" for b in keep]
    ax = fig.add_subplot(grid[0])
    im = ax.imshow(J, cmap=plotstyle.sequential_cmap, vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            if i != j and A[i, j]:
                ax.text(j, i, str(A[i, j]), ha="center", va="center",
                        fontsize=7.6,
                        color="#fff" if J[i, j] > 0.55 else plotstyle.INK_SOFT)
    ax.set_xticks(range(n), labels, rotation=90, fontsize=7.4,
                  family="monospace")
    ax.set_yticks(range(n), labels, fontsize=7.4, family="monospace")
    ax.grid(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.02)
    bar.set_label("Jaccard overlap of the two groups", fontsize=8.6,
                  color=plotstyle.INK_SOFT)
    bar.outline.set_visible(False)
    panel(ax, "Group overlap", "Printed number is how many wind nodes both "
                               "groups contain.")

    ax2 = fig.add_subplot(grid[1])
    im2 = ax2.imshow(corr, cmap=plotstyle.diverging_cmap.reversed(),
                     vmin=-1, vmax=1)
    ax2.set_xticks(range(n), labels, rotation=90, fontsize=7.4,
                   family="monospace")
    ax2.set_yticks(range(n), labels, fontsize=7.4, family="monospace")
    ax2.grid(False)
    bar2 = fig.colorbar(im2, ax=ax2, fraction=0.042, pad=0.02)
    bar2.set_label("correlation across all 157 wind nodes", fontsize=8.6,
                   color=plotstyle.INK_SOFT)
    bar2.outline.set_visible(False)
    panel(ax2, "Sensitivity correlation",
          "Blue: one instruction helps both.  Red: helping one loads the other.")
    save(fig, "06_branch_similarity.png")


# --------------------------------------------------------------------------- #
# o4  the same question over all 980 branches
# --------------------------------------------------------------------------- #

def fig_all_lines(S, br, bus, cong):
    stressed = br.index[br["max_loading_pct"] >= STRESSED]
    tiers = [("congested — 10 branches at their rating", list(cong),
              plotstyle.STATUS["critical"]),
             (f"stressed — {len(stressed)} branches ever above {STRESSED:.0f}%",
              list(stressed), plotstyle.CATEGORICAL[1]),
             ("all 980 branches", list(S.index), plotstyle.CATEGORICAL[0])]

    fig = plt.figure(figsize=(15.2, 7.6))
    grid = fig.add_gridspec(1, 3, wspace=0.26, left=0.05, right=0.98,
                            top=0.775, bottom=0.115)
    header(fig, "Does breadth survive when you look past the congested ten",
           "The congested set is one draw from one synthetic week and will move "
           "with the weather, so the same question is asked of three widening "
           "sets of branches.\nA node that is broad across all 980 branches is "
           "broad because of network position, not because of this week.",
           top=0.775)

    ax = fig.add_subplot(grid[0])
    for lbl, subset, colour in tiers:
        reach = (S.loc[subset].abs() >= THRESHOLD).sum()
        share = reach / len(subset) * 100
        xs = np.sort(share)[::-1]
        ax.plot(np.arange(1, len(xs) + 1), xs, linewidth=1.8, color=colour,
                label=lbl)
    ax.set_xlabel("wind nodes, ranked")
    ax.set_ylabel(f"% of that branch set reached above {THRESHOLD:.0%}")
    ax.set_xlim(1, S.shape[1])
    ax.legend(frameon=False, fontsize=8)
    panel(ax, "Reach, as a share of each set",
          "A flat high curve means many nodes are broadly useful;\n"
          "a steep one means reach is concentrated in a few.")

    ax2 = fig.add_subplot(grid[1])
    reach_c = (S.loc[list(cong)].abs() >= THRESHOLD).sum()
    reach_a = (S.abs() >= THRESHOLD).sum()
    d = ax2.scatter(reach_a, reach_c, s=18 + 130 * np.sqrt(
        bus.loc[S.columns, "wind_MW"] / bus.loc[S.columns, "wind_MW"].max()),
        c=bus.loc[S.columns, "wind_MW"], cmap=plotstyle.sequential_cmap,
        linewidths=0.4, edgecolors=plotstyle.SURFACE)
    ax2.set_xlabel("branches reached, out of all 980")
    ax2.set_ylabel("congested branches reached, out of 10")
    bar = fig.colorbar(d, ax=ax2, fraction=0.042, pad=0.02)
    bar.set_label("installed wind, MW", fontsize=8.6, color=plotstyle.INK_SOFT)
    bar.outline.set_visible(False)
    panel(ax2, "Broad everywhere vs broad where it counts",
          "Top-left is a node that matters for today's congestion\n"
          "without being generally well connected.")

    ax3 = fig.add_subplot(grid[2])
    for lbl, subset, colour in tiers:
        M = S.loc[subset]
        big = M.abs() >= THRESHOLD
        helps = (big & (M > 0)).sum()
        hurts = (big & (M < 0)).sum()
        frac = np.where((helps + hurts) > 0, helps / (helps + hurts), np.nan)
        ax3.hist(frac[~np.isnan(frac)], bins=np.linspace(0, 1, 21),
                 histtype="step", linewidth=1.8, color=colour, label=lbl)
    ax3.set_xlabel("share of reached branches that curtailing RELIEVES")
    ax3.set_ylabel("wind nodes")
    ax3.legend(frameon=False, fontsize=8)
    panel(ax3, "Does breadth point the same way",
          "Near 1.0: curtailing helps everything it touches.  Near 0.5:\n"
          "the node helps and harms in equal measure and is not a lever.")
    save(fig, "07_beyond_congested.png")


# --------------------------------------------------------------------------- #
# o5  the matrix, ordered by reach
# --------------------------------------------------------------------------- #

def fig_matrix(S, br, bus, cong, sc):
    order = sc.sort_values(["reach", "net"], ascending=[False, False]).index
    M = S.loc[cong, order]

    fig = plt.figure(figsize=(16.2, 6.4))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.0, 2.6), hspace=0.06,
                            left=0.155, right=0.955, top=0.775, bottom=0.245)
    header(fig, "The whole matrix, ordered by reach",
           "Same data as the p1 heatmap, but wind nodes are sorted left to "
           "right by how many congested branches they reach, then by net "
           "relief.\nThe left edge is where the multi-branch levers are; the "
           "long pale tail on the right is 100-odd nodes that barely matter to "
           "today's congestion.",
           top=0.775)

    bars = fig.add_subplot(grid[0])
    r = sc.loc[order, "reach"].to_numpy()
    bars.bar(np.arange(len(order)), r, width=1.0,
             color=plotstyle.CATEGORICAL[0])
    bars.set_ylabel("branches\nreached", fontsize=8.6)
    bars.set_xlim(-0.5, len(order) - 0.5)
    bars.set_xticks([])
    bars.set_yticks(range(0, int(r.max()) + 1, 2))
    bars.grid(axis="x", visible=False)
    for s in ("top", "right"):
        bars.spines[s].set_visible(False)

    ax = fig.add_subplot(grid[1], sharex=bars)
    lim = float(np.nanpercentile(np.abs(M.to_numpy()), 99)) or 1e-6
    im = ax.imshow(M.to_numpy(), aspect="auto", interpolation="nearest",
                   cmap=plotstyle.diverging_cmap.reversed(), vmin=-lim, vmax=lim,
                   extent=(-0.5, len(order) - 0.5, len(cong) - 0.5, -0.5))
    ax.set_yticks(range(len(cong)),
                  [f"{b}  {int(br.at[b, 'hours_at_rating']):>3} h" for b in cong],
                  fontsize=8.0, family="monospace")
    step = 5
    ax.set_xticks(range(0, len(order), step),
                  [name_of(bus, n)[:15] for n in order[::step]],
                  rotation=90, fontsize=6.2)
    ax.set_xlabel("wind node, ordered by reach then net relief", labelpad=6)
    ax.grid(False)
    bar = fig.colorbar(im, ax=[bars, ax], fraction=0.016, pad=0.008)
    bar.set_label("sensitivity, MW per MW", fontsize=8.6,
                  color=plotstyle.INK_SOFT)
    bar.outline.set_visible(False)
    save(fig, "08_node_matrix.png")


# --------------------------------------------------------------------------- #
# o6  on the map
# --------------------------------------------------------------------------- #

def fig_map(S, br, bus, cong, sc):
    placed = figmod.placed_only(figmod.snap_stations(bus, br))
    wind = placed[placed["is_wind_node"]]
    common = [n for n in sc.index if n in wind.index]

    fig = plt.figure(figsize=(14.6, 8.2))
    grid = fig.add_gridspec(1, 2, wspace=0.02, left=0.01, right=0.965,
                            top=0.80, bottom=0.06)
    header(fig, "Reach and net relief, on the ground",
           "Left: how many of the ten congested branches each wind node "
           "reaches.  Right: its net weighted relief, which is what breadth is "
           "worth once the signs are counted.\nA node can be dark on the left "
           "and pale on the right — broad, but pulling in both directions at "
           "once.",
           top=0.80)

    for k, (col, cmap, lab, div) in enumerate([
            ("reach", plotstyle.sequential_cmap,
             "congested branches reached", False),
            ("net", plotstyle.diverging_cmap.reversed(),
             "net weighted relief", True)]):
        ax = fig.add_subplot(grid[0, k])
        figmod.draw_edges(ax, placed, br)
        for b0, b1 in zip(br.loc[cong, "bus0"], br.loc[cong, "bus1"]):
            if b0 in placed.index and b1 in placed.index:
                ax.plot([placed.at[b0, "x"], placed.at[b1, "x"]],
                        [placed.at[b0, "y"], placed.at[b1, "y"]],
                        color=plotstyle.STATUS["critical"], linewidth=2.0,
                        solid_capstyle="round", zorder=2)
        v = sc.loc[common, col]
        kw = dict(vmin=-float(np.nanpercentile(v.abs(), 96) or 1e-6),
                  vmax=float(np.nanpercentile(v.abs(), 96) or 1e-6)) if div \
            else dict(vmin=0, vmax=float(v.max()))
        order = v.abs().sort_values().index
        d = ax.scatter(wind.loc[order, "x"], wind.loc[order, "y"],
                       c=v[order], cmap=cmap, s=40, zorder=3,
                       linewidths=0.4, edgecolors=plotstyle.SURFACE, **kw)
        figmod.map_frame(ax, placed)
        bar = fig.colorbar(d, ax=ax, orientation="horizontal", fraction=0.04,
                           pad=0.02, shrink=0.72)
        bar.set_label(lab, fontsize=8.8, color=plotstyle.INK_SOFT)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=8.2)
        panel(ax, "Reach" if not div else "Net relief",
              "Red edges are the ten congested branches.")
    save(fig, "09_map_score.png")


def main():
    plotstyle.use()
    S, br, bus = read()
    cong = list(br[br["hours_at_rating"] > 0]
                .sort_values("hours_at_rating", ascending=False).index)
    sc = fig_breadth_depth(S, br, bus, cong)
    fig_systemwide(S, br, bus, cong, sc)
    fig_similarity(S, br, bus, cong)
    fig_all_lines(S, br, bus, cong)
    fig_matrix(S, br, bus, cong, sc)
    fig_map(S, br, bus, cong, sc)

    sc.insert(0, "station", [name_of(bus, n) for n in sc.index])
    sc.insert(1, "wind_MW", bus.loc[sc.index, "wind_MW"].to_numpy())
    sc.sort_values("net", ascending=False).round(5).to_csv(
        os.path.join(OUT, "node_scores.csv"))
    print("->", os.path.relpath(os.path.join(OUT, "node_scores.csv"), HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
