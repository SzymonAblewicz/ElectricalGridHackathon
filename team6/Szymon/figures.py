"""Figures for the PTDF / shift-factor work.

    python figures.py

Reads ``participant-kit new/results/`` and writes PNGs to
``participant-kit new/figures/new figures/``.  Run ptdf_all.py first.

    1_network.png       what is in the grid: nodes by class, edges by kind
    2_sensitivity.png   one monitored branch: which wind nodes it feels, and
                        the ranked curve a constraint group is cut from
    3_congestion.png    the congested branches across the week, and whether
                        their flow direction is stable

Nothing here recomputes anything.  Every number comes from the CSVs.
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
RESULTS = os.path.join(KIT, "results")
FIGS = os.path.join(KIT, "figures", "new figures")
sys.path.insert(0, KIT)

import plotstyle  # noqa: E402

DPI = 190

#: Node classes, in drawing order - last drawn sits on top.
NODE_CLASSES = (
    ("junction", "#c9c8c3"),
    ("load only", plotstyle.INK_MUTED),
    ("other generation", plotstyle.CATEGORICAL[1]),
    ("wind node", plotstyle.CATEGORICAL[0]),
)

LINE_GREY = "#dcdbd7"
TX_BLUE = "#a9bfd8"


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def read():
    def r(folder, name, **kw):
        return pd.read_csv(os.path.join(RESULTS, folder, name), **kw)
    buses = r("network", "buses.csv", dtype={"bus": str}).set_index("bus")
    branches = r("network", "branches.csv",
                 dtype={"bus0": str, "bus1": str}).set_index("branch")
    ptdf = r("ptdf", "ptdf_wind.csv", index_col=0)
    loading = r("timeseries", "loading.csv", index_col=0)
    signs = r("timeseries", "signs.csv", index_col=0)
    return buses, branches, ptdf, loading, signs


def placed_only(buses):
    """Buses with a usable position.

    The three AC-isolated terminals - Scotland, and the far ends of EWIC and
    Greenlink - carry (0, 0), which is in the Gulf of Guinea.  Left in, they
    stretch the map until Ireland is a dot.
    """
    ok = buses["x"].notna() & buses["y"].notna()
    ok &= ~((buses["x"].abs() < 1e-9) & (buses["y"].abs() < 1e-9))
    return buses[ok]


def snap_stations(buses, branches):
    """Collapse each substation to one drawing position.

    Two buses with the same station name are the same place - a 380 kV busbar
    and the 110 kV busbar behind the same transformer.  Where neither has a
    real geocode they are each placed at the mean of their own neighbours,
    which can pull them kilometres apart and draw a transformer as a long line
    across open country.  Coolnabacky's two busbars end up 43 km apart that
    way; the transformer between them is then the most prominent edge in the
    midlands and means nothing.

    Only 5 of 616 station groups are spread by more than 1 km, so this changes
    almost nothing else on the map.  Drawing only - no result file is touched.
    """
    out = buses.copy()
    real = out["station"].astype(str)
    is_star = real.str.startswith("star of ")

    # 1. Buses sharing a station name go to that station's mean position.
    named = out[~is_star]
    centre = named.groupby(real[~is_star])[["x", "y"]].transform("mean")
    out.loc[centre.index, ["x", "y"]] = centre

    # 2. A star point is internal to one transformer, so put it on top of the
    #    terminals it connects.
    for bus in out.index[is_star]:
        ends = pd.concat([branches.loc[branches["bus0"] == bus, "bus1"],
                          branches.loc[branches["bus1"] == bus, "bus0"]])
        ends = [e for e in ends if e in out.index]
        if ends:
            out.loc[bus, ["x", "y"]] = out.loc[ends, ["x", "y"]].mean()
    return out


def classify(buses):
    cls = pd.Series("junction", index=buses.index)
    cls[buses["load_MW"] > 0] = "load only"
    cls[buses["other_gen_MW"] > 0] = "other generation"
    cls[buses["is_wind_node"]] = "wind node"
    return cls


def draw_edges(ax, buses, branches, colour=LINE_GREY, lw=0.6, zorder=1):
    """Draw the transmission lines, and only those.

    **Transformers are deliberately not drawn as edges.**  A transformer sits
    inside a single substation - it steps 220 kV down to 110 kV on the same
    site - so it has no length on the ground and no route to draw.  Any
    apparent length it has on this map is an artifact of its two busbars
    being given different inferred positions, and drawing that produces a
    dashed line across open country that means nothing.  All 14 of the
    long-looking transformers on earlier versions of this figure were exactly
    that.

    They are still real DC-flow edges and 4 of the 10 congested branches are
    transformers, so they are shown where they belong: as a ring on the
    substation they sit in.  See :func:`mark_transformers`.
    """
    from matplotlib.collections import LineCollection
    seg = branches[branches["kind"] == "Line"]
    pts = [[(buses.at[a, "x"], buses.at[a, "y"]),
            (buses.at[c, "x"], buses.at[c, "y"])]
           for a, c in zip(seg["bus0"], seg["bus1"])
           if a in buses.index and c in buses.index]
    ax.add_collection(LineCollection(pts, colors=colour, linewidths=lw,
                                     zorder=zorder))


def mark_transformers(ax, buses, branches, names, colour, size=190, lw=2.0):
    """Ring the substations holding the named transformers.

    A transformer has a location but not a route, so a ring at its site is the
    honest way to put it on a map.  Duplicates collapse: two windings of one
    three-winding transformer are one site, one ring.
    """
    seen = set()
    for name in names:
        ends = [branches.at[name, "bus0"], branches.at[name, "bus1"]]
        ends = [e for e in ends if e in buses.index]
        if not ends:
            continue
        x = float(buses.loc[ends, "x"].mean())
        y = float(buses.loc[ends, "y"].mean())
        key = (round(x, 4), round(y, 4))
        if key in seen:
            continue
        seen.add(key)
        ax.scatter([x], [y], s=size, facecolors="none", edgecolors=colour,
                   linewidths=lw, zorder=5)
    return len(seen)


def map_frame(ax, buses, margin=0.28):
    ax.set_xlim(buses["x"].min() - margin, buses["x"].max() + margin)
    ax.set_ylim(buses["y"].min() - margin, buses["y"].max() + margin)
    ax.set_aspect(1 / np.cos(np.radians(float(buses["y"].mean()))))
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_title(ax, title, subtitle=None):
    """Title above subtitle above the axes, with room reserved for both.

    ``set_title`` only takes one string at one size, so both are drawn as
    offset annotations and an empty title reserves the space they need.
    """
    lines = 0 if not subtitle else subtitle.count("\n") + 1
    ax.set_title(" ", fontsize=11.5, loc="left", pad=16 + 11.5 * lines)
    ax.annotate(title, (0, 1.0), xycoords="axes fraction",
                xytext=(0, 9 + 11.5 * lines), textcoords="offset points",
                va="bottom", ha="left", fontsize=11.5, fontweight="semibold",
                color=plotstyle.INK, annotation_clip=False)
    if subtitle:
        ax.annotate(subtitle, (0, 1.0), xycoords="axes fraction",
                    xytext=(0, 5), textcoords="offset points",
                    va="bottom", ha="left", fontsize=8.4,
                    color=plotstyle.INK_SOFT, linespacing=1.5,
                    annotation_clip=False)


def figure_header(fig, title, subtitle, top):
    """Figure-level title and a wrapped standfirst above everything."""
    fig.suptitle(title, fontsize=15.5, x=0.018, ha="left", y=0.985,
                 fontweight="semibold")
    fig.text(0.018, 0.955, subtitle, fontsize=9, color=plotstyle.INK_SOFT,
             ha="left", va="top", linespacing=1.55)
    fig.subplots_adjust(top=top)


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    path = os.path.join(FIGS, name)
    fig.savefig(path, bbox_inches="tight", dpi=DPI,
                facecolor=plotstyle.SURFACE)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1. What is in the grid
# --------------------------------------------------------------------------- #

def figure_network(buses, branches):
    placed = placed_only(snap_stations(buses, branches))
    cls_all, cls = classify(buses), classify(placed)

    fig = plt.figure(figsize=(15.4, 9.8))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.26, 1.0),
                            height_ratios=(1.0, 1.18),
                            wspace=0.13, hspace=0.46,
                            left=0.015, right=0.975, top=0.845, bottom=0.055)
    figure_header(
        fig,
        f"WP2033 all-island network:  {len(buses)} nodes,  {len(branches)} edges",
        "All dots are the same size, so the map reads position and class only.  "
        "The 225 transformers are real DC-flow edges but sit inside a single\n"
        "substation, so they have no route to draw and are shown as a ring on "
        "their site instead.  "
        f"{int((~placed['has_coordinates']).sum())} of {len(placed)} node positions "
        "are inferred rather than surveyed.",
        top=0.845)

    # ---- the map -----------------------------------------------------
    ax = fig.add_subplot(grid[:, 0])
    draw_edges(ax, placed, branches)
    for name, colour in NODE_CLASSES:
        sel = placed[cls == name]
        if not len(sel):
            continue
        # Equal sizes: the map reads position and class only.  Capacity is on
        # the bar chart to the right, where it can carry a real axis.
        size = 22 if name in ("wind node", "other generation") else 9
        ax.scatter(sel["x"], sel["y"], s=size, c=colour, zorder=3,
                   linewidths=0.35, edgecolors=plotstyle.SURFACE)

    cong = branches[branches["hours_at_rating"] > 0]
    cong_lines = cong[cong["kind"] == "Line"]
    cong_tx = cong[cong["kind"] == "Transformer"]
    for b0, b1 in zip(cong_lines["bus0"], cong_lines["bus1"]):
        if b0 in placed.index and b1 in placed.index:
            ax.plot([placed.at[b0, "x"], placed.at[b1, "x"]],
                    [placed.at[b0, "y"], placed.at[b1, "y"]],
                    color=plotstyle.STATUS["critical"], linewidth=2.6,
                    solid_capstyle="round", zorder=4)
    n_tx_sites = mark_transformers(ax, placed, branches, cong_tx.index,
                                   plotstyle.STATUS["critical"])
    map_frame(ax, placed)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=7,
                      markerfacecolor=c, markeredgecolor="none",
                      label=f"{n}   {int((cls_all == n).sum())}")
               for n, c in reversed(NODE_CLASSES)]
    handles += [
        Line2D([], [], color=LINE_GREY, lw=1.8,
               label=f"transmission line   {int((branches['kind'] == 'Line').sum())}"),
        Line2D([], [], color=plotstyle.STATUS["critical"], lw=2.6,
               label=f"congested line   {len(cong_lines)}"),
        Line2D([], [], marker="o", linestyle="", markersize=10,
               markerfacecolor="none", markeredgecolor=plotstyle.STATUS["critical"],
               markeredgewidth=2.0,
               label=f"congested transformer   {len(cong_tx)}"
                     f"  ({n_tx_sites} site{'s' if n_tx_sites != 1 else ''})")]
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(-0.055, 1.045), frameon=True,
                    framealpha=0.96, edgecolor="none", fontsize=8.8,
                    labelspacing=0.58, handletextpad=0.8, borderpad=0.8)
    leg.get_frame().set_facecolor(plotstyle.SURFACE)

    # ---- nodes by class and voltage ----------------------------------
    top = fig.add_subplot(grid[0, 1])
    tab = (pd.crosstab(cls_all, buses["v_nom_kV"])
           .reindex([n for n, _ in reversed(NODE_CLASSES)]).fillna(0))
    keep = [c for c in (110.0, 220.0, 275.0, 380.0) if c in tab.columns]
    rare = [c for c in tab.columns if c not in keep]
    stack = tab[keep].copy()
    if rare:
        stack["other"] = tab[rare].sum(axis=1)
    shades = ["#123a68", "#256abf", "#5f9ae8", "#9ec5f4", "#cde2fb"]
    bottom = np.zeros(len(stack))
    for i, col in enumerate(stack.columns):
        top.barh(range(len(stack)), stack[col], left=bottom, height=0.58,
                 color=shades[i % len(shades)],
                 label=f"{col:g} kV" if col != "other" else "150 / 260 / 365 kV")
        bottom += stack[col].to_numpy()
    for i, total in enumerate(stack.sum(axis=1)):
        top.annotate(f"{int(total)}", (total, i), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=9,
                     color=plotstyle.INK_SOFT)
    top.set_yticks(range(len(stack)), stack.index, fontsize=9.5)
    top.set_xlabel("nodes")
    top.set_xlim(0, stack.sum(axis=1).max() * 1.14)
    top.legend(frameon=False, fontsize=8.2, ncol=2, loc="lower right",
               columnspacing=1.4)
    top.grid(axis="y", visible=False)
    panel_title(top, "Nodes by class and voltage",
                "Junctions are mostly 220 kV and above: 3-winding transformer "
                "star points and\nbranching hubs. Wind sits almost entirely "
                "at 110 kV.")

    # ---- where the wind is -------------------------------------------
    bot = fig.add_subplot(grid[1, 1])
    w = buses[buses["is_wind_node"]].sort_values("wind_MW", ascending=False)
    n_show = 15
    show = w.head(n_show).iloc[::-1]
    bars = bot.barh(range(len(show)), show["wind_MW"], height=0.68,
                    color=plotstyle.CATEGORICAL[0])
    bot.bar_label(bars, fmt="%.0f", padding=4, fontsize=8.2,
                  color=plotstyle.INK_SOFT)
    bot.set_yticks(range(len(show)),
                   [f"{s.title()}  ({int(f)})" for s, f in
                    zip(show["station"].astype(str), show["wind_farms"])],
                   fontsize=8.6)
    bot.set_xlabel("installed wind, MW      (farms at that node in brackets)")
    bot.set_xlim(0, show["wind_MW"].max() * 1.14)
    bot.grid(axis="y", visible=False)
    rest = w["wind_MW"].iloc[n_show:].sum()
    panel_title(bot, f"Where the {len(w)} wind nodes' {w['wind_MW'].sum():,.0f} MW sits",
                f"Top {n_show} shown; {rest:,.0f} MW spread across the other "
                f"{len(w) - n_show} nodes.\nThe largest are single offshore "
                "farms; onshore nodes aggregate several.")

    return save(fig, "1_network.png")


# --------------------------------------------------------------------------- #
# 2. One monitored branch
# --------------------------------------------------------------------------- #

def figure_sensitivity(buses, branches, ptdf, monitored):
    row = ptdf.loc[monitored] * float(branches.at[monitored, "sign"])
    placed = placed_only(snap_stations(buses, branches))
    wind = placed[placed["is_wind_node"]]
    vals = row.reindex(wind.index)
    meta = branches.loc[monitored]

    fig = plt.figure(figsize=(15.4, 8.8))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.10, 1.0), wspace=0.11,
                            left=0.015, right=0.972, top=0.815, bottom=0.095)
    figure_header(
        fig,
        f"Shift factors on {monitored}",
        f"A {meta['kind'].lower()} rated {meta['s_nom']:.0f} MVA, at its rating "
        f"for {int(meta['hours_at_rating'])} of the week's 168 hours.\n"
        "Sensitivity is signed along the circuit's actual flow direction, so a "
        "positive value means curtailing there unloads it.  Grey edges are "
        "transmission lines; transformers are not drawn.",
        top=0.815)

    # ---- map ---------------------------------------------------------
    left = fig.add_subplot(grid[0, 0])
    draw_edges(left, placed, branches)
    limit = float(np.nanpercentile(vals.abs(), 98)) or 1e-6
    order = vals.abs().sort_values().index
    dots = left.scatter(
        wind.loc[order, "x"], wind.loc[order, "y"], c=vals[order],
        # Reversed ramp: positive means curtailing there RELIEVES the circuit,
        # and relief should not be the alarm colour.
        cmap=plotstyle.diverging_cmap.reversed(), vmin=-limit, vmax=limit,
        s=38, zorder=3, linewidths=0.4, edgecolors=plotstyle.SURFACE)
    b0, b1 = meta["bus0"], meta["bus1"]
    if b0 in placed.index and b1 in placed.index:
        if meta["kind"] == "Line":
            left.plot([placed.at[b0, "x"], placed.at[b1, "x"]],
                      [placed.at[b0, "y"], placed.at[b1, "y"]],
                      color=plotstyle.STATUS["critical"], linewidth=3.0,
                      solid_capstyle="round", zorder=4)
        else:
            # A transformer has a site, not a route - ring it (see draw_edges).
            mark_transformers(left, placed, branches, [monitored],
                              plotstyle.STATUS["critical"], size=260, lw=2.4)
        left.annotate(monitored,
                      (np.mean([placed.at[b0, "x"], placed.at[b1, "x"]]),
                       np.mean([placed.at[b0, "y"], placed.at[b1, "y"]])),
                      xytext=(22, 26), textcoords="offset points", fontsize=9.5,
                      color=plotstyle.INK, fontweight="semibold",
                      arrowprops=dict(arrowstyle="-", color=plotstyle.INK_SOFT,
                                      linewidth=0.9))
    map_frame(left, placed)
    bar = fig.colorbar(dots, ax=left, orientation="horizontal", fraction=0.042,
                       pad=0.02, shrink=0.72)
    bar.set_label("MW off the circuit per MW curtailed at that node",
                  color=plotstyle.INK_SOFT, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.5)
    panel_title(left, "Which wind nodes the circuit can feel",
                "All dots are the same size.  Blue relieves, red worsens, "
                "pale is invisible to this circuit.")

    # ---- ranked curve ------------------------------------------------
    right = fig.add_subplot(grid[0, 1])
    ranked = vals.reindex(vals.abs().sort_values(ascending=False).index).abs()
    xs = np.arange(1, len(ranked) + 1)
    right.fill_between(xs, ranked.to_numpy(), color=plotstyle.CATEGORICAL[0],
                       alpha=0.14, linewidth=0)
    right.plot(xs, ranked.to_numpy(), color=plotstyle.CATEGORICAL[0],
               linewidth=1.9)
    for thr, style, colour in ((0.10, (0, (1.6, 2.2)), plotstyle.INK_SOFT),
                               (0.05, (0, (4.5, 2.2)), plotstyle.INK_MUTED)):
        cnt = int((ranked >= thr).sum())
        right.axhline(thr, color=colour, linewidth=1.0, linestyle=style)
        right.annotate(f"{thr:.0%} threshold  →  {cnt} nodes in the group",
                       (len(ranked), thr), xytext=(-6, 5),
                       textcoords="offset points", fontsize=9, ha="right",
                       va="bottom", color=colour)
    right.set_xlabel("wind nodes, ranked by |sensitivity|")
    right.set_ylabel("|sensitivity|,  MW per MW")
    right.set_xlim(1, len(ranked))
    right.set_ylim(0, float(ranked.max()) * 1.06)
    panel_title(right, "The curve a constraint group is cut from",
                "The knee is what makes a threshold work at all. Where to cut "
                "it is a judgement,\nnot a standard — EirGrid's own "
                "thresholds are not published.")

    return save(fig, "2_sensitivity.png")


# --------------------------------------------------------------------------- #
# 3. The week
# --------------------------------------------------------------------------- #

def figure_congestion(branches, loading, signs):
    cong = branches[branches["hours_at_rating"] > 0].sort_values(
        "hours_at_rating", ascending=False)
    L = loading.loc[cong.index].to_numpy() * 100.0
    S = signs.loc[cong.index].to_numpy()
    hours = np.arange(L.shape[1])
    labels = [f"{i}    {k[0]}    {int(h):>3} h"
              for i, k, h in zip(cong.index, cong["kind"], cong["hours_at_rating"])]

    fig = plt.figure(figsize=(15.4, 9.2))
    grid = fig.add_gridspec(2, 1, height_ratios=(1.45, 1.0), hspace=0.40,
                            left=0.175, right=0.90, top=0.815, bottom=0.075)
    figure_header(
        fig,
        f"The {len(cong)} congested branches across the week",
        "168 hourly snapshots of the kit's synthetic winter week; a branch counts as "
        "congested if it reaches its rating in at least one hour.\n"
        f"{int((cong['kind'] == 'Transformer').sum())} of the {len(cong)} are "
        "transformers, which a lines-only model would miss entirely.",
        top=0.815)

    # ---- loading heatmap ---------------------------------------------
    top = fig.add_subplot(grid[0])
    im = top.imshow(L, aspect="auto", cmap=plotstyle.sequential_cmap,
                    vmin=0, vmax=100, interpolation="nearest",
                    extent=(-0.5, L.shape[1] - 0.5, len(cong) - 0.5, -0.5))
    at_rating = np.argwhere(L >= 99.9)
    top.scatter(at_rating[:, 1], at_rating[:, 0], s=9,
                color=plotstyle.STATUS["critical"], marker="s", linewidths=0)
    top.set_yticks(range(len(cong)), labels, fontsize=8.4, family="monospace")
    top.set_xticks(np.arange(0, L.shape[1] + 1, 24))
    bar = fig.colorbar(im, ax=top, fraction=0.020, pad=0.010)
    bar.set_label("loading, % of rating", color=plotstyle.INK_SOFT, fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=8.5)
    top.grid(False)
    panel_title(top, "Loading hour by hour",
                "Red squares mark hours at the rating.  Label is  "
                "branch  ·  L line / T transformer  ·  hours at rating.")

    # ---- sign stability ----------------------------------------------
    bot = fig.add_subplot(grid[1])
    for i, name in enumerate(cong.index):
        flip = S[i] != cong.at[name, "sign"]
        busy = L[i] > 50
        keep = ~flip
        bot.scatter(hours[keep], np.full(int(keep.sum()), i), s=6,
                    color=plotstyle.CATEGORICAL[0], alpha=0.6, linewidths=0)
        sel = flip & busy
        bot.scatter(hours[sel], np.full(int(sel.sum()), i), s=16,
                    color=plotstyle.STATUS["critical"], linewidths=0)
    bot.set_yticks(range(len(cong)), cong.index, fontsize=8.4,
                   family="monospace")
    bot.set_xlabel("hour of the week")
    bot.set_xlim(-0.5, L.shape[1] - 0.5)
    bot.set_xticks(np.arange(0, L.shape[1] + 1, 24))
    bot.set_ylim(len(cong) - 0.5, -0.5)
    bot.grid(axis="y", visible=False)
    agree = cong["sign_agreement_when_busy"].astype(float)
    panel_title(bot, "Flow direction stability",
                "Blue: same sign as the branch's own peak hour.  Red: flipped "
                "while above 50% loaded.\n"
                f"Agreement when busy is {agree.min():.3f} at worst and "
                f"{agree.mean():.3f} on average — one snapshot's sign is "
                "a safe basis.")

    return save(fig, "3_congestion.png")


def main():
    plotstyle.use()
    buses, branches, ptdf, loading, signs = read()
    monitored = branches["hours_at_rating"].idxmax()
    for path in (figure_network(buses, branches),
                 figure_sensitivity(buses, branches, ptdf, monitored),
                 figure_congestion(branches, loading, signs)):
        print("->", os.path.relpath(path, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
