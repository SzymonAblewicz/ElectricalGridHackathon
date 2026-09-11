"""Everything the protocol run found, printed and drawn.

    python protocol_report.py

Reads what protocol.py and cascade_protocol.py wrote and produces the numbers
in the form they should be quoted in, plus the figures.  Nothing is computed
here that was not computed there - this file only reads, aggregates and draws,
so a number in a figure can always be traced to a row in cases.csv.

Writes figures/ and summary.csv into
``participant-kit new/results/N-1 contingencies/protocol/``.
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
OUT = os.path.join(KIT, "results", "N-1 contingencies", "protocol")
FIG = os.path.join(OUT, "figures")
sys.path.insert(0, KIT)

import plotstyle  # noqa: E402

C = plotstyle.CATEGORICAL
S = plotstyle.STATUS
INK, SOFT, MUTED = plotstyle.INK, plotstyle.INK_SOFT, plotstyle.INK_MUTED

ARMS = [("eir", "EirGrid", C[1]), ("ovl", "Overlap only", C[3]),
        ("solo", "Per-circuit + veto", C[6]),
        ("veto", "Overlap + veto", C[0]), ("lp", "Safe optimum (LP)", C[2])]


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("   figures/%s" % name)


def bars(ax, labels, vals, colours, fmt="%d", note=None):
    """Horizontal bars with the value written on the end of each one."""
    y = np.arange(len(labels))[::-1]
    ax.barh(y, vals, height=0.62, color=colours, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, color=INK)
    span = max(vals) if max(vals) > 0 else 1.0
    for yi, v in zip(y, vals):
        ax.text(v + span * 0.02, yi, fmt % v, va="center", ha="left",
                fontsize=9, color=INK, fontweight="bold")
    ax.set_xlim(0, span * 1.22)
    ax.grid(axis="y", visible=False)
    if note:
        ax.set_xlabel(note)


# --------------------------------------------------------------------------- #

def fig_safety(d):
    """Does the remedy break a circuit that was healthy before it ran."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 3.6))
    lab = [n for _, n, _ in ARMS]
    col = [c for _, _, c in ARMS]

    ev = [int((d[t + "_new_overloads"] > 0).sum()) for t, _, _ in ARMS]
    bars(a1, lab, ev, [S["critical"] if v else S["good"] for v in ev],
         note="events where the remedy pushed a healthy circuit over its rating")
    a1.set_title("The remedy's own damage")

    mw = [float(d[t + "_new_MW"].sum()) for t, _, _ in ARMS]
    bars(a2, lab, mw, [S["critical"] if v > 0.5 else S["good"] for v in mw],
         fmt="%.1f MW",
         note="total MW those circuits were driven over their rating by")
    a2.set_title("How badly")

    fig.suptitle("One branch trips, something overloads, a remedy is dispatched",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.005, -0.06,
             "%s (outage, hour) events, WP2033 all-island.  Islanding outages "
             "excluded.  A circuit counts as over at 100.1%% of rating."
             % f"{len(d):,}", fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "01_safety.png")


def fig_cost(d):
    """How much wind each method has to cut to do the same job."""
    both = d[(d.eir_cleared == 1) & (d.veto_cleared == 1) & (d.lp_cleared == 1)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.0))

    # Plotting the three MW distributions on top of each other hides the
    # result: on a log axis they look identical.  What matters is the
    # per-event ratio, so that is what is drawn.
    base = both[both.eir_mw > 1e-6]
    for t, name, c in ARMS:
        if t in ("eir", "ovl"):
            continue
        v = np.sort((base[t + "_mw"] / base.eir_mw).to_numpy())
        a1.plot(v, np.arange(1, len(v) + 1) / len(v) * 100, lw=2,
                color=c, label=name, zorder=3)
    a1.axvline(1.0, color=MUTED, lw=1, ls="--", zorder=2)
    a1.text(1.02, 6, "same cost as EirGrid", fontsize=8, color=MUTED)
    a1.set_xlim(0, 1.6)
    a1.set_xlabel("MW curtailed, as a fraction of what EirGrid curtails")
    a1.set_ylabel("% of events at or below")
    a1.set_title("Cost of the same outcome")
    a1.legend(frameon=False, fontsize=8.5, loc="upper left")

    # The overlap idea, isolated.  Both series here carry the veto and the same
    # greedy step; the ONLY difference is whether a node is ranked on the worst
    # circuit alone or on the sum across every circuit that is over.  With one
    # circuit over, those are the same quantity and the two curves must sit on
    # top of each other - which is the check that the comparison is clean.
    q = both[both.solo_mw > 1e-6] if "solo_mw" in both else both.iloc[:0]
    g = q.groupby("n_over")
    ks = [k for k in sorted(g.groups) if len(g.get_group(k)) >= 20][:8]
    if not ks:
        a2.axis("off")
        ks = None
    for t, name, c in ([("solo", "ranked on the worst circuit", C[6]),
                        ("veto", "ranked on all of them (overlap)", C[0])]
                       if ks else []):
        med = [100 * (1 - g.get_group(k)[t + "_mw"].median()
                      / g.get_group(k).eir_mw.median()) for k in ks]
        a2.plot(ks, med, "o-", lw=2, ms=6, color=c, zorder=3)
        a2.annotate(name, (ks[-1], med[-1]), textcoords="offset points",
                    xytext=(6, 0), fontsize=8.5, color=c, va="center")
    if ks:
        a2.set_xlabel("circuits over their rating at the same moment")
        a2.set_ylabel("% less wind curtailed than EirGrid")
        a2.set_title("What the overlap ranking adds")
        a2.set_xticks(ks)
        a2.set_xlim(min(ks) - 0.3, max(ks) + 3.2)

    fig.suptitle("Same job, different bill",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.005, -0.05,
             "Only the %s events every method clears, so the comparison is "
             "like for like." % f"{len(both):,}",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "02_cost.png")


def fig_collateral(d, col):
    """Which circuits the unchecked remedy breaks, by name."""
    if col is None or not len(col):
        return
    e = col[col.arm == "eir"]
    if not len(e):
        return
    t = (e.groupby(["broken", "kind"])
         .agg(events=("broken", "size"), worst=("loading_pct", "max"),
              mw=("over_MW", "max"))
         .reset_index().nlargest(12, "events").iloc[::-1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.8, 0.42 * len(t) + 2.4),
                                 gridspec_kw={"width_ratios": [1, 1]})
    lab = ["%s%s" % (b, "  (transformer)" if k == "Transformer" else "")
           for b, k in zip(t.broken, t.kind)]
    bars(a1, lab[::-1], list(t.events)[::-1], [S["critical"]] * len(t),
         note="events in which this circuit was pushed over")
    a1.set_title("Circuits broken by the unchecked remedy")
    bars(a2, lab[::-1], list(t.worst)[::-1], [S["serious"]] * len(t),
         fmt="%.0f%%", note="worst loading reached, % of rating")
    a2.set_title("How far over they were driven")
    a2.axvline(100, color=MUTED, lw=1, ls="--", zorder=2)
    fig.suptitle("The collateral has names",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.005, -0.04,
             "Every one of these was inside its rating before the remedy ran.  "
             "The veto and the LP break none of them.",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "03_collateral.png")


def fig_cascade(c):
    """Where the chain stops."""
    if c is None or not len(c):
        return
    arms = [("nothing", "No remedy", S["critical"]), ("eir", "EirGrid", C[1]),
            ("veto", "Overlap + veto", C[0]), ("lp", "Safe optimum (LP)", C[2])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 3.8))
    lab = [n for _, n, _ in arms]
    lost = [int(c[a + "_lost"].sum()) for a, _, _ in arms]
    bars(a1, lab, lost, [col for _, _, col in arms],
         note="total circuits lost across all %d starting trips" % len(c))
    a1.set_title("Circuits lost to the chain")

    casc = [int((c[a + "_lost"] > 0).sum()) for a, _, _ in arms]
    bars(a2, lab, casc, [col for _, _, col in arms],
         note="starting trips that took at least one further circuit with them")
    a2.set_title("Trips that spread")

    fig.suptitle("Second-order: does the chain stop",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.005, -0.06,
             "A circuit over its rating at the end of a round is taken to trip.  "
             "No thermal clock: this is the pessimistic end of the range.",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "04_cascade.png")


def fig_reach(d):
    """What the protocol cannot do, and why it is not the method's fault."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 3.8))

    n = len(d)
    cats = ["a safe remedy exists\n(LP feasible)",
            "no safe remedy exists\n(proved infeasible)"]
    vals = [int((d.lp_feasible == 1).sum()), int((d.lp_feasible == 0).sum())]
    bars(a1, cats, vals, [S["good"], S["critical"]],
         note="events, of %s" % f"{n:,}")
    a1.set_title("Can wind curtailment fix it at all?")

    q = d[d.lp_feasible == 1]
    miss = [100 * (1 - q[t + "_cleared"].mean()) for t, _, _ in ARMS]
    bars(a2, [nm for _, nm, _ in ARMS], miss,
         [c for _, _, c in ARMS], fmt="%.1f%%",
         note="% of solvable events the method failed to clear")
    a2.set_title("Headroom each method leaves on the table")

    fig.suptitle("Failure to clear is an ordering problem, not a physics one",
                 x=0.005, ha="left", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.005, -0.06,
             "The LP is the same safety rule stated exactly.  Where it succeeds "
             "and a greedy rule fails, the greedy rule gave up, not the grid.",
             fontsize=8, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "05_reach.png")


# --------------------------------------------------------------------------- #

def line(k, v):
    print("  %-56s %s" % (k, v))


def main():
    plotstyle.use()
    d = pd.read_csv(os.path.join(OUT, "cases.csv"))
    col = None
    p = os.path.join(OUT, "collateral.csv")
    if os.path.exists(p):
        col = pd.read_csv(p)
    c = None
    p = os.path.join(OUT, "cascade.csv")
    if os.path.exists(p):
        c = pd.read_csv(p)
    isl = None
    p = os.path.join(OUT, "cases_islanding.csv")
    if os.path.exists(p):
        isl = pd.read_csv(p)

    n = len(d)
    print("\n" + "=" * 78)
    print("N-1 REMEDIAL PROTOCOL - %s events, WP2033 all-island" % f"{n:,}")
    print("=" * 78)

    print("\nWHAT THE SWEEP COVERED")
    line("(outage, hour) events with at least one overload", f"{n:,}")
    line("distinct outages that cause one", "%d" % d.outage.nunique())
    line("hours of the week in which at least one occurs", "%d" % d.hour.nunique())
    line("distinct circuits ever driven over rating", "%d" % d.worst_branch.nunique())
    line("worst single overload seen", "%.1f%% of rating" % d.worst_pct.max())
    if isl is not None:
        line("islanding events, reported separately", f"{len(isl):,}")

    print("\nHEADLINE")
    hdr = "  %-18s%10s%11s%14s%11s%10s%8s" % (
        "", "cleared", "med MW", "broke a line", "circuits", "MW over", "nodes")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t, name, _ in ARMS:
        print("  %-18s%9.1f%%%11.1f%9d/%-6s%11d%10.1f%8.0f"
              % (name, 100 * d[t + "_cleared"].mean(), d[t + "_mw"].median(),
                 int((d[t + "_new_overloads"] > 0).sum()), f"{n:,}",
                 int(d[t + "_new_overloads"].sum()), d[t + "_new_MW"].sum(),
                 d[t + "_nodes"].median()))

    print("\nSAFETY")
    e = int(d.eir_new_overloads.sum())
    line("healthy circuits pushed over by the EirGrid method",
         "%d, in %d events" % (e, int((d.eir_new_overloads > 0).sum())))
    line("  worst loading it drove one to",
         "%.1f%% of rating" % d.eir_worst_new_pct.max())
    line("healthy circuits pushed over by net-relief ordering alone",
         "%d, in %d events" % (int(d.ovl_new_overloads.sum()),
                               int((d.ovl_new_overloads > 0).sum())))
    line("healthy circuits pushed over with the veto",
         "%d" % int(d.veto_new_overloads.sum()))
    line("healthy circuits pushed over by the LP",
         "%d" % int(d.lp_new_overloads.sum()))
    line("total MW the collateral circuits were driven over rating by",
         "EirGrid %.1f, overlap-only %.1f, veto %.1f, LP %.1f"
         % (d.eir_new_MW.sum(), d.ovl_new_MW.sum(),
            d.veto_new_MW.sum(), d.lp_new_MW.sum()))
    line("already-overloaded circuits made worse, EirGrid",
         "%d events" % int((d.eir_worsened > 0).sum()))
    line("already-overloaded circuits made worse, veto / LP",
         "%d / %d events" % (int((d.veto_worsened > 0).sum()),
                             int((d.lp_worsened > 0).sum())))

    print("\nCOST, on the events every method clears")
    both = d[(d.eir_cleared == 1) & (d.veto_cleared == 1) & (d.lp_cleared == 1)]
    line("events in the like-for-like set", f"{len(both):,}")
    for t, name, _ in ARMS:
        line("  %s" % name,
             "median %.2f MW, mean %.2f MW, total %.0f MW"
             % (both[t + "_mw"].median(), both[t + "_mw"].mean(),
                both[t + "_mw"].sum()))
    if both.eir_mw.sum() > 0:
        line("total wind saved by the veto method vs EirGrid",
             "%.0f MW (%.1f%%)"
             % (both.eir_mw.sum() - both.veto_mw.sum(),
                100 * (1 - both.veto_mw.sum() / both.eir_mw.sum())))
    line("veto method's distance from the proven optimum",
         "%.1f%% more MW than the LP"
         % (100 * (both.veto_mw.sum() / max(both.lp_mw.sum(), 1e-9) - 1)))
    line("EirGrid group size vs veto group size (median)",
         "%.0f nodes vs %.0f nodes"
         % (both.eir_nodes.median(), both.veto_nodes.median()))

    print("\nWHAT THE OVERLAP RANKING ADDS, ISOLATED")
    print("  Both columns carry the veto and the same greedy step.  The only")
    print("  difference is whether a node is ranked on the worst circuit alone")
    print("  or on the sum across every circuit that is over.  With one circuit")
    print("  over they are the same quantity, so that row must tie.")
    g = both.groupby("n_over")
    print("  %-11s%9s%12s%13s%12s%11s" % ("circuits", "events", "EirGrid MW",
                                          "worst-only", "overlap", "overlap"))
    print("  %-11s%9s%12s%13s%12s%11s" % ("over", "", "", "MW", "MW", "gain"))
    for k in sorted(g.groups):
        q = g.get_group(k)
        if len(q) < 20:
            continue
        a, so, b = q.eir_mw.median(), q.solo_mw.median(), q.veto_mw.median()
        print("  %-11d%9s%12.2f%13.2f%12.2f%10.1f%%"
              % (k, f"{len(q):,}", a, so, b,
                 100 * (1 - b / so) if so > 0 else 0))
    tot_s, tot_v = both.solo_mw.sum(), both.veto_mw.sum()
    line("overall MW saved by ranking on all circuits, not one",
         "%.0f MW of %.0f (%.2f%%)"
         % (tot_s - tot_v, tot_s, 100 * (1 - tot_v / max(tot_s, 1e-9))))
    m = both[both.n_over > 1]
    if len(m):
        line("  on the %s events with >1 circuit over" % f"{len(m):,}",
             "%.0f MW of %.0f (%.2f%%)"
             % (m.solo_mw.sum() - m.veto_mw.sum(), m.solo_mw.sum(),
                100 * (1 - m.veto_mw.sum() / max(m.solo_mw.sum(), 1e-9))))
    one = both[both.n_over == 1]
    if len(one):
        line("  tie check on the %s single-circuit events" % f"{len(one):,}",
             "max |worst-only - overlap| = %.2e MW"
             % float((one.solo_mw - one.veto_mw).abs().max()))

    print("\nREACH - what curtailment genuinely cannot do")
    line("events where a safe remedy provably exists",
         "%s of %s  (%.2f%%)" % (f"{int((d.lp_feasible==1).sum()):,}", f"{n:,}",
                                 100 * d.lp_feasible.mean()))
    line("events where none exists, proved by LP infeasibility",
         "%d" % int((d.lp_feasible == 0).sum()))
    q = d[d.lp_feasible == 1]
    for t, name, _ in ARMS:
        line("  %s failed to clear a solvable event" % name,
             "%d times (%.1f%%)" % (int((q[t + "_cleared"] == 0).sum()),
                                    100 * (1 - q[t + "_cleared"].mean())))

    bad = d[d.lp_feasible == 0]
    if len(bad):
        print("\n  the %d events wind provably cannot fix:" % len(bad))
        t = (bad.groupby(["outage", "worst_branch"])
             .agg(events=("hour", "size"), worst=("worst_pct", "max"),
                  over=("over_MW", "max"), resid=("lp_resid_MW", "max"),
                  wind=("wind_avail_MW", "median")).reset_index()
             .nlargest(8, "events"))
        print("    %-22s%-22s%7s%8s%10s%10s"
              % ("outage", "circuit over", "events", "worst", "left MW", "wind MW"))
        for _, r in t.iterrows():
            print("    %-22s%-22s%7d%7.0f%%%10.1f%10.0f"
                  % (r.outage[:21], r.worst_branch[:21], r.events, r.worst,
                     r.resid, r.wind))
        line("  median wind running at those hours",
             "%.0f MW" % bad.wind_avail_MW.median())
        line("  median overload the LP could not remove",
             "%.1f MW of %.1f" % (bad.lp_resid_MW.median(), bad.over_MW.median()))

    if col is not None and len(col):
        print("\nTHE CIRCUITS THE UNCHECKED REMEDY BREAKS")
        e = col[col.arm == "eir"]
        t = (e.groupby(["broken", "kind"])
             .agg(events=("broken", "size"), worst=("loading_pct", "max"))
             .reset_index().nlargest(10, "events"))
        print("  %-34s%-14s%9s%10s" % ("circuit", "kind", "events", "worst"))
        for _, r in t.iterrows():
            print("  %-34s%-14s%9d%9.1f%%"
                  % (r.broken[:33], r.kind, r.events, r.worst))

    if c is not None and len(c):
        print("\nSECOND ORDER - where the chain stops")
        hdr = "  %-18s%13s%9s%12s%12s" % ("", "circuits lost", "worst",
                                          "trips that", "MW cut")
        print(hdr)
        print("  %-18s%13s%9s%12s%12s" % ("", "in total", "chain", "spread", ""))
        print("  " + "-" * (len(hdr) - 2))
        for a, name in (("nothing", "No remedy"), ("eir", "EirGrid"),
                        ("veto", "Overlap + veto"), ("lp", "Safe optimum (LP)")):
            print("  %-18s%13d%9d%9d/%-4d%12.0f"
                  % (name, int(c[a + "_lost"].sum()), int(c[a + "_lost"].max()),
                     int((c[a + "_lost"] > 0).sum()), len(c), c[a + "_cut"].sum()))

    print("\nROBUSTNESS - does the conclusion survive the choices made?")
    for fn, what in (("cases_uniform.csv", "uniform slack, not load slack"),
                     ("cases_t002.csv", "group threshold 0.02, not 0.05"),
                     ("cases_t01.csv", "group threshold 0.10, not 0.05")):
        fp = os.path.join(OUT, fn)
        if not os.path.exists(fp):
            continue
        r = pd.read_csv(fp)
        line("  %s" % what, "%s events" % f"{len(r):,}")
        for t, name, _ in ARMS:
            if t + "_cleared" not in r.columns:
                continue
            line("      %s" % name,
                 "%.1f%% cleared, %d circuits broken in %d events"
                 % (100 * r[t + "_cleared"].mean(),
                    int(r[t + "_new_overloads"].sum()),
                    int((r[t + "_new_overloads"] > 0).sum())))

    print("\nFIGURES")
    fig_safety(d)
    fig_cost(d)
    fig_collateral(d, col)
    fig_reach(d)
    fig_cascade(c)

    rec = {"events": n, "outages": int(d.outage.nunique())}
    for t, name, _ in ARMS:
        rec["%s_cleared_pct" % t] = round(100 * d[t + "_cleared"].mean(), 2)
        rec["%s_events_broke_a_line" % t] = int((d[t + "_new_overloads"] > 0).sum())
        rec["%s_lines_broken" % t] = int(d[t + "_new_overloads"].sum())
        rec["%s_median_MW" % t] = round(float(d[t + "_mw"].median()), 3)
        rec["%s_total_MW_common" % t] = round(float(both[t + "_mw"].sum()), 1)
        rec["%s_median_nodes" % t] = float(d[t + "_nodes"].median())
        rec["%s_collateral_MW" % t] = round(float(d[t + "_new_MW"].sum()), 2)
    rec["lp_feasible_pct"] = round(100 * float(d.lp_feasible.mean()), 3)
    pd.DataFrame([rec]).to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print("\n-> %s" % os.path.relpath(OUT, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
