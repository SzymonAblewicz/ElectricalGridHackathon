"""What the corrected comparison says.

    python merit2_report.py                          the main run
    python merit2_report.py --file cases_WP2033_all-island_uniform.csv
    python merit2_report.py --all                     pool every file present

THE RULE THIS SCRIPT ENFORCES
-----------------------------
MW is comparable between two arms only on events where BOTH cleared the
overload.  An arm that gives up early looks cheap.  So every headline is on the
paired subset and the clear-rate is printed beside it.

Paired Wilcoxon on the per-event differences; bootstrap interval on the ratio
of totals, resampled by event, because a handful of large events dominate the
sum - the median event and the annual tonnage are different questions and the
answer differs between them.

WHAT EACH COMPARISON IS FOR
---------------------------
A  node-at-a-time, ordering held fixed.  Pure effect of the group definition.
B  the veto.
C  the two defects in the original greedy, measured.
D  distance from the exact LP optimum - the ceiling on any group rule.
E  groups dispatched as UNITS.  This is the comparison the method was proposed
   for: a real constraint group cannot re-solve after every node.
F  stale effectiveness factors - a group drawn under one system condition and
   applied under another.
G  by number of circuits congested at once.  Overlap cannot do anything when
   only one circuit is over, so the all-events average understates it and the
   multi-circuit rows are the honest place to look.
H  safety.
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "participant-kit new", "results",
                   "N-1 contingencies", "merit2")

NAMES = [("eir", "EirGrid pro-rata (reference)"),
         ("blind", "greedy, worst-circuit rank  [CONTROL]"),
         ("sum", "greedy, net = sum of relief"),
         ("need", "greedy, net = need-weighted"),
         ("mmx", "greedy, net = bottleneck"),
         ("bvet", "greedy, worst-circuit + veto"),
         ("nvet", "greedy, need-weighted + veto  [METHOD]"),
         ("svet", "greedy, sum + veto"),
         ("oldstep", "greedy, ORIGINAL step (the bug)"),
         ("stale", "group from a different condition"),
         ("roster", "stale roster, live factors"),
         ("gper", "GROUP per-circuit"),
         ("gnet", "GROUP net across circuits"),
         ("gperv", "GROUP per-circuit + veto"),
         ("gnetv", "GROUP net + veto      [METHOD]"),
         ("lp", "LP optimum (floor)")]

ROI_CONSTRAINT_PCT = 6.6        # EirGrid/SONI annual report 2025, RoI
ALL_ISLAND_WIND_GWH = 13288.0   # 2024
ALL_ISLAND_DD_GWH = 2181.0      # 2024


def boot(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    m = len(a)
    if m < 2:
        return (np.nan, np.nan)
    idx = rng.integers(0, m, size=(n, m))
    r = 100.0 * (1.0 - b[idx].sum(1) / np.maximum(a[idx].sum(1), 1e-9))
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def pair(d, x, y, sub=None):
    m = (d[x + "_cleared"] == 1) & (d[y + "_cleared"] == 1)
    if sub is not None:
        m = m & sub
    s = d[m]
    if len(s) < 2:
        return None
    a = s[x + "_mw"].to_numpy(float)
    b = s[y + "_mw"].to_numpy(float)
    diff = a - b
    nz = a > 1e-6
    pct = 100.0 * diff[nz] / a[nz]
    try:
        p = (wilcoxon(a, b, zero_method="zsplit").pvalue
             if (np.abs(diff) > 1e-9).any() else 1.0)
    except ValueError:
        p = 1.0
    lo, hi = boot(a, b)
    return dict(n=len(s), ta=a.sum(), tb=b.sum(),
                sav=100.0 * (a.sum() - b.sum()) / max(a.sum(), 1e-9),
                lo=lo, hi=hi, med=float(np.median(pct)) if len(pct) else 0.0,
                win=int((diff > 1e-6).sum()), loss=int((diff < -1e-6).sum()), p=p)


def row(label, r):
    if r is None:
        print("  %-46s      -- too few paired events --" % label)
        return
    print("  %-46s %6d %9.0f %9.0f %8.2f%%  [%6.2f,%6.2f] %7.1f%% %6d/%-6d %8.1e"
          % (label, r["n"], r["ta"], r["tb"], r["sav"], r["lo"], r["hi"],
             r["med"], r["win"], r["loss"], r["p"]))


def head():
    print("  %-46s %6s %9s %9s %9s %16s %8s %13s %9s"
          % ("control -> treatment", "n", "ctrl MW", "trt MW", "saving",
             "95% CI", "median", "win/loss", "p"))
    print("  " + "-" * 142)


def main(argv):
    if "--file" in argv:
        files = [os.path.join(OUT, argv[argv.index("--file") + 1])]
    elif "--all" in argv:
        files = sorted(f for f in glob.glob(os.path.join(OUT, "cases_*.csv")))
    else:
        files = [os.path.join(OUT, "cases_WP2033_all-island.csv")]
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("no case files yet")
        return 1
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    tags = [t for t, _ in NAMES if t + "_mw" in d.columns]

    print("=" * 144)
    print("CORRECTED COMPARISON   %d events   %d outages   %d hours"
          % (len(d), d.outage.nunique(), d.hour.nunique()))
    for f in files:
        print("    %-54s %7d events" % (os.path.basename(f),
                                        len(pd.read_csv(f))))
    print("  multi-circuit events (2+ congested at once): %d of %d (%.1f%%)"
          % (int((d.n_over >= 2).sum()), len(d), 100 * (d.n_over >= 2).mean()))
    print("=" * 144)

    print("\nPER-ARM, ALL EVENTS")
    print("  %-46s %8s %10s %10s %9s %9s"
          % ("", "cleared", "total MW", "median MW", "new ovl", "events"))
    print("  " + "-" * 98)
    for t, nm in NAMES:
        if t not in tags:
            continue
        print("  %-46s %7.1f%% %10.0f %10.2f %9d %9d"
              % (nm, 100 * d[t + "_cleared"].mean(), d[t + "_mw"].sum(),
                 d[t + "_mw"].median(), int(d[t + "_new_overloads"].sum()),
                 int((d[t + "_new_overloads"] > 0).sum())))

    print("\n\nA. NODE-AT-A-TIME, ORDERING HELD FIXED - pure group-definition effect")
    head()
    for y, lab in (("sum", "plain sum"), ("need", "need-weighted"),
                   ("mmx", "bottleneck")):
        if y in tags:
            row("blind -> %-8s  %s" % (y, lab), pair(d, "blind", y))

    print("\n\nB. THE VETO")
    head()
    for x, y, lab in (("blind", "bvet", "veto alone"),
                      ("bvet", "nvet", "overlap, both vetoed"),
                      ("blind", "nvet", "both changes"),
                      ("eir", "nvet", "vs EirGrid as published")):
        if x in tags and y in tags:
            row("%-6s -> %-8s  %s" % (x, y, lab), pair(d, x, y))

    print("\n\nC. THE TWO DEFECTS IN THE ORIGINAL GREEDY, MEASURED")
    head()
    if "oldstep" in tags:
        row("oldstep -> svet     corrected step + veto retry",
            pair(d, "oldstep", "svet"))
        row("oldstep -> nvet     corrected, need-weighted",
            pair(d, "oldstep", "nvet"))

    print("\n\nD. DISTANCE FROM THE EXACT OPTIMUM - the ceiling on any group rule")
    head()
    for t, nm in NAMES:
        if t in tags and t != "lp":
            row("%-6s -> lp        %s" % (t, nm.strip()), pair(d, t, "lp"))

    print("\n\nE. GROUPS DISPATCHED AS UNITS - the comparison the method is for")
    head()
    for x, y, lab in (("gper", "gnet", "net group vs per-circuit groups"),
                      ("gperv", "gnetv", "same, both vetoed"),
                      ("gper", "gnetv", "the method vs per-circuit"),
                      ("eir", "gnetv", "the method vs EirGrid as published")):
        if x in tags and y in tags:
            row("%-6s -> %-8s  %s" % (x, y, lab), pair(d, x, y))

    print("\n\nF. A GROUP DRAWN UNDER A DIFFERENT SYSTEM CONDITION")
    head()
    for x, y, lab in (("stale", "blind", "live factors instead of study-case"),
                      ("stale", "roster", "of which: the ranking/sizing"),
                      ("roster", "blind", "of which: the roster")):
        if x in tags and y in tags:
            row("%-6s -> %-8s  %s" % (x, y, lab), pair(d, x, y))

    print("\n\nG. BY NUMBER OF CIRCUITS CONGESTED AT ONCE")
    print("   overlap is structurally incapable of helping when only one circuit")
    print("   is over, so the all-events average is diluted by those events.")
    for x, y in (("blind", "nvet"), ("gper", "gnetv")):
        if x not in tags or y not in tags:
            continue
        print("\n   %s -> %s" % (x, y))
        m = (d[x + "_cleared"] == 1) & (d[y + "_cleared"] == 1)
        s = d[m].copy()
        s["sv"] = s[x + "_mw"] - s[y + "_mw"]
        print("   %-10s %8s %11s %11s %10s %10s %12s"
              % ("circuits", "events", "ctrl MW", "method MW", "saving",
                 "win rate", "ctrl vs LP"))
        print("   " + "-" * 84)
        for k, grp in s.groupby(np.clip(s.n_over, 1, 6)):
            ta, tb = grp[x + "_mw"].sum(), grp[y + "_mw"].sum()
            gap = 100 * (ta - grp.lp_mw.sum()) / max(ta, 1e-9)
            print("   %-10s %8d %11.1f %11.1f %9.2f%% %9.1f%% %11.2f%%"
                  % ("%d%s" % (k, "+" if k == 6 else ""), len(grp), ta, tb,
                     100 * (ta - tb) / max(ta, 1e-9),
                     100 * (grp.sv > 1e-6).mean(), gap))
        r = pair(d, x, y, sub=(d.n_over >= 2))
        if r:
            print("   multi-circuit only:  n=%d  %.0f -> %.0f MW  saving %.2f%% "
                  "[%.2f, %.2f]  win/loss %d/%d  p=%.1e"
                  % (r["n"], r["ta"], r["tb"], r["sav"], r["lo"], r["hi"],
                     r["win"], r["loss"], r["p"]))

    print("\n\nH. SAFETY - remedies that pushed some other branch over")
    print("  %-46s %10s %12s %12s %12s"
          % ("", "events", "circuits", "MW over", "cleared"))
    print("  " + "-" * 96)
    for t, nm in NAMES:
        if t not in tags:
            continue
        print("  %-46s %10d %12d %12.1f %11.1f%%"
              % (nm, int((d[t + "_new_overloads"] > 0).sum()),
                 int(d[t + "_new_overloads"].sum()), d[t + "_new_MW"].sum(),
                 100 * d[t + "_cleared"].mean()))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
