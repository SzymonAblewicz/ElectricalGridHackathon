"""Read merit.py's cases and answer one question: what does the overlap method
save when the ordering is held fixed on both sides?

    python merit_report.py
    python merit_report.py --file cases_uniform.csv

WHAT IS COMPARED, AND WHY IT IS LIKE-FOR-LIKE
---------------------------------------------
Every arm dispatches highest-sensitivity-first.  So the ordering, which is the
fairness-sensitive part and not the contribution, cancels.  What is left is the
2x2:

                     no veto        veto
    per-circuit      blind          solo
    net              ovl            veto

blind is the control.  veto is the method.  ovl and solo split the difference
into its two halves and show whether they interact.

MW curtailed is only comparable between two arms on events where *both* cleared
the overload.  An arm that gives up early looks cheap.  Every headline number
below is on the paired subset, and the events each arm fails are counted
separately so nothing hides.

Paired Wilcoxon on the per-event differences, and a bootstrap interval on the
ratio of totals, because a handful of large events dominate the sum and the
median and the total say different things.
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "participant-kit new", "results",
                   "N-1 contingencies", "merit")

ARMS = [("eir", "EirGrid pro-rata (reference)"),
        ("blind", "per-circuit, merit  [CONTROL]"),
        ("ovl", "net, merit"),
        ("solo", "per-circuit, merit, veto"),
        ("veto", "net, merit, veto  [METHOD]"),
        ("lp", "LP optimum (floor)")]

# EirGrid/SONI Annual Renewable Constraint and Curtailment Report 2025.
# Constraint is the local-network half of dispatch-down - the half a constraint
# group is for.  Curtailment (SNSP/MUON) is system-wide and out of scope here.
ROI_WIND_DD_PCT = 11.3          # 2025: 6.6 constraint + 4.7 curtailment
ROI_CONSTRAINT_PCT = 6.6
ALL_ISLAND_WIND_GWH = 13288.0   # 2024
ALL_ISLAND_DD_GWH = 2181.0      # 2024, 14.0%


def boot_ratio(a, b, n=5000, seed=0):
    """Bootstrap CI on sum(b)/sum(a) - resampled by event, which is the unit."""
    rng = np.random.default_rng(seed)
    m = len(a)
    idx = rng.integers(0, m, size=(n, m))
    r = b[idx].sum(1) / np.maximum(a[idx].sum(1), 1e-9)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def pair(d, x, y):
    """One arm against another on the events both of them cleared."""
    m = (d[x + "_cleared"] == 1) & (d[y + "_cleared"] == 1)
    s = d[m]
    a = s[x + "_mw"].to_numpy(float)
    b = s[y + "_mw"].to_numpy(float)
    if not len(s):
        return None
    diff = a - b                       # positive = y saved MW
    tot_a, tot_b = a.sum(), b.sum()
    lo, hi = boot_ratio(a, b)
    # per-event saving, on events where the control actually cut something
    nz = a > 1e-6
    pct = 100.0 * diff[nz] / a[nz]
    try:
        w = wilcoxon(a, b, zero_method="zsplit").pvalue if (diff != 0).any() else 1.0
    except ValueError:
        w = 1.0
    return dict(
        n=len(s), tot_a=tot_a, tot_b=tot_b,
        tot_save_pct=100.0 * (tot_a - tot_b) / max(tot_a, 1e-9),
        ci=(100 * (1 - hi), 100 * (1 - lo)),
        med_a=float(np.median(a)), med_b=float(np.median(b)),
        med_pct=float(np.median(pct)) if len(pct) else 0.0,
        wins=int((diff > 1e-6).sum()), losses=int((diff < -1e-6).sum()),
        ties=int((np.abs(diff) <= 1e-6).sum()), p=w,
        only_x=int(((d[x + "_cleared"] == 1) & (d[y + "_cleared"] == 0)).sum()),
        only_y=int(((d[x + "_cleared"] == 0) & (d[y + "_cleared"] == 1)).sum()))


def line(label, r):
    if r is None:
        print("  %-34s no paired events" % label)
        return
    print("  %-34s %7d %10.1f %10.1f %8.2f%%  [%5.2f, %5.2f]  %7.1f%%  %5d/%-5d %9.1e"
          % (label, r["n"], r["tot_a"], r["tot_b"], r["tot_save_pct"],
             r["ci"][0], r["ci"][1], r["med_pct"], r["wins"], r["losses"], r["p"]))


def main(argv):
    fn = argv[argv.index("--file") + 1] if "--file" in argv else "cases.csv"
    d = pd.read_csv(os.path.join(OUT, fn))
    print("=" * 100)
    print("MERIT-ORDER-HELD-FIXED COMPARISON      %s      %d events, %d outages, %d hours"
          % (fn, len(d), d.outage.nunique(), d.hour.nunique()))
    print("=" * 100)

    # ---------------------------------------------------------------- arms
    print("\nPER-ARM, ALL %d EVENTS" % len(d))
    print("  %-34s %8s %10s %10s %9s %9s %8s"
          % ("", "cleared", "total MW", "median MW", "new ovl", "events", "nodes"))
    print("  " + "-" * 92)
    for t, name in ARMS:
        print("  %-34s %7.1f%% %10.0f %10.2f %9d %9d %8.0f"
              % (name, 100 * d[t + "_cleared"].mean(), d[t + "_mw"].sum(),
                 d[t + "_mw"].median(), int(d[t + "_new_overloads"].sum()),
                 int((d[t + "_new_overloads"] > 0).sum()),
                 d[t + "_nodes"].median()))

    # ------------------------------------------------------------- the 2x2
    print("\n\nTHE 2x2 - every pair is like-for-like (both arms cleared)")
    print("  positive saving = the second arm used less wind")
    print("  %-34s %7s %10s %10s %9s %18s %8s %11s %9s"
          % ("control -> treatment", "n", "ctrl MW", "trt MW", "saving",
             "95% CI", "median", "win/loss", "p"))
    print("  " + "-" * 132)
    for x, y, lab in (("blind", "ovl", "overlap only"),
                      ("blind", "solo", "veto only"),
                      ("blind", "veto", "BOTH = the method"),
                      ("solo", "veto", "overlap, given veto"),
                      ("ovl", "veto", "veto, given overlap"),
                      ("eir", "veto", "vs EirGrid as published"),
                      ("blind", "lp", "how much is left on the table")):
        line("%-7s -> %-7s  %s" % (x, y, lab), pair(d, x, y))

    # ------------------------------------------------------------- safety
    print("\n\nSAFETY - remedies that created a NEW overload somewhere else")
    print("  %-34s %10s %12s %12s" % ("", "events", "circuits", "MW over"))
    print("  " + "-" * 70)
    for t, name in ARMS:
        print("  %-34s %10d %12d %12.1f"
              % (name, int((d[t + "_new_overloads"] > 0).sum()),
                 int(d[t + "_new_overloads"].sum()), d[t + "_new_MW"].sum()))

    # --------------------------------------------------- where it comes from
    print("\n\nWHERE THE OVERLAP SAVING LIVES  (blind -> veto, by how many "
          "circuits were over at once)")
    m = (d.blind_cleared == 1) & (d.veto_cleared == 1)
    s = d[m].copy()
    s["save"] = s.blind_mw - s.veto_mw
    b = s.groupby(np.clip(s.n_over, 1, 5))
    print("  %-12s %8s %12s %12s %10s %10s"
          % ("circuits", "events", "ctrl MW", "method MW", "saving", "win rate"))
    print("  " + "-" * 70)
    for k, grp in b:
        ta, tb = grp.blind_mw.sum(), grp.veto_mw.sum()
        print("  %-12s %8d %12.1f %12.1f %9.2f%% %9.1f%%"
              % ("%d%s" % (k, "+" if k == 5 else ""), len(grp), ta, tb,
                 100 * (ta - tb) / max(ta, 1e-9),
                 100 * (grp.save > 1e-6).mean()))

    # ---------------------------------------------------------- real scale
    r = pair(d, "blind", "veto")
    if r:
        f = r["tot_save_pct"] / 100.0
        print("\n\nWHAT THAT FRACTION IS WORTH, IF IT HOLDS ON THE REAL SYSTEM")
        print("  The method acts on network constraint - the local half of")
        print("  dispatch-down.  RoI 2025: constraint %.1f%% of wind, curtailment"
              % ROI_CONSTRAINT_PCT)
        print("  %.1f%%, together %.1f%%.  All-island wind 2024: %.0f GWh."
              % (ROI_WIND_DD_PCT - ROI_CONSTRAINT_PCT, ROI_WIND_DD_PCT,
                 ALL_ISLAND_WIND_GWH))
        con_gwh = ALL_ISLAND_WIND_GWH * ROI_CONSTRAINT_PCT / 100.0
        print("  constraint volume, all-island scale      %8.0f GWh/yr" % con_gwh)
        print("  x %.2f%% saving                           %8.0f GWh/yr"
              % (r["tot_save_pct"], con_gwh * f))
        print("  as a share of ALL dispatch-down (%.0f GWh)  %8.2f%%"
              % (ALL_ISLAND_DD_GWH, 100 * con_gwh * f / ALL_ISLAND_DD_GWH))
        print("  as a share of wind generated              %8.3f%%"
              % (100 * con_gwh * f / ALL_ISLAND_WIND_GWH))
        print("  homes-year equivalent (4.2 MWh/home)      %8.0f homes"
              % (con_gwh * f * 1000 / 4.2))
        print("\n  This is a scaling, not a measurement.  It assumes the saving")
        print("  fraction found on post-N-1 congestion in this network carries")
        print("  to the real annual constraint tonnage.  Say so when you quote it.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
