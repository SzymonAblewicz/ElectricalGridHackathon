"""Which wind nodes are ever dispatched down, across every N-1 event.

    python wind_usage.py

protocol.py answers whether the remedy is safe.  This answers a different
question, at the level of individual wind farms: over every (outage, hour)
event in the study - the same 20,203 the headline numbers come from - does
the safe (net-sensitivity + veto) remedy ever ask this farm to cut, and if so
how much and how often.

A farm that is never asked, in any of the 20,203 events, needs no constraint
group at all: it can be dispatched purely on economics and run at whatever the
wind gives it, because nothing in the safe method ever finds it worth curtailing
without breaking something else.

Only the veto arm is run here - eir/ovl/lp add nothing to this question and
tripling the per-event cost for no reason would be wasted.

Writes wind_usage.csv into
``participant-kit new/results/N-1 contingencies/protocol/``.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
OUT = os.path.join(KIT, "results", "N-1 contingencies", "protocol")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402
import protocol         # noqa: E402

TOL = protocol.TOL


def main():
    gridkit.quiet()
    print("loading network")
    n = gridkit.load("WP2033", "all-island")
    br, K, b0, phi, P0, rank = ptdf_all.build(n)
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)

    buses, edges = n.buses.index, br.index
    E, NB = len(br), len(buses)
    s = br["s_nom"].to_numpy(float)
    p = ptdf_all.injections(n, n.snapshots).reindex(buses).fillna(0.0).to_numpy()

    g = n.generators
    wind = g.index[g["carrier"] == "wind"]
    outp = n.generators_t.p[wind].T.groupby(g.loc[wind, "bus"].to_numpy()).sum().T
    wnames = list(outp.columns)
    wcol = np.array([buses.get_loc(x) for x in wnames])
    wnom = g[g["carrier"] == "wind"].groupby("bus")["p_nom"].sum().reindex(wnames).fillna(0.0)
    W = len(wnames)

    dem = n.loads_t.p_set.mean() if len(n.loads_t.p_set.columns) else n.loads["p_set"]
    wt = np.zeros(NB)
    for name, val in dem.items():
        wt[buses.get_loc(n.loads.at[name, "bus"])] += float(val)
    wt /= wt.sum()

    isl = protocol.islanding_mask(br, buses)
    M = P0 @ K
    den = 1.0 - np.diag(M)
    fastok = (~isl) & (np.abs(den) > 1e-3)

    hours = np.arange(len(n.snapshots))
    cut_total = np.zeros(W)          # MW, summed over every event
    cut_events = np.zeros(W, int)    # how many events this farm was cut in
    avail_total = np.zeros(W)        # MW available, summed over every event it saw an overload
    n_events = 0

    t0 = time.perf_counter()
    for k in range(E):
        if isl[k]:
            continue
        bk = b0.copy()
        bk[k] = 0.0
        if fastok[k]:
            lod = M[:, k] / den[k]
            lod[k] = -1.0
            P = P0 + np.outer(lod, P0[k, :])
            P[k, :] = 0.0
        else:
            P = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        F = P @ (p + (K @ (bk * phi))[:, None]) - (bk * phi)[:, None]
        V = -(P[:, wcol] - (P @ wt)[:, None])
        live = np.ones(E, bool)
        live[k] = False

        for h in hours:
            f = F[:, h]
            over = list(np.nonzero((np.abs(f) > s * TOL) & live)[0])
            if not over:
                continue
            n_events += 1
            avail = outp.iloc[h].reindex(wnames).fillna(0.0).to_numpy()
            avail_total += avail
            _, u = protocol.arm_greedy(V, f, over, avail, s, live, veto=True)
            cut_total += u
            cut_events += (u > 1e-6)

        if (k + 1) % 200 == 0:
            print("  %4d/%d  %.1f min  (%d events so far)"
                  % (k + 1, E, (time.perf_counter() - t0) / 60, n_events))

    d = pd.DataFrame({
        "node": wnames,
        "p_nom_MW": wnom.reindex(wnames).to_numpy(),
        "cut_events": cut_events,
        "cut_total_MW": np.round(cut_total, 2),
        "avail_total_MW": np.round(avail_total, 1),
    })
    d["pct_events_cut"] = np.round(100 * d.cut_events / max(n_events, 1), 4)
    d["pct_MW_offered_kept"] = np.round(
        100 * (1 - d.cut_total_MW / d.avail_total_MW.replace(0, np.nan)), 3)
    d = d.sort_values("cut_events")
    d.to_csv(os.path.join(OUT, "wind_usage.csv"), index=False)

    never = d[d.cut_events == 0]
    print("\ntotal events (outage, hour) with an overload: %d" % n_events)
    print("wind nodes: %d   total nameplate: %.0f MW" % (len(d), d.p_nom_MW.sum()))
    print("\nnodes NEVER cut by the safe (net-sensitivity + veto) remedy, in any "
          "of the %d events:" % n_events)
    print("  count: %d of %d (%.1f%%)" % (len(never), len(d), 100 * len(never) / len(d)))
    print("  their combined nameplate capacity: %.0f MW (%.1f%% of the %.0f MW fleet)"
          % (never.p_nom_MW.sum(), 100 * never.p_nom_MW.sum() / d.p_nom_MW.sum(),
             d.p_nom_MW.sum()))
    print("\ntotal MW cut across the whole study (safe remedy): %.0f" % d.cut_total_MW.sum())
    print("total MW offered across the whole study: %.0f" % d.avail_total_MW.sum())
    print("\n-> %s" % os.path.relpath(os.path.join(OUT, "wind_usage.csv"), HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
