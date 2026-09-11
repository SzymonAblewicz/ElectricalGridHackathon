"""Same ordering on both sides.  The 2x2 that isolates the overlap method.

    python merit.py                    all outages x 168 hours
    python merit.py --limit 60         quick check
    python merit.py --hours 24         24 hours spread across the week

WHY THIS SCRIPT EXISTS
----------------------
The first comparison run (protocol.py) changed several things at once:

    EirGrid's published arithmetic  = per-circuit group  +  pro-rata split
    the proposed method             = net-sensitivity group + headroom veto
                                      + effectiveness ordering

Three changes together, and the biggest single term turned out to be the
*ordering* - merit order instead of pro-rata.  That is a fairness change, not a
topology one, and it is not the contribution being claimed.  Quoting it as the
result would be wrong.

So: hold the ordering fixed.  Every arm below dispatches highest-sensitivity
node first, one node at a time, same step rule, same stopping rule, same
threshold.  What varies is only the two things that are the contribution:

    GROUP   per-circuit (worst circuit first, rank on that circuit alone)
            vs net (one group over every circuit currently over, rank on the
            summed relief)
    VETO    off vs on (a node may not be cut past the point where it pushes
            some other branch over)

That is a 2x2.  Four arms, one factor each way:

    blind    per-circuit, no veto     <- the control.  EirGrid's group
                                         definition, given the *same* merit
                                         ordering as the treatment so the
                                         ordering cancels out of the difference.
    ovl      net,         no veto     <- overlap only
    solo     per-circuit, veto        <- veto only
    veto     net,         veto        <- the method

    eir      EirGrid exactly (pro-rata) - kept for reference, not the control
    lp       the exact optimum - the floor no ordering rule can beat

blind -> veto is the number to quote.  blind -> ovl and blind -> solo say which
half of the method it came from, and whether the two interact.
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
OUT = os.path.join(KIT, "results", "N-1 contingencies", "merit")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402
import protocol as P    # noqa: E402


ARMS = (("eir", "EirGrid pro-rata"),
        ("blind", "per-circuit, merit"),
        ("ovl", "net, merit"),
        ("solo", "per-circuit, merit, veto"),
        ("veto", "net, merit, veto"),
        ("lp", "LP optimum"))


def main(argv):
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    nhours = int(argv[argv.index("--hours") + 1]) if "--hours" in argv else None
    slack = argv[argv.index("--slack") + 1] if "--slack" in argv else "load"
    sfx = "" if slack == "load" else "_" + slack
    if "--threshold" in argv:
        P.THRESHOLD = float(argv[argv.index("--threshold") + 1])
        sfx += "_t%s" % str(P.THRESHOLD).replace(".", "")
    os.makedirs(OUT, exist_ok=True)
    gridkit.quiet()

    print("loading network  (slack reference: %s, threshold %.3f)"
          % (slack, P.THRESHOLD))
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
    W = len(wcol)

    if slack == "uniform":
        wt = None
    else:
        dem = n.loads_t.p_set.mean() if len(n.loads_t.p_set.columns) else n.loads["p_set"]
        wt = np.zeros(NB)
        for name, val in dem.items():
            wt[buses.get_loc(n.loads.at[name, "bus"])] += float(val)
        wt /= wt.sum()

    hours = (np.linspace(0, len(n.snapshots) - 1, nhours).astype(int)
             if nhours else np.arange(len(n.snapshots)))
    isl = P.islanding_mask(br, buses)
    todo = range(E if limit is None else min(limit, E))
    print("%d buses, %d branches, %d wind nodes, %d hours" % (NB, E, W, len(hours)))

    M = P0 @ K
    den = 1.0 - np.diag(M)
    fastok = (~isl) & (np.abs(den) > 1e-3)

    def post(k):
        bk = b0.copy()
        bk[k] = 0.0
        if fastok[k]:
            lod = M[:, k] / den[k]
            lod[k] = -1.0
            Pk = P0 + np.outer(lod, P0[k, :])
            Pk[k, :] = 0.0
        else:
            Pk = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        F = Pk @ (p + (K @ (bk * phi))[:, None]) - (bk * phi)[:, None]
        return Pk, F

    rows, broke = [], []
    t0 = time.perf_counter()
    for k in todo:
        if isl[k]:
            continue
        Pk, F = post(k)
        V = -Pk[:, wcol] if wt is None else -(Pk[:, wcol] - (Pk @ wt)[:, None])
        live = np.ones(E, bool)
        live[k] = False

        for h in hours:
            f = F[:, h]
            over = list(np.nonzero((np.abs(f) > s * P.TOL) & live)[0])
            if not over:
                continue
            avail = outp.iloc[h].reindex(wnames).fillna(0.0).to_numpy()
            e_worst = max(over, key=lambda e: abs(f[e]) / s[e])
            base = dict(outage=edges[k], hour=int(h),
                        kind=br["kind"].iloc[k], n_over=len(over),
                        worst_branch=edges[e_worst],
                        worst_kind=br["kind"].iloc[e_worst],
                        over_MW=round(float(sum(abs(f[e]) - s[e] for e in over)), 3),
                        worst_pct=round(100.0 * float((np.abs(f) / s)[live].max()), 2),
                        wind_avail_MW=round(float(avail.sum()), 1))

            res = {}
            res["eir"] = P.arm_eirgrid(V, f, over, avail, s, live)
            # the 2x2 - identical code path, one flag each
            res["blind"] = P.arm_greedy(V, f, over, avail, s, live,
                                        veto=False, net_rank=False)
            res["ovl"] = P.arm_greedy(V, f, over, avail, s, live,
                                      veto=False, net_rank=True)
            res["solo"] = P.arm_greedy(V, f, over, avail, s, live,
                                       veto=True, net_rank=False)
            res["veto"] = P.arm_greedy(V, f, over, avail, s, live,
                                       veto=True, net_rank=True)
            okL, uL, residL = P.arm_lp(V, f, over, avail, s, live)
            res["lp"] = (f + V @ uL, uL)

            for t, _ in ARMS:
                c, u = res[t]
                sc, new = P.score(c, f, over, s, live, u, t)
                base.update(sc)
                for e in new:
                    broke.append(dict(arm=t, outage=edges[k], hour=int(h),
                                      broken=edges[e], kind=br["kind"].iloc[e],
                                      over_MW=round(abs(c[e]) - s[e], 2)))
            base["lp_feasible"] = int(okL)
            rows.append(base)

        if (k + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print("  %4d/%d  %5.1f min  ~%5.1f min left  (%d events)"
                  % (k + 1, len(todo), el / 60,
                     el / (k + 1) * (len(todo) - k - 1) / 60, len(rows)))

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "cases%s.csv" % sfx), index=False)
    if broke:
        pd.DataFrame(broke).to_csv(os.path.join(OUT, "collateral%s.csv" % sfx),
                                   index=False)
    print("\ndone in %.1f min, %d events" % ((time.perf_counter() - t0) / 60, len(d)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
