"""The 2x2, with the greedy fixed and the ranking rule swept.

    python merit2.py                              WP2033 all-island, 168 h
    python merit2.py --scenario SV2024
    python merit2.py --limit 60 --hours 24        quick check
    python merit2.py --slack uniform              robustness re-run

Differs from merit.py in two ways: it calls arms2.arm (see that module for the
two defects it repairs), and it runs four ranking rules instead of two so the
question "does breadth beat depth" is answered by the data rather than by one
arbitrary score.

ARMS
----
    eir       EirGrid as published - per-circuit group, pro-rata split.
              Reference only.  Its ordering differs, so it is not the control.

    ------- everything below dispatches highest-score-first, same step,
    ------- same stopping rule, same threshold.  Ordering cancels.

    blind     rank on the worst violated circuit alone.  A per-circuit
              constraint group in merit order.  THE CONTROL.
    sum       rank on the summed relief over every violated circuit.
    need      rank on need-weighted mean relief over violated circuits.
    mmx       rank on the bottleneck - worst fractional progress.
    bvet      blind + headroom veto
    nvet      need  + headroom veto      THE METHOD
    svet      sum   + headroom veto
    oldstep   sum + veto, with the ORIGINAL single-circuit step, so the size
              of the defect is visible rather than asserted.
    lp        exact minimum MW.  The floor.  Nothing greedy can beat it, and
              the gap from blind to lp is the entire space any better group
              definition has to work in.
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
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402
import protocol as P    # noqa: E402
import arms2            # noqa: E402

OUT = os.path.join(KIT, "results", "N-1 contingencies", "merit2")

SPEC = [("blind", dict(rank="worst", veto=False)),
        ("sum",   dict(rank="sum",   veto=False)),
        ("need",  dict(rank="need",  veto=False)),
        ("mmx",   dict(rank="minmax", veto=False)),
        ("bvet",  dict(rank="worst", veto=True)),
        ("nvet",  dict(rank="need",  veto=True)),
        ("svet",  dict(rank="sum",   veto=True)),
        ("oldstep", dict(rank="sum", veto=True, step="worst"))]
#: groups drawn from the intact network and applied after a trip.  stale =
#: roster, ranking and step size all from the study case; roster = stale list
#: but ranked and sized on the true post-outage sensitivities, so the
#: difference between the two isolates the roster.
STALE = [("stale", dict(roster_only=False)),
         ("roster", dict(roster_only=True))]
#: groups dispatched as UNITS - a set with participation factors, scaled by one
#: lambda, which is what a constraint group actually is.  gper handles one
#: congested circuit at a time and re-cuts the shared members; gnet spans every
#: congested circuit and cuts them once.  Same participation rule in both, so
#: the equity posture is identical and only the group definition differs.
GROUPS = [("gper", dict(net=False, veto=False)),
          ("gnet", dict(net=True,  veto=False)),
          ("gperv", dict(net=False, veto=True)),
          ("gnetv", dict(net=True,  veto=True))]
TAGS = (["eir"] + [t for t, _ in SPEC] + [t for t, _ in STALE]
        + [t for t, _ in GROUPS] + ["lp"])


def main(argv):
    def opt(k, d=None, cast=str):
        return cast(argv[argv.index(k) + 1]) if k in argv else d

    limit = opt("--limit", None, int)
    nhours = opt("--hours", None, int)
    slack = opt("--slack", "load")
    scen = opt("--scenario", "WP2033")
    scope = opt("--scope", "all-island")
    if "--threshold" in argv:
        P.THRESHOLD = float(argv[argv.index("--threshold") + 1])
    sfx = "_%s_%s" % (scen, scope)
    if slack != "load":
        sfx += "_" + slack
    if "--threshold" in argv:
        sfx += "_t%s" % str(P.THRESHOLD).replace(".", "")
    os.makedirs(OUT, exist_ok=True)
    gridkit.quiet()

    print("%s / %s   slack=%s   threshold=%.3f" % (scen, scope, slack, P.THRESHOLD))
    n = gridkit.load(scen, scope)
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
    print("%d buses, %d branches, %d wind nodes, %d hours, %d islanding outages"
          % (NB, E, len(wcol), len(hours), int(isl.sum())))

    # the intact-network sensitivity matrix - what a precomputed constraint
    # group list is drawn from, before anything has tripped
    V0 = -P0[:, wcol] if wt is None else -(P0[:, wcol] - (P0 @ wt)[:, None])

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
        return Pk, Pk @ (p + (K @ (bk * phi))[:, None]) - (bk * phi)[:, None]

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
            base = dict(scenario=scen, outage=edges[k], hour=int(h),
                        kind=br["kind"].iloc[k], n_over=len(over),
                        worst_branch=edges[e_worst],
                        worst_kind=br["kind"].iloc[e_worst],
                        over_MW=round(float(sum(abs(f[e]) - s[e] for e in over)), 3),
                        worst_pct=round(100.0 * float((np.abs(f) / s)[live].max()), 2),
                        wind_avail_MW=round(float(avail.sum()), 1))

            res = {"eir": P.arm_eirgrid(V, f, over, avail, s, live)}
            for t, kw in SPEC:
                res[t] = arms2.arm(V, f, over, avail, s, live, **kw)
            for t, kw in STALE:
                res[t] = arms2.arm_stale(V, V0, f, over, avail, s, live, **kw)
            for t, kw in GROUPS:
                res[t] = arms2.arm_group(V, f, over, avail, s, live, **kw)
            okL, uL, residL = P.arm_lp(V, f, over, avail, s, live)
            res["lp"] = (f + V @ uL, uL)

            for t in TAGS:
                c, u = res[t]
                sc, new = P.score(c, f, over, s, live, u, t)
                base.update(sc)
                for e in new:
                    broke.append(dict(arm=t, scenario=scen, outage=edges[k],
                                      hour=int(h), broken=edges[e],
                                      kind=br["kind"].iloc[e],
                                      over_MW=round(abs(c[e]) - s[e], 2)))
            base["lp_feasible"] = int(okL)
            rows.append(base)

        if (k + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print("  %4d/%d  %5.1f min  ~%5.1f min left  (%d events)"
                  % (k + 1, len(todo), el / 60,
                     el / (k + 1) * (len(todo) - k - 1) / 60, len(rows)), flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "cases%s.csv" % sfx), index=False)
    if broke:
        pd.DataFrame(broke).to_csv(os.path.join(OUT, "collateral%s.csv" % sfx),
                                   index=False)
    print("\ndone in %.1f min, %d events -> cases%s.csv"
          % ((time.perf_counter() - t0) / 60, len(d), sfx))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
