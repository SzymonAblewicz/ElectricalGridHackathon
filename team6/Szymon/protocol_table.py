"""The lookup table the protocol reads, and how long the live half of it takes.

    python protocol_table.py

protocol.py asks whether the method works.  This writes the thing an operator
would actually load, and times the part that has to happen while a conductor is
heating up.

THE SPLIT THAT MAKES IT A LOOKUP
--------------------------------
Two of the three quantities the protocol needs are fixed by the network's
topology and reactances alone.  They do not move with the weather, the dispatch
or the hour:

    which circuits overload when branch k trips     - from the post-trip PTDF
    which wind nodes can relieve each of them       - from the post-trip PTDF
    how many megawatts to cut                       - needs the live flows

So the first two are precomputed here, once, for every single-branch outage.
Only the third is live, and it is a matrix-vector product and a small LP.

WHAT IS IN THE TABLE
--------------------
groups.csv, one row per (outage, overloaded circuit, wind node):

    sens_target     MW off this circuit per MW cut at this node
    sens_net        the same summed over every circuit that outage congests.
                    This is the overlap score, and it is the ordering the
                    protocol uses - a node that takes 0.09 off three circuits
                    outranks one that takes 0.20 off one of them.
    in_eirgrid      would the published threshold rule have included it
    harms           how many *other* live branches this node pushes towards
                    their limit, at 0.05 or worse.  A node with a high
                    sens_target and a large harms count is exactly the trap
                    the veto exists to catch.

trips.csv, one row per (outage, overloaded circuit): what to look up, and the
size of the group it points at.

Writes into ``participant-kit new/results/N-1 contingencies/protocol/lookup/``.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import linprog

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
OUT = os.path.join(KIT, "results", "N-1 contingencies", "protocol", "lookup")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402
import protocol         # noqa: E402

TOL = protocol.TOL
T = protocol.THRESHOLD


def main():
    os.makedirs(OUT, exist_ok=True)
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
    wg = g.index[g["carrier"] == "wind"]
    outp = n.generators_t.p[wg].T.groupby(g.loc[wg, "bus"].to_numpy()).sum().T
    wnames = list(outp.columns)
    wcol = np.array([buses.get_loc(x) for x in wnames])

    dem = n.loads_t.p_set.mean() if len(n.loads_t.p_set.columns) else n.loads["p_set"]
    wt = np.zeros(NB)
    for name, val in dem.items():
        wt[buses.get_loc(n.loads.at[name, "bus"])] += float(val)
    wt /= wt.sum()

    isl = protocol.islanding_mask(br, buses)
    M = P0 @ K
    den = 1.0 - np.diag(M)
    fastok = (~isl) & (np.abs(den) > 1e-3)
    pe = p + (K @ (b0 * phi))[:, None]

    rows, trips = [], []
    tim = {"ptdf": [], "flows": [], "screen": [], "group": [], "lp": []}
    t0 = time.perf_counter()

    for k in range(E):
        if isl[k]:
            continue
        bk = b0.copy()
        bk[k] = 0.0

        t = time.perf_counter()
        if fastok[k]:
            lod = M[:, k] / den[k]
            lod[k] = -1.0
            P = P0 + np.outer(lod, P0[k, :])
            P[k, :] = 0.0
        else:
            P = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        tim["ptdf"].append(time.perf_counter() - t)

        F = P @ (p + (K @ (bk * phi))[:, None]) - (bk * phi)[:, None]
        V = -(P[:, wcol] - (P @ wt)[:, None])
        live = np.ones(E, bool)
        live[k] = False
        load = np.abs(F) / s[:, None]
        load[k] = 0.0
        ever = np.nonzero((load > TOL).any(axis=1))[0]
        if not len(ever):
            continue

        # the hour this outage bites hardest - the one the table is written for
        h = int(load[ever].max(axis=0).argmax())
        f = F[:, h]

        t = time.perf_counter()
        _ = np.nonzero((np.abs(f) > s * TOL) & live)[0]
        tim["screen"].append(time.perf_counter() - t)
        t = time.perf_counter()
        _ = P @ pe[:, h]
        tim["flows"].append(time.perf_counter() - t)

        over = list(np.nonzero((np.abs(f) > s * TOL) & live)[0])
        rel = protocol.relief(V, f)
        net = rel[over].sum(axis=0)
        # how many other live branches does cutting here push towards its limit
        harm = ((rel < -T) & live[:, None]).sum(axis=0)
        avail = outp.iloc[h].reindex(wnames).fillna(0.0).to_numpy()

        t = time.perf_counter()
        lm = np.where(np.abs(f) > s * TOL, np.abs(f), s * TOL)
        caps = np.array([protocol.cap(V, f, lm, live, i) for i in range(len(wnames))])
        tim["group"].append(time.perf_counter() - t)

        t = time.perf_counter()
        protocol.arm_lp(V, f, over, avail, s, live)
        tim["lp"].append(time.perf_counter() - t)

        for e in ever:
            se = rel[e] if e in over else (-V[e] * (np.sign(f[e]) or 1.0))
            grp = np.nonzero(net > T)[0]
            grp = grp[np.argsort(-net[grp])]
            trips.append(dict(
                outage=edges[k], circuit=edges[e], kind=br["kind"].iloc[e],
                s_nom=s[e], worst_loading_pct=round(100 * float(load[e].max()), 2),
                hours_over=int((load[e] > TOL).sum()),
                group_size=len(grp),
                eirgrid_group_size=int((se >= T).sum()),
                vetoed_from_eirgrid=int(((se >= T) & (caps < 1e-6)).sum()),
                best_node=wnames[int(np.argmax(net))] if len(grp) else "",
            ))
            for r, i in enumerate(grp):
                rows.append(dict(
                    outage=edges[k], circuit=edges[e], rank=r + 1,
                    node=wnames[i], sens_target=round(float(se[i]), 5),
                    sens_net=round(float(net[i]), 5),
                    in_eirgrid=int(se[i] >= T), harms=int(harm[i]),
                    veto_cap_MW=round(float(caps[i]), 2),
                    avail_MW=round(float(avail[i]), 2)))

        if (k + 1) % 200 == 0:
            print("  %4d/%d  %.1f min" % (k + 1, E, (time.perf_counter() - t0) / 60))

    gr = pd.DataFrame(rows)
    tr = pd.DataFrame(trips)
    gr.to_csv(os.path.join(OUT, "groups.csv"), index=False)
    tr.to_csv(os.path.join(OUT, "trips.csv"), index=False)

    print("\nTHE TABLE")
    print("  outages with at least one overload   %d" % tr.outage.nunique())
    print("  (outage, circuit) entries            %d" % len(tr))
    print("  distinct circuits ever at risk       %d" % tr.circuit.nunique())
    print("  (outage, circuit, node) rows         %s" % f"{len(gr):,}")
    print("  median group size, overlap order     %.0f nodes"
          % tr.group_size.median())
    print("  median group size, EirGrid threshold %.0f nodes"
          % tr.eirgrid_group_size.median())
    print("  EirGrid members the veto excludes    %d of %d (%.1f%%)"
          % (tr.vetoed_from_eirgrid.sum(), tr.eirgrid_group_size.sum(),
             100 * tr.vetoed_from_eirgrid.sum() / max(tr.eirgrid_group_size.sum(), 1)))
    print("  nodes that harm >= 1 other branch    %.1f%% of group rows"
          % (100 * (gr.harms > 0).mean()))
    print("  median branches harmed per node      %.0f" % gr.harms.median())

    print("\nONLINE COST, per trip (median over %d outages)" % len(tim["ptdf"]))
    for key, name in (("ptdf", "rebuild the post-trip PTDF (precomputable)"),
                      ("flows", "flows on all 980 branches"),
                      ("screen", "find what is over its rating"),
                      ("group", "cap every wind node against every branch"),
                      ("lp", "solve for the minimum safe dispatch")):
        v = np.median(tim[key]) * 1000
        print("  %-44s %8.3f ms" % (name, v))
    live_ms = sum(np.median(tim[key]) for key in ("flows", "screen", "group", "lp"))
    print("  %-44s %8.3f ms" % ("TOTAL live decision", live_ms * 1000))
    print("\n-> %s" % os.path.relpath(OUT, HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
