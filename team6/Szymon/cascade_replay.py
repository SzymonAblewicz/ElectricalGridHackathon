"""Replay one cascade round by round, under two remedies, and export it.

    python cascade_replay.py                       the default case
    python cascade_replay.py 1122-1742-1 50        any (outage, hour)

cascade_protocol.py counts what is lost.  This records *how* it is lost: every
round, every branch loading, every circuit that trips, every megawatt
instructed and where, for each remedy side by side - so the same starting trip
can be watched playing out two ways.

WHAT A ROUND IS
---------------
    recompute the exact DC flows with everything that is out of service out
    -> whatever is over its rating is the next thing to fail
    -> the remedy is dispatched
    -> the MOST overloaded circuit still over its rating trips
    -> repeat

**One circuit trips per round, and it is the worst-loaded one.**  That is not a
simplification for the sake of a tidy picture, it is the closer reading of how
protection behaves: a time-overcurrent relay operates faster the further past
its rating the circuit is, so of several overloaded circuits the worst goes
first, and the flows the others see are the flows *after* it has gone.  Tripping
every overloaded circuit simultaneously - the other common convention - implies
they all reach their limit at the same instant, which is the one thing a
time-overcurrent characteristic says they do not do.

There is still no clock.  Rounds are ordinal, not minutes, and the model cannot
tell you whether a round is seconds or an hour apart.

The starting event is a single branch out of service - a true N-1.  Everything
that happens after it is a consequence of that one outage, or of the remedy
dispatched against it; no second independent fault is ever injected.

LOAD AT RISK
------------
When the chain splits the network, each island is checked for whether the
generation still connected to it covers its own demand:

    deficit = max(0, load_in_island - generation_in_island)

summed over islands.  That is load with nothing left to serve it - the closest
this model gets to saying "blackout", and it stops there, because what happens
next is frequency collapse and this is a DC power flow.

Writes replay.json into
``participant-kit new/results/N-1 contingencies/protocol/replay/``.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
OUT = os.path.join(KIT, "results", "N-1 contingencies", "protocol", "replay")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402
import protocol         # noqa: E402

TOL = protocol.TOL

#: A circuit only counts as over, and only trips, if it is over by more than
#: this many MW.  The remedy solves to exactly the rating, so a bare test sits
#: on the constraint and a difference of 1e-12 in the flow decides whether a
#: circuit trips - and that decides the whole chain.
MARGIN = protocol.MARGIN
MAX_ROUNDS = 22

#: (outage, hour) pairs to export.  Picked from cascade.csv on one rule: the
#: no-remedy chain is long, the published method still loses circuits and
#: splits the network, and the veto holds it.  Nothing is hand-tuned - these
#: are rows of a 250-case run, and the run is in the same folder.
CASES = [("4384-43844-1", 51), ("4242-4462-1", 6), ("1701-1981-1", 50),
         ("1122-1742-1", 50), ("1392-4943-1", 55)]

ARMS = ["nothing", "eir", "veto", "lp"]


def main(argv):
    if len(argv) >= 2:
        cases = [(argv[0], int(argv[1]))]
    else:
        cases = CASES
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
    Pinj = ptdf_all.injections(n, n.snapshots).reindex(buses).fillna(0.0).to_numpy()
    pos = {v: i for i, v in enumerate(buses)}
    ia = np.array([pos[x] for x in br["bus0"]])
    iz = np.array([pos[x] for x in br["bus1"]])

    g = n.generators
    wgen = g.index[g["carrier"] == "wind"]
    outp = n.generators_t.p[wgen].T.groupby(g.loc[wgen, "bus"].to_numpy()).sum().T
    wnames = list(outp.columns)
    wcol = np.array([buses.get_loc(x) for x in wnames])

    dem = n.loads_t.p_set.mean() if len(n.loads_t.p_set.columns) else n.loads["p_set"]
    wt = np.zeros(NB)
    for name, val in dem.items():
        wt[buses.get_loc(n.loads.at[name, "bus"])] += float(val)
    wt /= wt.sum()

    base_comp = connected_components(
        csr_matrix((np.ones(E), (ia, iz)), shape=(NB, NB)), directed=False)[0]

    def comps(dead):
        m = np.ones(E, bool)
        m[list(dead)] = False
        return connected_components(
            csr_matrix((np.ones(int(m.sum())), (ia[m], iz[m])), shape=(NB, NB)),
            directed=False)

    def at_risk(dead, pr):
        """MW of demand sitting in an island that cannot cover itself."""
        ncomp, lab = comps(dead)
        if ncomp <= base_comp:
            return 0.0, ncomp
        tot = 0.0
        for c in range(ncomp):
            net = float(pr[lab == c].sum())
            if net < -1e-6:
                tot += -net
        return tot, ncomp

    def state(dead, x, h):
        bk = b0.copy()
        bk[list(dead)] = 0.0
        Pk = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        pr = Pinj[:, h].copy()
        if x.sum() > 0:
            np.add.at(pr, wcol, -x)
            pr = pr + x.sum() * wt
        f = Pk @ (pr + (K @ (bk * phi))) - bk * phi
        V = -(Pk[:, wcol] - (Pk @ wt)[:, None])
        return f, V, pr

    def replay(k, h, arm):
        dead = [k]
        x = np.zeros(len(wnames))
        avail0 = outp.iloc[h].reindex(wnames).fillna(0.0).to_numpy()
        rounds = []
        for r in range(MAX_ROUNDS):
            ncomp, _ = comps(dead)
            f, V, pr = state(dead, x, h)
            live = np.ones(E, bool)
            live[dead] = False
            load = np.abs(f) / s
            load[~live] = 0.0
            over = list(np.nonzero((np.abs(f) > s * TOL + MARGIN) & live)[0])
            risk, _ = at_risk(dead, pr)

            before = dict(
                round=r + 1, dead=[edges[i] for i in dead],
                over=[edges[e] for e in over],
                loading=[round(float(v), 4) for v in load],
                islands=int(ncomp - base_comp), load_at_risk=round(risk, 1),
                cut_MW=round(float(x.sum()), 1),
                cut_nodes=int((x > 1e-6).sum()))
            if not over:
                before["action"] = "nothing over rating - the chain stops here"
                before["trips"] = []
                w = [round(float(v), 2) for v in np.maximum(avail0 - x, 0.0)]
                before["wind_before"] = w
                before["wind_after"] = w
                rounds.append(before)
                break

            avail = np.maximum(avail0 - x, 0.0)
            if arm == "nothing":
                cur, u = f, np.zeros(len(wnames))
            elif arm == "eir":
                cur, u = protocol.arm_eirgrid(V, f, over, avail, s, live)
            elif arm == "veto":
                cur, u = protocol.arm_greedy(V, f, over, avail, s, live, veto=True)
            else:
                ok, u, _ = protocol.arm_lp(V, f, over, avail, s, live)
                cur = f + V @ u
            x = x + u
            # the worst-loaded circuit goes first, and alone
            hot = np.nonzero((np.abs(cur) > s * TOL + MARGIN) & live)[0]
            still = ([int(hot[np.argmax((np.abs(cur) / s)[hot])])]
                     if len(hot) else [])

            after = np.abs(cur) / s
            after[~live] = 0.0
            before["loading_after"] = [round(float(v), 4) for v in after]
            # what every farm is generating before and after this round's
            # instruction.  The map sizes each farm by these, so a dispatch-down
            # is visible as the dot shrinking rather than as a number in a table.
            before["wind_before"] = [round(float(v), 2) for v in np.maximum(avail0 - (x - u), 0.0)]
            before["wind_after"] = [round(float(v), 2) for v in np.maximum(avail0 - x, 0.0)]
            before["instructed_MW"] = round(float(u.sum()), 1)
            # this round's group size.  cut_nodes is the running total and the
            # instructed list is truncated for size, so neither answers "how
            # many farms did this instruction touch" - which is the number the
            # comparison turns on.
            before["instructed_nodes"] = int((u > 1e-6).sum())
            before["instructed"] = sorted(
                [{"node": wnames[i], "MW": round(float(u[i]), 1)}
                 for i in np.nonzero(u > 1e-6)[0]],
                key=lambda d: -d["MW"])[:12]
            before["trips"] = [edges[e] for e in still]
            before["action"] = (
                "no remedy" if arm == "nothing" else
                "cut %.1f MW at %d node%s" % (u.sum(), int((u > 1e-6).sum()),
                                              "" if (u > 1e-6).sum() == 1 else "s"))
            rounds.append(before)
            if not still:
                break
            dead = dead + still
        risk, ncomp = at_risk(dead, state(dead, x, h)[2])
        return dict(arm=arm, rounds=rounds, lost=len(dead) - 1,
                    islands=int(ncomp - base_comp),
                    load_at_risk=round(risk, 1),
                    total_cut=round(float(x.sum()), 1),
                    dead=[edges[i] for i in dead])

    idx = {v: i for i, v in enumerate(edges)}

    # Drawing positions follow the same two rules as every other map in this
    # work, so the pictures are comparable: substations sharing a name collapse
    # to one point, and buses with no usable geocode are not drawn.  Both live
    # in figures.py; importing them beats reimplementing them slightly wrong.
    import figures as figmod
    bmeta = pd.read_csv(os.path.join(KIT, "results", "network", "buses.csv"),
                        index_col=0)
    bmeta = figmod.snap_stations(bmeta, br)
    placed = figmod.placed_only(bmeta)
    px = bmeta["x"].reindex(buses).to_numpy(float)
    py = bmeta["y"].reindex(buses).to_numpy(float)
    ok = np.array([b in placed.index for b in buses])
    px = np.where(ok, px, np.nan)
    py = np.where(ok, py, np.nan)

    wnom = (n.generators[n.generators["carrier"] == "wind"]
            .groupby("bus")["p_nom"].sum().reindex(wnames).fillna(0.0).to_numpy())
    payload = {
        "buses": [{"x": None if not np.isfinite(px[i]) else round(float(px[i]), 5),
                   "y": None if not np.isfinite(py[i]) else round(float(py[i]), 5),
                   "kv": float(n.buses["v_nom"].iloc[i])}
                  for i in range(len(buses))],
        "branches": [{"id": str(e), "a": int(ia[i]), "b": int(iz[i]),
                      "s": round(float(s[i]), 1),
                      "t": 1 if br["kind"].iloc[i] == "Transformer" else 0}
                     for i, e in enumerate(edges)],
        "wind": [{"id": str(w), "bus": int(wcol[j]),
                  "p_nom": round(float(wnom[j]), 1)}
                 for j, w in enumerate(wnames)],
        "cases": [],
    }

    for name, h in cases:
        if name not in idx:
            print("  no branch called %s - skipped" % name)
            continue
        k = idx[name]
        print("\n%s at hour %d" % (name, h))
        rec = {"outage": name, "hour": int(h),
               "s_nom": float(s[k]), "kind": str(br["kind"].iloc[k]),
               "arms": {}}
        for arm in ARMS:
            r = replay(k, h, arm)
            rec["arms"][arm] = r
            print("   %-8s %2d circuits lost, %d rounds, %6.1f MW cut, "
                  "%d islands, %.0f MW load at risk"
                  % (arm, r["lost"], len(r["rounds"]), r["total_cut"],
                     r["islands"], r["load_at_risk"]))
        payload["cases"].append(rec)

    p = os.path.join(OUT, "replay.json")
    with open(p, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print("\n-> %s  (%.1f MB)" % (os.path.relpath(p, HERE),
                                  os.path.getsize(p) / 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
