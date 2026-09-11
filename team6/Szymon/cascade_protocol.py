"""Does the protocol actually stop a cascade?  N-1-1 and beyond.

    python cascade_protocol.py                 the standard 600-case sample
    python cascade_protocol.py --cases 120     a smaller one

protocol.py answers a single question: after one trip, does the remedy break
anything.  This answers the question that matters operationally - if the remedy
does break something, where does it end?

THE MODEL OF A CASCADE
----------------------
One round is: recompute flows exactly with everything that is out of service
out of service; anything over its rating is a candidate to fail; apply the
remedy; anything *still* over its rating after the remedy trips; repeat.

That is the standard DC cascade model and its assumption is worth stating
plainly: a circuit over its rating at the end of a round is taken to trip, and
a circuit under it is taken to survive.  Real protection has a time-overcurrent
characteristic, so a circuit at 101% survives far longer than one at 160% and
may never trip at all.  This model has no clock in it, so it reads as the
pessimistic end of the range.  Rounds are ordinal, not minutes.

Four remedies, identical starting conditions:

    NOTHING   the domino, unopposed.  The baseline the protocol is measured
              against.
    EIRGRID   per-circuit group, threshold membership, pro-rata split.
    VETO      net-relief ordering with the hard cap.
    LP        minimum MW subject to every live branch ending in limit.

CARRYING CURTAILMENT BETWEEN ROUNDS
-----------------------------------
Wind cut in round 1 is still cut in round 2, and the injection vector says so:

    p_round = p - sum_n x_n e_n + (sum_n x_n) w

with w the load share that takes the balancing megawatt.  So each round is a
genuine power flow on the network as it then stands, not a flow correction
applied on top of a previous flow correction.

A cascade that splits the network is stopped and recorded as ``islanded``.
Past that point the DC pseudoinverse smears the island's imbalance across its
buses and the flows stop meaning anything; calling that a number of lines lost
would be inventing precision.

Writes cascade.csv and cascade_detail.csv into
``participant-kit new/results/N-1 contingencies/protocol/``.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
OUT = os.path.join(KIT, "results", "N-1 contingencies", "protocol")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402
import protocol         # noqa: E402  - the same four remedies, not a copy

TOL = protocol.TOL

#: A chain this long has already answered the question.  The cost of each round
#: grows with the number of circuits down - a run that loses sixty of them
#: spends minutes in the remedy solvers to tell you what round four told you -
#: so the chain is cut here and the case is recorded as ``runaway``.
MAX_ROUNDS = 4


def main(argv):
    ncase = int(argv[argv.index("--cases") + 1]) if "--cases" in argv else 600
    gridkit.quiet()
    print("loading network")
    n = gridkit.load("WP2033", "all-island")
    br, K, b0, phi, P0, rank = ptdf_all.build(n)
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)

    buses, edges = n.buses.index, br.index
    E, NB = len(br), len(buses)
    s = br["s_nom"].to_numpy(float)
    P = ptdf_all.injections(n, n.snapshots).reindex(buses).fillna(0.0).to_numpy()
    pos = {v: i for i, v in enumerate(buses)}
    ia = np.array([pos[x] for x in br["bus0"]])
    iz = np.array([pos[x] for x in br["bus1"]])

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

    base_comp = connected_components(
        csr_matrix((np.ones(E), (ia, iz)), shape=(NB, NB)), directed=False)[0]

    def split(dead):
        m = np.ones(E, bool)
        m[list(dead)] = False
        c = connected_components(
            csr_matrix((np.ones(int(m.sum())), (ia[m], iz[m])), shape=(NB, NB)),
            directed=False)[0]
        return c > base_comp

    def flows(dead, x, h):
        """Exact DC flows with `dead` out of service and `x` MW of wind cut."""
        bk = b0.copy()
        bk[list(dead)] = 0.0
        Pk = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        pr = P[:, h].copy()
        if x is not None and x.sum() > 0:
            np.add.at(pr, wcol, -x)
            pr = pr + x.sum() * wt
        f = Pk @ (pr + (K @ (bk * phi))) - bk * phi
        V = -(Pk[:, wcol] - (Pk @ wt)[:, None])
        return f, V

    def run(k, h, arm):
        """Follow the chain from one starting trip."""
        dead = {k}
        x = np.zeros(len(wnames))
        avail0 = outp.iloc[h].reindex(wnames).fillna(0.0).to_numpy()
        rounds, islanded, runaway = 0, False, False
        for _ in range(MAX_ROUNDS):
            if len(dead) > 12:
                runaway = True
                break
            if split(dead):
                islanded = True
                break
            f, V = flows(dead, x, h)
            live = np.ones(E, bool)
            live[list(dead)] = False
            over = list(np.nonzero((np.abs(f) > s * TOL + protocol.MARGIN) & live)[0])
            if not over:
                break
            rounds += 1
            avail = np.maximum(avail0 - x, 0.0)
            if arm == "nothing":
                cur = f
            elif arm == "eir":
                cur, u = protocol.arm_eirgrid(V, f, over, avail, s, live)
                x = x + u
            elif arm == "veto":
                cur, u = protocol.arm_greedy(V, f, over, avail, s, live, veto=True)
                x = x + u
            else:
                ok, u, _ = protocol.arm_lp(V, f, over, avail, s, live)
                cur, x = f + V @ u, x + u
            still = list(np.nonzero((np.abs(cur) > s * TOL + protocol.MARGIN) & live)[0])
            if not still:
                break
            dead |= set(still)
        return dict(lost=len(dead) - 1, rounds=rounds, cut=round(float(x.sum()), 1),
                    islanded=int(islanded), runaway=int(runaway))

    # ---- pick the starting trips -------------------------------------
    src = os.path.join(OUT, "cases.csv")
    if not os.path.exists(src):
        raise SystemExit("run protocol.py first - cascade starts from its cases")
    d = pd.read_csv(src)
    d = d[~d.islanding]
    # every event where a remedy was seen to break something, then the most
    # severe of the rest, then a random spread so the sample is not all tail
    hot = d[(d.eir_new_overloads > 0) | (d.ovl_new_overloads > 0)]
    if len(hot) > ncase // 2:
        hot = hot.sample(ncase // 2, random_state=7)
    rest = d.drop(hot.index)
    sev = rest.nlargest(max(0, (ncase - len(hot)) // 2), "worst_pct")
    pool = rest.drop(sev.index)
    rnd = pool.sample(min(len(pool), max(0, ncase - len(hot) - len(sev))),
                      random_state=7)
    pick = pd.concat([hot, sev, rnd])
    print("%d starting trips: %d where a remedy broke something, %d most "
          "severe, %d random" % (len(pick), len(hot), len(sev), len(rnd)))

    idx = {v: i for i, v in enumerate(edges)}
    rows = []
    t0 = time.perf_counter()
    for j, (_, r) in enumerate(pick.iterrows()):
        k, h = idx[r.outage], int(r.hour)
        row = dict(outage=r.outage, hour=h, n_over=r.n_over,
                   worst_pct=r.worst_pct)
        for arm in ("nothing", "eir", "veto", "lp"):
            for key, val in run(k, h, arm).items():
                row[arm + "_" + key] = val
        rows.append(row)
        if (j + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print("  %4d/%d  %5.1f min  ~%5.1f min left"
                  % (j + 1, len(pick), el / 60,
                     el / (j + 1) * (len(pick) - j - 1) / 60))

    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(OUT, "cascade.csv"), index=False)
    print("\ndone in %.1f min, %d starting trips\n" % (
        (time.perf_counter() - t0) / 60, len(c)))

    hdr = "%-9s%12s%10s%12s%11s%10s%10s" % (
        "", "lines lost", "worst", "cascaded", "islanded", "runaway", "MW cut")
    print(hdr)
    print("-" * len(hdr))
    for arm, name in (("nothing", "NOTHING"), ("eir", "EirGrid"),
                      ("veto", "+Veto"), ("lp", "LP")):
        print("%-9s%12d%10d%9d/%-4d%11d%10d%10.0f"
              % (name, int(c[arm + "_lost"].sum()), int(c[arm + "_lost"].max()),
                 int((c[arm + "_lost"] > 0).sum()), len(c),
                 int(c[arm + "_islanded"].sum()), int(c[arm + "_runaway"].sum()),
                 c[arm + "_cut"].sum()))
    print("\n-> %s" % os.path.relpath(os.path.join(OUT, "cascade.csv"), HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
