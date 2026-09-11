"""Find the N-1 outage that makes the clearest cascade demonstration.

    python cascade_search.py                  sweep and rank
    python cascade_search.py --cases 1500     widen the sweep

cascade_replay.py exports one case in detail.  This decides *which* case, by
running the chain for every candidate and ranking the results, so the example
on the slide is the best one in the network rather than the first one found.

WHAT IT IS LOOKING FOR
----------------------
A demonstration is only worth showing if all of these hold at once:

    the outage is a LINE, and so is everything it breaks   - a transformer
        outage is just as real but nobody reads it off a map at a glance
    ONE circuit overloads to begin with                    - so the chain that
        follows is unambiguously the remedy's doing, not the fault's
    both circuits are big                                  - a 110 kV spur
        going over teaches nothing; a 400 MVA corridor does
    today's remedy makes it worse, repeatedly              - the more rounds
        the shift-factor group needs, the clearer the blindness is
    the veto ends it in round one, losing nothing

SPEED
-----
Each round needs the PTDF of a network with one more branch out.  Rebuilding
the pseudoinverse costs ~310 ms; doing it by rank-one update from the PTDF
already in hand costs ~1 ms, because taking branch j out of a network whose
PTDF is P needs only column j of P K:

    c    = P[:, bus0(j)] - P[:, bus1(j)]
    lodf = c / (1 - c[j]),  lodf[j] = -1
    P'   = P + outer(lodf, P[j, :]),  then row j zeroed

Applied successively that is exact for any number of outages, and it is what
makes sweeping thousands of chains possible.  The identity is checked against a
full rebuild on a sample at startup and the check is printed.

Writes cascade_ranked.csv into
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
import protocol         # noqa: E402

TOL = protocol.TOL

#: A circuit only counts as over, and only trips, if it is over by more than
#: this many MW.  Without it the test sits exactly on the constraint the
#: remedy solves to, and a difference of 1e-12 in the flow decides whether a
#: circuit trips - which then decides the whole chain.  A kilowatt is far
#: below anything a thermal rating means.
MARGIN = protocol.MARGIN
MAX_ROUNDS = 20


def main(argv):
    ncand = int(argv[argv.index("--cases") + 1]) if "--cases" in argv else 1500
    gridkit.quiet()
    print("loading network")
    n = gridkit.load("WP2033", "all-island")
    br, K, b0, phi, P0, rank = ptdf_all.build(n)
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)

    buses, edges = n.buses.index, br.index
    E, NB = len(br), len(buses)
    s = br["s_nom"].to_numpy(float)
    kind = br["kind"].to_numpy()
    Pinj = ptdf_all.injections(n, n.snapshots).reindex(buses).fillna(0.0).to_numpy()
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

    def drop(P, j):
        """PTDF with branch j additionally out, by exact rank-one update."""
        c = P[:, ia[j]] - P[:, iz[j]]
        den = 1.0 - c[j]
        if abs(den) < 1e-3:
            return None                       # the outage islands - bail out
        lod = c / den
        lod[j] = -1.0
        Q = P + np.outer(lod, P[j, :])
        Q[j, :] = 0.0
        return Q

    # the identity, checked rather than asserted
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(8):
        j = int(rng.integers(E))
        Q = drop(P0, j)
        if Q is None:
            continue
        bk = b0.copy()
        bk[j] = 0.0
        R = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        worst = max(worst, float(np.abs(Q - R).max()))
    print("rank-one update vs full rebuild: max %.1e\n" % worst)

    def flows(P, dead, x, h):
        bk = b0.copy()
        bk[list(dead)] = 0.0
        pr = Pinj[:, h].copy()
        if x.sum() > 0:
            np.add.at(pr, wcol, -x)
            pr = pr + x.sum() * wt
        f = P @ (pr + (K @ (bk * phi))) - bk * phi
        V = -(P[:, wcol] - (P @ wt)[:, None])
        return f, V, pr

    def chain(k, h, arm):
        P = drop(P0, k)
        if P is None:
            return None
        dead = [k]
        x = np.zeros(len(wnames))
        avail0 = outp.iloc[h].reindex(wnames).fillna(0.0).to_numpy()
        rounds = 0
        seq = []
        for _ in range(MAX_ROUNDS):
            f, V, pr = flows(P, dead, x, h)
            live = np.ones(E, bool)
            live[dead] = False
            load = np.abs(f) / s
            over = list(np.nonzero((np.abs(f) > s * TOL + MARGIN) & live)[0])
            if not over:
                break
            rounds += 1
            avail = np.maximum(avail0 - x, 0.0)
            if arm == "eir":
                cur, u = protocol.arm_eirgrid(V, f, over, avail, s, live)
            elif arm == "veto":
                cur, u = protocol.arm_greedy(V, f, over, avail, s, live, veto=True)
            else:
                cur, u = f, np.zeros(len(wnames))
            x = x + u
            hot = np.nonzero((np.abs(cur) > s * TOL + MARGIN) & live)[0]
            if not len(hot):
                break
            j = int(hot[np.argmax((np.abs(cur) / s)[hot])])
            seq.append((edges[j], kind[j], float(s[j]), j not in over))
            dead.append(j)
            P = drop(P, j)
            if P is None:
                break
        m = np.ones(E, bool)
        m[dead] = False
        nc = connected_components(
            csr_matrix((np.ones(int(m.sum())), (ia[m], iz[m])), shape=(NB, NB)),
            directed=False)[0]
        return dict(lost=len(dead) - 1, rounds=rounds, cut=float(x.sum()),
                    islands=int(nc - base_comp), seq=seq)

    # ---- candidates --------------------------------------------------
    d = pd.read_csv(os.path.join(OUT, "cases.csv"))
    smap = dict(zip(edges, s))
    d["out_s"] = d.outage.map(smap)
    d["hit_s"] = d.worst_branch.map(smap)
    # The only conditions that are really required are that the starting
    # outage is a line and that the veto arm ends it cleanly - everything else
    # (how long today's remedy takes, what it breaks, how big the circuits
    # are) is what the sweep is for, so it is not filtered on in advance.
    c = d[(d.kind == "Line") & (d.worst_kind == "Line") &
          (d.veto_cleared == 1) & (d.veto_new_overloads == 0)].copy()
    if "--narrow" in argv:
        c = c[(c.n_over <= 2) & (c.eir_new_overloads > 0)]
    c["size"] = np.minimum(c.out_s, c.hit_s)
    c = c.nlargest(min(ncand, len(c)), "size")
    print("%d candidates (line outage, line overloads, veto ends it clean)\n"
          % len(c))

    idx = {v: i for i, v in enumerate(edges)}
    rows = []
    t0 = time.perf_counter()
    for j, (_, r) in enumerate(c.iterrows()):
        k, h = idx[r.outage], int(r.hour)
        e = chain(k, h, "eir")
        v = chain(k, h, "veto")
        if e is None or v is None:
            continue
        nth = chain(k, h, "nothing")
        caused = sum(1 for t in e["seq"] if t[3])
        rows.append(dict(
            outage=r.outage, hour=h, out_s=r.out_s, hit=r.worst_branch,
            hit_s=r.hit_s, worst_pct=r.worst_pct, n_over=r.n_over,
            eir_lost=e["lost"], eir_rounds=e["rounds"],
            eir_caused=caused, eir_islands=e["islands"], eir_cut=round(e["cut"], 1),
            eir_all_lines=int(all(t[1] == "Line" for t in e["seq"])),
            eir_min_s=round(min([t[2] for t in e["seq"]] or [0]), 0),
            veto_lost=v["lost"], veto_cut=round(v["cut"], 1),
            veto_rounds=v["rounds"],
            nothing_lost=nth["lost"], nothing_islands=nth["islands"],
            size=float(min(r.out_s, r.hit_s)),
            chain=" > ".join(t[0] for t in e["seq"][:6])))
        if (j + 1) % 300 == 0:
            print("  %4d/%d  %.1f min" % (j + 1, len(c), (time.perf_counter() - t0) / 60))

    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "cascade_ranked.csv"), index=False)
    print("\nswept %d chains in %.1f min\n" % (len(R), (time.perf_counter() - t0) / 60))

    good = R[(R.veto_lost == 0) & (R.eir_lost > 0) & (R.eir_all_lines == 1)]
    print("%d where the veto loses nothing and today's remedy loses only lines"
          % len(good))
    show = ["outage", "out_s", "hit", "hit_s", "hour", "n_over", "worst_pct",
            "eir_lost", "eir_rounds", "eir_caused", "eir_islands", "eir_cut",
            "veto_cut", "nothing_lost", "nothing_islands", "eir_min_s"]
    print("\nRANKED - longest chain under today's remedy:")
    print(good.sort_values(["eir_lost", "size"], ascending=False)
          .head(18)[show].to_string(index=False))
    print("\nRANKED - biggest circuits, among chains of 2+ lost:")
    big = good[good.eir_lost >= 2]
    if len(big):
        print(big.nlargest(14, "size")[show].to_string(index=False))
    print("\nRANKED - most damage today's remedy CAUSES (not merely fails to stop):")
    print(good.sort_values(["eir_caused", "size"], ascending=False)
          .head(12)[show].to_string(index=False))
    print("\n-> %s" % os.path.relpath(os.path.join(OUT, "cascade_ranked.csv"), HERE))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
