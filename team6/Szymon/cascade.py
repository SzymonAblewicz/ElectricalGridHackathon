"""Cascade simulation: does one bad remedy start a chain of failures?

    python cascade.py

A line trips.  Something else overloads.  A remedy is applied.  Anything still
over its rating then trips too, and the whole thing repeats.  That is a cascade.

Two remedies are run from the same starting trip, at the same hour, on the same
network:

    EIRGRID   per-line group by shift-factor threshold, split pro-rata by
              output, no check on what the cut does elsewhere.
              [WDT] Appendix 2 p.72 for membership, Appendix 1 p.66 for the split.

    OURS      one group across all congested lines, ordered by net sensitivity,
              each node capped so the running total can never push any other
              branch past its rating.

Each round is the same pipeline with more entries of ``b`` set to zero.  No new
maths, no approximation.

Writes ``cascade.csv`` into ``participant-kit new/results/N-1 contingencies/``.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
OUT = os.path.join(KIT, "results", "N-1 contingencies")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402

T = 0.05
OVER = 1.001
MAX_ROUNDS = 8
TRIP = 1.001


def main():
    gridkit.quiet()
    n = gridkit.load("WP2033", "all-island")
    br, K, b0, phi, _, _ = ptdf_all.build(n)
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)

    buses = n.buses.index
    s = br["s_nom"].to_numpy(float)
    p = ptdf_all.injections(n, n.snapshots).reindex(buses).fillna(0.0).to_numpy()
    g = n.generators
    w = g.index[g["carrier"] == "wind"]
    outp = n.generators_t.p[w].T.groupby(g.loc[w, "bus"].to_numpy()).sum().T
    wb = list(outp.columns)
    wc = np.array([buses.get_loc(x) for x in wb])
    E = len(br)

    def solve(dead):
        """Flows and PTDF with every branch in `dead` out of service."""
        bk = b0.copy()
        bk[list(dead)] = 0.0
        P = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        F = P @ (p + (K @ (bk * phi))[:, None]) - (bk * phi)[:, None]
        return P, F

    def headroom(v, cur, live, viol):
        """How far this cut can go before it breaks something.

        A branch that is *already* over its rating is one we are trying to
        relieve, not protect - the only requirement there is that we do not
        make it worse.  A branch under its rating must stay under it.
        """
        # A branch already past its limit only has to not get worse; every
        # other branch may go up to its rating.  The same 0.1% tolerance used
        # everywhere else matters here: the optimiser parks branches at
        # exactly 100.0%, and demanding strictly-under leaves zero room and
        # blocks the whole remedy.
        lim = np.maximum(np.abs(cur), s * OVER)
        with np.errstate(divide="ignore", invalid="ignore"):
            up = np.where(v > 1e-12, (lim - cur) / v, np.inf)
            dn = np.where(v < -1e-12, (-lim - cur) / v, np.inf)
        r = np.minimum(up, dn)
        r[~live] = np.inf
        r[list(viol)] = np.inf          # the branches we are fixing
        return max(0.0, float(np.min(r)))

    def eirgrid(P, f, over, avail, live):
        """Per-line group, pro-rata by output, no cross-line check."""
        cur, left, tot = f.copy(), avail.copy(), 0.0
        V = -P[:, wc]
        for e in sorted(over, key=lambda x: -(abs(f[x]) - s[x])):
            for _ in range(60):
                need = abs(cur[e]) - s[e]
                if need <= 1e-6:
                    break
                sg = np.sign(cur[e]) or 1.0
                Se = (P[:, wc] * sg)[e]
                grp = np.nonzero((Se >= T) & (left > 1e-9))[0]
                if len(grp) == 0:
                    break
                eff = float((Se[grp] * left[grp]).sum() / left[grp].sum())
                if eff <= 0:
                    break
                take = min(need / eff, left[grp].sum()) * left[grp] / left[grp].sum()
                cur = cur + V[:, grp] @ take
                left[grp] -= take
                tot += take.sum()
        return cur, tot

    def ours(P, f, over, avail, live):
        """Net sensitivity across all congested lines, capped by live headroom."""
        cur, left, tot = f.copy(), avail.copy(), 0.0
        V = -P[:, wc]
        for _ in range(800):
            viol = [e for e in over if abs(cur[e]) > s[e] * OVER]
            if not viol:
                break
            sg = np.sign(cur)
            sg[sg == 0] = 1.0
            Sn = (P[:, wc] * sg[:, None])[viol]
            sc = Sn.sum(axis=0)
            i = int(np.argmax(np.where(left > 1e-9, sc, -9)))
            if sc[i] <= T:
                break
            e0 = viol[int(np.argmax([abs(cur[e]) - s[e] for e in viol]))]
            cap = min(left[i], headroom(V[:, i], cur, live, viol))
            if cap <= 1e-9:
                left[i] = 0.0
                continue
            step = min(cap, max((abs(cur[e0]) - s[e0])
                                / max(Sn[viol.index(e0)][i], 1e-9), 0.05))
            cur = cur + step * V[:, i]
            left[i] -= step
            tot += step
        return cur, tot

    def run(start, hour, remedy):
        """Follow the chain: overload -> remedy -> whatever is still over trips."""
        dead = {start}
        rounds, cut = 0, 0.0
        while rounds < MAX_ROUNDS:
            P, F = solve(dead)
            live = np.ones(E, bool)
            live[list(dead)] = False
            f = F[:, hour]
            over = list(np.nonzero((np.abs(f) / s > OVER) & live)[0])
            if not over:
                break
            rounds += 1
            avail = outp.iloc[hour].reindex(wb).fillna(0.0).to_numpy()
            cur, tot = remedy(P, f, over, avail, live)
            cut += tot
            still = [e for e in np.nonzero((np.abs(cur) / s > TRIP) & live)[0]]
            if not still:
                break
            dead |= set(still)                    # they trip
        return len(dead) - 1, rounds, cut

    rng = np.random.default_rng(7)
    rows = []
    for k in rng.choice(980, 90, replace=False):
        for hour in (56, 64, 101):
            P, F = solve({k})
            live = np.ones(E, bool)
            live[k] = False
            f = F[:, hour]
            if not ((np.abs(f) / s > OVER) & live).any():
                continue
            le, re, ce = run(k, hour, eirgrid)
            lo, ro, co = run(k, hour, ours)
            rows.append(dict(trip=br.index[k], hour=hour,
                             eir_lines_lost=le, eir_rounds=re, eir_cut=round(ce, 1),
                             our_lines_lost=lo, our_rounds=ro, our_cut=round(co, 1)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "cascade.csv"), index=False)

    print(f"starting trips simulated: {len(d)}\n")
    print(f"  EirGrid method : {int(d.eir_lines_lost.sum()):4d} lines lost, "
          f"max chain {int(d.eir_lines_lost.max())}, "
          f"{int((d.eir_lines_lost > 0).sum())} events cascaded")
    print(f"  ours           : {int(d.our_lines_lost.sum()):4d} lines lost, "
          f"max chain {int(d.our_lines_lost.max())}, "
          f"{int((d.our_lines_lost > 0).sum())} events cascaded")
    print()
    worst = d.nlargest(8, "eir_lines_lost")
    print("worst cascades under the EirGrid method:")
    print(worst[["trip", "hour", "eir_lines_lost", "eir_rounds",
                 "our_lines_lost", "our_rounds"]].to_string(index=False))
    print(f"\n-> {os.path.relpath(os.path.join(OUT, 'cascade.csv'), HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
