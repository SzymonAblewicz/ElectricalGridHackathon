"""The cascade-stopping protocol, tested on every N-1 outage at every hour.

    python protocol.py                 all 980 outages x 168 hours
    python protocol.py --limit 40      first 40 outages, for a check
    python protocol.py --hours 8       8 hours spread across the week
    python protocol.py --slack uniform robustness re-run

WHAT IS BEING TESTED
--------------------
A branch trips.  Post-trip flows are recomputed exactly and compared to the
ratings.  Whatever is over its rating is the next thing to fail, so a remedy is
applied: dispatch wind down until it is back under.  The question is whether
the remedy itself breaks something else.

Five ways of choosing the remedy are run on identical inputs:

    NOTHING   no remedy.  The overload stands.  Context only.

    EIRGRID   one group per congested circuit, membership by shift factor
              >= 0.05 on that circuit, split pro-rata by output, circuits
              handled worst-first.  No check on what the cut does anywhere
              else.  This is the published Wind Dispatch Tool arithmetic
              ([WDT] App. 2 p.72 for membership, App. 1 p.66 for the split)
              and it is the baseline, not a straw man.

    OVERLAP   one group across all congested circuits at once, nodes taken in
              order of *net* relief - the sum of their sensitivities over every
              circuit that is currently over.  Still no safety check.  This arm
              exists to separate the two halves of the idea.

    VETO      OVERLAP plus a hard cap: a node may only be cut as far as leaves
              every other branch in the network under its rating, and leaves
              every already-overloaded branch no worse than it was.

    LP        the same safety rule stated exactly and handed to a solver:
              minimise total MW curtailed subject to every live branch ending
              at or under its rating.  Infeasible means wind curtailment
              genuinely cannot fix this outage without breaking something -
              a proof, not a greedy failure.

WHAT A MEGAWATT OF CURTAILMENT DOES
-----------------------------------
Cutting x MW at wind node n drops the injection there by x.  A megawatt cannot
vanish, so x MW appears elsewhere; *where* is part of the definition of a shift
factor.  The default here is the system-operator convention the kit documents -
spread across every load in proportion to its size:

    V[:, n] = -( PTDF[:, n] - PTDF @ w ),      w = load share per bus

so the flow vector after cutting x is  f + V x, exactly.  --slack uniform
re-runs everything against the pseudoinverse's own reference instead.

SCOPE
-----
Outages that split the network into islands are screened out of the main
result and reported separately.  An island is a generation-balance problem -
the DC pseudoinverse smears the imbalance across the island and the resulting
"flows" are not a dispatch anyone would run.  Curtailing wind is not the remedy
for it and pretending otherwise would inflate every number here.

Writes to participant-kit new/results/N-1 contingencies/protocol/.
"""

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
OUT = os.path.join(KIT, "results", "N-1 contingencies", "protocol")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402

#: Loading above this counts as over.  The 0.1% margin is not cosmetic: the
#: dispatch optimiser parks binding branches at exactly 1.000 and the
#: arithmetic lands a few parts in 1e-10 either side, so a bare "> 1.0"
#: reports phantom violations in the intact network.
TOL = 1.001

#: Shift factor at or above which a wind node joins an EirGrid-style group.
THRESHOLD = 0.05

#: Cuts below this are noise, not instructions.
EPS = 1e-9

#: A branch is only "pushed over" if it is over by more than this many MW.
#: Guards the exact constraint boundary; see score().
MARGIN = 1e-3


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #

def islanding_mask(br, buses):
    """True for each branch whose removal disconnects part of the network.

    Structural, from the graph alone - no pseudoinverse, no tolerance to pick.
    """
    pos = {v: i for i, v in enumerate(buses)}
    a = np.array([pos[x] for x in br["bus0"]])
    z = np.array([pos[x] for x in br["bus1"]])
    nb, ne = len(buses), len(br)
    base = connected_components(
        csr_matrix((np.ones(ne), (a, z)), shape=(nb, nb)), directed=False)[0]
    out = np.zeros(ne, bool)
    for k in range(ne):
        m = np.ones(ne, bool)
        m[k] = False
        c = connected_components(
            csr_matrix((np.ones(int(m.sum())), (a[m], z[m])), shape=(nb, nb)),
            directed=False)[0]
        out[k] = c > base
    return out


# --------------------------------------------------------------------------- #
# the remedies
# --------------------------------------------------------------------------- #

def relief(V, f):
    """Rel[e, n]: MW that |f_e| falls by, per MW curtailed at node n.

    Positive helps, negative hurts.  V[e, n] is the signed change in f_e;
    pointing it along the existing flow direction is what turns a PTDF into
    the quantity a dispatch decision is actually made on.
    """
    sg = np.sign(f)
    sg[sg == 0] = 1.0
    return -V * sg[:, None]


def cap(V, f, lim, live, n):
    """Largest cut at node n that leaves every live branch inside lim.

    Solved on the signed flow, not on |f|, so a flow that passes through zero
    and grows the other way is still caught.
    """
    v = V[:, n]
    with np.errstate(divide="ignore", invalid="ignore"):
        up = np.where(v > 1e-12, (lim - f) / v, np.inf)
        dn = np.where(v < -1e-12, (-lim - f) / v, np.inf)
    r = np.minimum(up, dn)
    r[~live] = np.inf
    return max(0.0, float(np.nanmin(r)))


def arm_eirgrid(V, f, over, avail, s, live):
    """Per-circuit group, membership by threshold, split pro-rata by output."""
    cur, left = f.copy(), avail.copy()
    used = np.zeros(len(avail))
    for e in sorted(over, key=lambda x: -(abs(f[x]) - s[x])):
        for _ in range(200):
            need = abs(cur[e]) - s[e] * TOL
            if need <= 1e-6:
                break
            sg = np.sign(cur[e]) or 1.0
            rel = -V[e] * sg
            grp = np.nonzero((rel >= THRESHOLD) & (left > EPS))[0]
            if not len(grp):
                break
            tot = left[grp].sum()
            eff = float((rel[grp] * left[grp]).sum() / tot)
            if eff <= 0:
                break
            take = min(need / eff, tot) * left[grp] / tot
            cur = cur + V[:, grp] @ take
            left[grp] -= take
            used[grp] += take
    return cur, used


def arm_greedy(V, f, over, avail, s, live, veto, net_rank=True):
    """One node at a time, worst circuit first; veto adds the hard cap.

    ``net_rank`` is the whole of the overlap idea and the only thing that
    separates this arm from the next one.  With it, a node is chosen on the sum
    of its relief over *every* circuit currently over its rating - so a node
    taking 0.09 off three circuits outranks one taking 0.20 off the worst.
    Without it, the node is chosen on its relief on the worst circuit alone,
    which is how a per-circuit constraint group is drawn.

    Everything else - the step, the stopping rule, the veto - is identical, so
    the difference between the two runs is the ranking and nothing else.
    """
    cur, left = f.copy(), avail.copy()
    used = np.zeros(len(avail))
    lim = s * TOL
    for _ in range(2000):
        viol = [e for e in over if abs(cur[e]) > s[e] * TOL]
        if not viol:
            break
        rel = relief(V, cur)
        e0 = max(viol, key=lambda e: abs(cur[e]) - s[e])
        net = rel[viol].sum(axis=0) if net_rank else rel[e0]
        net = np.where(left > EPS, net, -np.inf)
        i = int(np.argmax(net))
        if not np.isfinite(net[i]) or net[i] <= THRESHOLD:
            break
        room = left[i]
        if veto:
            # a branch already over may not get worse; every other branch has
            # its rating.  Both are the same statement about lim.
            lm = np.where(np.abs(cur) > lim, np.abs(cur), lim)
            room = min(room, cap(V, cur, lm, live, i))
        if room <= EPS:
            left[i] = 0.0
            continue
        r0 = rel[e0, i]
        want = (abs(cur[e0]) - s[e0] * TOL) / r0 if r0 > 1e-9 else room
        step = float(min(room, max(want, 1e-3)))
        cur = cur + step * V[:, i]
        left[i] -= step
        used[i] += step
    return cur, used


def arm_lp(V, f, over, avail, s, live):
    """Minimum MW of curtailment that ends with every live branch in limit.

    Two stages, both exact LPs:

    1. can it be done at all?  minimise total MW subject to
       -lim <= f + Vx <= lim on every live branch.  Feasible means a safe
       remedy exists and the objective is the cheapest one there is - no
       ordering heuristic can beat it and none can be blamed for a failure.
    2. if not, how close can it get?  minimise the residual overload on the
       congested circuits while still not breaking anything healthy.  What
       comes back is the part of the problem wind cannot reach.
    """
    W = len(avail)
    lv = np.nonzero(live)[0]
    # Drop branches that cannot reach their rating even if every wind node in
    # the network went to zero: |f| + sum_n |V[k,n]| * avail_n < rating.  That
    # is an exact bound on the constraint, not a screen, so the LP that comes
    # back is the same LP - there are just fewer rows of it.
    reach = np.abs(V[lv]) @ avail
    keep = (np.abs(f[lv]) + reach) > s[lv] * TOL - 1e-9
    keep[np.isin(lv, over)] = True
    lv = lv[keep]
    A = V[lv]
    lim = s[lv] * TOL
    bounds = [(0.0, float(a)) for a in avail]
    ub = np.concatenate([lim - f[lv], lim + f[lv]])
    Aub = np.vstack([A, -A])
    r1 = linprog(np.ones(W), A_ub=Aub, b_ub=ub, bounds=bounds, method="highs")
    if r1.status == 0:
        return True, r1.x, 0.0

    # stage 2 - healthy branches stay hard, congested ones get a priced slack
    hot = np.isin(lv, over)
    hi = np.nonzero(hot)[0]
    nz = len(hi)
    Z = np.zeros((2 * len(lv), nz))
    Z[hi, np.arange(nz)] = -1.0
    Z[len(lv) + hi, np.arange(nz)] = -1.0
    A2 = np.hstack([Aub, Z])
    c2 = np.concatenate([np.zeros(W), np.ones(nz)])
    r2 = linprog(c2, A_ub=A2, b_ub=ub,
                 bounds=bounds + [(0.0, None)] * nz, method="highs")
    if r2.status != 0:
        return False, np.zeros(W), np.nan
    return False, r2.x[:W], float(r2.x[W:].sum())


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def score(cur, f, over, s, live, used, tag):
    """What the remedy achieved, and what it cost elsewhere.

    The MARGIN matters and is not cosmetic.  The LP's whole job is to push
    branches up to their rating and no further, so its answer sits exactly on
    the constraint and lands a few parts in 1e-12 either side of it.  Without a
    margin that reads as "the LP broke 29 circuits", every one of them by less
    than 0.12 MW.  It did not; it landed on the line, which is what it was
    asked to do.  A tenth of a megawatt is below anything a rating means.
    """
    lim = s * TOL
    was_hot = np.abs(f) > lim + MARGIN
    now_hot = np.abs(cur) > lim + MARGIN
    new = np.nonzero(now_hot & ~was_hot & live)[0]
    worse = np.nonzero(was_hot & live & (np.abs(cur) > np.abs(f) + MARGIN))[0]
    left_over = [e for e in over if abs(cur[e]) > s[e] * TOL + MARGIN]
    resid = float(sum(abs(cur[e]) - s[e] for e in left_over))
    ld = np.abs(cur) / s
    out = {
        tag + "_cleared": int(not left_over),
        tag + "_mw": round(float(used.sum()), 3),
        tag + "_nodes": int((used > 1e-6).sum()),
        tag + "_new_overloads": int(len(new)),
        # counting circuits says how often; this says how badly, and it is the
        # number that decides whether the collateral matters operationally
        tag + "_new_MW": round(float((np.abs(cur[new]) - s[new]).sum()), 3),
        tag + "_worsened": int(len(worse)),
        tag + "_resid_MW": round(resid, 3),
        tag + "_worst_new_pct": round(100.0 * float(ld[new].max()), 2) if len(new) else 0.0,
        tag + "_max_load_pct": round(100.0 * float(ld[live].max()), 2),
    }
    return out, new


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv):
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    nhours = int(argv[argv.index("--hours") + 1]) if "--hours" in argv else None
    slack = argv[argv.index("--slack") + 1] if "--slack" in argv else "load"
    sfx = "" if slack == "load" else "_" + slack
    if "--threshold" in argv:
        # the group-membership cut-off is a choice, not a law.  Re-running the
        # baseline at other values is how you tell "the threshold is wrong"
        # apart from "nothing checks the rest of the network".
        global THRESHOLD
        THRESHOLD = float(argv[argv.index("--threshold") + 1])
        sfx += "_t%s" % str(THRESHOLD).replace(".", "")
    os.makedirs(OUT, exist_ok=True)
    gridkit.quiet()

    print("loading network  (slack reference: %s)" % slack)
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

    # the balancing megawatt
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
    isl = islanding_mask(br, buses)
    todo = range(E if limit is None else min(limit, E))
    print("%d buses, %d branches, %d wind nodes, %d hours" % (NB, E, W, len(hours)))
    print("%d outages island the network - screened out of the main result, "
          "reported separately\n" % int(isl.sum()))

    # Post-outage PTDF by exact rank-one update.  Zeroing b[k] removes one
    # rank-one term from the Laplacian, so the new PTDF is the old one plus an
    # outer product - no second pseudoinverse.  M[e, k] is the flow on e per
    # unit transfer across branch k's own terminals; 1 - M[k, k] is the
    # denominator, and it goes to zero exactly when the outage islands.  Those
    # are done the slow, safe way instead.  Checked against the pseudoinverse
    # on a sample every run; the check is printed, not assumed.
    M = P0 @ K
    den = 1.0 - np.diag(M)
    fastok = (~isl) & (np.abs(den) > 1e-3)
    print("exact rank-one update usable on %d of %d outages; the other %d "
          "rebuild the pseudoinverse" % (int(fastok.sum()), E, int((~fastok).sum())))

    def post(k):
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
        return P, F

    def slow(k):
        bk = b0.copy()
        bk[k] = 0.0
        P = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        return P, P @ (p + (K @ (bk * phi))[:, None]) - (bk * phi)[:, None]

    chk = np.random.default_rng(0).choice(np.nonzero(fastok)[0],
                                          min(20, int(fastok.sum())), replace=False)
    worst = max(np.abs(post(int(k))[1] - slow(int(k))[1]).max() for k in chk)
    print("rank-one vs pseudoinverse on %d sampled outages: max %.1e MW"
          % (len(chk), worst))
    print()

    rows, isl_rows, broke = [], [], []
    t0 = time.perf_counter()
    for k in todo:
        P, F = post(k)
        V = -P[:, wcol] if wt is None else -(P[:, wcol] - (P @ wt)[:, None])
        live = np.ones(E, bool)
        live[k] = False

        for h in hours:
            f = F[:, h]
            over = list(np.nonzero((np.abs(f) > s * TOL) & live)[0])
            if not over:
                continue
            avail = outp.iloc[h].reindex(wnames).fillna(0.0).to_numpy()
            e_worst = max(over, key=lambda e: abs(f[e]) / s[e])
            base = dict(outage=edges[k], hour=int(h), islanding=bool(isl[k]),
                        kind=br["kind"].iloc[k], n_over=len(over),
                        worst_branch=edges[e_worst],
                        worst_kind=br["kind"].iloc[e_worst],
                        over_MW=round(float(sum(abs(f[e]) - s[e] for e in over)), 3),
                        worst_pct=round(100.0 * float((np.abs(f) / s)[live].max()), 2),
                        wind_avail_MW=round(float(avail.sum()), 1))

            cE, uE = arm_eirgrid(V, f, over, avail, s, live)
            cO, uO = arm_greedy(V, f, over, avail, s, live, veto=False)
            cV, uV = arm_greedy(V, f, over, avail, s, live, veto=True)
            # identical to cV except the ranking, so cV - cS is the overlap
            cS, uS = arm_greedy(V, f, over, avail, s, live, veto=True,
                                net_rank=False)
            okL, uL, residL = arm_lp(V, f, over, avail, s, live)
            cL = f + V @ uL

            for c, u, t in ((cE, uE, "eir"), (cO, uO, "ovl"),
                            (cS, uS, "solo"), (cV, uV, "veto"), (cL, uL, "lp")):
                sc, new = score(c, f, over, s, live, u, t)
                base.update(sc)
                for e in new:
                    broke.append(dict(arm=t, outage=edges[k], hour=int(h),
                                      broken=edges[e], kind=br["kind"].iloc[e],
                                      s_nom=s[e],
                                      loading_pct=round(100 * abs(c[e]) / s[e], 2),
                                      over_MW=round(abs(c[e]) - s[e], 2)))
            base["lp_feasible"] = int(okL)
            base["lp_resid_MW"] = round(float(residL), 3) if np.isfinite(residL) else np.nan
            (isl_rows if isl[k] else rows).append(base)

        if (k + 1) % 25 == 0:
            el = time.perf_counter() - t0
            print("  %4d/%d  %5.1f min  ~%5.1f min left  (%d cases)"
                  % (k + 1, len(todo), el / 60,
                     el / (k + 1) * (len(todo) - k - 1) / 60, len(rows)))

    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "cases%s.csv" % sfx), index=False)
    if isl_rows:
        pd.DataFrame(isl_rows).to_csv(
            os.path.join(OUT, "cases_islanding%s.csv" % sfx), index=False)
    if broke:
        pd.DataFrame(broke).to_csv(
            os.path.join(OUT, "collateral%s.csv" % sfx), index=False)

    print("\ndone in %.1f min" % ((time.perf_counter() - t0) / 60))
    print("cases (non-islanding): %d   islanding: %d\n" % (len(d), len(isl_rows)))
    if len(d):
        report(d)
    return 0


def report(d):
    """The headline table, printed the way it should be quoted."""
    N = len(d)
    hdr = "%-9s%9s%9s%9s%10s%15s%7s" % (
        "", "cleared", "MW cut", "new ovl", "worsened", "events w/ new", "nodes")
    print(hdr)
    print("-" * len(hdr))
    for t, name in (("eir", "EirGrid"), ("ovl", "Overlap"),
                    ("solo", "Solo+veto"), ("veto", "+Veto"), ("lp", "LP")):
        print("%-9s%8.1f%%%9.1f%9d%10d%10d/%-4d%7.0f"
              % (name, 100 * d[t + "_cleared"].mean(), d[t + "_mw"].median(),
                 int(d[t + "_new_overloads"].sum()),
                 int(d[t + "_worsened"].sum()),
                 int((d[t + "_new_overloads"] > 0).sum()), N,
                 d[t + "_nodes"].median()))
    print()
    both = d[(d.eir_cleared == 1) & (d.veto_cleared == 1)]
    if len(both):
        print("on the %d events both EirGrid and +Veto clear:" % len(both))
        print("   EirGrid median %.1f MW, +Veto %.1f MW, LP optimum %.1f MW"
              % (both.eir_mw.median(), both.veto_mw.median(), both.lp_mw.median()))
    print("\nLP says a safe remedy exists in %.1f%% of events (%d genuinely "
          "impossible)" % (100 * d.lp_feasible.mean(),
                           int((d.lp_feasible == 0).sum())))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
