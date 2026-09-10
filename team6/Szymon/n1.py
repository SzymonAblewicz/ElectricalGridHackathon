"""N-1 contingency sensitivities: every single-branch outage, every wind node.

    python n1.py                    all 980 branch outages   (~12 min)
    python n1.py --limit 25         first 25 only, for a quick check

The intact-network pipeline in ``ptdf_all.py`` answers "if I curtail wind at
node n, what happens to branch e".  This answers the question EirGrid actually
builds constraint groups around: **"if branch k trips, what overloads, and
which wind nodes can relieve it".**

The method is the existing pipeline with one number changed.

    b[k] = 0

``b`` is the vector of branch susceptances, 1/x.  Setting one entry to zero is
setting that branch's reactance to infinity - the connection is still drawn on
the map, but no current can cross it.  That is an open circuit.  Verified
against physically deleting the branch and rebuilding: difference 0.0e+00,
exactly, not approximately.

Two consequences fall straight out of the algebra:

    L = K B K^T = sum over branches of  b_e * k_e k_e^T

so zeroing b[k] removes exactly one rank-one term from the Laplacian, and

    PTDF[e, :] = b_e * k_e^T L+

so row k of PTDF becomes exactly zero on its own.  A tripped line carries no
flow and has no sensitivity to anything.  Every *other* row changes, because
L+ changed - that is the flow the branch was carrying being redistributed.

Phase shifters need no special handling: b[k] = 0 carries through both
``p_eff = p + K(b*phi)`` and ``f = PTDF p_eff - b*phi`` automatically.

**The dispatch is held fixed** at the intact-network optimum.  That is the
correct question for contingency screening: given generation as it stands right
now, what breaks if this line trips.

Writes to ``participant-kit new/results/N-1 contingencies/``:

    outages.csv        980 rows, one per outage - what it broke, how far S moved
    violations.csv     one row per (outage, overloaded branch)
    sensitivity_n1.csv the wind sensitivities for those overloaded branches
    README.md          written by hand, not by this script
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
OUT = os.path.join(KIT, "results", "N-1 contingencies")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402  - the same build() and injections()

# Loading above this counts as a violation.  The 0.1% margin is deliberate:
# the optimiser holds binding branches at exactly 1.000, and the arithmetic
# lands a few parts in 1e-10 either side of that.  A bare `> 1.0` would report
# five phantom violations in the intact case.
OVERLOAD = 1.001


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def components(K, b, dropped):
    """How many electrical islands remain once ``dropped`` is out of service.

    Structural, not numerical - O(V+E) rather than another pseudoinverse.
    """
    live = np.ones(K.shape[1], dtype=bool)
    live[dropped] = False
    rows, cols = np.nonzero(K[:, live])
    edge_of = cols
    ends = {}
    for r, e in zip(rows, edge_of):
        ends.setdefault(e, []).append(r)
    pairs = [v for v in ends.values() if len(v) == 2]
    if not pairs:
        return K.shape[0]
    a = np.array([p[0] for p in pairs])
    z = np.array([p[1] for p in pairs])
    n = K.shape[0]
    A = csr_matrix((np.ones(len(a)), (a, z)), shape=(n, n))
    return connected_components(A, directed=False)[0]


def stage4(PTDF, K, b, phi, p, s_nom):
    """Flows, loading and per-branch signs at every snapshot.

    Exactly Stage 4 of ``shift-factor-ptdf-report.md``, unchanged.
    """
    p_eff = p + (K @ (b * phi))[:, None]
    F = PTDF @ p_eff - (b * phi)[:, None]
    loading = np.abs(F) / s_nom[:, None]
    signs = np.sign(F)
    signs[signs == 0] = 1.0
    peak = loading.argmax(axis=1)                  # each branch's own worst hour
    sign_peak = signs[np.arange(len(F)), peak]
    return F, loading, sign_peak, peak


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv):
    limit = None
    if "--limit" in argv:
        limit = int(argv[argv.index("--limit") + 1])
    os.makedirs(OUT, exist_ok=True)
    gridkit.quiet()

    print("loading network and solving the intact dispatch ...")
    n = gridkit.load("WP2033", "all-island")
    br, K, b, phi, PTDF0, rank0 = ptdf_all.build(n)
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)

    buses, edges = n.buses.index, br.index
    snaps = n.snapshots
    s_nom = br["s_nom"].to_numpy(float)
    p = ptdf_all.injections(n, snaps).reindex(buses).fillna(0.0).to_numpy()

    wind = n.generators[n.generators["carrier"] == "wind"].groupby("bus")["p_nom"].sum()
    wcol = np.array([buses.get_loc(w) for w in wind.index])
    wnames = list(wind.index)
    E, W = len(br), len(wcol)
    print(f"{len(buses)} buses, {E} branches, {W} wind nodes, {len(snaps)} snapshots")

    # ---- the intact case, as the baseline every outage is measured against ---
    F0, load0, sign0, _ = stage4(PTDF0, K, b, phi, p, s_nom)
    S0 = PTDF0[:, wcol] * sign0[:, None]
    base_islands = components(K, b, [])
    print(f"intact: rank {rank0}/{len(buses)}, {base_islands} islands, "
          f"{int((load0 > OVERLOAD).any(axis=1).sum())} branches over rating\n")

    todo = range(E if limit is None else min(limit, E))
    rows, viol, sens = [], [], []
    t0 = time.perf_counter()

    for k in todo:
        bk = b.copy()
        bk[k] = 0.0                                   # <- the entire change

        L = (K * bk) @ K.T
        PTDF = (bk[:, None] * K.T) @ np.linalg.pinv(L, hermitian=True)
        F, load, sign, peak = stage4(PTDF, K, bk, phi, p, s_nom)
        S = PTDF[:, wcol] * sign[:, None]

        islands = components(K, b, [k])
        alive = np.ones(E, bool)
        alive[k] = False                              # the tripped branch itself

        over = (load > OVERLOAD).any(axis=1) & alive
        worst = load.max(axis=1)
        dS = np.abs(S - S0)
        dS[k] = 0.0

        rows.append(dict(
            outage=edges[k], kind=br["kind"].iloc[k], s_nom=s_nom[k],
            islands_created=islands - base_islands,
            n_violations=int(over.sum()),
            worst_loading_pct=100.0 * float(worst[alive].max()),
            overload_MW=float(np.clip(np.abs(F) - s_nom[:, None], 0, None)[alive].max()),
            max_dS=float(dS.max()), mean_dS=float(dS.mean()),
            frac_S_moved_1pct=float((dS > 0.01).mean()),
        ))

        for e in np.nonzero(over)[0]:
            hrs = int((load[e] > OVERLOAD).sum())
            viol.append(dict(
                outage=edges[k], branch=edges[e], kind=br["kind"].iloc[e],
                s_nom=s_nom[e], hours_over=hrs,
                peak_loading_pct=100.0 * float(worst[e]),
                peak_overload_MW=float(np.abs(F[e]).max() - s_nom[e]),
                intact_loading_pct=100.0 * float(load0[e].max()),
                n_helpful_nodes=int((S[e] >= 0.05).sum()),
                best_sensitivity=float(S[e].max()),
            ))
            sens.append(pd.Series(S[e], index=wnames,
                                  name=(edges[k], edges[e])))

        if (k + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(f"  {k+1:4d}/{len(todo)}  {el:6.0f}s elapsed  "
                  f"~{el/(k+1)*(len(todo)-k-1):5.0f}s left  "
                  f"({len(viol)} violations so far)")

    # ---- write -----------------------------------------------------------
    out = pd.DataFrame(rows).set_index("outage")
    out.round(6).to_csv(os.path.join(OUT, "outages.csv"))

    v = pd.DataFrame(viol)
    if len(v):
        v.round(4).to_csv(os.path.join(OUT, "violations.csv"), index=False)
        M = pd.DataFrame(sens)
        M.index = pd.MultiIndex.from_tuples(M.index, names=["outage", "branch"])
        M.round(6).to_csv(os.path.join(OUT, "sensitivity_n1.csv"))

    el = time.perf_counter() - t0
    print(f"\ndone in {el/60:.1f} min")
    print(f"  outages run          {len(out)}")
    print(f"  outages that island  {int((out['islands_created'] > 0).sum())}")
    print(f"  outages with a violation  {int((out['n_violations'] > 0).sum())}")
    print(f"  distinct branches ever overloaded  "
          f"{v['branch'].nunique() if len(v) else 0}")
    print(f"  (outage, branch) violation pairs   {len(v)}")
    print(f"\n-> {os.path.relpath(OUT, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
