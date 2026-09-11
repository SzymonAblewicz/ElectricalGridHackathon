"""Does the cascade protocol hold up in different weather?

    python n1_weather.py

Runs all 980 single-branch outages at a handful of hours spanning the week,
from the calmest to the windiest.  For each hour it reports:

    how many outages cause an overload,
    how often safe wind curtailment clears it,
    and how many new overloads the remedy caused.

The PTDF for an outage does not depend on the weather, so each outage is
rebuilt once and then evaluated at every chosen hour.

Writes ``weather.csv`` into ``participant-kit new/results/N-1 contingencies/``.
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
OUT = os.path.join(KIT, "results", "N-1 contingencies")
sys.path.insert(0, KIT)
sys.path.insert(0, HERE)

import gridkit          # noqa: E402
import ptdf_all         # noqa: E402

T = 0.05          # group cut
OVERLOAD = 1.001  # 0.1% margin, see n1.py
N_HOURS = 6       # hours sampled across the week, calmest to windiest


def main():
    gridkit.quiet()
    n = gridkit.load("WP2033", "all-island")
    br, K, b, phi, _, _ = ptdf_all.build(n)
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)

    buses = n.buses.index
    s = br["s_nom"].to_numpy(float)
    p = ptdf_all.injections(n, n.snapshots).reindex(buses).fillna(0.0).to_numpy()

    g = n.generators
    wind = g.index[g["carrier"] == "wind"]
    out = n.generators_t.p[wind].T.groupby(g.loc[wind, "bus"].to_numpy()).sum().T
    wb = list(out.columns)
    wc = np.array([buses.get_loc(x) for x in wb])
    cap = float(g.loc[wind, "p_nom"].sum())
    fleet = out.sum(axis=1).to_numpy()

    # hours spanning the range of fleet output, calmest to windiest
    order = np.argsort(fleet)
    hours = sorted(order[np.linspace(0, len(order) - 1, N_HOURS).round().astype(int)])
    print(f"{cap:,.0f} MW of installed wind")
    for h in hours:
        print(f"  hour {h:3d}  fleet {fleet[h]:7.0f} MW  "
              f"capacity factor {fleet[h]/cap:.0%}")
    print()

    E = len(br)
    tally = {h: dict(outages_with_overload=0, cases=0, cleared=0,
                     new_overloads=0, wind_cut_MW=0.0, no_lever=0)
             for h in hours}

    t0 = time.perf_counter()
    for k in range(E):
        bk = b.copy()
        bk[k] = 0.0
        Pk = (bk[:, None] * K.T) @ np.linalg.pinv((K * bk) @ K.T, hermitian=True)
        F = Pk @ (p + (K @ (bk * phi))[:, None]) - (bk * phi)[:, None]
        alive = np.ones(E, bool)
        alive[k] = False

        for h in hours:
            f = F[:, h]
            load = np.abs(f) / s
            over = np.nonzero((load > OVERLOAD) & alive)[0]
            if len(over) == 0:
                continue
            tally[h]["outages_with_overload"] += 1

            sg = np.sign(f)
            sg[sg == 0] = 1.0
            S = Pk[:, wc] * sg[:, None]
            avail = out.iloc[h].reindex(wb).fillna(0.0).to_numpy()

            # Flow change per MW cut at each wind node.  PTDF rows sum to zero,
            # so the distributed-slack correction vanishes and this is simply
            # minus the node's PTDF column.
            V = -Pk[:, wc]

            for e in over:
                need = abs(f[e]) - s[e]
                grp = np.nonzero((S[e] >= T) & (avail > 0))[0]
                seq = grp[np.argsort(-S[e][grp])]

                # Greedy, cumulative.  Work down from the strongest node and
                # stop each one at the point where the flows *as they now
                # stand* would push some other branch past its rating.  The
                # headroom is recomputed after every node, so simultaneous
                # cuts can never add up past a limit.
                cur = f.copy()
                cut = np.zeros(len(wb))
                rel = 0.0
                for i in seq:
                    if rel >= need - 1e-9:
                        break
                    v = V[:, i]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        up = np.where(v > 1e-12, (s - cur) / v, np.inf)
                        dn = np.where(v < -1e-12, (-s - cur) / v, np.inf)
                    room = np.minimum(up, dn)
                    room[~alive] = np.inf
                    room[e] = np.inf                  # the branch we are fixing
                    limit = max(0.0, float(np.min(room)))
                    take = min(avail[i], limit, (need - rel) / S[e][i])
                    if take <= 0:
                        continue
                    cut[i] = take
                    cur = cur + take * v
                    rel += take * S[e][i]

                l2 = np.abs(cur) / s
                t = tally[h]
                t["cases"] += 1
                t["cleared"] += int(rel >= need - 1e-6)
                t["wind_cut_MW"] += float(cut.sum())
                t["no_lever"] += int(len(grp) == 0)
                t["new_overloads"] += int(((l2 > OVERLOAD) & alive
                                           & (load <= OVERLOAD)).sum())

        if (k + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {k+1:4d}/{E}  {el:5.0f}s  ~{el/(k+1)*(E-k-1):4.0f}s left")

    rows = []
    for h in hours:
        t = tally[h]
        rows.append(dict(
            hour=h, fleet_MW=round(fleet[h], 0),
            capacity_factor=round(fleet[h] / cap, 3),
            outages_run=E,
            outages_with_overload=t["outages_with_overload"],
            overload_cases=t["cases"],
            cleared=t["cleared"],
            clear_rate=round(t["cleared"] / max(t["cases"], 1), 3),
            no_wind_lever=t["no_lever"],
            new_overloads=t["new_overloads"],
            wind_cut_MW=round(t["wind_cut_MW"], 1)))
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(OUT, "weather.csv"), index=False)
    print()
    print(d.to_string(index=False))
    print(f"\ntotal new overloads caused, all hours: {int(d.new_overloads.sum())}")
    print(f"-> {os.path.relpath(os.path.join(OUT, 'weather.csv'), HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
