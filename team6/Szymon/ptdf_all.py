"""Full PTDF for every branch e and every wind node n, over every snapshot.

    python ptdf_all.py              wind nodes only (157 columns)
    python ptdf_all.py --full       also write the all-754-node matrix

Pipeline, per shift-factor-ptdf-report.md:
    1. K, B, phi, L = K B K^T
    2. L+ = pinv(L)
    3. PTDF = B K^T L+                          snapshot-independent
    4. p_eff = p + K B phi ; f = PTDF p_eff - B phi ; sign(f[e])
                                                snapshot-dependent

Stage 4 is the only part that moves hour to hour, so it is evaluated at all
168 snapshots rather than one.  The signed sensitivity for any hour is
``ptdf_wind[e, n] * signs[e, hour]`` - the two files below are the whole
answer and the 980 x 157 x 168 product is never materialised.

Writes to ``participant-kit new/results/``:
    network/branches.csv        980 rows, one per edge
    network/buses.csv           754 rows, one per node
    ptdf/ptdf_wind.csv          980 x 157, every edge x every wind node
    ptdf/ptdf_full.csv          980 x 754, only with --full
    timeseries/flows.csv        980 x 168, MW
    timeseries/loading.csv      980 x 168, |flow| / s_nom
    timeseries/signs.csv        980 x 168, +1 / -1
    sensitivity/sensitivity.csv 153,860 rows, signed at each branch's peak hour
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.join(HERE, "participant-kit new")
OUT = os.path.join(KIT, "results")
sys.path.insert(0, KIT)

import gridkit  # noqa: E402

#: Which subfolder of results/ each file belongs in.
FOLDERS = {
    "buses.csv": "network", "branches.csv": "network",
    "ptdf_wind.csv": "ptdf", "ptdf_full.csv": "ptdf",
    "sensitivity.csv": "sensitivity",
    "flows.csv": "timeseries", "loading.csv": "timeseries",
    "signs.csv": "timeseries",
}


def out_path(name):
    """Full path for one result file, creating its folder."""
    folder = os.path.join(OUT, FOLDERS[name])
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def build(n):
    """Stages 1-3.  Returns (branch frame, K, b, phi, PTDF, rank)."""
    n.calculate_dependent_values()
    frames = []
    for kind, f in (("Line", n.lines), ("Transformer", n.transformers)):
        if not len(f):
            continue
        shift = f["phase_shift"] if "phase_shift" in f.columns else 0.0
        frames.append(pd.DataFrame({
            "branch": f.index, "kind": kind,
            "bus0": f["bus0"].to_numpy(), "bus1": f["bus1"].to_numpy(),
            "x_pu": f["x_pu_eff"].to_numpy(float),
            "susceptance": 1.0 / f["x_pu_eff"].to_numpy(float),
            "phase_shift_rad": np.radians(np.asarray(shift, dtype=float)),
            "s_nom": f["s_nom"].to_numpy(float),
        }))
    br = pd.concat(frames, ignore_index=True).set_index("branch")

    buses = n.buses.index
    pos = {v: i for i, v in enumerate(buses)}
    K = np.zeros((len(buses), len(br)))
    for e, (b0, b1) in enumerate(zip(br["bus0"], br["bus1"])):
        K[pos[b0], e] += 1.0
        K[pos[b1], e] -= 1.0

    b = br["susceptance"].to_numpy(float)
    phi = br["phase_shift_rad"].to_numpy(float)
    L = (K * b) @ K.T
    rank = np.linalg.matrix_rank(L)
    PTDF = (b[:, None] * K.T) @ np.linalg.pinv(L, hermitian=True)
    return br, K, b, phi, PTDF, rank


def injections(n, snapshots):
    """Net MW per bus at every snapshot.  Buses x snapshots."""
    p = pd.DataFrame(0.0, index=n.buses.index, columns=snapshots)
    gp = n.generators_t.p.loc[snapshots]
    gsign = n.generators["sign"].reindex(gp.columns).fillna(1.0)
    contrib = (gp * gsign).T.groupby(n.generators["bus"].reindex(gp.columns).to_numpy()).sum()
    p.loc[contrib.index] += contrib
    ld = n.loads_t.p_set.loc[snapshots]
    lcontrib = ld.T.groupby(n.loads["bus"].reindex(ld.columns).to_numpy()).sum()
    p.loc[lcontrib.index] -= lcontrib
    if len(n.links):
        p0 = n.links_t.p0.loc[snapshots]
        for name in p0.columns:
            p.loc[n.links.at[name, "bus0"]] -= p0[name].to_numpy()
            p.loc[n.links.at[name, "bus1"]] += (
                p0[name].to_numpy() * n.links.at[name, "efficiency"])
    return p


def main(argv):
    want_full = "--full" in argv
    os.makedirs(OUT, exist_ok=True)
    gridkit.quiet()

    n = gridkit.load("WP2033", "all-island")
    br, K, b, phi, PTDF, rank = build(n)
    buses, edges = n.buses.index, br.index

    # ---- Stage 4, at every snapshot ----------------------------------
    gridkit.solve(n)
    gridkit.freeze_dispatch(n)
    n.lpf(n.snapshots)
    snaps = n.snapshots

    p = injections(n, snaps).reindex(buses).fillna(0.0).to_numpy()   # bus x t
    p_eff = p + (K @ (b * phi))[:, None]
    F = PTDF @ p_eff - (b * phi)[:, None]                            # edge x t

    truth = pd.concat([n.lines_t.p0, n.transformers_t.p0], axis=1)[edges]
    resid = np.abs(F - truth.to_numpy().T).max()

    signs = np.sign(F)
    signs[signs == 0] = 1.0                       # report section 4.8 guard
    loading = np.abs(F) / br["s_nom"].to_numpy(float)[:, None]

    flows_df = pd.DataFrame(F, index=edges, columns=snaps)
    signs_df = pd.DataFrame(signs, index=edges, columns=snaps).astype(int)
    load_df = pd.DataFrame(loading, index=edges, columns=snaps)

    # Each branch's own peak-loading hour is the hour that matters for it.
    peak_i = load_df.to_numpy().argmax(axis=1)
    sign_peak = signs[np.arange(len(edges)), peak_i]
    # How often the sign agrees with that, over hours the branch is >50% loaded.
    busy = loading > 0.5
    agree = np.where(busy.sum(axis=1) > 0,
                     (signs == sign_peak[:, None]).sum(axis=1, where=busy)
                     / np.maximum(busy.sum(axis=1), 1),
                     np.nan)
    flips = (signs != signs[:, [0]]).any(axis=1)

    # ---- metadata ----------------------------------------------------
    g = n.generators
    real = g[g["carrier"] != "load shedding"]
    wind = g[g["carrier"] == "wind"]
    wmw = wind.groupby("bus")["p_nom"].sum()

    bus_meta = pd.DataFrame({
        "bus": buses,
        "station": n.buses["psse_name"].to_numpy(),
        "v_nom_kV": n.buses["v_nom"].to_numpy(),
        "jurisdiction": n.buses["jurisdiction"].to_numpy(),
        "x": n.buses["x"].to_numpy(), "y": n.buses["y"].to_numpy(),
        "has_coordinates": n.buses["has_coordinates"].to_numpy(),
        "ac_component": n.buses["sub_network"].to_numpy(),
        "is_wind_node": buses.isin(wmw.index),
        "wind_farms": wind.groupby("bus").size().reindex(buses).fillna(0).astype(int).to_numpy(),
        "wind_MW": wmw.reindex(buses).fillna(0.0).to_numpy(),
        "other_generators": real[real["carrier"] != "wind"].groupby("bus").size()
                            .reindex(buses).fillna(0).astype(int).to_numpy(),
        "other_gen_MW": real[real["carrier"] != "wind"].groupby("bus")["p_nom"].sum()
                        .reindex(buses).fillna(0.0).to_numpy(),
        "load_MW": n.loads.groupby("bus")["p_set"].sum().reindex(buses).fillna(0.0).to_numpy(),
    }).set_index("bus")

    br_meta = br.copy()
    br_meta["is_phase_shifter"] = np.abs(phi) > 1e-9
    br_meta["peak_snapshot"] = snaps[peak_i]
    br_meta["peak_flow_MW"] = F[np.arange(len(edges)), peak_i]
    br_meta["max_loading_pct"] = 100.0 * loading.max(axis=1)
    br_meta["mean_loading_pct"] = 100.0 * loading.mean(axis=1)
    br_meta["hours_at_rating"] = (loading >= 0.999).sum(axis=1)
    br_meta["sign"] = sign_peak.astype(int)
    br_meta["sign_agreement_when_busy"] = agree
    br_meta["sign_ever_flips"] = flips

    # ---- write -------------------------------------------------------
    wind_buses = bus_meta.index[bus_meta["is_wind_node"]]
    ptdf_wind = pd.DataFrame(PTDF, index=edges, columns=buses)[wind_buses]

    br_meta.to_csv(out_path("branches.csv"))
    bus_meta.to_csv(out_path("buses.csv"))
    ptdf_wind.to_csv(out_path("ptdf_wind.csv"))
    flows_df.to_csv(out_path("flows.csv"))
    load_df.to_csv(out_path("loading.csv"))
    signs_df.to_csv(out_path("signs.csv"))
    if want_full:
        pd.DataFrame(PTDF, index=edges, columns=buses).to_csv(
            out_path("ptdf_full.csv"))
    else:
        stale = os.path.join(OUT, FOLDERS["ptdf_full.csv"], "ptdf_full.csv")
        if os.path.exists(stale):
            os.remove(stale)

    long = ptdf_wind.stack().rename("ptdf").reset_index()
    long.columns = ["branch", "bus", "ptdf"]
    long["sensitivity"] = long["ptdf"] * long["branch"].map(br_meta["sign"])
    long["wind_MW"] = long["bus"].map(bus_meta["wind_MW"])
    long["wind_farms"] = long["bus"].map(bus_meta["wind_farms"])
    long.to_csv(out_path("sensitivity.csv"), index=False)

    # ---- report ------------------------------------------------------
    nl = int((br["kind"] == "Line").sum())
    print(f"nodes {len(buses)}   edges {len(br)} ({nl} lines + {len(br)-nl} transformers)")
    print(f"wind nodes {len(wind_buses)}   wind farms {len(wind)}   {wind['p_nom'].sum():,.0f} MW")
    print(f"Laplacian rank {rank}/{len(buses)} -> {len(buses)-rank} AC components")
    print(f"snapshots {len(snaps)}  ({snaps[0]} .. {snaps[-1]})")
    print()
    print("checks")
    print(f"  1'p           max |.| = {np.abs(p.sum(axis=0)).max():.2e}")
    print(f"  1'p_eff       max |.| = {np.abs(p_eff.sum(axis=0)).max():.2e}")
    print(f"  PTDF row sums max |.| = {np.abs(PTDF.sum(axis=1)).max():.2e}")
    print(f"  flows vs n.lpf() over all {len(snaps)} snapshots: {resid:.2e} MW")
    print()
    print("sign stability across the week")
    print(f"  branches whose sign never flips: {int((~flips).sum())} of {len(edges)}")
    busy_any = busy.sum(axis=1) > 0
    print(f"  of branches ever >50% loaded ({int(busy_any.sum())}): "
          f"mean agreement with peak-hour sign {np.nanmean(agree[busy_any]):.3f}")
    cong = br_meta[br_meta["hours_at_rating"] > 0]
    print(f"\ncongested branches ({len(cong)}):")
    print(cong[["kind", "s_nom", "hours_at_rating", "max_loading_pct",
                "sign", "sign_agreement_when_busy"]].round(3).to_string())
    print()
    for name in sorted(FOLDERS):
        path = os.path.join(OUT, FOLDERS[name], name)
        if os.path.exists(path):
            print(f"  results/{FOLDERS[name]}/{name:18s} "
                  f"{os.path.getsize(path)/1e6:8.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
