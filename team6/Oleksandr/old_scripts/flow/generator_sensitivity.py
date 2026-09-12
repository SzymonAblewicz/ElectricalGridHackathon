# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "pypsa"]
# ///
"""Every real generator's sensitivity on every branch of WP2033_all-island, one row per generator.

Szymon's results are per bus: `ptdf_full.csv` is 980 branches x 754 buses. A generator has no
sensitivity of its own - it inherits its bus's column - so this is a join, not a computation:
look up each generator's bus and transpose. Two files come out:

  generator_ptdf_WP2033.csv         raw PTDF, on the arbitrary bus0 -> bus1 axis of each branch.
                                    Pure network: independent of dispatch, weather and hour.
  generator_sensitivity_WP2033.csv  PTDF x sign(f), each branch signed at its own peak-loading
                                    hour (`sign` in branches.csv). Positive = reducing this
                                    generator's output UNLOADS the branch; EirGrid's convention.

Layout: generator, bus, carrier, p_nom_MW, then one column per branch (980).

Balancing is Szymon's: the compensating MW is spread uniformly over all buses. Changing that
convention shifts every value in a branch column by the same constant - rankings are unchanged,
absolute values are not. Load-shedding dummies (213 of 929) are dropped; they are placeholders
for unserved demand, not plants.
"""
from pathlib import Path

import pandas as pd
import pypsa

ROOT = Path(__file__).resolve().parents[3]
KIT = ROOT / "team6" / "Szymon" / "participant-kit new"
RESULTS = KIT / "results"
OUT = Path(__file__).resolve().parents[1] / "data"

# AC-isolated behind DC links; Szymon's ptdf README says their columns are near-degenerate.
ISOLATED = {"86221", "GB_EWIC", "GB_GREENLINK"}


def main():
    n = pypsa.Network(KIT / "networks" / "WP2033_all-island.nc")
    gens = n.generators.loc[n.generators["carrier"] != "load shedding", ["bus", "carrier", "p_nom"]]
    gens = gens.assign(bus=gens["bus"].astype(str))

    ptdf = pd.read_csv(RESULTS / "ptdf" / "ptdf_full.csv", index_col=0)
    sign = pd.read_csv(RESULTS / "network" / "branches.csv", index_col=0)["sign"]

    missing = set(gens["bus"]) - set(ptdf.columns)
    assert not missing, f"generator buses absent from ptdf_full: {missing}"
    assert sign.index.equals(ptdf.index), "branches.csv and ptdf_full.csv rows disagree"
    if (sign == 0).any():
        print(f"warning: {(sign == 0).sum()} branches have sign 0 - their signed column is all zero")

    meta = pd.DataFrame({"generator": gens.index, "bus": gens["bus"].to_numpy(),
                         "carrier": gens["carrier"].to_numpy(), "p_nom_MW": gens["p_nom"].to_numpy()})

    OUT.mkdir(exist_ok=True)
    for name, matrix in [("generator_ptdf_WP2033.csv", ptdf),
                         ("generator_sensitivity_WP2033.csv", ptdf.mul(sign, axis=0))]:
        per_gen = matrix[gens["bus"]].T.reset_index(drop=True)
        table = pd.concat([meta, per_gen], axis=1)
        table.to_csv(OUT / name, index=False, float_format="%.6g")
        print(f"{name}: {table.shape[0]} generators x {per_gen.shape[1]} branches")

    stranded = gens[gens["bus"].isin(ISOLATED)]
    if len(stranded):
        print(f"{len(stranded)} generators on AC-isolated buses (treat their rows with care):")
        print(stranded.to_string())


if __name__ == "__main__":
    main()
