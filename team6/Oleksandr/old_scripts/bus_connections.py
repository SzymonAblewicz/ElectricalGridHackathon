# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "matplotlib"]
# ///
"""How many branches touch each bus, and how many buses sit at each voltage.

A bus's connection count is how many times it appears as bus0/bus1 across
lines.csv and transformers.csv - two parallel circuits between the same pair
of buses count as two connections, not one. links.csv (the DC interconnector
links, rated in p_nom rather than s_nom) is excluded, matching the graph
get_weighted_graph.py builds.

Writes one PDF with three histograms: connections per bus, buses per voltage
level, and buses grouped by which set of voltage levels their neighbours sit
at. That last one is where a bus with a 110kV line and a 220kV transformer
both landing on it shows up under "110+220" rather than under either alone -
in this data every line joins two buses at the same voltage, so a bus lands
in a multi-voltage group only via a transformer to a different level. To run
it: set CASE below, then run the file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # written to a file, never shown
import matplotlib.pyplot as plt  # noqa: E402  (needs the backend set above)

FOLDER = Path(__file__).resolve().parent
PYPSA_DIR = FOLDER.parent.parent / "grid_TF_Wind" / "data" / "pypsa"
OUT_DIR = FOLDER / "data" / "graphs"


# ---- what to run ---- #

CASE = "TYTFS2024_SV2024_V35_transmission"


def voltage_group_counts(voltage_groups: pd.Series) -> pd.Series:
    """Buses per voltage group, ordered by how many voltages they span."""
    counts = voltage_groups.value_counts()
    return counts.reindex(sorted(counts.index, key=lambda g: (g.count("+"), g)))


def main() -> None:
    case_dir = PYPSA_DIR / CASE
    buses = pd.read_csv(case_dir / "buses.csv", dtype={"name": str})
    v_nom = buses.set_index("name")["v_nom"]

    ends = {"bus0": str, "bus1": str}
    lines = pd.read_csv(case_dir / "lines.csv", dtype=ends)
    transformers = pd.read_csv(case_dir / "transformers.csv", dtype=ends)
    bus0 = pd.concat([lines["bus0"], transformers["bus0"]], ignore_index=True)
    bus1 = pd.concat([lines["bus1"], transformers["bus1"]], ignore_index=True)
    branch_ends = pd.concat([bus0, bus1])

    connections = branch_ends.value_counts().reindex(buses["name"], fill_value=0)

    # Each branch is one edge (bus0, bus1); the far end's voltage is what the
    # near end is "connected to". Two rows per branch, one per direction, so
    # every bus sees the voltage of every neighbour it has.
    neighbour_voltage = pd.concat(
        [pd.DataFrame({"bus": bus0, "neighbour_v": bus1.map(v_nom)}),
         pd.DataFrame({"bus": bus1, "neighbour_v": bus0.map(v_nom)})],
        ignore_index=True,
    )
    voltage_groups = (
        neighbour_voltage.groupby("bus")["neighbour_v"]
        .apply(lambda vs: "+".join(f"{v:g}" for v in sorted(set(vs))))
    )
    per_group = voltage_group_counts(voltage_groups)

    # Same voltage-group breakdown, restricted to buses at or above a given
    # connection count - printed only, since each is a filtered view of the
    # third chart above rather than a chart of its own.
    degree_thresholds = [3, 4]
    per_group_by_threshold = {
        min_conn: voltage_group_counts(
            voltage_groups.reindex(connections[connections >= min_conn].index)
        )
        for min_conn in degree_thresholds
    }

    fig, (left, mid, right) = plt.subplots(1, 3, figsize=(17, 5))

    left.hist(connections, bins=range(connections.min(), connections.max() + 2))
    left.set_xlabel("connections per bus")
    left.set_ylabel("number of buses")
    left.set_title(f"{CASE}\nbus connection count")

    per_voltage = buses["v_nom"].value_counts().sort_index()
    mid.bar(per_voltage.index.astype(str), per_voltage.to_numpy())
    mid.set_xlabel("voltage level (kV)")
    mid.set_ylabel("number of buses")
    mid.set_title("buses per voltage level")

    bars = right.bar(per_group.index, per_group.to_numpy())
    right.bar_label(bars)
    right.set_xlabel("neighbouring voltage level(s), kV")
    right.set_ylabel("number of buses")
    right.set_title("buses by voltage level(s) they connect to")
    right.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    out = OUT_DIR / CASE
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "bus_connections.pdf", dpi=150)
    plt.close(fig)

    print(f"{len(buses)} buses, {len(lines) + len(transformers)} branches")
    print(connections.describe())
    print(per_voltage)
    print(per_group)
    for min_conn, breakdown in per_group_by_threshold.items():
        n_buses = (connections >= min_conn).sum()
        print(f"\nvoltage group breakdown, buses with {min_conn}+ connections ({n_buses} buses):")
        print(breakdown)
    print(f"wrote {out / 'bus_connections.pdf'}")


if __name__ == "__main__":
    main()
