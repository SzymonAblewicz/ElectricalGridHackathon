# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Do the spectral cut corridors land on the lines the capped dispatch actually fills?

Two scripts here answer "which corridors matter" in unrelated ways, and this scores how far
they agree, branch by branch:

  get_flow_graph.py     clusters the graph and calls a corridor *cut* when its two ends fall
                        in different clusters. A structural claim about one snapshot: these
                        are the seams of the network.

  congested_lines.py    solves the capacity-constrained dispatch over all 168 hours and
                        reports how full each branch runs. An operational claim: this is the
                        steel that is working.

A cut corridor carrying nothing, or a saturated line the clustering treats as interior, is
where the two readings part company - and that is what this file is for.

Per branch, `is_corridor` is 1 when the branch belongs to a cut corridor and 0 otherwise, and
`loading` is its mean loading over the week. The disagreement between them,

    d = | is_corridor - loading |

is 0 when the two agree perfectly (a cut corridor running full, or an interior line running
empty) and 1 when they contradict outright. That is then mapped to an agreement score by a
cubic, clipped into [0, 1]:

    agreement = clip( c3 d^3 + c2 d^2 + c1 d + c0 ,  0, 1)

with COEFFICIENTS below. The cubic replaces the plain `1 - d`: it runs from ~0.994 at d = 0
to ~-0.012 at d = 1, but flatter at both ends and steeper through the middle, so a small
mismatch is forgiven and a mid-range one is punished. It is not monotone over the whole
interval - it rises to 1.009 at d = 0.062 and dips to -0.014 at d = 0.977 - which is why the
clip is not cosmetic: without it the score leaves [0, 1] at both ends.

On the two inputs. `loading` is read from the **capped** congested-lines CSV, the
`capacity_constrained` dispatch whose LOPF holds every branch at or under its rating, so the
column already lives in [0, 1] - the uncapped run reaches 188% and would not. It is a mean
over 168 hours, while the clustering describes the single snapshot in the corridors filename.
The comparison is worth making, but that gap is real: a low score can mean the clustering is
wrong about a seam, or only that the seam was busy in an hour the week-average dilutes.

Reads two CSVs already on disk; nothing is re-solved and no network is loaded.

Writes to `data/agreement/<CASE>/`:

  agreement_<run>_<column>.csv   every branch, worst agreement first

Set the constants below and run the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

FOLDER = Path(__file__).resolve().parent
sys.path.insert(0, str(FOLDER))  # graph_lib sits beside this one

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import graph_lib as gwg  # noqa: E402


# ---- what to run ---- #

CASE = "WP2033_all-island"

#: The clustering to score, under `data/graphs/<CASE>/flow/<STAGE>/`. Written by
#: get_flow_graph.py (stage "clustering") or its Fiedler variant (stage "fiedler"). Only cut
#: corridors are listed in these files, which is all this needs: a branch named by no row is
#: interior by construction.
STAGE = "clustering"
CORRIDORS = "corridors_sym_loading_k6_20300120_1700.csv"

#: The capped congested-lines run, under `data/congestion/<CASE>/`. Must be a `_capped` file:
#: see the docstring on why the uncapped one breaks the [0, 1] assumption. Which LOADING_FRACTION
#: it was written at does not matter here - that sets the `congested` flag, not the loading.
CONGESTED = "congested_lines_WP2033_all-island_lf0.7_capped.csv"

#: Which loading column carries the operational reading: "mean_loading" over the week, or
#: "max_loading" for its worst hour.
COLUMN = "mean_loading"

#: Cubic in d, highest power first. p(0) ~ 0.994, p(1) ~ -0.012, so it stands in for `1 - d`.
COEFFICIENTS = (2.67829813, -4.17329136, 0.48856531, 0.99442608)

#: A branch is called a disagreement below this score, for the summary only.
POOR = 0.5


# ---- the score ---- #


def agreement_score(disagreement: np.ndarray) -> np.ndarray:
    """COEFFICIENTS evaluated on `disagreement`, clipped into [0, 1].

    The clip carries weight at both ends: the cubic peaks at 1.009 around d = 0.062 and
    bottoms out at -0.014 around d = 0.977, so raw output leaves the unit interval for both
    near-perfect agreement and near-total contradiction.
    """
    return np.clip(np.polyval(COEFFICIENTS, disagreement), 0.0, 1.0)


def cut_branches(corridors: pd.DataFrame) -> set[str]:
    """Every branch name named by a cut corridor row.

    `cluster.corridors()` merges parallel circuits into one corridor and keeps their names in
    a space-joined `branches` field, so one cut row can name several branches. All of them
    inherit the cut: the circuits of a corridor share its two ends, so they cross the same
    cluster boundary.
    """
    return {name for row in corridors["branches"] for name in str(row).split()}


def compare(congested: pd.DataFrame, corridors: pd.DataFrame,
            column: str = COLUMN) -> pd.DataFrame:
    """One row per branch: its cut flag, its loading, their disagreement and the score.

    Indexed and ordered by `congested`, which carries every branch of the case, so interior
    branches are scored too - they are most of the network and a clustering that calls them
    interior while they run full is exactly what this is looking for.

    `loading` is clipped into [0, 1] before the subtraction. The capped dispatch should not
    exceed a rating at all, so the clip is a guard, not a correction; it raises instead if a
    cut corridor names a branch the congested table has never heard of, because that means
    the two files describe different networks and the score would be meaningless.
    """
    if column not in congested.columns:
        raise ValueError(f"no {column!r} column in the congested table; "
                         f"available: {', '.join(congested.columns[:10])}")

    branches = congested.index.astype(str)
    cut = cut_branches(corridors)
    missing = sorted(cut - set(branches))
    if missing:
        raise ValueError(
            f"{len(missing)} branch(es) named by a cut corridor are absent from the congested "
            f"table, so the two files disagree about the network: {missing[:5]}")

    loading = congested[column].to_numpy(float).clip(0.0, 1.0)
    is_corridor = np.isin(branches, list(cut)).astype(float)
    disagreement = np.abs(is_corridor - loading)

    frame = pd.DataFrame({
        "kind": congested["kind"],
        "bus0": congested["bus0"].astype(str),
        "bus1": congested["bus1"].astype(str),
        "s_nom": congested["s_nom"],
        "is_corridor": is_corridor.astype(int),
        "loading": loading,
        "disagreement": disagreement,
        "agreement": agreement_score(disagreement),
    })
    frame.index.name = congested.index.name or "branch"
    return frame.sort_values(["agreement", "s_nom"], ascending=[True, False])


# ---- entry point ---- #


def main() -> None:
    corridors_path = gwg.OUT_DIR / CASE / "flow" / STAGE / CORRIDORS
    congested_path = gwg.DATA_DIR / "congestion" / CASE / CONGESTED
    for path in (corridors_path, congested_path):
        if not path.is_file():
            available = (sorted(p.name for p in path.parent.glob("*.csv"))
                         if path.parent.is_dir() else [])
            raise FileNotFoundError(
                f"no such file: {path}\n"
                f"available: {', '.join(available) if available else '(no CSVs in that folder)'}")
    if "_capped" not in CONGESTED:
        raise ValueError(f"{CONGESTED} is not a capped run; the uncapped dispatch exceeds "
                         f"ratings and breaks the [0, 1] loading the score assumes")

    corridors = pd.read_csv(corridors_path)
    congested = pd.read_csv(congested_path, index_col=0)
    frame = compare(congested, corridors, COLUMN)

    run = CORRIDORS.removeprefix("corridors_").removesuffix(".csv")
    out = gwg.DATA_DIR / "agreement" / CASE
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"agreement_{run}_{COLUMN}.csv"
    frame.to_csv(path)

    corridor_rows = frame[frame["is_corridor"] == 1]
    interior = frame[frame["is_corridor"] == 0]
    poor = frame[frame["agreement"] < POOR]
    worst = "\n".join(
        f"    {name:<22} {'cut' if row['is_corridor'] else 'interior':<9} "
        f"loading {row['loading']:>6.1%}   score {row['agreement']:.3f}"
        for name, row in frame.head(10).iterrows())
    print(
        f"{CASE} - {len(frame)} branches, {COLUMN} vs {run}\n"
        f"  agreement  mean {frame['agreement'].mean():.3f} over all branches, "
        f"{poor['agreement'].count()} below {POOR:g}\n"
        f"  cut        {len(corridor_rows)} branches in cut corridors, mean loading "
        f"{corridor_rows['loading'].mean():.1%}, mean score "
        f"{corridor_rows['agreement'].mean():.3f}\n"
        f"  interior   {len(interior)} branches, mean loading "
        f"{interior['loading'].mean():.1%}, mean score {interior['agreement'].mean():.3f}\n"
        f"  worst\n{worst}\n"
        f"  wrote      {path}"
    )


if __name__ == "__main__":
    main()
