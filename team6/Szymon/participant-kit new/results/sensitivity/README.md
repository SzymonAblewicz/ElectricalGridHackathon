# results / sensitivity

**One file. `sensitivity.csv` — 153,860 rows, 6 columns, 11.19 MB.**

The full PTDF in long format, with the Stage 4 sign correction applied. One row
per (branch, wind node) pair: **980 branches × 157 wind nodes = 153,860**.

This is the file to build constraint groups from.

---

## Columns

| column | what it is |
|---|---|
| `branch` | edge name — 755 lines and 225 transformers |
| `bus` | wind node ID. Join to `../network/buses.csv` for the station name |
| `ptdf` | raw Stage 3 value, on the arbitrary `bus0→bus1` axis |
| `sensitivity` | **`ptdf × sign(f[branch])`** — signed along the actual flow |
| `wind_MW` | installed wind at that node |
| `wind_farms` | how many separate farms sit behind that node |

---

## `sensitivity` is the column you want

```
sensitivity > 0   curtailing at this node UNLOADS this branch
sensitivity < 0   curtailing at this node LOADS IT FURTHER
```

That matches EirGrid's stated convention.

**Why the correction is not cosmetic.** `bus0`/`bus1` orientation is set by
whatever order the buses appeared in the data file — it has no physical meaning.
On any branch where that labelling runs against the real flow, the raw `ptdf`
column is inverted, which **reverses the ranking** and hands you exactly the
wrong wind farms. Roughly half of all branches are affected, and a
sign-flipped list looks completely plausible on inspection.

Multiplying by `sign(f)` re-points the axis along the actual direction of flow.

---

## Which hour the sign comes from

Each branch is signed at **its own peak-loading hour**, not one global hour. A
branch that peaks at 03:00 and one that peaks at 18:00 each get their own. The
hour used is in `../network/branches.csv` as `peak_snapshot`.

**Is one hour enough?** Checked, not assumed. Over the hours each branch is
above 50% loaded, agreement with its peak-hour sign is **0.986 on average and
0.874 at worst**. For all ten congested branches it is 1.000 for eight of them.
So yes — but the check is in `../timeseries/signs.csv` if you want to redo it.

**For any other hour**, rebuild directly rather than using this column:

```python
sens_at_hour = ptdf_wind[e] * signs.loc[e, hour]
```

That is why `ptdf_wind.csv` and `signs.csv` are kept separately — the full
980 × 157 × 168 product would be 25.8 million values, and it is never worth
materialising when two small files reconstruct any slice of it.

---

## Building a constraint group

```python
import pandas as pd
s = pd.read_csv("sensitivity.csv", dtype={"bus": str})

group = s[(s.branch == "3581-89516-1") & (s.sensitivity.abs() >= 0.05)]
group = group.sort_values("sensitivity", key=abs, ascending=False)
```

At a 5% threshold that circuit admits 44 of 157 nodes; at 10%, 19.

**The threshold is a judgement, not a standard.** EirGrid draws its groups
case by case and does not publish the cut-offs. 5% and 10% here are illustrative
reference points only. The ranked curve in
`../../figures/new figures/2_sensitivity.png` shows the shape you are cutting —
there is a real knee after about 15 nodes, which is why any threshold in that
region behaves sensibly.

---

## Caveats

- **Only the `sensitivity` column depends on the dispatch.** The `ptdf` column is
  pure network. Swap in real EirGrid generation and `ptdf` is unchanged;
  `sensitivity` gets recomputed from new flows.
- **Ranking is robust to the balancing convention.** Changing from uniform slack
  to load-weighted or single-slack shifts every value in a row by the same
  constant, so the order does not move.
- **Linearity limit.** "Positive means relief" holds until curtailing far enough
  drives the flow through zero, after which it grows the other way. Not a
  practical concern at realistic curtailment volumes, but it is why the
  statement is not unconditional.
- All 980 branches are present, including the 88 dead-end stubs that carry no
  flow. Filter on `../network/branches.csv` if you only want real candidates.
