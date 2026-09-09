# results / sensitivity

**PTDF re-pointed along each branch's real flow direction. This is the form
every question is actually asked in, so everything worth looking at lives here.**

```
sensitivity[e, n]  =  ptdf[e, n] × sign(f[e])

    positive → curtailing at node n UNLOADS branch e
    negative → curtailing at node n LOADS IT FURTHER
```

That matches EirGrid's stated convention.

---

## Layout

```
sensitivity/
├── sensitivity_wind.csv        980 × 157 matrix, mirrors ../ptdf/ptdf_wind.csv
├── sensitivity_wind_long.csv   153,860 rows, same data, one row per pair
├── node_scores.csv             157 rows, one per wind node — the overlap scores
├── viewers/                    two interactive HTML tables
└── figures/                    nine plots, 01 → 09, in reading order
```

**Start with `viewers/ptdf_viewer_withOverlaps.html`.** Open it in a browser —
no server, no internet, no dependencies.

| | |
|---|---|
| `viewers/ptdf_viewer.html` | the plain table: sort, filter, search, transpose |
| `viewers/ptdf_viewer_withOverlaps.html` | all of that **plus live overlap scoring** |

The overlap viewer adds a third view (**Node scores**), a **threshold slider**
that recomputes every score as you move it, a **branch-set selector** (congested
/ stressed / all 980) and a **weighting toggle** (hours at rating vs equal). Six
summary cards update live, including how much the top-20 changes between ranking
by depth and ranking by net relief.

Both show sensitivity by default with a toggle back to raw PTDF, and mark with
`↺` the branches whose stored `bus0→bus1` axis runs against real flow.

Regenerate everything: `python ptdf_all.py --full`, then `python explore_ptdf.py`
and `python overlap.py`.

---

## The two CSVs

Same numbers, two shapes. The matrix mirrors `../ptdf/ptdf_wind.csv` row for row
and column for column, so the two compare entry by entry. The long file is one
row per (branch, node) pair — the shape spreadsheets and `groupby` want — and
carries per-node metadata inline.

Long-file columns: `branch`, `bus`, `ptdf` (raw), `sensitivity` (signed),
`wind_MW`, `wind_farms`.

### Why the sign correction is not cosmetic

`bus0`/`bus1` orientation comes from whatever order the buses appeared in the
source file. It has **no physical meaning**. Where that labelling runs against
real flow, raw `ptdf` is inverted — reversing the ranking and handing you exactly
the wrong wind farms. Roughly half of all branches are affected, and a
sign-flipped list looks entirely plausible. Flagged as `sign = −1` in
`../network/branches.csv`.

### Which hour the sign comes from

Each branch is signed at **its own peak-loading hour**, listed as
`peak_snapshot` in `../network/branches.csv`. Checked, not assumed: over the
hours each branch spends above 50% loading, agreement with its peak-hour sign is
**0.986 on average, 0.874 at worst**, and 1.000 for eight of the ten congested
branches.

For any other hour, rebuild rather than using this file:

```python
sens_at_hour = ptdf_wind[e] * signs.loc[e, hour]
```

The full 980 × 157 × 168 product is 25.8 million values and is never worth
materialising when two small files reconstruct any slice of it.

### Why there is no `sensitivity_full`

Non-wind nodes are not curtailable here, so a 754-column signed version would be
~55 MB of purely derived data. One line rebuilds it:
`ptdf_full.mul(branches["sign"], axis=0)`.

More generally: **sensitivity carries no information that PTDF and the
per-branch sign column do not.** It ships because it is the form the work is
done in.

---

## `node_scores.csv` — the overlap scores

One row per wind node. **The question:** a node with a large shift factor on one
circuit relieves that circuit and nothing else. A node with a moderate factor on
five congested circuits may be worth more system-wide. Ranking by
`max |sensitivity|` cannot tell the two apart.

**The catch, and why these score rather than count:** curtailing a node relieves
every branch where its sensitivity is positive and **worsens** every branch where
it is negative. A node touching six branches with three signs each way is broad
and nearly worthless.

| column | meaning |
|---|---|
| `reach` | congested branches where `abs(sensitivity) >= 0.05` |
| `helps` / `hurts` | of those, how many it relieves / worsens |
| `depth` | max `abs(sensitivity)` on any congested branch |
| `relief` / `harm` | weighted sum of positive / negative sensitivities |
| `net` | `relief − harm`, branches weighted by **hours at their rating** |
| `net_flat` | same, every congested branch weighted equally |

Two weightings because they answer different questions. `net` is "what is worth
curtailing *this week*" and is dominated by `3581-89516-1`, which binds 118 of
168 hours. `net_flat` is "what would be worth curtailing if congestion were
spread evenly" — closer to weather-robust. **They correlate at only 0.65.**

### What the numbers say

| | |
|---|---|
| Spearman, depth vs breadth-aware | **0.61** |
| Top-20 overlap between the two rankings | **11/20** |
| Median rank move | **29 places** (max 110) |
| Nodes reaching ≥ 2 congested branches | **109 of 157** |
| Of those, mixed signs | **38** |
| Nodes with `net < 0` — curtailing makes things **worse** | **45 of 157** |

Three things follow:

1. **Breadth is the norm, not the exception.** 109 of 157 reach two or more
   branches, so `max |sensitivity|` discards information about most of the fleet.
2. **Breadth without sign agreement is not a lever.** A third of broad nodes pull
   both ways at once, and 45 nodes are net-negative overall. A naive "curtail the
   well-connected farms" rule would pick some of them.
3. **The ranking genuinely changes.** Nearly half the top 20 differs. Which
   weighting is right depends on whether you are optimising for this week's
   congestion or for the network — that is a decision, not a computation.

---

## `figures/` — nine plots in reading order

| | |
|---|---|
| `01_heatmap.png` | all 10 congested branches × all 157 nodes at once, ordered so each group is a block. **Two rows are flat grey**: `T3662-36671-1` (75 h) and `T54640-54630-1` are insensitive to wind anywhere — no curtailment relieves them |
| `02_concentration.png` | each branch's ranked curve, and how many nodes reach a given share of its total. Steep = tight group; flat = curtailment is a blunt tool there |
| `03_group_overlap_upset.png` | UpSet, not Venn (a Venn cannot honestly draw eight sets). 133 of 157 clear 5% on at least one branch, 109 on more than one |
| `04_breadth_vs_depth.png` | the core plot: depth against reach, plus a slopegraph showing how the top 15 reorders when breadth is counted |
| `05_systemwide_score.png` | best 22 nodes, each bar split by *which* branch the value comes from. Single colour = specialist, stacked = real multi-branch lever |
| `06_branch_similarity.png` | Jaccard overlap between groups, and correlation of full sensitivity rows. Anti-correlated branches **cannot** be relieved by one instruction |
| `07_beyond_congested.png` | the same question over 10 congested / 28 stressed / all 980 branches — because the congested set is one draw from one synthetic week |
| `08_node_matrix.png` | the full matrix sorted by reach then net relief. Left edge is where the levers are |
| `09_map_score.png` | reach and net relief on the map. **Reach is spread island-wide; net relief concentrates in Donegal**, with a band in NI where curtailing actively hurts |

---

## Caveats

- **The 5% threshold is illustrative, not a standard.** EirGrid draws groups case
  by case and does not publish cut-offs. `reach` moves if you change it — the
  overlap viewer's slider exists so you can see by how much.
- **Only the signed values depend on the dispatch.** `../ptdf/` is pure network.
  Swap in real EirGrid generation and PTDF is unchanged; the signs and the
  congested set get recomputed.
- **Everything overlap-related rests on the congested set**, which comes from the
  kit's **synthetic** week. `07_beyond_congested.png` exists to show how much the
  conclusion depends on that.
- **Base case only.** Under N−1 the congested set would be larger and the overlap
  structure different.
- **Ranking is robust to the balancing convention** — changing slack shifts every
  value in a row by the same constant.
- **Linearity limit.** "Positive means relief" holds until curtailing drives the
  flow through zero. Not a practical concern at realistic volumes, but not
  unconditional either.
- All 980 branches are present, including the 88 dead-end stubs that carry no
  flow. Filter on `../network/branches.csv` for real candidates.
