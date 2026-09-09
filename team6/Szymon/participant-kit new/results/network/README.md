# results / network

**Two files describing the grid itself — the labels everything else joins to.**

| file | rows | cols | size |
|---|---|---|---|
| `buses.csv` | 754 | 14 | 0.06 MB |
| `branches.csv` | 980 | 17 | 0.17 MB |

Small enough to open in a spreadsheet. Start here.

---

## buses.csv — one row per node

| column | meaning |
|---|---|
| `bus` | ID. The column headers of `../ptdf/*.csv` are these |
| `station` | substation name from the PSSE file |
| `v_nom_kV` | 110 (535), 220 (160), 380 (34), 275 (20), other (5) |
| `jurisdiction` | IE 524, NI 121, unlabelled 107, GB 2 |
| `x`, `y` | longitude, latitude |
| `has_coordinates` | **false = position inferred for drawing only.** 251 of 754 |
| `ac_component` | 0 for the 751-bus mainland; 1/2/3 are AC-isolated DC-link terminals |
| `is_wind_node` | true for 157 |
| `wind_farms` | separate wind generators behind this bus (max 17) |
| `wind_MW` | installed wind |
| `other_generators`, `other_gen_MW` | non-wind, excluding load-shedding dummies |
| `load_MW` | demand |

### Node classes

Not a column — derived, in this priority order:

| class | count | test | significance |
|---|---|---|---|
| wind node | 157 | `is_wind_node` | what you curtail |
| other generation | 108 | `other_gen_MW > 0` | sets the base flow; picks up the slack when wind backs off |
| load only | 102 | `load_MW > 0` | the sinks that decide flow direction, and so every sign |
| junction | 387 | neither | see below |

**The 387 junctions are three different things:**

- **210** are 3-winding transformer star points — a fictitious middle node of a
  real transformer, not a place on the ground.
- **131** have degree ≥ 3: genuine branching hubs. Essential.
- **88** are dead-end stubs of degree 1. **Verified useless**: max flow
  `8.3 × 10⁻¹⁰ MW` across all 168 hours, against 1,039 MW elsewhere. Kept anyway
   — pruning them saves nothing measurable and risks a bug.

### Three buses to know about

`86221` (Scotland), `GB_EWIC`, `GB_GREENLINK` are the far ends of DC links. They
carry coordinates `(0, 0)`, they are AC-isolated, and they are why the
Laplacian's nullspace is 4-dimensional rather than 1. Excluded from every map.
No wind on them.

---

## branches.csv — one row per edge

980 rows: **755 lines + 225 transformers**. Both are real DC-flow edges.

### Network columns

| column | meaning |
|---|---|
| `branch` | name. The row index of every other results file |
| `kind` | `Line` or `Transformer` |
| `bus0`, `bus1` | endpoints. **Orientation is arbitrary** — data-file order, no physical meaning |
| `x_pu` | per-unit reactance, 1 MVA base (PyPSA's `x_pu_eff`) |
| `susceptance` | `1/x_pu`, in MW per radian |
| `phase_shift_rad` | zero except for 2 phase shifters, at 3° and 17° |
| `s_nom` | rating, MVA — TYTFS `RATE1`, a planning number |
| `is_phase_shifter` | true for exactly 2 |

Two things that look like data errors and are not:

- **114 transformers connect two buses at the same voltage.** Busbar couplers.
- **20 transformers have negative `x_pu`.** All are 3-winding star legs, where a
  negative leg is standard and correct.

### Result columns

| column | meaning |
|---|---|
| `peak_snapshot` | the hour this branch is most loaded — where its sign comes from |
| `peak_flow_MW` | signed flow at that hour |
| `max_loading_pct`, `mean_loading_pct` | over the 168-hour week |
| `hours_at_rating` | hours at ≥ 99.9% of `s_nom` |
| `sign` | +1 or −1. Multiply raw PTDF by this to get flow-aligned sensitivity |
| `sign_agreement_when_busy` | fraction of hours above 50% loading sharing the peak-hour sign |
| `sign_ever_flips` | true if the sign differs from hour 0 at any point — 570 of 980, almost all while idle |

### The 10 congested branches

`hours_at_rating > 0`:

| branch | kind | rating | hours | sign agreement |
|---|---|---|---|---|
| `3581-89516-1` | Line | 123 | 118 | 1.000 |
| `T3662-36671-1` | **Transformer** | 582 | 75 | 0.982 |
| `2781-4951-1` | Line | 106 | 37 | 1.000 |
| `1122-11260-1` | Line | 513 | 32 | 1.000 |
| `T3464-3462-34641-1-w1` | **Transformer** | 500 | 20 | 1.000 |
| `T3464-3462-34641-1-w2` | **Transformer** | 500 | 20 | 1.000 |
| `1122-1742-1` | Line | 513 | 9 | 1.000 |
| `3082-3122-1` | Line | 634 | 7 | 1.000 |
| `T54640-54630-1` | **Transformer** | 582 | 4 | 0.874 |
| `3691-4041-1` | Line | 192 | 2 | 1.000 |

**4 of 10 are transformers**, including the second-worst constraint in the
network. A lines-only model would miss all four.

---

## Caveats

- `has_coordinates = False` on 251 buses: those positions are the mean of their
  neighbours, for drawing only. Never use one as a location.
- `s_nom` is `RATE1`, a continuous planning rating. 18 transmission branches in
  the source file carry a 9999 MVA placeholder instead of a real rating.
- Every loading and sign column comes from a dispatch over the kit's
  **synthetic** week. The network columns above them do not.
