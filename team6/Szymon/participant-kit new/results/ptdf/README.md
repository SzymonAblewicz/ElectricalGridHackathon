# results / ptdf

**Two files. This is the core output — everything else is metadata or downstream.**

| file | shape | size |
|---|---|---|
| `ptdf_wind.csv` | 980 × 157 | 3.35 MB |
| `ptdf_full.csv` | 980 × 754 | 16.02 MB |

Rows are branches (755 lines + 225 transformers), columns are buses. First
column is the branch name; the header row is bus IDs.

`ptdf_wind.csv` is `ptdf_full.csv` with only the 157 wind-node columns kept.
Use it unless you specifically need a non-wind node — same numbers, a fifth the
size. `ptdf_full.csv` is only written when `ptdf_all.py` is run with `--full`.

---

## What one entry means

```
PTDF[e, n]  =  change in flow on branch e, in MW,
               per 1 MW injected at bus n,
               with the balancing withdrawal spread uniformly over all buses
```

**Dimensionless** — MW per MW. `0.20` means 20% of a megawatt injected at `n`
appears on branch `e`.

**Both signs occur and both are meaningful.** A negative entry means injection
at `n` pushes flow in the `bus1 → bus0` direction along `e`. It does **not** yet
mean "makes congestion worse" — that reading needs the sign correction, which is
in [`../sensitivity/`](../sensitivity/).

**`PTDF = B Kᵀ L⁺`.** Stage 3 of the pipeline. It is the exact analytic
derivative `∂f_e/∂p_n` of the DC model, not a finite-difference estimate — every
entry falls out of one pseudoinverse.

---

## Two properties worth knowing

**Rows sum to zero.** Injecting 1 MW at every bus simultaneously moves nothing,
so `Σₙ PTDF[e, n] = 0` for every branch. Measured: max `3.6 × 10⁻¹⁰`. This is a
free correctness check on any row you pull.

**These numbers do not depend on the weather, the dispatch, the hour, or the
scenario's generation data.** They are a property of topology and reactance
alone. Change the wind profiles, change the costs, change the demand — this file
is unchanged. That is why it is separated from `timeseries/`.

---

## The balancing convention

A shift factor is "MW on the circuit per MW at the node", but a megawatt cannot
appear on its own — the definition is incomplete until you say where the
balancing megawatt goes. **Here it is spread uniformly over all buses**, which
is what the Moore–Penrose pseudoinverse's own reference gives.

To convert to any other convention, subtract a column:

```python
# single slack at bus `ref`
ptdf[e] - ptdf.loc[e, ref]

# load-weighted distributed slack (what a system operator usually means)
ptdf.loc[e] - (ptdf.loc[e] * load_share).sum()
```

**This does not change the ranking.** Subtracting a reference takes the same
constant off every node in a row, so it cannot reorder nodes, cannot change
which clear a threshold, and cannot change the resulting constraint group. Only
do it to match someone else's published numbers.

---

## Reading a slice

```python
import pandas as pd
ptdf = pd.read_csv("ptdf_wind.csv", index_col=0)

# one monitored circuit, wind nodes ranked by how much it feels them
row = ptdf.loc["3581-89516-1"]
row.reindex(row.abs().sort_values(ascending=False).index).head(10)
```

The full file loads in about a second and sits in ~6 MB of RAM as float64.

---

## Caveats

- **Not signed for congestion relief.** Raw PTDF is on the arbitrary `bus0→bus1`
  axis from the data file. Use `sensitivity/` for the flow-aligned version.
- **Phase shifters do not appear here and should not.** A phase shifter is a
  fixed device setting, not a function of injection, so it differentiates away:
  `∂f/∂p = PTDF` whether or not the network has shifters. It affects the base
  flows in `timeseries/`, not these values.
- **Four AC components.** Buses `86221`, `GB_EWIC` and `GB_GREENLINK` are
  AC-isolated behind DC links, so "uniform slack" is per-component, not
  island-wide. No wind sits on them. Their columns are near-degenerate — ignore
  them.
- DC approximation throughout: no reactive power, no losses, no voltage.
