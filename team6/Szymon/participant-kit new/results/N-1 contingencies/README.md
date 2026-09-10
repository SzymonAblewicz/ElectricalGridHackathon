# N-1 contingencies

Every branch in the network taken out of service in turn — **980 outages** — and
the full sensitivity pipeline rerun for each one.

The intact-network results in [`../`](../) answer *"if I curtail wind at node n,
what happens to branch e."* This answers the question EirGrid actually builds
constraint groups around:

> **"If branch k trips, what overloads, and which wind farms can relieve it?"**

Runtime **7.9 minutes**. Regenerate with:

```
python n1.py              # the sweep
python n1_figures.py      # figures 01-05
```

Both live in `team6/Szymon/`.

---

## The headline

| | |
|---|---|
| branches congested with the network **intact** | **10** |
| branches congested under **some single trip** | **61** |
| **congested only after a fault** | **51** |
| congested intact but never post-fault | **0** |

**A base-case study finds 10 of 61 — it misses 84% of the problem.** The
intact-network set is not a different set; it is a small subset. Every branch
congested with the network whole is also congested post-fault.

The worst case is stark. When `3554-39444-1` trips, branch `4664-5464-1` hits
**146% of its rating — 461 MW over.** With the network whole that branch sits at
**75%**, comfortably inside its limit and invisible to any base-case study.

---

## What's here

| file | rows | what it holds |
|---|---|---|
| `outages.csv` | 980 | one row per outage: what it broke, how far the sensitivity matrix moved |
| `violations.csv` | 1,745 | one row per *(outage, overloaded branch)* pair |
| `sensitivity_n1.csv` | 1,745 × 157 | the wind sensitivities for each of those pairs |
| `figures/` | 5 | see below |

**The sweep computes 981 sensitivity matrices** — one per outage plus the intact
case — each 980 branches × 157 wind farms, **150,782,800 entries in total.**
Only the 1,745 violated pairs are written to disk. Keeping all of it would be
~1.2 GB and almost all of it unused: a branch that never overloads needs no
constraint group.

---

## How it was produced

**One number changes per outage.**

```python
b[k] = 0
```

`b` is the vector of branch susceptances, `1/x`. Setting one entry to zero sets
that branch's reactance to infinity — the connection is still drawn on the map,
but no current can cross it. **That is an open circuit.**

Everything downstream is the existing pipeline, unmodified. `n1.py` imports
`build()` and `injections()` from `ptdf_all.py` as the same objects.

### Why one number is enough

The Laplacian is a sum over branches:

```
L  =  K B Kᵀ  =  Σ over branches e of   b_e · k_e k_eᵀ
```

where `k_e` is column e of the incidence matrix — `+1` at one end, `−1` at the
other. Zeroing `b[k]` **removes exactly one rank-one term.** Nothing else in the
sum is touched.

And because row e of the PTDF is

```
PTDF[e, :]  =  b_e · k_eᵀ L⁺
```

setting `b_k = 0` makes **row k exactly zero, on its own.** A tripped line
carries no flow and has no sensitivity to anything. Every *other* row changes,
because `L⁺` changed — that is the flow the branch was carrying being
redistributed across the rest of the network.

Phase shifters need no special handling: `b_k = 0` carries through both
`p_eff = p + K(b·φ)` and `f = PTDF·p_eff − b·φ` automatically.

### The dispatch is frozen

Generation is solved once for the intact network and held fixed. That is the
correct question for contingency screening: **given generation exactly as it
stands, what breaks if this line trips now?** It is not a re-optimisation.

---

## Findings

### 1. Over half of all single trips break something

| | |
|---|---|
| outages causing at least one violation | **551 of 980 (56%)** |
| outages that split the network into a new island | **267 of 980 (27%)** |
| total *(outage, branch)* violation pairs | **1,745** |

Split by type: 430 of 755 line outages cause a violation, against 121 of 225
transformer outages.

### 2. A few branches are structurally fragile

`3581-89516-1` — already the binding constraint in the intact case — overloads
under **321 different single trips.** Then `1122-1742-1` (219), the Coolnabacky
transformer pair `T3464-…-w1/w2` (218 and 217), and `3082-3122-1` (199).

These are not unlucky circuits. They are load-bearing, and almost anything
happening nearby pushes them over.

### 3. The sensitivity shift is concentrated, not diffuse

Comparing each outage's 980 × 157 sensitivity matrix against the intact one,
entry by entry:

| | median | 99th pct | max |
|---|---|---|---|
| largest single entry change | **0.44** | 1.73 | 2.00 |
| mean change across all entries | 0.0003 | 0.0023 | 0.0032 |
| share of entries moving > 0.01 | **0.29%** | 3.8% | 5.3% |

**Read these two rows together.** Individual entries move enormously — up to
2.0 MW/MW, which is larger than most sensitivities are in absolute terms. But
the *average* movement is near zero, and fewer than one entry in three hundred
shifts meaningfully.

A trip does not redraw the picture. **It rewrites a small part of it, violently.**

### 4. One fixed group per branch works — membership is nearly binary

Take every wind farm that is ever in a branch's group, and ask what share of
that branch's contingencies it appears in:

| appears in… | share of farm–branch pairs |
|---|---|
| **100% of cases** | **67%** |
| ≥90% of cases | **81%** |
| 10–90% — genuinely borderline | **9%** |
| under 10% | 10% |

**Only 9% of farms are borderline.** The rest are permanent members or freak
appearances. There is no middle band, and that is precisely what makes a single
fixed group defensible.

Per branch, comparing the **core** (in every case) with the **union** (in at
least one):

| | |
|---|---|
| median core ÷ union, all 61 branches | **0.91** |
| same, restricted to the 40 branches seen under >1 trip | **0.66** |
| branches where core = union exactly | 27 of 61 |
| median extra farms if you take the union rather than the core | **0** |

### 5. And the fixed group is the intact-network group

Comparing each branch's intact group against the farms appearing in at least
half its contingencies:

| | |
|---|---|
| **exactly identical membership** | **31 of 61 branches (51%)** |
| median overlap where not identical | **0.88** |
| the ≥90% group vs the ≥50% group | identical every time |

```
branch                  intact group    in >=90% of N-1 cases
3581-89516-1                 26                  26
1122-1742-1                  33                  33
T3464-...-w1 / w2            56                  56
3082-3122-1                  19                  19
3691-4041-1                   8                   8
```

**This was not the expected answer, and it reframes the contribution.**

The gap between base-case and N-1 analysis is **not** that base-case groups are
wrong. For a branch you already knew about, the intact group is very nearly the
right group after a fault too.

The gap is that **base-case analysis only finds 10 of the 61 branches that need
a group at all.**

So this is not *better groups for known lines.* It is **groups for the 51 lines
that have no group, because nothing in a base-case study says they need one.**

---

## Figures

| figure | the one question it answers |
|---|---|
| `01_one_fixed_group.png` | **Can one fixed group serve every contingency?** Membership is nearly binary; the middle band is empty |
| `02_what_stays_and_moves.png` | **For a given branch, who stays and who comes and goes?** The four busiest branches, every farm sorted by how often it is in. The cliff is the finding |
| `03_hidden_congestion.png` | **How many branches does a base-case study miss?** The 10 / 0 / 51 split, and the largest post-fault overloads |
| `04_intact_is_the_fixed_group.png` | **Is the fixed group just the intact group?** Size and membership, side by side |
| `05_map.png` | **Where does it happen?** Left: circuits whose loss breaks something. Right: circuits that break, red where a base-case study never flags them |

---

## Verification

| check | result |
|---|---|
| zeroing `b[k]` vs physically deleting branch k and rebuilding | **0.0e+00** — exact, on every branch tested |
| tripped branch's own PTDF row | **0.0e+00** — falls out of the algebra, not imposed |
| islanding count, structural test vs the LODF denominator test | 267 vs 303 — two independent methods, same order |
| intact-case flows vs PyPSA's own DC solver (inherited) | 3.9 × 10⁻⁹ MW |

The first is the one that matters: the shortcut is not an approximation of
rebuilding the network. It *is* rebuilding the network.

---

## Limits — read before quoting a number

- **Islanding.** 267 outages disconnect part of the network. Injections in an
  isolated fragment no longer balance, so `pinv` returns a least-squares answer
  rather than a physical one. Flows near such a fragment should not be trusted.
  **The headline survives without them:** excluding all 267, the sweep still
  finds 58 congested branches, 51 of them post-fault only.

- **Frozen dispatch.** Generation does not respond to the outage. A real control
  room would redispatch conventional plant first and only then constrain wind.
  So these overloads are an upper bound on what wind would actually be asked to
  fix.

- **N-1 only.** No N-1-1. That is 479,710 pairs at this cost — about 67 hours.
  Screening first (only pairs among branches that already violate alone) brings
  it back to minutes, but it has not been done.

- **Line and transformer outages only.** Busbar outages are genuine N-1 events
  and appear throughout EirGrid's own document, but our model carries buses
  without breakers or busbar sections, so a named sectionalising-cubicle outage
  cannot be reproduced faithfully.

- **Everything inherited from the intact run still applies** — DC approximation,
  synthetic weather, planning ratings. See [`../README.md`](../README.md).

- **This is not a reproduction of EirGrid's groups.** Theirs are built on a
  deliberate worst case (low load, all renewables at maximum, summer ratings)
  with a per-line threshold chosen by judgement. Ours run a normal optimised
  dispatch at a fixed 5% cut. The two are comparable as *methods*, not as
  outputs.
