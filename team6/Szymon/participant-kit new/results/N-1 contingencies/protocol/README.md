# A protocol for stopping cascades, and what testing it found

Every single-branch outage, at every hour of the week — **20,203 events where a
trip drives at least one other circuit over its rating** — with four different
remedies dispatched against each one and the result measured.

Regenerate with:

```
python protocol.py            # the sweep            ~22 min
python protocol_table.py      # the lookup table     ~4 min
python cascade_protocol.py    # second order (N-1-1) ~65 min
python protocol_report.py     # numbers and figures  instant
```

All four live in `team6/Szymon/`.

---

## The protocol

> A line or node open-circuits. We recompute the flow everywhere:
>
> ```
> f = PTDF · p
> ```
>
> `p` is the net power injection at each bus — generation minus load, in MW.
> `PTDF` spreads those injections across the lines. Compare each result to its
> rating. **Any line above 100% is the one that will overheat and break next.**
>
> We then take that line's precomputed constraint group and dispatch those
> nodes down, so the congestion is relieved **without overloading anything
> else.**

Nothing in step 2 or 3 is computed live: every single-branch outage is
precomputed. Online it is a table read and one matrix–vector product.

**Why the flow vector and not the sensitivity matrix.** Sensitivity says how a
flow *responds* to curtailment; it says nothing about how much flow is on the
line right now. A circuit can absorb a large redistributed flow while its
sensitivity barely moves, and another can have its sensitivity rewritten while
sitting at 40% loaded. Tested directly: flagging the top 1% of circuits by
sensitivity change catches **0 of 47** actual overloads and raises 147 false
alarms. The flow vector is a direct calculation and it is exact.

---

## The headline

| | cleared | median MW cut | events where it broke a healthy circuit | circuits broken | MW driven over | group |
|---|---|---|---|---|---|---|
| EirGrid threshold | 92.0% | 2.2 | **642** | **873** | **675** | 19 nodes |
| Net sensitivity alone | 94.3% | 1.4 | **1,085** | **1,354** | **2,637** | 1 node |
| Net sensitivity **+ veto** | 89.9% | 1.4 | **0** | **0** | **0** | 1 node |
| Per-circuit **+ veto** | 89.5% | 1.3 | **0** | **0** | **0** | 1 node |
| **Safe optimum (LP)** | **99.8%** | 1.8 | **0** | **0** | **0** | 1 node |

Four things fall out of that table, and they do not all point the same way.

---

## 1. The published method breaks circuits. This is the result.

In **642 of 20,203 events**, dispatching an EirGrid-style constraint group
pushed a circuit that was *inside its rating* over it. **873 circuit-breakages
in total**, and they have names:

| circuit | | events | worst loading |
|---|---|---|---|
| `2781-4951-1` | line | **532** | 104.7% |
| `3581-89516-1` | line | 231 | 102.3% |
| `T89510-89515-1` | transformer | 44 | 100.6% |
| `T89515-89516-1` | transformer | 44 | 100.6% |
| `1122-1742-1` | line | 24 | 100.6% |
| 4 others | | 16 | ≤100.5% |

Every one of these was healthy before the remedy ran. The mechanism is simple
and it is structural, not a tuning error: **membership in an EirGrid group is
decided by a node's shift factor on the constrained circuit, and nothing in the
calculation looks at what that node does to any other circuit.**

Magnitude matters as much as count, so it is reported: the total excess is
**675 MW**, worst single case 104.7% of rating. These are modest overloads, not
instant failures — but they are new constraint violations created by the action
taken to remove one.

---

## 2. The veto removes all of it, and costs almost nothing

The rule, stated once:

```
a cut of x MW at node n is allowed only if, for EVERY live branch k,

        | f_k  +  V[k,n] · x |  ≤  rating_k

  and no branch already over its rating is made worse.
```

`V[k,n]` is the change in branch k's flow per MW curtailed at node n. Every
branch in the network, not just the congested ones — which is the whole point,
because **the circuits that got broken were never in the congested set**, so any
rule that only inspects congested circuits cannot see them coming.

Result: **0 breakages in 20,203 events.** Not "fewer" — zero, and zero again
under the LP, which states the same rule exactly and hands it to a solver.

It costs **24% less wind, not more.** On the 17,856 events both methods clear,
EirGrid curtails a median 2.18 MW against 1.37 MW — because it splits the cut
pro-rata across 19 nodes including ineffective ones, while the safe method
targets the effective ones. The safe method is both safer and cheaper.

---

## 3. Net sensitivity *on its own* makes things worse — a negative result

This is the one to be careful about.

Ranking nodes by net sensitivity across all congested circuits, with **no** veto,
broke **1,354 circuits in 1,085 events** — *more* than the EirGrid method, and
the total excess is **2,637 MW, four times worse**.

The reason is mechanical. Net-sensitivity ordering concentrates the whole
instruction on the single most effective node (median group: **1 node** against
EirGrid's 19). A large cut at one node moves the rest of the network much harder
than the same megawatts spread across nineteen.

**So the overlap idea is not a safety mechanism.** The safety comes entirely
from the veto. The two are independent and should be presented as such.

---

## 4. The overlap ranking does not pay for itself either

Isolated properly: two arms identical in every respect — same veto, same greedy
step, same stopping rule — differing *only* in whether a node is ranked on the
worst circuit alone or on the sum across every circuit currently over.

| circuits over at once | events | EirGrid MW | ranked on worst circuit | ranked on all (overlap) | overlap gain |
|---|---|---|---|---|---|
| 1 | 12,437 | 1.20 | 0.74 | 0.74 | 0.0% |
| 2 | 1,849 | 9.80 | 6.42 | 6.50 | −1.3% |
| 3 | 2,622 | 18.67 | 11.05 | 11.26 | −1.9% |
| 4 | 359 | 326.4 | 267.5 | 273.7 | −2.3% |
| 5 | 417 | 596.1 | 432.6 | 494.1 | −14.2% |
| 6 | 149 | 395.3 | 330.9 | 330.9 | 0.0% |
| 7 | 21 | 713.5 | 546.1 | 617.4 | −13.1% |

**Overall the overlap ranking curtails 5.0% MORE wind, not less.** On the 5,419
events with more than one circuit over, 5.7% more.

The single-circuit row is the check that the comparison is clean: with one
circuit over, "sum across all congested circuits" and "the worst circuit" are
the same quantity, so the two arms must agree exactly. They do —
`max |difference| = 0.00e+00 MW` across 12,437 events.

It buys a little clearance back (89.9% vs 89.5%) but it is a wash, and on MW it
is slightly negative. **In this network, on this problem, overlap scoring is not
the win.** The honest reading is that once the veto is in place the binding
constraint is the *cap*, not the ordering — and fixing the worst circuit first
tends to relieve the others as a side effect, because circuits that congest
together are electrically close.

---

## 5. "Wind cannot fix 31% of these" was wrong

The greedy rule fails to clear about 10% of events. That is the greedy rule
giving up, not the grid refusing.

The LP settles it. Minimise total MW curtailed subject to every live branch
ending at or under its rating; infeasibility is then a *proof* that no safe
wind-only remedy exists.

| | |
|---|---|
| events where a safe remedy provably exists | **20,153 of 20,203 — 99.75%** |
| events where none exists, proved infeasible | **50** |
| events the LP failed to clear, of the solvable ones | **0** |

The 50 genuinely impossible ones cluster on a handful of outages — `2712-34620-1`
driving `3462-5142-2` to 166%, the `3082/3091` transformer pair, `1201-2561-1`
into `2561-3781-1`. At those hours a median **1,555 MW of wind is running** and
it still is not enough: the median unremovable residual is **20.6 MW of a 79.3 MW
overload**. Those are the cases that escalate to conventional redispatch. They
are a result, not a gap.

---

## What is actually novel here

Not automated remedial action — EirGrid already runs special protection schemes
(West Group 4 at Cauteen, the NI runback scheme, and others). What is new is:

1. **Systematic coverage.** A precomputed entry for *every* single-branch
   outage, not a hand-built scheme per contingency.
2. **A do-no-harm check on the remedy.** The cut is validated against all 980
   branches before it is issued. The published threshold rule has no equivalent
   and this is where its 873 breakages come from.
3. **A proof of reach.** The LP says not just "here is a remedy" but "no safe
   wind-only remedy exists", which is the information an operator needs to
   escalate.

---

## Method, and what it rests on

**Exactness.** Post-trip flows come from the same pipeline as the intact case
with one susceptance zeroed, `b[k] = 0`. The post-trip PTDF is built by an exact
rank-one update of the intact one; checked against a full pseudoinverse rebuild
on sampled outages every run, **max discrepancy 8.0 × 10⁻⁹ MW**.

**The balancing megawatt.** Curtailing x MW at a node means x MW appears
elsewhere, and *where* is part of the definition of a shift factor. The default
here is the system-operator convention — spread across every load in proportion
to size. Re-run under the pseudoinverse's own uniform reference, every
conclusion holds: EirGrid breaks 349 circuits in 246 events, the veto and the LP
break none.

**Scope: islanding outages are excluded.** 267 of 980 outages disconnect part
of the network; their 6,406 events are in `cases_islanding.csv` and are not in
any number above. An island is a generation-balance problem — the DC
pseudoinverse smears the imbalance across it and the resulting flows are not a
dispatch anyone would run. Curtailing wind is not the remedy for it.

**No branch is over rating in the intact network** (checked: 0 at any of the 168
hours, 10 sit at exactly 100%), so every overload measured here is genuinely
caused by the outage.

### What this does not establish

- **No thermal clock.** The model has no conductor heating in it. It cannot tell
  you whether you have minutes or hours before an overloaded circuit fails, and
  no number here should be quoted as a timescale.
- **The veto is exact only inside the DC model.** It guarantees no violation of
  the same linearisation that produced the flows. Real AC behaviour and model
  error mean an operational version would want a margin below rating.
- **Synthetic weather.** Every hourly figure comes from the kit's generated
  week. The topology-dependent results (which circuits, which groups) do not
  move with weather; the event counts and MW figures do.
- **The dispatch is frozen** at the intact optimum, which is the correct
  question for contingency screening but is not a re-optimisation.

---

## Files

| file | rows | what it holds |
|---|---|---|
| `cases.csv` | 20,203 | one row per (outage, hour) event, all five arms scored |
| `cases_islanding.csv` | 6,406 | the same for islanding outages, out of scope |
| `cases_uniform.csv` | 3,106 | the slack-convention robustness re-run |
| `collateral.csv` | — | one row per circuit broken, by arm, with how far over |
| `cascade.csv` | — | second-order chains, four remedies |
| `summary.csv` | 1 | the headline numbers, machine-readable |
| `lookup/trips.csv` | — | outage → circuits that overload |
| `lookup/groups.csv` | — | (outage, circuit) → ordered nodes, sensitivities, veto caps |
| `figures/` | 5 | see below |

| figure | shows |
|---|---|
| `01_safety.png` | how often each remedy broke a healthy circuit, and by how many MW |
| `02_cost.png` | MW cost relative to EirGrid, and the overlap ranking isolated |
| `03_collateral.png` | the broken circuits by name |
| `04_cascade.png` | second order — where the chain stops |
| `05_reach.png` | what curtailment provably cannot do |
