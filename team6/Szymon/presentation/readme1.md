# How EirGrid actually does it — and where we differ

Written 2026-09-10. Everything here is either quoted from a source or measured
from our own run. Nothing is inferred.

## Sources

| tag | document |
|---|---|
| **[WDT]** | EirGrid/SONI, *Wind Dispatch Tool Constraint Group Overview*, 1 Feb 2024, 74 pp. |
| **[REB]** | EirGrid/SONI, *Guide to Rebalancing — Scheduling and Dispatch Programme*, Nov 2025, 17 pp. |
| **[HACK]** | TPSA Hackathon 2026 Problem Sheet, Sept 2026 |
| **[OURS]** | `participant-kit new/results/`, produced by `ptdf_all.py`. Network `WP2033_all-island`: 754 buses, 980 branches, 157 wind nodes. Flows checked against PyPSA's independent DC solver: agree to 3.9e-09 MW over all 168 hours. |

---

## 1. How a constraint group is built

**One group per problem line.** [WDT] Appendix 2, pp. 71–72:

- Take one violation. Note the flow on the affected circuit.
- Drop one wind farm at each candidate node by 10 MW; add 10 MW at a fixed
  distant conventional generator. Re-run. The change in flow per MW is the
  **shift factor**.
- *"Only one wind farm per node requires studying as each will have the same
  effectiveness"* (p. 71).
- Keep nodes above a threshold; that becomes the group.

**Three details that matter:**

1. **Sign is used.** *"A shift factor greater than 0 indicates that the
   renewable generator is effective... When the shift factor is 0 or below, it
   is not effective... and will not be included in the constraint group"*
   (p. 72). Farms that would worsen the line are excluded.
2. **The threshold is not a fixed number.** *"This threshold will differ for
   each contingency issue... In general, the value will be based on a
   significant step change in the shift factors"* (p. 72). It is a judgement
   call, read off a gap in the sorted list, per line.
3. **Units are amps per MW**, not MW per MW. Their worked example is
   2.9 A / 10 MW = 0.29 (p. 74). **Not comparable to our numbers.**

⚠️ [HACK] p. 6, Algorithm 0 uses `|SF| ≥ τ` — ignoring sign. This contradicts
[WDT] p. 72. Follow [WDT].

**Most groups are post-fault, not base case.** Nearly every group in [WDT]
§§2–8 is defined "for the loss of..." — e.g. NW Group 1 exists because losing
Binbane–Cathaleen's Fall overloads the Letterkenny busbar (p. 12). Only a few
(SW 3a, SW 3b, West 5) are base-case.

## 2. How the cut is issued

**Absolute MW, never a percentage.**

- Operator selects a group and *"a MW reduction level"* [WDT] p. 6.
- Split pro-rata by **current output** — [WDT] Appendix 1, p. 66:
  `setpoint = output − (group cut) × (output / Σ outputs)`
  Same formula in [REB] p. 5.
- Each farm receives *"individual MW setpoints"* [WDT] p. 6.

**The split ignores effectiveness entirely.** Shift factor decides *who is in
the group*. It plays no part in *how much each one loses*.

**Equal percentages don't survive.** At first application everyone loses the
same share (25% each, [WDT] p. 67) — but only because output equalled
availability. After the wind moves, a second cut gives 75% / 37.5% / 62.5% /
66.7% (p. 68), because *"this will always be pro-rata based on the
actual output and does not consider the changing availability."* That
divergence is why the 2025 rebalancing programme exists: farms *"reported being
disproportionately impacted... raised repeatedly with the Regulatory
Authorities"* [REB] p. 3.

## 3. What we measured

**Ordering the same group by effectiveness instead of by output.** Same farms,
same available MW, [WDT]'s own membership rules (positive shift factor above
threshold). Only the split differs. 245 binding branch-hours across 8 of the
10 congested branches, three target sizes. [OURS]

| | saving in MW cut |
|---|---|
| median | **18%** |
| mean | 22% |
| best | 65% |
| worst | **0%** |

By branch (5% threshold, median):

```
1122-1742-1              33 farms   59%
3082-3122-1              19 farms   42%
3581-89516-1             16 farms   28%
T3464-3462-34641-1-w1/w2 54 farms   22%
3691-4041-1               8 farms   17%
2781-4951-1               4 farms    0%
1122-11260-1              1 farm     0%
```

**The saving comes entirely from how much farms within a group differ from each
other.** A one-member group has nothing to reorder.

Two congested branches produced no group at all: `T3662-36671-1` (75 h at its
limit) and `T54640-54630-1`. Both have the *same* sensitivity — 0.0013 — for
every one of the 157 wind farms, which is just 1/754, the numerical background.
**No wind farm can relieve them.** An empty group there is the right answer.

**Note:** ordering by effectiveness can never do worse than ignoring it. That
is arithmetic, not a result. The only question was the size of the gap.

## 4. Overlap

Six stations each sit in six different groups — four intact-network, two
outage-driven — plus the nationwide group. [WDT] pp. 12, 15, 16, 22, 23, 24,
33–34, 65:

| Station | Groups |
|---|---|
| Ardnagappary, Binbane, Lenalea, Sorne Hill, Trillick | NW 1, 3, 4, 5, 6 · West 2 |
| Meentycat | NW **2**, 3, 4, 5, 6 · West 2 |

Each group was built by its own separate study (§1). **A farm's six shift
factors are never combined into one ranking.** The stored group definitions
carry no cross-line information.

⚠️ **This is a statement about the stored data, not about what operators know.**
EirGrid explicitly tracks interaction in real time:

- *"the often interacting nature of constraints and curtailment"* [WDT] p. 66
- *"modelling of the impact of contingency events"* [WDT] p. 5
- Sequential group application when the first is exhausted [WDT] pp. 60–61
- *"layered application of constraint and curtailment actions"* [REB] p. 6

**Do not claim EirGrid cannot see this.** The claim is narrower: interaction is
handled reactively at dispatch time by an operator, and is not priced into
membership.

## 5. What we can and cannot say

**Defensible:**

> Membership is decided by shift factor, one line at a time. The MW split
> within a group is pro-rata by output and ignores effectiveness entirely.
> Ordering that split by effectiveness saves a median 18% of the wind cut in
> our model — up to 65%, and nothing at all where a group's farms are alike.

**Two limits that must be attached:**

1. **Our model has no line outages.** [WDT]'s groups are mostly built for
   post-fault conditions. We are computing a different quantity for a different
   situation, not a variant of theirs.
2. **One week of synthetic weather.** No hour in it ever happened. The set of
   congested branches is one draw and will move with real data. The PTDF
   matrices will not — they are topology and reactance only.

**Do not say:** "we invented net sensitivity" (standard practice), "this is new
to Ireland" (unsupported), "no approximations" (DC model drops voltage,
reactive power and losses), or the earlier 52% / 77% figures (both were
computed with installed capacity instead of actual output — superseded by §3).

## 6. Open questions [HACK] itself asks

- **3.2** *"does there need to exist one group for each problematic line"*
- **3.3** *"Is there an optimal strategy to ensure a safe and effective
  incentive structure is enforced?"* — this is the §3 result, and the fairness
  tension in §2 is the reason it is not simply "use merit order".
