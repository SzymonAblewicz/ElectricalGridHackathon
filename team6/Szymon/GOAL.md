# The goal, and how far the current work is from it

Written 2026-09-10. Read this before adding anything — it exists so the work
does not drift away from what it is for.

---

## The goal, in Szymon's words

> "the final goal is to just create constraint groups (and create them better
> than eirgrid) and be able to later run a simulation where i track how good my
> constraint groups truly are at relieving congestion in practice — and if i
> automate it in a live 24 hours scenario where i give some windy weather at
> each wind node to simulate weather throughout the day, then i can measure
> power generation at each wind node and current flow throughout the day,
> measuring congestion, dispatch down rate and occurrences, and how congestion
> is relieved throughout the day."

Plus three things said later:

- **Real EirGrid data** for generation and wind speed, not synthetic profiles.
- **The group-selection method is the contribution.** "I will create constraint
  groups in a more complex way" — the point is to *improve the selection
  process*, not to reproduce EirGrid's.
- The overlap idea: a node with a **moderate** shift factor on several congested
  branches may beat one with a **high** factor on a single branch.

---

## Where the work actually is

| goal | status |
|---|---|
| sensitivity for every wind node × every branch | **done** — 980 × 157, verified to 3.9e-09 MW against an independent solver |
| overlap scoring (breadth vs depth) | **done** — `node_scores.csv`, 9 figures, live in the viewer |
| **create constraint groups** | **not started** — no selection algorithm exists |
| **simulate and measure the groups** | **not started** |
| real EirGrid weather and generation | **not started** — everything uses the kit's synthetic week |
| better than EirGrid | **not testable as stated** (see below) |

**What exists is the ingredients, not the dish.** Sensitivities, scores and
tooling are built and checked. Nothing yet selects a group, applies it, or
measures whether it worked.

---

## The five real deviations

**1. No constraint groups yet.** Thresholding `|sensitivity| >= 0.05` is shown in
the figures as an *illustration*, not as a method. The "more complex way" is
still to be designed. This is the actual contribution and it is unwritten.

**2. No simulation.** The 24-hour tracking loop — weather in, dispatch, flows,
congestion, dispatch-down, relief — does not exist. The pieces are all present
(`gridkit.solve`, `line_loading`, `dispatch_down`, hourly `p_max_pu`), and a full
168-hour solve takes 59 seconds, so this is feasible, not done.

**3. Synthetic weather.** Every hourly number here comes from the kit's generated
week — a spatially-correlated Gaussian wind field. **No hour in it ever
happened.** Consequences: the set of 10 congested branches is one draw from one
week and will change with real data, and every dispatch-down figure is
model-internal.
*Not affected:* `ptdf/` is topology and reactance only and will not change at all.

**4. "Better than EirGrid" cannot be tested as stated.** Their constraint-group
thresholds are not public, so there is nothing to compare against. The
defensible substitute is **your selection method vs a baseline method inside your
own model**, same network, same weather, same everything else. That is a real
result; a comparison against EirGrid's published curtailment figures is not.

**5. Base case only — and this one bites.** EirGrid's own definition of
effectiveness is measured against **N-1 and N-1-1 overloads**, not the intact
network. The current work is intact-network only. Claiming an improvement on
their method while ignoring the contingencies that method is built around is a
gap that has to be either closed or stated plainly.

---

## One scope change that was not a deviation

The model carries **754 nodes and 980 edges**, not "wind nodes and transmission
lines". That is necessary physics, not scope creep — and it paid off: **4 of the
10 congested branches are transformers**, including the second-worst constraint
in the network. A lines-only model would have missed 40% of the targets.

The deliverable is still small: **157 wind nodes × 10 congested branches**.

---

## The next two pieces, in order

1. **A group-selection method** that uses the overlap scoring, and a baseline to
   compare it against.
2. **The 24-hour loop** that applies both and measures dispatch-down and relief.

Real EirGrid data can be substituted at any point after that — it changes the
signs and the congested set, and nothing in `ptdf/`.
