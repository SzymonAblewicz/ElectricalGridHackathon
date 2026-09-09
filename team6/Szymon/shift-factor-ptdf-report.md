# Shift Factor / PTDF — Computation Pipeline

**Author:** Szymon Ablewicz
**Team 6 — Electrical Grid Hackathon 2026**
**Scope:** intact (base-case) network only. No outages, no open circuits, no contingencies.

---

## Purpose

We need a number that answers one question:

> If we reduce power generation at wind node `n` by 1 MW, how much does congestion on transmission line `e` change?

This is the quantity EirGrid calls a **shift factor** and the literature calls a **PTDF** (Power Transfer Distribution Factor). This report specifies exactly how we compute it, why each step exists, and which steps are optional.

---

# The pipeline

## 1. Build the network matrices

```
K = incidence          buses × branches,  +1 at from, −1 at to
B = diag(1/x)          branches × branches
L = K B Kᵀ             buses × buses
```

`K` encodes the topology, `B` encodes how easily each line carries power. Together they give `L`, the conductance matrix of the whole grid — so `Lθ = p` is just KCL written at every bus.

Per-unit reactances are fine; PTDF is invariant to any global scaling of `B`.

## 2. Take the Moore–Penrose pseudoinverse

```
L⁺ = pinv(L)           np.linalg.pinv / scipy.linalg.pinv
```

`L` is singular: for a connected network its nullspace is spanned by the all-ones vector, since adding a constant to every angle changes nothing. So it has no true inverse.

Moore–Penrose is the specific choice that returns the **minimum-norm solution orthogonal to that nullspace** — meaning `θ = L⁺p` has angles summing to zero. Physically, that's balancing the injected MW uniformly across every bus rather than at one arbitrary slack.

Only expensive operation here, and you do it once.

## 3. Compute the PTDF

```
PTDF = B Kᵀ L⁺         branches × buses, dimensionless
```

`L⁺` turns injections into angles; `B Kᵀ` turns angles into branch flows via Ohm's law. Composed, they take injections straight to flows. This matrix is the metric.

## 4. Correct the sign direction

```
f = PTDF @ p                              net injection per bus, sums to ≈ 0
sensitivity = PTDF[e,:] * sign(f[e])
```

`from`/`to` orientation is arbitrary — whatever order the buses appeared in your data. So on any line where that labelling runs against the actual flow, the entire sensitivity list comes out inverted, which reverses the ranking and hands you the wrong wind farms.

Multiplying by `sign(f[e])` re-points the axis along the real flow direction. Then **positive means curtailing there relieves congestion**, matching EirGrid.

---

**Debug check, 2 lines, run once:** `np.allclose(f, n.lpf_flows)`. Catches a flipped orientation in `K`, the one bug that otherwise fails silently. Produces nothing you keep.

**Conditional — reference subtraction:** `PTDF[e,:] − PTDF[e,ref]`. Only if you're matching someone else's numbers, since it just shifts every node by the same constant and can't change your ranking. You'd need it against a kit-provided PTDF table, a library cross-check (MATPOWER's `makePTDF` and friends default to single-slack, not distributed), or EirGrid's own figures, which balance at a remote conventional generator.

---
---

# Detailed explanation

Everything below expands the four stages above. The pipeline itself does not change.

---

## 0. The model we are working in: DC power flow

Before Stage 1 makes sense, it is worth being explicit about the approximation, because every property we exploit later depends on it.

The **DC power flow** approximation linearises the AC power flow equations under three assumptions [1, 2]:

1. All bus voltage magnitudes are held at 1 per-unit.
2. Branch resistance is negligible relative to reactance (`r << x`), so losses are ignored.
3. Voltage angle differences across branches are small, so `sin(θᵢ − θⱼ) ≈ θᵢ − θⱼ`.

Under these, active power flow on a branch becomes exactly linear in the angle difference:

```
f_e = (θ_from(e) − θ_to(e)) / x_e
```

**Why this matters for us:** linearity is the whole reason we get a closed-form sensitivity matrix. In full AC we would have to perturb an injection and re-solve a nonlinear system for every wind node — which is precisely what EirGrid does in their load-flow tool. In DC, the derivative is exact and available analytically, once, for every line-node pair simultaneously.

**What we give up.** Stott, Jardim and Alsaç [2] give the standard critical assessment of DC power flow accuracy and its failure modes. The two that matter here:

- **No reactive power.** Real thermal limits are current limits, and current tracks apparent power `|S| = √(P² + Q²)`. DC only sees `P`. On a congested transmission circuit `P` dominates, so directions are reliable and magnitudes are approximate.
- **No voltage-stability representation.** A number of real EirGrid constraint groups exist for voltage-stability reasons rather than thermal ones [3]. DC cannot represent these at all. They are out of scope for this pipeline and should be excluded explicitly rather than given a number.

**Circuit-theory reading.** Under DC, the grid is an ordinary resistive network and the whole thing is standard nodal analysis:

| Circuit theory | DC power flow |
|---|---|
| node voltage | bus voltage angle `θ` |
| conductance `G = 1/R` | susceptance `1/x` |
| injected current source | net power injection `p` (MW) |
| nodal conductance matrix **G** | `L = K B Kᵀ` |
| **G**·V = I  (KCL at every node) | `L θ = p` |
| Ohm's law on a branch | `f = B Kᵀ θ` |

This is the classical bus-admittance / nodal formulation described in any power systems text [1, 4].

---

## Stage 1 — Build the network matrices

### 1.1 What `K` is

`K` is the **bus–branch incidence matrix**, dimensions `n_bus × n_branch`. Column `e` has exactly two non-zeros:

- `+1` in the row of the branch's `from` bus
- `−1` in the row of the branch's `to` bus

This encodes topology only — which buses connect to which. It carries no electrical information.

**The orientation is arbitrary.** Which bus we call `from` and which we call `to` is decided by whatever order they appeared in the data file. This is harmless for Stages 1–3 but is exactly the thing Stage 4 has to clean up. Flag it now so it is not a surprise later.

### 1.2 What `B` is

`B = diag(1/x)`, dimensions `n_branch × n_branch`, diagonal. Each entry is the branch susceptance — how readily that line carries power for a given angle difference. Low reactance = electrically "short" = carries more.

This is where the electrical properties of the network enter. `K` says what is connected; `B` says how strongly.

### 1.3 What `L` is, and why it is the conductance matrix

```
L = K B Kᵀ
```

This is the **weighted graph Laplacian** of the network, weighted by susceptance. Equivalently, in power-systems language, it is the DC bus admittance matrix (`Bbus` in MATPOWER's notation [5]).

Its structure is the standard nodal matrix:

- Off-diagonal `L[i,j] = −1/x` for a branch between `i` and `j` (0 if not connected)
- Diagonal `L[i,i] = Σ 1/x` over all branches incident on bus `i`

Applying it to the angle vector gives:

```
L θ = p
```

which is **Kirchhoff's Current Law written at every bus simultaneously**: net power injected at a bus equals the sum of power flowing out of it along its branches. Verifying this by hand on a 3-bus example is the fastest way to convince yourself the matrices are built correctly.

### 1.4 Why per-unit reactances are fine — the scale-invariance argument

**Claim:** PTDF is unchanged if you multiply every susceptance by any constant `c`.

**Proof.** Let `B → cB`. Then:

- `L = K(cB)Kᵀ = cL`
- `L⁺ = (cL)⁺ = L⁺/c`  (a Moore–Penrose property of scalar multiples)
- `PTDF = (cB)Kᵀ(L⁺/c) = B Kᵀ L⁺` — the `c` cancels. ∎

**Practical consequence:** you do not need an MVA base, and you do not need to convert per-unit reactances into MW/radian. Feed the reactances in whatever consistent units the data provides. This eliminates a common and hard-to-spot source of error.

**Important caveat — this does *not* extend to the angles.** From the same algebra, `θ = L⁺p` scales as `θ → θ/c`. So:

- **`PTDF` and `f` are always correct**, regardless of `B`'s units.
- **`θ` is only physically meaningful in radians if `B` is genuinely in MW/radian** and `p` in MW.

We never use `θ` directly in this pipeline, so this costs us nothing. But do not print the angles and expect them to be in radians unless you have been careful with units.

---

## Stage 2 — Take the Moore–Penrose pseudoinverse

### 2.1 Why `L` cannot be inverted

**Claim:** `L·1 = 0`, where `1` is the all-ones vector.

**Proof.** Each column of `K` contains exactly one `+1` and one `−1`. Therefore every column sums to zero, so `Kᵀ1 = 0`. Then:

```
L·1 = K B Kᵀ·1 = K B (Kᵀ·1) = K B · 0 = 0     ∎
```

**Physical meaning.** Adding the same constant to every bus angle changes no angle *difference*, so it changes no flow. The absolute angle reference is unobservable. In circuit terms: we have not grounded any node, so node voltages are only defined up to a common offset.

**Consequence:** `L` is singular and `np.linalg.inv(L)` will either fail or return numerical garbage. This is a structural property of the problem, not a data defect.

**Standard result on the nullspace dimension.** For a graph Laplacian, the multiplicity of the zero eigenvalue equals the number of connected components [6]. So:

- Connected network → nullity exactly 1 → nullspace = span{**1**}, as stated in the pipeline.
- **If the network model has islands, nullity > 1** and the nullspace is spanned by the indicator vectors of each component.

> **Practical check for the kit data.** Confirm the network is connected before trusting anything downstream. If the model contains isolated buses or disconnected sub-networks, the "uniform slack" interpretation in §2.3 breaks — balancing would be spread across buses that are electrically unreachable. Check the number of near-zero eigenvalues of `L`, or run a connected-components pass on the graph.

### 2.2 What Moore–Penrose specifically gives us

The Moore–Penrose pseudoinverse `L⁺` is the unique matrix satisfying the four Penrose conditions [7]. `pinv` in both NumPy and SciPy computes it via SVD.

For our symmetric singular `L`, the property we actually rely on is:

```
range(L⁺) = range(Lᵀ) = range(L) = null(L)⊥
```

So `θ = L⁺p` is guaranteed orthogonal to the nullspace, i.e. `1ᵀθ = 0` — **the bus angles sum to zero.** Among the infinitely many valid angle vectors (all differing by a constant offset), Moore–Penrose picks the one centred on zero.

This is why we use `pinv` rather than the textbook alternative of deleting the slack bus row and column and inverting the remainder. Both are valid; `pinv` avoids having to nominate an arbitrary slack bus, and gives a convention that is symmetric across the network.

### 2.3 Why this amounts to *uniform distributed slack*

This is the claim in the pipeline that most deserves a proof, since it defines what our numbers mean.

**Claim.** The PTDF column produced by `L⁺` for bus `n` is exactly the flow pattern from injecting 1 MW at `n` and withdrawing `1/N` MW at each of the `N` buses.

**Proof.** That balanced injection is `p = eₙ − (1/N)·1`. Since `1 ∈ null(L)` and `L` is symmetric, `1 ∈ null(L⁺)`, so `L⁺·1 = 0`. Therefore:

```
L⁺p = L⁺eₙ − (1/N)·L⁺·1 = L⁺eₙ − 0 = L⁺eₙ     ∎
```

So `L⁺eₙ` — the raw `n`-th column, which corresponds to an *unbalanced* unit injection — produces precisely the same flows as the *balanced* uniform-slack injection. The pseudoinverse silently supplies the uniform withdrawal for us.

**This is the answer to "where does the balancing MW go?"** Every sensitivity has to specify this, because power must balance: if a wind farm drops 1 MW, something absorbs it. Our convention is "uniformly across all buses." EirGrid's convention is "at a remote conventional generator" [3]. They are different conventions and will give different numbers — see the Conditional stage, §C, for when that matters.

### 2.4 Why `p` must sum to zero

`p` is only in `range(L)` when `1ᵀp = 0` (since `range(L) = null(L)⊥`). If injections do not balance:

- `L⁺p` still returns something — the **least-squares** solution, not an exact one.
- The imbalance is absorbed silently, with **no error and no warning.**

> Assert `abs(p.sum()) < tol` before computing flows in Stage 4. This is a two-line guard against a failure mode that otherwise produces plausible-looking wrong numbers.

### 2.5 Cost

`pinv` is an SVD, so `O(n³)` in the number of buses. It is the only expensive operation in the pipeline, it is done **once**, and the resulting matrix is reused for every line, every wind node, and every dispatch snapshot. Everything after Stage 2 is matrix multiplication and indexing.

---

## Stage 3 — Compute the PTDF

```
PTDF = B Kᵀ L⁺
```

Dimensions: `(n_branch × n_branch)(n_branch × n_bus)(n_bus × n_bus)` → **`n_branch × n_bus`**.

### 3.1 Reading it as two composed steps

```
PTDF  =   B Kᵀ    ·    L⁺
        [angles      [injections
         → flows]     → angles]
```

- `L⁺` solves the network: given injections, produce bus angles. This is the nodal analysis solve.
- `B Kᵀ` reads flows off the angles: `Kᵀθ` gives the angle *difference* across each branch, and multiplying by `1/x` applies Ohm's law. This is exactly `f_e = (θ_from − θ_to)/x_e`.

Pre-multiplying them means we never solve the network again. One matrix takes injections straight to flows, for any injection vector.

### 3.2 What an entry means

`PTDF[e, n]` = **change in active power flow on branch `e` (MW), per MW injected at bus `n`**, with the balancing withdrawal distributed uniformly.

It is **dimensionless** — MW per MW. A value of `0.20` means 20% of the injected power appears on that branch.

**Signs are meaningful and both occur.** A negative entry simply means injection at `n` pushes flow in the `to → from` direction along `e`. It is not an error and it does not yet mean "makes congestion worse" — that interpretation requires Stage 4.

### 3.3 Why this is exact, and why not to use finite differences

Because DC power flow is linear, `PTDF[e,n]` is the *exact analytic derivative* `∂f_e/∂p_n` of the model — not an estimate. Every entry for every line-node pair falls out of the single inversion in Stage 2.

The alternative — perturb one wind node by 10 MW, re-solve, take the difference, divide — is what EirGrid does in their AC load-flow tool [3], and is necessary there because AC is nonlinear. In DC it would be strictly worse: same answer, but contaminated by floating-point cancellation, and requiring one re-solve per wind node instead of one inversion total.

### 3.4 Relation to the terminology

"PTDF", "shift factor", "injection shift factor" (ISF) and "generation shift factor" (GSF) all name the same underlying sensitivity [1, 5]. The only substantive difference between them is the balancing convention discussed in §2.3 — where the compensating withdrawal is assumed to occur. A shift factor is a PTDF with the withdrawal point left implicit. Textbooks and European sources tend to say PTDF; North American system operators tend to say shift factor.

---

## Stage 4 — Correct the sign direction

### 4.1 The problem

Two quantities live on the *same arbitrary axis* defined by the `from`/`to` labelling in `K`:

- `f[e]` — positive means power flows `from → to`
- `PTDF[e,:]` — change in flow measured on that same axis

That axis was set by data-file row order. It has no physical meaning.

Congestion, however, is about **magnitude** — `|f|` against the thermal rating. So:

| Base flow on `e` | Overload means | Relief requires |
|---|---|---|
| positive (`from → to`) | `f` too large **positive** | `PTDF[e,n] > 0` |
| negative (`to → from`) | `f` too large **negative** | `PTDF[e,n] < 0` |

**So the sign that indicates "relieves congestion" flips depending on an arbitrary labelling choice.**

### 4.2 The fix

```
f = PTDF @ p
sensitivity = PTDF[e,:] * sign(f[e])
```

- `f[e] > 0` → axis already runs along the real flow → `sign = +1`, row unchanged.
- `f[e] < 0` → axis runs backwards → `sign = −1`, row flipped.

The result is a sensitivity measured on an axis that **always points along the actual direction of flow.**

### 4.3 Verification that this matches EirGrid's convention

EirGrid's stated convention: positive shift factor means reducing generation at that node relieves congestion; negative means it worsens it [3].

Check ours. Let `S = PTDF[e,n]·sign(f[e])`. Reducing generation by `r > 0` MW at `n` is an injection change of `−r`, so the change in the flow-aligned quantity is `−S·r`.

- If `S > 0`: change is `−S·r < 0` → flow shrinks along its own direction → magnitude drops → **congestion relieved.** ✓
- If `S < 0`: change is positive → flow grows along its own direction → magnitude rises → **congestion worsened.** ✓

Both match. The convention is correct.

**Intuition:** think of the change in flow as a small vector along the line. Positive `S` means the change points *against* the existing flow (magnitude falls). Negative `S` means it points *with* it (magnitude rises).

### 4.4 Why this is not cosmetic

Multiplying a row by `−1` **reverses the ranking**. The wind farms that appeared most effective become the least effective and vice versa.

Since the `from`/`to` labelling is essentially random with respect to flow direction, roughly **half of all monitored lines** will be affected. And there is no way to detect this by inspecting the numbers — a sign-flipped list looks entirely plausible.

Skipping Stage 4 does not give slightly-off numbers. It gives a completely wrong constraint group on half the lines.

### 4.5 Why `p` is needed here

Stages 1–3 are pure network properties — they depend only on topology and reactance, and are valid for every hour of the year. Stage 4 is the first point that requires a **dispatch snapshot**: actual net injections per bus for the hour being studied.

This is unavoidable. Flow direction depends on what is actually generating and consuming, not on the wires alone.

`p` must satisfy `p.sum() ≈ 0` (see §2.4).

### 4.6 Edge case: `f[e] ≈ 0`

`sign(0)` returns `0` in NumPy, which would zero out the entire sensitivity row silently.

Guard it. In practice a line carrying no flow is not congested and should not be in the monitored set, so this should not arise for lines we care about — but a bare `np.sign` on a full flow vector will hit it.

### 4.7 Linearity caveat

Under the linear model, curtailing far enough drives the flow through zero and it then grows in the opposite direction. "Positive `S` = relief" therefore holds only up to that crossing point.

Not a practical concern at the curtailment magnitudes involved, but it is why the statement is not unconditionally true.

---

# Optional stages

## D. Debug check — 2 lines, run once, then delete

```python
assert np.allclose(f, n.lpf_flows, atol=1e-9)
```

**This is not a pipeline stage.** It produces no output that is used anywhere downstream.

**What it does.** It compares two independent routes to the same DC flows:

1. Our analytic route: `f = PTDF @ p`
2. An independent DC power flow solver — e.g. PyPSA's `n.lpf()` [8]

Both should agree to ~1e-10 MW, since branch flows are independent of slack convention given balanced injections.

**What it catches.** Almost exclusively: a flipped or wrong orientation in `K`. That is the single most common construction error and it fails *silently* — you get a full, plausible-looking, entirely wrong matrix.

**What it does NOT establish.** This is an internal arithmetic-consistency check between two DC methods. It says nothing about:

- whether DC is a good approximation of the real AC system,
- whether the network model matches the real Irish grid,
- whether our numbers agree with EirGrid's.

Be precise about this in any writeup. It is a correctness check on our implementation, not a validation against reality.

**Cost:** two lines, once, at the start. Then forget it.

## C. Conditional — reference subtraction

```
S = PTDF[e,:] − PTDF[e,ref]
```

### C.1 Why it is usually unnecessary

This subtracts **the same constant from every wind node** in the row. Therefore it:

- cannot change the ranking of wind nodes,
- cannot change which nodes clear a threshold,
- cannot change the resulting constraint group.

Our uniform-slack numbers from Stage 3 are already a valid, internally consistent sensitivity. **Default: skip this stage.**

### C.2 What it actually does

Subtracting the column for bus `ref` converts our distributed-slack ISF into the **single-slack ISF with slack at `ref`** — equivalently, the two-point transfer factor for "inject at `n`, withdraw at `ref`".

This difference is *slack-independent*: any change of balancing convention adds a common term to every column, which cancels in the subtraction. That is why a column difference is the standard way to move between conventions [1, 5].

### C.3 When you DO need it

Only when comparing our numbers against **someone else's numbers**, because then we must match their balancing convention:

| Comparison target | What to set `ref` to |
|---|---|
| A kit-provided PTDF or shift-factor table | Whatever slack bus the kit documents |
| A library cross-check | Its slack bus. Most implementations — e.g. MATPOWER's `makePTDF` [5] — take a slack argument and default to **single-slack**, not distributed |
| EirGrid's published figures [3] | A large conventional generator bus, since their studies balance at a remote conventional generator |

If we are only ranking wind farms within our own model, this stage does nothing at all.

---

# Deliberately excluded from this pipeline

| Excluded | Reason |
|---|---|
| **Rating normalisation** (`sensitivity / rating_e`) | Only needed to compare across *different* lines, or to feed a multi-constraint LP. Irrelevant to computing a single line's sensitivity. |
| **LODF / post-contingency** | Base case only, by scope decision. Real network limits usually bind post-trip (N−1), so this is the most likely future extension. Roughly 10 lines on top of the same PTDF matrix. |
| **A/MW unit conversion** | Presentation only. `A/MW ≈ dimensionless × 1000/(√3·V_kV)`. Add it *only* if a comparison target is in amps. Never compute with it — it is not comparable across voltage levels, and it assumes zero reactive flow, which DC cannot verify. |

---

# Summary of what is and is not required

| Stage | Status | One-line reason |
|---|---|---|
| 1. Build `K`, `B`, `L` | Required | The network |
| 2. `L⁺ = pinv(L)` | Required | `L` is singular; fixes the angle reference |
| 3. `PTDF = B Kᵀ L⁺` | Required | This is the metric |
| 4. Sign correction | Required | Otherwise ~half the lines rank backwards |
| D. Debug check | Optional, 2 lines | Catches silent `K` orientation bugs |
| C. Reference subtraction | Conditional | Only to match an external convention |

---

# References

**[1]** Wood, A. J., Wollenberg, B. F., Sheblé, G. B. — *Power Generation, Operation, and Control*, 3rd ed., Wiley, 2013. Standard reference for DC power flow, shift factors, and the PTDF/ISF/GSF terminology.

**[2]** Stott, B., Jardim, J., Alsaç, O. — "DC Power Flow Revisited", *IEEE Transactions on Power Systems*, 24(3), 2009, pp. 1290–1300. Critical assessment of DC power flow assumptions, accuracy and failure modes.

**[3]** EirGrid / SONI — *Wind Dispatch Tool Constraint Group Overview*, v1, 1 February 2024. Primary TSO operational document. Defines the shift factor as `SF = Δf_e / ΔP_n` (change in post-contingency current per MW reduced), computed offline by 10 MW perturbation in a load-flow tool with balancing at a remote conventional generator; sign convention positive = relieves. Also the source for the existence of voltage-stability-driven constraint groups.
https://cms.eirgrid.ie/sites/default/files/publications/Wind-Dispatch-Tool-Constraint-Group-Overview_0.pdf

**[4]** Grainger, J. J., Stevenson, W. D. — *Power System Analysis*, McGraw-Hill, 1994. Bus admittance matrix and nodal formulation.

**[5]** Zimmerman, R. D., Murillo-Sánchez, C. E., Thomas, R. J. — "MATPOWER: Steady-State Operations, Planning, and Analysis Tools for Power Systems Research and Education", *IEEE Transactions on Power Systems*, 26(1), 2011, pp. 12–19. Reference implementation of `makePTDF` and `Bbus`; documents the slack-bus argument.

**[6]** Godsil, C., Royle, G. — *Algebraic Graph Theory*, Springer, 2001. Graph Laplacian; multiplicity of the zero eigenvalue equals the number of connected components.

**[7]** Penrose, R. — "A generalized inverse for matrices", *Mathematical Proceedings of the Cambridge Philosophical Society*, 51(3), 1955, pp. 406–413. Defines the pseudoinverse and its four characterising conditions.

**[8]** Brown, T., Hörsch, J., Schlachtberger, D. — "PyPSA: Python for Power System Analysis", *Journal of Open Research Software*, 6(4), 2018. Source of the independent `lpf()` DC solver used in the debug check.

---

## Citation confidence — read this

For honesty about what has and has not been checked:

- **The derivations in §1.4, §2.1, §2.3, §2.4 and §4.3 are proved inline in this document.** They do not depend on any citation being accurate — check the algebra directly.
- **References [1], [2], [4], [5], [6], [7], [8]** are cited at the level of author / title / venue / year. They are standard, widely-cited works and the attributions are reliable, but **specific page numbers, equation numbers, and exact wording have not been verified against the sources in preparing this document.** Verify before quoting any of them directly in a submission.
- **Reference [3] (EirGrid WDT document)** — the specific claims attributed to it here (shift factor definition, 10 MW perturbation, remote conventional generator balancing, sign convention, voltage-stability groups) come from prior research notes, not from re-reading the PDF while writing this. The URL is included. **Verify directly before citing in a submission**, particularly the balancing-generator detail, which drives the Conditional stage §C.3.
- **No claim in this document asserts agreement between our numbers and EirGrid's published figures.** We have not established that, and the thresholds EirGrid uses for constraint group membership are not public [3], so it may not be establishable.
