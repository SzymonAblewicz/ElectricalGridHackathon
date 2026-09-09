# Shift Factor / PTDF — Computation Pipeline

**Author:** Szymon Ablewicz
**Team 6 — Electrical Grid Hackathon 2026**
**Scope:** intact (base-case) network only. No outages, no open circuits, no contingencies.

> **Revision note — Stage 4 only.** Stages 1, 2 and 3 are unchanged from the original version of this
> document. Stage 4 has been corrected to carry the **phase-shifting transformer term**, which the
> original omitted. The correction is derived in §4.2, verified numerically in §4.9, and the original
> form is kept in full in **Appendix A** with an explicit statement of when it is still valid.
> Nothing else in the pipeline moves. `PTDF` itself is *identical* under both versions — the
> phase-shift term does not enter it.

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
φ = phase shifts       branches, radians (zero for every ordinary line)
L = K B Kᵀ             buses × buses
```

`K` encodes the topology, `B` encodes how easily each line carries power. Together they give `L`, the conductance matrix of the whole grid — so `Lθ = p` is just KCL written at every bus.

Per-unit reactances are fine for the PTDF; it is invariant to any global scaling of `B`. `φ` is zero for every line and every ordinary transformer, and non-zero only for phase shifters — it is not used until Stage 4, and there `B` must be in true MW/radian (§1.5).

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
p_eff = p + K B φ                         net injection per bus, sums to ≈ 0
f     = PTDF @ p_eff − B φ                actual branch flows, MW
sensitivity = PTDF[e,:] * sign(f[e])
```

`from`/`to` orientation is arbitrary — whatever order the buses appeared in your data. So on any line where that labelling runs against the actual flow, the entire sensitivity list comes out inverted, which reverses the ranking and hands you the wrong wind farms.

Multiplying by `sign(f[e])` re-points the axis along the real flow direction. Then **positive means curtailing there relieves congestion**, matching EirGrid.

A phase shifter imposes an angle of its own, so it acts as a pair of equal-and-opposite injections. That is why it enters as a modified injection vector `p_eff` and a constant `−Bφ`, and why `PTDF` is untouched by it. With no shifters in the network `φ = 0` and this collapses to `f = PTDF @ p`.

---

**Debug check, 2 lines, run once:** `np.allclose(f, n.lpf_flows)`. Catches a flipped orientation in `K` and a dropped phase-shift term — the two bugs that otherwise fail silently. Produces nothing you keep.

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

**One extension to this base model.** The relations above assume no branch imposes an angle of its own. A **phase-shifting transformer** does, which generalises the flow equation to `f_e = b_e(θᵢ − θⱼ − φ_e)` and the bus equation to `Lθ = p + K B φ`. This is introduced in §1.5 and handled in §4.2. It leaves `PTDF` completely unchanged (§4.4), which is why it does not appear until Stage 4. On a network with no phase shifters the two forms are identical.

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

**Practical consequence:** for **the PTDF** you do not need an MVA base, and you do not need to convert per-unit reactances into MW/radian. Feed the reactances in whatever consistent units the data provides. This eliminates a common and hard-to-spot source of error.

**Caveat 1 — this does *not* extend to the angles.** From the same algebra, `θ = L⁺p` scales as `θ → θ/c`. So:

- **`PTDF` is always correct**, regardless of `B`'s units.
- **`θ` is only physically meaningful in radians if `B` is genuinely in MW/radian** and `p` in MW.

We never use `θ` directly in this pipeline, so this costs us nothing. But do not print the angles and expect them to be in radians unless you have been careful with units.

**Caveat 2 — and this one does cost us. Scale invariance does *not* extend to Stage 4 when phase shifters are present.** Stage 4 forms `p_eff = p + K B φ`, which **adds** a `B`-weighted term to a vector in MW. Under `B → cB` that term scales but `p` does not, so the sum is not proportional to anything and the resulting flows are simply wrong.

Concretely:

| Quantity | `B` in arbitrary units? |
|---|---|
| `PTDF = B Kᵀ L⁺` | Fine — `c` cancels (proved above) |
| `f = PTDF @ p` (no shifters, `φ = 0`) | Fine — no `B`-weighted additive term |
| `p_eff = p + K B φ` and `f = PTDF @ p_eff − B φ` | **`B` must be in genuine MW/radian** |

So the rule is: Stages 1–3 tolerate any scaling of `B`; **Stage 4 does not, once `φ ≠ 0`.** If you build `B` yourself from raw per-unit reactances, `b_e = S_base / x_pu,e` with `S_base` the MVA base of the per-unit system.

**On the participant kit specifically:** PyPSA's `x_pu_eff` is per-unit on a **1 MVA base**, so `1/x_pu_eff` is *already* in MW/radian and no conversion is needed. This was checked directly against the raw data rather than assumed: for line `1021-2121-1`, `x = 7.634011 Ω` at `v_nom = 110 kV` gives `x_pu_eff · v_nom² / x = 1.000 MVA` exactly. See §4.9.

### 1.5 What `φ` is

`φ` is the vector of **phase-shift angles**, one per branch, in radians. It is zero for every ordinary line and every ordinary transformer, and non-zero only for **phase-shifting transformers** — devices that deliberately impose a fixed angle across themselves in order to push power onto or off a particular route.

For a branch with a phase shift, DC flow is not `b(θᵢ − θⱼ)` but

```
f_e = b_e (θᵢ − θⱼ − φ_e)
```

Nothing in Stages 1–3 uses `φ`; it enters only at Stage 4, and §4.2 shows exactly how and why `PTDF` is unaffected.

**Where to get it.** In PyPSA it is `transformers["phase_shift"]`, stored in **degrees** — convert with `np.radians`. Lines have no such column; treat them as zero.

**Do not assume it is zero.** The kit's `WP2033_all-island` network has two phase-shifting transformers, one of them at **17°**. Ignoring them puts the flow on the most heavily loaded circuit in the network out by **78.8 MW** (§4.9).

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

### 4.2 Getting `f` right: the phase-shift term

To take `sign(f[e])` we first need `f[e]`, and this is the one place in the pipeline where the naive expression is not good enough.

**The model.** A phase-shifting transformer imposes a fixed angle `φ_e` across itself, so its flow is

```
f_e = b_e (θᵢ − θⱼ − φ_e)          i.e.   f = B(Kᵀθ − φ)
```

**Propagate it to the bus equation.** KCL still says `K f = p` — injections must equal the net flow out of each bus. Substituting:

```
K B (Kᵀθ − φ) = p
K B Kᵀ θ      = p + K B φ
L θ           = p + K B φ
```

So the shifter appears on the right-hand side, in exactly the position an injection occupies. **That is the physical content: a phase shifter behaves as a pair of equal-and-opposite injections at its two ends.** It is not a new kind of object in the model.

**Collapse it into one substitution.** Define

```
p_eff = p + K B φ
```

Then the entire Stage 2–3 machinery applies unchanged:

```
θ = L⁺ p_eff
f = B Kᵀ θ − B φ
  = (B Kᵀ L⁺) p_eff − B φ
  = PTDF · p_eff − B φ
```

Same `L⁺`. Same `PTDF`. Same single matrix multiply. The shifter costs one modified input vector and one constant subtraction — **no new operator and no change to Stages 1–3.**

**Three properties that confirm this is the right form:**

**(i) `p_eff` still sums to zero.** Every column of `K` has one `+1` and one `−1`, so `Kᵀ1 = 0`, hence

```
1ᵀ p_eff = 1ᵀp + 1ᵀK B φ = 0 + (Kᵀ1)ᵀ B φ = 0
```

So the balance precondition of §2.4 carries over **unchanged**. Nothing new to check, and the pseudoinverse is still being handed a vector in `range(L)`. Verified numerically at `1ᵀp_eff = 0.000e+00` (§4.9).

**(ii) It is an affine offset, not a different model.** Expanding the substitution back out:

```
f = PTDF·p + (PTDF·K − I) B φ
  = PTDF·p + c
```

where `c = (B Kᵀ L⁺ K − I) B φ` **does not depend on `p` at all**. So the original `f = PTDF @ p` is not a different model — it is this model with the constant `c` discarded. `c` is computed once and reused for every snapshot. (Both forms agree to `1.7e-13 MW`; §4.9.)

**(iii) `φ = 0` recovers the original exactly.** Every line and every ordinary transformer has `φ_e = 0`, so on a network with no phase shifters `p_eff = p`, `Bφ = 0`, and this reduces term-by-term to `f = PTDF @ p`. See Appendix A.

### 4.3 The fix

```
p_eff = p + K B φ
f     = PTDF @ p_eff − B φ
sensitivity = PTDF[e,:] * sign(f[e])
```

- `f[e] > 0` → axis already runs along the real flow → `sign = +1`, row unchanged.
- `f[e] < 0` → axis runs backwards → `sign = −1`, row flipped.

The result is a sensitivity measured on an axis that **always points along the actual direction of flow.**

### 4.4 Why `PTDF` is untouched by any of this

Worth stating plainly, because it is the reason the correction is confined to Stage 4:

```
∂f/∂p = ∂(PTDF·p_eff − Bφ)/∂p = PTDF · ∂p_eff/∂p = PTDF · I = PTDF
```

`φ` is a **fixed device setting**, not a function of `p`. Differentiating with respect to injections kills it. Both the `K B φ` term and the `− B φ` term are constants and vanish.

**Consequence: the shift factor for every edge `e` and every wind node `n` is numerically identical whether or not you model phase shifters.** The correction does not improve the sensitivity values — they were already exact. It fixes the *base flows*, and therefore the *orientation* those sensitivities are reported on, and therefore which lines you identify as congested in the first place.

### 4.5 Verification that this matches EirGrid's convention

EirGrid's stated convention: positive shift factor means reducing generation at that node relieves congestion; negative means it worsens it [3].

Check ours. Let `S = PTDF[e,n]·sign(f[e])`. Reducing generation by `r > 0` MW at `n` is an injection change of `−r`, so the change in the flow-aligned quantity is `−S·r`.

- If `S > 0`: change is `−S·r < 0` → flow shrinks along its own direction → magnitude drops → **congestion relieved.** ✓
- If `S < 0`: change is positive → flow grows along its own direction → magnitude rises → **congestion worsened.** ✓

Both match. The convention is correct.

**Intuition:** think of the change in flow as a small vector along the line. Positive `S` means the change points *against* the existing flow (magnitude falls). Negative `S` means it points *with* it (magnitude rises).

### 4.6 Why the sign correction is not cosmetic

Multiplying a row by `−1` **reverses the ranking**. The wind farms that appeared most effective become the least effective and vice versa.

Since the `from`/`to` labelling is essentially random with respect to flow direction, roughly **half of all monitored lines** will be affected. And there is no way to detect this by inspecting the numbers — a sign-flipped list looks entirely plausible.

Skipping Stage 4 does not give slightly-off numbers. It gives a completely wrong constraint group on half the lines.

### 4.7 Why `p` is needed here

Stages 1–3 are pure network properties — they depend only on topology and reactance, and are valid for every hour of the year. Stage 4 is the first point that requires a **dispatch snapshot**: actual net injections per bus for the hour being studied.

This is unavoidable. Flow direction depends on what is actually generating and consuming, not on the wires alone.

`p` must satisfy `p.sum() ≈ 0` (see §2.4). As shown in §4.2(i), `p_eff` inherits this automatically — adding the phase-shift term introduces no new balance condition.

### 4.8 Edge cases

**`f[e] ≈ 0`.** `sign(0)` returns `0` in NumPy, which would zero out the entire sensitivity row silently.

Guard it. In practice a line carrying no flow is not congested and should not be in the monitored set, so this should not arise for lines we care about — but a bare `np.sign` on a full flow vector will hit it. On the kit's network, 6 branches sit close enough to zero flow that their sign is numerical noise (§4.9) — they are all lightly loaded and none is a monitoring candidate, but they are there.

**Linearity caveat.** Under the linear model, curtailing far enough drives the flow through zero and it then grows in the opposite direction. "Positive `S` = relief" therefore holds only up to that crossing point.

Not a practical concern at the curtailment magnitudes involved, but it is why the statement is not unconditionally true.

### 4.9 Numerical verification on the kit's network

Everything in §4.2 was checked directly against the participant kit's `WP2033_all-island` network (754 buses, 980 DC branches, 2 phase shifters at 3° and 17°) rather than asserted. Flows were compared against **PyPSA's own `n.lpf()` DC solver** [8], which is an independent implementation.

**Setup:** dispatch solved with HiGHS, frozen into `p_set`, `n.lpf()` run, injections taken from the resulting generator/load/link powers.

| Check | Result |
|---|---|
| `1ᵀ p` (balance precondition, §2.4) | `0.000e+00` |
| `1ᵀ p_eff` (balance preserved, §4.2(i)) | `0.000e+00` |
| `f = PTDF·p_eff − Bφ` vs `n.lpf()` | **`1.7e-09 MW`** ✔ |
| `f = PTDF·p` vs `n.lpf()` (Appendix A form) | **`7.88e+01 MW`** ✘ |
| `PTDF·p_eff − Bφ` vs `PTDF·p + c` (§4.2(ii)) | `1.7e-13 MW` — the two algebraic forms agree |
| PyPSA per-unit base, from `x_pu_eff·v_nom²/x` | `1.000 MVA` exactly, so `1/x_pu_eff` is already MW/rad (§1.4) |

**What omitting the term actually costs.** Not a uniform small error — a badly skewed one:

- **272 of 980 branches** wrong by more than 1 MW; **63** wrong by more than 10 MW.
- **13 branches (1.3%) get the opposite sign of `f`**, so their entire sensitivity row is flipped and their ranking reversed. A further 6 differ only at `|f| < 1 MW`, which is the §4.8 numerical-zero case rather than a genuine flip.
- Worst single error: **78.8 MW**.

**And the worst error lands on the worst possible branch.** Branch `3581-89516-1` — the Northern Ireland tie, and the kit's own documented example of a circuit worth studying:

| | flow | rating | loading |
|---|---|---|---|
| Correct (`PTDF·p_eff − Bφ`) | 123.00 MW | 123 MVA | **100.0 %** |
| Omitting the shifter term | 44.16 MW | 123 MVA | 35.9 % |

The two transformers in series with it (`T89510-89515-1`, `T89515-89516-1`) show the same 78.8 MW error at 98.4 % loading.

This is worse than a sign error. Under the original Stage 4 this circuit is **not congested at all** — it is at a third of its rating, would never be selected as a monitored line, and would never have a constraint group computed for it. The single binding constraint in the network disappears.

That is the concrete justification for the revision: not accuracy of the sensitivity numbers, which are unchanged (§4.4), but **correctly identifying which line is congested and which way its flow runs.**

---

# Optional stages

## D. Debug check — 2 lines, run once, then delete

```python
assert np.allclose(f, n.lpf_flows, atol=1e-9)
```

**This is not a pipeline stage.** It produces no output that is used anywhere downstream.

**What it does.** It compares two independent routes to the same DC flows:

1. Our analytic route: `f = PTDF @ p_eff − B φ`
2. An independent DC power flow solver — e.g. PyPSA's `n.lpf()` [8]

Both should agree to ~1e-9 MW, since branch flows are independent of slack convention given balanced injections.

**What it catches.** Two things, both of which fail *silently* and leave you with a full, plausible-looking, entirely wrong matrix:

1. **A flipped or wrong orientation in `K`** — the single most common construction error.
2. **A dropped or mis-signed phase-shift term.** This is why the check earns its place rather than being a formality: on the kit's network it is the difference between `1.7e-09 MW` and `7.9e+01 MW` (§4.9). If your residual comes back at tens of MW rather than ~1e-9, `φ` is the first thing to look at, and the branches with the largest residuals will be the ones sitting next to the shifters.

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
| 1. Build `K`, `B`, `φ`, `L` | Required | The network |
| 2. `L⁺ = pinv(L)` | Required | `L` is singular; fixes the angle reference |
| 3. `PTDF = B Kᵀ L⁺` | Required | This is the metric |
| 4. Sign correction, via `p_eff` | Required | Otherwise ~half the lines rank backwards, and congested lines are missed |
| D. Debug check | Optional, 2 lines | Catches silent `K` orientation and dropped-`φ` bugs |
| C. Reference subtraction | Conditional | Only to match an external convention |

**Where the snapshot dependence lives.** Stages 1–3 depend only on topology and reactance and are valid for every hour of the year: one `PTDF` matrix, computed once, `n_branch × n_bus`. Stage 4 is the only snapshot-dependent step, because `sign(f[e])` depends on what is generating and consuming that hour. So for the full every-edge × every-wind-node computation you get **one PTDF block for all time, and one signed sensitivity matrix per snapshot.** Pick the snapshot deliberately — peak loading on the monitored circuit is the usual choice — and state which one you used.

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

**[8]** Brown, T., Hörsch, J., Schlachtberger, D. — "PyPSA: Python for Power System Analysis", *Journal of Open Research Software*, 6(4), 2018. Source of the independent `lpf()` DC solver used in the debug check and in §4.9.

**[9]** Hackathon participant kit — `participant-kit/flowmath.py` (functions `branches`, `incidence`, `laplacian`, `pseudoinverse`, `ptdf`, `injections`, `angles`, `flows`) and `examples/d_ptdf.py`, `examples/f_shift_factors.py`. An independent implementation of Stages 1–3 identical to ours, and the source of the phase-shift treatment adopted in Stage 4. Its docstring for `angles()` states the same relations derived in §4.2: `F_e = b_e(θᵢ − θⱼ − φ_e)` and `Lθ = p + K B φ`. Local copy: `team6/Szymon/participant-kit new/`, unmodified control copy at `team6/Szymon/participant kit original (from the repo)/`.

**[10]** Network data — `TYTFS2024` / `WP2033_all-island`, as shipped in the participant kit (754 buses, 755 lines, 225 transformers, 929 generators, 168 hourly snapshots). Derived by the kit authors from EirGrid's Ten Year Transmission Forecast Statement 2024 study files. **All time series in it are synthetic**, and its `s_nom` ratings are TYTFS `RATE1` planning values, not operational limits — see the kit's own `README.md` LIMITATIONS section before quoting any absolute number from it.

---

## Citation confidence — read this

For honesty about what has and has not been checked:

- **The derivations in §1.4, §2.1, §2.3, §2.4, §4.2, §4.4 and §4.5 are proved inline in this document.** They do not depend on any citation being accurate — check the algebra directly.
- **§4.9 was executed, not asserted.** The numbers in it (`1.7e-09` vs `7.88e+01 MW`, 272 / 63 / 13 branch counts, the 100.0 % vs 35.9 % loading on `3581-89516-1`, and the 1.000 MVA per-unit base) come from running the computation on the kit's `WP2033_all-island` network and comparing against PyPSA's independent `n.lpf()` solver [8]. They are reproducible from the kit copy in this folder. Note they are for **one snapshot** of a solved dispatch — the exact figures move between hours, the conclusion does not.
- **References [1], [2], [4], [5], [6], [7], [8]** are cited at the level of author / title / venue / year. They are standard, widely-cited works and the attributions are reliable, but **specific page numbers, equation numbers, and exact wording have not been verified against the sources in preparing this document.** Verify before quoting any of them directly in a submission.
- **Reference [3] (EirGrid WDT document)** — the specific claims attributed to it here (shift factor definition, 10 MW perturbation, remote conventional generator balancing, sign convention, voltage-stability groups) come from prior research notes, not from re-reading the PDF while writing this. The URL is included. **Verify directly before citing in a submission**, particularly the balancing-generator detail, which drives the Conditional stage §C.3.
- **No claim in this document asserts agreement between our numbers and EirGrid's published figures.** We have not established that, and the thresholds EirGrid uses for constraint group membership are not public [3], so it may not be establishable.

---
---

# Appendix A — the original Stage 4 (no phase-shift term)

This is the Stage 4 this pipeline originally specified. It is kept because it is **not wrong** — it is a special case, it is correct whenever its precondition holds, and it is what you should reach for on any network without phase shifters.

## A.1 The original form

```
f = PTDF @ p                              net injection per bus, sums to ≈ 0
sensitivity = PTDF[e,:] * sign(f[e])
```

## A.2 Exactly when it is valid

**It is exact if and only if `φ_e = 0` for every branch** — no phase-shifting transformers anywhere in the network.

That is not a rare condition. Ordinary lines never have a phase shift, ordinary transformers never have one, and many test networks and regional subsets contain no shifters at all. On such a network `p_eff = p`, `Bφ = 0`, and §4.2 reduces term-by-term to the expression above. The two forms are then **identical, not approximately equal**.

Check it in one line before relying on it:

```python
assert (np.abs(phi) < 1e-9).all(), "network has phase shifters — use the p_eff form"
```

## A.3 What goes wrong when it is not valid

The omitted quantity is `c = (B Kᵀ L⁺ K − I) B φ`, a constant vector independent of `p` (§4.2(ii)). It does not scale down with anything and it does not average out. On the kit's `WP2033_all-island` network (§4.9):

- 272 of 980 branches wrong by more than 1 MW, 63 by more than 10 MW, worst 78.8 MW
- 13 branches get the **opposite sign of `f`**, flipping their whole sensitivity row
- the most heavily loaded circuit in the network, `3581-89516-1`, reads **35.9 % loaded instead of 100 %** — it stops looking congested at all

The failure is concentrated, not diffuse: it is worst on exactly the branches electrically closest to the shifters, which on this network happen to include the binding constraint.

## A.4 Why it is still worth keeping in this document

**Three reasons, in order of practical value:**

1. **It is the right form on a shifter-free network.** Simpler, one fewer vector to build, one fewer place to get a sign or a unit wrong. If `φ = 0` everywhere, use it.

2. **It does not require `B` in physical units.** This is a genuine advantage. The `p_eff` form adds a `B`-weighted term to a vector in MW, so it needs `B` in true MW/radian (§1.4, Caveat 2). The original form is fully scale-invariant — `f = PTDF @ p` inherits `PTDF`'s immunity to `B → cB`, so you can feed it raw per-unit reactances with no MVA base at all and still get correct flows in MW. If you are working from a data source whose per-unit base you cannot establish, and the network has no shifters, this form removes an entire class of error.

3. **It isolates what the correction actually did.** `PTDF` is bit-for-bit identical under both forms (§4.4). Running the two side by side and differencing shows the phase-shift contribution in isolation, which is a clean way to demonstrate that the sensitivity values never changed and only the base flows did.

## A.5 Summary

| | Original (A.1) | Corrected (§4.3) |
|---|---|---|
| Expression | `f = PTDF @ p` | `f = PTDF @ p_eff − Bφ` |
| `PTDF` values | identical | identical |
| Exact when | `φ = 0` everywhere | always |
| Needs `B` in MW/rad | no | **yes** |
| On `WP2033_all-island` | 78.8 MW error, misses the binding constraint | matches `n.lpf()` to 1.7e-09 MW |

**Default: use the corrected form.** It is exact in both cases, and the only thing it asks of you is that `B` be in real units — which, on the participant kit, it already is.
