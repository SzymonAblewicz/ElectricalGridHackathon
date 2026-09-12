# Plan: higher-order Cheeger multicut of the flow graph (approved 2026-09-11)

**New file:** `team6/Oleksandr/flow/cheeger_multicut.py`. No edits to existing files.

Basis: Lee, Oveis Gharan & Trevisan, "Multiway spectral partitioning and higher-order Cheeger
inequalities", JACM 2014 (arXiv:1111.1055), Theorem 1.1: lambda_k/2 <= rho_G(k) <= O(k^2) sqrt(lambda_k).

1. **Graph** - fiedler_bisection's `build_flow` / `weighted_graph` (constants pushed in); SNAPSHOT
   155, K = 9, headroom and loading each run separately.
2. **Embedding** - bottom k eigenpairs of L_sym (`spectrum.spectrum`), F = D^-1/2 U, mass d|F|^2.
3. **Regions** - deterministic stand-in for the paper's random partition: k centres by mass-weighted
   farthest point in the radial distance d_F, nearest-centre cells.
4. **Localise** - h_i = |F| min(1, margin/EPS) inside cell i (taper to 0 at the cell edge), so
   supports are disjoint.
5. **Sweep** - per cell, lowest cut/vol prefix in the full graph -> k disjoint cores.
6. **Leftovers** - absorbed layer by layer into the core with most edge weight.
7. **Output** - `flow/cheeger/`: graph PDF + npz per weighting; clusters (with `core` flag), sets,
   corridors, embedding npz, maps per run; printed certificate lambda_k/2 vs max phi (cores, final).

Decisions: higher-order Cheeger (not recursive, not k-selection); leftovers assigned (full
partition); deterministic centres rather than random partitions with restarts; no upper bound
printed (the O(k^2) constant is not given).
