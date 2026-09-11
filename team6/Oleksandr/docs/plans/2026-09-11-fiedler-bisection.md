# Plan: recursive Fiedler bisection of the flow graph (approved 2026-09-11)

**New file:** `team6/Oleksandr/flow/fiedler_bisection.py`. No edits to existing files.

1. **Graph** - `get_flow_graph`'s power flow run once (dispatch -> rebalance -> lpf), then weighted
   by `headroom` and by `loading`, each separately, through `gfg.branch_weights` /
   `gfg.generator_weights` -> `gwg.adjacency`.
2. **Bisection** - start with one cluster; K-1 times: lambda_2 of L_sym for every cluster
   (cached, only the two changed clusters are recomputed), cut the cluster with the smallest one.
   The cut is a sweep along the random-walk Fiedler vector u_2 / sqrt(d), best prefix under the
   cut function. A disconnected cluster (lambda_2 = 0) is split along its components.
3. **Cut functions** - `ncut` (cut/vol S + cut/vol S') and `ratiocut` (cut/|S| + cut/|S'|), each
   run separately: 2 weightings x 2 cuts = 4 partitions.
4. **Output** - `data/graphs/<case>/flow/fiedler/`: per weighting the full graph via
   `gwg.plot` + `gwg.save`; per run clusters/corridors CSV, overview PDF, per-cluster maps,
   `fiedler_<run>.csv` (the Fiedler vector used at every cut) and `steps_<run>.csv`.

Decisions: "better" = smallest lambda_2 of L_sym; sweep cut rather than sign/median split;
RatioCut as "the usual" cut beside Ncut; defaults as local_conductance (kit, WP2033_all-island,
hour 155, K = 9).
