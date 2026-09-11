# Plan: local conductance clustering of the flow graph (approved 2026-09-11)

**New file:** `team6/Oleksandr/flow/local_conductance.py`. No edits to `get_flow_graph.py`,
`cluster.py` or `graph_lib.py`.

1. **Graph** - the exact flow-weighted graph `get_flow_graph.main()` builds (dispatch ->
   rebalance -> lpf -> headroom/loading/inverse_flow -> `gwg.adjacency`), by importing its
   helpers and pushing this script's constants into their module globals.
2. **Local clusters** - personalised PageRank from a seed (exact, one sparse `splu` solve per
   round for all seeds), sweep over `p/d`, keep the minimum-conductance prefix,
   `phi(S) = cut(S) / min(vol S, vol V\S)`, vectorised.
3. **Partition by peeling** - K-1 rounds: sweep from every unassigned bus on the unassigned
   subgraph, keep the lowest-phi set subject to `MIN_BUSES` and `MAX_VOLUME_FRACTION`,
   label it, remove it. Leaves orphaned by the removal ride with the set. Remainder = cluster K.
   Optional `PHI_MAX` stops early.
4. **Output** - `data/graphs/<case>/flow/local/`: clusters/corridors CSV, overview PDF,
   per-cluster maps (reusing cluster.py / cluster_maps.py), plus `conductance_<run>.csv`.

Decisions: full partition (not seeded single clusters); exact PPR rather than ACL push;
size floor counted in buses (default 5) so a bus with its generators is not a "cluster";
defaults as get_flow_graph (kit, WP2033_all-island, headroom, hour 155, K = 9).
