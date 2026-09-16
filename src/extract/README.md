# `extract`

Extractors: given an e-graph and its roots, choose one node per reachable
e-class. Every extractor implements

```rust
pub trait Extractor: Sync {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult;
}
```

and is registered by name in `extractors()` (`src/lib.rs`), which the CLI
(`src/main.rs`) and the tests (`src/test.rs`) run.

## Start here

| You want | Use |
|---|---|
| An optimal extraction with a solver (size, depth, size under a depth budget, weighted) | [`ilp/`](ilp/README.md) |
| A fast extraction with no solver | [`greedy/`](greedy/mod.rs): `GreedySizeExtractor`, `GreedyDepthExtractor` |
| The initial expression back out of an e-graph | `greedy::InitialExtractor` |

## Node weights

Each node of an `egraph_serialize::EGraph` carries:

| Field | Meaning | Used as |
|---|---|---|
| `cost` | size of the node | **size** (area when area-weighted) |
| `delay` | depth of the node | **depth** (delay when delay-weighted) |
| `initial` | node belongs to the initial expression | warm start / fallback |

## Results

| Type | File | What |
|---|---|---|
| `ExtractionResult` | `mod.rs` | `class -> chosen node`; may be incomplete or cyclic |
| `CompleteExtraction` | `mod.rs` | an `ExtractionResult` validated for its roots (complete, acyclic); safe to cost |

`ExtractionResult::validate` / `is_complete` say why a result is invalid;
`check` panics instead. `dag_cost` sums shared nodes once (size); `dag_depth`
is the critical path (depth); `tree_cost` counts shared nodes every time.

## Files

| File | Extractor | Optimal for | Status |
|---|---|---|---|
| `ilp/` | `SizeExtractor`, `DepthExtractor`, `SizeConstrainedDepthExtractor`, `WeightedSizeDepthExtractor` | DAG objective (given time) | **current** |
| `greedy/` | `GreedySizeExtractor`, `GreedyDepthExtractor`, `InitialExtractor` | depth (exact); size heuristic | **current** |
| `bottom_up.rs`, `faster_bottom_up.rs`, `prio_queue.rs` | tree-cost extractors | tree cost | upstream |
| `faster_greedy_dag.rs` | `FasterGreedyDagExtractor` (= `GreedySizeExtractor`) | none (heuristic) | upstream |
| `greedy_dag.rs`, `global_greedy_dag.rs` | older DAG heuristics | none | upstream, unregistered |
| `ilp_cbc.rs` | `IlpExtractor`, `DualIlpExtractor`, `DelayBudgetIlpExtractor` | | **deprecated**, adapters over `ilp::solve` |
| `faster_ilp_cbc.rs` | `FasterIlpExtractor` (simplified model, lazy cycle blocking) | | **deprecated** |
| `warm.rs` | re-exports under the old `extract::warm` paths | | **deprecated** paths |

The ILP extractors need a MILP backend feature (`ilp-cbc`, `ilp-gurobi` or
`ilp-highs`); see [`../milp/README.md`](../milp/README.md).
