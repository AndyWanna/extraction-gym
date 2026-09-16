# `extract::ilp`

Optimal DAG extraction as a mixed-integer program, with a time limit, a warm
start, and a guaranteed-complete result.

## Using it

```rust
use extraction_gym::extract::ilp::*;

// egraph: nodes carry cost (size), delay (depth) and initial flags
let options = IlpOptions::default()                  // 10 s, WarmStart::Initial, 1 thread
    .with_time_limit(Duration::from_secs(60));
let outcome = SizeConstrainedDepthExtractor { depth_budget, options }
    .solve(&egraph, &roots)?;                        // Err only for an invalid request

let extraction = outcome.extraction;                 // CompleteExtraction
match outcome.report.outcome {                       // SolveReport
    SolveOutcome::Optimal => {}
    SolveOutcome::Incumbent => {}                    // time limit hit, solver's best
    SolveOutcome::Fallback { reason } => {}          // solver failed; see below
}
```

`solve` runs on the compiled-in backend; `solve_with::<M>()` pins one. Each
extractor also implements `Extractor` (panicking on an invalid request).

## The four extractors

| Extractor | Objective (`IlpObjective`) | With area / delay weights |
|---|---|---|
| `SizeExtractor` | min size | min area |
| `DepthExtractor` | min depth | min delay (prefer `greedy::GreedyDepthExtractor`: exact and fast) |
| `SizeConstrainedDepthExtractor { depth_budget }` | min size s.t. depth <= budget | min area without worsening delay |
| `WeightedSizeDepthExtractor { size_weight, depth_weight }` | min `size_weight*size + depth_weight*depth` | area/delay trade-off |

## Warm start and fallback

Two candidate extractions are computed before every solve:

* **initial**: `greedy::InitialExtractor`, from the `Node::initial` flags
* **greedy**: `GreedyDepthExtractor` for `Depth` and `SizeConstrainedDepth`,
  otherwise `GreedySizeExtractor`

The **fallback** is the better of the two under the objective. With a depth
budget, candidates within budget are compared by size; if none is, the
shallowest is returned with `report.depth_budget_violated = true`.

| `WarmStart` | Seed |
|---|---|
| `None` | no MIP start |
| `Initial` (default) | the initial candidate; error if nothing is flagged |
| `Greedy` | the fallback |

The fallback is returned when the solver produces no solution in time, the
budget is infeasible, the solution is invalid, a timed-out incumbent is worse
than the fallback, or the backend panics. The reason is in
`SolveOutcome::Fallback { reason }`.

## Flow of `solve` (`solve.rs`)

```text
validate request ──> candidates (fallback.rs) ──> pick fallback + seed
        │
        └─> run_solver: model::build ─> apply_seed (warm.rs) ─> solve ─> decode ─> CompleteExtraction
                                                                                │
                              Optimal / Incumbent <── ok ──┤── none/invalid/worse/panic ──> Fallback
```

## Files

| File | Contents |
|---|---|
| `mod.rs` | the four extractors and re-exports |
| `objective.rs` | `IlpObjective`, the size/depth vocabulary, `evaluate`, the greedy pairing |
| `options.rs` | `IlpOptions`, `WarmStart`, `DEFAULT_TIME_LIMIT` |
| `solve.rs` | `solve`, `IlpOutcome`, `IlpError`; the only place the fallback is decided |
| `fallback.rs` | the candidates and the rule for picking between them |
| `model.rs` | the one MILP model: selection, depth (arrival time) and cycle-blocking rows, objective |
| `warm.rs` | pushing a seed into every model column; `arrival_times` for the budget check |
| `decode.rs` | reading the selection back out of a solution |
| `report.rs` | `SolveReport`, `SolveOutcome` |
| `tests.rs` | the guarantees above, on random e-graphs |

## The model, briefly

Binary `x_n` per node and `a_c` per class; `sum x_n = a_c`, a selected node
selects its child classes, roots are selected. Cycles are blocked exactly with
a topological level per class, so any feasible solution is a valid
extraction. Depth objectives add an arrival time `T_c >= depth_n + T_child`
per class (big-M on `x_n`); a budget caps every `T_c`. Details in `model.rs`.
