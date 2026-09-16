# `milp`

A solver-agnostic MILP interface, so the ILP extractors in
[`extract::ilp`](../extract/ilp/README.md) build one model and run it on CBC,
Gurobi or HiGHS.

## Choosing a backend

Exactly one Cargo feature, fixed at build time (two at once is a
`compile_error!`):

| Feature | Backend | Warm start | Threads | Notes |
|---|---|---|---|---|
| `ilp-cbc` | COIN-OR CBC (`coin_cbc`) | refused (untrustworthy) | 1 | behavioural baseline; needs libCbc installed |
| `ilp-gurobi` | Gurobi (`grb`, Gurobi 10 ABI) | yes | yes | needs `GUROBI_HOME`, `GRB_LICENSE_FILE` and the Gurobi lib dir on `LD_LIBRARY_PATH` at build **and** run time; one licence token per thread |
| `ilp-highs` | HiGHS (`highs-sys` FFI) | yes | yes | vendors and cmake-builds HiGHS (~4 min cold); needs `cmake` |

`DefaultMilp` is the enabled backend. Everything ILP-related is behind the
internal `ilp-any` feature, which each backend enables; never enable it
directly. egraphrl pins `ilp-gurobi` in its `Cargo.toml`.

## The interface (`mod.rs`)

| Item | What |
|---|---|
| `MilpModel` | build columns/rows/objective, set time limit, threads, log file, raw parameters and MIP start; `solve()` keeps the model live |
| `MilpSolution` | owned solution: column values, objective, bound, gap, and the predicates `has_solution` / `ran_to_completion` / `is_infeasible` |
| `MilpStatus` | `Optimal`, `Infeasible`, `TimeoutWithSolution`, `TimeoutNoSolution`, derived from the predicates |
| `conformance` | shared backend tests: time limit respected, rows added to a live model, warm start does not break the answer |

Branch on the predicates, not on `MilpStatus`, when the distinction matters:
CBC reports a proven-infeasible model as finished.

## Files

| File | Contents |
|---|---|
| `mod.rs` | the traits, backend selection, conformance tests |
| `cbc.rs` | CBC backend; delegates call-for-call to `coin_cbc` |
| `gurobi.rs` | Gurobi backend; licensing notes at the top |
| `highs.rs` | HiGHS backend; overrides the MIP gap defaults to match CBC |
| `stage.rs` | buffer that pushes new columns/rows to Gurobi/HiGHS in bulk at `solve()` (per-coefficient edits on a live model are quadratic) |
| `trajectory.rs` | parses Gurobi/HiGHS logs into an incumbent-vs-time trajectory for `SolveReport` |

## Running the backend tests

The Gurobi conformance tests are `#[ignore]`d because each test thread takes a
licence token:

```bash
cargo test --release --features ilp-gurobi -- --ignored --test-threads=1 milp::gurobi
```
