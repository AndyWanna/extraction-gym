//! Gurobi backend for [`MilpModel`], via the `grb` crate (Gurobi 10 ABI).
//!
//! # Licensing
//!
//! `grb` keeps one *thread-local* `GRBenv` ([`grb::Env::DEFAULT_ENV`]) and every
//! [`grb::Model`] created on that thread is a child of it. One thread that
//! touches Gurobi therefore consumes exactly one license token, no matter how
//! many models it builds, and the token is held until the thread exits.
//!
//! Consequences under a single-token licence:
//!
//! * The extraction flow is single-threaded, so a normal run takes one token.
//! * `cargo test` runs test functions on N worker threads. If more than one of
//!   them constructs a `GurobiModel`, that is N tokens. Run Gurobi tests with
//!   `--test-threads=1`, and never run two Gurobi processes at once.
//!
//! There is no way to fix this from inside this file: `grb::Env` holds an `Rc`
//! and is `!Send`, so a single process-wide environment cannot legally be shared
//! between threads.
//!
//! Creating the default environment also writes a `gurobi.log` into the current
//! working directory (that filename is hard-coded by `grb`).
//!
//! # Why the staging buffer
//!
//! See [`super::stage`]. The short version for Gurobi specifically: there is no
//! such thing as a free (`-inf <= row <= +inf`) constraint in the Gurobi C API —
//! a row is created with a sense and a right-hand side — but the trait says
//! [`MilpModel::add_row`] creates an unconstrained row and the bounds arrive in
//! a later call. So rows are buffered and only classified (`=` / `<=` / `>=` /
//! ranged) when they are pushed to the solver at the next `solve()`.
//!
//! Everything pushed stays in the live `GRBmodel`; re-solves after the
//! cycle-breaking loop adds rows are true incremental re-solves and Gurobi keeps
//! its incumbent and its cut pool.
//!
//! # Warm start
//!
//! [`MilpModel::set_col_initial_solution`] writes Gurobi's `Start` variable
//! attribute. Confirmed empirically on a 60-item strongly-correlated knapsack —
//! with the start seeded from a previous solution the Gurobi log prints
//!
//! ```text
//! Loaded user MIP start with objective -15119
//! ```

use super::stage::{snap_integral, Staged};
use super::{MilpModel, MilpSense, MilpSolution};

use grb::prelude::*;
use grb::parameter::Parameter;
use grb::{attr, param};
use grb::constr::{IneqExpr, RangeExpr};
use grb::expr::{Expr, LinExpr};

/// Objective reported when the solver has no incumbent. Mirrors CBC's `1e30`
/// sentinel so `solution.obj_value() > best_so_far` keeps working.
const NO_SOLUTION_OBJ: f64 = 1.0e30;

/// Right-hand side used for a row that is still `(-inf, +inf)` when it is
/// pushed. Gurobi cannot express "no constraint", so this is a vacuous
/// `expr <= 1e30`. Finite on purpose: Gurobi treats `|v| >= 1e100` as infinite
/// and rejects an infinite RHS. Never hit by the extractors in this repo (every
/// `add_row` is followed immediately by a bound call), and it warns loudly if it
/// ever is.
const VACUOUS_RHS: f64 = 1.0e30;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GurobiCol(u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct GurobiRow(u32);

/// How a staged row was encoded in the solver, so a later bound change can be
/// replayed with the right call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowKind {
    /// A plain `=` / `<=` / `>=` constraint: change `Sense`/`RHS` to retune.
    Simple,
    /// `GRBaddrangeconstr`, which materialises an auxiliary variable whose
    /// bounds *are* the row's bounds: change the aux var's LB/UB to retune.
    Ranged(Var),
}

struct RowRec {
    c: Constr,
    kind: RowKind,
}

/// Bound pattern of a staged row, decided at push time.
#[derive(Clone, Copy, Debug)]
enum RowCat {
    Eq(f64),
    Ge(f64),
    Le(f64),
    Range(f64, f64),
    Free,
}

fn categorize(lower: f64, upper: f64) -> RowCat {
    let lo_fin = lower.is_finite();
    let hi_fin = upper.is_finite();
    match (lo_fin, hi_fin) {
        (true, true) if lower == upper => RowCat::Eq(lower),
        (true, true) => RowCat::Range(lower, upper),
        (true, false) => RowCat::Ge(lower),
        (false, true) => RowCat::Le(upper),
        (false, false) => RowCat::Free,
    }
}

/// Warn on both `log` and stderr. `env_logger`'s default filter is `error`, so a
/// bare `log::warn!` would be invisible in exactly the situation where the user
/// most needs to see it (a parameter that silently did not apply).
fn complain(msg: &str) {
    log::warn!("{msg}");
    eprintln!("WARNING: {msg}");
}

pub struct GurobiModel {
    m: Model,
    s: Staged,
    vars: Vec<Var>,
    rows: Vec<RowRec>,
}

impl GurobiModel {
    fn flush(&mut self) {
        // 1. objective sense
        if self.s.sense_dirty() {
            let sense = match self.s.sense() {
                MilpSense::Minimize => ModelSense::Minimize,
                MilpSense::Maximize => ModelSense::Maximize,
            };
            self.m
                .set_attr(attr::ModelSense, sense)
                .expect("gurobi: set ModelSense");
        }

        // 2. new columns
        let n_cols = self.s.num_cols();
        let first_col = n_cols - self.s.num_new_cols();
        for i in first_col..n_cols {
            let c = *self.s.col(i as u32);
            let vtype = if c.integral {
                VarType::Integer
            } else {
                VarType::Continuous
            };
            // Binary columns are declared Integer with [0,1] bounds rather than
            // VarType::Binary: identical to the solver, and it means a later
            // bound change (the extractors pin roots with set_col_lower(.., 1.0))
            // does not contradict the declared type.
            let v = self
                .m
                .add_var(
                    "",
                    vtype,
                    c.cost,
                    c.lower,
                    c.upper,
                    std::iter::empty::<(Constr, f64)>(),
                )
                .expect("gurobi: add_var");
            self.vars.push(v);
        }
        debug_assert_eq!(self.vars.len(), n_cols);

        // 3. new rows, each created with its full coefficient list in one call
        //    (GRBaddconstr takes the nonzeros; poking them in one at a time
        //    afterwards would be O(nnz) per poke).
        let n_rows = self.s.num_rows();
        let first_row = n_rows - self.s.num_new_rows();
        for i in first_row..n_rows {
            let (cat, expr) = {
                let r = self.s.row(i as u32);
                let mut le = LinExpr::new();
                for (&col, &w) in r.coeffs.iter() {
                    le.add_term(w, self.vars[col as usize]);
                }
                (categorize(r.lower, r.upper), le)
            };
            let rec = match cat {
                RowCat::Eq(v) => self.add_simple(expr, ConstrSense::Equal, v),
                RowCat::Ge(v) => self.add_simple(expr, ConstrSense::Greater, v),
                RowCat::Le(v) => self.add_simple(expr, ConstrSense::Less, v),
                RowCat::Range(lo, hi) => {
                    let (aux, c) = self
                        .m
                        .add_range(
                            "",
                            RangeExpr {
                                expr: Expr::Linear(expr),
                                lb: lo,
                                ub: hi,
                            },
                        )
                        .expect("gurobi: add_range");
                    RowRec {
                        c,
                        kind: RowKind::Ranged(aux),
                    }
                }
                RowCat::Free => {
                    complain(&format!(
                        "gurobi: row {i} still has bounds (-inf, +inf) at solve time; \
                         Gurobi has no free constraint, encoding it as `<= {VACUOUS_RHS:e}`"
                    ));
                    self.add_simple(expr, ConstrSense::Less, VACUOUS_RHS)
                }
            };
            self.rows.push(rec);
        }
        debug_assert_eq!(self.rows.len(), n_rows);

        // Make the new objects real before touching them by attribute.
        self.m.update().expect("gurobi: update");

        // 4. bound / cost changes on already-pushed columns
        let dirty_cols = self.s.dirty_cols();
        if !dirty_cols.is_empty() {
            let mut lb = Vec::with_capacity(dirty_cols.len());
            let mut ub = Vec::with_capacity(dirty_cols.len());
            let mut ob = Vec::with_capacity(dirty_cols.len());
            let mut vt = Vec::with_capacity(dirty_cols.len());
            for &ci in &dirty_cols {
                let c = *self.s.col(ci);
                let v = self.vars[ci as usize];
                lb.push((v, c.lower));
                ub.push((v, c.upper));
                ob.push((v, c.cost));
                vt.push((
                    v,
                    if c.integral {
                        VarType::Integer
                    } else {
                        VarType::Continuous
                    },
                ));
            }
            self.m
                .set_obj_attr_batch(attr::LB, lb)
                .expect("gurobi: set LB");
            self.m
                .set_obj_attr_batch(attr::UB, ub)
                .expect("gurobi: set UB");
            self.m
                .set_obj_attr_batch(attr::Obj, ob)
                .expect("gurobi: set Obj");
            self.m
                .set_obj_attr_batch(attr::VType, vt)
                .expect("gurobi: set VType");
        }

        // 5. bound changes on already-pushed rows
        for ri in self.s.dirty_rows() {
            let (lower, upper) = {
                let r = self.s.row(ri);
                (r.lower, r.upper)
            };
            let cat = categorize(lower, upper);
            let rec = &self.rows[ri as usize];
            match (rec.kind, cat) {
                (RowKind::Simple, RowCat::Eq(v)) => self.retune(ri, ConstrSense::Equal, v),
                (RowKind::Simple, RowCat::Ge(v)) => self.retune(ri, ConstrSense::Greater, v),
                (RowKind::Simple, RowCat::Le(v)) => self.retune(ri, ConstrSense::Less, v),
                (RowKind::Simple, RowCat::Free) => {
                    self.retune(ri, ConstrSense::Less, VACUOUS_RHS)
                }
                (RowKind::Ranged(aux), RowCat::Range(lo, hi)) => {
                    self.m
                        .set_obj_attr(attr::LB, &aux, lo)
                        .expect("gurobi: set range LB");
                    self.m
                        .set_obj_attr(attr::UB, &aux, hi)
                        .expect("gurobi: set range UB");
                }
                // A row that was pushed one-sided and later becomes two-sided
                // (or vice versa) would need the constraint to be deleted and
                // re-added, which would renumber rows. No extractor in this repo
                // does that -- every add_row is bounded once, immediately, before
                // the first solve -- so refuse loudly instead of silently
                // solving a different model.
                (kind, cat) => panic!(
                    "gurobi: row {ri} changed bound category after being pushed to the solver \
                     ({kind:?} -> {cat:?}); this backend does not support that (it would \
                     require deleting and re-adding the constraint)"
                ),
            }
        }

        // 6. coefficient changes on already-pushed rows
        for (ri, ci, w) in self.s.dirty_coeffs() {
            let c = self.rows[ri as usize].c;
            let v = self.vars[ci as usize];
            self.m
                .set_coeff(&v, &c, w)
                .expect("gurobi: set_coeff");
        }

        // 7. MIP start. `mark_flushed` clears the staged map, so this fires only
        //    for the solve the seed was set for.
        if !self.s.warm().is_empty() {
            let pairs: Vec<(Var, f64)> = self
                .s
                .warm()
                .iter()
                .map(|(&ci, &v)| (self.vars[ci as usize], v))
                .collect();
            self.m
                .set_obj_attr_batch(attr::Start, pairs)
                .expect("gurobi: set Start");
        }

        self.m.update().expect("gurobi: update");
        self.s.mark_flushed();
    }

    fn add_simple(&mut self, expr: LinExpr, sense: ConstrSense, rhs: f64) -> RowRec {
        let c = self
            .m
            .add_constr(
                "",
                IneqExpr {
                    lhs: Expr::Linear(expr),
                    sense,
                    rhs: Expr::Constant(rhs),
                },
            )
            .expect("gurobi: add_constr");
        RowRec {
            c,
            kind: RowKind::Simple,
        }
    }

    fn retune(&mut self, ri: u32, sense: ConstrSense, rhs: f64) {
        let c = self.rows[ri as usize].c;
        self.m
            .set_obj_attr(attr::Sense, &c, sense)
            .expect("gurobi: set Sense");
        self.m
            .set_obj_attr(attr::RHS, &c, rhs)
            .expect("gurobi: set RHS");
    }
}

impl MilpModel for GurobiModel {
    const NAME: &'static str = "Gurobi";
    type Col = GurobiCol;
    type Row = GurobiRow;
    type Solution = GurobiSolution;

    fn new() -> Self {
        let mut m = Model::new("extraction-gym").expect(
            "gurobi: failed to create a model -- check GUROBI_HOME, GRB_LICENSE_FILE \
             and LD_LIBRARY_PATH, and that a licence token is available",
        );
        // Silence first, so the parameter changes below do not each echo a
        // "Set parameter ..." line. (The banner Gurobi prints when the
        // thread-local environment is created -- "Set parameter TokenServer" --
        // happens before any Rust code here runs and cannot be suppressed
        // without allocating a second environment, i.e. a second license
        // token.) CBC parity: `coin_cbc` starts quiet.
        m.set_param(param::OutputFlag, 0)
            .expect("gurobi: set OutputFlag");
        // CBC parity: CBC's default MIP gap is 0, Gurobi's is 1e-4 relative.
        // Leaving Gurobi's default on would let it stop 0.01% away from the
        // optimum and report Optimal, which breaks both the A/B objective
        // comparison against CBC and the extractors' own
        // `EPSILON_ALLOWANCE = 1e-5` assertions. Re-widen via
        // `set_raw_parameter("MIPGap", ..)` if a benchmark wants to.
        m.set_param(param::MIPGap, 0.0)
            .expect("gurobi: set MIPGap");
        m.set_param(param::MIPGapAbs, 0.0)
            .expect("gurobi: set MIPGapAbs");
        // CBC parity, and required under any CPU allocation: Gurobi's `Threads`
        // defaults to 0 = "use every core on the machine", which ignores the
        // allocation entirely, so one process can oversubscribe the whole box. CBC
        // is single-threaded, so pinning to 1 is also what makes the A/B
        // timing comparison honest. Widen via
        // `set_raw_parameter("Threads", "N")` only if that many CPUs are
        // actually allocated.
        m.set_param(param::Threads, 1)
            .expect("gurobi: set Threads");
        GurobiModel {
            m,
            s: Staged::new(),
            vars: Vec::new(),
            rows: Vec::new(),
        }
    }

    fn add_col(&mut self) -> Self::Col {
        GurobiCol(self.s.add_col())
    }

    fn add_binary(&mut self) -> Self::Col {
        GurobiCol(self.s.add_binary())
    }

    fn set_col_lower(&mut self, col: Self::Col, value: f64) {
        self.s.set_col_lower(col.0, value);
    }

    fn set_col_upper(&mut self, col: Self::Col, value: f64) {
        self.s.set_col_upper(col.0, value);
    }

    fn set_obj_coeff(&mut self, col: Self::Col, value: f64) {
        self.s.set_obj_coeff(col.0, value);
    }

    fn add_row(&mut self) -> Self::Row {
        GurobiRow(self.s.add_row())
    }

    fn set_row_lower(&mut self, row: Self::Row, value: f64) {
        self.s.set_row_lower(row.0, value);
    }

    fn set_row_upper(&mut self, row: Self::Row, value: f64) {
        self.s.set_row_upper(row.0, value);
    }

    fn set_row_equal(&mut self, row: Self::Row, value: f64) {
        self.s.set_row_equal(row.0, value);
    }

    fn set_weight(&mut self, row: Self::Row, col: Self::Col, weight: f64) {
        self.s.set_weight(row.0, col.0, weight);
    }

    fn set_obj_sense(&mut self, sense: MilpSense) {
        self.s.set_obj_sense(sense);
    }

    fn set_time_limit_seconds(&mut self, seconds: u32) {
        // Applied to the model's own environment immediately, so it is in force
        // for the very next solve (the cycle loop re-sets it every iteration
        // with the remaining budget).
        self.m
            .set_param(param::TimeLimit, f64::from(seconds))
            .expect("gurobi: set TimeLimit");
    }

    fn set_log_level(&mut self, level: u32) {
        self.m
            .set_param(param::OutputFlag, i32::from(level > 0))
            .expect("gurobi: set OutputFlag");
    }

    /// `LogFile` + `OutputFlag = 1` + `LogToConsole = 0`: the B&B progress
    /// table goes to `path` and nothing goes to stdout, so the benchmark's own
    /// output is unaffected while the trajectory becomes recoverable (see
    /// [`super::trajectory`]).
    fn set_log_file(&mut self, path: &str) {
        if let Err(e) = self.m.set_param(param::LogFile, path.to_string()) {
            complain(&format!("gurobi: could not set LogFile={path:?} ({e}); ignored"));
            return;
        }
        let _ = self.m.set_param(param::OutputFlag, 1);
        let _ = self.m.set_param(param::LogToConsole, 0);
    }

    /// Overrides the single-threaded default from [`MilpModel::set_threads`]:
    /// Gurobi's branch-and-bound is genuinely parallel. See the trait docs for
    /// the slot-allocation invariant this must respect.
    fn set_threads(&mut self, threads: u32) {
        // Gurobi's `Threads` is i32 and 0 means "all cores" -- exactly the
        // default we must never fall back into, so clamp 0 up to 1 rather than
        // passing it through.
        let n = i32::try_from(threads.max(1)).unwrap_or(i32::MAX);
        self.m
            .set_param(param::Threads, n)
            .expect("gurobi: set Threads");
    }

    fn set_raw_parameter(&mut self, key: &str, value: &str) {
        let p = match Parameter::new(key) {
            Ok(p) => p,
            Err(e) => {
                complain(&format!(
                    "gurobi: parameter name {key:?} is not a valid C string ({e}); ignored"
                ));
                return;
            }
        };
        // Gurobi has no "what type is this parameter" query, so probe it with a
        // get: only the correctly-typed getter succeeds.
        let as_int: grb::Result<i32> = self.m.get_param(&p);
        if as_int.is_ok() {
            match value.parse::<i32>() {
                Ok(v) => {
                    if let Err(e) = self.m.set_param(&p, v) {
                        complain(&format!("gurobi: setting {key}={value} failed ({e}); ignored"));
                    }
                }
                Err(_) => complain(&format!(
                    "gurobi: parameter {key} is an int but {value:?} is not an int; ignored"
                )),
            }
            return;
        }
        let as_dbl: grb::Result<f64> = self.m.get_param(&p);
        if as_dbl.is_ok() {
            match value.parse::<f64>() {
                Ok(v) => {
                    if let Err(e) = self.m.set_param(&p, v) {
                        complain(&format!("gurobi: setting {key}={value} failed ({e}); ignored"));
                    }
                }
                Err(_) => complain(&format!(
                    "gurobi: parameter {key} is a double but {value:?} is not a number; ignored"
                )),
            }
            return;
        }
        let as_str: grb::Result<String> = self.m.get_param(&p);
        if as_str.is_ok() {
            if let Err(e) = self.m.set_param(&p, value.to_string()) {
                complain(&format!("gurobi: setting {key}={value} failed ({e}); ignored"));
            }
            return;
        }
        complain(&format!(
            "gurobi: unknown parameter {key:?} (value {value:?}); ignored"
        ));
    }

    fn set_col_initial_solution(&mut self, col: Self::Col, value: f64) {
        self.s.set_warm(col.0, value);
    }

    fn set_initial_solution(&mut self, solution: &Self::Solution) {
        self.s.set_warm_all(
            solution
                .values
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v)),
        );
    }

    fn solve(&mut self) -> Self::Solution {
        self.flush();
        self.m.optimize().expect("gurobi: optimize");

        // `Model::status()` fails on statuses newer than the ones grb 3.0.1
        // knows about (WORK_LIMIT, MEM_LIMIT). Treat that as "stopped early",
        // which is what those statuses mean anyway.
        let status = self.m.status().ok();
        let sol_count: i32 = self.m.get_attr(attr::SolCount).unwrap_or(0);
        let has_solution = sol_count > 0;

        let obj_value = if has_solution {
            self.m.get_attr(attr::ObjVal).unwrap_or(NO_SOLUTION_OBJ)
        } else {
            NO_SOLUTION_OBJ
        };

        // Dual bound. Meaningless before any relaxation is solved, hence the
        // finiteness filter.
        let best_bound = self
            .m
            .get_attr(attr::ObjBound)
            .ok()
            .filter(|b: &f64| b.is_finite() && b.abs() < NO_SOLUTION_OBJ);

        let mut values = if has_solution {
            self.m
                .get_obj_attr_batch(attr::X, self.vars.iter().copied())
                .unwrap_or_else(|_| vec![0.0; self.vars.len()])
        } else {
            vec![0.0; self.vars.len()]
        };
        // See `stage::snap_integral`: Gurobi's `X` for an integer variable is
        // only integral to within IntFeasTol, and the extractors test it with a
        // bare `> 0.0`.
        snap_integral(&self.s, &mut values);

        if status == Some(Status::InfOrUnbd) {
            complain(
                "gurobi: model reported INF_OR_UNBD; treating it as infeasible. \
                 (The extraction models always have a bounded objective, so this \
                 is a presolve artefact -- set DualReductions=0 to disambiguate.)",
            );
        }

        GurobiSolution {
            values,
            obj_value,
            has_solution,
            status,
            best_bound,
        }
    }
}

pub struct GurobiSolution {
    values: Vec<f64>,
    obj_value: f64,
    has_solution: bool,
    status: Option<Status>,
    best_bound: Option<f64>,
}

impl MilpSolution for GurobiSolution {
    type Col = GurobiCol;

    fn col(&self, col: Self::Col) -> f64 {
        self.values[col.0 as usize]
    }

    fn obj_value(&self) -> f64 {
        self.obj_value
    }

    fn is_infeasible(&self) -> bool {
        matches!(self.status, Some(Status::Infeasible) | Some(Status::InfOrUnbd))
    }

    fn ran_to_completion(&self) -> bool {
        // Mirrors CBC's `Status::Finished`: the solver stopped because it was
        // done, not because a limit cut it off. A proven-infeasible model
        // counts as finished (the trait docs call this out explicitly).
        matches!(
            self.status,
            Some(Status::Optimal)
                | Some(Status::Infeasible)
                | Some(Status::InfOrUnbd)
                | Some(Status::Unbounded)
                | Some(Status::CutOff)
        )
    }

    fn has_solution(&self) -> bool {
        self.has_solution
    }

    fn best_bound(&self) -> Option<f64> {
        self.best_bound
    }

    fn status_detail(&self) -> String {
        match self.status {
            Some(s) => format!("{s:?}"),
            None => "Unknown (status code not recognised by grb 3.0.1)".to_string(),
        }
    }
}

// NOTE ON RUNNING THESE: every thread that touches Gurobi takes a licence
// token; under a single-token licence run these serially. `cargo test` uses one
// thread per test by default, so these are `#[ignore]`d to keep a plain
// `cargo test` from grabbing three tokens at once. Run them deliberately:
//
//   cargo test --release -p extraction-gym --features ilp-gurobi \
//       -- --ignored --test-threads=1
#[cfg(test)]
mod tests {
    use super::GurobiModel;
    use crate::milp::conformance;

    #[test]
    #[ignore = "takes a Gurobi license token; run with --ignored --test-threads=1"]
    fn time_limit_is_respected() {
        conformance::time_limit_is_respected::<GurobiModel>();
    }

    #[test]
    #[ignore = "takes a Gurobi license token; run with --ignored --test-threads=1"]
    fn incremental_rows_on_live_model() {
        conformance::incremental_rows_on_live_model::<GurobiModel>();
    }

    #[test]
    #[ignore = "takes a Gurobi license token; run with --ignored --test-threads=1"]
    fn warm_start_does_not_break_the_answer() {
        conformance::warm_start_does_not_break_the_answer::<GurobiModel>();
    }
}
