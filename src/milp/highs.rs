//! HiGHS backend for [`MilpModel`].
//!
//! Talks to HiGHS through `highs-sys` (raw C FFI), deliberately *not* through
//! the high-level `highs` crate: that wrapper consumes the model to solve it and
//! exposes neither incremental row addition on a live model nor a MIP start,
//! both of which the extractors require.
//!
//! # Shape of the implementation
//!
//! One `Highs` object is created in [`HighsModel::new`] and lives until drop.
//! Newly created columns/rows are buffered in [`Staged`] and appended to that
//! same live object in bulk at the start of each [`HighsModel::solve`]; nothing
//! is ever rebuilt or re-loaded. See the [`super::stage`] module docs for why
//! the buffering step exists (short version: `Highs_changeCoeff` is `O(nnz)`,
//! so building a model one coefficient at a time is quadratic).
//!
//! # Parity with CBC
//!
//! Two HiGHS defaults would silently make this backend solve a *different*
//! problem from CBC, so [`HighsModel::new`] overrides them:
//!
//! * `mip_rel_gap` defaults to `1e-4` and `mip_abs_gap` to `1e-6`, i.e. HiGHS
//!   stops and reports `kOptimal` at a 0.01% gap. CBC's default gaps are 0, and
//!   the extractors assert that a "finished" solve is a true optimum (see the
//!   `assert!((cost - solution.obj_value()).abs() < EPSILON_ALLOWANCE)` and the
//!   cross-extractor DAG-optimality checks in `src/test.rs`, whose
//!   `EPSILON_ALLOWANCE` is `1e-5`). Both gaps are therefore set to 0.
//!   Re-widen them with `set_raw_parameter("mip_rel_gap", "1e-4")` if a
//!   benchmark deliberately wants the looser stopping rule.
//!
//! * `threads` defaults to 0 = "use every core". CBC is single-threaded, and a
//!   caller that runs many `HighsModel`s concurrently gives each process a
//!   share of a fixed CPU allocation, not the whole box — so "use every core" per process means
//!   every concurrent process trying to grab every core, i.e. severe
//!   self-contention, and it silently breaks the fairness of a CBC vs. HiGHS
//!   comparison (HiGHS would get to use multiple threads per file where CBC
//!   never could). [`HighsModel::new`] therefore pins `threads` to `1` via
//!   `set_raw_parameter("threads", "1")`. Widen it back with
//!   `set_raw_parameter("threads", "N")` if a benchmark deliberately wants
//!   multi-threaded HiGHS.
//!
//! # Warm start: real, not a no-op
//!
//! [`MilpModel::set_col_initial_solution`] maps to `Highs_setSparseSolution`,
//! which lands in `Highs::setSolution(n, idx, val)`; for a MIP that triggers
//! `completeSolutionFromDiscreteAssignment()`, which fills in the continuous
//! variables and hands the point to `HighsMipSolver` as
//! `kSolutionSourceUserSolution`. Confirmed empirically on a 60-item
//! strongly-correlated knapsack: with the start seeded from a previous
//! solution, the HiGHS log prints
//!
//! ```text
//! MIP start solution is feasible, objective value is -15119
//! ```
//!
//! One subtlety this costs: `Highs_addCols`/`Highs_addRows` both call
//! `invalidateModelStatusSolutionAndInfo()`, which throws the stored solution
//! away. The start is therefore re-applied at the end of every flush rather
//! than once at seed time.
//!
//! # Infinity
//!
//! HiGHS treats `1e30` as infinite and *rejects* `f64::INFINITY` (it fails the
//! `assess` bound checks). Every bound crossing the FFI therefore goes through
//! [`hinf`].

use std::ffi::CString;
use std::os::raw::c_void;

use highs_sys::*;

use super::stage::{snap_integral, Staged};
use super::{MilpModel, MilpSense, MilpSolution};

/// HiGHS' notion of infinity. Anything at or beyond this magnitude is infinite.
const HIGHS_INF: f64 = 1.0e30;

/// Sentinel objective reported when no incumbent exists. Mirrors CBC's
/// `COIN_DBL_MAX` behaviour, which `faster_ilp_cbc` relies on ("no solution
/// compares worse than anything real" — it does
/// `solution.obj_value() > initial_result_cost` on a timed-out solve).
const NO_SOLUTION_OBJ: f64 = 1.0e30;

/// Clamp a bound into HiGHS' finite-or-`±1e30` representation.
fn hinf(v: f64) -> f64 {
    if v >= HIGHS_INF || v == f64::INFINITY {
        HIGHS_INF
    } else if v <= -HIGHS_INF || v == f64::NEG_INFINITY {
        -HIGHS_INF
    } else {
        v
    }
}

/// Opaque column handle: an index into the staged column list, which is also
/// the HiGHS column index (columns are only ever appended, never deleted).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HighsCol(u32);

/// Opaque row handle; same invariant as [`HighsCol`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct HighsRow(u32);

/// A live HiGHS model plus the not-yet-pushed delta.
pub struct HighsModel {
    h: *mut c_void,
    s: Staged,
}

// The `Highs` object is not shared between threads by this crate; each model
// owns its handle exclusively and never hands the pointer out.
unsafe impl Send for HighsModel {}

impl Drop for HighsModel {
    fn drop(&mut self) {
        unsafe { Highs_destroy(self.h) }
    }
}

/// An owned snapshot of a HiGHS solve. Column values are copied out at
/// `solve()` time so the solution does not borrow the model (trait rule 3) and
/// stays valid after further model mutation.
pub struct HighsSolution {
    col_value: Vec<f64>,
    obj: f64,
    model_status: HighsInt,
    /// `kHighsSolutionStatus*`: 0 none, 1 infeasible, 2 feasible.
    primal_status: HighsInt,
    /// `mip_dual_bound` info value, snapshotted at solve time (the info block
    /// is invalidated by any later model change, so it cannot be read lazily).
    dual_bound: Option<f64>,
}

fn model_status_name(s: HighsInt) -> &'static str {
    match s {
        MODEL_STATUS_NOTSET => "NotSet",
        MODEL_STATUS_LOAD_ERROR => "LoadError",
        MODEL_STATUS_MODEL_ERROR => "ModelError",
        MODEL_STATUS_PRESOLVE_ERROR => "PresolveError",
        MODEL_STATUS_SOLVE_ERROR => "SolveError",
        MODEL_STATUS_POSTSOLVE_ERROR => "PostsolveError",
        MODEL_STATUS_MODEL_EMPTY => "ModelEmpty",
        MODEL_STATUS_OPTIMAL => "Optimal",
        MODEL_STATUS_INFEASIBLE => "Infeasible",
        MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE => "UnboundedOrInfeasible",
        MODEL_STATUS_UNBOUNDED => "Unbounded",
        MODEL_STATUS_OBJECTIVE_BOUND => "ObjectiveBound",
        MODEL_STATUS_OBJECTIVE_TARGET => "ObjectiveTarget",
        MODEL_STATUS_REACHED_TIME_LIMIT => "TimeLimit",
        MODEL_STATUS_REACHED_ITERATION_LIMIT => "IterationLimit",
        MODEL_STATUS_UNKNOWN => "Unknown",
        MODEL_STATUS_REACHED_SOLUTION_LIMIT => "SolutionLimit",
        MODEL_STATUS_REACHED_INTERRUPT => "Interrupt",
        MODEL_STATUS_REACHED_MEMORY_LIMIT => "MemoryLimit",
        _ => "Unrecognised",
    }
}

impl MilpSolution for HighsSolution {
    type Col = HighsCol;

    fn col(&self, col: HighsCol) -> f64 {
        // A column added after this solution was taken has no value here.
        // Returning 0.0 matches "not selected", which is the only sane reading;
        // in practice the extractors never query such a column.
        self.col_value.get(col.0 as usize).copied().unwrap_or(0.0)
    }

    fn obj_value(&self) -> f64 {
        self.obj
    }

    fn is_infeasible(&self) -> bool {
        self.model_status == MODEL_STATUS_INFEASIBLE
    }

    /// "The solver terminated of its own accord." Mirrors CBC's
    /// `status() == Finished`, which is also true for a proven-infeasible or
    /// proven-unbounded model — hence those statuses count as completion here
    /// too. Everything that means "a limit cut me off" (time / iteration /
    /// solution / interrupt / memory) and everything that means "I do not know"
    /// (`Unknown`, the error statuses) counts as *not* finished, so the callers
    /// fall back to the greedy-DAG extraction rather than trusting a partial
    /// answer.
    fn ran_to_completion(&self) -> bool {
        matches!(
            self.model_status,
            MODEL_STATUS_OPTIMAL
                | MODEL_STATUS_INFEASIBLE
                | MODEL_STATUS_UNBOUNDED
                | MODEL_STATUS_UNBOUNDED_OR_INFEASIBLE
                | MODEL_STATUS_MODEL_EMPTY
        )
    }

    fn has_solution(&self) -> bool {
        self.primal_status == SOLUTION_STATUS_FEASIBLE
    }

    fn best_bound(&self) -> Option<f64> {
        self.dual_bound
    }

    fn status_detail(&self) -> String {
        format!(
            "{}, primal_solution_status = {}",
            model_status_name(self.model_status),
            match self.primal_status {
                SOLUTION_STATUS_NONE => "None",
                SOLUTION_STATUS_INFEASIBLE => "Infeasible",
                SOLUTION_STATUS_FEASIBLE => "Feasible",
                _ => "Unrecognised",
            }
        )
    }
}

impl HighsModel {
    /// Raw HiGHS handle, for HiGHS-only code. The staged delta is *not*
    /// necessarily in it — call [`Self::flush`] first if that matters.
    pub fn raw(&self) -> *mut c_void {
        self.h
    }

    fn set_bool_opt(&mut self, name: &str, value: bool) {
        let c = CString::new(name).expect("option name contains a NUL");
        unsafe {
            Highs_setBoolOptionValue(self.h, c.as_ptr(), if value { 1 } else { 0 });
        }
    }

    fn set_double_opt(&mut self, name: &str, value: f64) {
        let c = CString::new(name).expect("option name contains a NUL");
        unsafe {
            Highs_setDoubleOptionValue(self.h, c.as_ptr(), value);
        }
    }

    /// Push the staged delta into the live HiGHS object.
    ///
    /// Order matters: new columns must exist before rows that reference them,
    /// and a coefficient replay on an old row may reference a brand-new column,
    /// so it goes last.
    pub fn flush(&mut self) {
        unsafe {
            if self.s.sense_dirty() {
                let sense = match self.s.sense() {
                    MilpSense::Minimize => OBJECTIVE_SENSE_MINIMIZE,
                    MilpSense::Maximize => OBJECTIVE_SENSE_MAXIMIZE,
                };
                Highs_changeObjectiveSense(self.h, sense);
            }

            // 1. modifications to already-pushed columns
            for c in self.s.dirty_cols() {
                let d = *self.s.col(c);
                Highs_changeColBounds(self.h, c as HighsInt, hinf(d.lower), hinf(d.upper));
                Highs_changeColCost(self.h, c as HighsInt, d.cost);
                Highs_changeColIntegrality(
                    self.h,
                    c as HighsInt,
                    if d.integral {
                        VAR_TYPE_INTEGER
                    } else {
                        VAR_TYPE_CONTINUOUS
                    },
                );
            }

            // 2. new columns, in one call
            let n_new_cols = self.s.num_new_cols();
            if n_new_cols > 0 {
                let mut costs = Vec::with_capacity(n_new_cols);
                let mut lower = Vec::with_capacity(n_new_cols);
                let mut upper = Vec::with_capacity(n_new_cols);
                let mut integral = Vec::with_capacity(n_new_cols);
                for (idx, d) in self.s.new_cols() {
                    costs.push(d.cost);
                    lower.push(hinf(d.lower));
                    upper.push(hinf(d.upper));
                    if d.integral {
                        integral.push(idx as HighsInt);
                    }
                }
                Highs_addCols(
                    self.h,
                    n_new_cols as HighsInt,
                    costs.as_ptr(),
                    lower.as_ptr(),
                    upper.as_ptr(),
                    0,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                );
                // No bulk "set integrality for this set of columns" that takes
                // an index list *and* a single type, so one call each. Cheap:
                // it is an O(1) vector write per column.
                for idx in integral {
                    Highs_changeColIntegrality(self.h, idx, VAR_TYPE_INTEGER);
                }
            }

            // 3. bound changes on already-pushed rows
            for r in self.s.dirty_rows() {
                let d = self.s.row(r);
                let (lo, up) = (hinf(d.lower), hinf(d.upper));
                Highs_changeRowBounds(self.h, r as HighsInt, lo, up);
            }

            // 4. new rows, in one call (CSR)
            let n_new_rows = self.s.num_new_rows();
            if n_new_rows > 0 {
                let mut lower = Vec::with_capacity(n_new_rows);
                let mut upper = Vec::with_capacity(n_new_rows);
                let mut starts = Vec::with_capacity(n_new_rows);
                let mut index: Vec<HighsInt> = Vec::new();
                let mut value: Vec<f64> = Vec::new();
                for (_, d) in self.s.new_rows() {
                    starts.push(index.len() as HighsInt);
                    lower.push(hinf(d.lower));
                    upper.push(hinf(d.upper));
                    for (&c, &w) in &d.coeffs {
                        index.push(c as HighsInt);
                        value.push(w);
                    }
                }
                Highs_addRows(
                    self.h,
                    n_new_rows as HighsInt,
                    lower.as_ptr(),
                    upper.as_ptr(),
                    index.len() as HighsInt,
                    starts.as_ptr(),
                    if index.is_empty() {
                        std::ptr::null()
                    } else {
                        index.as_ptr()
                    },
                    if value.is_empty() {
                        std::ptr::null()
                    } else {
                        value.as_ptr()
                    },
                );
            }

            // 5. coefficient replays on already-pushed rows
            for (r, c, w) in self.s.dirty_coeffs() {
                Highs_changeCoeff(self.h, r as HighsInt, c as HighsInt, w);
            }
        }

        self.s.mark_flushed();
    }

    /// Hand the staged MIP start to HiGHS, if there is one.
    ///
    /// HiGHS calls `invalidateModelStatusSolutionAndInfo()` from `addCols` /
    /// `addRows`, so any user solution set before a model change is discarded.
    /// The start is therefore kept staged and re-applied here, after the flush,
    /// on every solve.
    fn apply_warm_start(&mut self) {
        let warm = self.s.warm();
        if warm.is_empty() {
            return;
        }
        let n = self.s.num_cols();
        let mut index: Vec<HighsInt> = Vec::with_capacity(warm.len());
        let mut value: Vec<f64> = Vec::with_capacity(warm.len());
        for (&c, &v) in warm {
            if (c as usize) < n {
                index.push(c as HighsInt);
                value.push(v);
            }
        }
        if index.is_empty() {
            return;
        }
        // Sparse (partial) start: HiGHS fills the unspecified columns itself in
        // `Highs::completeSolutionFromDiscreteAssignment()` and then feeds the
        // completed assignment to the MIP solver as a user incumbent
        // (`kSolutionSourceUserSolution`). This is a genuine MIP start, not a
        // no-op. Note it is not free — completing a partial assignment can
        // involve solving a sub-MIP.
        unsafe {
            Highs_setSparseSolution(
                self.h,
                index.len() as HighsInt,
                index.as_ptr(),
                value.as_ptr(),
            );
        }
    }
}

impl MilpModel for HighsModel {
    const NAME: &'static str = "HiGHS";

    type Col = HighsCol;
    type Row = HighsRow;
    type Solution = HighsSolution;

    fn new() -> Self {
        let h = unsafe { Highs_create() };
        let mut m = HighsModel {
            h,
            s: Staged::new(),
        };
        // CBC parity: prove optimality rather than stopping at a 0.01% gap.
        // See the module docs.
        m.set_double_opt("mip_rel_gap", 0.0);
        m.set_double_opt("mip_abs_gap", 0.0);
        // CBC parity: `coin_cbc` starts quiet, and only `faster_ilp_cbc` ever
        // calls `set_log_level(0)`. Without this, `ilp_cbc`'s extractors would
        // dump a full HiGHS MIP log per solve into the benchmark output.
        // `set_log_level(1..)` turns it back on.
        m.set_bool_opt("output_flag", false);
        // CBC parity + concurrent-benchmark correctness: see the module docs'
        // "Parity with CBC" section. Override with
        // `set_raw_parameter("threads", "N")` if ever wanted.
        m.set_raw_parameter("threads", "1");
        m
    }

    fn add_col(&mut self) -> HighsCol {
        HighsCol(self.s.add_col())
    }

    fn add_binary(&mut self) -> HighsCol {
        HighsCol(self.s.add_binary())
    }

    fn set_col_lower(&mut self, col: HighsCol, value: f64) {
        self.s.set_col_lower(col.0, value)
    }

    fn set_col_upper(&mut self, col: HighsCol, value: f64) {
        self.s.set_col_upper(col.0, value)
    }

    fn set_obj_coeff(&mut self, col: HighsCol, value: f64) {
        self.s.set_obj_coeff(col.0, value)
    }

    fn add_row(&mut self) -> HighsRow {
        HighsRow(self.s.add_row())
    }

    fn set_row_lower(&mut self, row: HighsRow, value: f64) {
        self.s.set_row_lower(row.0, value)
    }

    fn set_row_upper(&mut self, row: HighsRow, value: f64) {
        self.s.set_row_upper(row.0, value)
    }

    fn set_row_equal(&mut self, row: HighsRow, value: f64) {
        self.s.set_row_equal(row.0, value)
    }

    fn set_weight(&mut self, row: HighsRow, col: HighsCol, weight: f64) {
        self.s.set_weight(row.0, col.0, weight)
    }

    fn set_obj_sense(&mut self, sense: MilpSense) {
        self.s.set_obj_sense(sense)
    }

    /// HiGHS' `time_limit` option, in seconds. Applied to the `Highs` object
    /// immediately (options are independent of the staged model delta), so it
    /// is in force for the very next `Highs_run`.
    ///
    /// `u32::MAX` becomes `4294967295.0` s (~136 years), i.e. effectively
    /// unbounded but still a real number, and `0` becomes `0.0`, which makes
    /// HiGHS return `kTimeLimit` essentially immediately — the behaviour
    /// `faster_ilp_cbc` wants when its own budget is exhausted.
    fn set_time_limit_seconds(&mut self, seconds: u32) {
        self.set_double_opt("time_limit", f64::from(seconds));
    }

    /// `0` silences HiGHS entirely (`output_flag = false`); anything else turns
    /// its log back on. HiGHS has no finer verbosity dial worth exposing here.
    fn set_log_level(&mut self, level: u32) {
        self.set_bool_opt("output_flag", level > 0);
    }

    /// `log_file` + `output_flag = true` + `log_to_console = false`: the MIP
    /// progress table goes to `path` and nothing goes to stdout, so the
    /// benchmark's own output is unaffected while the trajectory becomes
    /// recoverable (see [`super::trajectory`]).
    fn set_log_file(&mut self, path: &str) {
        self.set_raw_parameter("log_file", path);
        self.set_bool_opt("output_flag", true);
        self.set_bool_opt("log_to_console", false);
    }

    /// Overrides the single-threaded default from [`MilpModel::set_threads`].
    /// See the trait docs for the slot-allocation invariant this must respect.
    fn set_threads(&mut self, threads: u32) {
        // HiGHS reads `threads = 0` as "all cores" -- the default we must never
        // fall back into -- so clamp 0 up to 1 rather than passing it through.
        self.set_raw_parameter("threads", &threads.max(1).to_string());
    }

    /// Set any HiGHS option by name.
    ///
    /// CBC's parameter names (`"seconds"`, `"loglevel"`, ...) do not exist in
    /// HiGHS, so a key copied from CBC code will not be understood. Rather than
    /// swallow that silently — which would mean a benchmark quietly running
    /// without the knob it thinks it set — this looks the option up with
    /// `Highs_getOptionType`, dispatches on the real type, and **reports every
    /// failure on stderr as well as through `log::warn!`**. `eprintln!` and not
    /// only `log` because `env_logger`'s default filter hides warnings, and an
    /// invisible warning is the same as no warning.
    ///
    /// The call is still non-fatal (the trait documents unknown keys as
    /// ignored, and CBC ignores them), so nothing panics; the operator just
    /// cannot miss it.
    fn set_raw_parameter(&mut self, key: &str, value: &str) {
        fn complain(key: &str, value: &str, why: &str) {
            let msg = format!("HiGHS: ignoring parameter {key:?} = {value:?}: {why}");
            log::warn!("{msg}");
            eprintln!("WARNING: {msg}");
        }

        let name = match CString::new(key) {
            Ok(n) => n,
            Err(_) => return complain(key, value, "key contains a NUL byte"),
        };
        let mut ty: HighsInt = -1;
        let rc = unsafe { Highs_getOptionType(self.h, name.as_ptr(), &mut ty) };
        if rc != STATUS_OK {
            return complain(key, value, "no HiGHS option of that name");
        }

        // kHighsOptionType: 0 bool, 1 int, 2 double, 3 string.
        let rc = match ty {
            0 => {
                let b = match value.trim().to_ascii_lowercase().as_str() {
                    "1" | "true" | "on" | "yes" => 1,
                    "0" | "false" | "off" | "no" => 0,
                    _ => return complain(key, value, "expected a boolean"),
                };
                unsafe { Highs_setBoolOptionValue(self.h, name.as_ptr(), b) }
            }
            1 => match value.trim().parse::<i32>() {
                Ok(v) => unsafe { Highs_setIntOptionValue(self.h, name.as_ptr(), v as HighsInt) },
                Err(_) => return complain(key, value, "expected an integer"),
            },
            2 => match value.trim().parse::<f64>() {
                Ok(v) => unsafe { Highs_setDoubleOptionValue(self.h, name.as_ptr(), v) },
                Err(_) => return complain(key, value, "expected a float"),
            },
            3 => {
                let v = match CString::new(value) {
                    Ok(v) => v,
                    Err(_) => return complain(key, value, "value contains a NUL byte"),
                };
                unsafe { Highs_setStringOptionValue(self.h, name.as_ptr(), v.as_ptr()) }
            }
            other => return complain(key, value, &format!("unhandled option type {other}")),
        };
        if rc != STATUS_OK {
            complain(key, value, "HiGHS rejected the value");
        }
    }

    /// Seed one column of the MIP start. Staged; handed to HiGHS as a *sparse*
    /// (partial) user solution just before the next `Highs_run`.
    fn set_col_initial_solution(&mut self, col: HighsCol, value: f64) {
        self.s.set_warm(col.0, value)
    }

    /// Seed the MIP start from a previous solution of this model. Columns added
    /// since `solution` was produced are simply absent from the start, which is
    /// fine — HiGHS completes partial assignments. (CBC crashes in the same
    /// situation; this backend does not.)
    fn set_initial_solution(&mut self, solution: &HighsSolution) {
        self.s.set_warm_all(
            solution
                .col_value
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v)),
        );
    }

    fn solve(&mut self) -> HighsSolution {
        self.flush();
        self.apply_warm_start();

        unsafe {
            Highs_run(self.h);
        }

        let n = self.s.num_cols();
        let n_rows = self.s.num_rows();
        let mut col_value = vec![0.0f64; n.max(1)];
        let mut row_value = vec![0.0f64; n_rows.max(1)];

        let model_status = unsafe { Highs_getModelStatus(self.h) };

        let mut primal_status: HighsInt = SOLUTION_STATUS_NONE;
        let info = CString::new("primal_solution_status").unwrap();
        unsafe {
            Highs_getIntInfoValue(self.h, info.as_ptr(), &mut primal_status);
        }

        if primal_status == SOLUTION_STATUS_FEASIBLE {
            unsafe {
                Highs_getSolution(
                    self.h,
                    col_value.as_mut_ptr(),
                    std::ptr::null_mut(),
                    row_value.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
            }
        }
        col_value.truncate(n);
        snap_integral(&self.s, &mut col_value);

        let obj = if primal_status == SOLUTION_STATUS_FEASIBLE {
            unsafe { Highs_getObjectiveValue(self.h) }
        } else {
            // See NO_SOLUTION_OBJ: callers compare this against a real cost.
            NO_SOLUTION_OBJ
        };

        // Snapshot the dual bound now: HiGHS invalidates its info block on the
        // next model change, and the solution must stay valid past that.
        let mut bound = 0.0f64;
        let bound_key = CString::new("mip_dual_bound").unwrap();
        let bound_rc = unsafe { Highs_getDoubleInfoValue(self.h, bound_key.as_ptr(), &mut bound) };
        let dual_bound = if bound_rc == STATUS_OK && bound.is_finite() && bound.abs() < HIGHS_INF {
            Some(bound)
        } else {
            None
        };

        HighsSolution {
            col_value,
            obj,
            model_status,
            primal_status,
            dual_bound,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::HighsModel;
    use crate::milp::conformance;

    #[test]
    fn time_limit_is_respected() {
        conformance::time_limit_is_respected::<HighsModel>();
    }

    #[test]
    fn incremental_rows_on_live_model() {
        conformance::incremental_rows_on_live_model::<HighsModel>();
    }

    #[test]
    fn warm_start_does_not_break_the_answer() {
        conformance::warm_start_does_not_break_the_answer::<HighsModel>();
    }
}
