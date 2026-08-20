//! COIN-OR CBC backend for [`MilpModel`].
//!
//! This is the behavioural baseline: every method is a direct, semantics-
//! preserving delegation to the `coin_cbc` call the extractors previously made
//! inline. Nothing here may add, reorder or reinterpret a solver interaction —
//! benchmarks of other backends are only meaningful if the CBC path is
//! bit-for-bit what it was before the abstraction was introduced.

use coin_cbc::raw::Status as CbcRawStatus;
use coin_cbc::{Col, Model, Row, Sense, Solution};

use super::{MilpModel, MilpSense, MilpSolution};

/// CBC model handle. A thin newtype over `coin_cbc::Model`.
///
/// `coin_cbc::Model` is a pure Rust-side *description* of the problem; each
/// `solve()` loads it into a fresh `Cbc_Model`. That means "incremental" solving
/// here is incremental only in the sense that the description is extended in
/// place between solves (which is exactly what the cycle-breaking loop needs);
/// CBC itself restarts from scratch. This was true before the abstraction too.
#[derive(Default, Clone)]
pub struct CbcModel(Model);

impl CbcModel {
    /// Access the underlying `coin_cbc::Model` (escape hatch for CBC-only code).
    pub fn inner(&self) -> &Model {
        &self.0
    }
    /// Mutable access to the underlying `coin_cbc::Model`.
    pub fn inner_mut(&mut self) -> &mut Model {
        &mut self.0
    }
}

/// An owned CBC solution. `coin_cbc::Solution` owns its own `raw::Model` (it is
/// produced by `Model::solve(&self)`, which builds a fresh raw model), so it
/// does not borrow `CbcModel` and satisfies the trait's owned-solution rule.
pub struct CbcSolution(Solution);

impl CbcSolution {
    /// Access the underlying `coin_cbc::Solution`.
    pub fn inner(&self) -> &Solution {
        &self.0
    }
}

/// CBC reports `Cbc_getObjValue() == 1e30` (`COIN_DBL_MAX`) when it has no
/// feasible solution. Used only to split "stopped early" into the
/// with/without-solution statuses; no pre-existing behaviour depends on it.
const CBC_NO_SOLUTION_OBJ: f64 = 1e29;

impl MilpSolution for CbcSolution {
    type Col = Col;

    fn col(&self, col: Col) -> f64 {
        self.0.col(col)
    }

    fn obj_value(&self) -> f64 {
        self.0.raw().obj_value()
    }

    /// `Cbc_isProvenInfeasible`. Exactly the predicate the extractors used.
    ///
    /// Note that CBC reports `Status::Finished` *as well* for a model it proved
    /// infeasible, which is why this stays a separate predicate from
    /// [`Self::ran_to_completion`] rather than being folded into a single enum.
    fn is_infeasible(&self) -> bool {
        self.0.raw().is_proven_infeasible()
    }

    /// `Cbc_status() == Finished`. The extractors' "the solver did not stop
    /// early" test, spelled `solution.raw().status() != Status::Finished`
    /// before the abstraction.
    ///
    /// Note what is deliberately *not* consulted: `is_continuous_unbounded()`.
    /// The pre-abstraction code never looked at it, and folding it in would
    /// change which branch an unbounded-but-finished model takes.
    fn ran_to_completion(&self) -> bool {
        self.0.raw().status() == CbcRawStatus::Finished
    }

    /// Best-effort: CBC has no exposed "number of solutions found" accessor in
    /// this binding, but `Cbc_getObjValue` returns the `1e30` `COIN_DBL_MAX`
    /// sentinel when there is no incumbent.
    ///
    /// Nothing in the pre-abstraction code consulted this, so no existing
    /// behaviour depends on the heuristic; it only refines
    /// [`MilpStatus::TimeoutWithSolution`] vs [`MilpStatus::TimeoutNoSolution`]
    /// for new code.
    fn has_solution(&self) -> bool {
        self.0.raw().obj_value().abs() < CBC_NO_SOLUTION_OBJ
    }

    /// `Cbc_getBestPossibleObjValue`, i.e. CBC's dual bound. Read-only; adds no
    /// solver interaction to the baseline path (it is only called by the
    /// reporting code, after the solve has already returned).
    fn best_bound(&self) -> Option<f64> {
        let b = self.0.raw().best_possible_value();
        if b.is_finite() && b.abs() < CBC_NO_SOLUTION_OBJ {
            Some(b)
        } else {
            None
        }
    }

    /// `"{Status:?}, {SecondaryStatus:?}"`, matching the previous log format
    /// exactly (the call sites used `"... status {:?}, {:?}, obj = {}"`).
    fn status_detail(&self) -> String {
        let raw = self.0.raw();
        format!("{:?}, {:?}", raw.status(), raw.secondary_status())
    }
}

impl MilpModel for CbcModel {
    const NAME: &'static str = "CBC";

    type Col = Col;
    type Row = Row;
    type Solution = CbcSolution;

    fn new() -> Self {
        CbcModel(Model::default())
    }

    fn add_col(&mut self) -> Col {
        self.0.add_col()
    }

    fn add_binary(&mut self) -> Col {
        self.0.add_binary()
    }

    fn set_col_lower(&mut self, col: Col, value: f64) {
        self.0.set_col_lower(col, value)
    }

    fn set_col_upper(&mut self, col: Col, value: f64) {
        self.0.set_col_upper(col, value)
    }

    fn set_obj_coeff(&mut self, col: Col, value: f64) {
        self.0.set_obj_coeff(col, value)
    }

    fn add_row(&mut self) -> Row {
        self.0.add_row()
    }

    fn set_row_lower(&mut self, row: Row, value: f64) {
        self.0.set_row_lower(row, value)
    }

    fn set_row_upper(&mut self, row: Row, value: f64) {
        self.0.set_row_upper(row, value)
    }

    fn set_row_equal(&mut self, row: Row, value: f64) {
        self.0.set_row_equal(row, value)
    }

    fn set_weight(&mut self, row: Row, col: Col, weight: f64) {
        self.0.set_weight(row, col, weight)
    }

    fn set_obj_sense(&mut self, sense: MilpSense) {
        self.0.set_obj_sense(match sense {
            MilpSense::Minimize => Sense::Minimize,
            MilpSense::Maximize => Sense::Maximize,
        })
    }

    /// CBC's `seconds` parameter. Formatted exactly as before
    /// (`&seconds.to_string()`), so `u32::MAX` still becomes `"4294967295"` and
    /// `0` still becomes `"0"`.
    fn set_time_limit_seconds(&mut self, seconds: u32) {
        self.0.set_parameter("seconds", &seconds.to_string())
    }

    /// CBC's `loglevel` parameter (`"0"` silences it).
    fn set_log_level(&mut self, level: u32) {
        self.0.set_parameter("loglevel", &level.to_string())
    }

    fn set_raw_parameter(&mut self, key: &str, value: &str) {
        self.0.set_parameter(key, value)
    }

    /// **CBC's MIP start is not used by this crate.**
    ///
    /// Two independent problems, both documented at the (previously
    /// `if false`-gated) call sites in `faster_ilp_cbc.rs`:
    ///
    /// 1. `Cbc_setMIPStart` via `coin_cbc::Model::set_initial_solution` reads a
    ///    solution vector sized to the *previous* model and crashes when the
    ///    model has since gained columns — which is exactly what the
    ///    cycle-breaking loop does every iteration.
    /// 2. Seeding before the first solve produced *wrong* results. `coin_cbc`'s
    ///    `initial_solution` is a dense vector defaulting to `0.0` for every
    ///    column the caller did not set, so any model with continuous helper
    ///    columns (the level / arrival-time variables in `ilp_cbc.rs`) is
    ///    handed a start that violates its own rows, and CBC does not reject
    ///    it cleanly.
    ///
    /// Rather than paper over that, the extractors ask
    /// [`MilpModel::supports_warm_start`] and skip warm starting entirely on
    /// CBC, reporting `warm_start_applied = none` with a reason. The methods
    /// below still delegate faithfully so the backend stays a complete
    /// implementation of the trait.
    fn supports_warm_start(&self) -> bool {
        false
    }

    fn set_col_initial_solution(&mut self, col: Col, value: f64) {
        self.0.set_col_initial_solution(col, value)
    }

    fn set_initial_solution(&mut self, solution: &CbcSolution) {
        self.0.set_initial_solution(&solution.0)
    }

    fn solve(&mut self) -> CbcSolution {
        // `coin_cbc::Model::solve` takes `&self`; the trait takes `&mut self`
        // because other backends (Gurobi) need it.
        CbcSolution(self.0.solve())
    }
}
