//! Solver-neutral staging buffer shared by the HiGHS and Gurobi backends.
//!
//! # Why this exists
//!
//! [`MilpModel`](super::MilpModel) is shaped by what the CBC-era extractors do:
//! create a row, then set its bounds and its coefficients one at a time,
//! afterwards. CBC can absorb that pattern for free because `coin_cbc::Model`
//! *is* a Rust-side description that only becomes a solver model at `solve()`.
//! HiGHS and Gurobi are different: they own a live problem, and
//!
//! * a row must have real bounds the moment it is created (Gurobi has no
//!   `(-inf, +inf)` constraint at all), and
//! * poking one coefficient at a time into a live sparse matrix
//!   (`Highs_changeCoeff` / `GRBchgcoeffs`) is `O(nnz)` per poke, which turns
//!   model construction quadratic on the e-graphs this project actually runs.
//!
//! So both backends buffer *newly created* columns and rows here and push them
//! to the solver in bulk, once, at the start of the next `solve()`.
//!
//! # This is still an incremental, live model
//!
//! What is buffered is only the **delta since the last solve**. Everything that
//! has already been pushed stays in the solver: the cycle-breaking loop in
//! `faster_ilp_cbc` adds rows, calls `solve()` again, and those rows are
//! *appended* to the existing `Highs`/`GRBmodel` object. The model is never
//! rebuilt, never re-loaded, never round-tripped through a file, and Gurobi
//! keeps its incumbent across the re-solve. That is exactly the property the
//! trait docs (point 2) demand.
//!
//! Mutations of things that *have* already been pushed (a bound change, a
//! coefficient change) cannot be batched away, so they are recorded in the
//! `dirty_*` sets and replayed with the solver's targeted modification calls at
//! the next flush.

use super::MilpSense;
use indexmap::{IndexMap, IndexSet};

/// A staged column. Defaults match the trait contract: `add_col` gives
/// `[0, +inf)` continuous with objective coefficient 0, `add_binary` gives
/// `[0, 1]` integral.
#[derive(Clone, Copy, Debug)]
pub struct ColDef {
    pub lower: f64,
    pub upper: f64,
    pub cost: f64,
    pub integral: bool,
}

/// A staged row. Defaults to `(-inf, +inf)` with no coefficients, per the trait
/// contract for [`MilpModel::add_row`](super::MilpModel::add_row).
#[derive(Clone, Debug, Default)]
pub struct RowDef {
    pub lower: f64,
    pub upper: f64,
    /// column index -> coefficient. `IndexMap` (not a hash map) so the nonzero
    /// order handed to the solver is insertion order and therefore
    /// deterministic across runs — a solver fed the same matrix in a different
    /// order can take a different branching path and report a different
    /// (equally optimal) solution, which would make A/B comparisons noise.
    pub coeffs: IndexMap<u32, f64>,
}

/// Round the values of integral columns to the nearest integer, in place.
///
/// **Not cosmetic.** A MIP solver only guarantees its integer variables are
/// integral to within its integer-feasibility tolerance (`1e-6` by default in
/// both HiGHS and Gurobi), and it hands back the raw LP values. CBC happens to
/// return clean `0.0`/`1.0`, and `faster_ilp_cbc` was written against that: it
/// classifies a variable as selected with a bare `solution.col(v) > 0.0`, then
///
/// ```text
/// assert_eq!(1, var.variables.iter().filter(|&n| solution.col(*n) > 0.0).count());
/// ```
///
/// A class variable coming back as `3e-13` instead of `0.0` therefore reads as
/// "active" while none of its members does, and that assertion fires. This was
/// observed with HiGHS on a large extraction model.
///
/// Snapping here rather than loosening the extractor's threshold keeps the
/// comparison against the CBC baseline exact: "integral columns come back
/// integral" is part of the behaviour the trait promises, so each backend owes
/// it, and the CBC path stays untouched.
pub fn snap_integral(s: &Staged, values: &mut [f64]) {
    for (i, v) in values.iter_mut().enumerate() {
        if i < s.num_cols() && s.col(i as u32).integral {
            *v = v.round();
        }
    }
}

/// Buffered model description plus the bookkeeping needed to push only the
/// delta to a live solver.
#[derive(Clone, Debug)]
pub struct Staged {
    cols: Vec<ColDef>,
    rows: Vec<RowDef>,
    /// `cols[..pushed_cols]` are already in the solver.
    pushed_cols: usize,
    /// `rows[..pushed_rows]` are already in the solver.
    pushed_rows: usize,
    /// Already-pushed columns whose bounds / cost / integrality changed.
    dirty_cols: IndexSet<u32>,
    /// Already-pushed rows whose bounds changed.
    dirty_rows: IndexSet<u32>,
    /// `(row, col)` pairs on already-pushed rows whose coefficient changed.
    dirty_coeffs: IndexSet<(u32, u32)>,
    /// Staged MIP start, column index -> value. Partial by construction:
    /// `set_col_initial_solution` seeds one column at a time.
    warm: IndexMap<u32, f64>,
    sense: MilpSense,
    sense_dirty: bool,
}

impl Default for Staged {
    fn default() -> Self {
        Staged {
            cols: Vec::new(),
            rows: Vec::new(),
            pushed_cols: 0,
            pushed_rows: 0,
            dirty_cols: IndexSet::new(),
            dirty_rows: IndexSet::new(),
            dirty_coeffs: IndexSet::new(),
            warm: IndexMap::new(),
            // The trait specifies Minimize as the default; `faster_ilp_cbc`
            // never calls set_obj_sense and relies on it.
            sense: MilpSense::Minimize,
            sense_dirty: false,
        }
    }
}

impl Staged {
    pub fn new() -> Self {
        Self::default()
    }

    // -- construction -------------------------------------------------------

    pub fn add_col(&mut self) -> u32 {
        let idx = self.cols.len() as u32;
        self.cols.push(ColDef {
            lower: 0.0,
            upper: f64::INFINITY,
            cost: 0.0,
            integral: false,
        });
        idx
    }

    pub fn add_binary(&mut self) -> u32 {
        let idx = self.cols.len() as u32;
        self.cols.push(ColDef {
            lower: 0.0,
            upper: 1.0,
            cost: 0.0,
            integral: true,
        });
        idx
    }

    pub fn add_row(&mut self) -> u32 {
        let idx = self.rows.len() as u32;
        self.rows.push(RowDef {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
            coeffs: IndexMap::new(),
        });
        idx
    }

    // -- mutation -----------------------------------------------------------

    fn touch_col(&mut self, col: u32) {
        if (col as usize) < self.pushed_cols {
            self.dirty_cols.insert(col);
        }
    }

    fn touch_row(&mut self, row: u32) {
        if (row as usize) < self.pushed_rows {
            self.dirty_rows.insert(row);
        }
    }

    pub fn set_col_lower(&mut self, col: u32, value: f64) {
        self.cols[col as usize].lower = value;
        self.touch_col(col);
    }

    pub fn set_col_upper(&mut self, col: u32, value: f64) {
        self.cols[col as usize].upper = value;
        self.touch_col(col);
    }

    pub fn set_obj_coeff(&mut self, col: u32, value: f64) {
        self.cols[col as usize].cost = value;
        self.touch_col(col);
    }

    pub fn set_row_lower(&mut self, row: u32, value: f64) {
        self.rows[row as usize].lower = value;
        self.touch_row(row);
    }

    pub fn set_row_upper(&mut self, row: u32, value: f64) {
        self.rows[row as usize].upper = value;
        self.touch_row(row);
    }

    pub fn set_row_equal(&mut self, row: u32, value: f64) {
        let r = &mut self.rows[row as usize];
        r.lower = value;
        r.upper = value;
        self.touch_row(row);
    }

    /// Set (or, with `0.0`, remove) a coefficient. Matches
    /// `coin_cbc::Model::set_weight`, which replaces rather than accumulates.
    pub fn set_weight(&mut self, row: u32, col: u32, weight: f64) {
        let r = &mut self.rows[row as usize];
        if weight == 0.0 {
            r.coeffs.shift_remove(&col);
        } else {
            r.coeffs.insert(col, weight);
        }
        if (row as usize) < self.pushed_rows {
            self.dirty_coeffs.insert((row, col));
        }
    }

    pub fn set_obj_sense(&mut self, sense: MilpSense) {
        if sense != self.sense {
            self.sense = sense;
            self.sense_dirty = true;
        }
    }

    pub fn set_warm(&mut self, col: u32, value: f64) {
        self.warm.insert(col, value);
    }

    /// Replace the staged MIP start wholesale (used by `set_initial_solution`).
    pub fn set_warm_all(&mut self, values: impl IntoIterator<Item = (u32, f64)>) {
        self.warm.clear();
        self.warm.extend(values);
    }

    // -- inspection ---------------------------------------------------------

    pub fn num_cols(&self) -> usize {
        self.cols.len()
    }

    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    pub fn col(&self, col: u32) -> &ColDef {
        &self.cols[col as usize]
    }

    pub fn row(&self, row: u32) -> &RowDef {
        &self.rows[row as usize]
    }

    pub fn sense(&self) -> MilpSense {
        self.sense
    }

    pub fn sense_dirty(&self) -> bool {
        self.sense_dirty
    }

    pub fn warm(&self) -> &IndexMap<u32, f64> {
        &self.warm
    }

    /// Columns created since the last [`Self::mark_flushed`], with their index.
    pub fn new_cols(&self) -> impl Iterator<Item = (u32, &ColDef)> {
        (self.pushed_cols..self.cols.len()).map(move |i| (i as u32, &self.cols[i]))
    }

    pub fn num_new_cols(&self) -> usize {
        self.cols.len() - self.pushed_cols
    }

    /// Rows created since the last [`Self::mark_flushed`], with their index.
    pub fn new_rows(&self) -> impl Iterator<Item = (u32, &RowDef)> {
        (self.pushed_rows..self.rows.len()).map(move |i| (i as u32, &self.rows[i]))
    }

    pub fn num_new_rows(&self) -> usize {
        self.rows.len() - self.pushed_rows
    }

    pub fn dirty_cols(&self) -> Vec<u32> {
        self.dirty_cols.iter().copied().collect()
    }

    pub fn dirty_rows(&self) -> Vec<u32> {
        self.dirty_rows.iter().copied().collect()
    }

    /// `(row, col, coefficient)` triples to replay on already-pushed rows. The
    /// coefficient is looked up now, so a `set_weight(.., 0.0)` correctly
    /// reports `0.0` (i.e. "delete this entry from the solver's matrix").
    pub fn dirty_coeffs(&self) -> Vec<(u32, u32, f64)> {
        self.dirty_coeffs
            .iter()
            .map(|&(r, c)| {
                (
                    r,
                    c,
                    self.rows[r as usize].coeffs.get(&c).copied().unwrap_or(0.0),
                )
            })
            .collect()
    }

    /// Everything staged is now in the solver.
    ///
    /// Clears the warm start too: a MIP start is consumed by the solve it was
    /// set for. Re-applying it on later solves of the cycle-breaking loop feeds
    /// back a point containing the cycle that was just blocked, which the solver
    /// then spends budget rejecting.
    pub fn mark_flushed(&mut self) {
        self.pushed_cols = self.cols.len();
        self.pushed_rows = self.rows.len();
        self.dirty_cols.clear();
        self.dirty_rows.clear();
        self.dirty_coeffs.clear();
        self.sense_dirty = false;
        self.warm.clear();
    }
}
