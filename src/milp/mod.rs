/*!
A solver-agnostic MILP (mixed-integer linear programming) abstraction.

The ILP extractors in [`crate::extract`] used to talk directly to COIN-OR CBC
through the `coin_cbc` crate. This module factors that surface into a trait so
the same extraction models can be handed to a different solver (Gurobi, HiGHS,
...) and benchmarked against each other without touching the extractor logic.

# What the trait has to support

The two extractors constrain the design more than a generic "build an LP and
solve it" API would:

1. **MIP warm start.** The greedy-DAG solution is (optionally) fed to the solver
   as an incumbent via [`MilpModel::set_col_initial_solution`] /
   [`MilpModel::set_initial_solution`]. Any backend must be able to accept a
   full or partial starting assignment before solving.

2. **Incremental constraint addition on a live model.** `faster_ilp_cbc`'s
   cycle-breaking loop solves, inspects the returned solution for cycles, adds
   rows banning each cycle it found, and re-solves *the same model object*. The
   trait therefore must not require rebuilding/re-serializing the model between
   solves: [`MilpModel::solve`] takes `&mut self` and leaves the model intact
   and further mutable afterwards.

   Note that "live model" here means live at the *Rust* level. CBC itself has no
   incremental API — `coin_cbc::Model::solve` re-loads the problem into a fresh
   `Cbc_Model` every call — but the model *description* (columns, rows, bounds,
   objective) is retained and extended in place, which is what the extractor
   needs. Backends that do support true incremental re-solves (Gurobi) can
   simply keep their handle open and get the speedup for free.

3. **Owned solutions.** [`MilpModel::Solution`] must not borrow the model, so
   the extractor can hold a solution and mutate the model in the same scope
   (exactly what the cycle loop does). Backends that keep solution values inside
   the solver object (Gurobi) must snapshot the column values on `solve`.

# Handle types

Columns and rows are opaque per-backend handles. They are stored in
`IndexSet`/`IndexMap` by the extractors, so the bound is `Copy + Eq + Hash`.
Deliberately *no* `Debug`/`Ord` bound: `coin_cbc::Col` does not implement
`Debug`, and requiring more than the extractors actually need would rule out
backends for no reason.
*/

// --- Backend selection -----------------------------------------------------
//
// Cargo features are additive, so two backend features enabled at once cannot
// be resolved by precedence without silently ignoring one of them. That would
// be a miserable thing to debug in a benchmark ("why is Gurobi as slow as
// CBC?"), so it is a hard error instead. Pick exactly one:
//
//   --features ilp-cbc      (default, via the `ilp` alias)
//   --features ilp-gurobi
//   --features ilp-highs

#[cfg(all(feature = "ilp-cbc", feature = "ilp-gurobi"))]
compile_error!(
    "extraction-gym: features `ilp-cbc` and `ilp-gurobi` are mutually exclusive \
     (exactly one MILP backend may be enabled). Build the two configurations \
     separately and compare them."
);
#[cfg(all(feature = "ilp-cbc", feature = "ilp-highs"))]
compile_error!(
    "extraction-gym: features `ilp-cbc` and `ilp-highs` are mutually exclusive \
     (exactly one MILP backend may be enabled). Build the two configurations \
     separately and compare them."
);
#[cfg(all(feature = "ilp-gurobi", feature = "ilp-highs"))]
compile_error!(
    "extraction-gym: features `ilp-gurobi` and `ilp-highs` are mutually exclusive \
     (exactly one MILP backend may be enabled). Build the two configurations \
     separately and compare them."
);

#[cfg(all(
    feature = "ilp-any",
    not(any(feature = "ilp-cbc", feature = "ilp-gurobi", feature = "ilp-highs"))
))]
compile_error!(
    "extraction-gym: the internal `ilp-any` feature was enabled without a MILP \
     backend. Enable one of `ilp-cbc`, `ilp-gurobi`, `ilp-highs` instead of \
     enabling `ilp-any` directly."
);

// Backend implementations.
#[cfg(feature = "ilp-cbc")]
pub mod cbc;

#[cfg(feature = "ilp-gurobi")]
pub mod gurobi;

#[cfg(feature = "ilp-highs")]
pub mod highs;

// Shared delta-push staging buffer. CBC does not need it (`coin_cbc::Model` is
// itself a Rust-side description), the two live-solver backends do.
#[cfg(any(feature = "ilp-gurobi", feature = "ilp-highs"))]
pub(crate) mod stage;

// Solver-log -> incumbent-trajectory parser (see `MilpModel::set_log_file`).
pub mod trajectory;

/// The MILP backend the ILP extractors use when instantiated through their
/// non-generic aliases (`CbcExtractor`, `FasterCbcExtractor`, ...). Exactly one
/// backend feature may be on, so this is unambiguous.
#[cfg(feature = "ilp-cbc")]
pub type DefaultMilp = cbc::CbcModel;

#[cfg(feature = "ilp-gurobi")]
pub type DefaultMilp = gurobi::GurobiModel;

#[cfg(feature = "ilp-highs")]
pub type DefaultMilp = highs::HighsModel;

// --- Core types ------------------------------------------------------------

/// Direction of the objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilpSense {
    Minimize,
    Maximize,
}

/// Solver-neutral summary of a [`MilpModel::solve`] call.
///
/// This deliberately collapses each backend's much larger status space onto the
/// four cases that matter. It is derived from the three orthogonal predicates
/// on [`MilpSolution`]; see [`MilpSolution::status`].
///
/// Beware that `Infeasible` and "did the solver finish" are *not* mutually
/// exclusive in every backend: CBC reports `Status::Finished` for a model it
/// proved infeasible. The extractors therefore branch on the raw predicates,
/// not on this enum, wherever the pre-abstraction behaviour depended on the
/// distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MilpStatus {
    /// The solver ran to completion with a proven-optimal incumbent.
    Optimal,
    /// The model was proven infeasible.
    Infeasible,
    /// The solver stopped early (time limit, node limit, ...) but has a
    /// feasible incumbent available.
    TimeoutWithSolution,
    /// The solver stopped early with no feasible solution found. Reading column
    /// values in this state is meaningless.
    TimeoutNoSolution,
}

/// A solved assignment. Owned — must not borrow the model it came from (see the
/// module docs, point 3).
pub trait MilpSolution {
    /// The column handle type of the model that produced this solution.
    type Col: Copy + Eq + std::hash::Hash;

    /// Value of `col` in the solution.
    fn col(&self, col: Self::Col) -> f64;

    /// Objective value of the incumbent. Backends report a large sentinel (CBC:
    /// `1e30`) when no solution was found; the extractors rely on that
    /// "no solution compares worse than anything real" behaviour.
    fn obj_value(&self) -> f64;

    // -- the three orthogonal outcome predicates ----------------------------
    //
    // Kept separate rather than folded into `status()` because the extractors'
    // control flow distinguishes combinations that a single enum cannot express
    // (CBC can be simultaneously "finished" and "proven infeasible"). Every
    // backend can answer all three.

    /// The model was proven to have no feasible solution.
    fn is_infeasible(&self) -> bool;

    /// The solver terminated of its own accord rather than being cut off by a
    /// limit (time / nodes / interrupt). This is *not* the same as "found an
    /// optimal solution": a proven-infeasible model also ran to completion.
    fn ran_to_completion(&self) -> bool;

    /// A feasible incumbent is available, so reading column values is
    /// meaningful.
    fn has_solution(&self) -> bool;

    /// Best proven bound on the objective (the dual/"best possible" value) at
    /// the moment the solve stopped.
    ///
    /// This is what makes a *timed-out* run interpretable: two configurations
    /// that both hit the wall with the same incumbent are not equally good if
    /// one closed the bound further. Every backend can report it
    /// (`Cbc_getBestPossibleObjValue`, HiGHS `mip_dual_bound`, Gurobi
    /// `ObjBound`), but it is `Option` because a backend that has not even
    /// solved a relaxation has nothing meaningful to say.
    ///
    /// Default `None` so a new backend compiles before it wires this up.
    fn best_bound(&self) -> Option<f64> {
        None
    }

    /// Relative MIP gap `|obj - bound| / max(|obj|, 1e-10)`.
    ///
    /// Derived from [`Self::obj_value`] and [`Self::best_bound`] rather than
    /// read from the solver, so the number means the same thing in all three
    /// backends (Gurobi's `MIPGap` and HiGHS' `mip_gap` use slightly different
    /// denominators). `None` when there is no incumbent or no bound.
    fn gap(&self) -> Option<f64> {
        if !self.has_solution() {
            return None;
        }
        let obj = self.obj_value();
        let bound = self.best_bound()?;
        if !obj.is_finite() || !bound.is_finite() {
            return None;
        }
        Some((obj - bound).abs() / obj.abs().max(1e-10))
    }

    /// Solver-neutral status, derived from the three predicates above.
    fn status(&self) -> MilpStatus {
        if self.is_infeasible() {
            MilpStatus::Infeasible
        } else if self.ran_to_completion() {
            MilpStatus::Optimal
        } else if self.has_solution() {
            MilpStatus::TimeoutWithSolution
        } else {
            MilpStatus::TimeoutNoSolution
        }
    }

    /// Backend-specific status text, for logging only. CBC renders this as
    /// `"{Status:?}, {SecondaryStatus:?}"` so the log lines are unchanged from
    /// the pre-abstraction code.
    fn status_detail(&self) -> String;
}

/// A mutable MILP model that can be built up incrementally and solved
/// repeatedly.
///
/// Column/row creation returns opaque handles; every mutator takes them by
/// value. Semantics are specified to match COIN-OR CBC, since the CBC backend
/// is the behavioural baseline all others are compared against:
///
/// * a new column has bounds `[0, +inf)` and objective coefficient `0`
///   ([`Self::add_col`]);
/// * a new binary column has bounds `[0, 1]` and is integral
///   ([`Self::add_binary`]);
/// * a new row has bounds `(-inf, +inf)` (i.e. no constraint) and no
///   coefficients ([`Self::add_row`]);
/// * the default objective sense is [`MilpSense::Minimize`] — note that
///   `faster_ilp_cbc` never calls [`Self::set_obj_sense`] and relies on this.
pub trait MilpModel: Sized {
    /// Short human-readable backend name, used in log messages (`"CBC"`).
    const NAME: &'static str;

    /// Opaque column handle. Stored in hash containers by the extractors.
    type Col: Copy + Eq + std::hash::Hash;
    /// Opaque row handle.
    type Row: Copy + Eq + std::hash::Hash;
    /// Owned solution type.
    type Solution: MilpSolution<Col = Self::Col>;

    /// An empty model: no columns, no rows, minimizing.
    fn new() -> Self;

    // -- columns ------------------------------------------------------------

    /// Add a continuous column with bounds `[0, +inf)`.
    fn add_col(&mut self) -> Self::Col;
    /// Add an integral column with bounds `[0, 1]`.
    fn add_binary(&mut self) -> Self::Col;
    /// Set a column's lower bound.
    fn set_col_lower(&mut self, col: Self::Col, value: f64);
    /// Set a column's upper bound.
    fn set_col_upper(&mut self, col: Self::Col, value: f64);
    /// Set a column's objective coefficient (replacing any previous value).
    fn set_obj_coeff(&mut self, col: Self::Col, value: f64);

    // -- rows ---------------------------------------------------------------

    /// Add an unconstrained row `(-inf, +inf)` with no coefficients.
    fn add_row(&mut self) -> Self::Row;
    /// Set a row's lower bound (`>=`).
    fn set_row_lower(&mut self, row: Self::Row, value: f64);
    /// Set a row's upper bound (`<=`).
    fn set_row_upper(&mut self, row: Self::Row, value: f64);
    /// Constrain a row to equal `value` (sets both bounds).
    fn set_row_equal(&mut self, row: Self::Row, value: f64);
    /// Set the coefficient of `col` in `row` (replacing any previous value; a
    /// weight of `0.0` removes the entry).
    fn set_weight(&mut self, row: Self::Row, col: Self::Col, weight: f64);

    // -- objective ----------------------------------------------------------

    fn set_obj_sense(&mut self, sense: MilpSense);

    // -- solver parameters --------------------------------------------------
    //
    // Only two parameters are actually used by the extractors, and every solver
    // has both under a different name, so they get typed methods rather than
    // being pushed through the stringly-typed escape hatch.

    /// Wall-clock limit for each [`Self::solve`] call, in seconds.
    /// `u32::MAX` means "effectively unbounded"; `0` means "stop immediately".
    fn set_time_limit_seconds(&mut self, seconds: u32);

    /// Solver log verbosity. `0` silences the solver; higher is more verbose.
    /// The scale is backend-defined beyond `0 == quiet`.
    fn set_log_level(&mut self, level: u32);

    /// Redirect the solver's own log to `path` (and turn it back on, since
    /// every extractor calls `set_log_level(0)` for CBC parity).
    ///
    /// This exists for one reason: the incumbent-vs-time trajectory. Gurobi and
    /// HiGHS both print a timestamped line every time the incumbent or the
    /// bound moves, so a 1-hour run's log also contains its 10-minute state.
    /// Parsing that (see [`super::milp::trajectory`]) is much cheaper than
    /// wiring three different callback APIs.
    ///
    /// The default implementation is for backends with no log-file parameter
    /// (CBC writes to stdout only): it warns loudly rather than silently
    /// producing no file, because an empty trajectory that looks like "the
    /// solver never improved" would be actively misleading.
    fn set_log_file(&mut self, path: &str) {
        let msg = format!(
            "{}: this backend has no log-file parameter; the requested MILP log \
             {path:?} will NOT be written (no incumbent trajectory for this run)",
            Self::NAME
        );
        log::warn!("{msg}");
        eprintln!("WARNING: {msg}");
    }

    /// Number of threads the solver may use for one `solve()`.
    ///
    /// Every extractor in this crate calls this with `1` unless the caller asks
    /// for more, and `1` is what each backend's `new()` pins. That default is a
    /// safety property, not a preference:
    ///
    /// > **threads x concurrent solver processes must not exceed the process's
    /// > CPU allocation.**
    ///
    /// Both HiGHS (`threads = 0`) and Gurobi (`Threads = 0`) default to "use
    /// every core on the machine", ignoring any cgroup or scheduler allocation.
    /// Running N such processes under one allocation oversubscribes it N-fold.
    ///
    /// The default implementation is for single-threaded backends (CBC): it
    /// accepts the call, does nothing, and warns loudly if more than one thread
    /// was requested so nobody believes they got parallelism they did not get.
    fn set_threads(&mut self, threads: u32) {
        if threads > 1 {
            let msg = format!(
                "{}: this backend is single-threaded and has no thread-count parameter; \
                 the requested {threads} solver threads are IGNORED (the solve stays on 1 thread)",
                Self::NAME
            );
            log::warn!("{msg}");
            eprintln!("WARNING: {msg}");
        }
    }

    /// Backend-specific parameter escape hatch. Keys and values are passed
    /// through verbatim; unknown keys are ignored (best-effort). Provided so a
    /// backend-specific tuning knob does not require widening the trait.
    fn set_raw_parameter(&mut self, key: &str, value: &str);

    // -- warm start ---------------------------------------------------------

    /// Whether this backend's MIP start can be trusted.
    ///
    /// `false` means the extractors must run **without** a warm start and say
    /// so, rather than seeding one and hoping. Only CBC returns `false`; see
    /// [`cbc::CbcModel::supports_warm_start`] for the evidence. Reporting an
    /// unsupported configuration is always preferable to silently returning a
    /// wrong answer, which is exactly what CBC's MIP start has been observed
    /// to do here.
    fn supports_warm_start(&self) -> bool {
        true
    }

    /// Seed one column of the starting incumbent.
    fn set_col_initial_solution(&mut self, col: Self::Col, value: f64);
    /// Seed the starting incumbent from a previous solution of this model.
    /// Backends may misbehave if the model has gained columns since `solution`
    /// was produced (CBC crashes) — the caller is responsible for that.
    fn set_initial_solution(&mut self, solution: &Self::Solution);

    // -- solving ------------------------------------------------------------

    /// Solve the model as it currently stands and return an owned solution.
    /// The model remains usable and mutable afterwards, so callers may add
    /// rows/columns and call `solve` again.
    fn solve(&mut self) -> Self::Solution;
}

// --- Shared backend conformance tests ---------------------------------------

/// Backend-agnostic checks that a new [`MilpModel`] implementation actually
/// honours the two contract points that are easy to get silently wrong:
/// the wall-clock limit, and incremental row addition to a *live* model.
///
/// Deliberately not gated on a particular backend feature so a third backend
/// can reuse it; the per-backend test module just instantiates these.
#[cfg(test)]
pub(crate) mod conformance {
    use super::*;
    use std::time::Instant;

    /// Deterministic pseudo-random ints, so the "hard" instance below is the
    /// same on every machine and every run (a flaky timing test is worse than
    /// no timing test).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    /// A Cornuejols-Dawande "market split" instance: `rows` equality
    /// constraints over `cols` binary variables with coefficients in `[0, 99]`
    /// and right-hand sides at half the row sum. These are the standard
    /// small-but-brutal MIPs — a few hundred bytes of model that no solver can
    /// close quickly — which is exactly what a time-limit test needs.
    fn build_market_split<M: MilpModel>(m: &mut M, rows: usize, cols: usize) {
        let mut rng = Lcg(0x5EED_1234_ABCD_0001);
        let vars: Vec<M::Col> = (0..cols).map(|_| m.add_binary()).collect();
        // Non-trivial objective so the solver has to prove optimality, not just
        // find any feasible point.
        for (j, &v) in vars.iter().enumerate() {
            m.set_obj_coeff(v, 1.0 + (j % 7) as f64);
        }
        for _ in 0..rows {
            let coeffs: Vec<f64> = (0..cols).map(|_| (rng.next() % 100) as f64).collect();
            let rhs = (coeffs.iter().sum::<f64>() / 2.0).floor();
            let row = m.add_row();
            m.set_row_equal(row, rhs);
            for (j, &c) in coeffs.iter().enumerate() {
                if c != 0.0 {
                    m.set_weight(row, vars[j], c);
                }
            }
        }
    }

    fn timed_solve<M: MilpModel>(limit: u32) -> (f64, bool) {
        let mut m = M::new();
        m.set_log_level(0);
        build_market_split(&mut m, 6, 50);
        m.set_time_limit_seconds(limit);
        let t = Instant::now();
        let sol = m.solve();
        (t.elapsed().as_secs_f64(), sol.ran_to_completion())
    }

    /// The limit must be *honoured* and must *scale with its value* — a backend
    /// that ignored `set_time_limit_seconds` but happened to be fast, or one
    /// that hard-coded some limit of its own, would pass a single-point check.
    pub fn time_limit_is_respected<M: MilpModel>() {
        let (t_short, done_short) = timed_solve::<M>(3);
        assert!(
            !done_short,
            "{}: the 'hard' market-split instance was solved to completion inside 3s, \
             so this test proves nothing about the time limit; make the instance harder",
            M::NAME
        );
        assert!(
            (2.0..8.0).contains(&t_short),
            "{}: 3s time limit produced a {t_short:.2}s solve",
            M::NAME
        );

        let (t_long, _) = timed_solve::<M>(9);
        assert!(
            (8.0..16.0).contains(&t_long),
            "{}: 9s time limit produced a {t_long:.2}s solve",
            M::NAME
        );
        assert!(
            t_long > t_short + 3.0,
            "{}: 3s limit took {t_short:.2}s and 9s limit took {t_long:.2}s -- the limit \
             value is not actually reaching the solver",
            M::NAME
        );
    }

    /// Rows added after a solve must land on the *same* model object and take
    /// effect on the next solve. This is what `faster_ilp_cbc`'s cycle-breaking
    /// loop does, and a backend that rebuilt the model from the staging buffer
    /// each time would also pass a naive "does it solve" test.
    pub fn incremental_rows_on_live_model<M: MilpModel>() {
        let mut m = M::new();
        m.set_log_level(0);
        let x = m.add_binary();
        let y = m.add_binary();
        // maximise-by-minimising-negatives: min -x - 2y  =>  wants x=y=1.
        m.set_obj_coeff(x, -1.0);
        m.set_obj_coeff(y, -2.0);

        let s1 = m.solve();
        assert!(s1.has_solution(), "{}: first solve found nothing", M::NAME);
        assert_eq!(s1.col(x), 1.0, "{}: expected x=1", M::NAME);
        assert_eq!(s1.col(y), 1.0, "{}: expected y=1", M::NAME);
        assert!((s1.obj_value() - (-3.0)).abs() < 1e-9, "{}: obj", M::NAME);

        // x + y <= 1, added to the live model, then re-solve.
        let r = m.add_row();
        m.set_row_upper(r, 1.0);
        m.set_weight(r, x, 1.0);
        m.set_weight(r, y, 1.0);

        let s2 = m.solve();
        assert!(s2.has_solution(), "{}: second solve found nothing", M::NAME);
        assert!(
            (s2.obj_value() - (-2.0)).abs() < 1e-9,
            "{}: the row added between solves did not take effect (obj {})",
            M::NAME,
            s2.obj_value()
        );

        // And again: x + y <= 0 forces the empty selection.
        let r2 = m.add_row();
        m.set_row_upper(r2, 0.0);
        m.set_weight(r2, x, 1.0);
        m.set_weight(r2, y, 1.0);
        let s3 = m.solve();
        assert!(s3.has_solution(), "{}: third solve found nothing", M::NAME);
        assert!(
            s3.obj_value().abs() < 1e-9,
            "{}: second incremental row did not take effect (obj {})",
            M::NAME,
            s3.obj_value()
        );
    }


    /// A warm start must be accepted and must not corrupt the answer. This
    /// cannot distinguish "used the start" from "ignored the start" — see the
    /// HiGHS module docs for why that distinction is checked by reading the
    /// solver log instead.
    pub fn warm_start_does_not_break_the_answer<M: MilpModel>() {
        let mut m = M::new();
        m.set_log_level(0);
        let x = m.add_binary();
        let y = m.add_binary();
        m.set_obj_coeff(x, -1.0);
        m.set_obj_coeff(y, -2.0);
        let r = m.add_row();
        m.set_row_upper(r, 1.0);
        m.set_weight(r, x, 1.0);
        m.set_weight(r, y, 1.0);

        // Feasible but suboptimal start: x=1, y=0 (objective -1; optimum -2).
        m.set_col_initial_solution(x, 1.0);
        m.set_col_initial_solution(y, 0.0);

        let s = m.solve();
        assert!(s.has_solution(), "{}: no solution with a warm start", M::NAME);
        assert!(
            (s.obj_value() - (-2.0)).abs() < 1e-9,
            "{}: warm start changed the optimum (obj {})",
            M::NAME,
            s.obj_value()
        );
    }
}
