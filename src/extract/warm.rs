/*!
Warm-start mode and the machine-readable report of what the solver actually
did, shared by every ILP extractor. Seed construction and validation live in
[`super::ilp::warm`].
*/

use crate::milp::trajectory::Incumbent;
use crate::milp::MilpStatus;

/// Which starting incumbent to hand the solver.
///
/// The three values are what a caller selects between when choosing a seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WarmStartMode {
    /// No MIP start. The historical behaviour and still the default.
    None,
    /// Seed from the caller-supplied *initial* extraction — typically the
    /// input design's own mapping. This is the seed that is feasible by
    /// construction under a design-derived delay budget, because that budget
    /// *is* the input design's critical-path delay.
    ///
    /// If the caller supplies no seed, [`super::ilp_cbc`] refuses the warm
    /// start with a note and solves with no MIP start (it used to degrade to
    /// [`Self::Greedy`]; see that variant).
    ///
    #[default]
    Initial,
    /// Seed from a greedy extraction of the same e-graph.
    ///
    /// **Refused by [`super::ilp_cbc`]'s extractors.** Greedy is unbounded in
    /// how much worse than the caller's input mapping it can be, and it was
    /// also the acceptance bar, so a run could ship a result larger than its
    /// own input. Seed selection now happens entirely caller-side. Still
    /// honoured by [`super::faster_ilp_cbc`], which is no longer on
    /// the default extraction path.
    Greedy,
}

impl WarmStartMode {
    pub fn as_str(self) -> &'static str {
        match self {
            WarmStartMode::None => "none",
            WarmStartMode::Initial => "initial",
            WarmStartMode::Greedy => "greedy",
        }
    }
}

impl std::fmt::Display for WarmStartMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for WarmStartMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(WarmStartMode::None),
            "initial" => Ok(WarmStartMode::Initial),
            "greedy" => Ok(WarmStartMode::Greedy),
            other => Err(format!(
                "unknown warm-start mode {other:?} (expected none|initial|greedy)"
            )),
        }
    }
}

/// Everything a caller needs to tell one ILP extraction run apart from another
/// in the benchmark matrix, plus enough solver state to say whether a *timed
/// out* run got closer than its neighbour.
///
/// Final area alone cannot answer that question on instances where every
/// configuration times out, which is why the objective, the bound, the gap and
/// the time-to-first-incumbent are all here.
#[derive(Debug, Clone)]
pub struct SolveReport {
    /// `MilpModel::NAME` of the backend that was compiled in.
    pub backend: &'static str,
    /// Which model this report describes (`"area-ilp"`, `"delay-budget"`, ...).
    pub model: &'static str,
    /// Threads actually requested of the solver.
    pub threads: u32,
    /// Wall-clock limit handed to the solver, in seconds.
    pub timeout_seconds: u32,
    /// The mode the caller asked for.
    pub warm_start_requested: WarmStartMode,
    /// The mode that was actually applied. Differs from the request when the
    /// seed failed validation, when the backend does not support MIP starts, or
    /// when `Initial` degraded to `Greedy` for want of a seed.
    pub warm_start_applied: WarmStartMode,
    /// Human-readable reason whenever `applied != requested`, and any repair
    /// statistics when they are equal.
    pub warm_start_note: Option<String>,
    /// DAG cost of the seed that was actually applied. Lets a reader see how
    /// much of the final objective the warm start already accounted for.
    pub warm_start_objective: Option<f64>,
    /// DAG cost of the greedy fallback extraction, i.e. what is returned when
    /// the solver's answer is discarded. Reported unconditionally so a
    /// `returned = initial_fallback` row is still interpretable.
    pub initial_cost: Option<f64>,
    /// Solver-neutral outcome of the final solve.
    pub status: Option<MilpStatus>,
    /// Backend's own status text, for debugging.
    pub status_detail: Option<String>,
    /// Objective value of the final incumbent, if there was one.
    pub objective: Option<f64>,
    /// Best proven bound at the moment the solve stopped.
    pub best_bound: Option<f64>,
    /// `|obj - bound| / max(|obj|, 1e-10)`.
    pub gap: Option<f64>,
    /// Wall-clock time spent inside the solve loop (excludes model building).
    pub solve_wall_secs: f64,
    /// How many times `solve()` was called (the cycle-breaking loop re-solves).
    pub num_solves: usize,
    /// True when the extractor discarded the solver's answer and returned the
    /// greedy fallback instead — the objective below is then the *solver's*,
    /// not the returned extraction's.
    pub returned_fallback: bool,
    /// Path the solver log was written to, if `--milp-log` was given.
    pub milp_log: Option<String>,
    /// Incumbent-vs-time points recovered from that log.
    pub trajectory: Vec<Incumbent>,
    /// Seconds to the first feasible incumbent, from the trajectory.
    pub time_to_first_incumbent_secs: Option<f64>,
}

impl SolveReport {
    pub fn new(
        backend: &'static str,
        model: &'static str,
        threads: u32,
        timeout_seconds: u32,
        requested: WarmStartMode,
    ) -> Self {
        SolveReport {
            backend,
            model,
            threads,
            timeout_seconds,
            warm_start_requested: requested,
            warm_start_applied: WarmStartMode::None,
            warm_start_note: None,
            warm_start_objective: None,
            initial_cost: None,
            status: None,
            status_detail: None,
            objective: None,
            best_bound: None,
            gap: None,
            solve_wall_secs: 0.0,
            num_solves: 0,
            returned_fallback: false,
            milp_log: None,
            trajectory: Vec::new(),
            time_to_first_incumbent_secs: None,
        }
    }

    /// Record that the warm start was refused, and why. Logged at `warn` *and*
    /// on stderr: `env_logger`'s default filter hides warnings, and a silently
    /// dropped warm start would make a whole column of the benchmark matrix
    /// quietly identical to the `none` column.
    pub fn refuse_warm_start(&mut self, why: impl Into<String>) {
        let why = why.into();
        if self.warm_start_requested != WarmStartMode::None {
            let msg = format!(
                "warm start ({}) NOT applied for the {} model on {}: {why}; \
                 running without a MIP start",
                self.warm_start_requested, self.model, self.backend
            );
            log::warn!("{msg}");
            eprintln!("WARNING: {msg}");
        }
        self.warm_start_applied = WarmStartMode::None;
        self.warm_start_note = Some(why);
    }

    /// Record that the warm start was applied, optionally with a note (e.g.
    /// "repaired 3 classes", or "degraded initial -> greedy").
    pub fn accept_warm_start(&mut self, applied: WarmStartMode, note: Option<String>) {
        self.warm_start_applied = applied;
        if let Some(n) = &note {
            log::info!(
                "warm start ({applied}) applied for the {} model on {}: {n}",
                self.model,
                self.backend
            );
        }
        self.warm_start_note = note;
    }

    /// Fill in the solver-state fields from the final solution.
    pub fn record_solution<S: crate::milp::MilpSolution>(&mut self, sol: &S) {
        self.status = Some(sol.status());
        self.status_detail = Some(sol.status_detail());
        self.objective = sol.has_solution().then(|| sol.obj_value());
        self.best_bound = sol.best_bound();
        self.gap = sol.gap();
    }

    /// Parse the solver log (if any) into a trajectory and derive the
    /// time-to-first-incumbent from it.
    pub fn load_trajectory(&mut self) {
        let Some(path) = &self.milp_log else { return };
        let traj = crate::milp::trajectory::parse_log_for_backend(
            std::path::Path::new(path),
            self.backend,
        );
        self.time_to_first_incumbent_secs = crate::milp::trajectory::time_to_first_incumbent(&traj);
        self.trajectory = traj;
    }
}

pub use super::ilp::warm::{arrival_times, build_seed, topological_levels, try_dag_cost, Seed};
