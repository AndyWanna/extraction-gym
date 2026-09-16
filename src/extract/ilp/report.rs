use std::time::Duration;

use super::WarmStart;
use crate::milp::trajectory::Incumbent;
use crate::milp::MilpStatus;

/// Where an ILP extractor's returned extraction came from.
#[derive(Debug, Clone, PartialEq)]
pub enum SolveOutcome {
    /// The solver proved it optimal.
    Optimal,
    /// The solver stopped early (e.g. time limit) and this is its best
    /// feasible solution, which was no worse than the fallback.
    Incumbent,
    /// The solver produced nothing usable (or something worse than the
    /// fallback), so the fallback extraction was returned.
    Fallback { reason: String },
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
    /// Which objective this report describes (`IlpObjective::name`).
    pub model: &'static str,
    /// Threads actually requested of the solver.
    pub threads: u32,
    /// Wall-clock limit handed to the solver (`None` = unbounded).
    pub time_limit: Option<Duration>,
    /// The mode the caller asked for.
    pub warm_start_requested: WarmStart,
    /// The mode that was actually applied. Differs from the request when the
    /// seed failed validation or the backend does not support MIP starts.
    pub warm_start_applied: WarmStart,
    /// Human-readable reason whenever `applied != requested`, and any repair
    /// statistics when they are equal.
    pub warm_start_note: Option<String>,
    /// Objective value of the seed that was actually applied. Lets a reader
    /// see how much of the final objective the warm start already accounted
    /// for.
    pub warm_start_objective: Option<f64>,
    /// Objective value of the fallback extraction, i.e. what is returned when
    /// the solver produces nothing usable. Reported unconditionally.
    pub fallback_objective: Option<f64>,
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
    /// Wall-clock time to extract and validate the initial expression, if it
    /// was needed.
    pub initial_wall_secs: Option<f64>,
    /// Wall-clock time to compute the greedy extraction (`WarmStart::Greedy`).
    pub greedy_wall_secs: Option<f64>,
    /// Wall-clock time to build the model and push the warm start.
    pub build_wall_secs: f64,
    /// Wall-clock time spent inside the solve loop (excludes model building).
    pub solve_wall_secs: f64,
    /// How many times `solve()` was called (the cycle-breaking loop re-solves).
    pub num_solves: usize,
    /// Where the returned extraction came from. On `Fallback` the solver
    /// fields above describe the *solver's* attempt, not the returned
    /// extraction.
    pub outcome: SolveOutcome,
    /// The depth budget enforced, with `DepthBudget::Initial` resolved.
    pub depth_budget: Option<f64>,
    /// True when the returned extraction exceeds the depth budget. Only
    /// possible on `Fallback`, when no candidate met the budget.
    pub depth_budget_violated: bool,
    /// Path the solver log was written to, if one was requested.
    pub solver_log: Option<String>,
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
        time_limit: Option<Duration>,
        requested: WarmStart,
    ) -> Self {
        SolveReport {
            backend,
            model,
            threads,
            time_limit,
            warm_start_requested: requested,
            warm_start_applied: WarmStart::None,
            warm_start_note: None,
            warm_start_objective: None,
            fallback_objective: None,
            status: None,
            status_detail: None,
            objective: None,
            best_bound: None,
            gap: None,
            initial_wall_secs: None,
            greedy_wall_secs: None,
            build_wall_secs: 0.0,
            solve_wall_secs: 0.0,
            num_solves: 0,
            outcome: SolveOutcome::Fallback {
                reason: "not solved".to_string(),
            },
            depth_budget: None,
            depth_budget_violated: false,
            solver_log: None,
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
        if self.warm_start_requested != WarmStart::None {
            let msg = format!(
                "warm start ({}) NOT applied for the {} model on {}: {why}; \
                 running without a MIP start",
                self.warm_start_requested, self.model, self.backend
            );
            log::warn!("{msg}");
            eprintln!("WARNING: {msg}");
        }
        self.warm_start_applied = WarmStart::None;
        self.warm_start_note = Some(why);
    }

    /// Record that the warm start was applied, optionally with a note (e.g.
    /// "repaired 3 classes").
    pub fn accept_warm_start(&mut self, applied: WarmStart, note: Option<String>) {
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
        let Some(path) = &self.solver_log else {
            return;
        };
        let traj = crate::milp::trajectory::parse_log_for_backend(
            std::path::Path::new(path),
            self.backend,
        );
        self.time_to_first_incumbent_secs = crate::milp::trajectory::time_to_first_incumbent(&traj);
        self.trajectory = traj;
    }
}
