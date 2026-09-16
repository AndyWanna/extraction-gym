/* The DAG-optimal ILP extractors, under their original names.

Each struct here is one `ilp::IlpObjective` over the shared model in
`ilp::model`:

  IlpExtractor              Size
  DualIlpExtractor          WeightedSizeDepth { size_weight: beta, depth_weight: alpha }
  DelayBudgetIlpExtractor   SizeConstrainedDepth { depth_budget: max_delay }

When the solver produces no solution these return an EMPTY selection for the
caller to floor against its own candidates.

The solver is abstracted behind `crate::milp::MilpModel`, so the same model can
be built for CBC / Gurobi / HiGHS. The public extractor structs default to
`milp::DefaultMilp` (the one backend feature that is enabled), and
`*_with::<M>()` methods let a caller pin a backend explicitly. The historical
`*CbcExtractor` names are kept as type aliases so downstream crates don't have
to change.
*/

use super::faster_ilp_cbc::WarmConfig;
use super::ilp::decode::decode_selection;
use super::ilp::model::{self, IlpVars};
use super::ilp::{warm as seed, IlpObjective};
use super::warm::{SolveReport, WarmStartMode};
use super::*;
use crate::milp::{DefaultMilp, MilpModel, MilpSolution};
use std::time::Instant;

/// DAG-optimal ILP extractor with a compile-time timeout. Backend-generic; use
/// [`CbcExtractorWithTimeout`] for the previous (CBC) spelling.
pub struct IlpExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32>;

/// Legacy name for [`IlpExtractorWithTimeout`].
///
/// NOTE: a type alias to a *unit* struct cannot be used in expression position
/// (`CbcExtractorWithTimeout::<10>` as a value is rejected by rustc), so
/// construct it with `::new()` or use the neutral name directly.
pub type CbcExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32> =
    IlpExtractorWithTimeout<TIMEOUT_IN_SECONDS>;

impl<const TIMEOUT_IN_SECONDS: u32> Extractor for IlpExtractorWithTimeout<TIMEOUT_IN_SECONDS> {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract::<DefaultMilp>(
            egraph,
            roots,
            TIMEOUT_IN_SECONDS,
            1,
            &WarmConfig::default(),
        )
        .0;
    }
}

impl<const TIMEOUT_IN_SECONDS: u32> IlpExtractorWithTimeout<TIMEOUT_IN_SECONDS> {
    /// Constructor, so the type aliases can also be used in expression
    /// position (`CbcExtractorWithTimeout::<10>::new()`).
    pub const fn new() -> Self {
        Self
    }

    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        extract::<M>(egraph, roots, TIMEOUT_IN_SECONDS, 1, &WarmConfig::default()).0
    }
}

/// DAG-optimal ILP extractor with a runtime timeout. Backend-generic; use
/// [`CbcExtractor`] for the previous (CBC) spelling.
pub struct IlpExtractor {
    /// Solver time limit in seconds (`u32::MAX` for unbounded).
    pub timeout_seconds: u32,
    /// Solver threads for each solve. `1` unless the caller opted in; see
    /// [`MilpModel::set_threads`] for the slot-allocation invariant.
    pub threads: u32,
    /// Warm start + solver-log configuration; default = no MIP start, no log.
    pub warm: WarmConfig,
}

impl Default for IlpExtractor {
    fn default() -> Self {
        IlpExtractor {
            timeout_seconds: u32::MAX,
            threads: 1,
            warm: WarmConfig::default(),
        }
    }
}

/// Legacy name for [`IlpExtractor`].
pub type CbcExtractor = IlpExtractor;

impl Extractor for IlpExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract::<DefaultMilp>(
            egraph,
            roots,
            self.timeout_seconds,
            self.threads,
            &self.warm,
        )
        .0;
    }
}

impl IlpExtractor {
    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        self.extract_with_report::<M>(egraph, roots).0
    }

    /// Same as [`Self::extract_with`], plus the [`SolveReport`].
    pub fn extract_with_report<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> (ExtractionResult, SolveReport) {
        extract::<M>(
            egraph,
            roots,
            self.timeout_seconds,
            self.threads,
            &self.warm,
        )
    }
}

fn extract<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    run::<M>(
        egraph,
        roots,
        IlpObjective::Size,
        timeout_seconds,
        threads,
        warm,
    )
}

fn extract_dual<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    alpha: f64,
    beta: f64,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let objective = IlpObjective::WeightedSizeDepth {
        size_weight: beta,
        depth_weight: alpha,
    };
    run::<M>(egraph, roots, objective, timeout_seconds, threads, warm)
}

fn extract_delay_budget<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    max_delay: f64,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let objective = IlpObjective::SizeConstrainedDepth {
        depth_budget: max_delay,
    };
    run::<M>(egraph, roots, objective, timeout_seconds, threads, warm)
}

fn run<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    objective: IlpObjective,
    timeout_seconds: u32,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let mut report = SolveReport::new(
        M::NAME,
        objective.name(),
        threads,
        timeout_seconds,
        warm.mode,
    );
    let mut model = M::new();
    model.set_time_limit_seconds(timeout_seconds);
    model.set_threads(threads);
    if let Some(path) = &warm.milp_log {
        model.set_log_file(path);
        report.milp_log = Some(path.clone());
    }

    let vars = model::build(&mut model, egraph, roots, objective);
    apply_warm_start(&mut model, egraph, roots, &vars, warm, &mut report);

    let solve_clock = Instant::now();
    let solution = model.solve();
    report.num_solves = 1;
    report.solve_wall_secs = solve_clock.elapsed().as_secs_f64();
    report.record_solution(&solution);
    report.load_trajectory();
    log::info!(
        "{} {} status {}",
        objective.name(),
        M::NAME,
        solution.status_detail(),
    );

    if !solution.has_solution() {
        log::info!(
            "{} returned no solution ({}); returning an empty selection for the \
             caller to floor against its own candidates",
            M::NAME,
            solution.status_detail()
        );
        report.returned_fallback = true;
        return (ExtractionResult::default(), report);
    }
    if !solution.ran_to_completion() {
        // Keep the incumbent: the model blocks cycles exactly and enforces any
        // depth budget, so every feasible solution is a valid answer.
        log::info!(
            "Unfinished {} solution, but an incumbent exists (obj = {}); returning it",
            M::NAME,
            solution.obj_value(),
        );
    }

    (decode_selection(&solution, egraph, &vars.classes), report)
}

/// Warm-start entry point.
///
/// Everything here is refusable: any problem with the candidate ends in
/// [`SolveReport::refuse_warm_start`] and a solve with no MIP start, never a
/// panic and never an infeasible point pushed to the solver.
fn apply_warm_start<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
    roots: &[ClassId],
    vars: &IlpVars<M::Col>,
    warm: &WarmConfig,
    report: &mut SolveReport,
) {
    let preferred = match warm.mode {
        WarmStartMode::None => return,
        WarmStartMode::Greedy => {
            return report.refuse_warm_start(
                "greedy seeding is not supported by the ilp_cbc extractors; pass a \
                 caller-supplied seed with WarmStartMode::Initial",
            )
        }
        WarmStartMode::Initial => match &warm.seed {
            Some(s) => s,
            None => {
                return report.refuse_warm_start(
                    "no initial extraction was supplied; running with no MIP start",
                )
            }
        },
    };
    if !model.supports_warm_start() {
        return report.refuse_warm_start(format!(
            "{} does not support a trustworthy MIP start (see CbcModel::supports_warm_start)",
            M::NAME
        ));
    }

    // Repair a class the seed cannot fill with its cheapest member.
    let seed = match seed::build_seed(egraph, roots, Some(preferred), &ExtractionResult::default())
    {
        Ok(s) => s,
        Err(why) => return report.refuse_warm_start(why),
    };
    let mut note = (seed.repaired > 0).then(|| {
        format!(
            "{} class(es) repaired off the preferred node",
            seed.repaired
        )
    });

    let arrival = match &vars.depth {
        None => None,
        Some(_) => match seed::arrival_times(egraph, roots, &seed.selection) {
            Ok(t) => Some(t),
            Err(why) => return report.refuse_warm_start(why),
        },
    };

    // The budget caps EVERY arrival column, so a single interior class over
    // budget makes the point infeasible.
    if let (Some(budget), Some(times)) = (vars.depth.as_ref().and_then(|d| d.budget), &arrival) {
        let worst = times.values().cloned().fold(0.0f64, f64::max);
        if worst > budget {
            return report.refuse_warm_start(format!(
                "candidate seed violates the depth budget: worst class arrival {worst:.9} > \
                 budget {budget:.9} (over by {:.3e})",
                worst - budget
            ));
        }
        let extra = format!("depth slack {:.3e} under the budget", budget - worst);
        note = Some(match note {
            Some(n) => format!("{n}; {extra}"),
            None => extra,
        });
    }

    seed::push_seed(
        model,
        egraph,
        roots,
        vars,
        &seed.selection,
        arrival.as_ref(),
    );
    report.warm_start_objective = seed::try_dag_cost(egraph, roots, &seed.selection);
    report.accept_warm_start(WarmStartMode::Initial, note);
}

/// Joint delay+area ILP extractor. Backend-generic; use [`DualCbcExtractor`]
/// for the previous (CBC) spelling.
pub struct DualIlpExtractor {
    /// Solver time limit in seconds (u32::MAX for unbounded).
    pub timeout_seconds: u32,
    /// Weight on the critical-path delay.
    pub alpha: f64,
    /// Weight on the total area.
    pub beta: f64,
    /// Solver threads for each solve. `1` unless the caller opted in; see
    /// [`MilpModel::set_threads`] for the slot-allocation invariant.
    pub threads: u32,
    /// Warm start + solver-log configuration; default = no MIP start, no log.
    pub warm: WarmConfig,
}

impl Default for DualIlpExtractor {
    fn default() -> Self {
        DualIlpExtractor {
            timeout_seconds: u32::MAX,
            alpha: 1.0,
            beta: 1.0,
            threads: 1,
            warm: WarmConfig::default(),
        }
    }
}

/// Legacy name for [`DualIlpExtractor`].
pub type DualCbcExtractor = DualIlpExtractor;

impl Extractor for DualIlpExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        extract_dual::<DefaultMilp>(
            egraph,
            roots,
            self.timeout_seconds,
            self.alpha,
            self.beta,
            self.threads,
            &self.warm,
        )
        .0
    }
}

impl DualIlpExtractor {
    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        self.extract_with_report::<M>(egraph, roots).0
    }

    /// Same as [`Self::extract_with`], plus the [`SolveReport`].
    pub fn extract_with_report<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> (ExtractionResult, SolveReport) {
        extract_dual::<M>(
            egraph,
            roots,
            self.timeout_seconds,
            self.alpha,
            self.beta,
            self.threads,
            &self.warm,
        )
    }
}

/// Minimum-area-under-a-delay-budget ILP extractor. Backend-generic; use
/// [`DelayBudgetCbcExtractor`] for the previous (CBC) spelling.
pub struct DelayBudgetIlpExtractor {
    /// Solver time limit in seconds (u32::MAX for unbounded).
    pub timeout_seconds: u32,
    /// Hard upper bound on the critical-path delay of the extraction.
    pub max_delay: f64,
    /// Solver threads for each solve. `1` unless the caller opted in; see
    /// [`MilpModel::set_threads`] for the slot-allocation invariant.
    pub threads: u32,
    /// Warm start + solver-log configuration; default = no MIP start, no log.
    pub warm: WarmConfig,
}

impl Default for DelayBudgetIlpExtractor {
    fn default() -> Self {
        DelayBudgetIlpExtractor {
            timeout_seconds: u32::MAX,
            max_delay: f64::INFINITY,
            threads: 1,
            warm: WarmConfig::default(),
        }
    }
}

/// Legacy name for [`DelayBudgetIlpExtractor`].
pub type DelayBudgetCbcExtractor = DelayBudgetIlpExtractor;

impl Extractor for DelayBudgetIlpExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        extract_delay_budget::<DefaultMilp>(
            egraph,
            roots,
            self.timeout_seconds,
            self.max_delay,
            self.threads,
            &self.warm,
        )
        .0
    }
}

impl DelayBudgetIlpExtractor {
    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        self.extract_with_report::<M>(egraph, roots).0
    }

    /// Same as [`Self::extract_with`], plus the [`SolveReport`].
    pub fn extract_with_report<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> (ExtractionResult, SolveReport) {
        extract_delay_budget::<M>(
            egraph,
            roots,
            self.timeout_seconds,
            self.max_delay,
            self.threads,
            &self.warm,
        )
    }
}
