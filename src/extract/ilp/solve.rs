/*!
The only entry point that runs an ILP solve, and the only place that decides
whether the solver's answer or the fallback is returned.
*/

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Once;
use std::time::Instant;

use super::decode::decode_selection;
use super::fallback::{self, Candidate};
use super::model::{self, IlpVars};
use super::{warm, DepthBudget, IlpObjective, IlpOptions, SolveOutcome, SolveReport, WarmStart};
use crate::extract::greedy::initial::{has_initial_nodes, InitialExtractor};
use crate::milp::{MilpModel, MilpSolution};
use crate::*;

/// A complete extraction and how it was obtained.
pub struct IlpOutcome {
    pub extraction: CompleteExtraction,
    pub report: SolveReport,
}

/// Why an ILP extraction produced no extraction.
#[derive(Debug, Clone)]
pub enum IlpError {
    NoRoots,
    /// A negative or non-finite size/depth, weight or budget.
    InvalidInput(String),
    /// `WarmStart::Initial` or `DepthBudget::Initial` was requested but the
    /// e-graph's `Node::initial` flags do not describe a complete extraction
    /// of the roots.
    MissingInitial(String),
    /// The e-graph has no acyclic extraction of the roots at all.
    NoValidExtraction(String),
    /// The solver produced no usable solution and, with `WarmStart::None`,
    /// there is no fallback extraction to return.
    NoSolution {
        reason: String,
        report: Box<SolveReport>,
    },
}

impl std::fmt::Display for IlpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IlpError::NoRoots => write!(f, "no roots to extract"),
            IlpError::InvalidInput(why) => write!(f, "invalid input: {why}"),
            IlpError::MissingInitial(why) => write!(
                f,
                "the initial expression is needed but not flagged with Node::initial: {why}"
            ),
            IlpError::NoValidExtraction(why) => {
                write!(f, "the e-graph has no valid extraction: {why}")
            }
            IlpError::NoSolution { reason, .. } => write!(
                f,
                "{reason}, and WarmStart::None has no fallback extraction"
            ),
        }
    }
}

impl std::error::Error for IlpError {}

/// Extract `roots` from `egraph` minimising `objective`.
///
/// The warm start also decides the fallback, returned when the solver
/// produces nothing usable (no solution within the time limit, an infeasible
/// depth budget, an invalid solution, an incumbent worse than the fallback, a
/// panic in the backend):
///
/// | `WarmStart` | Fallback |
/// |---|---|
/// | `None` | none: `IlpError::NoSolution` |
/// | `Initial` | the initial extraction |
/// | `Greedy` | the better of the initial (if flagged) and greedy extractions |
///
/// Only what the warm start needs is computed, since greedy extraction can be
/// slow on large e-graphs. See [`SolveReport::outcome`].
pub fn solve<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    objective: IlpObjective,
    options: &IlpOptions,
) -> Result<IlpOutcome, IlpError> {
    let needs_initial = options.warm_start != WarmStart::None
        || objective.depth_budget() == Some(DepthBudget::Initial);
    let required = options.warm_start == WarmStart::Initial
        || objective.depth_budget() == Some(DepthBudget::Initial);

    let clock = Instant::now();
    let initial = if !needs_initial {
        None
    } else if !has_initial_nodes(egraph) {
        None
    } else {
        let extraction = InitialExtractor.extract(egraph, roots);
        match CompleteExtraction::try_new(extraction, egraph, roots) {
            Ok(e) => Some(e),
            Err(why) if required => return Err(IlpError::MissingInitial(why)),
            Err(why) => {
                log::warn!("ignoring the flagged initial expression: {why}");
                None
            }
        }
    };
    if required && initial.is_none() {
        return Err(IlpError::MissingInitial("no node is flagged".to_string()));
    }
    let initial_wall_secs = needs_initial.then(|| clock.elapsed().as_secs_f64());
    solve_with_initial::<M>(
        egraph,
        roots,
        objective,
        options,
        initial,
        initial_wall_secs,
    )
}

/// [`solve`] with the initial extraction supplied by the caller rather than
/// read from `Node::initial`. For the legacy extractors, which take a seed.
pub(crate) fn solve_with_initial<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    objective: IlpObjective,
    options: &IlpOptions,
    initial: Option<CompleteExtraction>,
    initial_wall_secs: Option<f64>,
) -> Result<IlpOutcome, IlpError> {
    let initial_depth = initial
        .as_ref()
        .map(|e| e.dag_depth(egraph, roots).into_inner());
    let objective = objective.resolve_budget(initial_depth).ok_or_else(|| {
        IlpError::MissingInitial("DepthBudget::Initial needs the initial expression".to_string())
    })?;
    validate(egraph, roots, objective)?;
    if objective == IlpObjective::Depth {
        warn_depth_objective();
    }

    let mut report = SolveReport::new(
        M::NAME,
        objective.name(),
        options.threads,
        options.time_limit,
        options.warm_start,
    );
    report.depth_budget = objective.budget_value();
    report.initial_wall_secs = initial_wall_secs;

    let with_greedy = options.warm_start == WarmStart::Greedy;
    let clock = Instant::now();
    let candidates = fallback::candidates(egraph, roots, objective, initial, with_greedy)?;
    if with_greedy {
        report.greedy_wall_secs = Some(clock.elapsed().as_secs_f64());
    }
    let fallback = fallback::best(objective, &candidates);
    let (best, budget_violated) = match fallback {
        Some((best, violated)) => (Some(&candidates[best]), violated),
        None => (None, false),
    };
    report.fallback_objective = best.map(|c| c.objective);

    let seed = match options.warm_start {
        WarmStart::None => None,
        WarmStart::Initial | WarmStart::Greedy => best,
    };

    let solved = catch_unwind(AssertUnwindSafe(|| {
        run_solver::<M>(egraph, roots, objective, options, seed, &mut report)
    }))
    .unwrap_or_else(|panic| {
        let msg = panic
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| panic.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        Err(format!("the solver panicked: {msg}"))
    });

    let reason = match (solved, best) {
        (Ok((extraction, true)), _) => {
            report.outcome = SolveOutcome::Optimal;
            return Ok(IlpOutcome { extraction, report });
        }
        (Ok((extraction, false)), best) => {
            let value = objective.evaluate(egraph, roots, &extraction);
            match best {
                Some(b) if !budget_violated && value > b.objective + EPSILON_ALLOWANCE => format!(
                    "the solver's incumbent ({value}) is worse than the {} extraction ({})",
                    b.name, b.objective
                ),
                _ => {
                    report.outcome = SolveOutcome::Incumbent;
                    return Ok(IlpOutcome { extraction, report });
                }
            }
        }
        (Err(reason), _) => reason,
    };

    let Some((best, _)) = fallback else {
        report.outcome = SolveOutcome::Fallback {
            reason: reason.clone(),
        };
        return Err(IlpError::NoSolution {
            reason,
            report: Box::new(report),
        });
    };
    log::warn!(
        "{} ILP on {}: {reason}; returning the {} extraction",
        objective.name(),
        M::NAME,
        candidates[best].name
    );
    report.outcome = SolveOutcome::Fallback { reason };
    report.depth_budget_violated = budget_violated;
    let extraction = candidates.into_iter().nth(best).unwrap().extraction;
    Ok(IlpOutcome { extraction, report })
}

fn validate(egraph: &EGraph, roots: &[ClassId], objective: IlpObjective) -> Result<(), IlpError> {
    if roots.is_empty() {
        return Err(IlpError::NoRoots);
    }
    let non_negative = |what: &str, v: f64| {
        if v.is_finite() && v >= 0.0 {
            Ok(())
        } else {
            Err(IlpError::InvalidInput(format!("{what} is {v}")))
        }
    };
    for (nid, node) in &egraph.nodes {
        non_negative(&format!("size of node {nid}"), node.cost.into_inner())?;
        non_negative(&format!("depth of node {nid}"), node.delay.into_inner())?;
    }
    let (size_weight, depth_weight) = objective.weights();
    non_negative("size_weight", size_weight)?;
    non_negative("depth_weight", depth_weight)?;
    if let Some(budget) = objective.budget_value() {
        non_negative("depth_budget", budget)?;
    }
    Ok(())
}

fn warn_depth_objective() {
    static WARN: Once = Once::new();
    WARN.call_once(|| {
        let msg = "the Depth ILP objective is solved exactly and much faster by \
                   greedy::GreedyDepthExtractor, and as an ILP it is degenerate (slow to \
                   prove optimal, may return needlessly large extractions). Prefer the \
                   greedy extractor, or WeightedSizeDepth with a small size_weight to break \
                   ties towards smaller extractions";
        log::warn!("{msg}");
        eprintln!("WARNING: {msg}");
    });
}

/// Build, seed and solve the model. `Ok((extraction, proven_optimal))`, or
/// `Err(why)` when there is no usable solution.
fn run_solver<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    objective: IlpObjective,
    options: &IlpOptions,
    seed: Option<&Candidate>,
    report: &mut SolveReport,
) -> Result<(CompleteExtraction, bool), String> {
    let mut model = M::new();
    model.set_time_limit_seconds(options.time_limit_seconds());
    model.set_threads(options.threads);
    for (key, value) in &options.raw_params {
        model.set_raw_parameter(key, value);
    }
    if let Some(path) = &options.solver_log {
        let path = path.to_string_lossy().into_owned();
        model.set_log_file(&path);
        report.solver_log = Some(path);
    }

    let build_clock = Instant::now();
    let vars = model::build(&mut model, egraph, roots, objective);
    if let Some(seed) = seed {
        apply_seed(
            &mut model,
            egraph,
            roots,
            &vars,
            seed,
            options.warm_start,
            report,
        );
    }

    report.build_wall_secs = build_clock.elapsed().as_secs_f64();

    let solve_clock = Instant::now();
    let solution = model.solve();
    report.num_solves = 1;
    report.solve_wall_secs = solve_clock.elapsed().as_secs_f64();
    report.record_solution(&solution);
    report.load_trajectory();
    log::info!(
        "{} ILP on {}: status {}",
        objective.name(),
        M::NAME,
        solution.status_detail()
    );

    if !solution.has_solution() {
        return Err(format!(
            "{} returned no solution ({})",
            M::NAME,
            solution.status_detail()
        ));
    }
    let selection = decode_selection(&solution, egraph, &vars.classes);
    let extraction = CompleteExtraction::try_new(selection, egraph, roots)
        .map_err(|why| format!("the solver's solution is not a valid extraction: {why}"))?;
    let proven_optimal = solution.ran_to_completion() && !solution.is_infeasible();
    Ok((extraction, proven_optimal))
}

/// Hand `seed` to the solver, or record why not. Never fails the solve.
fn apply_seed<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
    roots: &[ClassId],
    vars: &IlpVars<M::Col>,
    seed: &Candidate,
    mode: WarmStart,
    report: &mut SolveReport,
) {
    if !model.supports_warm_start() {
        return report.refuse_warm_start(format!(
            "{} does not support a trustworthy MIP start (see CbcModel::supports_warm_start)",
            M::NAME
        ));
    }

    let arrival = match &vars.depth {
        None => None,
        Some(_) => match warm::arrival_times(egraph, roots, &seed.extraction) {
            Ok(t) => Some(t),
            Err(why) => return report.refuse_warm_start(why),
        },
    };

    // The budget caps EVERY arrival column, so a single class over budget
    // makes the point infeasible.
    let mut note = format!("seeded from the {} extraction", seed.name);
    if let (Some(budget), Some(times)) = (vars.depth.as_ref().and_then(|d| d.budget), &arrival) {
        let worst = times.values().cloned().fold(0.0f64, f64::max);
        if worst > budget {
            return report.refuse_warm_start(format!(
                "the {} extraction violates the depth budget: depth {worst:.9} > budget \
                 {budget:.9} (over by {:.3e})",
                seed.name,
                worst - budget
            ));
        }
        note += &format!("; depth slack {:.3e} under the budget", budget - worst);
    }

    warm::push_seed(
        model,
        egraph,
        roots,
        vars,
        &seed.extraction,
        arrival.as_ref(),
    );
    report.warm_start_objective = Some(seed.objective);
    report.accept_warm_start(mode, Some(note));
}
