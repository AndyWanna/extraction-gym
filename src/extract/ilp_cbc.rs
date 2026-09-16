/* The DAG-optimal ILP extractors, under their original names.

Each struct here is one `ilp::IlpObjective` solved by `ilp::solve`:

  IlpExtractor              Size
  DualIlpExtractor          WeightedSizeDepth { size_weight: beta, depth_weight: alpha }
  DelayBudgetIlpExtractor   SizeConstrainedDepth { depth_budget: max_delay }

so they share its guarantee: the result is complete unless the input itself is
invalid (then it is empty). New code should use the `ilp` extractors directly.

The public extractor structs default to `milp::DefaultMilp` (the one backend
feature that is enabled), and `*_with::<M>()` methods let a caller pin a
backend explicitly. The historical `*CbcExtractor` names are kept as type
aliases so downstream crates don't have to change.
*/

use super::faster_ilp_cbc::WarmConfig;
use super::ilp::options::time_limit_from_seconds;
use super::ilp::solve::solve_with_initial;
use super::ilp::{warm as seed, IlpObjective, IlpOptions, SolveOutcome, SolveReport, WarmStart};
use super::*;
use crate::milp::{DefaultMilp, MilpModel};
use std::path::PathBuf;

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

/// Adapter from the legacy fields onto [`ilp::solve`](super::ilp::solve).
///
/// `warm.seed` plays the role of the initial extraction (repaired onto this
/// e-graph by `build_seed`). `WarmStartMode::Initial` without a seed runs with
/// no MIP start, as it always has.
fn run<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    objective: IlpObjective,
    timeout_seconds: u32,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let initial = warm.seed.as_ref().and_then(|preferred| {
        let seed = seed::build_seed(egraph, roots, Some(preferred), &ExtractionResult::default());
        match seed.and_then(|s| CompleteExtraction::try_new(s.selection, egraph, roots)) {
            Ok(e) => Some(e),
            Err(why) => {
                log::warn!("ignoring the supplied seed: {why}");
                None
            }
        }
    });
    let warm_start = match (warm.mode, &initial) {
        (WarmStart::Initial, None) => WarmStart::None,
        (mode, _) => mode,
    };
    let options = IlpOptions {
        time_limit: time_limit_from_seconds(timeout_seconds),
        threads,
        warm_start,
        solver_log: warm.milp_log.as_ref().map(PathBuf::from),
        raw_params: Vec::new(),
    };
    match solve_with_initial::<M>(egraph, roots, objective, &options, initial) {
        Ok(outcome) => (outcome.extraction.into_inner(), outcome.report),
        Err(e) => {
            log::warn!(
                "{} ILP extraction: {e}; returning an empty selection",
                objective.name()
            );
            let mut report = SolveReport::new(
                M::NAME,
                objective.name(),
                threads,
                options.time_limit,
                warm.mode,
            );
            report.outcome = SolveOutcome::Fallback {
                reason: e.to_string(),
            };
            (ExtractionResult::default(), report)
        }
    }
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
