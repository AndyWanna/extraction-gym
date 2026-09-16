/*!
ILP extraction.

Four extractors, each one [`IlpObjective`] over the single model in
[`model`](self::model):

| Extractor | Minimises | With area / delay weights |
|---|---|---|
| [`SizeExtractor`] | size | area |
| [`DepthExtractor`] | depth | critical-path delay |
| [`SizeConstrainedDepthExtractor`] | size, subject to depth `<= depth_budget` | area without worsening delay |
| [`WeightedSizeDepthExtractor`] | `size_weight * size + depth_weight * depth` | area/delay trade-off |

Every extractor takes [`IlpOptions`] (time limit, default
[`DEFAULT_TIME_LIMIT`] = 10 s; threads; [`WarmStart`], default initial) and
always returns a complete extraction: if the solver fails, the better of the
initial and greedy extractions is returned instead (see [`solve`]).

Sizes come from `Node::cost`, depths from `Node::delay`, and the initial
expression from `Node::initial`:

```ignore
let options = IlpOptions::default().with_time_limit(Duration::from_secs(60)); // default 10 s
let outcome = SizeConstrainedDepthExtractor { depth_budget, options }.solve(&egraph, &roots)?;
let extraction = outcome.extraction;   // CompleteExtraction
let report = outcome.report;           // SolveReport: outcome, status, gap, ...
```
*/

pub(crate) mod decode;
pub(crate) mod fallback;
pub(crate) mod model;
pub mod objective;
pub mod options;
pub mod report;
pub mod solve;
pub mod warm;

pub use objective::IlpObjective;
pub use options::{IlpOptions, WarmStart, DEFAULT_TIME_LIMIT};
pub use report::{SolveOutcome, SolveReport};
pub use solve::{solve, IlpError, IlpOutcome};

use crate::milp::{DefaultMilp, MilpModel};
use crate::*;

/// Minimises size. See [`IlpObjective::Size`].
#[derive(Debug, Clone, Default)]
pub struct SizeExtractor {
    pub options: IlpOptions,
}

/// Minimises depth. See [`IlpObjective::Depth`] for why
/// `greedy::GreedyDepthExtractor` is usually the better choice.
#[derive(Debug, Clone, Default)]
pub struct DepthExtractor {
    pub options: IlpOptions,
}

/// Minimises size subject to depth `<= depth_budget`. See
/// [`IlpObjective::SizeConstrainedDepth`].
#[derive(Debug, Clone)]
pub struct SizeConstrainedDepthExtractor {
    pub depth_budget: f64,
    pub options: IlpOptions,
}

/// Minimises `size_weight * size + depth_weight * depth`. See
/// [`IlpObjective::WeightedSizeDepth`].
#[derive(Debug, Clone)]
pub struct WeightedSizeDepthExtractor {
    pub size_weight: f64,
    pub depth_weight: f64,
    pub options: IlpOptions,
}

macro_rules! ilp_extractor {
    ($ty:ident, |$s:ident| $objective:expr) => {
        impl $ty {
            pub fn objective(&self) -> IlpObjective {
                let $s = self;
                $objective
            }

            /// Solve on the compiled-in backend.
            pub fn solve(
                &self,
                egraph: &EGraph,
                roots: &[ClassId],
            ) -> Result<IlpOutcome, IlpError> {
                self.solve_with::<DefaultMilp>(egraph, roots)
            }

            /// Solve on backend `M`.
            pub fn solve_with<M: MilpModel>(
                &self,
                egraph: &EGraph,
                roots: &[ClassId],
            ) -> Result<IlpOutcome, IlpError> {
                solve::<M>(egraph, roots, self.objective(), &self.options)
            }
        }

        /// Panics if the request is invalid ([`IlpError`]); never because of
        /// the solver. Use [`Self::solve`] to handle the error or read the
        /// report.
        impl Extractor for $ty {
            fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
                match self.solve(egraph, roots) {
                    Ok(outcome) => outcome.extraction.into_inner(),
                    Err(e) => panic!("{} ILP extraction: {e}", self.objective().name()),
                }
            }
        }
    };
}

ilp_extractor!(SizeExtractor, |_s| IlpObjective::Size);
ilp_extractor!(DepthExtractor, |_s| IlpObjective::Depth);
ilp_extractor!(SizeConstrainedDepthExtractor, |s| {
    IlpObjective::SizeConstrainedDepth {
        depth_budget: s.depth_budget,
    }
});
ilp_extractor!(WeightedSizeDepthExtractor, |s| {
    IlpObjective::WeightedSizeDepth {
        size_weight: s.size_weight,
        depth_weight: s.depth_weight,
    }
});

#[cfg(test)]
mod tests;
