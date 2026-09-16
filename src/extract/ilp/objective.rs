/*!
What an ILP extractor minimises.

Vocabulary, used throughout [`super`]:

* **size** of a node is `Node::cost`. The size of an extraction counts every
  shared node once (`ExtractionResult::dag_cost`). With area weights it is area.
* **depth** of a node is `Node::delay`. The depth of an extraction is the
  largest sum of depths along any root-to-leaf path
  (`ExtractionResult::dag_depth`). With delay weights it is the critical-path
  delay.
*/

use crate::extract::greedy::{GreedyDepthExtractor, GreedySizeExtractor};
use crate::*;

/// The quantity an ILP extraction minimises.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IlpObjective {
    /// Minimise size.
    Size,
    /// Minimise depth.
    ///
    /// Provided for completeness only: min depth is solved exactly and far
    /// faster by `greedy::GreedyDepthExtractor`, and as an ILP it is badly
    /// degenerate (anything off the critical path is free), so it is slow to
    /// prove optimal and may return needlessly large extractions. Prefer the
    /// greedy extractor, or [`Self::WeightedSizeDepth`] with a small
    /// `size_weight` to break ties towards smaller extractions.
    Depth,
    /// Minimise size subject to depth `<= depth_budget`.
    SizeConstrainedDepth { depth_budget: f64 },
    /// Minimise `size_weight * size + depth_weight * depth`.
    WeightedSizeDepth { size_weight: f64, depth_weight: f64 },
}

impl IlpObjective {
    pub fn name(&self) -> &'static str {
        match self {
            IlpObjective::Size => "size",
            IlpObjective::Depth => "depth",
            IlpObjective::SizeConstrainedDepth { .. } => "size-constrained-depth",
            IlpObjective::WeightedSizeDepth { .. } => "weighted-size-depth",
        }
    }

    pub fn depth_budget(&self) -> Option<f64> {
        match self {
            IlpObjective::SizeConstrainedDepth { depth_budget } => Some(*depth_budget),
            _ => None,
        }
    }

    /// `(size_weight, depth_weight)` of the objective function.
    pub(crate) fn weights(&self) -> (f64, f64) {
        match *self {
            IlpObjective::Size | IlpObjective::SizeConstrainedDepth { .. } => (1.0, 0.0),
            IlpObjective::Depth => (0.0, 1.0),
            IlpObjective::WeightedSizeDepth {
                size_weight,
                depth_weight,
            } => (size_weight, depth_weight),
        }
    }

    /// Whether the model needs arrival-time (depth) variables.
    pub(crate) fn uses_depth(&self) -> bool {
        !matches!(self, IlpObjective::Size)
    }

    /// Value of this objective for `extraction`, in the same units as the
    /// solver's objective. For [`Self::SizeConstrainedDepth`] this is the
    /// size alone; the budget is checked separately.
    pub fn evaluate(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
        extraction: &CompleteExtraction,
    ) -> f64 {
        let (size_weight, depth_weight) = self.weights();
        let mut value = 0.0;
        if size_weight != 0.0 {
            value += size_weight * extraction.dag_cost(egraph, roots).into_inner();
        }
        if depth_weight != 0.0 {
            value += depth_weight * extraction.dag_depth(egraph, roots).into_inner();
        }
        value
    }

    /// The solver-free extractor paired with this objective, used for the
    /// `WarmStart::Greedy` seed and the fallback. A depth budget pairs with
    /// the depth extractor, the only one guaranteed to meet any attainable
    /// budget.
    pub(crate) fn greedy_extractor(&self) -> (&'static str, Box<dyn Extractor>) {
        match self {
            IlpObjective::Size | IlpObjective::WeightedSizeDepth { .. } => {
                ("greedy-size", GreedySizeExtractor.boxed())
            }
            IlpObjective::Depth | IlpObjective::SizeConstrainedDepth { .. } => {
                ("greedy-depth", GreedyDepthExtractor.boxed())
            }
        }
    }
}
