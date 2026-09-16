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
}
