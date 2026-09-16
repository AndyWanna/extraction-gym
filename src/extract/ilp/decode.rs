use super::model::ClassVars;
use crate::milp::MilpSolution;
use crate::*;

/// Read the selected node of every active class out of `solution`.
///
/// Binaries are read with a 0.5 threshold, since solvers return them within an
/// integrality tolerance. Does not check the result: a class marked active
/// with no active member is simply left out, and validation catches it.
pub(crate) fn decode_selection<S: MilpSolution>(
    solution: &S,
    egraph: &EGraph,
    classes: &IndexMap<ClassId, ClassVars<S::Col>>,
) -> ExtractionResult {
    let mut result = ExtractionResult::default();
    for (id, var) in classes {
        if solution.col(var.active) < 0.5 {
            continue;
        }
        if let Some(node_idx) = var.nodes.iter().position(|&n| solution.col(n) > 0.5) {
            result.choose(id.clone(), egraph[id].nodes[node_idx].clone());
        }
    }
    result
}
