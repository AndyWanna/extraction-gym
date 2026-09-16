use super::model::ClassVars;
use crate::milp::MilpSolution;
use crate::*;

/// Read the selected node of every active class out of `solution`.
///
/// Does not check the result: a class marked active with no active member
/// (numerically impossible for a feasible solution) is simply left out.
pub(crate) fn decode_selection<S: MilpSolution>(
    solution: &S,
    egraph: &EGraph,
    classes: &IndexMap<ClassId, ClassVars<S::Col>>,
) -> ExtractionResult {
    let mut result = ExtractionResult::default();
    for (id, var) in classes {
        if solution.col(var.active) <= 0.0 {
            continue;
        }
        if let Some(node_idx) = var.nodes.iter().position(|&n| solution.col(n) > 0.0) {
            result.choose(id.clone(), egraph[id].nodes[node_idx].clone());
        }
    }
    result
}
