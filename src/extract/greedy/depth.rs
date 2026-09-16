use super::fixpoint::extract_superior;
use crate::*;

/// Minimum-depth extractor, where the depth of a node is its `Node::delay`
/// plus the largest depth among its children.
///
/// Despite the name this is **exact**: depth takes a `max` over children
/// rather than a sum over shared nodes, so a bottom-up fixpoint finds the
/// optimum in `O(E log V)` (see [`super::fixpoint`]). Requires non-negative
/// delays.
pub struct GreedyDepthExtractor;

impl Extractor for GreedyDepthExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        extract_superior(
            egraph,
            |_| true,
            |node, children| node.delay + children.iter().copied().max().unwrap_or_default(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashMap;

    /// Minimum depth per class by plain Bellman-Ford relaxation: slow, but
    /// obviously correct.
    fn reference_min_depths(egraph: &EGraph) -> FxHashMap<ClassId, Cost> {
        let mut depth: FxHashMap<ClassId, Cost> = FxHashMap::default();
        loop {
            let mut changed = false;
            for class in egraph.classes().values() {
                for nid in &class.nodes {
                    let node = &egraph[nid];
                    let children: Option<Vec<Cost>> = node
                        .children
                        .iter()
                        .map(|c| depth.get(egraph.nid_to_cid(c)).copied())
                        .collect();
                    let Some(children) = children else { continue };
                    let d = node.delay + children.into_iter().max().unwrap_or_default();
                    if depth.get(&class.id).map_or(true, |&old| d < old) {
                        depth.insert(class.id.clone(), d);
                        changed = true;
                    }
                }
            }
            if !changed {
                return depth;
            }
        }
    }

    #[test]
    fn matches_reference_on_random_egraphs() {
        for _ in 0..200 {
            let egraph = crate::test::generate_random_egraph();
            let roots = egraph.root_eclasses.clone();
            let result = GreedyDepthExtractor.extract(&egraph, &roots);
            CompleteExtraction::try_new(result.clone(), &egraph, &roots).unwrap();

            let reference = reference_min_depths(&egraph);
            let expected = roots.iter().map(|r| reference[r]).max().unwrap();
            let got = result.dag_depth(&egraph, &roots);
            assert!(
                (got - expected).abs() < EPSILON_ALLOWANCE,
                "{got} != {expected}"
            );
        }
    }
}
