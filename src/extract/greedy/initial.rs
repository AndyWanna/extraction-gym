use super::fixpoint::extract_superior;
use crate::*;

/// Rebuilds the initial expression from the nodes flagged `Node::initial`.
///
/// Usually each class holds at most one initial node and this is just a walk.
/// But rewriting can prove two subexpressions of the initial expression
/// equivalent and union their classes, leaving a class with several initial
/// nodes, one of which may be an ancestor of the other. Choosing the one of
/// **minimum height** (levels above the leaves, over initial nodes only)
/// always yields an acyclic extraction: every initial node's children hold an
/// initial node of strictly smaller height, so by induction every class that
/// holds an initial node gets a choice.
///
/// Returns an empty extraction when the e-graph has no initial nodes; see
/// [`has_initial_nodes`].
pub struct InitialExtractor;

impl Extractor for InitialExtractor {
    fn extract(&self, egraph: &EGraph, _roots: &[ClassId]) -> ExtractionResult {
        let one = Cost::new(1.0).unwrap();
        extract_superior(
            egraph,
            |node| node.initial,
            |_, children| one + children.iter().copied().max().unwrap_or_default(),
        )
    }
}

/// Whether any node of `egraph` is flagged `Node::initial`.
pub fn has_initial_nodes(egraph: &EGraph) -> bool {
    egraph.nodes.values().any(|node| node.initial)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(eclass: &str, children: &[&str], initial: bool) -> Node {
        Node {
            op: "op".into(),
            children: children.iter().map(|c| NodeId::from(*c)).collect(),
            eclass: eclass.into(),
            cost: Cost::new(1.0).unwrap(),
            delay: Cost::new(1.0).unwrap(),
            initial,
        }
    }

    #[test]
    fn follows_initial_nodes_only() {
        // f(g(x)), plus a non-initial cheaper `y` in g's class.
        let mut egraph = EGraph::default();
        egraph.add_node("x", node("X", &[], true));
        egraph.add_node("g", node("G", &["x"], true));
        egraph.add_node("y", node("G", &[], false));
        egraph.add_node("f", node("F", &["g"], true));
        egraph.root_eclasses.push("F".into());

        let roots = egraph.root_eclasses.clone();
        let result = InitialExtractor.extract(&egraph, &roots);
        CompleteExtraction::try_new(result.clone(), &egraph, &roots).unwrap();
        assert_eq!(result.choices[&ClassId::from("G")], NodeId::from("g"));
    }

    #[test]
    fn unioned_ancestor_and_descendant_stay_acyclic() {
        // Initial expression f(g(x)); rewriting proved f(g(x)) == x, so f and
        // x share class FX, and f's grandchild is its own class.
        let mut egraph = EGraph::default();
        egraph.add_node("x", node("FX", &[], true));
        egraph.add_node("g", node("G", &["x"], true));
        egraph.add_node("f", node("FX", &["g"], true));
        egraph.root_eclasses.push("FX".into());

        let roots = egraph.root_eclasses.clone();
        let result = InitialExtractor.extract(&egraph, &roots);
        CompleteExtraction::try_new(result.clone(), &egraph, &roots).unwrap();
        assert_eq!(result.choices[&ClassId::from("FX")], NodeId::from("x"));
    }

    #[test]
    fn no_initial_nodes_is_incomplete() {
        let mut egraph = EGraph::default();
        egraph.add_node("x", node("X", &[], false));
        egraph.root_eclasses.push("X".into());
        let roots = egraph.root_eclasses.clone();
        assert!(!has_initial_nodes(&egraph));
        assert!(!InitialExtractor
            .extract(&egraph, &roots)
            .is_complete(&egraph, &roots));
    }
}
