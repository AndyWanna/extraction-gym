/*!
Exact bottom-up extraction for costs of the form `cost(node) = f(node, child costs)`
where `f` is monotone and *superior* (`f >= every child cost`), e.g.
`delay + max(children)`.

This is Knuth's generalisation of Dijkstra's algorithm: a class is finalised
when it is the cheapest unfinalised candidate, and a node only becomes a
candidate once all of its child classes are finalised. Superiority means no
later candidate can undercut a finalised class, so every class gets its
optimal cost. Children are always finalised before their parents, so the
choices are acyclic by construction.

Note this is NOT exact for DAG size, where shared children are counted once.
*/

use crate::*;
use rustc_hash::FxHashMap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Choose one node per class, over the nodes for which `include` holds,
/// minimising `node_cost(node, &child_class_costs)`. Classes with no finite
/// acyclic choice get no entry.
pub(crate) fn extract_superior(
    egraph: &EGraph,
    include: impl Fn(&Node) -> bool,
    node_cost: impl Fn(&Node, &[Cost]) -> Cost,
) -> ExtractionResult {
    let n2c = |nid: &NodeId| egraph.nid_to_cid(nid);

    let nodes: Vec<&NodeId> = egraph
        .classes()
        .values()
        .flat_map(|class| class.nodes.iter())
        .filter(|nid| include(&egraph[*nid]))
        .collect();

    // For each node: how many distinct child classes are not finalised yet.
    // For each class: the nodes that have it as a child.
    let mut pending_children: Vec<usize> = Vec::with_capacity(nodes.len());
    let mut parents: FxHashMap<&ClassId, Vec<usize>> = FxHashMap::default();
    // Min-heap on (cost, insertion order, node index); the order keeps ties
    // deterministic.
    let mut heap: BinaryHeap<Reverse<(Cost, usize, usize)>> = BinaryHeap::new();
    let mut pushes = 0usize;

    for (idx, nid) in nodes.iter().enumerate() {
        let mut child_classes: Vec<&ClassId> = egraph[*nid].children.iter().map(n2c).collect();
        child_classes.sort();
        child_classes.dedup();
        for cid in &child_classes {
            parents.entry(cid).or_default().push(idx);
        }
        pending_children.push(child_classes.len());
        if child_classes.is_empty() {
            heap.push(Reverse((node_cost(&egraph[*nid], &[]), pushes, idx)));
            pushes += 1;
        }
    }

    let mut finalised: FxHashMap<ClassId, Cost> = FxHashMap::default();
    let mut result = ExtractionResult::default();

    while let Some(Reverse((cost, _, idx))) = heap.pop() {
        let nid = nodes[idx];
        let cid = n2c(nid);
        if finalised.contains_key(cid) {
            continue;
        }
        finalised.insert(cid.clone(), cost);
        result.choose(cid.clone(), nid.clone());

        for &parent in parents.get(cid).map(Vec::as_slice).unwrap_or_default() {
            pending_children[parent] -= 1;
            if pending_children[parent] == 0 {
                let node = &egraph[nodes[parent]];
                let child_costs: Vec<Cost> =
                    node.children.iter().map(|c| finalised[n2c(c)]).collect();
                heap.push(Reverse((node_cost(node, &child_costs), pushes, parent)));
                pushes += 1;
            }
        }
    }

    result
}
