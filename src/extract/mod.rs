use indexmap::IndexMap;
use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::HashMap;

pub use crate::*;

pub mod bottom_up;
pub mod faster_bottom_up;
pub mod faster_greedy_dag;
#[cfg(feature = "ilp-any")]
pub mod faster_ilp_cbc;
pub mod global_greedy_dag;
pub mod greedy;
#[cfg(feature = "ilp-any")]
pub mod ilp;
pub mod greedy_dag;
#[cfg(feature = "ilp-any")]
pub mod ilp_cbc;
pub mod prio_queue;
/// Warm-start construction/validation + the per-run solver report shared by
/// both ILP extractors.
#[cfg(feature = "ilp-any")]
pub mod warm;

#[cfg(feature = "ilp-any")]
#[allow(deprecated)]
pub use warm::{SolveReport, WarmStartMode};
#[cfg(feature = "ilp-any")]
pub use ilp::{DepthBudget, IlpError, IlpObjective, IlpOptions, IlpOutcome, SolveOutcome, WarmStart};

// Allowance for floating point values to be considered equal
pub const EPSILON_ALLOWANCE: f64 = 0.00001;

pub trait Extractor: Sync {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult;

    fn boxed(self) -> Box<dyn Extractor>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }
}

pub trait MapGet<K, V> {
    fn get(&self, key: &K) -> Option<&V>;
}

impl<K, V> MapGet<K, V> for HashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        HashMap::get(self, key)
    }
}

impl<K, V> MapGet<K, V> for FxHashMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        FxHashMap::get(self, key)
    }
}

impl<K, V> MapGet<K, V> for IndexMap<K, V>
where
    K: Eq + std::hash::Hash,
{
    fn get(&self, key: &K) -> Option<&V> {
        IndexMap::get(self, key)
    }
}

#[derive(Default, Clone)]
pub struct ExtractionResult {
    pub choices: IndexMap<ClassId, NodeId>,
}

/// An [`ExtractionResult`] proven valid for a specific set of roots (see
/// [`ExtractionResult::validate`]). The only constructor validates, so holding
/// one means the selection can be costed and converted without panicking.
#[derive(Clone)]
pub struct CompleteExtraction(ExtractionResult);

impl CompleteExtraction {
    pub fn try_new(
        result: ExtractionResult,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> Result<Self, String> {
        if roots.is_empty() {
            return Err("no roots to extract".to_string());
        }
        result.validate(egraph, roots)?;
        Ok(CompleteExtraction(result))
    }

    pub fn into_inner(self) -> ExtractionResult {
        self.0
    }
}

impl std::ops::Deref for CompleteExtraction {
    type Target = ExtractionResult;
    fn deref(&self) -> &ExtractionResult {
        &self.0
    }
}

#[derive(Clone, Copy)]
enum Status {
    Doing,
    Done,
}

impl ExtractionResult {
    /// Panics unless this is a valid extraction of `egraph.root_eclasses`
    /// (see [`Self::validate`]).
    pub fn check(&self, egraph: &EGraph) {
        // should be a root
        assert!(!egraph.root_eclasses.is_empty());
        if let Err(why) = self.validate(egraph, &egraph.root_eclasses) {
            panic!("invalid extraction: {why}");
        }
    }

    /// Whether this is a valid extraction of `roots`: see [`Self::validate`].
    pub fn is_complete(&self, egraph: &EGraph, roots: &[ClassId]) -> bool {
        self.validate(egraph, roots).is_ok()
    }

    /// `Ok` iff every class reachable from `roots` through the chosen nodes has
    /// a choice, every choice lives in the class it is chosen for, and the
    /// chosen nodes form no cycle. Choices for unreachable classes are ignored.
    pub fn validate(&self, egraph: &EGraph, roots: &[ClassId]) -> Result<(), String> {
        // Nodes should match the class they are selected into.
        for (cid, nid) in &self.choices {
            if egraph[nid].eclass != *cid {
                return Err(format!(
                    "node {nid} is chosen for class {cid} but belongs to {}",
                    egraph[nid].eclass
                ));
            }
        }

        // All the nodes the roots depend upon should be selected. This walk
        // must precede `find_cycles`, which indexes `choices` directly.
        let mut todo: Vec<ClassId> = roots.to_vec();
        let mut visited: FxHashSet<ClassId> = Default::default();
        while let Some(cid) = todo.pop() {
            if !visited.insert(cid.clone()) {
                continue;
            }
            let Some(nid) = self.choices.get(&cid) else {
                return Err(format!("no choice for reachable class {cid}"));
            };
            for child in &egraph[nid].children {
                todo.push(egraph.nid_to_cid(child).clone());
            }
        }

        // No cycles
        let cycles = self.find_cycles(egraph, roots);
        if !cycles.is_empty() {
            return Err(format!(
                "chosen nodes form a cycle through class {}",
                cycles[0]
            ));
        }
        Ok(())
    }

    pub fn choose(&mut self, class_id: ClassId, node_id: NodeId) {
        self.choices.insert(class_id, node_id);
    }

    pub fn find_cycles(&self, egraph: &EGraph, roots: &[ClassId]) -> Vec<ClassId> {
        // let mut status = vec![Status::Todo; egraph.classes().len()];
        let mut status = IndexMap::<ClassId, Status>::default();
        let mut cycles = vec![];
        for root in roots {
            // let root_index = egraph.classes().get_index_of(root).unwrap();
            self.cycle_dfs(egraph, root, &mut status, &mut cycles)
        }
        cycles
    }

    fn cycle_dfs(
        &self,
        egraph: &EGraph,
        class_id: &ClassId,
        status: &mut IndexMap<ClassId, Status>,
        cycles: &mut Vec<ClassId>,
    ) {
        match status.get(class_id).cloned() {
            Some(Status::Done) => (),
            Some(Status::Doing) => cycles.push(class_id.clone()),
            None => {
                status.insert(class_id.clone(), Status::Doing);
                let node_id = &self.choices[class_id];
                let node = &egraph[node_id];
                for child in &node.children {
                    let child_cid = egraph.nid_to_cid(child);
                    self.cycle_dfs(egraph, child_cid, status, cycles)
                }
                status.insert(class_id.clone(), Status::Done);
            }
        }
    }

    pub fn tree_cost(&self, egraph: &EGraph, roots: &[ClassId]) -> Cost {
        let node_roots = roots
            .iter()
            .map(|cid| self.choices[cid].clone())
            .collect::<Vec<NodeId>>();
        self.tree_cost_rec(egraph, &node_roots, &mut HashMap::new())
    }

    fn tree_cost_rec(
        &self,
        egraph: &EGraph,
        roots: &[NodeId],
        memo: &mut HashMap<NodeId, Cost>,
    ) -> Cost {
        let mut cost = Cost::default();
        for root in roots {
            if let Some(c) = memo.get(root) {
                cost += *c;
                continue;
            }
            let class = egraph.nid_to_cid(root);
            let node = &egraph[&self.choices[class]];
            let inner = node.cost + self.tree_cost_rec(egraph, &node.children, memo);
            memo.insert(root.clone(), inner);
            cost += inner;
        }
        cost
    }

    // this will loop if there are cycles
    pub fn dag_cost(&self, egraph: &EGraph, roots: &[ClassId]) -> Cost {
        let mut costs: IndexMap<ClassId, Cost> = IndexMap::new();
        let mut todo: Vec<ClassId> = roots.to_vec();
        while let Some(cid) = todo.pop() {
            let node_id = &self.choices[&cid];
            let node = &egraph[node_id];
            if costs.insert(cid.clone(), node.cost).is_some() {
                continue;
            }
            for child in &node.children {
                todo.push(egraph.nid_to_cid(child).clone());
            }
        }
        costs.values().sum()
    }

    /// Critical-path delay of the extracted DAG: the maximum over root-to-leaf
    /// paths of the sum of per-node `delay`. Companion to `dag_cost` (which
    /// sums area). This is the quantity the dual ILP extractor minimises as its
    /// delay term. Assumes the extraction is acyclic (see `check`); a cycle
    /// guard keeps it terminating (returning a placeholder) rather than looping.
    pub fn dag_depth(&self, egraph: &EGraph, roots: &[ClassId]) -> Cost {
        let mut memo: HashMap<ClassId, Cost> = HashMap::new();
        roots
            .iter()
            .map(|r| self.class_depth(egraph, r, &mut memo))
            .max()
            .unwrap_or_default()
    }

    fn class_depth(
        &self,
        egraph: &EGraph,
        class_id: &ClassId,
        memo: &mut HashMap<ClassId, Cost>,
    ) -> Cost {
        if let Some(d) = memo.get(class_id) {
            return *d;
        }
        // Cycle guard: temporarily record 0 so a back-edge terminates.
        memo.insert(class_id.clone(), Cost::default());
        let node = &egraph[&self.choices[class_id]];
        let mut child_max = Cost::default();
        for child in &node.children {
            let cid = egraph.nid_to_cid(child);
            let d = self.class_depth(egraph, cid, memo);
            if d > child_max {
                child_max = d;
            }
        }
        let total = node.delay + child_max;
        memo.insert(class_id.clone(), total);
        total
    }

    pub fn node_sum_cost<M>(&self, egraph: &EGraph, node: &Node, costs: &M) -> Cost
    where
        M: MapGet<ClassId, Cost>,
    {
        node.cost
            + node
                .children
                .iter()
                .map(|n| {
                    let cid = egraph.nid_to_cid(n);
                    costs.get(cid).unwrap_or(&INFINITY)
                })
                .sum::<Cost>()
    }
}

#[cfg(test)]
mod complete_tests {
    use super::*;

    /// `a(b)` with `b` a leaf, in two classes.
    fn two_class_egraph() -> EGraph {
        let mut egraph = EGraph::default();
        let leaf = |eclass: &str, children: Vec<NodeId>| Node {
            op: "op".into(),
            children,
            eclass: eclass.into(),
            cost: Cost::new(1.0).unwrap(),
            delay: Cost::new(1.0).unwrap(),
            initial: false,
        };
        egraph.add_node("b.0", leaf("b", vec![]));
        egraph.add_node("a.0", leaf("a", vec!["b.0".into()]));
        egraph.root_eclasses.push("a".into());
        egraph
    }

    #[test]
    fn try_new_rejects_empty_result() {
        let egraph = two_class_egraph();
        let roots = egraph.root_eclasses.clone();
        assert!(CompleteExtraction::try_new(ExtractionResult::default(), &egraph, &roots).is_err());
    }

    #[test]
    fn try_new_rejects_missing_child() {
        let egraph = two_class_egraph();
        let roots = egraph.root_eclasses.clone();
        let mut partial = ExtractionResult::default();
        partial.choose("a".into(), "a.0".into());
        assert!(!partial.is_complete(&egraph, &roots));
        assert!(CompleteExtraction::try_new(partial, &egraph, &roots).is_err());
    }

    #[test]
    fn try_new_accepts_complete_result() {
        let egraph = two_class_egraph();
        let roots = egraph.root_eclasses.clone();
        let mut full = ExtractionResult::default();
        full.choose("a".into(), "a.0".into());
        full.choose("b".into(), "b.0".into());
        assert!(CompleteExtraction::try_new(full, &egraph, &roots).is_ok());
    }
}
