/* An ILP extractor that returns the optimal DAG-extraction.

This extractor is simple so that it's easy to see that it's correct.

If the timeout is reached, it will return the result of the faster-greedy-dag extractor.
*/

use super::*;
use coin_cbc::{Col, Model, Sense};
use indexmap::IndexSet;

struct ClassVars {
    active: Col,
    nodes: Vec<Col>,
}

pub struct CbcExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32>;

impl<const TIMEOUT_IN_SECONDS: u32> Extractor for CbcExtractorWithTimeout<TIMEOUT_IN_SECONDS> {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract(egraph, roots, TIMEOUT_IN_SECONDS);
    }
}

pub struct CbcExtractor;

impl Extractor for CbcExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract(egraph, roots, std::u32::MAX);
    }
}

fn extract(egraph: &EGraph, roots: &[ClassId], timeout_seconds: u32) -> ExtractionResult {
    let mut model = Model::default();

    model.set_parameter("seconds", &timeout_seconds.to_string());

    let vars: IndexMap<ClassId, ClassVars> = egraph
        .classes()
        .values()
        .map(|class| {
            let cvars = ClassVars {
                active: model.add_binary(),
                nodes: class.nodes.iter().map(|_| model.add_binary()).collect(),
            };
            (class.id.clone(), cvars)
        })
        .collect();

    for (class_id, class) in &vars {
        // class active == some node active
        // sum(for node_active in class) == class_active
        let row = model.add_row();
        model.set_row_equal(row, 0.0);
        model.set_weight(row, class.active, -1.0);
        for &node_active in &class.nodes {
            model.set_weight(row, node_active, 1.0);
        }

        let childrens_classes_var = |nid: NodeId| {
            egraph[&nid]
                .children
                .iter()
                .map(|n| egraph[n].eclass.clone())
                .map(|n| vars[&n].active)
                .collect::<IndexSet<_>>()
        };

        for (node_id, &node_active) in egraph[class_id].nodes.iter().zip(&class.nodes) {
            for child_active in childrens_classes_var(node_id.clone()) {
                // node active implies child active, encoded as:
                //   node_active <= child_active
                //   node_active - child_active <= 0
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, node_active, 1.0);
                model.set_weight(row, child_active, -1.0);
            }
        }
    }

    model.set_obj_sense(Sense::Minimize);
    for class in egraph.classes().values() {
        for (node_id, &node_active) in class.nodes.iter().zip(&vars[&class.id].nodes) {
            let node = &egraph[node_id];
            let node_cost = node.cost.into_inner();
            assert!(node_cost >= 0.0);

            if node_cost != 0.0 {
                model.set_obj_coeff(node_active, node_cost);
            }
        }
    }

    for root in roots {
        model.set_col_lower(vars[root].active, 1.0);
    }

    block_cycles(&mut model, &vars, &egraph);

    let solution = model.solve();
    log::info!(
        "CBC status {:?}, {:?}, obj = {}",
        solution.raw().status(),
        solution.raw().secondary_status(),
        solution.raw().obj_value(),
    );

    if solution.raw().status() != coin_cbc::raw::Status::Finished {
        assert!(timeout_seconds != std::u32::MAX);

        let initial_result =
            super::faster_greedy_dag::FasterGreedyDagExtractor.extract(egraph, roots);
        log::info!("Unfinished CBC solution");
        return initial_result;
    }

    let mut result = ExtractionResult::default();

    for (id, var) in &vars {
        let active = solution.col(var.active) > 0.0;
        if active {
            let node_idx = var
                .nodes
                .iter()
                .position(|&n| solution.col(n) > 0.0)
                .unwrap();
            let node_id = egraph[id].nodes[node_idx].clone();
            result.choose(id.clone(), node_id);
        }
    }

    return result;
}

/*

 To block cycles, we enforce that a topological ordering exists on the extraction.
 Each class is mapped to a variable (called its level).  Then for each node,
 we add a constraint that if a node is active, then the level of the class the node
 belongs to must be less than than the level of each of the node's children.

 To create a cycle, the levels would need to decrease, so they're blocked. For example,
 given a two class cycle: if class A, has level 'l', and class B has level 'm', then
 'l' must be less than 'm', but because there is also an active node in class B that
 has class A as a child, 'm' must be less than 'l', which is a contradiction.
*/

fn block_cycles(model: &mut Model, vars: &IndexMap<ClassId, ClassVars>, egraph: &EGraph) {
    let mut levels: IndexMap<ClassId, Col> = Default::default();
    for c in vars.keys() {
        let var = model.add_col();
        levels.insert(c.clone(), var);
        //model.set_col_lower(var, 0.0);
        // It solves the benchmarks about 5% faster without this
        //model.set_col_upper(var, vars.len() as f64);
    }

    // If n.variable is true, opposite_col will be false and vice versa.
    let mut opposite: IndexMap<Col, Col> = Default::default();
    for c in vars.values() {
        for n in &c.nodes {
            let opposite_col = model.add_binary();
            opposite.insert(*n, opposite_col);
            let row = model.add_row();
            model.set_row_equal(row, 1.0);
            model.set_weight(row, opposite_col, 1.0);
            model.set_weight(row, *n, 1.0);
        }
    }

    for (class_id, c) in vars {
        for i in 0..c.nodes.len() {
            let n_id = &egraph[class_id].nodes[i];
            let n = &egraph[n_id];
            let var = c.nodes[i];

            let children_classes = n
                .children
                .iter()
                .map(|n| egraph[n].eclass.clone())
                .collect::<IndexSet<_>>();

            if children_classes.contains(class_id) {
                // Self loop - disable this node.
                // This is clumsier than calling set_col_lower(var,0.0),
                // but means it'll be infeasible (rather than producing an
                // incorrect solution) if var corresponds to a root node.
                let row = model.add_row();
                model.set_weight(row, var, 1.0);
                model.set_row_equal(row, 0.0);
                continue;
            }

            for cc in children_classes {
                assert!(*levels.get(class_id).unwrap() != *levels.get(&cc).unwrap());

                let row = model.add_row();
                model.set_row_lower(row, 1.0);
                model.set_weight(row, *levels.get(class_id).unwrap(), -1.0);
                model.set_weight(row, *levels.get(&cc).unwrap(), 1.0);

                // If n.variable is 0, then disable the contraint.
                model.set_weight(row, *opposite.get(&var).unwrap(), (vars.len() + 1) as f64);
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Dual (area + critical-path delay) ILP extractor.                          */
/*                                                                           */
/* Minimises   alpha * (critical-path delay)  +  beta * (total area)         */
/*                                                                           */
/* Area is the additive sum of per-node `cost`, exactly like `extract`.      */
/* Delay is the longest weighted path from the roots down to the leaves,     */
/* using per-node `delay`. The max-over-paths is linearised with one         */
/* arrival-time variable T_c per e-class and a big-M selection guard:        */
/*                                                                           */
/*   T_c >= delay_n + T_cc - M*(1 - x_n)     for each child class cc of n    */
/*   T_c >= delay_n        - M*(1 - x_n)     for childless n                 */
/*                                                                           */
/* with 0 <= T_c <= S (S = sum of all delays). Capping T_c at S is what      */
/* makes the big-M sound: it bounds delay_n + T_cc <= 2S, so M = 2S is safe. */
/* The objective then minimises alpha*t + beta*sum(area*x) where t >= T_root.*/
/* ------------------------------------------------------------------------- */

/// Per-node area contribution (the additive cost term).
fn node_area(node: &Node) -> f64 {
    node.cost.into_inner()
}

/// Per-node delay contribution (the critical-path term).
fn node_delay(node: &Node) -> f64 {
    node.delay.into_inner()
}

pub struct DualCbcExtractor {
    /// Solver time limit in seconds (u32::MAX for unbounded).
    pub timeout_seconds: u32,
    /// Weight on the critical-path delay.
    pub alpha: f64,
    /// Weight on the total area.
    pub beta: f64,
}

impl Extractor for DualCbcExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        extract_dual(egraph, roots, self.timeout_seconds, self.alpha, self.beta)
    }
}

/// Builds the selection variables/constraints (identical to the sum-cost
/// `extract`) plus the arrival-time variables `T_c` per e-class with their
/// big-M constraints:
///
///   T_c >= delay_n + T_cc - M*(1 - x_n)   for each child class cc of n
///   T_c >= delay_n        - M*(1 - x_n)   for childless n
///
/// bounded by `0 <= T_c <= S` (S = sum of all delays), which is what makes
/// `M = 2S` a sound big-M. Shared by every extractor in this module that
/// needs critical-path delay, so the arrival-time formulation lives in one
/// place. Returns the class selection vars, the arrival-time column per
/// class, and `S`.
fn build_selection_and_arrival(
    model: &mut Model,
    egraph: &EGraph,
) -> (IndexMap<ClassId, ClassVars>, IndexMap<ClassId, Col>, f64) {
    let vars: IndexMap<ClassId, ClassVars> = egraph
        .classes()
        .values()
        .map(|class| {
            let cvars = ClassVars {
                active: model.add_binary(),
                nodes: class.nodes.iter().map(|_| model.add_binary()).collect(),
            };
            (class.id.clone(), cvars)
        })
        .collect();

    // Selection constraints: class active == some node active, and a node
    // being active implies each of its child classes is active. Identical to
    // the sum-cost `extract`.
    for (class_id, class) in &vars {
        let row = model.add_row();
        model.set_row_equal(row, 0.0);
        model.set_weight(row, class.active, -1.0);
        for &node_active in &class.nodes {
            model.set_weight(row, node_active, 1.0);
        }

        let childrens_classes_var = |nid: NodeId| {
            egraph[&nid]
                .children
                .iter()
                .map(|n| egraph[n].eclass.clone())
                .map(|n| vars[&n].active)
                .collect::<IndexSet<_>>()
        };

        for (node_id, &node_active) in egraph[class_id].nodes.iter().zip(&class.nodes) {
            for child_active in childrens_classes_var(node_id.clone()) {
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, node_active, 1.0);
                model.set_weight(row, child_active, -1.0);
            }
        }
    }

    // Arrival-time variables, bounded by S = sum of all delays.
    let s: f64 = egraph
        .classes()
        .values()
        .flat_map(|c| c.nodes.iter())
        .map(|nid| node_delay(&egraph[nid]))
        .sum();
    let big_m = if s > 0.0 { 2.0 * s } else { 1.0 };

    let arr: IndexMap<ClassId, Col> = vars
        .keys()
        .map(|c| {
            let v = model.add_col();
            model.set_col_lower(v, 0.0);
            model.set_col_upper(v, s.max(0.0));
            (c.clone(), v)
        })
        .collect();

    for (class_id, class) in &vars {
        let t_c = arr[class_id];
        for (node_id, &x_n) in egraph[class_id].nodes.iter().zip(&class.nodes) {
            let d_n = node_delay(&egraph[node_id]);
            let child_classes = egraph[node_id]
                .children
                .iter()
                .map(|n| egraph[n].eclass.clone())
                .collect::<IndexSet<_>>();

            if child_classes.is_empty() {
                // T_c - M*x_n >= d_n - M
                let row = model.add_row();
                model.set_row_lower(row, d_n - big_m);
                model.set_weight(row, t_c, 1.0);
                model.set_weight(row, x_n, -big_m);
            } else {
                for cc in child_classes {
                    // T_c - T_cc - M*x_n >= d_n - M
                    let row = model.add_row();
                    model.set_row_lower(row, d_n - big_m);
                    model.set_weight(row, t_c, 1.0);
                    model.set_weight(row, arr[&cc], -1.0);
                    model.set_weight(row, x_n, -big_m);
                }
            }
        }
    }

    (vars, arr, s)
}

fn extract_dual(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    alpha: f64,
    beta: f64,
) -> ExtractionResult {
    let mut model = Model::default();
    model.set_parameter("seconds", &timeout_seconds.to_string());

    let (vars, arr, s) = build_selection_and_arrival(&mut model, egraph);

    // Objective: alpha * t + beta * sum(area * x), where t >= max root arrival.
    model.set_obj_sense(Sense::Minimize);

    let t = model.add_col();
    model.set_col_lower(t, 0.0);
    model.set_col_upper(t, s.max(0.0));
    if alpha != 0.0 {
        model.set_obj_coeff(t, alpha);
    }
    for root in roots {
        // t - T_root >= 0
        let row = model.add_row();
        model.set_row_lower(row, 0.0);
        model.set_weight(row, t, 1.0);
        model.set_weight(row, arr[root], -1.0);
    }

    if beta != 0.0 {
        for class in egraph.classes().values() {
            for (node_id, &node_active) in class.nodes.iter().zip(&vars[&class.id].nodes) {
                let area = beta * node_area(&egraph[node_id]);
                if area != 0.0 {
                    model.set_obj_coeff(node_active, area);
                }
            }
        }
    }

    for root in roots {
        model.set_col_lower(vars[root].active, 1.0);
    }

    block_cycles(&mut model, &vars, &egraph);

    let solution = model.solve();
    log::info!(
        "Dual CBC status {:?}, {:?}, obj = {}",
        solution.raw().status(),
        solution.raw().secondary_status(),
        solution.raw().obj_value(),
    );

    if solution.raw().status() != coin_cbc::raw::Status::Finished {
        assert!(timeout_seconds != std::u32::MAX);
        // NOTE: the fallback optimises area only (not the dual objective).
        log::info!("Unfinished dual CBC solution; falling back to area-greedy DAG");
        return super::faster_greedy_dag::FasterGreedyDagExtractor.extract(egraph, roots);
    }

    let mut result = ExtractionResult::default();
    for (id, var) in &vars {
        let active = solution.col(var.active) > 0.0;
        if active {
            let node_idx = var
                .nodes
                .iter()
                .position(|&n| solution.col(n) > 0.0)
                .unwrap();
            let node_id = egraph[id].nodes[node_idx].clone();
            result.choose(id.clone(), node_id);
        }
    }

    result
}

/* ------------------------------------------------------------------------- */
/* Delay-budget (area-under-timing-constraint) ILP extractor.                */
/*                                                                           */
/* Minimises   sum(area_n * x_n)                                            */
/* subject to  critical-path delay <= max_delay                             */
/*                                                                           */
/* Reuses the arrival-time machinery from `build_selection_and_arrival`.    */
/* Because delays are non-negative, every active class's arrival time is    */
/* transitively bounded by whichever root it feeds, so capping every T_c's  */
/* upper bound at `max_delay` (not just the roots') is sound: it can never  */
/* cut off a feasible solution, and it makes an unattainable budget surface */
/* as a provable ILP infeasibility rather than a silently wrong answer.     */
/* ------------------------------------------------------------------------- */

pub struct DelayBudgetCbcExtractor {
    /// Solver time limit in seconds (u32::MAX for unbounded).
    pub timeout_seconds: u32,
    /// Hard upper bound on the critical-path delay of the extraction.
    pub max_delay: f64,
}

impl Extractor for DelayBudgetCbcExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        extract_delay_budget(egraph, roots, self.timeout_seconds, self.max_delay)
    }
}

fn extract_delay_budget(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    max_delay: f64,
) -> ExtractionResult {
    let mut model = Model::default();
    model.set_parameter("seconds", &timeout_seconds.to_string());

    let (vars, arr, s) = build_selection_and_arrival(&mut model, egraph);

    // Tighten every arrival-time variable's upper bound to the budget. This
    // enforces T_root <= max_delay for every root with no extra rows, and is
    // sound for every class (see comment above).
    let budget = max_delay.min(s.max(0.0)).max(0.0);
    for &t_c in arr.values() {
        model.set_col_upper(t_c, budget);
    }

    // Objective: minimise area alone.
    model.set_obj_sense(Sense::Minimize);
    for class in egraph.classes().values() {
        for (node_id, &node_active) in class.nodes.iter().zip(&vars[&class.id].nodes) {
            let area = node_area(&egraph[node_id]);
            if area != 0.0 {
                model.set_obj_coeff(node_active, area);
            }
        }
    }

    for root in roots {
        model.set_col_lower(vars[root].active, 1.0);
    }

    block_cycles(&mut model, &vars, &egraph);

    let solution = model.solve();
    log::info!(
        "Delay-budget CBC status {:?}, {:?}, obj = {}",
        solution.raw().status(),
        solution.raw().secondary_status(),
        solution.raw().obj_value(),
    );

    if solution.raw().is_proven_infeasible() {
        log::info!("Infeasible, returning empty solution");
        return ExtractionResult::default();
    }

    if solution.raw().status() != coin_cbc::raw::Status::Finished {
        assert!(timeout_seconds != std::u32::MAX);
        // NOTE: the fallback optimises area only and ignores max_delay
        // entirely, so it is not guaranteed to respect the delay budget.
        log::info!("Unfinished delay-budget CBC solution; falling back to area-greedy DAG");
        return super::faster_greedy_dag::FasterGreedyDagExtractor.extract(egraph, roots);
    }

    let mut result = ExtractionResult::default();
    for (id, var) in &vars {
        let active = solution.col(var.active) > 0.0;
        if active {
            let node_idx = var
                .nodes
                .iter()
                .position(|&n| solution.col(n) > 0.0)
                .unwrap();
            let node_id = egraph[id].nodes[node_idx].clone();
            result.choose(id.clone(), node_id);
        }
    }

    result
}
