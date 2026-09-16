/*!
The one MILP model behind every ILP extractor.

```text
x_n       binary, node n selected           a_c   binary, class c selected
sum_{n in c} x_n = a_c                      x_n <= a_cc   for each child class cc of n
a_root = 1
```

Cycles are blocked exactly (see [`block_cycles`]): every feasible solution is
acyclic, so any incumbent the solver returns is a valid extraction.

Objectives that involve depth add one arrival time `T_c` per class:

```text
T_c >= depth_n + T_cc - M*(1 - x_n)   for each child class cc of n
T_c >= depth_n        - M*(1 - x_n)   for childless n
```

with `0 <= T_c <= S` (`S` = sum of all node depths), so `M = 2S` is sound.
With a depth budget every `T_c` is instead capped at the budget, which
enforces the budget without extra rows and makes `M = budget + max node depth`
sound and much better conditioned. Delays are non-negative, so capping every
class (not just the roots) never cuts off a feasible solution.
*/

use super::IlpObjective;
use crate::milp::{MilpModel, MilpSense};
use crate::*;
use indexmap::{IndexMap, IndexSet};

/// Selection variables for one e-class.
pub(crate) struct ClassVars<C> {
    pub active: C,
    /// One per member node, in `Class::nodes` order.
    pub nodes: Vec<C>,
}

/// The auxiliary columns [`block_cycles`] creates, kept so a warm start can
/// seed them: an unseeded continuous column may be read as `0.0`, which
/// violates the level rows.
pub(crate) struct CycleVars<C> {
    /// One continuous "topological level" column per class.
    pub levels: IndexMap<ClassId, C>,
    /// `opposite[x_n] == 1 - x_n`, the big-M switch that disables a level
    /// row when its node is not selected.
    pub opposite: IndexMap<C, C>,
}

pub(crate) struct DepthVars<C> {
    /// `T_c` per class.
    pub arrival: IndexMap<ClassId, C>,
    /// Upper bound applied to every `T_c`.
    pub cap: f64,
    /// The depth budget the model enforces (the requested one clamped into
    /// `[0, S]`), if any.
    pub budget: Option<f64>,
    /// `t >= T_root` for every root, when depth is in the objective.
    pub root_depth: Option<C>,
}

pub(crate) struct IlpVars<C> {
    pub classes: IndexMap<ClassId, ClassVars<C>>,
    pub cycles: CycleVars<C>,
    pub depth: Option<DepthVars<C>>,
}

pub(crate) fn build<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
    roots: &[ClassId],
    objective: IlpObjective,
) -> IlpVars<M::Col> {
    let classes = add_selection(model, egraph);
    let depth = objective
        .uses_depth()
        .then(|| add_depth(model, egraph, roots, &classes, objective.budget_value()));

    model.set_obj_sense(MilpSense::Minimize);
    let (size_weight, depth_weight) = objective.weights();
    if let Some(t) = depth.as_ref().and_then(|d| d.root_depth) {
        if depth_weight != 0.0 {
            model.set_obj_coeff(t, depth_weight);
        }
    }
    if size_weight != 0.0 {
        for class in egraph.classes().values() {
            for (node_id, &x_n) in class.nodes.iter().zip(&classes[&class.id].nodes) {
                let size = egraph[node_id].cost.into_inner();
                debug_assert!(size >= 0.0);
                if size != 0.0 {
                    model.set_obj_coeff(x_n, size_weight * size);
                }
            }
        }
    }

    for root in roots {
        model.set_col_lower(classes[root].active, 1.0);
    }

    let cycles = block_cycles(model, &classes, egraph);
    IlpVars {
        classes,
        cycles,
        depth,
    }
}

fn child_classes(egraph: &EGraph, node_id: &NodeId) -> IndexSet<ClassId> {
    egraph[node_id]
        .children
        .iter()
        .map(|n| egraph[n].eclass.clone())
        .collect()
}

fn add_selection<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
) -> IndexMap<ClassId, ClassVars<M::Col>> {
    let vars: IndexMap<ClassId, ClassVars<M::Col>> = egraph
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
        let row = model.add_row();
        model.set_row_equal(row, 0.0);
        model.set_weight(row, class.active, -1.0);
        for &node_active in &class.nodes {
            model.set_weight(row, node_active, 1.0);
        }

        // node active implies child active: node_active - child_active <= 0
        for (node_id, &node_active) in egraph[class_id].nodes.iter().zip(&class.nodes) {
            for child in child_classes(egraph, node_id) {
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, node_active, 1.0);
                model.set_weight(row, vars[&child].active, -1.0);
            }
        }
    }
    vars
}

fn add_depth<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
    roots: &[ClassId],
    classes: &IndexMap<ClassId, ClassVars<M::Col>>,
    requested_budget: Option<f64>,
) -> DepthVars<M::Col> {
    let node_depths = || {
        egraph
            .classes()
            .values()
            .flat_map(|c| c.nodes.iter())
            .map(|nid| egraph[nid].delay.into_inner())
    };
    let s: f64 = node_depths().sum::<f64>().max(0.0);
    // The largest single-node depth (~ the biggest library cell), which unlike
    // S does not grow with the e-graph.
    let d_max: f64 = node_depths().fold(0.0, f64::max);

    let budget = requested_budget.map(|b| b.min(s).max(0.0));
    let big_m = match budget {
        Some(b) => b + d_max,
        None => 2.0 * s,
    };
    let big_m = if big_m > 0.0 { big_m } else { 1.0 };
    let cap = budget.unwrap_or(s);
    if let (Some(b), Some(requested)) = (budget, requested_budget) {
        log::info!(
            "Depth budget: S={s:.2}, D_max={d_max:.2}, budget={b:.2} (requested {requested:.2}), big_M={big_m:.2}"
        );
    }

    let arrival: IndexMap<ClassId, M::Col> = classes
        .keys()
        .map(|c| {
            let v = model.add_col();
            model.set_col_lower(v, 0.0);
            model.set_col_upper(v, cap);
            (c.clone(), v)
        })
        .collect();

    for (class_id, class) in classes {
        let t_c = arrival[class_id];
        for (node_id, &x_n) in egraph[class_id].nodes.iter().zip(&class.nodes) {
            let d_n = egraph[node_id].delay.into_inner();
            let children = child_classes(egraph, node_id);
            if children.is_empty() {
                // T_c - M*x_n >= d_n - M
                let row = model.add_row();
                model.set_row_lower(row, d_n - big_m);
                model.set_weight(row, t_c, 1.0);
                model.set_weight(row, x_n, -big_m);
            }
            for cc in children {
                // T_c - T_cc - M*x_n >= d_n - M
                let row = model.add_row();
                model.set_row_lower(row, d_n - big_m);
                model.set_weight(row, t_c, 1.0);
                model.set_weight(row, arrival[&cc], -1.0);
                model.set_weight(row, x_n, -big_m);
            }
        }
    }

    // A budget is enforced by the column caps alone; otherwise depth is in the
    // objective and needs `t >= T_root`.
    let root_depth = budget.is_none().then(|| {
        let t = model.add_col();
        model.set_col_lower(t, 0.0);
        model.set_col_upper(t, s);
        for root in roots {
            // t - T_root >= 0
            let row = model.add_row();
            model.set_row_lower(row, 0.0);
            model.set_weight(row, t, 1.0);
            model.set_weight(row, arrival[root], -1.0);
        }
        t
    });

    DepthVars {
        arrival,
        cap,
        budget,
        root_depth,
    }
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
fn block_cycles<M: MilpModel>(
    model: &mut M,
    vars: &IndexMap<ClassId, ClassVars<M::Col>>,
    egraph: &EGraph,
) -> CycleVars<M::Col> {
    let mut levels: IndexMap<ClassId, M::Col> = Default::default();
    for c in vars.keys() {
        let var = model.add_col();
        levels.insert(c.clone(), var);
        //model.set_col_lower(var, 0.0);
        // It solves the benchmarks about 5% faster without this
        //model.set_col_upper(var, vars.len() as f64);
    }

    // If n.variable is true, opposite_col will be false and vice versa.
    let mut opposite: IndexMap<M::Col, M::Col> = Default::default();
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
        for (n_id, &var) in egraph[class_id].nodes.iter().zip(&c.nodes) {
            let children = child_classes(egraph, n_id);

            if children.contains(class_id) {
                // Self loop - disable this node.
                // This is clumsier than calling set_col_lower(var,0.0),
                // but means it'll be infeasible (rather than producing an
                // incorrect solution) if var corresponds to a root node.
                let row = model.add_row();
                model.set_weight(row, var, 1.0);
                model.set_row_equal(row, 0.0);
                continue;
            }

            for cc in children {
                let row = model.add_row();
                model.set_row_lower(row, 1.0);
                model.set_weight(row, levels[class_id], -1.0);
                model.set_weight(row, levels[&cc], 1.0);

                // If n.variable is 0, then disable the contraint.
                model.set_weight(row, opposite[&var], (vars.len() + 1) as f64);
            }
        }
    }

    CycleVars { levels, opposite }
}
