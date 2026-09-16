/*!
Handing a warm-start seed to the solver.

A seed here is always a `CompleteExtraction` (the initial or greedy
extraction), so it is complete and acyclic before it arrives. What remains is
making it a feasible point for the *whole* model, not just the selection
binaries:

* [`push_seed`] sets every auxiliary column too: the `1 - x` switches, the
  topological levels ([`topological_levels`]) and the arrival times. A solver
  that reads an unseeded continuous column as `0.0` (CBC does) would otherwise
  see the level rows violated.
* [`arrival_times`] computes depth with the ILP's own recurrence
  (`T_c = d_n + max_cc T_cc`), so the depth-budget check in `solve` is done in
  the solver's arithmetic. A design-derived budget is exactly tight, where one
  ulp decides feasibility.

A seed that fails a check is refused (`SolveReport::refuse_warm_start`) and the
solve runs without a start; it is never a hard failure.
*/
use indexmap::{IndexMap, IndexSet};

use super::model::IlpVars;
use crate::milp::MilpModel;
use crate::{ClassId, EGraph, ExtractionResult};

/// Arrival time (critical-path delay) per class under `selection`, computed
/// with the **same recurrence the ILP's rows encode**:
///
/// ```text
/// T_c = delay(n_c) + max over child classes cc of n_c of T_cc      (0 if none)
/// ```
///
/// Using the ILP's own formulation rather than an independent walk is the whole
/// point: a budget check against a differently-associated sum is a check
/// against a different number, and at a tight budget that difference decides
/// feasibility.
///
/// Returns `Err` on a cycle (the recurrence has no fixed point then).
pub fn arrival_times(
    egraph: &EGraph,
    roots: &[ClassId],
    selection: &ExtractionResult,
) -> Result<IndexMap<ClassId, f64>, String> {
    let mut arrival: IndexMap<ClassId, f64> = IndexMap::default();
    for root in roots {
        arrival_of(
            egraph,
            root,
            selection,
            &mut arrival,
            &mut IndexSet::default(),
        )?;
    }
    Ok(arrival)
}

fn arrival_of(
    egraph: &EGraph,
    cid: &ClassId,
    selection: &ExtractionResult,
    memo: &mut IndexMap<ClassId, f64>,
    on_stack: &mut IndexSet<ClassId>,
) -> Result<f64, String> {
    if let Some(v) = memo.get(cid) {
        return Ok(*v);
    }
    if !on_stack.insert(cid.clone()) {
        return Err(format!("cycle through class {cid} while timing the seed"));
    }
    let nid = selection
        .choices
        .get(cid)
        .ok_or_else(|| format!("seed has no choice for class {cid}"))?;
    let node = &egraph[nid];
    let mut child_max = 0.0f64;
    for child in &node.children {
        let ccid = egraph.nid_to_cid(child).clone();
        let t = arrival_of(egraph, &ccid, selection, memo, on_stack)?;
        if t > child_max {
            child_max = t;
        }
    }
    let total = node.delay.into_inner() + child_max;
    on_stack.shift_remove(cid);
    memo.insert(cid.clone(), total);
    Ok(total)
}

/// Longest-path depth from the roots along the selected edges, which is a valid
/// assignment for `ilp_cbc::block_cycles`' `level` columns:
/// the rows demand `level_child >= level_parent + 1` for every *selected* edge,
/// and merely `level_parent - level_child <= |classes|` for unselected ones.
/// Depths lie in `[0, |classes| - 1]`, so both hold.
pub fn topological_levels(
    egraph: &EGraph,
    roots: &[ClassId],
    selection: &ExtractionResult,
) -> IndexMap<ClassId, f64> {
    // Reverse post-order = topological order (parents before children).
    let mut order: Vec<ClassId> = Vec::new();
    let mut state: IndexMap<ClassId, u8> = IndexMap::default();
    for root in roots {
        post_order(egraph, root, selection, &mut state, &mut order);
    }
    order.reverse();

    let mut level: IndexMap<ClassId, f64> = IndexMap::default();
    for cid in &order {
        level.entry(cid.clone()).or_insert(0.0);
    }
    for cid in &order {
        let here = *level.get(cid).unwrap_or(&0.0);
        let Some(nid) = selection.choices.get(cid) else {
            continue;
        };
        for child in &egraph[nid].children {
            let ccid = egraph.nid_to_cid(child).clone();
            let e = level.entry(ccid).or_insert(0.0);
            if *e < here + 1.0 {
                *e = here + 1.0;
            }
        }
    }
    level
}

fn post_order(
    egraph: &EGraph,
    cid: &ClassId,
    selection: &ExtractionResult,
    state: &mut IndexMap<ClassId, u8>,
    order: &mut Vec<ClassId>,
) {
    if state.get(cid).is_some() {
        return;
    }
    state.insert(cid.clone(), 1);
    if let Some(nid) = selection.choices.get(cid) {
        for child in &egraph[nid].children {
            let ccid = egraph.nid_to_cid(child).clone();
            post_order(egraph, &ccid, selection, state, order);
        }
    }
    order.push(cid.clone());
}

/// Hand `seed` to the solver as a MIP start, including values for every
/// auxiliary column (levels, `1 - x` switches, arrival times, root depth) so
/// the point is feasible for the whole model, not just the binaries.
///
/// `seed` must be complete and acyclic for `roots`, and `arrival` must be
/// [`arrival_times`] of it whenever the model has depth variables.
pub(crate) fn push_seed<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
    roots: &[ClassId],
    vars: &IlpVars<M::Col>,
    seed: &ExtractionResult,
    arrival: Option<&IndexMap<ClassId, f64>>,
) {
    // Selection binaries.
    for (cid, cvars) in &vars.classes {
        let chosen = seed.choices.get(cid);
        model.set_col_initial_solution(cvars.active, if chosen.is_some() { 1.0 } else { 0.0 });
        for (nid, &col) in egraph[cid].nodes.iter().zip(&cvars.nodes) {
            let on = chosen == Some(nid);
            model.set_col_initial_solution(col, if on { 1.0 } else { 0.0 });
            // opposite == 1 - x
            if let Some(&opp) = vars.cycles.opposite.get(&col) {
                model.set_col_initial_solution(opp, if on { 0.0 } else { 1.0 });
            }
        }
    }

    // Topological levels for the cycle-blocking rows.
    let levels = topological_levels(egraph, roots, seed);
    for (cid, &col) in &vars.cycles.levels {
        model.set_col_initial_solution(col, *levels.get(cid).unwrap_or(&0.0));
    }

    // Arrival times, clamped into their column bounds so a last-ulp overshoot
    // cannot put the start outside them, and the root depth.
    if let (Some(depth), Some(times)) = (&vars.depth, arrival) {
        for (cid, &col) in &depth.arrival {
            let v = times.get(cid).cloned().unwrap_or(0.0).clamp(0.0, depth.cap);
            model.set_col_initial_solution(col, v);
        }
        if let Some(t) = depth.root_depth {
            let t_val = roots
                .iter()
                .map(|r| times.get(r).cloned().unwrap_or(0.0))
                .fold(0.0, f64::max);
            model.set_col_initial_solution(t, t_val.clamp(0.0, depth.cap));
        }
    }
}
