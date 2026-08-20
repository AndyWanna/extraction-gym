/* An ILP extractor that returns the optimal DAG-extraction.

This extractor is simple so that it's easy to see that it's correct.

If the timeout is reached, it will return the result of the faster-greedy-dag extractor.

The solver is abstracted behind `crate::milp::MilpModel`, so the same model can
be built for CBC / Gurobi / HiGHS. The free functions here are generic over the
backend `M`; the public extractor structs default to `milp::DefaultMilp` (the
one backend feature that is enabled), and `*_with::<M>()` methods let a caller
pin a backend explicitly. The historical `*CbcExtractor` names are kept as type
aliases so downstream crates don't have to change.
*/

use super::faster_ilp_cbc::WarmConfig;
use super::warm::{self, SolveReport, WarmStartMode};
use super::*;
use crate::milp::{DefaultMilp, MilpModel, MilpSense, MilpSolution};
use indexmap::IndexSet;
use std::time::Instant;

/// Selection variables for one e-class: one "class is active" column plus one
/// column per member node. Generic over the backend's column handle.
struct ClassVars<C> {
    active: C,
    nodes: Vec<C>,
}

/// The auxiliary columns `block_cycles` creates, returned so a warm start can
/// give them values too.
///
/// A MIP start that fixes only the binaries leaves these unset, and CBC in
/// particular reads an unset column as `0.0` — which violates
/// `level_child >= level_parent + 1` for every selected edge. That is one of
/// the two concrete ways the previously-disabled warm-start code fed the solver
/// an infeasible point. Seeding them explicitly removes the guesswork for every
/// backend, not just CBC.
struct CycleVars<C> {
    /// One continuous "topological level" column per class.
    levels: IndexMap<ClassId, C>,
    /// `opposite[x_n] == 1 - x_n`, the big-M switch that disables a level
    /// constraint when its node is not selected.
    opposite: IndexMap<C, C>,
}

/// DAG-optimal ILP extractor with a compile-time timeout. Backend-generic; use
/// [`CbcExtractorWithTimeout`] for the previous (CBC) spelling.
pub struct IlpExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32>;

/// Legacy name for [`IlpExtractorWithTimeout`].
///
/// NOTE: a type alias to a *unit* struct cannot be used in expression position
/// (`CbcExtractorWithTimeout::<10>` as a value is rejected by rustc), so
/// construct it with `::new()` or use the neutral name directly.
pub type CbcExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32> =
    IlpExtractorWithTimeout<TIMEOUT_IN_SECONDS>;

impl<const TIMEOUT_IN_SECONDS: u32> Extractor for IlpExtractorWithTimeout<TIMEOUT_IN_SECONDS> {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract::<DefaultMilp>(
            egraph,
            roots,
            TIMEOUT_IN_SECONDS,
            1,
            &WarmConfig::default(),
        )
        .0;
    }
}

impl<const TIMEOUT_IN_SECONDS: u32> IlpExtractorWithTimeout<TIMEOUT_IN_SECONDS> {
    /// Constructor, so the type aliases can also be used in expression
    /// position (`CbcExtractorWithTimeout::<10>::new()`).
    pub const fn new() -> Self {
        Self
    }

    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        extract::<M>(egraph, roots, TIMEOUT_IN_SECONDS, 1, &WarmConfig::default()).0
    }
}

/// DAG-optimal ILP extractor with a runtime timeout. Backend-generic; use
/// [`CbcExtractor`] for the previous (CBC) spelling.
pub struct IlpExtractor {
    /// Solver time limit in seconds (`u32::MAX` for unbounded).
    pub timeout_seconds: u32,
    /// Solver threads for each solve. `1` unless the caller opted in; see
    /// [`MilpModel::set_threads`] for the slot-allocation invariant.
    pub threads: u32,
    /// Warm start + solver-log configuration; default = no MIP start, no log.
    pub warm: WarmConfig,
}

impl Default for IlpExtractor {
    fn default() -> Self {
        IlpExtractor {
            timeout_seconds: u32::MAX,
            threads: 1,
            warm: WarmConfig::default(),
        }
    }
}

/// Legacy name for [`IlpExtractor`].
pub type CbcExtractor = IlpExtractor;

impl Extractor for IlpExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract::<DefaultMilp>(
            egraph,
            roots,
            self.timeout_seconds,
            self.threads,
            &self.warm,
        )
        .0;
    }
}

impl IlpExtractor {
    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        self.extract_with_report::<M>(egraph, roots).0
    }

    /// Same as [`Self::extract_with`], plus the [`SolveReport`].
    pub fn extract_with_report<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> (ExtractionResult, SolveReport) {
        extract::<M>(egraph, roots, self.timeout_seconds, self.threads, &self.warm)
    }
}

fn extract<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let mut report = SolveReport::new(M::NAME, "dag-ilp", threads, timeout_seconds, warm.mode);
    let mut model = M::new();

    model.set_time_limit_seconds(timeout_seconds);
    model.set_threads(threads);
    if let Some(path) = &warm.milp_log {
        model.set_log_file(path);
        report.milp_log = Some(path.clone());
    }

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

    model.set_obj_sense(MilpSense::Minimize);
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

    let cyc = block_cycles(&mut model, &vars, &egraph);

    apply_warm_start(
        &mut model, egraph, roots, &vars, &cyc, None, warm, &mut report,
    );

    let solve_clock = Instant::now();
    let solution = model.solve();
    report.num_solves = 1;
    report.solve_wall_secs = solve_clock.elapsed().as_secs_f64();
    report.record_solution(&solution);
    report.load_trajectory();
    log::info!(
        "{} status {}, obj = {}",
        M::NAME,
        solution.status_detail(),
        solution.obj_value(),
    );

    if !solution.has_solution() {
        // No greedy stand-in: the CALLER holds the input mapping and its own
        // seed and floors against them. An empty
        // result says "the solver produced nothing", which is the only fact
        // this crate knows.
        log::info!(
            "{} returned no solution ({}); returning an empty selection for the \
             caller to floor against its own candidates",
            M::NAME,
            solution.status_detail()
        );
        report.returned_fallback = true;
        return (ExtractionResult::default(), report);
    }

    if !solution.ran_to_completion() {
        debug_assert!(timeout_seconds != std::u32::MAX);
        // Keep the incumbent. `has_solution` is checked above, so reaching here
        // means one EXISTS -- the old unconditional fallback threw away a valid,
        // strictly better answer on every timeout. Returning it is sound:
        // `block_cycles` encodes a topological order, so any feasible solution
        // is acyclic by construction, and this model has no budget to violate.
        // Same fix as the delay-budget extractor below.
        log::info!(
            "Unfinished {} solution, but an incumbent exists (obj = {}); returning it",
            M::NAME,
            solution.obj_value(),
        );
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

    return (result, report);
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

    CycleVars { levels, opposite }
}

/// The arrival-time part of a warm start, for the two delay-aware models.
struct ArrivalSeed<'a, C> {
    /// `T_c` column per class.
    arr: &'a IndexMap<ClassId, C>,
    /// Upper bound that was applied to every `T_c` (`S` for dual, the budget
    /// for delay-budget). Seeded values are clamped into `[0, cap]` so a
    /// last-ulp overshoot cannot put the start outside its own column bounds.
    cap: f64,
    /// The dual model's extra `t >= T_root` objective column.
    t: Option<C>,
    /// Hard delay budget the seed must satisfy, if the model has one. This is
    /// the check the task calls for: evaluate the candidate in the ILP's own
    /// arrival recurrence and compare against the budget the ILP enforces.
    budget: Option<f64>,
}

/// Warm-start entry point shared by the three `ilp_cbc` models.
///
/// Everything here is refusable: any problem with the candidate ends in
/// [`SolveReport::refuse_warm_start`] and a solve with no MIP start, never a
/// panic and never an infeasible point pushed to the solver.
#[allow(clippy::too_many_arguments)]
fn apply_warm_start<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
    roots: &[ClassId],
    vars: &IndexMap<ClassId, ClassVars<M::Col>>,
    cyc: &CycleVars<M::Col>,
    arrival: Option<ArrivalSeed<M::Col>>,
    warm: &WarmConfig,
    report: &mut SolveReport,
) {
    if warm.mode == WarmStartMode::None {
        return;
    }
    if !model.supports_warm_start() {
        report.refuse_warm_start(format!(
            "{} does not support a trustworthy MIP start (see CbcModel::supports_warm_start)",
            M::NAME
        ));
        return;
    }

    // A seed is only ever the caller's. This crate no longer computes one:
    // exgym's own area-greedy was both the seed AND the acceptance bar, and it
    // carries no bound relative to the input mapping the caller is holding,
    // so a run could ship something bigger than its own input. The caller
    // picks the best of its own candidates — including the input mapping —
    // and passes the winner here.
    let (preferred, effective, mut note) = match warm.mode {
        WarmStartMode::Greedy => {
            return report.refuse_warm_start(
                "greedy seeding was removed: this crate no longer computes a seed of its \
                 own (it was unbounded in how much worse than the input mapping it could \
                 be). Pass a caller-supplied seed with WarmStartMode::Initial.",
            )
        }
        WarmStartMode::Initial => match &warm.seed {
            Some(s) => (Some(s), WarmStartMode::Initial, None),
            None => {
                return report.refuse_warm_start(
                    "no initial extraction was supplied and there is no greedy seed to \
                     degrade to; running with no MIP start",
                )
            }
        },
        WarmStartMode::None => unreachable!(),
    };

    // Repair fallback for a class the preferred seed cannot fill: `build_seed`
    // drops through to the cheapest member of that class (a LOCAL choice inside
    // the class the walk already committed to), not to a whole greedy extraction.
    let repair = ExtractionResult::default();
    let seed = match warm::build_seed(egraph, roots, preferred, &repair) {
        Ok(s) => s,
        Err(why) => return report.refuse_warm_start(why),
    };
    if seed.repaired > 0 {
        let extra = format!("{} class(es) repaired off the preferred node", seed.repaired);
        note = Some(match note {
            Some(n) => format!("{n}; {extra}"),
            None => extra,
        });
    }

    // Arrival times, in the ILP's own recurrence.
    let arrival_times = match &arrival {
        Some(_) => match warm::arrival_times(egraph, roots, &seed.selection) {
            Ok(t) => Some(t),
            Err(why) => return report.refuse_warm_start(why),
        },
        None => None,
    };

    // THE feasibility gate for the delay-budget model. Note it is checked
    // against EVERY class, not just the roots: `extract_delay_budget` caps every
    // arrival column at the budget, so a single interior class over budget makes
    // the whole point infeasible.
    if let (Some(a), Some(times)) = (&arrival, &arrival_times) {
        if let Some(budget) = a.budget {
            let worst = times.values().cloned().fold(0.0f64, f64::max);
            if worst > budget {
                return report.refuse_warm_start(format!(
                    "candidate seed violates the delay budget in the ILP's own arrival \
                     formulation: worst class arrival {worst:.9} > budget {budget:.9} \
                     (over by {:.3e}). Greedy seeds minimise area and carry no delay \
                     guarantee; a design-derived budget is also exactly tight, so any \
                     positive margin is a violation.",
                    worst - budget
                ));
            }
            let extra = format!("delay slack {:.3e} under the budget", budget - worst);
            note = Some(match note {
                Some(n) => format!("{n}; {extra}"),
                None => extra,
            });
        }
    }

    // -- push the point ------------------------------------------------------
    // Selection binaries.
    for (cid, cvars) in vars {
        let chosen = seed.selection.choices.get(cid);
        model.set_col_initial_solution(cvars.active, if chosen.is_some() { 1.0 } else { 0.0 });
        for (nid, &col) in egraph[cid].nodes.iter().zip(&cvars.nodes) {
            let on = chosen == Some(nid);
            model.set_col_initial_solution(col, if on { 1.0 } else { 0.0 });
            // opposite == 1 - x
            if let Some(&opp) = cyc.opposite.get(&col) {
                model.set_col_initial_solution(opp, if on { 0.0 } else { 1.0 });
            }
        }
    }

    // Topological levels for the cycle-blocking rows.
    let levels = warm::topological_levels(egraph, roots, &seed.selection);
    for (cid, &col) in &cyc.levels {
        model.set_col_initial_solution(col, *levels.get(cid).unwrap_or(&0.0));
    }

    // Arrival times (and the dual model's `t`).
    if let (Some(a), Some(times)) = (&arrival, &arrival_times) {
        let mut t_val = 0.0f64;
        for (cid, &col) in a.arr {
            let v = times.get(cid).cloned().unwrap_or(0.0).clamp(0.0, a.cap);
            model.set_col_initial_solution(col, v);
        }
        for root in roots {
            t_val = t_val.max(times.get(root).cloned().unwrap_or(0.0));
        }
        if let Some(t) = a.t {
            model.set_col_initial_solution(t, t_val.clamp(0.0, a.cap));
        }
    }

    report.warm_start_objective = warm::try_dag_cost(egraph, roots, &seed.selection);
    report.accept_warm_start(effective, note);
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

/// Joint delay+area ILP extractor. Backend-generic; use [`DualCbcExtractor`]
/// for the previous (CBC) spelling.
pub struct DualIlpExtractor {
    /// Solver time limit in seconds (u32::MAX for unbounded).
    pub timeout_seconds: u32,
    /// Weight on the critical-path delay.
    pub alpha: f64,
    /// Weight on the total area.
    pub beta: f64,
    /// Solver threads for each solve. `1` unless the caller opted in; see
    /// [`MilpModel::set_threads`] for the slot-allocation invariant.
    pub threads: u32,
    /// Warm start + solver-log configuration; default = no MIP start, no log.
    pub warm: WarmConfig,
}

impl Default for DualIlpExtractor {
    fn default() -> Self {
        DualIlpExtractor {
            timeout_seconds: u32::MAX,
            alpha: 1.0,
            beta: 1.0,
            threads: 1,
            warm: WarmConfig::default(),
        }
    }
}

/// Legacy name for [`DualIlpExtractor`].
pub type DualCbcExtractor = DualIlpExtractor;

impl Extractor for DualIlpExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        extract_dual::<DefaultMilp>(
            egraph,
            roots,
            self.timeout_seconds,
            self.alpha,
            self.beta,
            self.threads,
            &self.warm,
        )
        .0
    }
}

impl DualIlpExtractor {
    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        self.extract_with_report::<M>(egraph, roots).0
    }

    /// Same as [`Self::extract_with`], plus the [`SolveReport`].
    pub fn extract_with_report<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> (ExtractionResult, SolveReport) {
        extract_dual::<M>(
            egraph,
            roots,
            self.timeout_seconds,
            self.alpha,
            self.beta,
            self.threads,
            &self.warm,
        )
    }
}

/// Builds the selection variables/constraints (identical to the sum-cost
/// `extract`) plus the arrival-time variables `T_c` per e-class with their
/// big-M constraints:
///
///   T_c >= delay_n + T_cc - M*(1 - x_n)   for each child class cc of n
///   T_c >= delay_n        - M*(1 - x_n)   for childless n
///
/// bounded by `0 <= T_c <= S` (S = sum of all delays). The big-M is chosen by
/// the caller-supplied `big_m_fn(s, d_max)` where `s` = sum of all node delays
/// and `d_max` = the max single-node delay: the loose but always-sound `2S`
/// (`|s, _| 2*s`, used by `dual`, whose arrival cols keep their `[0, S]` bound),
/// or the tight `budget + d_max` (used by `delay_budget`, which re-caps every
/// arrival col at `budget`, so `M >= budget + max(node_delay)` is exactly the
/// sound bound — see the arrival-row derivation). Shared by every extractor in
/// this module that needs critical-path delay. Returns the class selection
/// vars, the arrival-time column per class, and `S`.
fn build_selection_and_arrival<M: MilpModel>(
    model: &mut M,
    egraph: &EGraph,
    big_m_fn: impl Fn(f64, f64) -> f64,
) -> (
    IndexMap<ClassId, ClassVars<M::Col>>,
    IndexMap<ClassId, M::Col>,
    f64,
) {
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
    // Max single-node delay: the physical quantity the tight big-M scales with
    // (bounded by the biggest library cell, unlike S which grows with #nodes).
    let d_max: f64 = egraph
        .classes()
        .values()
        .flat_map(|c| c.nodes.iter())
        .map(|nid| node_delay(&egraph[nid]))
        .fold(0.0_f64, f64::max);
    let big_m = {
        let m = big_m_fn(s, d_max);
        if m > 0.0 { m } else { 1.0 }
    };

    let arr: IndexMap<ClassId, M::Col> = vars
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

#[allow(clippy::too_many_arguments)]
fn extract_dual<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    alpha: f64,
    beta: f64,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let mut report = SolveReport::new(M::NAME, "dual-ilp", threads, timeout_seconds, warm.mode);
    let mut model = M::new();
    model.set_time_limit_seconds(timeout_seconds);
    model.set_threads(threads);
    if let Some(path) = &warm.milp_log {
        model.set_log_file(path);
        report.milp_log = Some(path.clone());
    }

    // Dual keeps arrival cols on their [0, S] bound, so the sound big-M is 2S.
    let (vars, arr, s) = build_selection_and_arrival(&mut model, egraph, |s, _d_max| 2.0 * s);

    // Objective: alpha * t + beta * sum(area * x), where t >= max root arrival.
    model.set_obj_sense(MilpSense::Minimize);

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

    let cyc = block_cycles(&mut model, &vars, &egraph);

    // No budget: every complete acyclic selection satisfies the dual model, so
    // the arrival seed is informational (it just saves the solver the LP) and
    // can never be rejected on feasibility grounds.
    apply_warm_start(
        &mut model,
        egraph,
        roots,
        &vars,
        &cyc,
        Some(ArrivalSeed {
            arr: &arr,
            cap: s.max(0.0),
            t: Some(t),
            budget: None,
        }),
        warm,
        &mut report,
    );

    let solve_clock = Instant::now();
    let solution = model.solve();
    report.num_solves = 1;
    report.solve_wall_secs = solve_clock.elapsed().as_secs_f64();
    report.record_solution(&solution);
    report.load_trajectory();
    log::info!(
        "Dual {} status {}, obj = {}",
        M::NAME,
        solution.status_detail(),
        solution.obj_value(),
    );

    if !solution.has_solution() {
        // No greedy stand-in: the CALLER holds the input mapping and its own
        // seed and floors against them. An empty
        // result says "the solver produced nothing", which is the only fact
        // this crate knows.
        log::info!(
            "{} returned no solution ({}); returning an empty selection for the \
             caller to floor against its own candidates",
            M::NAME,
            solution.status_detail()
        );
        report.returned_fallback = true;
        return (ExtractionResult::default(), report);
    }

    if !solution.ran_to_completion() {
        debug_assert!(timeout_seconds != std::u32::MAX);
        // Keep the incumbent -- see the identical fix in the area extractor.
        // The area-greedy fallback optimises AREA ONLY, not `alpha*delay +
        // beta*area`, so discarding a dual incumbent here traded a better joint
        // solution for one that ignores the objective the caller asked for.
        log::info!(
            "Unfinished dual {} solution, but an incumbent exists (obj = {}); returning it",
            M::NAME,
            solution.obj_value(),
        );
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

    (result, report)
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

/// Minimum-area-under-a-delay-budget ILP extractor. Backend-generic; use
/// [`DelayBudgetCbcExtractor`] for the previous (CBC) spelling.
pub struct DelayBudgetIlpExtractor {
    /// Solver time limit in seconds (u32::MAX for unbounded).
    pub timeout_seconds: u32,
    /// Hard upper bound on the critical-path delay of the extraction.
    pub max_delay: f64,
    /// Solver threads for each solve. `1` unless the caller opted in; see
    /// [`MilpModel::set_threads`] for the slot-allocation invariant.
    pub threads: u32,
    /// Warm start + solver-log configuration; default = no MIP start, no log.
    pub warm: WarmConfig,
}

impl Default for DelayBudgetIlpExtractor {
    fn default() -> Self {
        DelayBudgetIlpExtractor {
            timeout_seconds: u32::MAX,
            max_delay: f64::INFINITY,
            threads: 1,
            warm: WarmConfig::default(),
        }
    }
}

/// Legacy name for [`DelayBudgetIlpExtractor`].
pub type DelayBudgetCbcExtractor = DelayBudgetIlpExtractor;

impl Extractor for DelayBudgetIlpExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        extract_delay_budget::<DefaultMilp>(
            egraph,
            roots,
            self.timeout_seconds,
            self.max_delay,
            self.threads,
            &self.warm,
        )
        .0
    }
}

impl DelayBudgetIlpExtractor {
    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        self.extract_with_report::<M>(egraph, roots).0
    }

    /// Same as [`Self::extract_with`], plus the [`SolveReport`].
    pub fn extract_with_report<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> (ExtractionResult, SolveReport) {
        extract_delay_budget::<M>(
            egraph,
            roots,
            self.timeout_seconds,
            self.max_delay,
            self.threads,
            &self.warm,
        )
    }
}

fn extract_delay_budget<M: MilpModel>(
    egraph: &EGraph,
    roots: &[ClassId],
    timeout_seconds: u32,
    max_delay: f64,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let mut report = SolveReport::new(
        M::NAME,
        "delay-budget-ilp",
        threads,
        timeout_seconds,
        warm.mode,
    );
    let mut model = M::new();
    model.set_time_limit_seconds(timeout_seconds);
    model.set_threads(threads);
    if let Some(path) = &warm.milp_log {
        model.set_log_file(path);
        report.milp_log = Some(path.clone());
    }

    // Tight, well-conditioned big-M: since we cap every arrival col at `budget`
    // (below), the sound bound for the arrival relaxation is exactly
    // `M = budget + D_max` (D_max = max single-node delay ~ the biggest library
    // cell, NOT a path delay). With real per-gate delays on every node (the
    // boolean-op primitives now carry their AN2/OR2/XOR2/INV delays instead of a
    // 1e4/1e9 sentinel), D_max is physical (~100 ps), so this M is ~1x the RHS
    // scale and the solver's tolerances stay well-conditioned — unlike the loose
    // 2S, whose S was inflated by the old sentinels into a ~1e7 coefficient.
    let (vars, arr, s) = build_selection_and_arrival(&mut model, egraph, |s, d_max| {
        let budget = max_delay.min(s.max(0.0)).max(0.0);
        budget + d_max
    });

    // Tighten every arrival-time variable's upper bound to the budget. This
    // enforces T_root <= max_delay for every root with no extra rows, and is
    // sound for every class (see comment above).
    let budget = max_delay.min(s.max(0.0)).max(0.0);
    let d_max: f64 = egraph
        .classes()
        .values()
        .flat_map(|c| c.nodes.iter())
        .map(|nid| node_delay(&egraph[nid]))
        .fold(0.0_f64, f64::max);
    log::info!(
        "Delay-budget: S={:.2}, D_max={:.2}, budget={:.2} (max_delay={:.2}), big_M={:.2}",
        s, d_max, budget, max_delay, budget + d_max,
    );
    for &t_c in arr.values() {
        model.set_col_upper(t_c, budget);
    }

    // Objective: minimise area alone.
    model.set_obj_sense(MilpSense::Minimize);
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

    let cyc = block_cycles(&mut model, &vars, &egraph);

    // The budget is the one hard side constraint in this crate's models, so it
    // is where a warm start can genuinely be infeasible. `apply_warm_start`
    // evaluates the candidate through `warm::arrival_times` — the same
    // recurrence these rows encode — and refuses (with the margin in the log)
    // rather than seeding an infeasible point.
    apply_warm_start(
        &mut model,
        egraph,
        roots,
        &vars,
        &cyc,
        Some(ArrivalSeed {
            arr: &arr,
            cap: budget,
            t: None,
            budget: Some(budget),
        }),
        warm,
        &mut report,
    );

    let solve_clock = Instant::now();
    let solution = model.solve();
    report.num_solves = 1;
    report.solve_wall_secs = solve_clock.elapsed().as_secs_f64();
    report.record_solution(&solution);
    report.load_trajectory();
    log::info!(
        "Delay-budget {} status {}, obj = {}",
        M::NAME,
        solution.status_detail(),
        solution.obj_value(),
    );

    if solution.is_infeasible() {
        log::info!("Infeasible, returning empty solution");
        return (ExtractionResult::default(), report);
    }

    if !solution.ran_to_completion() {
        debug_assert!(timeout_seconds != std::u32::MAX);
        // Fall back ONLY when the solver has no incumbent to return.
        //
        // This previously fell back on ANY timeout, discarding a perfectly good
        // answer. Unlike the area extractor's model, this one needs no
        // solve/find-cycles/re-block loop: `block_cycles` (see the call above)
        // encodes a topological ordering directly into the constraints, and the
        // arrival-time rows bound every path by `max_delay`. So ANY feasible
        // solution here is acyclic AND within budget by construction -- there is
        // nothing left to validate before returning it. (Structural nodes carry
        // delay EPS = 1e-6 rather than 0, so a zero-delay cycle cannot sneak
        // past the arrival-time rows either.)
        //
        // The fallback, by contrast, optimises area only and ignores max_delay,
        // so the old code could discard a budget-feasible incumbent in favour of
        // a possibly budget-violating greedy result.
        if !solution.has_solution() {
            log::info!(
                "Unfinished delay-budget {} solution with no incumbent; returning an \
                 empty selection for the caller to floor against its own candidates",
                M::NAME
            );
            report.returned_fallback = true;
            return (ExtractionResult::default(), report);
        }
        log::info!(
            "Unfinished delay-budget {} solution, but an incumbent exists (obj = {}); \
             returning it -- acyclic and within budget by construction",
            M::NAME,
            solution.obj_value(),
        );
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

    (result, report)
}
