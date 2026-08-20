/*
Not the default DAG-area extractor any more; see ilp_cbc::IlpExtractor.

The lazy cycle loop below discards any timed-out solution that still contains a
cycle, so at a short time cap it returns the greedy seed almost every time.
ilp_cbc encodes a topological order up front, so every feasible solution is
acyclic by construction.

Also note `initial_result` below is exgym's own area-greedy, used here as the
seed, the acceptance bar AND `remove_high_cost`'s pruning bound. It is unbounded
in how much worse than the caller's input mapping it can be. `ilp_cbc` no longer
computes a greedy of any kind; do not copy this pattern back.

Produces a dag-cost optimal extraction of an Egraph.

This can take >10 hours to run on some egraphs, so there's the option to provide a timeout.

To operate:
1) It simplifies the egraph by removing nodes that can't be selected in the optimal
solution, as well as collapsing other classes down.
2) It then sends the problem to the COIN-OR CBC solver to find an extraction (or timeout).
It allows the solver to generate solutions that contain cycles
3) The solution from the solver is checked, and if the extraction contains a cycle, extra
constraints are added to block the cycle and the solver is called again.

In SAT solving, it's common to call a solver incrementally. Each time you call the SAT
solver with more clauses to the SAT solver (constraining the solution further), and
allowing the SAT solver to reuse its previous work.

So there are two uses of "incremental", one is gradually sending more of the problem to the solver,
and the other is the solver being able to re-use the previous work when it receives additional parts
of the problem. In the case here, we're just referring to sending extra pieces of the problem to
the solver. COIN-OR CBC doesn't provide an interface that allows us to call it and reuse what it
has discovered previously.

In the case of COIN-OR CBC, we're sending extra constraints each time we're solving, these
extra constraints are prohibiting cycles that were found in the solutions that COIN-OR CBC
previously produced.

Obviously, we could add constraints to block all the cycles the first time we call COIN-OR CBC,
so we'd only need to call the solver once. However, for the problems in our test-set, lots of these
constraints don't change the answer, they're removing cycles from high-cost extractions.  These
extra constraints do slow down solving though - and for our test-set it gives a faster runtime when
we incrementally add constraints that break cycles when they occur in the lowest cost extraction.

We've experimented with two ways to break cycles.

One approach is by enforcing a topological sort on nodes. Each node has a level, and each edge
can only connect from a lower level to a higher level node.

Another approach, is by explicity banning cycles. Say in an extraction that the solver generates
we find a cycle A->B->A. Say there are two edges, edgeAB, and edgeBA, which connect A->B, then B->A.
Then any solution that contains both edgeAB, and edgeBA will contain a cycle.  So we add a constraint
that at most one of these two edges can be active. If we check through the whole extraction for cycles,
and ban each cycle that we find, then try solving again, we'll get a new solution which, if it contains
cycles, will not contain any of the cycles we've previously seen. We repeat this until timeout, or until
we get an optimal solution without cycles.


*/

use super::warm::{SolveReport, WarmStartMode};
use super::*;
use crate::milp::{DefaultMilp, MilpModel, MilpSolution};
use indexmap::IndexSet;
use std::fmt;
use std::time::{Instant, SystemTime};

#[derive(Debug)]
pub struct Config {
    pub pull_up_costs: bool,
    pub remove_self_loops: bool,
    pub remove_high_cost_nodes: bool,
    pub remove_more_expensive_subsumed_nodes: bool,
    pub remove_unreachable_classes: bool,
    pub pull_up_single_parent: bool,
    pub take_intersection_of_children_in_class: bool,
    pub move_min_cost_of_members_to_class: bool,
    pub find_extra_roots: bool,
    pub remove_empty_classes: bool,
    pub return_improved_on_timeout: bool,
    pub remove_single_zero_cost: bool,
}

impl Config {
    pub const fn default() -> Self {
        Self {
            pull_up_costs: true,
            remove_self_loops: true,
            remove_high_cost_nodes: true,
            remove_more_expensive_subsumed_nodes: true,
            remove_unreachable_classes: true,
            pull_up_single_parent: true,
            take_intersection_of_children_in_class: true,
            move_min_cost_of_members_to_class: false,
            find_extra_roots: true,
            remove_empty_classes: true,
            return_improved_on_timeout: true,
            remove_single_zero_cost: true,
        }
    }
}

struct NodeILP<C> {
    variable: C,
    cost: Cost,
    member: NodeId,
    children_classes: IndexSet<ClassId>,
}

struct ClassILP<C> {
    active: C,
    members: Vec<NodeId>,
    variables: Vec<C>,
    costs: Vec<Cost>,
    // Initially this contains the children of each member (respectively), but
    // gets edited during the run, so mightn't match later on.
    childrens_classes: Vec<IndexSet<ClassId>>,
}

impl<C: Copy> fmt::Debug for ClassILP<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "classILP[{}] {{ node: {:?}, children: {:?},  cost: {:?} }}",
            self.members(),
            self.members,
            self.childrens_classes,
            self.costs
        )
    }
}

impl<C: Copy> ClassILP<C> {
    fn remove(&mut self, idx: usize) {
        self.variables.remove(idx);
        self.costs.remove(idx);
        self.members.remove(idx);
        self.childrens_classes.remove(idx);
    }

    fn remove_node(&mut self, node_id: &NodeId) {
        if let Some(idx) = self.members.iter().position(|n| n == node_id) {
            self.remove(idx);
        }
    }

    fn members(&self) -> usize {
        self.variables.len()
    }

    fn check(&self) {
        assert_eq!(self.variables.len(), self.costs.len());
        assert_eq!(self.variables.len(), self.members.len());
        assert_eq!(self.variables.len(), self.childrens_classes.len());
    }

    fn as_nodes(&self) -> Vec<NodeILP<C>> {
        self.variables
            .iter()
            .zip(&self.costs)
            .zip(&self.members)
            .zip(&self.childrens_classes)
            .map(|(((variable, &cost_), member), children_classes)| NodeILP {
                variable: *variable,
                cost: cost_,
                member: member.clone(),
                children_classes: children_classes.clone(),
            })
            .collect()
    }

    fn get_children_of_node(&self, node_id: &NodeId) -> &IndexSet<ClassId> {
        let idx = self.members.iter().position(|n| n == node_id).unwrap();
        &self.childrens_classes[idx]
    }

    #[allow(dead_code)]
    fn get_variable_for_node(&self, node_id: &NodeId) -> Option<C> {
        if let Some(idx) = self.members.iter().position(|n| n == node_id) {
            return Some(self.variables[idx]);
        }
        None
    }
}

/// Warm-start configuration shared by every ILP extractor in this crate.
///
/// Kept as one field on the extractor structs rather than two, so adding warm
/// starts did not turn every struct-literal construction site into a churn
/// point (`..Default::default()` covers it).
#[derive(Default, Clone)]
pub struct WarmConfig {
    /// Which starting incumbent to use.
    pub mode: WarmStartMode,
    /// The caller-supplied "initial" extraction for [`WarmStartMode::Initial`],
    /// expressed over the same serialized e-graph the extractor is given.
    /// `None` with `mode == Initial` degrades to `Greedy` with a note.
    pub seed: Option<ExtractionResult>,
    /// Where to write the solver's own log, enabling the incumbent-vs-time
    /// trajectory. `None` keeps the solver silent (the historical behaviour).
    pub milp_log: Option<String>,
}

/// Simplifying DAG-optimal ILP extractor with a compile-time timeout.
/// Backend-generic; [`FasterCbcExtractorWithTimeout`] is the legacy spelling.
pub struct FasterIlpExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32>;

/// Legacy name for [`FasterIlpExtractorWithTimeout`].
/// NOTE: a type alias to a *unit* struct cannot be used in expression position
/// (`FasterCbcExtractorWithTimeout::<10>` as a value is rejected by rustc), so
/// construct it with `::new()` or use the neutral name directly.
pub type FasterCbcExtractorWithTimeout<const TIMEOUT_IN_SECONDS: u32> =
    FasterIlpExtractorWithTimeout<TIMEOUT_IN_SECONDS>;

// Some problems take >36,000 seconds to optimise.
impl<const TIMEOUT_IN_SECONDS: u32> Extractor
    for FasterIlpExtractorWithTimeout<TIMEOUT_IN_SECONDS>
{
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract::<DefaultMilp>(
            egraph,
            roots,
            &Config::default(),
            TIMEOUT_IN_SECONDS,
            1,
            &WarmConfig::default(),
        )
        .0;
    }
}

impl<const TIMEOUT_IN_SECONDS: u32> FasterIlpExtractorWithTimeout<TIMEOUT_IN_SECONDS> {
    /// Constructor, so the type aliases can also be used in expression
    /// position (`FasterCbcExtractorWithTimeout::<10>::new()`).
    pub const fn new() -> Self {
        Self
    }

    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        extract::<M>(
            egraph,
            roots,
            &Config::default(),
            TIMEOUT_IN_SECONDS,
            1,
            &WarmConfig::default(),
        )
        .0
    }
}

/// Simplifying DAG-optimal ILP extractor with a runtime timeout.
/// Backend-generic; [`FasterCbcExtractor`] is the legacy spelling.
pub struct FasterIlpExtractor {
    /// Solver time limit in seconds (`u32::MAX` for unbounded).
    ///
    /// NOTE: `u32::MAX` really does mean "run until it finishes", which on a
    /// hard instance means "hang". Callers that cannot supervise the process
    /// must pass a real limit.
    pub timeout_seconds: u32,
    /// Solver threads for each solve. `1` unless the caller opted in; see
    /// [`MilpModel::set_threads`] for the slot-allocation invariant.
    pub threads: u32,
    /// Warm start + solver-log configuration. Defaults to "no MIP start,
    /// no log", i.e. exactly the historical behaviour.
    pub warm: WarmConfig,
}

impl Default for FasterIlpExtractor {
    fn default() -> Self {
        FasterIlpExtractor {
            timeout_seconds: u32::MAX,
            threads: 1,
            warm: WarmConfig::default(),
        }
    }
}

/// Legacy name for [`FasterIlpExtractor`].
pub type FasterCbcExtractor = FasterIlpExtractor;

impl Extractor for FasterIlpExtractor {
    fn extract(&self, egraph: &EGraph, roots: &[ClassId]) -> ExtractionResult {
        return extract::<DefaultMilp>(
            egraph,
            roots,
            &Config::default(),
            self.timeout_seconds,
            self.threads,
            &self.warm,
        )
        .0;
    }
}

impl FasterIlpExtractor {
    /// Same as [`Extractor::extract`], but with the MILP backend pinned.
    pub fn extract_with<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> ExtractionResult {
        self.extract_with_report::<M>(egraph, roots).0
    }

    /// Same as [`Self::extract_with`], but also returns the [`SolveReport`]:
    /// backend, threads, timeout, warm start *as applied*, solver status,
    /// objective, bound, gap and (with `warm.milp_log` set) the
    /// incumbent-vs-time trajectory.
    pub fn extract_with_report<M: MilpModel>(
        &self,
        egraph: &EGraph,
        roots: &[ClassId],
    ) -> (ExtractionResult, SolveReport) {
        extract::<M>(
            egraph,
            roots,
            &Config::default(),
            self.timeout_seconds,
            self.threads,
            &self.warm,
        )
    }
}

fn extract<M: MilpModel>(
    egraph: &EGraph,
    roots_slice: &[ClassId],
    config: &Config,
    timeout: u32,
    threads: u32,
    warm: &WarmConfig,
) -> (ExtractionResult, SolveReport) {
    let mut report = SolveReport::new(M::NAME, "area-ilp", threads, timeout, warm.mode);
    // todo from now on we don't use roots_slice - be good to prevent using it any more.
    let mut roots = roots_slice.to_vec();
    roots.sort();
    roots.dedup();

    let simp_start_time = std::time::Instant::now();

    let mut model = M::new();
    //silence verbose stdout output
    model.set_log_level(0);
    model.set_threads(threads);
    // ... unless a trajectory was asked for, in which case route the solver's
    // own log to a file (still nothing on stdout). See `MilpModel::set_log_file`.
    if let Some(path) = &warm.milp_log {
        model.set_log_file(path);
        report.milp_log = Some(path.clone());
    }

    let n2c = |nid: &NodeId| egraph.nid_to_cid(nid);

    let mut vars: IndexMap<ClassId, ClassILP<M::Col>> = egraph
        .classes()
        .values()
        .map(|class| {
            let cvars = ClassILP {
                active: model.add_binary(),
                variables: class.nodes.iter().map(|_| model.add_binary()).collect(),
                costs: class.nodes.iter().map(|n| egraph[n].cost).collect(),
                members: class.nodes.clone(),
                childrens_classes: class
                    .nodes
                    .iter()
                    .map(|n| {
                        egraph[n]
                            .children
                            .iter()
                            .map(|c| n2c(c).clone())
                            .collect::<IndexSet<ClassId>>()
                    })
                    .collect(),
            };
            (class.id.clone(), cvars)
        })
        .collect();

    let initial_result = super::faster_greedy_dag::FasterGreedyDagExtractor.extract(egraph, &roots);
    let initial_result_cost = initial_result.dag_cost(egraph, &roots);
    // Reported unconditionally: on timeout the extractor may return this
    // instead of the solver's answer, and a reader who cannot tell the two
    // apart would mistake "greedy" for "what the solver achieved".
    report.initial_cost = Some(initial_result_cost.into_inner());

    // For classes where we know the choice already, we set the nodes early.
    let mut result = ExtractionResult::default();

    //This could be much more efficient, but it only takes less than 5 seconds for all our benchmarks.
    //The ILP solver takes the time.
    for _i in 1..3 {
        remove_with_loops(&mut vars, &roots, config);
        remove_high_cost(&mut vars, initial_result_cost, &roots, config);
        remove_more_expensive_subsumed_nodes(&mut vars, config);
        remove_unreachable_classes(&mut vars, &roots, config);
        pull_up_with_single_parent(&mut vars, &roots, config);
        pull_up_costs(&mut vars, &roots, config);
        remove_single_zero_cost(&mut vars, &mut result, &roots, config);
        find_extra_roots(&mut vars, &mut roots, config);
        remove_empty_classes(&mut vars, config);
    }

    for (classid, class) in &vars {
        if class.members() == 0 {
            if roots.contains(classid) {
                log::info!("Infeasible, root has no possible children, returning empty solution");
                report.status = Some(crate::milp::MilpStatus::Infeasible);
                return (ExtractionResult::default(), report);
            }

            model.set_col_upper(class.active, 0.0);
            continue;
        }

        if class.members() == 1 && class.childrens_classes[0].is_empty() && class.costs[0] == 0.0 {
            continue;
        }

        // class active == some node active
        // sum(for node_active in class) == class_active

        let row = model.add_row();
        model.set_row_equal(row, 0.0);
        model.set_weight(row, class.active, -1.0);
        for &node_active in &class.variables.iter().collect::<IndexSet<_>>() {
            model.set_weight(row, *node_active, 1.0);
        }

        let childrens_classes_var =
            |cc: &IndexSet<ClassId>| cc.iter().map(|n| vars[n].active).collect::<IndexSet<_>>();

        let mut intersection: IndexSet<M::Col> = Default::default();

        if config.take_intersection_of_children_in_class {
            // otherwise the intersection is empty (i.e. disabled.)
            intersection = childrens_classes_var(&class.childrens_classes[0].clone());
        }

        for childrens_classes in &class.childrens_classes[1..] {
            intersection = intersection
                .intersection(&childrens_classes_var(childrens_classes))
                .cloned()
                .collect();
        }

        // A class being active implies that all in the intersection
        // of it's children are too.
        for c in &intersection {
            let row = model.add_row();
            model.set_row_upper(row, 0.0);
            model.set_weight(row, class.active, 1.0);
            model.set_weight(row, *c, -1.0);
        }

        for (childrens_classes, &node_active) in
            class.childrens_classes.iter().zip(&class.variables)
        {
            for child_active in childrens_classes_var(childrens_classes) {
                // node active implies child active, encoded as:
                //   node_active <= child_active
                //   node_active - child_active <= 0
                if !intersection.contains(&child_active) {
                    let row = model.add_row();
                    model.set_row_upper(row, 0.0);
                    model.set_weight(row, node_active, 1.0);
                    model.set_weight(row, child_active, -1.0);
                }
            }
        }
    }

    for root in &roots {
        model.set_col_lower(vars[root].active, 1.0);
    }

    let mut objective_fn_terms = 0;

    for (_class_id, c_var) in &vars {
        let mut min_cost = 0.0;

        /* Moves the minimum of all the nodes up onto the class.
        Most helpful when the members of the class all have the same cost.
        For example if the members' costs are [1,1,1], three terms get
        replaced by one in the objective function.
        */

        if config.move_min_cost_of_members_to_class {
            min_cost = c_var
                .costs
                .iter()
                .min()
                .unwrap_or(&Cost::default())
                .into_inner();
        }

        if min_cost != 0.0 {
            model.set_obj_coeff(c_var.active, min_cost);
            objective_fn_terms += 1;
        }

        for (&node_active, &node_cost) in c_var.variables.iter().zip(c_var.costs.iter()) {
            if *node_cost - min_cost != 0.0 {
                model.set_obj_coeff(node_active, *node_cost - min_cost);
            }
        }
    }

    log::info!("Objective function terms: {}", objective_fn_terms);

    // -- warm start -----------------------------------------------------------
    //
    // For the AREA model every complete acyclic selection is feasible (there is
    // no side constraint to violate), so the only way to seed an infeasible
    // point is to seed an *incomplete* or *inconsistent* one — which is exactly
    // what the old `if false`-gated code did, because it walked the greedy
    // result against the SIMPLIFIED `vars` without checking that the chosen
    // node survived simplification. `seed_simplified` rebuilds the selection
    // against `vars` itself and refuses if it cannot.
    if warm.mode != WarmStartMode::None {
        if !model.supports_warm_start() {
            report.refuse_warm_start(format!(
                "{} does not support a trustworthy MIP start (see CbcModel::supports_warm_start)",
                M::NAME
            ));
        } else {
            let (candidate, effective, mut note) = match warm.mode {
                WarmStartMode::Greedy => (&initial_result, WarmStartMode::Greedy, None),
                WarmStartMode::Initial => match &warm.seed {
                    Some(s) => (s, WarmStartMode::Initial, None),
                    None => (
                        &initial_result,
                        WarmStartMode::Greedy,
                        Some(
                            "no initial extraction was supplied; degraded to the greedy seed"
                                .to_string(),
                        ),
                    ),
                },
                WarmStartMode::None => unreachable!(),
            };
            match seed_simplified(&vars, &roots, candidate, &initial_result) {
                Ok((assignment, repaired)) => {
                    // Report the cost of the seed AS APPLIED (post-repair), not
                    // of the raw candidate: with repairs those are different
                    // numbers, and the one that explains the solver's starting
                    // incumbent is the applied one. `result` already holds the
                    // classes `remove_single_zero_cost` decided up front, which
                    // are no longer in `vars`.
                    let mut applied = result.clone();
                    for (cid, &idx) in &assignment {
                        applied.choose(cid.clone(), vars[cid].members[idx].clone());
                    }
                    report.warm_start_objective =
                        super::warm::try_dag_cost(egraph, &roots, &applied);
                    apply_seed(&vars, &mut model, &assignment);
                    if repaired > 0 {
                        let extra = format!(
                            "{repaired} class(es) repaired off the preferred node \
                             (removed by simplification or re-classed)"
                        );
                        note = Some(match note {
                            Some(n) => format!("{n}; {extra}"),
                            None => extra,
                        });
                    }
                    report.accept_warm_start(effective, note);
                }
                Err(why) => report.refuse_warm_start(why),
            }
        }
    }

    log::info!(
        "Time spent before solving: {}ms",
        simp_start_time.elapsed().as_millis()
    );

    let start_time = SystemTime::now();
    let solve_clock = Instant::now();

    loop {
        // Set the solver limit based on how long has passed already.
        if let Ok(difference) = SystemTime::now().duration_since(start_time) {
            let seconds = timeout.saturating_sub(difference.as_secs().try_into().unwrap());
            model.set_time_limit_seconds(seconds);
        } else {
            model.set_time_limit_seconds(0);
        }

        //This starts from scratch solving each time. I've looked quickly
        //at the API and didn't see how to call it incrementally.
        let solution = model.solve();
        report.num_solves += 1;
        report.solve_wall_secs = solve_clock.elapsed().as_secs_f64();
        report.record_solution(&solution);
        log::info!(
            "{} status {}, obj = {}",
            M::NAME,
            solution.status_detail(),
            solution.obj_value(),
        );

        if solution.is_infeasible() {
            log::info!("Infeasible, returning empty solution");
            report.load_trajectory();
            return (ExtractionResult::default(), report);
        }

        let stopped_without_finishing = !solution.ran_to_completion();

        if stopped_without_finishing {
            log::info!("{} stopped before finishing", M::NAME);

            if !config.return_improved_on_timeout
                || solution.obj_value() > initial_result_cost.into_inner()
            {
                log::info!(
                    "Unfinished {} solution returned, solver: {}, initial: {}",
                    M::NAME,
                    solution.obj_value(),
                    initial_result_cost
                );
                report.returned_fallback = true;
                report.load_trajectory();
                return (initial_result, report);
            }
        }

        let mut cost = 0.0;
        for (id, var) in &vars {
            let active = solution.col(var.active) > 0.0;

            if active {
                assert!(var.members() > 0);
                let mut node_idx = 0;
                if var.members() != 1 {
                    assert_eq!(
                        1,
                        var.variables
                            .iter()
                            .filter(|&n| solution.col(*n) > 0.0)
                            .count()
                    );

                    node_idx = var
                        .variables
                        .iter()
                        .position(|&n| solution.col(n) > 0.0)
                        .unwrap();
                }

                let node_id = var.members[node_idx].clone();
                cost += var.costs[node_idx].into_inner();
                result.choose(id.clone(), node_id);
            }
        }

        let cycles = find_cycles_in_result(&result, &vars, &roots);

        log::info!("Cost of solution {cost}");
        log::info!("Initial result {}", initial_result_cost.into_inner());
        log::info!("Cost of extraction {}", result.dag_cost(egraph, &roots));
        log::info!("Cost from solver {}", solution.obj_value());

        if stopped_without_finishing {
            log::info!("Timed out");
            report.load_trajectory();
            if cycles.is_empty() {
                // The reported cost of the solution sometimes differs to the dag cost, so we're
                // a bit carefu..
                let extraction_dag_cost = result.dag_cost(egraph, &roots);

                // Not sure if this will ever fail..
                result.check(egraph);
                if extraction_dag_cost < initial_result_cost {
                    log::info!(
                        "Returning result of incomplete search saving: {}",
                        initial_result_cost - extraction_dag_cost
                    );
                    return (result, report);
                } else {
                    report.returned_fallback = true;
                    return (initial_result, report);
                }
            } else {
                log::info!("Found cycle in solution, but solver timed out");
                report.returned_fallback = true;
                return (initial_result, report);
            }
        }

        if cycles.is_empty() {
            assert!(cost <= initial_result_cost.into_inner() + EPSILON_ALLOWANCE);
            assert!((result.dag_cost(egraph, &roots) - cost).abs() < EPSILON_ALLOWANCE);
            assert!((cost - solution.obj_value()).abs() < EPSILON_ALLOWANCE);

            report.load_trajectory();
            return (result, report);
        } else {
            log::info!("Refining by blocking cycles: {}", cycles.len());
            for c in &cycles {
                block_cycle(&mut model, c, &vars);
            }
        }

        // NOTE: no re-seed here. `MilpModel::set_initial_solution` takes a
        // solution of the *previous* model, and `block_cycle` has just added
        // columns; CBC's version segfaults on that mismatch, and the seed would
        // in any case be the solution the solver already has. The MIP start set
        // before the first solve is the one that matters.
    }
}

/// Build a complete, model-consistent seed against the **simplified** `vars`.
///
/// This is the piece the original code was missing. `vars` is not the e-graph:
/// nodes have been deleted and children sets rewritten (`pull_up_with_single_
/// parent` *adds* a class's descendants to its parent's child set). So a seed
/// has to be constructed by walking `vars` itself — committing to a node, then
/// descending through *that node's* `childrens_classes` — which makes it
/// satisfy the "node active implies child active" and the intersection rows by
/// construction. Seeding from the e-graph's own structure instead, as the old
/// code did, produces a point that violates rows the solver is about to add.
///
/// Returns the per-class `(active, chosen index)` assignment and how many
/// classes had to be repaired off the preferred choice. `Err` means "do not
/// warm start": every error case here would be an infeasible point.
type SeedAssignment = IndexMap<ClassId, usize>;

fn seed_simplified<C: Copy>(
    vars: &IndexMap<ClassId, ClassILP<C>>,
    roots: &[ClassId],
    preferred: &ExtractionResult,
    fallback: &ExtractionResult,
) -> Result<(SeedAssignment, usize), String> {
    let mut chosen: SeedAssignment = IndexMap::default();
    let mut repaired = 0usize;
    let mut todo: Vec<ClassId> = roots.to_vec();
    // `Doing`/`Done` colouring, to reject a cyclic seed rather than loop.
    let mut done: FxHashSet<ClassId> = Default::default();

    while let Some(cid) = todo.pop() {
        if !done.insert(cid.clone()) {
            continue;
        }
        let class = vars
            .get(&cid)
            .ok_or_else(|| format!("seed reached class {cid}, which simplification removed"))?;
        if class.members() == 0 {
            return Err(format!(
                "seed reached class {cid}, which has no remaining members (it is pinned inactive)"
            ));
        }
        // Prefer the seed's node, then the fallback's, then the cheapest
        // survivor. Every one of those is feasible for a pure-area model.
        let idx = preferred
            .choices
            .get(&cid)
            .and_then(|nid| class.members.iter().position(|m| m == nid))
            .or_else(|| {
                repaired += 1;
                fallback
                    .choices
                    .get(&cid)
                    .and_then(|nid| class.members.iter().position(|m| m == nid))
            })
            .or_else(|| {
                class
                    .costs
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.cmp(b.1))
                    .map(|(i, _)| i)
            })
            .ok_or_else(|| format!("no seedable member for class {cid}"))?;

        for child in &class.childrens_classes[idx] {
            todo.push(child.clone());
        }
        chosen.insert(cid, idx);
    }

    // Acyclicity, checked over the simplified children sets the model uses.
    if seed_has_cycle(vars, roots, &chosen) {
        return Err("candidate seed selection contains a cycle".to_string());
    }
    Ok((chosen, repaired))
}

fn seed_has_cycle<C: Copy>(
    vars: &IndexMap<ClassId, ClassILP<C>>,
    roots: &[ClassId],
    chosen: &SeedAssignment,
) -> bool {
    fn dfs<C: Copy>(
        vars: &IndexMap<ClassId, ClassILP<C>>,
        chosen: &SeedAssignment,
        cid: &ClassId,
        state: &mut IndexMap<ClassId, u8>,
    ) -> bool {
        match state.get(cid) {
            Some(2) => return false,
            Some(_) => return true,
            None => {}
        }
        state.insert(cid.clone(), 1);
        if let (Some(class), Some(&idx)) = (vars.get(cid), chosen.get(cid)) {
            for child in &class.childrens_classes[idx] {
                if dfs(vars, chosen, child, state) {
                    return true;
                }
            }
        }
        state.insert(cid.clone(), 2);
        false
    }
    let mut state: IndexMap<ClassId, u8> = IndexMap::default();
    roots.iter().any(|r| dfs(vars, chosen, r, &mut state))
}

/// Push a validated [`SeedAssignment`] into the model as a MIP start. Every
/// column of the selection model gets an explicit value (including the zeros),
/// so no backend has to guess at an unspecified variable.
fn apply_seed<M: MilpModel>(
    vars: &IndexMap<ClassId, ClassILP<M::Col>>,
    model: &mut M,
    chosen: &SeedAssignment,
) {
    for (class_id, class_vars) in vars {
        let active = chosen.get(class_id).copied();
        model.set_col_initial_solution(
            class_vars.active,
            if active.is_some() { 1.0 } else { 0.0 },
        );
        for (i, col) in class_vars.variables.iter().enumerate() {
            model.set_col_initial_solution(*col, if active == Some(i) { 1.0 } else { 0.0 });
        }
    }
}

/* If a class has one node, and that node is zero cost, and it has no children, then we
can fill the answer into the extraction result without doing any more work. If it
has children, we need to setup the dependencies.

Intuitively, whenever we find a class that has a single node that is zero cost, our work
is done, we can't do any better for that class, so we can select it. Additionally, we
don't care if any other node depends on this class, because this class is zero cost,
we can ignore all references to it.

This is really like deleting empty classes, except there we delete the parent classes,
and here we delete just children of nodes in the parent classes.

*/
fn remove_single_zero_cost<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    extraction_result: &mut ExtractionResult,
    roots: &[ClassId],
    config: &Config,
) {
    if config.remove_single_zero_cost {
        let mut zero: FxHashSet<ClassId> = Default::default();
        for (class_id, details) in &*vars {
            if details.childrens_classes.len() == 1
                && details.childrens_classes[0].is_empty()
                && details.costs[0] == 0.0
                && !roots.contains(&class_id.clone())
            {
                zero.insert(class_id.clone());
            }
        }

        if zero.is_empty() {
            return;
        }

        let mut removed = 0;
        let mut extras = 0;
        let fresh = IndexSet::<ClassId>::new();
        let child_to_parents = child_to_parents(&vars);

        // Remove all references to those in zero.
        for e in &zero {
            let parents = child_to_parents.get(e).unwrap_or(&fresh);
            for parent in parents {
                for i in (0..vars[parent].childrens_classes.len()).rev() {
                    if vars[parent].childrens_classes[i].contains(e) {
                        vars[parent].childrens_classes[i].remove(e);
                        removed += 1;
                    }
                }

                // Like with empty classes, we might have discovered a new candidate class.
                // It's rare in our benchmarks so I haven't implemented it yet.
                if vars[parent].childrens_classes.len() == 1
                    && vars[parent].childrens_classes[0].is_empty()
                    && vars[parent].costs[0] == 0.0
                    && !roots.contains(&e.clone())
                {
                    extras += 1;
                    // this should be called in a loop like we delete empty classes.
                }
            }
        }
        // Add into the extraction result
        for e in &zero {
            extraction_result.choose(e.clone(), vars[e].members[0].clone());
        }

        // Remove the classes themselves.
        vars.retain(|class_id, _| !zero.contains(class_id));

        log::info!(
            "Zero cost & zero children removed: {} links removed: {removed}, extras:{extras}",
            zero.len()
        );
    }
}

fn child_to_parents<C: Copy>(
    vars: &IndexMap<ClassId, ClassILP<C>>,
) -> IndexMap<ClassId, IndexSet<ClassId>> {
    let mut child_to_parents: IndexMap<ClassId, IndexSet<ClassId>> = IndexMap::new();

    for (class_id, class_vars) in vars.iter() {
        for kids in &class_vars.childrens_classes {
            for child_class in kids {
                child_to_parents
                    .entry(child_class.clone())
                    .or_insert_with(IndexSet::new)
                    .insert(class_id.clone());
            }
        }
    }
    child_to_parents
}

/* If a node in a class has (a) equal or higher cost compared to another in that same class, and (b) its
  children are a superset of the other's, then it can be removed.
*/
fn remove_more_expensive_subsumed_nodes<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    config: &Config,
) {
    if config.remove_more_expensive_subsumed_nodes {
        let mut removed = 0;

        for class in vars.values_mut() {
            let mut children = class.as_nodes();
            children.sort_by_key(|e| (e.children_classes.len(), e.cost));

            let mut i = 0;
            while i < children.len() {
                for j in ((i + 1)..children.len()).rev() {
                    let node_b = &children[j];

                    // This removes some extractions with the same cost.
                    if children[i].cost <= node_b.cost
                        && children[i]
                            .children_classes
                            .is_subset(&node_b.children_classes)
                    {
                        class.remove_node(&node_b.member.clone());
                        children.remove(j);
                        removed += 1;
                    }
                }
                i += 1;
            }
        }

        log::info!("Removed more expensive subsumed nodes: {removed}");
    }
}

// Remove any classes that can't be reached from a root.
fn remove_unreachable_classes<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    roots: &[ClassId],
    config: &Config,
) {
    if config.remove_unreachable_classes {
        let mut reachable_classes: IndexSet<ClassId> = IndexSet::default();
        reachable(&*vars, roots, &mut reachable_classes);
        let initial_size = vars.len();
        vars.retain(|class_id, _| reachable_classes.contains(class_id));
        log::info!("Unreachable classes: {}", initial_size - vars.len());
    }
}

// Any node that has an empty class as a child, can't be selected, so remove the node,
// if that makes another empty class, then remove its parents
fn remove_empty_classes<C: Copy>(vars: &mut IndexMap<ClassId, ClassILP<C>>, config: &Config) {
    if config.remove_empty_classes {
        let mut empty_classes: std::collections::VecDeque<ClassId> = Default::default();
        for (classid, detail) in vars.iter() {
            if detail.members() == 0 {
                empty_classes.push_back(classid.clone());
            }
        }

        let mut removed = 0;
        let fresh = IndexSet::<ClassId>::new();

        let mut child_to_parents: IndexMap<ClassId, IndexSet<ClassId>> = IndexMap::new();

        for (class_id, class_vars) in vars.iter() {
            for kids in &class_vars.childrens_classes {
                for child_class in kids {
                    child_to_parents
                        .entry(child_class.clone())
                        .or_insert_with(IndexSet::new)
                        .insert(class_id.clone());
                }
            }
        }

        let mut done = FxHashSet::<ClassId>::default();

        while let Some(e) = empty_classes.pop_front() {
            if !done.insert(e.clone()) {
                continue;
            }
            let parents = child_to_parents.get(&e).unwrap_or(&fresh);
            for parent in parents {
                for i in (0..vars[parent].childrens_classes.len()).rev() {
                    if vars[parent].childrens_classes[i].contains(&e) {
                        vars[parent].remove(i);
                        removed += 1;
                    }
                }

                if vars[parent].members() == 0 {
                    empty_classes.push_back(parent.clone());
                }
            }
        }

        log::info!("Nodes removed that point to empty classes: {}", removed);
    }
}

// Any class that is a child of each node in a root, is also a root.
fn find_extra_roots<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    roots: &mut Vec<ClassId>,
    config: &Config,
) {
    if config.find_extra_roots {
        let mut extra = 0;
        let mut i = 0;
        // newly added roots will also be processed in one pass through.
        while i < roots.len() {
            let r = roots[i].clone();

            let details = vars.get(&r).unwrap();
            if details.childrens_classes.len() == 0 {
                continue;
            }

            let mut intersection = details.childrens_classes[0].clone();

            for childrens_classes in &details.childrens_classes[1..] {
                intersection = intersection
                    .intersection(childrens_classes)
                    .cloned()
                    .collect();
            }

            for r in &intersection {
                if !roots.contains(r) {
                    roots.push(r.clone());
                    extra += 1;
                }
            }
            i += 1;
        }

        log::info!("Extra roots discovered: {extra}");
    }
}

/*
For each class with one parent, move the minimum costs of the members to each node in the parent that points to it.

if we iterated through these in order, from child to parent, to parent, to parent.. it could be done in one pass.
*/
fn pull_up_costs<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    roots: &[ClassId],
    config: &Config,
) {
    if config.pull_up_costs {
        let mut count = 0;
        let mut changed = true;
        let child_to_parent = classes_with_single_parent(&*vars);

        while (count < 10) && changed {
            log::info!("Classes with a single parent: {}", child_to_parent.len());
            changed = false;
            count += 1;
            for (child, parent) in &child_to_parent {
                if child == parent {
                    continue;
                }
                if roots.contains(child) {
                    continue;
                }
                if vars[child].members() == 0 {
                    continue;
                }

                // Get the minimum cost of members of the children
                let min_cost = vars[child]
                    .costs
                    .iter()
                    .min()
                    .unwrap_or(&Cost::default())
                    .into_inner();

                assert!(min_cost >= 0.0);
                if min_cost == 0.0 {
                    continue;
                }
                changed = true;

                // Now remove it from each member
                for c in &mut vars[child].costs {
                    *c -= min_cost;
                    assert!(c.into_inner() >= 0.0);
                }
                // Add it onto each node in the parent that refers to this class.
                let indices: Vec<_> = vars[parent]
                    .childrens_classes
                    .iter()
                    .enumerate()
                    .filter(|&(_, c)| c.contains(child))
                    .map(|(id, _)| id)
                    .collect();

                assert!(!indices.is_empty());

                for id in indices {
                    vars[parent].costs[id] += min_cost;
                }
            }
        }
    }
}

/* If a class has a single parent class,
then move the children from the child to the parent class.

There could be a long chain of single parent classes - which this handles
(badly) by looping through a few times.

*/

fn pull_up_with_single_parent<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    roots: &[ClassId],
    config: &Config,
) {
    if config.pull_up_single_parent {
        for _i in 0..10 {
            let child_to_parent = classes_with_single_parent(&*vars);
            log::info!("Classes with a single parent: {}", child_to_parent.len());

            let mut pull_up_count = 0;
            for (child, parent) in &child_to_parent {
                if child == parent {
                    continue;
                }

                if roots.contains(child) {
                    continue;
                }

                if vars[child].members.len() != 1 {
                    continue;
                }

                if vars[child].childrens_classes.first().unwrap().is_empty() {
                    continue;
                }

                let found = vars[parent]
                    .childrens_classes
                    .iter()
                    .filter(|c| c.contains(child))
                    .count();

                if found != 1 {
                    continue;
                }

                let idx = vars[parent]
                    .childrens_classes
                    .iter()
                    .position(|e| e.contains(child))
                    .unwrap();

                let child_descendants = vars
                    .get(child)
                    .unwrap()
                    .childrens_classes
                    .first()
                    .unwrap()
                    .clone();

                let parent_descendants: &mut IndexSet<ClassId> = vars
                    .get_mut(parent)
                    .unwrap()
                    .childrens_classes
                    .get_mut(idx)
                    .unwrap();

                for e in &child_descendants {
                    parent_descendants.insert(e.clone());
                }

                vars.get_mut(child)
                    .unwrap()
                    .childrens_classes
                    .first_mut()
                    .unwrap()
                    .clear();

                pull_up_count += 1;
            }
            log::info!("Pull up count: {pull_up_count}");
            if pull_up_count == 0 {
                break;
            }
        }
    }
}

// Remove any nodes that alone cost more than the total of a solution.
// For example, if the lowest the sum of roots can be is 12, and we've found an approximate
// solution already that is 15, then any non-root node that costs more than 3 can't be selected
// in the optimal solution.

fn remove_high_cost<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    initial_result_cost: NotNan<f64>,
    roots: &[ClassId],
    config: &Config,
) {
    if config.remove_high_cost_nodes {
        debug_assert_eq!(
            roots.len(),
            roots.iter().collect::<std::collections::HashSet<_>>().len(),
            "All ClassId in roots must be unique"
        );

        let lowest_root_cost_sum: Cost = roots
            .iter()
            .filter_map(|root| vars[root].costs.iter().min())
            .sum();

        let mut removed = 0;

        for (class_id, class_details) in vars.iter_mut() {
            for i in (0..class_details.costs.len()).rev() {
                let cost = &class_details.costs[i];
                let this_root: Cost = if roots.contains(class_id) {
                    *class_details.costs.iter().min().unwrap()
                } else {
                    Cost::default()
                };

                if cost
                    > &(initial_result_cost - lowest_root_cost_sum + this_root + EPSILON_ALLOWANCE)
                {
                    class_details.remove(i);
                    removed += 1;
                }
            }
        }
        log::info!("Removed high-cost nodes: {}", removed);
    }
}

// Remove nodes with any (a) child pointing back to its own class,
// or (b) any child pointing to the sole root class.
fn remove_with_loops<C: Copy>(
    vars: &mut IndexMap<ClassId, ClassILP<C>>,
    roots: &[ClassId],
    config: &Config,
) {
    if config.remove_self_loops {
        let mut removed = 0;
        for (class_id, class_details) in vars.iter_mut() {
            for i in (0..class_details.childrens_classes.len()).rev() {
                if class_details.childrens_classes[i]
                    .iter()
                    .any(|cid| *cid == *class_id || (roots.len() == 1 && roots[0] == *cid))
                {
                    class_details.remove(i);
                    removed += 1;
                }
            }
        }

        log::info!("Omitted looping nodes: {}", removed);
    }
}

// Mapping from child class to parent classes
fn classes_with_single_parent<C: Copy>(
    vars: &IndexMap<ClassId, ClassILP<C>>,
) -> IndexMap<ClassId, ClassId> {
    let mut child_to_parents: IndexMap<ClassId, IndexSet<ClassId>> = IndexMap::new();

    for (class_id, class_vars) in vars.iter() {
        for kids in &class_vars.childrens_classes {
            for child_class in kids {
                child_to_parents
                    .entry(child_class.clone())
                    .or_insert_with(IndexSet::new)
                    .insert(class_id.clone());
            }
        }
    }

    // return classes with only one parent
    child_to_parents
        .into_iter()
        .filter_map(|(child_class, parents)| {
            if parents.len() == 1 {
                Some((child_class, parents.into_iter().next().unwrap()))
            } else {
                None
            }
        })
        .collect()
}

//Set of classes that can be reached from the [classes]
fn reachable<C: Copy>(
    vars: &IndexMap<ClassId, ClassILP<C>>,
    classes: &[ClassId],
    is_reachable: &mut IndexSet<ClassId>,
) {
    for class in classes {
        if is_reachable.insert(class.clone()) {
            let class_vars = vars.get(class).unwrap();
            for kids in &class_vars.childrens_classes {
                for child_class in kids {
                    reachable(vars, &[child_class.clone()], is_reachable);
                }
            }
        }
    }
}

// Adds constraints to stop the cycle.
fn block_cycle<M: MilpModel>(
    model: &mut M,
    cycle: &Vec<ClassId>,
    vars: &IndexMap<ClassId, ClassILP<M::Col>>,
) {
    if cycle.is_empty() {
        return;
    }
    let mut blocking = Vec::new();
    for i in 0..cycle.len() {
        let current_class_id = &cycle[i];
        let next_class_id = &cycle[(i + 1) % cycle.len()];

        let mut this_level = Vec::default();
        for node in &vars[current_class_id].as_nodes() {
            if node.children_classes.contains(next_class_id) {
                this_level.push(node.variable);
            }
        }

        assert!(!this_level.is_empty());

        if this_level.len() == 1 {
            blocking.push(this_level[0]);
        } else {
            let blocking_var = model.add_binary();
            blocking.push(blocking_var);
            for n in this_level {
                let row = model.add_row();
                model.set_row_upper(row, 0.0);
                model.set_weight(row, n, 1.0);
                model.set_weight(row, blocking_var, -1.0);
            }
        }
    }

    //One of the edges between nodes in the cycle shouldn't be activated:
    let row = model.add_row();
    model.set_row_upper(row, blocking.len() as f64 - 1.0);
    for b in blocking {
        model.set_weight(row, b, 1.0)
    }
}

#[derive(Clone)]
enum TraverseStatus {
    Doing,
    Done,
}

/*
Returns the simple cycles possible from the roots.

Because the number of simple cycles can be factorial in the number
of nodes, this can be very slow.

Imagine a 20 node complete graph with one root. From the first node you have
19 choices, then from the second 18 choices, etc.  When you get to the second
last node you go back to the root. There are about 10^17 length 18 cycles.

So we limit how many can be found.
*/
const CYCLE_LIMIT: usize = 1000;

fn find_cycles_in_result<C: Copy>(
    extraction_result: &ExtractionResult,
    vars: &IndexMap<ClassId, ClassILP<C>>,
    roots: &[ClassId],
) -> Vec<Vec<ClassId>> {
    let mut status = IndexMap::<ClassId, TraverseStatus>::default();
    let mut cycles = vec![];
    for root in roots {
        let mut stack = vec![];
        cycle_dfs(
            extraction_result,
            vars,
            root,
            &mut status,
            &mut cycles,
            &mut stack,
        )
    }
    cycles
}

fn cycle_dfs<C: Copy>(
    extraction_result: &ExtractionResult,
    vars: &IndexMap<ClassId, ClassILP<C>>,
    class_id: &ClassId,
    status: &mut IndexMap<ClassId, TraverseStatus>,
    cycles: &mut Vec<Vec<ClassId>>,
    stack: &mut Vec<ClassId>,
) {
    match status.get(class_id).cloned() {
        Some(TraverseStatus::Done) => (),
        Some(TraverseStatus::Doing) => {
            // Get the part of the stack between the first visit to the class and now.
            let mut cycle = vec![];
            if let Some(pos) = stack.iter().position(|id| id == class_id) {
                cycle.extend_from_slice(&stack[pos..]);
            }
            cycles.push(cycle);
        }
        None => {
            if cycles.len() > CYCLE_LIMIT {
                return;
            }
            status.insert(class_id.clone(), TraverseStatus::Doing);
            stack.push(class_id.clone());
            let node_id = &extraction_result.choices[class_id];
            for child_cid in vars[class_id].get_children_of_node(node_id) {
                cycle_dfs(extraction_result, vars, child_cid, status, cycles, stack)
            }
            let last = stack.pop();
            assert_eq!(*class_id, last.unwrap());
            status.insert(class_id.clone(), TraverseStatus::Done);
        }
    }
}

#[cfg(test)]
mod test {
    use super::Config;
    use crate::test::{generate_random_egraph, ELABORATE_TESTING};

    use crate::milp::DefaultMilp;
    use crate::{faster_ilp_cbc::extract, EPSILON_ALLOWANCE};
    use rand::Rng;
    pub type Cost = ordered_float::NotNan<f64>;

    pub fn generate_random_config() -> Config {
        let mut rng = rand::thread_rng();
        Config {
            pull_up_costs: rng.gen(),
            remove_self_loops: rng.gen(),
            remove_high_cost_nodes: rng.gen(),
            remove_more_expensive_subsumed_nodes: rng.gen(),
            remove_unreachable_classes: rng.gen(),
            pull_up_single_parent: rng.gen(),
            take_intersection_of_children_in_class: rng.gen(),
            move_min_cost_of_members_to_class: rng.gen(),
            find_extra_roots: rng.gen(),
            remove_empty_classes: rng.gen(),
            return_improved_on_timeout: rng.gen(),
            remove_single_zero_cost: rng.gen(),
        }
    }

    fn all_disabled() -> Config {
        return Config {
            pull_up_costs: false,
            remove_self_loops: false,
            remove_high_cost_nodes: false,
            remove_more_expensive_subsumed_nodes: false,
            remove_unreachable_classes: false,
            pull_up_single_parent: false,
            take_intersection_of_children_in_class: false,
            move_min_cost_of_members_to_class: false,
            find_extra_roots: false,
            remove_empty_classes: false,
            return_improved_on_timeout: false,
            remove_single_zero_cost: false,
        };
    }

    const CONFIGS_TO_TEST: i64 = 150;

    fn test_configs(config: &Vec<Config>, log_path: impl AsRef<std::path::Path>) {
        const RANDOM_EGRAPHS_TO_TEST: i64 = if ELABORATE_TESTING {
            1000000 / CONFIGS_TO_TEST
        } else {
            250 / CONFIGS_TO_TEST
        };

        for _ in 0..RANDOM_EGRAPHS_TO_TEST {
            let egraph = generate_random_egraph();

            if !log_path.as_ref().to_str().unwrap_or("").is_empty() {
                egraph.to_json_file(&log_path).unwrap();
            }

            let mut results: Option<Cost> = None;
            for c in config {
                let (extraction, _report) = extract::<DefaultMilp>(
                    &egraph,
                    &egraph.root_eclasses,
                    c,
                    u32::MAX,
                    1,
                    &super::WarmConfig::default(),
                );
                extraction.check(&egraph);
                let dag_cost = extraction.dag_cost(&egraph, &egraph.root_eclasses);
                if results.is_some() {
                    assert!(
                        (dag_cost.into_inner() - results.unwrap().into_inner()).abs()
                            < EPSILON_ALLOWANCE
                    );
                }
                results = Some(dag_cost);
            }
        }
    }

    macro_rules! create_tests {
    ($($name:ident),*) => {
        $(
            #[test]
            fn $name() {
                let mut configs = vec![Config::default(), all_disabled()];

                for _ in 0..CONFIGS_TO_TEST {
                    configs.push(generate_random_config());
                }
                test_configs(&configs, crate::test::test_save_path(stringify!($name)));
            }
        )*
    }
}

    // So the test runner uses more of my cores.
    create_tests!(
        random0, random1, random2, random3, random4, random5, random6, random7, random8, random9,
        random10
    );
}
