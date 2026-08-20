/*!
Warm starts for the ILP extractors, and the machine-readable report of what the
solver actually did.

# Why the warm-start code was disabled, and what changed

Both `set_initial_solution` call sites in [`super::faster_ilp_cbc`] were
`if false`-gated with the comment "using this causes the ILP solver to return
unsound results", and [`super::ilp_cbc`] had no warm-start code at all. The
working diagnosis — which the evidence in this repo supports — is that the
unsoundness came from **seeding an infeasible point**, not from warm starting as
such:

* The old seeding loop walked the *simplified* `vars` map and set
  `class_active = 1` for every class the greedy extraction had chosen. But the
  simplifier deletes nodes (`remove_high_cost`, `remove_more_expensive_subsumed_
  nodes`, ...) and *rewrites* children sets (`pull_up_with_single_parent` adds a
  child's descendants to its parent). So the greedy choice for a class was
  routinely a node that no longer had a column, leaving `class_active = 1` with
  no member node active — a direct violation of the `sum(nodes) == active` row.
* CBC additionally fills every unseeded column with `0.0`, which cannot be right
  for the continuous level / arrival-time columns in `ilp_cbc`.
* The registry's `delay-budget-ilp-cbc-timeout` hardcodes `max_delay: 100.0`,
  an arbitrary budget that is infeasible on most of the test corpus — and that
  is the confirmed cause of the eight long-standing `test::check0..check7`
  failures. "Arbitrary budget -> infeasible -> breakage" is already demonstrated
  here.

The fix is therefore not "warm start carefully" but **construct the seed against
the model the solver will actually see, then verify it before handing it over**:

1. [`build_seed`] walks the selection structure the model was built from,
   choosing the preferred node where it still exists and repairing to a fallback
   where it does not, so the result is complete and closed by construction.
2. [`arrival_times`] evaluates the candidate's critical-path delay using the
   *ILP's own* recurrence (`T_c = d_n + max_cc T_cc`), which is what the
   delay-budget extractor's rows enforce — so a budget check here is a check in
   the solver's arithmetic, not in a parallel implementation of it.
3. If any check fails, the run continues with **no warm start** and records why
   in [`SolveReport::warm_start_note`]. Never a hard failure, and never an
   infeasible point handed to the solver.

Two failure modes this must catch, and does:

* **Greedy seeds are not delay-preserving.** Greedy minimises area, so its
  critical path can exceed a budget derived from the input design.
* **The budget can be tight rather than slack.** A design-derived budget equals
  the input's own delay, putting the initial seed exactly on the constraint
  boundary, where a float discrepancy of one ulp is the difference between
  feasible and not.
*/

use indexmap::{IndexMap, IndexSet};

use crate::milp::trajectory::Incumbent;
use crate::milp::MilpStatus;
use crate::{ClassId, EGraph, ExtractionResult, NodeId};

/// Which starting incumbent to hand the solver.
///
/// The three values are what a caller selects between when choosing a seed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WarmStartMode {
    /// No MIP start. The historical behaviour and still the default.
    #[default]
    None,
    /// Seed from the caller-supplied *initial* extraction — typically the
    /// input design's own mapping. This is the seed that is feasible by
    /// construction under a design-derived delay budget, because that budget
    /// *is* the input design's critical-path delay.
    ///
    /// If the caller supplies no seed, [`super::ilp_cbc`] refuses the warm
    /// start with a note and solves with no MIP start (it used to degrade to
    /// [`Self::Greedy`]; see that variant).
    Initial,
    /// Seed from a greedy extraction of the same e-graph.
    ///
    /// **Refused by [`super::ilp_cbc`]'s extractors.** Greedy is unbounded in
    /// how much worse than the caller's input mapping it can be, and it was
    /// also the acceptance bar, so a run could ship a result larger than its
    /// own input. Seed selection now happens entirely caller-side. Still
    /// honoured by [`super::faster_ilp_cbc`], which is no longer on
    /// the default extraction path.
    Greedy,
}

impl WarmStartMode {
    pub fn as_str(self) -> &'static str {
        match self {
            WarmStartMode::None => "none",
            WarmStartMode::Initial => "initial",
            WarmStartMode::Greedy => "greedy",
        }
    }
}

impl std::fmt::Display for WarmStartMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for WarmStartMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(WarmStartMode::None),
            "initial" => Ok(WarmStartMode::Initial),
            "greedy" => Ok(WarmStartMode::Greedy),
            other => Err(format!(
                "unknown warm-start mode {other:?} (expected none|initial|greedy)"
            )),
        }
    }
}

/// Everything a caller needs to tell one ILP extraction run apart from another
/// in the benchmark matrix, plus enough solver state to say whether a *timed
/// out* run got closer than its neighbour.
///
/// Final area alone cannot answer that question on instances where every
/// configuration times out, which is why the objective, the bound, the gap and
/// the time-to-first-incumbent are all here.
#[derive(Debug, Clone)]
pub struct SolveReport {
    /// `MilpModel::NAME` of the backend that was compiled in.
    pub backend: &'static str,
    /// Which model this report describes (`"area-ilp"`, `"delay-budget"`, ...).
    pub model: &'static str,
    /// Threads actually requested of the solver.
    pub threads: u32,
    /// Wall-clock limit handed to the solver, in seconds.
    pub timeout_seconds: u32,
    /// The mode the caller asked for.
    pub warm_start_requested: WarmStartMode,
    /// The mode that was actually applied. Differs from the request when the
    /// seed failed validation, when the backend does not support MIP starts, or
    /// when `Initial` degraded to `Greedy` for want of a seed.
    pub warm_start_applied: WarmStartMode,
    /// Human-readable reason whenever `applied != requested`, and any repair
    /// statistics when they are equal.
    pub warm_start_note: Option<String>,
    /// DAG cost of the seed that was actually applied. Lets a reader see how
    /// much of the final objective the warm start already accounted for.
    pub warm_start_objective: Option<f64>,
    /// DAG cost of the greedy fallback extraction, i.e. what is returned when
    /// the solver's answer is discarded. Reported unconditionally so a
    /// `returned = initial_fallback` row is still interpretable.
    pub initial_cost: Option<f64>,
    /// Solver-neutral outcome of the final solve.
    pub status: Option<MilpStatus>,
    /// Backend's own status text, for debugging.
    pub status_detail: Option<String>,
    /// Objective value of the final incumbent, if there was one.
    pub objective: Option<f64>,
    /// Best proven bound at the moment the solve stopped.
    pub best_bound: Option<f64>,
    /// `|obj - bound| / max(|obj|, 1e-10)`.
    pub gap: Option<f64>,
    /// Wall-clock time spent inside the solve loop (excludes model building).
    pub solve_wall_secs: f64,
    /// How many times `solve()` was called (the cycle-breaking loop re-solves).
    pub num_solves: usize,
    /// True when the extractor discarded the solver's answer and returned the
    /// greedy fallback instead — the objective below is then the *solver's*,
    /// not the returned extraction's.
    pub returned_fallback: bool,
    /// Path the solver log was written to, if `--milp-log` was given.
    pub milp_log: Option<String>,
    /// Incumbent-vs-time points recovered from that log.
    pub trajectory: Vec<Incumbent>,
    /// Seconds to the first feasible incumbent, from the trajectory.
    pub time_to_first_incumbent_secs: Option<f64>,
}

impl SolveReport {
    pub fn new(
        backend: &'static str,
        model: &'static str,
        threads: u32,
        timeout_seconds: u32,
        requested: WarmStartMode,
    ) -> Self {
        SolveReport {
            backend,
            model,
            threads,
            timeout_seconds,
            warm_start_requested: requested,
            warm_start_applied: WarmStartMode::None,
            warm_start_note: None,
            warm_start_objective: None,
            initial_cost: None,
            status: None,
            status_detail: None,
            objective: None,
            best_bound: None,
            gap: None,
            solve_wall_secs: 0.0,
            num_solves: 0,
            returned_fallback: false,
            milp_log: None,
            trajectory: Vec::new(),
            time_to_first_incumbent_secs: None,
        }
    }

    /// Record that the warm start was refused, and why. Logged at `warn` *and*
    /// on stderr: `env_logger`'s default filter hides warnings, and a silently
    /// dropped warm start would make a whole column of the benchmark matrix
    /// quietly identical to the `none` column.
    pub fn refuse_warm_start(&mut self, why: impl Into<String>) {
        let why = why.into();
        if self.warm_start_requested != WarmStartMode::None {
            let msg = format!(
                "warm start ({}) NOT applied for the {} model on {}: {why}; \
                 running without a MIP start",
                self.warm_start_requested, self.model, self.backend
            );
            log::warn!("{msg}");
            eprintln!("WARNING: {msg}");
        }
        self.warm_start_applied = WarmStartMode::None;
        self.warm_start_note = Some(why);
    }

    /// Record that the warm start was applied, optionally with a note (e.g.
    /// "repaired 3 classes", or "degraded initial -> greedy").
    pub fn accept_warm_start(&mut self, applied: WarmStartMode, note: Option<String>) {
        self.warm_start_applied = applied;
        if let Some(n) = &note {
            log::info!(
                "warm start ({applied}) applied for the {} model on {}: {n}",
                self.model,
                self.backend
            );
        }
        self.warm_start_note = note;
    }

    /// Fill in the solver-state fields from the final solution.
    pub fn record_solution<S: crate::milp::MilpSolution>(&mut self, sol: &S) {
        self.status = Some(sol.status());
        self.status_detail = Some(sol.status_detail());
        self.objective = sol.has_solution().then(|| sol.obj_value());
        self.best_bound = sol.best_bound();
        self.gap = sol.gap();
    }

    /// Parse the solver log (if any) into a trajectory and derive the
    /// time-to-first-incumbent from it.
    pub fn load_trajectory(&mut self) {
        let Some(path) = &self.milp_log else { return };
        let traj = crate::milp::trajectory::parse_log_for_backend(
            std::path::Path::new(path),
            self.backend,
        );
        self.time_to_first_incumbent_secs =
            crate::milp::trajectory::time_to_first_incumbent(&traj);
        self.trajectory = traj;
    }
}

/* -------------------------------------------------------------------------- */
/* Seed construction and validation                                           */
/* -------------------------------------------------------------------------- */

/// Outcome of [`build_seed`].
pub struct Seed {
    /// One chosen node per class reachable from the roots. Complete and
    /// acyclic by construction.
    pub selection: ExtractionResult,
    /// How many classes had to fall back off the preferred choice (because the
    /// preferred node was absent or belonged to a different class after
    /// congruence). Reported, not fatal.
    pub repaired: usize,
}

/// Build a **complete, acyclic** selection over `egraph` from a `preferred`
/// partial choice, repairing to `fallback` wherever `preferred` cannot be used.
///
/// The walk starts at `roots` and only ever descends through the children of a
/// node it has already committed to, so the result satisfies the ILP's
/// "node active implies child class active" rows by construction — which is
/// precisely the property the old, disabled seeding code did *not* have.
///
/// Returns `Err` if the walk hits a class neither map can fill, or if the
/// committed selection contains a cycle. Both are refusals, not panics.
pub fn build_seed(
    egraph: &EGraph,
    roots: &[ClassId],
    preferred: Option<&ExtractionResult>,
    fallback: &ExtractionResult,
) -> Result<Seed, String> {
    let mut selection = ExtractionResult::default();
    let mut repaired = 0usize;
    // Iterative DFS so a deep e-graph cannot blow the stack.
    let mut todo: Vec<ClassId> = roots.to_vec();
    let mut seen: IndexSet<ClassId> = IndexSet::default();

    while let Some(cid) = todo.pop() {
        if !seen.insert(cid.clone()) {
            continue;
        }
        let pick = pick_node(egraph, &cid, preferred, fallback, &mut repaired)?;
        for child in &egraph[&pick].children {
            todo.push(egraph.nid_to_cid(child).clone());
        }
        selection.choose(cid, pick);
    }

    if !selection.find_cycles(egraph, roots).is_empty() {
        return Err("candidate seed selection contains a cycle".to_string());
    }
    Ok(Seed {
        selection,
        repaired,
    })
}

fn pick_node(
    egraph: &EGraph,
    cid: &ClassId,
    preferred: Option<&ExtractionResult>,
    fallback: &ExtractionResult,
    repaired: &mut usize,
) -> Result<NodeId, String> {
    // A choice is usable only if the node really lives in this class: after
    // congruence closure two classes the seed treated separately may have
    // merged, and the stale choice then belongs to a sibling.
    let usable = |nid: &NodeId| egraph[nid].eclass == *cid;

    if let Some(p) = preferred {
        if let Some(nid) = p.choices.get(cid) {
            if usable(nid) {
                return Ok(nid.clone());
            }
        }
        *repaired += 1;
    }
    if let Some(nid) = fallback.choices.get(cid) {
        if usable(nid) {
            return Ok(nid.clone());
        }
    }
    // Last resort: the cheapest member. Keeps the seed complete when the
    // fallback extraction did not reach this class (it can happen when the
    // preferred choice steers the walk somewhere greedy never went).
    egraph[cid]
        .nodes
        .iter()
        .min_by(|a, b| egraph[*a].cost.cmp(&egraph[*b].cost))
        .cloned()
        .ok_or_else(|| format!("class {cid} has no nodes to seed with"))
}

/// [`ExtractionResult::dag_cost`], but returning `None` instead of panicking on
/// a selection that does not cover every class reachable from `roots`.
///
/// `dag_cost` indexes `choices[cid]` directly, so a partial selection is a
/// panic — unacceptable in *reporting* code, which is the only thing that ever
/// looks at a seed's cost. A missing number is fine; a crash while writing the
/// metadata sidecar would lose the whole run.
pub fn try_dag_cost(
    egraph: &EGraph,
    roots: &[ClassId],
    selection: &ExtractionResult,
) -> Option<f64> {
    let mut costs: IndexMap<ClassId, f64> = IndexMap::default();
    let mut todo: Vec<ClassId> = roots.to_vec();
    while let Some(cid) = todo.pop() {
        let nid = selection.choices.get(&cid)?;
        let node = &egraph[nid];
        if costs.insert(cid, node.cost.into_inner()).is_some() {
            continue;
        }
        for child in &node.children {
            todo.push(egraph.nid_to_cid(child).clone());
        }
    }
    Some(costs.values().sum())
}

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
        arrival_of(egraph, root, selection, &mut arrival, &mut IndexSet::default())?;
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
