use std::time::Duration;

use super::*;
use crate::extract::greedy::{GreedyDepthExtractor, GreedySizeExtractor};
use crate::milp::DefaultMilp;
use crate::test::generate_random_egraph;

const EGRAPHS: usize = 20;

fn options(time_limit: Option<Duration>, warm_start: WarmStart) -> IlpOptions {
    IlpOptions {
        time_limit,
        warm_start,
        ..Default::default()
    }
}

fn min_depth(egraph: &EGraph, roots: &[ClassId]) -> f64 {
    let e = GreedyDepthExtractor.extract(egraph, roots);
    e.dag_depth(egraph, roots).into_inner()
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-6 * a.abs().max(b.abs()).max(1.0)
}

/// Flag the nodes of `extraction` reachable from the roots as initial.
fn flag_initial(egraph: &mut EGraph, extraction: &ExtractionResult) {
    let roots = egraph.root_eclasses.clone();
    let mut todo = roots;
    let mut seen = std::collections::HashSet::new();
    while let Some(cid) = todo.pop() {
        if !seen.insert(cid.clone()) {
            continue;
        }
        let nid = extraction.choices[&cid].clone();
        todo.extend(
            egraph[&nid]
                .children
                .iter()
                .map(|c| egraph.nid_to_cid(c).clone()),
        );
        egraph.nodes.get_mut(&nid).unwrap().initial = true;
    }
}

#[test]
fn zero_time_limit_still_returns_a_complete_extraction() {
    for _ in 0..EGRAPHS {
        let egraph = generate_random_egraph();
        let roots = egraph.root_eclasses.clone();
        let objectives = [
            IlpObjective::Size,
            IlpObjective::Depth,
            IlpObjective::SizeConstrainedDepth {
                depth_budget: min_depth(&egraph, &roots),
            },
            IlpObjective::WeightedSizeDepth {
                size_weight: 1.0,
                depth_weight: 1.0,
            },
        ];
        for objective in objectives {
            for warm_start in [WarmStart::None, WarmStart::Greedy] {
                let opts = options(Some(Duration::ZERO), warm_start);
                let outcome = solve::<DefaultMilp>(&egraph, &roots, objective, &opts).unwrap();
                assert!(outcome.extraction.is_complete(&egraph, &roots));
                assert!(!outcome.report.depth_budget_violated);
            }
        }
    }
}

#[test]
fn infeasible_budget_returns_the_shallowest_candidate_flagged() {
    let mut tested = 0;
    while tested < EGRAPHS {
        let egraph = generate_random_egraph();
        let roots = egraph.root_eclasses.clone();
        let best_depth = min_depth(&egraph, &roots);
        if best_depth <= 0.0 {
            continue;
        }
        tested += 1;
        let objective = IlpObjective::SizeConstrainedDepth {
            depth_budget: best_depth / 2.0,
        };
        let opts = options(None, WarmStart::None);
        let outcome = solve::<DefaultMilp>(&egraph, &roots, objective, &opts).unwrap();
        assert!(matches!(
            outcome.report.outcome,
            SolveOutcome::Fallback { .. }
        ));
        assert!(outcome.report.depth_budget_violated);
        let depth = outcome.extraction.dag_depth(&egraph, &roots).into_inner();
        assert!(close(depth, best_depth));
    }
}

#[test]
fn attainable_budget_is_met_and_no_larger_than_the_depth_extraction() {
    for _ in 0..EGRAPHS {
        let egraph = generate_random_egraph();
        let roots = egraph.root_eclasses.clone();
        let depth_extraction = GreedyDepthExtractor.extract(&egraph, &roots);
        let budget = depth_extraction.dag_depth(&egraph, &roots).into_inner();
        let objective = IlpObjective::SizeConstrainedDepth {
            depth_budget: budget,
        };
        let opts = options(None, WarmStart::Greedy);
        let outcome = solve::<DefaultMilp>(&egraph, &roots, objective, &opts).unwrap();
        let e = &outcome.extraction;
        assert!(!outcome.report.depth_budget_violated);
        assert!(e.dag_depth(&egraph, &roots).into_inner() <= budget);
        assert!(e.dag_cost(&egraph, &roots) <= depth_extraction.dag_cost(&egraph, &roots));
    }
}

#[test]
fn optimal_solutions_agree_with_the_objective_and_greedy_bounds() {
    for _ in 0..EGRAPHS {
        let egraph = generate_random_egraph();
        let roots = egraph.root_eclasses.clone();
        let opts = options(None, WarmStart::None);

        // Size: no larger than greedy.
        let size = solve::<DefaultMilp>(&egraph, &roots, IlpObjective::Size, &opts).unwrap();
        assert_eq!(size.report.outcome, SolveOutcome::Optimal);
        let greedy_size = GreedySizeExtractor.extract(&egraph, &roots);
        assert!(
            size.extraction.dag_cost(&egraph, &roots)
                <= greedy_size.dag_cost(&egraph, &roots) + EPSILON_ALLOWANCE
        );

        // Depth: the ILP encoding reproduces the exact greedy optimum.
        let depth = solve::<DefaultMilp>(&egraph, &roots, IlpObjective::Depth, &opts).unwrap();
        assert_eq!(depth.report.outcome, SolveOutcome::Optimal);
        let got = depth.extraction.dag_depth(&egraph, &roots).into_inner();
        assert!(close(got, min_depth(&egraph, &roots)));

        // Weighted: the solver's objective is the objective of the extraction.
        let weighted = IlpObjective::WeightedSizeDepth {
            size_weight: 0.3,
            depth_weight: 1.0,
        };
        let outcome = solve::<DefaultMilp>(&egraph, &roots, weighted, &opts).unwrap();
        assert_eq!(outcome.report.outcome, SolveOutcome::Optimal);
        let evaluated = weighted.evaluate(&egraph, &roots, &outcome.extraction);
        assert!(close(outcome.report.objective.unwrap(), evaluated));
    }
}

#[test]
fn initial_warm_start_requires_flagged_nodes() {
    let egraph = generate_random_egraph();
    let roots = egraph.root_eclasses.clone();
    let opts = options(None, WarmStart::Initial);
    let result = solve::<DefaultMilp>(&egraph, &roots, IlpObjective::Size, &opts);
    assert!(matches!(result, Err(IlpError::MissingInitial(_))));
}

#[test]
fn invalid_requests_are_errors() {
    let egraph = generate_random_egraph();
    let roots = egraph.root_eclasses.clone();
    let opts = options(None, WarmStart::None);
    let negative = IlpObjective::WeightedSizeDepth {
        size_weight: -1.0,
        depth_weight: 1.0,
    };
    assert!(matches!(
        solve::<DefaultMilp>(&egraph, &roots, negative, &opts),
        Err(IlpError::InvalidInput(_))
    ));
    assert!(matches!(
        solve::<DefaultMilp>(&egraph, &[], IlpObjective::Size, &opts),
        Err(IlpError::NoRoots)
    ));
}

/// A supplied feasible start must be accepted and returned rather than
/// nothing. Checks the solver log too: the returned answer alone cannot tell a
/// used start from a quick solve.
///
/// A zero time limit is no good here: Gurobi stops before it loads the start.
#[cfg(not(feature = "ilp-cbc"))]
#[test]
fn initial_warm_start_is_accepted_and_used() {
    for i in 0..EGRAPHS {
        let mut egraph = generate_random_egraph();
        let roots = egraph.root_eclasses.clone();
        let greedy = GreedySizeExtractor.extract(&egraph, &roots);
        flag_initial(&mut egraph, &greedy);

        let log =
            std::env::temp_dir().join(format!("exgym-warm-start-{}-{i}.log", std::process::id()));
        let mut opts = options(Some(Duration::from_secs(1)), WarmStart::Initial);
        opts.solver_log = Some(log.clone());
        let outcome = solve::<DefaultMilp>(&egraph, &roots, IlpObjective::Size, &opts).unwrap();
        let report = &outcome.report;
        assert_eq!(
            report.warm_start_applied,
            WarmStart::Initial,
            "{:?}",
            report.warm_start_note
        );
        assert!(
            !matches!(report.outcome, SolveOutcome::Fallback { .. }),
            "{:?}",
            report.outcome
        );
        assert!(
            report.objective.unwrap() <= report.warm_start_objective.unwrap() + EPSILON_ALLOWANCE
        );

        #[cfg(feature = "ilp-gurobi")]
        {
            let text = std::fs::read_to_string(&log).unwrap();
            assert!(
                text.contains("Loaded user MIP start"),
                "start not loaded:\n{text}"
            );
        }
        let _ = std::fs::remove_file(&log);
    }
}
