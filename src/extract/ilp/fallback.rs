/*!
The candidate extractions an ILP solve can fall back on, and the rule for
picking between them. The same pick seeds `WarmStart::Greedy`, so the seed and
the fallback can never disagree.
*/

use super::{IlpError, IlpObjective};
use crate::*;

pub(crate) struct Candidate {
    /// `"initial"`, `"greedy-size"` or `"greedy-depth"`.
    pub name: &'static str,
    pub extraction: CompleteExtraction,
    /// [`IlpObjective::evaluate`] of `extraction`.
    pub objective: f64,
    pub depth: f64,
}

impl Candidate {
    fn new(
        name: &'static str,
        extraction: CompleteExtraction,
        egraph: &EGraph,
        roots: &[ClassId],
        objective: IlpObjective,
    ) -> Self {
        Candidate {
            name,
            objective: objective.evaluate(egraph, roots, &extraction),
            depth: extraction.dag_depth(egraph, roots).into_inner(),
            extraction,
        }
    }
}

/// The initial extraction (if any), then the objective's greedy extraction if
/// `with_greedy`. Empty only when there is neither.
pub(crate) fn candidates(
    egraph: &EGraph,
    roots: &[ClassId],
    objective: IlpObjective,
    initial: Option<CompleteExtraction>,
    with_greedy: bool,
) -> Result<Vec<Candidate>, IlpError> {
    let mut candidates: Vec<Candidate> = initial
        .map(|e| Candidate::new("initial", e, egraph, roots, objective))
        .into_iter()
        .collect();
    if !with_greedy {
        return Ok(candidates);
    }

    let (name, extractor) = objective.greedy_extractor();
    match CompleteExtraction::try_new(extractor.extract(egraph, roots), egraph, roots) {
        Ok(e) => candidates.push(Candidate::new(name, e, egraph, roots, objective)),
        // Greedy finds a valid extraction whenever one exists, so with no
        // initial extraction to fall back on the e-graph has none at all.
        Err(why) if candidates.is_empty() => return Err(IlpError::NoValidExtraction(why)),
        Err(why) => {
            log::warn!("{name} extraction is invalid ({why}); falling back on initial only")
        }
    }
    Ok(candidates)
}

/// Index of the best candidate under `objective`, and whether it violates the
/// depth budget; `None` if there are no candidates. Ties go to the earlier
/// candidate, i.e. the initial one.
///
/// With a depth budget, candidates within budget are compared by size; if
/// none is, the shallowest is returned and flagged.
pub(crate) fn best(objective: IlpObjective, candidates: &[Candidate]) -> Option<(usize, bool)> {
    let min_by = |key: fn(&Candidate) -> f64, within: &dyn Fn(&Candidate) -> bool| {
        candidates
            .iter()
            .enumerate()
            .filter(|(_, c)| within(c))
            .min_by(|(_, a), (_, b)| key(a).total_cmp(&key(b)))
            .map(|(i, _)| i)
    };
    let any = |_: &Candidate| true;
    match objective.budget_value() {
        None => min_by(|c| c.objective, &any).map(|i| (i, false)),
        Some(budget) => match min_by(|c| c.objective, &|c| c.depth <= budget) {
            Some(i) => Some((i, false)),
            None => min_by(|c| c.depth, &any).map(|i| (i, true)),
        },
    }
}
