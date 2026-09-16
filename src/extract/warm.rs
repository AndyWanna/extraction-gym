/*!
Legacy paths for the warm-start mode, the solver report and seed
construction, which now live in [`super::ilp`].
*/

pub use super::ilp::report::{SolveOutcome, SolveReport};
pub use super::ilp::warm::{arrival_times, build_seed, topological_levels, try_dag_cost, Seed};

/// Legacy name for [`super::ilp::WarmStart`].
#[deprecated(note = "use ilp::WarmStart")]
pub type WarmStartMode = super::ilp::WarmStart;
