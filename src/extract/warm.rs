/*!
Legacy paths for the warm-start mode, the solver report and seed helpers.
*/

#[allow(deprecated)]
pub use super::faster_ilp_cbc::try_dag_cost;
pub use super::ilp::report::{SolveOutcome, SolveReport};
pub use super::ilp::warm::{arrival_times, topological_levels};
#[allow(deprecated)]
pub use super::ilp_cbc::{build_seed, Seed};

/// Legacy name for [`super::ilp::WarmStart`].
#[deprecated(note = "use ilp::WarmStart")]
pub type WarmStartMode = super::ilp::WarmStart;
