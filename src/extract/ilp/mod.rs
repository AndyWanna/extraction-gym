/*!
ILP extraction. Every ILP extractor is one [`IlpObjective`] over the single
model in [`model`].
*/

pub(crate) mod decode;
pub(crate) mod model;
pub mod objective;
pub mod warm;

pub use objective::IlpObjective;
