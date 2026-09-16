/*!
Solver-free extractors used as ILP warm starts and fallbacks.

| Extractor | Minimises | Optimal? |
|---|---|---|
| [`GreedySizeExtractor`] | DAG size (sum of `Node::cost` over shared nodes) | no (heuristic) |
| [`GreedyDepthExtractor`] | depth (max over paths of summed `Node::delay`) | **yes** |
| [`InitialExtractor`] | nothing: rebuilds the initial expression from `Node::initial` | n/a |

Every one of them returns a complete, acyclic extraction whenever one exists.
*/

pub mod depth;
mod fixpoint;
pub mod initial;

pub use super::faster_greedy_dag::FasterGreedyDagExtractor as GreedySizeExtractor;
pub use depth::GreedyDepthExtractor;
pub use initial::InitialExtractor;
