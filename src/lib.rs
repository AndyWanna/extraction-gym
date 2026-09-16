pub mod extract;
#[cfg(feature = "ilp-any")]
pub mod milp;

pub use extract::*;

pub use egraph_serialize::*;

use indexmap::IndexMap;
use ordered_float::NotNan;

pub type Cost = NotNan<f64>;
pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(f64::INFINITY) };

#[derive(PartialEq, Eq)]
pub enum Optimal {
    Tree,
    #[cfg(feature = "ilp-any")]
    Dag,
    Neither,
}

pub struct ExtractorDetail {
    pub extractor: Box<dyn Extractor>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub optimal: Optimal,
    pub use_for_bench: bool,
}

#[allow(deprecated)]
pub fn extractors() -> IndexMap<&'static str, ExtractorDetail> {
    let extractors: IndexMap<&'static str, ExtractorDetail> = [
        (
            "bottom-up",
            ExtractorDetail {
                extractor: extract::bottom_up::BottomUpExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "faster-bottom-up",
            ExtractorDetail {
                extractor: extract::faster_bottom_up::FasterBottomUpExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "prio-queue",
            ExtractorDetail {
                extractor: extract::prio_queue::PrioQueueExtractor.boxed(),
                optimal: Optimal::Tree,
                use_for_bench: true,
            },
        ),
        (
            "faster-greedy-dag",
            ExtractorDetail {
                extractor: extract::faster_greedy_dag::FasterGreedyDagExtractor.boxed(),
                optimal: Optimal::Neither,
                use_for_bench: true,
            },
        ),
        (
            "greedy-depth",
            ExtractorDetail {
                extractor: extract::greedy::GreedyDepthExtractor.boxed(),
                optimal: Optimal::Neither,
                use_for_bench: true,
            },
        ),
        /*(
            "global-greedy-dag",
            ExtractorDetail {
                extractor: extract::global_greedy_dag::GlobalGreedyDagExtractor.boxed(),
                optimal: Optimal::Neither,
                use_for_bench: true,
            },
        ),*/
        // The ILP extractors. Benchmark e-graphs carry no Node::initial flags,
        // so these seed from the greedy extraction rather than the initial one.
        #[cfg(feature = "ilp-any")]
        (
            "ilp-size-timeout",
            ExtractorDetail {
                extractor: extract::ilp::SizeExtractor {
                    options: bench_ilp_options(Some(10)),
                }
                .boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "ilp-size",
            ExtractorDetail {
                extractor: extract::ilp::SizeExtractor {
                    options: bench_ilp_options(None),
                }
                .boxed(),
                optimal: Optimal::Dag,
                use_for_bench: false, // takes >10 hours sometimes
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "ilp-depth-timeout",
            ExtractorDetail {
                extractor: extract::ilp::DepthExtractor {
                    options: bench_ilp_options(Some(10)),
                }
                .boxed(),
                optimal: Optimal::Neither,
                use_for_bench: false,
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "ilp-weighted-size-depth-timeout",
            ExtractorDetail {
                // Not a size optimum, so Optimal::Neither.
                extractor: extract::ilp::WeightedSizeDepthExtractor {
                    size_weight: 1.0,
                    depth_weight: 1.0,
                    options: bench_ilp_options(Some(10)),
                }
                .boxed(),
                optimal: Optimal::Neither,
                use_for_bench: false,
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "ilp-size-constrained-depth-timeout",
            ExtractorDetail {
                // The budget is an arbitrary placeholder: infeasible on some
                // e-graphs (returning the flagged shallowest fallback) and
                // slack on others.
                extractor: extract::ilp::SizeConstrainedDepthExtractor {
                    depth_budget: extract::ilp::DepthBudget::Fixed(100.0),
                    options: bench_ilp_options(Some(10)),
                }
                .boxed(),
                optimal: Optimal::Neither,
                use_for_bench: false,
            },
        ),
        // Deprecated extractors, kept under their original keys.
        #[cfg(feature = "ilp-any")]
        (
            "ilp-cbc-timeout",
            ExtractorDetail {
                extractor: extract::ilp_cbc::IlpExtractorWithTimeout::<10>.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "ilp-cbc",
            ExtractorDetail {
                extractor: extract::ilp_cbc::CbcExtractor {
                    timeout_seconds: u32::MAX,
                    threads: 1,
                    ..Default::default()
                }
                .boxed(),
                optimal: Optimal::Dag,
                use_for_bench: false, // takes >10 hours sometimes
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "faster-ilp-cbc-timeout",
            ExtractorDetail {
                extractor: extract::faster_ilp_cbc::FasterIlpExtractorWithTimeout::<10>.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "faster-ilp-cbc",
            ExtractorDetail {
                extractor: extract::faster_ilp_cbc::FasterCbcExtractor {
                    timeout_seconds: u32::MAX,
                    threads: 1,
                    ..Default::default()
                }
                .boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "dual-ilp-cbc-timeout",
            ExtractorDetail {
                // Minimises alpha*(critical-path delay) + beta*(area). Not a
                // sum-cost optimum, so Optimal::Neither and off by default in bench.
                extractor: extract::ilp_cbc::DualCbcExtractor {
                    timeout_seconds: 10,
                    alpha: 1.0,
                    beta: 1.0,
                    threads: 1,
                    ..Default::default()
                }
                .boxed(),
                optimal: Optimal::Neither,
                use_for_bench: false,
            },
        ),
        #[cfg(feature = "ilp-any")]
        (
            "delay-budget-ilp-cbc-timeout",
            ExtractorDetail {
                // Minimises area subject to critical-path delay <= max_delay.
                // max_delay is a fixed placeholder for benchmarking purposes
                // only -- it is arbitrary and may be infeasible or slack
                // depending on the egraph under test.
                extractor: extract::ilp_cbc::DelayBudgetCbcExtractor {
                    timeout_seconds: 10,
                    max_delay: 100.0,
                    threads: 1,
                    ..Default::default()
                }
                .boxed(),
                optimal: Optimal::Neither,
                use_for_bench: false,
            },
        ),
    ]
    .into_iter()
    .collect();
    extractors
}


#[cfg(feature = "ilp-any")]
fn bench_ilp_options(time_limit_seconds: Option<u64>) -> extract::ilp::IlpOptions {
    extract::ilp::IlpOptions {
        time_limit: time_limit_seconds.map(std::time::Duration::from_secs),
        warm_start: extract::ilp::WarmStart::Greedy,
        ..Default::default()
    }
}

#[cfg(test)]
pub mod test;
