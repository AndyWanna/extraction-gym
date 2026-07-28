pub mod extract;

pub use extract::*;

pub use egraph_serialize::*;

use indexmap::IndexMap;
use ordered_float::NotNan;

pub type Cost = NotNan<f64>;
pub const INFINITY: Cost = unsafe { NotNan::new_unchecked(f64::INFINITY) };

#[derive(PartialEq, Eq)]
pub enum Optimal {
    Tree,
    #[cfg(feature = "ilp-cbc")]
    Dag,
    Neither,
}

pub struct ExtractorDetail {
    pub extractor: Box<dyn Extractor>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub optimal: Optimal,
    pub use_for_bench: bool,
}

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
        /*(
            "global-greedy-dag",
            ExtractorDetail {
                extractor: extract::global_greedy_dag::GlobalGreedyDagExtractor.boxed(),
                optimal: Optimal::Neither,
                use_for_bench: true,
            },
        ),*/
        #[cfg(feature = "ilp-cbc")]
        (
            "ilp-cbc-timeout",
            ExtractorDetail {
                extractor: extract::ilp_cbc::CbcExtractorWithTimeout::<10>.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-cbc")]
        (
            "ilp-cbc",
            ExtractorDetail {
                extractor: extract::ilp_cbc::CbcExtractor {
                    timeout_seconds: u32::MAX,
                }
                .boxed(),
                optimal: Optimal::Dag,
                use_for_bench: false, // takes >10 hours sometimes
            },
        ),
        #[cfg(feature = "ilp-cbc")]
        (
            "faster-ilp-cbc-timeout",
            ExtractorDetail {
                extractor: extract::faster_ilp_cbc::FasterCbcExtractorWithTimeout::<10>.boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-cbc")]
        (
            "faster-ilp-cbc",
            ExtractorDetail {
                extractor: extract::faster_ilp_cbc::FasterCbcExtractor {
                    timeout_seconds: u32::MAX,
                }
                .boxed(),
                optimal: Optimal::Dag,
                use_for_bench: true,
            },
        ),
        #[cfg(feature = "ilp-cbc")]
        (
            "dual-ilp-cbc-timeout",
            ExtractorDetail {
                // Minimises alpha*(critical-path delay) + beta*(area). Not a
                // sum-cost optimum, so Optimal::Neither and off by default in bench.
                extractor: extract::ilp_cbc::DualCbcExtractor {
                    timeout_seconds: 10,
                    alpha: 1.0,
                    beta: 1.0,
                }
                .boxed(),
                optimal: Optimal::Neither,
                use_for_bench: false,
            },
        ),
        #[cfg(feature = "ilp-cbc")]
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


#[cfg(test)]
pub mod test;
