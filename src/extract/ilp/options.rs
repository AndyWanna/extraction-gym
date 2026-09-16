use std::path::PathBuf;
use std::time::Duration;

/// Which extraction to hand the solver as its starting incumbent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WarmStart {
    /// No MIP start.
    None,
    /// Start from the initial expression (`greedy::InitialExtractor`).
    /// Requires the e-graph to flag its initial nodes (`Node::initial`).
    #[default]
    Initial,
    /// Start from the better of the initial expression (if flagged) and the
    /// objective's greedy extraction, judged by the objective. This is the
    /// same extraction that is returned if the solver fails.
    Greedy,
}

impl WarmStart {
    pub fn as_str(self) -> &'static str {
        match self {
            WarmStart::None => "none",
            WarmStart::Initial => "initial",
            WarmStart::Greedy => "greedy",
        }
    }
}

impl std::fmt::Display for WarmStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for WarmStart {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Ok(WarmStart::None),
            "initial" => Ok(WarmStart::Initial),
            "greedy" => Ok(WarmStart::Greedy),
            other => Err(format!(
                "unknown warm start {other:?} (expected none|initial|greedy)"
            )),
        }
    }
}

/// Solver settings shared by every ILP extractor.
#[derive(Debug, Clone)]
pub struct IlpOptions {
    /// Wall-clock limit for the solve, rounded up to whole seconds. `None` is
    /// unbounded, which can take hours on large e-graphs. Zero stops the solver
    /// before it even loads a warm start, so it always returns the fallback.
    pub time_limit: Option<Duration>,
    /// Solver threads. Keep `threads x concurrent solver processes` within
    /// the CPU allocation; see `MilpModel::set_threads`.
    pub threads: u32,
    pub warm_start: WarmStart,
    /// Where to write the solver's own log (enables the incumbent trajectory
    /// in the report).
    pub solver_log: Option<PathBuf>,
    /// Backend-specific parameters, passed verbatim to
    /// `MilpModel::set_raw_parameter`.
    pub raw_params: Vec<(String, String)>,
}

impl Default for IlpOptions {
    fn default() -> Self {
        IlpOptions {
            time_limit: None,
            threads: 1,
            warm_start: WarmStart::default(),
            solver_log: None,
            raw_params: Vec::new(),
        }
    }
}

impl IlpOptions {
    pub fn with_time_limit(mut self, time_limit: Duration) -> Self {
        self.time_limit = Some(time_limit);
        self
    }

    pub fn with_warm_start(mut self, warm_start: WarmStart) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// The limit in the backend's whole seconds (rounded up; `u32::MAX` for
    /// unbounded).
    pub(crate) fn time_limit_seconds(&self) -> u32 {
        time_limit_seconds(self.time_limit)
    }
}

pub(crate) fn time_limit_seconds(time_limit: Option<Duration>) -> u32 {
    match time_limit {
        None => u32::MAX,
        Some(d) => d.as_secs_f64().ceil().min(u32::MAX as f64) as u32,
    }
}

/// Inverse of [`time_limit_seconds`], for the legacy `timeout_seconds` fields.
pub(crate) fn time_limit_from_seconds(seconds: u32) -> Option<Duration> {
    (seconds != u32::MAX).then(|| Duration::from_secs(seconds.into()))
}
