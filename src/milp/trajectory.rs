//! Recover an incumbent-vs-time trajectory from a solver's own log file.
//!
//! # Why parse a log instead of using a callback
//!
//! The benchmark matrix runs the same instance at 10 min / 1 h / 3 h. Without a
//! trajectory those are three separate runs whose only comparable output is the
//! final objective — and on instances where every configuration times out, the
//! final objective alone cannot say whether a configuration was *converging*.
//! With a trajectory, one 1-hour run also yields its own 10-minute point and
//! shows whether the gap was still closing when the wall hit.
//!
//! Gurobi and HiGHS both print a timestamped line every time the incumbent or
//! the bound moves, so the information is already there for free once
//! [`super::MilpModel::set_log_file`] points the log at a file. Three different
//! callback APIs (one of which, CBC's, is not exposed by `coin_cbc` at all)
//! would be far more code for the same data.
//!
//! # Robustness policy
//!
//! Log formats are not a stable API. Every line that does not parse is silently
//! skipped, and a log that yields nothing produces an empty trajectory rather
//! than an error: a missing trajectory must degrade the *reporting*, never the
//! run. Callers should treat "empty" as "unknown", not as "no improvement".

use std::path::Path;

/// One point on the incumbent/bound trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Incumbent {
    /// Which `solve()` call this came from. The cycle-breaking loop in
    /// `faster_ilp_cbc` solves repeatedly, and each solve restarts the solver's
    /// own clock, so `secs` is only comparable within one `solve_index`.
    pub solve_index: usize,
    /// Seconds since the start of that solve, as the solver reported them.
    pub secs: f64,
    /// Objective value of the incumbent at that moment.
    pub objective: f64,
    /// Best proven bound at that moment, when the log reported one.
    pub bound: Option<f64>,
}

/// Which log dialect to expect. Chosen from `MilpModel::NAME` by
/// [`parse_log_for_backend`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogDialect {
    Gurobi,
    Highs,
}

/// Parse `path` in the dialect matching `backend_name` (`MilpModel::NAME`).
/// Returns an empty vector for an unknown backend, a missing file, or a log
/// with no recognisable rows.
pub fn parse_log_for_backend(path: &Path, backend_name: &str) -> Vec<Incumbent> {
    let dialect = match backend_name {
        "Gurobi" => LogDialect::Gurobi,
        "HiGHS" => LogDialect::Highs,
        _ => return Vec::new(),
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("could not read MILP log {}: {e}", path.display());
            return Vec::new();
        }
    };
    parse_log(&text, dialect)
}

/// Parse an already-read log body. Split out from [`parse_log_for_backend`] so
/// it is unit-testable without touching the filesystem.
///
/// The parser is a small state machine rather than a per-line regex, for two
/// reasons that both showed up in real logs:
///
/// * A "Solving report" / "Explored N nodes" summary block repeats the final
///   objective in a shape that looks enough like a progress row to be picked
///   up, which would fabricate a duplicate trajectory point at t = 0.
/// * HiGHS runs a **separate sub-MIP** to complete a partial MIP start
///   ("Attempting to find feasible solution by solving MIP for user-supplied
///   values of N / M discrete variables"), and that sub-MIP prints its own
///   progress table with its own, unrelated objective. Feeding those numbers
///   into the trajectory would report the warm start as a wild objective swing.
///
/// So rows are only read while inside a genuine progress table, and a table
/// belonging to a warm-start completion sub-MIP is skipped entirely.
///
/// # `solve_index` is not a time axis, and objectives are not comparable across it
///
/// `faster_ilp_cbc` solves the model repeatedly, adding cycle-blocking rows each
/// round, so **each `solve_index` is a different, strictly more constrained
/// model** and its optimum is generally *higher* than the previous round's. One
/// observed run walked `4.5e-05 -> 0.2996 -> 0.3521 -> ... -> 1.0409` across
/// seven solves; only the last is the DAG optimum. A consumer that concatenated
/// those onto one axis would report the run as monotonically getting *worse*.
/// Filter to the final `solve_index` (or plot each separately). The single-solve
/// extractors in `ilp_cbc` do not have this problem.
///
/// Each segment is also normalised against its own reported primal bound
/// (`primal_bound - last row's objective`), which guards against a solver whose
/// progress table is in a shifted (presolved) objective space. On the logs seen
/// so far this shift is exactly zero, i.e. it is a safety net rather than a
/// correction; a segment with no readable primal bound is emitted unshifted.
pub fn parse_log(text: &str, dialect: LogDialect) -> Vec<Incumbent> {
    let mut out: Vec<Incumbent> = Vec::new();
    let mut pending: Vec<Incumbent> = Vec::new();
    let mut solve_index = 0usize;
    let mut in_table = false;
    // Set when the next table belongs to the MIP-start completion sub-MIP.
    let mut skip_table = false;
    // Set between a segment's end and its "Primal bound" line.
    let mut awaiting_bound = false;

    for line in text.lines() {
        let t = line.trim();

        if is_warm_start_submip(t, dialect) {
            skip_table = true;
            // Any pending segment ends here unshifted; the sub-MIP's own
            // Solving report must not be read as this segment's bound.
            flush(&mut out, &mut pending, None, &mut solve_index);
            awaiting_bound = false;
            continue;
        }
        if awaiting_bound {
            if let Some(v) = parse_primal_bound(t, dialect) {
                flush(&mut out, &mut pending, Some(v), &mut solve_index);
                awaiting_bound = false;
                continue;
            }
        }
        if is_table_header(t, dialect) {
            // A new table with the previous segment still unresolved: emit it
            // as-is rather than losing it.
            if awaiting_bound {
                flush(&mut out, &mut pending, None, &mut solve_index);
                awaiting_bound = false;
            }
            in_table = true;
            continue;
        }
        if is_segment_end(t, dialect) {
            in_table = false;
            if skip_table {
                // That was the completion sub-MIP, not a solve of our model.
                skip_table = false;
                pending.clear();
            } else if !pending.is_empty() {
                awaiting_bound = true;
            }
            continue;
        }
        if !in_table || skip_table || t.is_empty() {
            continue;
        }
        let Some((secs, obj, bound)) = parse_row(line, dialect) else {
            continue;
        };
        // Keep only points where the incumbent actually moved; a MIP log prints
        // a heartbeat row every few seconds with an unchanged incumbent, and
        // 3 hours of those is noise, not a trajectory.
        if let Some(last) = pending.last() {
            if last.objective == obj {
                continue;
            }
        }
        pending.push(Incumbent {
            solve_index,
            secs,
            objective: obj,
            bound,
        });
    }
    flush(&mut out, &mut pending, None, &mut solve_index);
    out
}

/// Move `pending` into `out`, shifting into true objective space if the
/// segment's reported primal bound is known, and advance the solve counter.
fn flush(
    out: &mut Vec<Incumbent>,
    pending: &mut Vec<Incumbent>,
    primal_bound: Option<f64>,
    solve_index: &mut usize,
) {
    if pending.is_empty() {
        return;
    }
    let shift = match (primal_bound, pending.last()) {
        (Some(pb), Some(last)) => pb - last.objective,
        _ => 0.0,
    };
    for mut p in pending.drain(..) {
        p.solve_index = *solve_index;
        if shift != 0.0 {
            p.objective += shift;
            p.bound = p.bound.map(|b| b + shift);
        }
        out.push(p);
    }
    *solve_index += 1;
}

/// The solver's own statement of the segment's true (un-presolved) incumbent.
fn parse_primal_bound(line: &str, dialect: LogDialect) -> Option<f64> {
    match dialect {
        LogDialect::Highs => line
            .strip_prefix("Primal bound")
            .and_then(|r| r.trim().parse::<f64>().ok()),
        // Gurobi's table is already in the original objective space, so there
        // is nothing to correct; returning None keeps the shift at zero.
        LogDialect::Gurobi => None,
    }
}

/// The line that announces HiGHS' MIP-start completion sub-MIP.
fn is_warm_start_submip(line: &str, dialect: LogDialect) -> bool {
    match dialect {
        LogDialect::Highs => line.starts_with("Attempting to find feasible solution"),
        // Gurobi completes a partial start internally without a separate log
        // section, so there is nothing to skip.
        LogDialect::Gurobi => false,
    }
}

/// The header line immediately above a progress table.
fn is_table_header(line: &str, dialect: LogDialect) -> bool {
    match dialect {
        LogDialect::Highs => line.contains("InQueue") && line.contains("Leaves"),
        LogDialect::Gurobi => line.contains("Incumbent") && line.contains("BestBd"),
    }
}

/// Anything that ends the current table / solve.
fn is_segment_end(line: &str, dialect: LogDialect) -> bool {
    match dialect {
        LogDialect::Highs => line.starts_with("Solving report"),
        LogDialect::Gurobi => {
            line.starts_with("Explored ")
                || line.starts_with("Cutting planes:")
                || line.starts_with("Optimal solution found")
                || line.starts_with("Time limit reached")
                || line.starts_with("Solution count")
        }
    }
}

/// Both dialects put the elapsed time last, formatted as `<number>s`.
fn parse_time_token(tok: &str) -> Option<f64> {
    let t = tok.strip_suffix('s')?;
    t.parse::<f64>().ok()
}

/// A number, or `None` for the placeholders both solvers use for "not known
/// yet" (`-`, `inf`, `-inf`, `1e+30`, `Large`).
fn parse_num(tok: &str) -> Option<f64> {
    if tok == "-" {
        return None;
    }
    let v: f64 = tok.parse().ok()?;
    if !v.is_finite() || v.abs() >= 1e29 {
        return None;
    }
    Some(v)
}

/// Pull `(secs, incumbent, bound)` out of one progress row, or `None` if the
/// line is not one. Both dialects have FIXED right-hand columns, which is what
/// makes positional parsing safe here (the left-hand columns vary: Gurobi
/// prefixes `H`/`*`, HiGHS prefixes a one-letter `Src` code).
fn parse_row(line: &str, dialect: LogDialect) -> Option<(f64, f64, Option<f64>)> {
    let toks: Vec<&str> = line.split_whitespace().collect();
    let n = toks.len();
    let secs = parse_time_token(toks.last()?)?;

    match dialect {
        // ... Incumbent  BestBd  Gap  It/Node  Time
        LogDialect::Gurobi => {
            if n < 5 {
                return None;
            }
            let obj = parse_num(toks[n - 5])?;
            let bound = parse_num(toks[n - 4]);
            Some((secs, obj, bound))
        }
        // ... BestBound  BestSol  Gap  Cuts  InLp  Confl.  LpIters  Time
        LogDialect::Highs => {
            if n < 8 {
                return None;
            }
            let bound = parse_num(toks[n - 8]);
            let obj = parse_num(toks[n - 7])?;
            Some((secs, obj, bound))
        }
    }
}

/// Seconds at which the first feasible incumbent appeared, within the first
/// solve that produced one. `None` if the trajectory is empty.
///
/// Note the "within the first solve": under `faster_ilp_cbc`'s cycle-breaking
/// loop that is the first solve of the *relaxed* model, which is the honest
/// answer to "how long until the solver had something" but not to "how long
/// until it had a cycle-free extraction".
pub fn time_to_first_incumbent(traj: &[Incumbent]) -> Option<f64> {
    traj.first().map(|i| i.secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gurobi_rows() {
        let log = "\
Gurobi Optimizer version 10.0.3 build v10.0.3rc0
    Nodes    |    Current Node    |     Objective Bounds      |     Work
 Expl Unexpl |  Obj  Depth IntInf | Incumbent    BestBd   Gap | It/Node Time

     0     0    1.00000    0   10          -    1.00000      -     -    0s
H    0     0                       2.0000000    1.00000  50.0%     -    1s
*  100    50              10       1.5000000    1.40000  6.67%   5.0    7s
";
        let t = parse_log(log, LogDialect::Gurobi);
        assert_eq!(t.len(), 2, "{t:?}");
        assert_eq!(t[0].objective, 2.0);
        assert_eq!(t[0].secs, 1.0);
        assert_eq!(t[0].bound, Some(1.0));
        assert_eq!(t[1].objective, 1.5);
        assert_eq!(t[1].secs, 7.0);
    }

    #[test]
    fn highs_rows() {
        let log = "\
        Nodes      |    B&B Tree     |            Objective Bounds              |  Dynamic Constraints |       Work
     Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time

         0       0         0   0.00%   -inf            inf                  inf        0      0      0         0     0.0s
 R       0       0         0   0.00%   1.0             3.0                66.66%       0      0      0        12     0.4s
         9       2         1   4.00%   1.2             2.0                40.00%       3      5      0       120     9.5s
";
        let t = parse_log(log, LogDialect::Highs);
        assert_eq!(t.len(), 2, "{t:?}");
        assert_eq!(t[0].objective, 3.0);
        assert_eq!(t[0].bound, Some(1.0));
        assert_eq!(t[0].secs, 0.4);
        assert_eq!(t[1].objective, 2.0);
        assert_eq!(t[1].secs, 9.5);
        assert_eq!(time_to_first_incumbent(&t), Some(0.4));
    }

    /// Regression for the exact shape a warm-started HiGHS run produces: a
    /// MIP-start completion sub-MIP with its own progress table and its own
    /// (unrelated) objective, followed by the real solve. Reading the sub-MIP's
    /// rows would report the warm start as a wild objective swing -- here from
    /// ~1.04 down to 4.5e-05 -- which is not this model's objective at all.
    #[test]
    fn highs_warm_start_submip_is_skipped() {
        let log = "\
MIP has 3545 rows; 4306 cols
Attempting to find feasible solution by solving MIP for user-supplied values of 1935 / 4306 discrete variables
Presolving model
Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time
 J       0       0         0   0.00%   -inf            0.082462           Large        0      0      0         0     0.1s
 T       0       0         0   0.00%   1.8e-05         4.5e-05           60.00%        0      0      0        50     0.1s

Solving report
  Status            Optimal
  Primal bound      1.040918
MIP start solution is feasible, objective value is 1.040918

Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time
 X       0       0         0   0.00%   0.5             1.040918           50.00%       0      0      0         0     0.2s
         9       2         1   4.00%   1.040918        1.040918            0.00%       3      5      0       120     1.7s

Solving report
";
        let t = parse_log(log, LogDialect::Highs);
        assert_eq!(t.len(), 1, "sub-MIP rows leaked into the trajectory: {t:?}");
        assert_eq!(t[0].objective, 1.040918);
        assert_eq!(t[0].secs, 0.2);
    }

    /// The objective-space safety net: when a segment's own "Primal bound"
    /// disagrees with the last row of its table, the whole segment is shifted
    /// to agree with it. (Zero shift on every log observed so far, but a
    /// silently shifted axis would corrupt every checkpoint the report
    /// derives, so it is checked.)
    #[test]
    fn highs_segment_is_normalised_to_its_primal_bound() {
        let log = "\
Src  Proc. InQueue |  Leaves   Expl. | BestBound       BestSol              Gap |   Cuts   InLp Confl. | LpIters     Time
 J       0       0         0   0.00%   -inf            0.082462           Large        0      0      0         0     0.1s
 T       0       0         0   0.00%   1.8e-05         4.5e-05           60.00%        0      0      0        50     0.3s

Solving report
  Status            Optimal
  Primal bound      1.040918
  Dual bound        1.040918
";
        let t = parse_log(log, LogDialect::Highs);
        assert_eq!(t.len(), 2, "{t:?}");
        // shift = 1.040918 - 4.5e-05
        assert!((t[1].objective - 1.040918).abs() < 1e-12, "{t:?}");
        assert!((t[0].objective - 1.1233350).abs() < 1e-6, "{t:?}");
    }

    #[test]
    fn garbage_is_ignored() {
        assert!(parse_log("nothing here\nat all\n", LogDialect::Gurobi).is_empty());
        assert!(parse_log("", LogDialect::Highs).is_empty());
    }
}
