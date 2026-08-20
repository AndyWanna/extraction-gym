//! Build script: only job is to make the HiGHS backend link.
//!
//! `highs-sys` (default features `build` + `highs_release`) vendors HiGHS and
//! builds it with cmake as a static C++ library, then emits
//! `cargo:rustc-link-lib=dylib=stdc++`. On some toolchains the compiler rustc
//! shells out to cannot find `libstdc++.so` on its own, so the final link dies
//! with `unable to find library -lstdc++`.
//!
//! Rather than exporting a global `RUSTFLAGS` (which would invalidate the build
//! cache of every other crate that shares `$CARGO_HOME`), ask the C compiler
//! where its libstdc++ lives and add exactly that directory to the link search
//! path. Cargo propagates `rustc-link-search` from build scripts in the
//! dependency graph to the final binary link, so this covers dependents too.
//!
//! No-op unless the `ilp-highs` feature is on.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=CC");

    if std::env::var_os("CARGO_FEATURE_ILP_HIGHS").is_none() {
        return;
    }
    if cfg!(not(target_os = "linux")) {
        return;
    }

    // Try the configured compiler first, then the usual names. A compiler that
    // cannot find the library echoes the bare name back instead of failing, and
    // some /usr/bin/cc installs do exactly that while the gcc on PATH
    // resolves it -- hence the list rather than a single candidate.
    let mut candidates: Vec<String> = Vec::new();
    for var in ["CXX", "CC"] {
        if let Ok(v) = std::env::var(var) {
            candidates.push(v);
        }
    }
    candidates.extend(["c++", "g++", "gcc", "cc"].iter().map(|s| s.to_string()));

    for cc in &candidates {
        let out = match Command::new(cc).arg("-print-file-name=libstdc++.so").output() {
            Ok(o) if o.status.success() => o,
            _ => continue,
        };
        let path = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim().to_string());
        if !path.is_absolute() {
            continue;
        }
        if let Some(dir) = path.parent() {
            // Canonicalize: gcc reports the path through its own lib dir with
            // `../../..` segments, which rustc accepts but is unreadable in -L.
            let dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
            println!("cargo:rustc-link-search=native={}", dir.display());
            return;
        }
    }

    println!(
        "cargo:warning=extraction-gym: could not locate libstdc++.so via any of {candidates:?}; \
         if the link fails with `unable to find library -lstdc++`, set \
         RUSTFLAGS=\"-L native=<dir containing libstdc++.so>\""
    );
}
