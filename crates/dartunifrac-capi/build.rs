//! Regenerate the C header `dartunifrac.h` on every build, so it can never drift
//! from the `#[unsafe(no_mangle)]` surface. cbindgen is a build-time-only
//! dependency.
//!
//! The header goes to `$DF_HEADER_DIR/dartunifrac.h` when that is set (an
//! absolute path), and to `$OUT_DIR/include/dartunifrac.h` otherwise. It never
//! goes into the source tree by default: two builds of one checkout (native and
//! wasm, debug and release) would race on the one file, and a read-only source
//! tree would fail the build. `make capi` sets `DF_HEADER_DIR` to the repo's
//! `include/` for the C harness; a consumer building this crate through cargo
//! sets it to a directory of its own and puts that on its include path.

use std::path::PathBuf;

fn main() {
    let crate_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let header_dir = match std::env::var_os("DF_HEADER_DIR") {
        Some(dir) => {
            let dir = PathBuf::from(dir);
            // A build script runs in the crate directory, so a relative path would
            // land somewhere the caller did not mean.
            assert!(
                dir.is_absolute(),
                "DF_HEADER_DIR must be an absolute path, got {dir:?}"
            );
            dir
        }
        None => PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR")).join("include"),
    };
    let header = header_dir.join("dartunifrac.h");

    let mut config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .expect("read crates/dartunifrac-capi/cbindgen.toml");

    // Emit `#define DF_VERSION "x.y.z"`, keyed off the crate version so it cannot
    // drift. cbindgen cannot evaluate `env!`, so the define is injected through
    // `after_includes` (appended to whatever is there, not replacing it).
    // `df_version()` returns the same string at runtime; a Rust test and the C
    // harness both check that the two agree.
    let version = std::env::var("CARGO_PKG_VERSION").expect("CARGO_PKG_VERSION");
    let define = format!("\n#define DF_VERSION \"{version}\"\n");
    config.after_includes = Some(match config.after_includes.take() {
        Some(existing) => format!("{existing}{define}"),
        None => define,
    });

    let bindings = cbindgen::generate_with_config(&crate_dir, config)
        .expect("cbindgen failed to generate the C header");

    std::fs::create_dir_all(&header_dir)
        .unwrap_or_else(|e| panic!("create {}: {e}", header_dir.display()));
    bindings.write_to_file(&header);

    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=build.rs");
    // Emitting any `rerun-if-*` disables Cargo's "watch the whole package"
    // default, so the injected version must be tracked explicitly: a bare
    // version bump, with no source change, would otherwise skip this script and
    // leave a stale `#define DF_VERSION` beside a newer `df_version()`.
    println!("cargo:rerun-if-env-changed=CARGO_PKG_VERSION");
    // A new header directory needs the header written there.
    println!("cargo:rerun-if-env-changed=DF_HEADER_DIR");
}
