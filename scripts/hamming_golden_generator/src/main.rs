//! Emits `crates/dartunifrac-core/tests/hamming_golden.rs`.
//!
//! The expected values come from `anndists`' nightly `std::simd` DistHamming --
//! the kernel being replaced -- so the generated test pins core's kernel to the
//! incumbent's exact f32 output. Nothing is transcribed by hand.
//!
//! The input generator is written into the emitted file verbatim from the same
//! string constant this binary uses, so the dumper and the test cannot drift.

use anndists::dist::{DistHamming, Distance};

/// Shared verbatim between this dumper and the emitted test. Any edit here
/// changes both sides at once, which is the point.
const GENERATOR: &str = r##"
/// Deterministic input pair. `b` starts as a copy of `a`, then each position is
/// redrawn with probability `p_permille/1000`, so p=0 gives identical vectors and
/// p=1000 gives independent ones. Values come from a small alphabet so that
/// coincidental matches occur too, not just the forced ones.
fn pair<T, F: Fn(u64) -> T>(len: usize, seed: u64, p_permille: u64, conv: F) -> (Vec<T>, Vec<T>) {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let a: Vec<u64> = (0..len).map(|_| next() % 251).collect();
    let b: Vec<u64> = a
        .iter()
        .map(|&x| if next() % 1000 < p_permille { next() % 251 } else { x })
        .collect();
    (
        a.into_iter().map(&conv).collect(),
        b.into_iter().map(&conv).collect(),
    )
}
"##;

// The generator, again, as real code. Kept textually identical to GENERATOR.
fn pair<T, F: Fn(u64) -> T>(len: usize, seed: u64, p_permille: u64, conv: F) -> (Vec<T>, Vec<T>) {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let a: Vec<u64> = (0..len).map(|_| next() % 251).collect();
    let b: Vec<u64> = a
        .iter()
        .map(|&x| if next() % 1000 < p_permille { next() % 251 } else { x })
        .collect();
    (
        a.into_iter().map(&conv).collect(),
        b.into_iter().map(&conv).collect(),
    )
}

/// The transform the binary applies at all four call sites, reproduced here so the
/// golden values pin the composed function and not just the Hamming step.
fn weighted_transform(d: f32) -> f32 {
    if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 }
}

fn main() {
    // Lengths chosen around the lane boundaries of every candidate width: 15/16/17
    // and 31/32/33 straddle L=16 and L=32, 255 and 4095 leave awkward residuals,
    // 2048 is the CLI default k.
    let lens: [usize; 13] = [1, 7, 15, 16, 17, 31, 32, 33, 100, 255, 2048, 4095, 4096];
    let ps: [u64; 4] = [0, 250, 500, 1000];

    let mut out = String::new();
    out.push_str(
        "// GENERATED FILE -- do not edit by hand.\n\
         //\n\
         // Expected values produced by anndists 0.1.5's nightly `std::simd` DistHamming --\n\
         // the kernel this crate replaced. They are not values this project chose; they are\n\
         // the incumbent's exact output, so any drift in the replacement breaks them.\n\
         //\n\
         // WHY THAT IS SUFFICIENT: anndists ends every width with\n\
         // `dist as f32 / va.len() as f32` over an exact u32 count -- exactly one\n\
         // floating-point operation. Any kernel that counts mismatches exactly and divides\n\
         // once is therefore bit-identical to it, whatever lane width or summation order it\n\
         // uses. These vectors turn that argument into a regression test. Values are\n\
         // compared as raw bits, not with an epsilon: the claim is byte parity.\n\
         //\n\
         // PROVENANCE. Emitting these values requires\n\
         // anndists 0.1.5 with its stdsimd feature, which requires\n\
         // `feature(portable_simd)`, which requires nightly Rust. Removing exactly that\n\
         // from the dependency graph is why this file exists, so the generator cannot live\n\
         // in this crate's dev-dependencies without undoing the work. The generator is\n\
         // checked in at `scripts/hamming_golden_generator/` instead, outside the\n\
         // workspace so a plain `cargo build` never pulls anndists in. To regenerate:\n\
         //\n\
         //     cd scripts/hamming_golden_generator\n\
         //     cargo +nightly run --release > ../../crates/dartunifrac-core/tests/hamming_golden.rs\n\
         //\n\
         // These vectors are the regression net, not the primary parity evidence. That is\n\
         // the byte-for-byte comparison of whole CLI runs against a pre-change binary.\n\n\
         use dartunifrac_core::unifrac_from_sketches;\n",
    );
    out.push_str(GENERATOR);

    for (width, conv_src) in [("u16", "|x| x as u16"), ("u32", "|x| x as u32"), ("u64", "|x| x")] {
        for (weighted, wname) in [(false, "unweighted"), (true, "weighted")] {
            out.push_str(&format!(
                "\n#[test]\nfn golden_{wname}_{width}() {{\n    \
                 // (len, seed, p_permille, expected f32 bits)\n    \
                 let cases: &[(usize, u64, u64, u32)] = &[\n"
            ));
            for &len in &lens {
                for &p in &ps {
                    let seed = 0x5DEE_CE66_D000_0000u64 ^ ((len as u64) << 8) ^ p;
                    let bits = match width {
                        "u16" => {
                            let (a, b) = pair(len, seed, p, |x| x as u16);
                            let d = DistHamming.eval(&a, &b);
                            (if weighted { weighted_transform(d) } else { d }).to_bits()
                        }
                        "u32" => {
                            let (a, b) = pair(len, seed, p, |x| x as u32);
                            let d = DistHamming.eval(&a, &b);
                            (if weighted { weighted_transform(d) } else { d }).to_bits()
                        }
                        _ => {
                            let (a, b) = pair(len, seed, p, |x| x);
                            let d = DistHamming.eval(&a, &b);
                            (if weighted { weighted_transform(d) } else { d }).to_bits()
                        }
                    };
                    out.push_str(&format!("        ({len}, {seed:#018x}, {p}, {bits:#010x}),\n"));
                }
            }
            out.push_str(&format!(
                "    ];\n    for &(len, seed, p, want) in cases {{\n        \
                 let (a, b) = pair::<{width}, _>(len, seed, p, {conv_src});\n        \
                 let got = unifrac_from_sketches(&a, &b, {weighted});\n        \
                 assert_eq!(\n            got.to_bits(),\n            want,\n            \
                 \"{width} {wname} len={{len}} p={{p}}: got {{got}} ({{:#010x}}) want {{:#010x}}\",\n            \
                 got.to_bits(),\n            want\n        );\n    }}\n}}\n"
            ));
        }
    }

    // The kernel folds its lane counters into a usize every LANES * u16::MAX = 32 *
    // 65535 positions. The table above stops at 4096, so without this nothing past a
    // single block has an expected value traceable to the incumbent kernel -- only
    // the hand-derived all-mismatch invariant. Two lengths either side of a block
    // boundary, with the SAME generator and densities, so the fold is covered at
    // realistic mismatch rates rather than only at 100%.
    out.push_str(
        "\n/// Straddles a lane-counter block boundary: the kernel folds its counters into a\n\
         /// usize every 32 * 65535 = 2,097,120 positions, so 2,097,120 is the last length\n\
         /// handled by a single block and 2,097,121 the first that needs two. anndists has\n\
         /// no such structure at all, so agreeing with it here is what shows the blocking\n\
         /// is invisible in the output.\n\
         #[test]\n\
         fn golden_across_a_counter_block_boundary_u16() {\n    \
         // (len, seed, p_permille, unweighted bits, weighted bits)\n    \
         let cases: &[(usize, u64, u64, u32, u32)] = &[\n",
    );
    for len in [32 * 65_535usize, 32 * 65_535 + 1] {
        for p in [250u64, 1000] {
            let seed = 0x5DEE_CE66_D000_0000u64 ^ ((len as u64) << 8) ^ p;
            let (a, b) = pair(len, seed, p, |x| x as u16);
            let d = DistHamming.eval(&a, &b);
            out.push_str(&format!(
                "        ({len}, {seed:#018x}, {p}, {:#010x}, {:#010x}),\n",
                d.to_bits(),
                weighted_transform(d).to_bits()
            ));
        }
    }
    out.push_str(
        "    ];\n    for &(len, seed, p, want_u, want_w) in cases {\n        \
         let (a, b) = pair::<u16, _>(len, seed, p, |x| x as u16);\n        \
         assert_eq!(\n            unifrac_from_sketches(&a, &b, false).to_bits(),\n            \
         want_u,\n            \"unweighted len={len} p={p}\"\n        );\n        \
         assert_eq!(\n            unifrac_from_sketches(&a, &b, true).to_bits(),\n            \
         want_w,\n            \"weighted len={len} p={p}\"\n        );\n    }\n}\n",
    );

    // Report what the goldens actually cover, so a reader can see they are not all
    // the same trivial value.
    let mut distinct = std::collections::BTreeSet::new();
    for &len in &lens {
        for &p in &ps {
            let seed = 0x5DEE_CE66_D000_0000u64 ^ ((len as u64) << 8) ^ p;
            let (a, b) = pair(len, seed, p, |x| x as u16);
            distinct.insert(DistHamming.eval(&a, &b).to_bits());
        }
    }
    eprintln!(
        "u16 goldens: {} cases, {} distinct expected values",
        lens.len() * ps.len(),
        distinct.len()
    );
    print!("{out}");
}
