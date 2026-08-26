// GENERATED FILE -- do not edit by hand.
//
// Expected values produced by anndists 0.1.5's nightly `std::simd` DistHamming --
// the kernel this crate replaced. They are not values this project chose; they are
// the incumbent's exact output, so any drift in the replacement breaks them.
//
// WHY THAT IS SUFFICIENT: anndists ends every width with
// `dist as f32 / va.len() as f32` over an exact u32 count -- exactly one
// floating-point operation. Any kernel that counts mismatches exactly and divides
// once is therefore bit-identical to it, whatever lane width or summation order it
// uses. These vectors turn that argument into a regression test. Values are
// compared as raw bits, not with an epsilon: the claim is byte parity.
//
// PROVENANCE. Emitting these values requires
// anndists 0.1.5 with its stdsimd feature, which requires
// `feature(portable_simd)`, which requires nightly Rust. Removing exactly that
// from the dependency graph is why this file exists, so the generator cannot live
// in this crate's dev-dependencies without undoing the work. The generator is
// checked in at `scripts/hamming_golden_generator/` instead, outside the
// workspace so a plain `cargo build` never pulls anndists in. To regenerate:
//
//     cd scripts/hamming_golden_generator
//     cargo +nightly run --release > ../../crates/dartunifrac-core/tests/hamming_golden.rs
//
// These vectors are the regression net, not the primary parity evidence. That is
// the byte-for-byte comparison of whole CLI runs against a pre-change binary.

use dartunifrac_core::unifrac_from_sketches;

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

#[test]
fn golden_unweighted_u16() {
    // (len, seed, p_permille, expected f32 bits)
    let cases: &[(usize, u64, u64, u32)] = &[
        (1, 0x5deece66d0000100, 0, 0x00000000),
        (1, 0x5deece66d00001fa, 250, 0x3f800000),
        (1, 0x5deece66d00000f4, 500, 0x3f800000),
        (1, 0x5deece66d00002e8, 1000, 0x3f800000),
        (7, 0x5deece66d0000700, 0, 0x00000000),
        (7, 0x5deece66d00007fa, 250, 0x3e124925),
        (7, 0x5deece66d00006f4, 500, 0x3e124925),
        (7, 0x5deece66d00004e8, 1000, 0x3f800000),
        (15, 0x5deece66d0000f00, 0, 0x00000000),
        (15, 0x5deece66d0000ffa, 250, 0x3e4ccccd),
        (15, 0x5deece66d0000ef4, 500, 0x3eeeeeef),
        (15, 0x5deece66d0000ce8, 1000, 0x3f800000),
        (16, 0x5deece66d0001000, 0, 0x00000000),
        (16, 0x5deece66d00010fa, 250, 0x3e000000),
        (16, 0x5deece66d00011f4, 500, 0x3f400000),
        (16, 0x5deece66d00013e8, 1000, 0x3f800000),
        (17, 0x5deece66d0001100, 0, 0x00000000),
        (17, 0x5deece66d00011fa, 250, 0x3e34b4b5),
        (17, 0x5deece66d00010f4, 500, 0x3e969697),
        (17, 0x5deece66d00012e8, 1000, 0x3f800000),
        (31, 0x5deece66d0001f00, 0, 0x00000000),
        (31, 0x5deece66d0001ffa, 250, 0x3e46318c),
        (31, 0x5deece66d0001ef4, 500, 0x3ee739ce),
        (31, 0x5deece66d0001ce8, 1000, 0x3f800000),
        (32, 0x5deece66d0002000, 0, 0x00000000),
        (32, 0x5deece66d00020fa, 250, 0x3ea00000),
        (32, 0x5deece66d00021f4, 500, 0x3f000000),
        (32, 0x5deece66d00023e8, 1000, 0x3f800000),
        (33, 0x5deece66d0002100, 0, 0x00000000),
        (33, 0x5deece66d00021fa, 250, 0x3e9b26ca),
        (33, 0x5deece66d00020f4, 500, 0x3eba2e8c),
        (33, 0x5deece66d00022e8, 1000, 0x3f800000),
        (100, 0x5deece66d0006400, 0, 0x00000000),
        (100, 0x5deece66d00064fa, 250, 0x3e75c28f),
        (100, 0x5deece66d00065f4, 500, 0x3f0ccccd),
        (100, 0x5deece66d00067e8, 1000, 0x3f800000),
        (255, 0x5deece66d000ff00, 0, 0x00000000),
        (255, 0x5deece66d000fffa, 250, 0x3e8c8c8d),
        (255, 0x5deece66d000fef4, 500, 0x3ef8f8f9),
        (255, 0x5deece66d000fce8, 1000, 0x3f800000),
        (2048, 0x5deece66d0080000, 0, 0x00000000),
        (2048, 0x5deece66d00800fa, 250, 0x3e7c0000),
        (2048, 0x5deece66d00801f4, 500, 0x3f012000),
        (2048, 0x5deece66d00803e8, 1000, 0x3f7f2000),
        (4095, 0x5deece66d00fff00, 0, 0x00000000),
        (4095, 0x5deece66d00ffffa, 250, 0x3e784f85),
        (4095, 0x5deece66d00ffef4, 500, 0x3f026827),
        (4095, 0x5deece66d00ffce8, 1000, 0x3f7eefef),
        (4096, 0x5deece66d0100000, 0, 0x00000000),
        (4096, 0x5deece66d01000fa, 250, 0x3e6e8000),
        (4096, 0x5deece66d01001f4, 500, 0x3eff2000),
        (4096, 0x5deece66d01003e8, 1000, 0x3f7f6000),
    ];
    for &(len, seed, p, want) in cases {
        let (a, b) = pair::<u16, _>(len, seed, p, |x| x as u16);
        let got = unifrac_from_sketches(&a, &b, false);
        assert_eq!(
            got.to_bits(),
            want,
            "u16 unweighted len={len} p={p}: got {got} ({:#010x}) want {:#010x}",
            got.to_bits(),
            want
        );
    }
}

#[test]
fn golden_weighted_u16() {
    // (len, seed, p_permille, expected f32 bits)
    let cases: &[(usize, u64, u64, u32)] = &[
        (1, 0x5deece66d0000100, 0, 0x00000000),
        (1, 0x5deece66d00001fa, 250, 0x3f800000),
        (1, 0x5deece66d00000f4, 500, 0x3f800000),
        (1, 0x5deece66d00002e8, 1000, 0x3f800000),
        (7, 0x5deece66d0000700, 0, 0x00000000),
        (7, 0x5deece66d00007fa, 250, 0x3d9d89d9),
        (7, 0x5deece66d00006f4, 500, 0x3d9d89d9),
        (7, 0x5deece66d00004e8, 1000, 0x3f800000),
        (15, 0x5deece66d0000f00, 0, 0x00000000),
        (15, 0x5deece66d0000ffa, 250, 0x3de38e3a),
        (15, 0x5deece66d0000ef4, 500, 0x3e9bd37b),
        (15, 0x5deece66d0000ce8, 1000, 0x3f800000),
        (16, 0x5deece66d0001000, 0, 0x00000000),
        (16, 0x5deece66d00010fa, 250, 0x3d888889),
        (16, 0x5deece66d00011f4, 500, 0x3f19999a),
        (16, 0x5deece66d00013e8, 1000, 0x3f800000),
        (17, 0x5deece66d0001100, 0, 0x00000000),
        (17, 0x5deece66d00011fa, 250, 0x3dc6318d),
        (17, 0x5deece66d00010f4, 500, 0x3e308d3f),
        (17, 0x5deece66d00012e8, 1000, 0x3f800000),
        (31, 0x5deece66d0001f00, 0, 0x00000000),
        (31, 0x5deece66d0001ffa, 250, 0x3ddb6db7),
        (31, 0x5deece66d0001ef4, 500, 0x3e955555),
        (31, 0x5deece66d0001ce8, 1000, 0x3f800000),
        (32, 0x5deece66d0002000, 0, 0x00000000),
        (32, 0x5deece66d00020fa, 250, 0x3e3da12f),
        (32, 0x5deece66d00021f4, 500, 0x3eaaaaab),
        (32, 0x5deece66d00023e8, 1000, 0x3f800000),
        (33, 0x5deece66d0002100, 0, 0x00000000),
        (33, 0x5deece66d00021fa, 250, 0x3e36db6e),
        (33, 0x5deece66d00020f4, 500, 0x3e638e39),
        (33, 0x5deece66d00022e8, 1000, 0x3f800000),
        (100, 0x5deece66d0006400, 0, 0x00000000),
        (100, 0x5deece66d00064fa, 250, 0x3e0ba2e9),
        (100, 0x5deece66d00065f4, 500, 0x3ec234f7),
        (100, 0x5deece66d00067e8, 1000, 0x3f800000),
        (255, 0x5deece66d000ff00, 0, 0x00000000),
        (255, 0x5deece66d000fffa, 250, 0x3e22e8bb),
        (255, 0x5deece66d000fef4, 500, 0x3ea47a08),
        (255, 0x5deece66d000fce8, 1000, 0x3f800000),
        (2048, 0x5deece66d0080000, 0, 0x00000000),
        (2048, 0x5deece66d00800fa, 250, 0x3e0fade6),
        (2048, 0x5deece66d00801f4, 500, 0x3eacac2c),
        (2048, 0x5deece66d00803e8, 1000, 0x3f7e4187),
        (4095, 0x5deece66d00fff00, 0, 0x00000000),
        (4095, 0x5deece66d00ffffa, 250, 0x3e0d4919),
        (4095, 0x5deece66d00ffef4, 500, 0x3eaef8f6),
        (4095, 0x5deece66d00ffce8, 1000, 0x3f7de21f),
        (4096, 0x5deece66d0100000, 0, 0x00000000),
        (4096, 0x5deece66d01000fa, 250, 0x3e06f7b9),
        (4096, 0x5deece66d01001f4, 500, 0x3ea9e3c8),
        (4096, 0x5deece66d01003e8, 1000, 0x3f7ec0c8),
    ];
    for &(len, seed, p, want) in cases {
        let (a, b) = pair::<u16, _>(len, seed, p, |x| x as u16);
        let got = unifrac_from_sketches(&a, &b, true);
        assert_eq!(
            got.to_bits(),
            want,
            "u16 weighted len={len} p={p}: got {got} ({:#010x}) want {:#010x}",
            got.to_bits(),
            want
        );
    }
}

#[test]
fn golden_unweighted_u32() {
    // (len, seed, p_permille, expected f32 bits)
    let cases: &[(usize, u64, u64, u32)] = &[
        (1, 0x5deece66d0000100, 0, 0x00000000),
        (1, 0x5deece66d00001fa, 250, 0x3f800000),
        (1, 0x5deece66d00000f4, 500, 0x3f800000),
        (1, 0x5deece66d00002e8, 1000, 0x3f800000),
        (7, 0x5deece66d0000700, 0, 0x00000000),
        (7, 0x5deece66d00007fa, 250, 0x3e124925),
        (7, 0x5deece66d00006f4, 500, 0x3e124925),
        (7, 0x5deece66d00004e8, 1000, 0x3f800000),
        (15, 0x5deece66d0000f00, 0, 0x00000000),
        (15, 0x5deece66d0000ffa, 250, 0x3e4ccccd),
        (15, 0x5deece66d0000ef4, 500, 0x3eeeeeef),
        (15, 0x5deece66d0000ce8, 1000, 0x3f800000),
        (16, 0x5deece66d0001000, 0, 0x00000000),
        (16, 0x5deece66d00010fa, 250, 0x3e000000),
        (16, 0x5deece66d00011f4, 500, 0x3f400000),
        (16, 0x5deece66d00013e8, 1000, 0x3f800000),
        (17, 0x5deece66d0001100, 0, 0x00000000),
        (17, 0x5deece66d00011fa, 250, 0x3e34b4b5),
        (17, 0x5deece66d00010f4, 500, 0x3e969697),
        (17, 0x5deece66d00012e8, 1000, 0x3f800000),
        (31, 0x5deece66d0001f00, 0, 0x00000000),
        (31, 0x5deece66d0001ffa, 250, 0x3e46318c),
        (31, 0x5deece66d0001ef4, 500, 0x3ee739ce),
        (31, 0x5deece66d0001ce8, 1000, 0x3f800000),
        (32, 0x5deece66d0002000, 0, 0x00000000),
        (32, 0x5deece66d00020fa, 250, 0x3ea00000),
        (32, 0x5deece66d00021f4, 500, 0x3f000000),
        (32, 0x5deece66d00023e8, 1000, 0x3f800000),
        (33, 0x5deece66d0002100, 0, 0x00000000),
        (33, 0x5deece66d00021fa, 250, 0x3e9b26ca),
        (33, 0x5deece66d00020f4, 500, 0x3eba2e8c),
        (33, 0x5deece66d00022e8, 1000, 0x3f800000),
        (100, 0x5deece66d0006400, 0, 0x00000000),
        (100, 0x5deece66d00064fa, 250, 0x3e75c28f),
        (100, 0x5deece66d00065f4, 500, 0x3f0ccccd),
        (100, 0x5deece66d00067e8, 1000, 0x3f800000),
        (255, 0x5deece66d000ff00, 0, 0x00000000),
        (255, 0x5deece66d000fffa, 250, 0x3e8c8c8d),
        (255, 0x5deece66d000fef4, 500, 0x3ef8f8f9),
        (255, 0x5deece66d000fce8, 1000, 0x3f800000),
        (2048, 0x5deece66d0080000, 0, 0x00000000),
        (2048, 0x5deece66d00800fa, 250, 0x3e7c0000),
        (2048, 0x5deece66d00801f4, 500, 0x3f012000),
        (2048, 0x5deece66d00803e8, 1000, 0x3f7f2000),
        (4095, 0x5deece66d00fff00, 0, 0x00000000),
        (4095, 0x5deece66d00ffffa, 250, 0x3e784f85),
        (4095, 0x5deece66d00ffef4, 500, 0x3f026827),
        (4095, 0x5deece66d00ffce8, 1000, 0x3f7eefef),
        (4096, 0x5deece66d0100000, 0, 0x00000000),
        (4096, 0x5deece66d01000fa, 250, 0x3e6e8000),
        (4096, 0x5deece66d01001f4, 500, 0x3eff2000),
        (4096, 0x5deece66d01003e8, 1000, 0x3f7f6000),
    ];
    for &(len, seed, p, want) in cases {
        let (a, b) = pair::<u32, _>(len, seed, p, |x| x as u32);
        let got = unifrac_from_sketches(&a, &b, false);
        assert_eq!(
            got.to_bits(),
            want,
            "u32 unweighted len={len} p={p}: got {got} ({:#010x}) want {:#010x}",
            got.to_bits(),
            want
        );
    }
}

#[test]
fn golden_weighted_u32() {
    // (len, seed, p_permille, expected f32 bits)
    let cases: &[(usize, u64, u64, u32)] = &[
        (1, 0x5deece66d0000100, 0, 0x00000000),
        (1, 0x5deece66d00001fa, 250, 0x3f800000),
        (1, 0x5deece66d00000f4, 500, 0x3f800000),
        (1, 0x5deece66d00002e8, 1000, 0x3f800000),
        (7, 0x5deece66d0000700, 0, 0x00000000),
        (7, 0x5deece66d00007fa, 250, 0x3d9d89d9),
        (7, 0x5deece66d00006f4, 500, 0x3d9d89d9),
        (7, 0x5deece66d00004e8, 1000, 0x3f800000),
        (15, 0x5deece66d0000f00, 0, 0x00000000),
        (15, 0x5deece66d0000ffa, 250, 0x3de38e3a),
        (15, 0x5deece66d0000ef4, 500, 0x3e9bd37b),
        (15, 0x5deece66d0000ce8, 1000, 0x3f800000),
        (16, 0x5deece66d0001000, 0, 0x00000000),
        (16, 0x5deece66d00010fa, 250, 0x3d888889),
        (16, 0x5deece66d00011f4, 500, 0x3f19999a),
        (16, 0x5deece66d00013e8, 1000, 0x3f800000),
        (17, 0x5deece66d0001100, 0, 0x00000000),
        (17, 0x5deece66d00011fa, 250, 0x3dc6318d),
        (17, 0x5deece66d00010f4, 500, 0x3e308d3f),
        (17, 0x5deece66d00012e8, 1000, 0x3f800000),
        (31, 0x5deece66d0001f00, 0, 0x00000000),
        (31, 0x5deece66d0001ffa, 250, 0x3ddb6db7),
        (31, 0x5deece66d0001ef4, 500, 0x3e955555),
        (31, 0x5deece66d0001ce8, 1000, 0x3f800000),
        (32, 0x5deece66d0002000, 0, 0x00000000),
        (32, 0x5deece66d00020fa, 250, 0x3e3da12f),
        (32, 0x5deece66d00021f4, 500, 0x3eaaaaab),
        (32, 0x5deece66d00023e8, 1000, 0x3f800000),
        (33, 0x5deece66d0002100, 0, 0x00000000),
        (33, 0x5deece66d00021fa, 250, 0x3e36db6e),
        (33, 0x5deece66d00020f4, 500, 0x3e638e39),
        (33, 0x5deece66d00022e8, 1000, 0x3f800000),
        (100, 0x5deece66d0006400, 0, 0x00000000),
        (100, 0x5deece66d00064fa, 250, 0x3e0ba2e9),
        (100, 0x5deece66d00065f4, 500, 0x3ec234f7),
        (100, 0x5deece66d00067e8, 1000, 0x3f800000),
        (255, 0x5deece66d000ff00, 0, 0x00000000),
        (255, 0x5deece66d000fffa, 250, 0x3e22e8bb),
        (255, 0x5deece66d000fef4, 500, 0x3ea47a08),
        (255, 0x5deece66d000fce8, 1000, 0x3f800000),
        (2048, 0x5deece66d0080000, 0, 0x00000000),
        (2048, 0x5deece66d00800fa, 250, 0x3e0fade6),
        (2048, 0x5deece66d00801f4, 500, 0x3eacac2c),
        (2048, 0x5deece66d00803e8, 1000, 0x3f7e4187),
        (4095, 0x5deece66d00fff00, 0, 0x00000000),
        (4095, 0x5deece66d00ffffa, 250, 0x3e0d4919),
        (4095, 0x5deece66d00ffef4, 500, 0x3eaef8f6),
        (4095, 0x5deece66d00ffce8, 1000, 0x3f7de21f),
        (4096, 0x5deece66d0100000, 0, 0x00000000),
        (4096, 0x5deece66d01000fa, 250, 0x3e06f7b9),
        (4096, 0x5deece66d01001f4, 500, 0x3ea9e3c8),
        (4096, 0x5deece66d01003e8, 1000, 0x3f7ec0c8),
    ];
    for &(len, seed, p, want) in cases {
        let (a, b) = pair::<u32, _>(len, seed, p, |x| x as u32);
        let got = unifrac_from_sketches(&a, &b, true);
        assert_eq!(
            got.to_bits(),
            want,
            "u32 weighted len={len} p={p}: got {got} ({:#010x}) want {:#010x}",
            got.to_bits(),
            want
        );
    }
}

#[test]
fn golden_unweighted_u64() {
    // (len, seed, p_permille, expected f32 bits)
    let cases: &[(usize, u64, u64, u32)] = &[
        (1, 0x5deece66d0000100, 0, 0x00000000),
        (1, 0x5deece66d00001fa, 250, 0x3f800000),
        (1, 0x5deece66d00000f4, 500, 0x3f800000),
        (1, 0x5deece66d00002e8, 1000, 0x3f800000),
        (7, 0x5deece66d0000700, 0, 0x00000000),
        (7, 0x5deece66d00007fa, 250, 0x3e124925),
        (7, 0x5deece66d00006f4, 500, 0x3e124925),
        (7, 0x5deece66d00004e8, 1000, 0x3f800000),
        (15, 0x5deece66d0000f00, 0, 0x00000000),
        (15, 0x5deece66d0000ffa, 250, 0x3e4ccccd),
        (15, 0x5deece66d0000ef4, 500, 0x3eeeeeef),
        (15, 0x5deece66d0000ce8, 1000, 0x3f800000),
        (16, 0x5deece66d0001000, 0, 0x00000000),
        (16, 0x5deece66d00010fa, 250, 0x3e000000),
        (16, 0x5deece66d00011f4, 500, 0x3f400000),
        (16, 0x5deece66d00013e8, 1000, 0x3f800000),
        (17, 0x5deece66d0001100, 0, 0x00000000),
        (17, 0x5deece66d00011fa, 250, 0x3e34b4b5),
        (17, 0x5deece66d00010f4, 500, 0x3e969697),
        (17, 0x5deece66d00012e8, 1000, 0x3f800000),
        (31, 0x5deece66d0001f00, 0, 0x00000000),
        (31, 0x5deece66d0001ffa, 250, 0x3e46318c),
        (31, 0x5deece66d0001ef4, 500, 0x3ee739ce),
        (31, 0x5deece66d0001ce8, 1000, 0x3f800000),
        (32, 0x5deece66d0002000, 0, 0x00000000),
        (32, 0x5deece66d00020fa, 250, 0x3ea00000),
        (32, 0x5deece66d00021f4, 500, 0x3f000000),
        (32, 0x5deece66d00023e8, 1000, 0x3f800000),
        (33, 0x5deece66d0002100, 0, 0x00000000),
        (33, 0x5deece66d00021fa, 250, 0x3e9b26ca),
        (33, 0x5deece66d00020f4, 500, 0x3eba2e8c),
        (33, 0x5deece66d00022e8, 1000, 0x3f800000),
        (100, 0x5deece66d0006400, 0, 0x00000000),
        (100, 0x5deece66d00064fa, 250, 0x3e75c28f),
        (100, 0x5deece66d00065f4, 500, 0x3f0ccccd),
        (100, 0x5deece66d00067e8, 1000, 0x3f800000),
        (255, 0x5deece66d000ff00, 0, 0x00000000),
        (255, 0x5deece66d000fffa, 250, 0x3e8c8c8d),
        (255, 0x5deece66d000fef4, 500, 0x3ef8f8f9),
        (255, 0x5deece66d000fce8, 1000, 0x3f800000),
        (2048, 0x5deece66d0080000, 0, 0x00000000),
        (2048, 0x5deece66d00800fa, 250, 0x3e7c0000),
        (2048, 0x5deece66d00801f4, 500, 0x3f012000),
        (2048, 0x5deece66d00803e8, 1000, 0x3f7f2000),
        (4095, 0x5deece66d00fff00, 0, 0x00000000),
        (4095, 0x5deece66d00ffffa, 250, 0x3e784f85),
        (4095, 0x5deece66d00ffef4, 500, 0x3f026827),
        (4095, 0x5deece66d00ffce8, 1000, 0x3f7eefef),
        (4096, 0x5deece66d0100000, 0, 0x00000000),
        (4096, 0x5deece66d01000fa, 250, 0x3e6e8000),
        (4096, 0x5deece66d01001f4, 500, 0x3eff2000),
        (4096, 0x5deece66d01003e8, 1000, 0x3f7f6000),
    ];
    for &(len, seed, p, want) in cases {
        let (a, b) = pair::<u64, _>(len, seed, p, |x| x);
        let got = unifrac_from_sketches(&a, &b, false);
        assert_eq!(
            got.to_bits(),
            want,
            "u64 unweighted len={len} p={p}: got {got} ({:#010x}) want {:#010x}",
            got.to_bits(),
            want
        );
    }
}

#[test]
fn golden_weighted_u64() {
    // (len, seed, p_permille, expected f32 bits)
    let cases: &[(usize, u64, u64, u32)] = &[
        (1, 0x5deece66d0000100, 0, 0x00000000),
        (1, 0x5deece66d00001fa, 250, 0x3f800000),
        (1, 0x5deece66d00000f4, 500, 0x3f800000),
        (1, 0x5deece66d00002e8, 1000, 0x3f800000),
        (7, 0x5deece66d0000700, 0, 0x00000000),
        (7, 0x5deece66d00007fa, 250, 0x3d9d89d9),
        (7, 0x5deece66d00006f4, 500, 0x3d9d89d9),
        (7, 0x5deece66d00004e8, 1000, 0x3f800000),
        (15, 0x5deece66d0000f00, 0, 0x00000000),
        (15, 0x5deece66d0000ffa, 250, 0x3de38e3a),
        (15, 0x5deece66d0000ef4, 500, 0x3e9bd37b),
        (15, 0x5deece66d0000ce8, 1000, 0x3f800000),
        (16, 0x5deece66d0001000, 0, 0x00000000),
        (16, 0x5deece66d00010fa, 250, 0x3d888889),
        (16, 0x5deece66d00011f4, 500, 0x3f19999a),
        (16, 0x5deece66d00013e8, 1000, 0x3f800000),
        (17, 0x5deece66d0001100, 0, 0x00000000),
        (17, 0x5deece66d00011fa, 250, 0x3dc6318d),
        (17, 0x5deece66d00010f4, 500, 0x3e308d3f),
        (17, 0x5deece66d00012e8, 1000, 0x3f800000),
        (31, 0x5deece66d0001f00, 0, 0x00000000),
        (31, 0x5deece66d0001ffa, 250, 0x3ddb6db7),
        (31, 0x5deece66d0001ef4, 500, 0x3e955555),
        (31, 0x5deece66d0001ce8, 1000, 0x3f800000),
        (32, 0x5deece66d0002000, 0, 0x00000000),
        (32, 0x5deece66d00020fa, 250, 0x3e3da12f),
        (32, 0x5deece66d00021f4, 500, 0x3eaaaaab),
        (32, 0x5deece66d00023e8, 1000, 0x3f800000),
        (33, 0x5deece66d0002100, 0, 0x00000000),
        (33, 0x5deece66d00021fa, 250, 0x3e36db6e),
        (33, 0x5deece66d00020f4, 500, 0x3e638e39),
        (33, 0x5deece66d00022e8, 1000, 0x3f800000),
        (100, 0x5deece66d0006400, 0, 0x00000000),
        (100, 0x5deece66d00064fa, 250, 0x3e0ba2e9),
        (100, 0x5deece66d00065f4, 500, 0x3ec234f7),
        (100, 0x5deece66d00067e8, 1000, 0x3f800000),
        (255, 0x5deece66d000ff00, 0, 0x00000000),
        (255, 0x5deece66d000fffa, 250, 0x3e22e8bb),
        (255, 0x5deece66d000fef4, 500, 0x3ea47a08),
        (255, 0x5deece66d000fce8, 1000, 0x3f800000),
        (2048, 0x5deece66d0080000, 0, 0x00000000),
        (2048, 0x5deece66d00800fa, 250, 0x3e0fade6),
        (2048, 0x5deece66d00801f4, 500, 0x3eacac2c),
        (2048, 0x5deece66d00803e8, 1000, 0x3f7e4187),
        (4095, 0x5deece66d00fff00, 0, 0x00000000),
        (4095, 0x5deece66d00ffffa, 250, 0x3e0d4919),
        (4095, 0x5deece66d00ffef4, 500, 0x3eaef8f6),
        (4095, 0x5deece66d00ffce8, 1000, 0x3f7de21f),
        (4096, 0x5deece66d0100000, 0, 0x00000000),
        (4096, 0x5deece66d01000fa, 250, 0x3e06f7b9),
        (4096, 0x5deece66d01001f4, 500, 0x3ea9e3c8),
        (4096, 0x5deece66d01003e8, 1000, 0x3f7ec0c8),
    ];
    for &(len, seed, p, want) in cases {
        let (a, b) = pair::<u64, _>(len, seed, p, |x| x);
        let got = unifrac_from_sketches(&a, &b, true);
        assert_eq!(
            got.to_bits(),
            want,
            "u64 weighted len={len} p={p}: got {got} ({:#010x}) want {:#010x}",
            got.to_bits(),
            want
        );
    }
}

/// Straddles a lane-counter block boundary: the kernel folds its counters into a
/// usize every 32 * 65535 = 2,097,120 positions, so 2,097,120 is the last length
/// handled by a single block and 2,097,121 the first that needs two. anndists has
/// no such structure at all, so agreeing with it here is what shows the blocking
/// is invisible in the output.
#[test]
fn golden_across_a_counter_block_boundary_u16() {
    // (len, seed, p_permille, unweighted bits, weighted bits)
    let cases: &[(usize, u64, u64, u32, u32)] = &[
        (2097120, 0x5deece66cfffe0fa, 250, 0x3e7e991f, 0x3e115ef5),
        (2097120, 0x5deece66cfffe3e8, 1000, 0x3f7efa77, 0x3f7df703),
        (2097121, 0x5deece66cfffe1fa, 250, 0x3e7f01b7, 0x3e11a32c),
        (2097121, 0x5deece66cfffe2e8, 1000, 0x3f7efe57, 0x3f7dfeb4),
    ];
    for &(len, seed, p, want_u, want_w) in cases {
        let (a, b) = pair::<u16, _>(len, seed, p, |x| x as u16);
        assert_eq!(
            unifrac_from_sketches(&a, &b, false).to_bits(),
            want_u,
            "unweighted len={len} p={p}"
        );
        assert_eq!(
            unifrac_from_sketches(&a, &b, true).to_bits(),
            want_w,
            "weighted len={len} p={p}"
        );
    }
}
