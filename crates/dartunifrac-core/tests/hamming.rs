//! Intent tests for the sketch-distance kernel.
//!
//! `tests/hamming_golden.rs` pins the kernel's f32 output to `anndists`' nightly
//! `std::simd` DistHamming, which is what M2 replaced. Those vectors prove parity
//! but they do not say *why* any particular behaviour is right, and they cannot
//! localize a fault. These tests carry the reasoning: what the transform means,
//! which branch exists for which input, and where the counter can wrap.

use dartunifrac_core::unifrac_from_sketches;

/// Sketches of different lengths are a caller bug, not a distance of some kind.
/// The kernel this replaced asserted (verified: it panics rather than truncating
/// to the shorter slice), and silently comparing a prefix would return a
/// plausible number for mismatched sketch sizes -- the worst possible failure for
/// an API whose entire contract is that two sketch sets are comparable.
#[test]
#[should_panic(expected = "sketches of different lengths are not comparable")]
fn sketches_of_different_lengths_panic_rather_than_comparing_a_prefix() {
    let a: [u16; 4] = [1, 2, 3, 4];
    let b: [u16; 3] = [1, 2, 3];
    unifrac_from_sketches(&a, &b, false);
}

/// Empty sketches divide 0 by 0. The incumbent returned NaN there, and weighted
/// mode turned that NaN into 1.0 -- because `NaN < 2.0` is false, so the
/// transform takes its `else` branch.
///
/// This is preserved deliberately rather than "fixed". It is also the ONLY way to
/// reach that `else` branch: `d` is a count over a length, so `d <= 1.0` for every
/// non-empty input, and at `d == 1.0` the transform yields `1.0/(2.0-1.0) == 1.0`
/// through the normal path. A future reader who deletes the branch as unreachable
/// dead code will change empty-sketch behaviour, and this test is what tells them.
#[test]
fn empty_sketches_give_nan_unweighted_and_one_weighted() {
    let e: [u16; 0] = [];
    assert!(
        unifrac_from_sketches(&e, &e, false).is_nan(),
        "unweighted empty sketches must stay NaN (0/0), as anndists returned"
    );
    assert_eq!(
        unifrac_from_sketches(&e, &e, true).to_bits(),
        1.0f32.to_bits(),
        "weighted empty sketches must be 1.0: NaN < 2.0 is false, so the transform's else branch"
    );
    // The same value through the ordinary path, showing the else branch is not
    // what produces 1.0 for a real all-mismatch pair.
    let a: [u16; 4] = [1, 2, 3, 4];
    let b: [u16; 4] = [9, 9, 9, 9];
    assert_eq!(unifrac_from_sketches(&a, &b, true).to_bits(), 1.0f32.to_bits());
}

/// A sample against itself has no unique branch mass, so UniFrac is 0 in both modes.
#[test]
fn identical_sketches_are_zero_in_both_modes() {
    let a: Vec<u32> = (0..2048u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
    assert_eq!(unifrac_from_sketches(&a, &a, false).to_bits(), 0.0f32.to_bits());
    assert_eq!(unifrac_from_sketches(&a, &a, true).to_bits(), 0.0f32.to_bits());
}

/// The weighted transform is `d_J / (2 - d_J)`, which is what turns a Jaccard
/// distance into normalized weighted UniFrac: `(1 - J_w)/(1 + J_w)`. Pinned at a
/// value where the arithmetic is checkable by eye -- half the slots differing
/// gives 0.5, and 0.5/1.5 is 1/3 -- so a sign slip or an inverted numerator
/// cannot pass.
#[test]
fn the_weighted_transform_maps_jaccard_distance_to_normalized_unifrac() {
    let k = 2048;
    let a: Vec<u16> = (0..k).map(|i| i as u16).collect();
    // Exactly half the positions differ.
    let b: Vec<u16> = (0..k).map(|i| if i % 2 == 0 { i as u16 } else { 0xFFFF }).collect();
    let d = unifrac_from_sketches(&a, &b, false);
    assert_eq!(d.to_bits(), 0.5f32.to_bits(), "half the slots differ, so d_J = 0.5");
    assert_eq!(
        unifrac_from_sketches(&a, &b, true).to_bits(),
        (0.5f32 / 1.5f32).to_bits(),
        "d/(2-d) at d=0.5 is 1/3"
    );
}

/// Residual handling: a mismatch that falls in the tail past the last full lane
/// chunk must still be counted. Lengths straddle the lane counts any reasonable
/// implementation would pick (16 and 32), and the single mismatch is placed at the
/// LAST position, which is in the residual for every length that is not a multiple
/// of the lane count. Drop the residual loop and these lengths report 0.
#[test]
fn a_mismatch_in_the_residual_tail_is_counted() {
    for len in [1usize, 7, 15, 16, 17, 31, 32, 33, 47, 63, 65, 100, 255, 2047, 2049] {
        let a: Vec<u16> = (0..len).map(|i| i as u16).collect();
        let mut b = a.clone();
        b[len - 1] ^= 1;
        let got = unifrac_from_sketches(&a, &b, false);
        assert_eq!(
            got.to_bits(),
            (1.0f32 / len as f32).to_bits(),
            "len={len}: exactly one mismatch, at the last position, so d = 1/{len} (got {got})"
        );
    }
}

/// Narrow lane counters are what make the kernel fast, and they are also the one
/// way it could silently produce a wrong answer: a counter that gains at most 1 per
/// chunk wraps once the chunk count exceeds its range. The kernel forecloses that
/// by construction rather than by guarding -- it processes the input in blocks of
/// `LANES * u16::MAX` positions and folds each block into a `usize`, so no counter
/// can reach its limit at any input length. One code path, no fast-path/fallback
/// split, nothing to get the threshold of wrong.
///
/// These lengths straddle that block boundary (32 lanes x 65535 = 2,097,120) and
/// then cross it, with every position differing -- the worst case, where each
/// counter climbs as fast as it possibly can. Break the blocking and a counter
/// wraps, losing 65536 from the count and moving the distance by ~3%. That is
/// invisible without a test at this scale, and in release it is silent, because
/// arithmetic overflow does not panic there.
#[test]
fn counters_cannot_wrap_because_the_input_is_processed_in_blocks() {
    for len in [
        32 * 65_535 - 1,
        32 * 65_535,          // exactly one full block
        32 * 65_535 + 1,      // one block plus one position
        2 * 32 * 65_535,      // exactly two full blocks
        2 * 32 * 65_535 + 17, // two blocks plus a partial chunk
    ] {
        let a: Vec<u16> = (0..len).map(|i| i as u16).collect();
        let b: Vec<u16> = a.iter().map(|x| x ^ 1).collect(); // every position differs
        assert_eq!(
            unifrac_from_sketches(&a, &b, false).to_bits(),
            1.0f32.to_bits(),
            "len={len}: all {len} positions differ, so d must be exactly 1.0"
        );
    }
}

/// The kernel is generic over sketch width because `--bbits` selects u16, u32 or
/// u64. The same logical input must give the same distance whichever width holds
/// it -- if it did not, `--bbits` would be changing results rather than only
/// changing memory.
#[test]
fn the_three_sketch_widths_agree_on_the_same_logical_input() {
    let k = 1000usize;
    let a16: Vec<u16> = (0..k).map(|i| (i % 300) as u16).collect();
    let b16: Vec<u16> = (0..k).map(|i| ((i * 7) % 300) as u16).collect();
    let a32: Vec<u32> = a16.iter().map(|&x| x as u32).collect();
    let b32: Vec<u32> = b16.iter().map(|&x| x as u32).collect();
    let a64: Vec<u64> = a16.iter().map(|&x| x as u64).collect();
    let b64: Vec<u64> = b16.iter().map(|&x| x as u64).collect();
    let d16 = unifrac_from_sketches(&a16, &b16, true);
    assert_eq!(d16.to_bits(), unifrac_from_sketches(&a32, &b32, true).to_bits());
    assert_eq!(d16.to_bits(), unifrac_from_sketches(&a64, &b64, true).to_bits());
}
