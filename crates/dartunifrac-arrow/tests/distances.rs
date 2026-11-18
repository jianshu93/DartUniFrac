//! Contracts of `trim_to_bbits` and `distances_reader` — the producer half.
//!
//! The distance kernel itself is core's and is pinned by core's own golden
//! tests. What is under test here is everything around it: that tiling the upper
//! triangle emits each pair once and only once, that where a pair lands in the
//! batch stream never changes its value, and that neither the batch size nor the
//! thread count can perturb the result.

mod common;

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use common::*;
use dartunifrac_arrow::{
    cross_distances_reader, distances_reader, output_schema, table_from_stream, trim_to_bbits,
    MarshalError, Sketches,
};
use dartunifrac_core::{build_sketches, unifrac_from_sketches};

/// Sketches of `n` samples, narrowed to 16 bits. The alphabet is small so that
/// pairs genuinely differ; see [`common::fake_sketches`].
fn sk16(n: usize) -> Sketches {
    trim_to_bbits(fake_sketches(n, 64, 8), 16).expect("16 is a supported width")
}

fn ids(n: usize) -> Vec<i64> {
    (0..n as i64).collect()
}

/// The naive thing the tiling has to agree with: every pair, in row-major order,
/// through the same kernel.
fn reference(sk: &Sketches, ids: &[i64], weighted: bool) -> Vec<(i64, i64, f32)> {
    macro_rules! go {
        ($rows:expr) => {{
            let mut out = Vec::new();
            for i in 0..$rows.len() {
                for j in (i + 1)..$rows.len() {
                    out.push((
                        ids[i],
                        ids[j],
                        unifrac_from_sketches(&$rows[i], &$rows[j], weighted),
                    ));
                }
            }
            out
        }};
    }
    match sk {
        Sketches::U16(r) => go!(r),
        Sketches::U32(r) => go!(r),
        Sketches::U64(r) => go!(r),
    }
}

/// Pairs are unique, so `(i, j)` is a total order and the distances need no
/// comparator — which matters, because they are compared for exact equality.
fn by_pair(mut v: Vec<(i64, i64, f32)>) -> Vec<(i64, i64, f32)> {
    v.sort_by_key(|t| (t.0, t.1));
    v
}

fn stream(sk: Sketches, ids: Vec<i64>, weighted: bool, batch_rows: usize) -> Vec<RecordBatch> {
    drain(distances_reader(sk, ids, weighted, batch_rows).expect("fixture is well formed"))
}

// ---------------------------------------------------------------- correctness

#[test]
fn the_tiled_stream_matches_a_serial_reference_pair_for_pair() {
    // 37 is deliberately not a multiple of the tile side, so the last tile row
    // and column are short and the clipping is exercised.
    let want = by_pair(reference(&sk16(37), &ids(37), false));
    assert_distances_vary(&want);
    let got = by_pair(triples(&stream(sk16(37), ids(37), false, 64)));
    assert_eq!(got, want, "tiling must not change which pair gets which value");
}

#[test]
fn the_tiled_stream_matches_a_serial_reference_when_weighted() {
    // The weighted transform is applied per pair, so it has to survive tiling
    // the same way the raw distance does.
    let want = by_pair(reference(&sk16(37), &ids(37), true));
    assert_distances_vary(&want);
    let got = by_pair(triples(&stream(sk16(37), ids(37), true, 64)));
    assert_eq!(got, want);
}

#[test]
fn the_serial_reference_still_matches_where_tiles_land_on_a_boundary() {
    // `batch_rows = 64` gives a tile side of 8. Clipping the last tile row and
    // column is the part of the arithmetic most likely to be subtly wrong, so
    // pin exact values one short of, exactly on, and one past a boundary.
    // Without this the property has to be re-derived from the formula by
    // whoever touches `Tile::counts` next.
    for n in [7usize, 8, 9, 15, 16, 17, 24, 25] {
        let want = by_pair(reference(&sk16(n), &ids(n), false));
        let got = by_pair(triples(&stream(sk16(n), ids(n), false, 64)));
        assert_eq!(got, want, "n = {n} disagrees with the serial reference");
        assert_eq!(got.len(), n * (n - 1) / 2, "n = {n} lost or duplicated pairs");
    }
}

#[test]
fn every_unordered_pair_appears_exactly_once() {
    // The property that makes this a *condensed* result: skipping tiles below
    // the diagonal must not drop pairs, and the diagonal tiles must not emit
    // each of their pairs twice.
    let n = 37;
    let got = triples(&stream(sk16(n), ids(n), false, 64));
    assert_eq!(got.len(), n * (n - 1) / 2, "expected the condensed count");
    assert!(got.iter().all(|t| t.0 < t.1), "every pair must be upper triangle");
    let mut seen: Vec<(i64, i64)> = got.iter().map(|t| (t.0, t.1)).collect();
    seen.sort_unstable();
    let before = seen.len();
    seen.dedup();
    assert_eq!(seen.len(), before, "a pair was emitted more than once");
}

/// An `r × c` rectangle in the order its tiles are visited: row tiles
/// ascending, each row tile's column tiles ascending, and within a tile each
/// row's columns ascending. Written out independently of the reader, so the two
/// can disagree.
fn rect_tile_major(r: usize, c: usize, side: usize) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for ti in 0..r.div_ceil(side) {
        for tj in 0..c.div_ceil(side) {
            for i in ti * side..((ti + 1) * side).min(r) {
                for j in tj * side..((tj + 1) * side).min(c) {
                    out.push((i as i64, j as i64));
                }
            }
        }
    }
    out
}

/// The upper triangle in the order the tiling visits it: the `n × n` square's
/// order with only `i < j` kept. Tiles below the diagonal hold no such pair,
/// and a diagonal tile keeps each row's columns past itself, which is the
/// triangle's walk: row tiles ascending, then column tiles from the diagonal
/// rightwards.
fn tile_major(n: usize, side: usize) -> Vec<(i64, i64)> {
    rect_tile_major(n, n, side).into_iter().filter(|(i, j)| i < j).collect()
}

/// Each triple's `(i, j)`, in stream order.
fn pairs(got: &[(i64, i64, f32)]) -> Vec<(i64, i64)> {
    got.iter().map(|t| (t.0, t.1)).collect()
}

#[test]
fn the_stream_order_is_tile_major() {
    // The order is part of the stream, and it is not row-major: once n exceeds
    // a tile's side it depends on batch_rows, though the pairs do not. Pinned so
    // that a change to the tiling cannot reorder the stream unseen.
    let order =
        |n: usize, batch_rows: usize| pairs(&triples(&stream(sk16(n), ids(n), false, batch_rows)));
    // n = 5 at side 2, derived by hand. Row-major would put (0, 4) fourth.
    assert_eq!(
        order(5, 4),
        vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (0, 4), (1, 4), (2, 3), (2, 4), (3, 4)]
    );
    // Partial tiles on both axes.
    for (n, batch_rows, side) in [(37, 64, 8), (17, 16, 4), (9, 4, 2)] {
        assert_eq!(order(n, batch_rows), tile_major(n, side), "n = {n}, side = {side}");
    }
}

#[test]
fn all_three_sketch_widths_stream_the_full_pair_set() {
    // The reader dispatches on the enum; a missing arm would be a width that
    // silently produces nothing.
    let n = 20;
    let raw = fake_sketches(n, 64, 8);
    for bbits in [16u8, 32, 64] {
        let sk = trim_to_bbits(raw.clone(), bbits).unwrap();
        let want = by_pair(reference(&sk, &ids(n), false));
        let sk = trim_to_bbits(raw.clone(), bbits).unwrap();
        let got = by_pair(triples(&stream(sk, ids(n), false, 64)));
        assert_eq!(got, want, "width {bbits} disagrees with its own reference");
        assert_eq!(got.len(), n * (n - 1) / 2, "width {bbits} lost pairs");
    }
}

#[test]
fn a_real_sketch_set_streams_its_distances() {
    // End to end against core rather than against synthetic sketches, so the
    // plumbing is pinned to what `build_sketches` actually returns.
    let table = table_from_stream(
        reader(schema_of(false), vec![batch(&test_rows(), false)]),
        3,
        None,
        false,
    )
    .unwrap();
    let set = build_sketches(&test_tree(), &table, &params(false), "").unwrap();
    let kept: Vec<i64> = set.kept.iter().map(|&s| s as i64).collect();
    assert_eq!(kept.len(), 3, "all three fixture samples should survive");

    let want = by_pair(reference(
        &trim_to_bbits(set.sketches.clone(), 16).unwrap(),
        &kept,
        false,
    ));
    let got = by_pair(triples(&stream(
        trim_to_bbits(set.sketches, 16).unwrap(),
        kept,
        false,
        64,
    )));
    assert_eq!(got, want);
}

// -------------------------------------------------------------------- framing

#[test]
fn the_pair_set_does_not_change_with_the_batch_size() {
    // Tile boundaries are an artefact of how the output is cut up. If any of
    // these disagree, a pair is falling into a gap between tiles or being
    // counted by two of them.
    let n = 37;
    let want = by_pair(triples(&stream(sk16(n), ids(n), false, 64)));
    for batch_rows in [1usize, 2, 3, 5, 17, 64, 1000, n * n * 10] {
        let got = by_pair(triples(&stream(sk16(n), ids(n), false, batch_rows)));
        assert_eq!(got, want, "batch_rows = {batch_rows} changed the result");
    }
}

#[test]
fn no_batch_exceeds_the_requested_row_count() {
    // The bound the caller is promised: memory is a function of batch_rows, so
    // a batch that overshoots it breaks the contract even if the values are
    // right.
    for batch_rows in [1usize, 2, 7, 64, 1000] {
        for b in stream(sk16(37), ids(37), false, batch_rows) {
            assert!(
                b.num_rows() <= batch_rows,
                "a batch of {} rows exceeds batch_rows = {batch_rows}",
                b.num_rows()
            );
        }
    }
}

#[test]
fn the_largest_batch_does_not_grow_with_the_number_of_samples() {
    // This is the whole point of tiling in 2D. A row-stripe scheme bounds only
    // the rows it reads, so its batches grow linearly in n; tiles bound both
    // axes. Both n here are larger than the tile side, so neither is clipped.
    let batch_rows = 64;
    let max_at = |n: usize| {
        stream(sk16(n), ids(n), false, batch_rows)
            .iter()
            .map(|b| b.num_rows())
            .max()
            .expect("there are pairs to emit")
    };
    let small = max_at(200);
    let large = max_at(400);
    assert_eq!(small, large, "batch size tracked n instead of batch_rows");
    assert!(small <= batch_rows);
    // ...and it is actually filling tiles, not emitting one pair per batch.
    assert!(small > 1, "batches of {small} row(s) would make the bound vacuous");
}

#[test]
// wasm32-wasip1 has no threads to spawn, so the pool cannot be built there and
// there is no thread count to vary. Marked ignored rather than compiled out, so
// the wasm run still reports it instead of silently listing one test fewer.
#[cfg_attr(target_family = "wasm", ignore = "rayon cannot spawn threads on wasm")]
fn output_is_identical_whatever_the_thread_count() {
    // Be clear about what this does and does not do. As written, the producer
    // hands every rayon task a disjoint span at a fixed offset, with no shared
    // accumulator and no order-dependent reduction, so it *cannot* vary with
    // thread count -- there is nothing to race on. This test is therefore a
    // regression guard, not a stress test: it fails the day someone replaces
    // the span fill with an unordered collect off a channel, which is exactly
    // the shape the bounded-queue variant of this design would take.
    //
    // It also runs inside an explicit pool, which is how M5's per-context pool
    // will drive the reader.
    let n = 37;
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool builds");
        pool.install(|| triples(&stream(sk16(n), ids(n), false, 64)))
    };
    let one = run(1);
    assert_distances_vary(&one);
    for threads in [2usize, 3, 8] {
        assert_eq!(
            run(threads),
            one,
            "{threads} threads produced a different order or different values"
        );
    }
}

#[test]
fn a_batch_row_request_of_zero_is_clamped_to_one_row_rather_than_refused() {
    // `bbits` is rejected when it is nonsense because a wrong width silently
    // changes every distance. `batch_rows` only decides where batches are cut,
    // so zero is clamped instead -- and clamped the way `coo_reader` already
    // clamps it, so the two halves of the crate agree rather than each having
    // its own rule.
    //
    // The clamp is load-bearing: remove it and `n.div_ceil(0)` panics, which
    // beneath an FFI boundary aborts the host process. This test is what fails
    // if someone decides the `.max(1)` is redundant.
    let n = 7;
    let want = by_pair(triples(&stream(sk16(n), ids(n), false, 64)));
    let batches = stream(sk16(n), ids(n), false, 0);
    assert!(
        batches.iter().all(|b| b.num_rows() == 1),
        "a zero request should behave as one row per batch"
    );
    assert_eq!(by_pair(triples(&batches)), want, "clamping changed the result");
}

#[test]
fn the_readers_schema_is_the_coo_output_schema() {
    // Not rebuilt here: the input and output index types are tied together in
    // coo.rs, and a second schema would be free to drift away from it.
    let r = distances_reader(sk16(10), ids(10), false, 64).unwrap();
    let schema = r.schema();
    assert_eq!(schema.as_ref(), &output_schema());
    for b in drain(r) {
        assert_eq!(b.schema(), schema);
    }
}

#[test]
fn fewer_than_two_samples_yields_a_schema_and_no_batches() {
    // A caller with one sample has no pairs; that is an empty result, not an
    // error, and the schema still has to be readable off the stream.
    for n in [0usize, 1] {
        let r = distances_reader(sk16(n), ids(n), false, 64).unwrap();
        assert_eq!(r.schema().as_ref(), &output_schema(), "n = {n}");
        assert!(drain(r).is_empty(), "n = {n} should emit no batches");
    }
}

#[test]
fn emitted_indices_are_the_callers_ids_not_tile_positions() {
    // With dense ids the two spaces coincide and a bug would be invisible, so
    // the ids here are sparse -- these are what `SketchSet::kept` looks like
    // after empty samples are dropped.
    let sparse = vec![100i64, 200, 300, 400];
    let got = by_pair(triples(&stream(sk16(4), sparse.clone(), false, 64)));
    let pairs: Vec<(i64, i64)> = got.iter().map(|t| (t.0, t.1)).collect();
    assert_eq!(
        pairs,
        vec![(100, 200), (100, 300), (100, 400), (200, 300), (200, 400), (300, 400)],
        "emitting 0..4 here would mean the id mapping was skipped"
    );
}

// ------------------------------------------------------------------ rejection

#[test]
fn ragged_sketches_are_an_error_not_a_panic() {
    // core's `count_mismatches` asserts equal lengths, and an assert beneath an
    // FFI boundary aborts the host process. Catching it here keeps it
    // recoverable.
    let mut raw = fake_sketches(4, 64, 8);
    raw[2].truncate(63);
    let sk = trim_to_bbits(raw, 16).unwrap();
    let err = distances_reader(sk, ids(4), false, 64).marshal_err();
    assert!(
        matches!(err, MarshalError::Sketch(_)),
        "expected a Sketch error, got {err}"
    );
}

#[test]
fn an_id_count_that_disagrees_with_the_sketch_count_is_an_error() {
    // Checked once up front rather than per row: the pairs come from this
    // crate, so a mismatch is a caller bug about the id table, not bad data in
    // the middle of a stream.
    let err = distances_reader(sk16(4), ids(3), false, 64).marshal_err();
    assert!(
        matches!(err, MarshalError::Sketch(_)),
        "expected a Sketch error, got {err}"
    );
}

#[test]
fn sketches_with_no_values_are_an_error_not_a_matrix_full_of_nan() {
    // `unifrac_from_sketches` divides the mismatch count by the sketch length,
    // so a zero-length sketch yields 0/0. Unweighted that is NaN and weighted it
    // is 1.0 -- in neither case does anything fail, so the caller would get a
    // full distance matrix that is silently meaningless.
    let err = distances_reader(trim_to_bbits(vec![vec![]; 3], 16).unwrap(), ids(3), false, 64)
        .marshal_err();
    assert!(
        matches!(err, MarshalError::Sketch(_)),
        "expected a Sketch error, got {err}"
    );
}

// ----------------------------------------------------------------------- trim

#[test]
fn trimming_keeps_the_low_bits() {
    // b-bit minwise hashing compares the *low* bits. Truncating from the other
    // end would still produce a u16 and still compare cleanly -- and would be
    // wrong -- so the expected values are written out rather than derived.
    let raw = vec![vec![0x1234_5678_9ABC_DEF0u64, 0x0000_0000_0000_FFFF]];
    match trim_to_bbits(raw.clone(), 16).unwrap() {
        Sketches::U16(s) => assert_eq!(s, vec![vec![0xDEF0u16, 0xFFFF]]),
        _ => panic!("bbits 16 must produce U16"),
    }
    match trim_to_bbits(raw.clone(), 32).unwrap() {
        Sketches::U32(s) => assert_eq!(s, vec![vec![0x9ABC_DEF0u32, 0x0000_FFFF]]),
        _ => panic!("bbits 32 must produce U32"),
    }
    match trim_to_bbits(raw.clone(), 64).unwrap() {
        Sketches::U64(s) => assert_eq!(s, raw, "bbits 64 is the identity"),
        _ => panic!("bbits 64 must produce U64"),
    }
}

#[test]
fn an_unsupported_bbits_is_an_error_not_a_silent_fallback_to_16() {
    // The binary warns and uses 16, which is safe there only because clap has
    // already restricted the value. Here the caller supplies it directly, and
    // quietly narrowing their sketches would change every distance in the
    // result while reporting success.
    for bbits in [0u8, 1, 8, 15, 17, 24, 31, 33, 63, 65, 255] {
        let err = trim_to_bbits(fake_sketches(2, 8, 8), bbits).marshal_err();
        assert!(
            matches!(err, MarshalError::Sketch(_)),
            "bbits = {bbits} should be rejected, got {err}"
        );
    }
}

// ---------------------------------------------------------------------- cross
//
// `cross_distances_reader` (M7): every pair of a rows × cols rectangle. The two
// sides are separate sketch sets, so each carries its own ids, and the same
// sample may sit on both sides; its pair with itself is then emitted like any
// other (G1: no special cases).

/// Two sets of `r` and `c` sketches of `k` values at `bbits`, drawn as one
/// sequence and split, so no sketch on one side repeats one on the other.
fn sides(r: usize, c: usize, k: usize, bbits: u8) -> (Sketches, Sketches) {
    let mut raw = fake_sketches(r + c, k, 8);
    let cols = raw.split_off(r);
    (trim_to_bbits(raw, bbits).unwrap(), trim_to_bbits(cols, bbits).unwrap())
}

/// The naive rectangle: every rows × cols pair, row-major, through the kernel.
fn cross_reference(
    rows: &Sketches,
    row_ids: &[i64],
    cols: &Sketches,
    col_ids: &[i64],
    weighted: bool,
) -> Vec<(i64, i64, f32)> {
    macro_rules! go {
        ($a:expr, $b:expr) => {{
            let mut out = Vec::new();
            for i in 0..$a.len() {
                for j in 0..$b.len() {
                    out.push((
                        row_ids[i],
                        col_ids[j],
                        unifrac_from_sketches(&$a[i], &$b[j], weighted),
                    ));
                }
            }
            out
        }};
    }
    match (rows, cols) {
        (Sketches::U16(a), Sketches::U16(b)) => go!(a, b),
        (Sketches::U32(a), Sketches::U32(b)) => go!(a, b),
        (Sketches::U64(a), Sketches::U64(b)) => go!(a, b),
        _ => panic!("the reference compares two sides of one width"),
    }
}

fn cross(
    rows: impl Into<Arc<Sketches>>,
    row_ids: Vec<i64>,
    cols: impl Into<Arc<Sketches>>,
    col_ids: Vec<i64>,
    weighted: bool,
    batch_rows: usize,
) -> Vec<RecordBatch> {
    drain(
        cross_distances_reader(rows, row_ids, cols, col_ids, weighted, batch_rows)
            .expect("fixture is well formed"),
    )
}

#[test]
fn the_cross_stream_matches_a_naive_rectangle_pair_for_pair() {
    // At batch_rows 16 the tile side is 4. The shapes straddle it on each axis
    // independently, down to a single row and a single column.
    for (r, c) in [(7, 9), (8, 8), (9, 17), (16, 1), (1, 16), (25, 24)] {
        for weighted in [false, true] {
            let (a, b) = sides(r, c, 64, 16);
            let want = by_pair(cross_reference(&a, &ids(r), &b, &ids(c), weighted));
            let got = by_pair(triples(&cross(a, ids(r), b, ids(c), weighted, 16)));
            assert_eq!(got.len(), r * c, "{r}×{c} lost or duplicated pairs");
            assert_eq!(got, want, "{r}×{c}, weighted = {weighted}, disagrees with the reference");
            assert_distances_vary(&got);
        }
    }
}

#[test]
fn the_cross_stream_covers_all_three_widths() {
    // Both sides are matched on their width together; a missing arm would be a
    // width that silently produces nothing.
    for bbits in [16u8, 32, 64] {
        let (a, b) = sides(9, 17, 64, bbits);
        let want = by_pair(cross_reference(&a, &ids(9), &b, &ids(17), true));
        let got = by_pair(triples(&cross(a, ids(9), b, ids(17), true, 16)));
        assert_eq!(got, want, "width {bbits} disagrees with its own reference");
    }
}

#[test]
fn each_side_keeps_its_own_ids() {
    // Disjoint, sparse id ranges, so emitting tile positions, or one side's ids
    // for the other's, cannot pass. Equality with the full product also proves
    // every pair appears exactly once.
    let row_ids: Vec<i64> = (0..5).map(|x| 100 + 3 * x).collect();
    let col_ids: Vec<i64> = (0..7).map(|x| 7 + 2 * x).collect();
    let (a, b) = sides(5, 7, 64, 16);
    let got = pairs(&by_pair(triples(&cross(a, row_ids.clone(), b, col_ids.clone(), false, 4))));
    let want: Vec<(i64, i64)> = row_ids
        .iter()
        .flat_map(|&i| col_ids.iter().map(move |&j| (i, j)))
        .collect();
    assert_eq!(got, want, "every rows × cols pair, once, under each side's own ids");
}

#[test]
fn the_same_set_on_both_sides_gives_the_full_square() {
    // G1: no special case. Every pair comes in both orders with one value, the
    // diagonal is zero, and the upper half is exactly df_distances' triangle.
    let n = 9;
    let sk = Arc::new(sk16(n));
    let got = by_pair(triples(&cross(sk.clone(), ids(n), sk.clone(), ids(n), false, 16)));
    assert_eq!(got.len(), n * n, "the full square");
    let d = |i: usize, j: usize| got[i * n + j].2;
    for i in 0..n {
        assert_eq!(d(i, i), 0.0, "sample {i}'s distance to itself");
        for j in 0..n {
            assert_eq!(d(i, j).to_bits(), d(j, i).to_bits(), "({i}, {j}) is not symmetric");
        }
    }
    let upper: Vec<(i64, i64, f32)> = got.iter().copied().filter(|t| t.0 < t.1).collect();
    let triangle = by_pair(triples(&drain(distances_reader(sk, ids(n), false, 16).unwrap())));
    assert_eq!(upper, triangle, "the upper half should be df_distances' triangle");
}

#[test]
fn an_empty_side_gives_a_schema_and_no_batches() {
    // G1: an empty side is an empty result, as df_distances gives for n < 2,
    // against any well-formed other side -- even sketches of another width,
    // since no pair is ever compared. (A side malformed in itself is refused
    // whatever the other holds; see the refusals below.)
    type Case = (&'static str, Sketches, Vec<i64>, Sketches, Vec<i64>);
    let empty = || trim_to_bbits(Vec::new(), 16).unwrap();
    let other_width = || trim_to_bbits(fake_sketches(5, 64, 8), 32).unwrap();
    let cases: Vec<Case> = vec![
        ("0×5", empty(), vec![], sk16(5), ids(5)),
        ("5×0", sk16(5), ids(5), empty(), vec![]),
        ("0×0", empty(), vec![], empty(), vec![]),
        ("0×5 of another width", empty(), vec![], other_width(), ids(5)),
    ];
    for (label, a, a_ids, b, b_ids) in cases {
        let r = cross_distances_reader(a, a_ids, b, b_ids, false, 16).unwrap();
        assert_eq!(r.schema().as_ref(), &output_schema(), "{label}");
        assert!(drain(r).is_empty(), "{label} should emit no batches");
    }
}

#[test]
#[cfg_attr(target_family = "wasm", ignore = "rayon cannot spawn threads on wasm")]
fn the_cross_stream_is_identical_whatever_the_thread_count() {
    // Order included: a regression guard on the fixed-offset span fill, run
    // inside an explicit pool as the C API's per-context pool will drive it.
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool builds");
        let (a, b) = sides(25, 24, 64, 16);
        pool.install(|| triples(&cross(a, ids(25), b, ids(24), true, 16)))
    };
    let one = run(1);
    assert_distances_vary(&one);
    for threads in [2usize, 4] {
        assert_eq!(
            run(threads),
            one,
            "{threads} threads produced a different order or different values"
        );
    }
}

#[test]
fn the_cross_pairs_do_not_change_with_the_batch_size_and_batches_are_bounded() {
    let want = {
        let (a, b) = sides(25, 24, 64, 16);
        by_pair(cross_reference(&a, &ids(25), &b, &ids(24), false))
    };
    for batch_rows in [1usize, 16, 65536] {
        let (a, b) = sides(25, 24, 64, 16);
        let batches = cross(a, ids(25), b, ids(24), false, batch_rows);
        assert!(
            batches.iter().all(|b| b.num_rows() <= batch_rows),
            "a batch exceeds batch_rows = {batch_rows}"
        );
        assert_eq!(by_pair(triples(&batches)), want, "batch_rows = {batch_rows} changed the pairs");
    }
}

#[test]
fn the_cross_order_is_row_tile_major() {
    let order = |r: usize, c: usize, batch_rows: usize| {
        let (a, b) = sides(r, c, 64, 16);
        pairs(&triples(&cross(a, ids(r), b, ids(c), false, batch_rows)))
    };
    // 3 × 3 at side 2, derived by hand.
    assert_eq!(
        order(3, 3, 4),
        vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    );
    for (r, c, batch_rows, side) in [(9, 17, 16, 4), (25, 24, 16, 4), (7, 3, 4, 2)] {
        assert_eq!(order(r, c, batch_rows), rect_tile_major(r, c, side), "{r}×{c}, side = {side}");
    }
}

/// The error a cross refuses with, as text, after checking its kind.
fn cross_refusal(
    rows: Sketches,
    row_ids: Vec<i64>,
    cols: Sketches,
    col_ids: Vec<i64>,
) -> String {
    let err = cross_distances_reader(rows, row_ids, cols, col_ids, false, 16).marshal_err();
    assert!(matches!(err, MarshalError::Sketch(_)), "expected a Sketch error, got {err}");
    err.to_string()
}

#[test]
fn sides_of_different_widths_are_refused_naming_both() {
    // Both non-empty: a 16-bit sketch read against a 32-bit one is not a
    // comparison at all. The capi's fingerprint check makes this unreachable
    // from C, but this crate is the validation boundary in its own right.
    let msg = cross_refusal(
        trim_to_bbits(fake_sketches(3, 64, 8), 16).unwrap(),
        ids(3),
        trim_to_bbits(fake_sketches(4, 64, 8), 32).unwrap(),
        ids(4),
    );
    for want in ["rows", "cols", "16", "32"] {
        assert!(msg.contains(want), "{msg:?} should name {want}");
    }
}

#[test]
fn sides_of_different_lengths_are_refused_naming_both() {
    // core's count_mismatches asserts equal lengths, and an assert beneath an
    // FFI boundary aborts the host process.
    let msg = cross_refusal(
        trim_to_bbits(fake_sketches(3, 64, 8), 16).unwrap(),
        ids(3),
        trim_to_bbits(fake_sketches(4, 32, 8), 16).unwrap(),
        ids(4),
    );
    for want in ["rows", "cols", "64", "32"] {
        assert!(msg.contains(want), "{msg:?} should name {want}");
    }
}

#[test]
fn a_ragged_side_is_refused_naming_that_side() {
    for (side, other) in [("rows", "cols"), ("cols", "rows")] {
        let mut raw = fake_sketches(4, 64, 8);
        raw[2].truncate(63);
        let ragged = trim_to_bbits(raw, 16).unwrap();
        let msg = if side == "rows" {
            cross_refusal(ragged, ids(4), sk16(3), ids(3))
        } else {
            cross_refusal(sk16(3), ids(3), ragged, ids(4))
        };
        assert!(msg.contains(side), "{msg:?} should name {side}");
        assert!(!msg.contains(other), "{msg:?} blames the wrong side");
    }
}

#[test]
fn an_id_count_that_disagrees_with_its_side_is_refused_naming_that_side() {
    for (side, other) in [("rows", "cols"), ("cols", "rows")] {
        let msg = if side == "rows" {
            cross_refusal(sk16(4), ids(3), sk16(5), ids(5))
        } else {
            cross_refusal(sk16(4), ids(4), sk16(5), ids(6))
        };
        assert!(msg.contains(side), "{msg:?} should name {side}");
        assert!(!msg.contains(other), "{msg:?} blames the wrong side");
    }
}

#[test]
fn sketches_with_no_values_are_refused_even_one_by_one() {
    // validate() refuses k = 0 only for a set of two or more, since a single
    // sample has no pairs. Across two sides one sample each already makes a
    // pair, and its distance would be 0/0.
    let msg = cross_refusal(
        trim_to_bbits(vec![vec![]], 16).unwrap(),
        ids(1),
        trim_to_bbits(vec![vec![]], 16).unwrap(),
        ids(1),
    );
    assert!(msg.contains("no values"), "{msg:?} should say the sketches hold no values");
}
