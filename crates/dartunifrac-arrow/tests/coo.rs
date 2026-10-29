//! Contracts of `coo_reader` — the output half.

mod common;

use arrow_array::RecordBatchReader;
use common::*;
use dartunifrac_arrow::{coo_reader, output_schema, table_from_stream, I, J};
use dartunifrac_core::build_sketches;

#[test]
fn the_index_columns_carry_the_same_type_as_the_input_sample_idx() {
    // The rule that keeps the two halves consistent: widen the input's accepted
    // index type later and this fails until the output is widened with it.
    let input = dartunifrac_arrow::input_schema(false);
    let want = input.field_with_name("sample_idx").unwrap().data_type();
    let out = output_schema();
    assert_eq!(out.field_with_name(I).unwrap().data_type(), want);
    assert_eq!(out.field_with_name(J).unwrap().data_type(), want);
}

#[test]
fn the_readers_schema_is_the_schema_of_every_batch_it_yields() {
    // Required of a RecordBatchReader, and the FFI stream export relies on it.
    let r = coo_reader(
        (0..7u32).map(|n| (n % 3, n % 3 + 1, n as f32)),
        vec![10, 20, 30, 40],
        3,
    );
    let schema = r.schema();
    assert_eq!(schema.as_ref(), &output_schema());
    for b in drain(r) {
        assert_eq!(b.schema(), schema);
    }
}

#[test]
fn pairs_are_batched_at_the_requested_size_with_a_short_final_batch() {
    let pairs: Vec<(u32, u32, f32)> = (0..7).map(|n| (0u32, 1u32, n as f32)).collect();
    let batches = drain(coo_reader(pairs.into_iter(), vec![0, 1], 3));
    assert_eq!(
        batches.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
        vec![3, 3, 1],
        "the tail must not be padded or dropped"
    );
}

#[test]
fn no_pairs_yields_no_batches_but_still_a_schema() {
    // A caller with fewer than two samples gets an empty result, not an error,
    // and still has to be able to read the schema off the stream.
    let r = coo_reader(std::iter::empty(), vec![], 1024);
    assert_eq!(r.schema().as_ref(), &output_schema());
    assert!(drain(r).is_empty());
}

#[test]
fn i_and_j_are_the_callers_sample_ids_not_compacted_positions() {
    // With every sample surviving these two spaces coincide, so the fixture
    // deliberately has gaps: ids 0, 2 and 5 out of six samples.
    let pairs = vec![(0u32, 1u32, 0.25f32), (0, 2, 0.5), (1, 2, 0.75)];
    let got = triples(&drain(coo_reader(pairs.into_iter(), vec![0, 2, 5], 1024)));
    assert_eq!(
        got,
        vec![(0, 2, 0.25), (0, 5, 0.5), (2, 5, 0.75)],
        "emitting 0,1,2 here would mean the kept[] mapping was skipped"
    );
}

#[test]
fn an_index_outside_the_id_table_is_an_error_not_a_panic() {
    // Only reachable from a bug in the layer above, but this crate sits under an
    // FFI boundary: an error is recoverable where a panic is a process-wide
    // hazard.
    let mut r = coo_reader(std::iter::once((0u32, 9u32, 1.0f32)), vec![0, 1], 16);
    assert!(r.next().expect("a batch should be attempted").is_err());
}

#[test]
fn the_ids_a_real_run_produces_are_the_input_sample_indices() {
    // End to end against core, so the id contract is pinned to what
    // `SketchSet::kept` actually contains rather than to a hand-written vector.
    // Sample 1 contributes nothing and is dropped, so kept is [0, 2].
    let rows = vec![(0i64, 3i64, 1.0f64), (2, 6, 1.0)];
    let table =
        table_from_stream(reader(schema_of(false), vec![batch(&rows, false)]), 3, None, false)
            .unwrap();
    let set = build_sketches(&test_tree(), &table, &params(false), "").unwrap();
    assert_eq!(set.kept, vec![0, 2], "sample 1 is empty and must be dropped");

    let ids: Vec<i64> = set.kept.iter().map(|&s| s as i64).collect();
    let got = triples(&drain(coo_reader(std::iter::once((0u32, 1u32, 0.5f32)), ids, 16)));
    assert_eq!(got, vec![(0, 2, 0.5)], "the pair is between input samples 0 and 2, not 0 and 1");
}
