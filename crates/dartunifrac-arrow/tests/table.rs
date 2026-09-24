//! Contracts of `table_from_stream`.
//!
//! The first group is the milestone's reason for existing: a table that went
//! through Arrow must be the table core would have been handed directly, no
//! matter how the producer batched or ordered the rows. The second group is the
//! validation the FFI boundary owes, since core deliberately trusts its input.

mod common;

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, RecordBatchIterator,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use common::*;
use dartunifrac_arrow::{table_from_stream, MarshalError};
use dartunifrac_core::{build_sketches, Table};

fn assert_same_table(got: &Table, want: &Table) {
    assert_eq!(got.n_samples, want.n_samples, "n_samples");
    assert_eq!(got.colptr, want.colptr, "colptr");
    assert_eq!(got.node, want.node, "node");
    assert_eq!(got.value, want.value, "value");
    assert_eq!(got.col_sums, want.col_sums, "col_sums");
}

fn weighted_sums() -> Vec<f64> {
    test_table().col_sums
}

/// The unweighted table core would be handed: no weights at all, and the
/// denominators it never reads left at zero.
fn presence_table() -> Table {
    let t = test_table();
    Table { value: vec![], col_sums: vec![0.0; t.n_samples], ..t }
}

// ---------------------------------------------------------------- equivalence

#[test]
fn a_streamed_table_is_the_table_core_would_have_been_given() {
    let rows = test_rows();
    let sums = weighted_sums();

    let w = table_from_stream(reader(schema_of(true), vec![batch(&rows, true)]), 3, Some(&sums), true)
        .unwrap();
    assert_same_table(&w, &test_table());

    let u = table_from_stream(reader(schema_of(false), vec![batch(&rows, false)]), 3, None, false)
        .unwrap();
    assert_same_table(&u, &presence_table());
}

#[test]
fn a_streamed_table_sketches_identically_to_a_direct_one() {
    // Compared through `build_sketches` rather than field by field, because
    // that is the only thing a caller can actually observe.
    let rows = test_rows();
    let sums = weighted_sums();
    for weighted in [false, true] {
        let cs = if weighted { Some(&sums[..]) } else { None };
        let streamed =
            table_from_stream(reader(schema_of(weighted), vec![batch(&rows, weighted)]), 3, cs, weighted)
                .unwrap();
        let direct = if weighted { test_table() } else { presence_table() };
        let p = params(weighted);
        assert_eq!(
            build_sketches(&test_tree(), &streamed, &p, "").unwrap(),
            build_sketches(&test_tree(), &direct, &p, "").unwrap(),
            "weighted={weighted}: marshaling must not be observable in the sketches"
        );
    }
}

#[test]
fn how_the_producer_batches_the_rows_changes_nothing() {
    // A caller's batch boundaries are an artefact of its scan, not of its data.
    let rows = test_rows();
    let sums = weighted_sums();
    let want = test_table();
    for n in [1usize, 2, 4, 9, 100] {
        let got = table_from_stream(
            reader(schema_of(true), batched(&rows, n, true)),
            3,
            Some(&sums),
            true,
        )
        .unwrap();
        assert_same_table(&got, &want);
    }
}

#[test]
fn stream_order_does_not_change_the_table() {
    // The direct proof that the sort runs: this fixture's node ids are ascending
    // within every sample, so a preserved arrival order would leave `node` in
    // the shuffled permutation and this assert would fail.
    let rows = shuffled(&test_rows());
    assert_no_sample_arrives_sorted(&rows, 3);
    let sums = weighted_sums();
    let got =
        table_from_stream(reader(schema_of(true), vec![batch(&rows, true)]), 3, Some(&sums), true)
            .unwrap();
    assert_same_table(&got, &test_table());
}

#[test]
fn stream_order_does_not_change_the_sketches() {
    let sums = weighted_sums();
    let p = params(true);
    let ordered =
        table_from_stream(reader(schema_of(true), vec![batch(&test_rows(), true)]), 3, Some(&sums), true)
            .unwrap();
    let jumbled_rows = shuffled(&test_rows());
    assert_no_sample_arrives_sorted(&jumbled_rows, 3);
    let jumbled = table_from_stream(
        reader(schema_of(true), batched(&jumbled_rows, 2, true)),
        3,
        Some(&sums),
        true,
    )
    .unwrap();
    assert_eq!(
        build_sketches(&test_tree(), &ordered, &p, "").unwrap(),
        build_sketches(&test_tree(), &jumbled, &p, "").unwrap(),
    );
}

#[test]
fn core_is_order_sensitive_so_the_sort_is_load_bearing() {
    // Without this, the two tests above could pass for the wrong reason: if
    // reordering a sample's entries never changed anything, sorting would be
    // pointless and they would hold whatever the implementation did.
    //
    // Core accumulates weights in f32, and `1.0 + 2^-24 + 2^-24` is 1.0 while
    // `2^-24 + 2^-24 + 1.0` is the next f32 up. Under ERS that single ulp moves
    // a rejection-sampling decision, which desynchronises the whole RNG stream
    // and changes *every* slot -- measured at 256/256 for tip counts from 3 to
    // 128. DartMinHash and TreeMinHash absorb a perturbation this small, so the
    // claim is specifically about ERS, not about all three methods.
    use dartunifrac_core::{Method, SketchParams, Tree, NO_PARENT};

    // Root 0 (no edge) -> node 1 (length 1.0) -> three tips. Node 1 accumulates
    // all three masses, so it is where summation order becomes visible.
    let tree = Tree {
        parent: vec![NO_PARENT, 0, 1, 1, 1],
        lens: vec![0.0, 1.0, 1.0, 1.0, 1.0],
    };
    let eps = f64::from(f32::EPSILON) / 2.0;
    let table_of = |node: Vec<usize>, value: Vec<f64>| Table {
        n_samples: 2,
        colptr: vec![0, 3, 4],
        node: [node, vec![2]].concat(),
        value: [value, vec![1.0]].concat(),
        col_sums: vec![1.0, 1.0],
    };
    let p = SketchParams {
        k: 256,
        method: Method::Ers,
        ers_l: 2048,
        seed: 1337,
        weighted: true,
        raw_counts: true,
        portable: false,
    };

    let big_first = table_of(vec![2, 3, 4], vec![1.0, eps, eps]);
    let big_last = table_of(vec![3, 4, 2], vec![eps, eps, 1.0]);
    assert_ne!(
        build_sketches(&tree, &big_first, &p, "").unwrap(),
        build_sketches(&tree, &big_last, &p, "").unwrap(),
        "if this ever passes, f32 accumulation stopped being order-dependent \
         and the sort in table_from_stream is no longer justified"
    );
}

// ------------------------------------------------------------------ the shape

#[test]
fn a_missing_required_column_is_rejected() {
    let schema = Arc::new(Schema::new(vec![Field::new("sample_idx", DataType::Int64, false)]));
    let cols: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![0i64]))];
    let b = RecordBatch::try_new(schema.clone(), cols).unwrap();
    match table_from_stream(reader(schema, vec![b]), 3, None, false).marshal_err() {
        MarshalError::Schema(m) => assert!(m.contains("node_idx"), "should name it; got {m:?}"),
        e => panic!("expected Schema, got {e:?}"),
    }
}

#[test]
fn a_column_of_the_wrong_type_is_rejected() {
    // Int32 is a plausible thing for a caller to project by accident.
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_idx", DataType::Int32, false),
        Field::new("node_idx", DataType::Int64, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(vec![0i32])),
        Arc::new(Int64Array::from(vec![3i64])),
    ];
    let b = RecordBatch::try_new(schema.clone(), cols).unwrap();
    match table_from_stream(reader(schema, vec![b]), 3, None, false).marshal_err() {
        MarshalError::Schema(m) => {
            assert!(m.contains("sample_idx"), "should name the column; got {m:?}");
            assert!(m.contains("Int32"), "should name what it found; got {m:?}");
        }
        e => panic!("expected Schema, got {e:?}"),
    }
}

#[test]
fn columns_the_crate_does_not_know_about_are_ignored() {
    // Matching by name rather than position is what makes this safe, and a
    // caller selecting a few extra columns should not have to care.
    let schema = Arc::new(Schema::new(vec![
        Field::new("junk", DataType::Float32, false),
        Field::new("node_idx", DataType::Int64, false),
        Field::new("sample_idx", DataType::Int64, false),
    ]));
    let rows = test_rows();
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Float32Array::from(vec![0.0f32; rows.len()])),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>())),
        Arc::new(Int64Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>())),
    ];
    let b = RecordBatch::try_new(schema.clone(), cols).unwrap();
    let got = table_from_stream(reader(schema, vec![b]), 3, None, false).unwrap();
    assert_same_table(&got, &presence_table());
}

#[test]
fn a_null_in_a_required_column_is_rejected() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_idx", DataType::Int64, true),
        Field::new("node_idx", DataType::Int64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![Some(0i64), None])),
        Arc::new(Int64Array::from(vec![Some(3i64), Some(6)])),
    ];
    let b = RecordBatch::try_new(schema.clone(), cols).unwrap();
    match table_from_stream(reader(schema, vec![b]), 3, None, false).marshal_err() {
        MarshalError::Schema(m) => assert!(m.contains("sample_idx"), "should name it; got {m:?}"),
        e => panic!("expected Schema, got {e:?}"),
    }
}

#[test]
fn an_index_beyond_32_bits_is_rejected_rather_than_truncated() {
    // `sample_idx as usize` on a 32-bit target keeps only the low word, so 2^32
    // would read as 0 and the row would be filed under sample 0 -- a wrong
    // answer, not an error. The bound check compares in i64 to stop that.
    //
    // On a 64-bit host this passes with or without the fix. Run the suite with
    // `--target wasm32-wasip1` under wasmtime to see it bite, which is the
    // target shape that actually ships.
    let rows = vec![(0i64, 3i64, 1.0f64), (1i64 << 32, 6, 1.0)];
    match table_from_stream(reader(schema_of(false), vec![batch(&rows, false)]), 3, None, false)
        .marshal_err()
    {
        MarshalError::Data(m) => assert!(m.contains("4294967296"), "should name it; got {m:?}"),
        e => panic!("expected Data, got {e:?}"),
    }
}

#[test]
fn a_node_index_beyond_32_bits_is_never_silently_narrowed() {
    // node_idx has no upper bound here -- core owns it -- but core can only
    // check the number it is handed. Either this platform can represent the
    // value and it must arrive intact, or it cannot and that must be an error.
    // Truncating to something core would then happily accept is the one
    // outcome ruled out.
    let big = 1i64 << 32;
    let rows = vec![(0i64, 3i64, 1.0f64), (1, big, 1.0)];
    match table_from_stream(reader(schema_of(false), vec![batch(&rows, false)]), 3, None, false) {
        Ok(t) => assert_eq!(
            t.node.iter().map(|&n| n as i64).collect::<Vec<_>>(),
            vec![3, big],
            "node_idx must reach core unchanged"
        ),
        Err(MarshalError::Data(m)) => {
            assert!(m.contains("usize"), "a narrow platform must say so; got {m:?}")
        }
        Err(e) => panic!("unexpected error {e:?}"),
    }
}

#[test]
fn a_null_weight_is_rejected() {
    // `weight_column` carries its own null check, and until now only
    // `index_column`'s was exercised: deleting this one would have gone unseen.
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_idx", DataType::Int64, false),
        Field::new("node_idx", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![0i64, 1])),
        Arc::new(Int64Array::from(vec![3i64, 6])),
        Arc::new(Float64Array::from(vec![Some(1.0f64), None])),
    ];
    let b = RecordBatch::try_new(schema.clone(), cols).unwrap();
    match table_from_stream(reader(schema, vec![b]), 3, Some(&weighted_sums()), true).marshal_err() {
        MarshalError::Schema(m) => assert!(m.contains("value"), "should name it; got {m:?}"),
        e => panic!("expected Schema, got {e:?}"),
    }
}

#[test]
fn presence_mode_needs_no_value_column_and_leaves_the_weights_empty() {
    // Table::value is the size of the whole feature table. Presence runs never
    // read it, so making a caller materialise one would cost real memory to
    // carry a constant.
    let got = table_from_stream(
        reader(schema_of(false), vec![batch(&test_rows(), false)]),
        3,
        None,
        false,
    )
    .unwrap();
    assert!(got.value.is_empty(), "presence mode must not materialise weights");
}

#[test]
fn presence_mode_ignores_a_value_column_that_is_present_anyway() {
    let got = table_from_stream(
        reader(schema_of(true), vec![batch(&test_rows(), true)]),
        3,
        None,
        false,
    )
    .unwrap();
    assert!(got.value.is_empty());
    assert_same_table(&got, &presence_table());
}

#[test]
fn weighted_mode_without_a_value_column_is_rejected() {
    match table_from_stream(
        reader(schema_of(false), vec![batch(&test_rows(), false)]),
        3,
        Some(&weighted_sums()),
        true,
    )
    .marshal_err()
    {
        MarshalError::Schema(m) => assert!(m.contains("value"), "should name it; got {m:?}"),
        e => panic!("expected Schema, got {e:?}"),
    }
}

// ------------------------------------------------------------------- the data

#[test]
fn a_sample_idx_past_n_samples_is_rejected() {
    let rows = vec![(0i64, 3i64, 1.0f64), (7, 6, 1.0)];
    match table_from_stream(reader(schema_of(false), vec![batch(&rows, false)]), 3, None, false)
        .marshal_err()
    {
        MarshalError::Data(m) => {
            assert!(m.contains('7'), "should name the value; got {m:?}");
            assert!(m.contains('3'), "should name the bound; got {m:?}");
        }
        e => panic!("expected Data, got {e:?}"),
    }
}

#[test]
fn a_negative_index_is_rejected() {
    for (rows, needle) in [
        (vec![(-1i64, 3i64, 1.0f64)], "sample_idx"),
        (vec![(0i64, -3i64, 1.0f64)], "node_idx"),
    ] {
        match table_from_stream(reader(schema_of(false), vec![batch(&rows, false)]), 3, None, false)
            .marshal_err()
        {
            MarshalError::Data(m) => assert!(m.contains(needle), "got {m:?}, wanted {needle:?}"),
            e => panic!("expected Data, got {e:?}"),
        }
    }
}

#[test]
fn a_non_finite_weight_is_rejected() {
    // Core's own docs flag this as the FFI boundary's job: a NaN weight would
    // otherwise propagate into the accumulator and poison a whole sample.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let rows = vec![(0i64, 3i64, 1.0f64), (1, 6, bad)];
        match table_from_stream(
            reader(schema_of(true), vec![batch(&rows, true)]),
            3,
            Some(&weighted_sums()),
            true,
        )
        .marshal_err()
        {
            MarshalError::Data(m) => assert!(m.contains("value"), "should name it; got {m:?}"),
            e => panic!("expected Data for {bad}, got {e:?}"),
        }
    }
}

#[test]
fn a_non_finite_col_sum_is_rejected() {
    // col_sums is the denominator of the very division whose numerator is
    // already guarded, so leaving it unchecked guards nothing. NaN slips past
    // core's `denom == 0.0` skip and poisons that sample's whole accumulator;
    // +inf divides every entry to zero, which core's `inc == 0.0` fast path then
    // discards, so the sample loses all its mass and is dropped as empty. Both
    // are wrong answers rather than errors.
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let sums = [10.0, bad, 14.0];
        match table_from_stream(
            reader(schema_of(true), vec![batch(&test_rows(), true)]),
            3,
            Some(&sums),
            true,
        )
        .marshal_err()
        {
            MarshalError::Data(m) => {
                assert!(m.contains("col_sums"), "should name it; got {m:?}");
                assert!(m.contains('1'), "should name the sample; got {m:?}");
            }
            e => panic!("expected Data for {bad}, got {e:?}"),
        }
    }
}

#[test]
fn a_zero_or_negative_col_sum_is_accepted() {
    // Zero is meaningful -- core skips such a sample by design -- and a negative
    // total is reachable from the BIOM path, which does not filter negative
    // counts. Only non-finite values are rejected.
    let sums = [0.0, -4.0, 14.0];
    let got = table_from_stream(
        reader(schema_of(true), vec![batch(&test_rows(), true)]),
        3,
        Some(&sums),
        true,
    )
    .unwrap();
    assert_eq!(got.col_sums, vec![0.0, -4.0, 14.0]);
}

#[test]
fn a_negative_weight_is_accepted_because_the_biom_reader_does_not_filter_them() {
    let rows = vec![(0i64, 3i64, -1.0f64), (1, 6, 2.0)];
    let got = table_from_stream(
        reader(schema_of(true), vec![batch(&rows, true)]),
        3,
        Some(&weighted_sums()),
        true,
    )
    .unwrap();
    assert_eq!(got.value, vec![-1.0, 2.0]);
}

// -------------------------------------------------------- denominators, shape

#[test]
fn weighted_mode_without_col_sums_is_rejected() {
    // They cannot be derived here: they are summed over the caller's whole
    // table, including features that resolve to no tip and so never reach this
    // crate at all.
    match table_from_stream(reader(schema_of(true), vec![batch(&test_rows(), true)]), 3, None, true)
        .marshal_err()
    {
        MarshalError::Data(m) => assert!(m.contains("col_sums"), "should name it; got {m:?}"),
        e => panic!("expected Data, got {e:?}"),
    }
}

#[test]
fn col_sums_of_the_wrong_length_is_rejected() {
    let short = [1.0, 2.0];
    match table_from_stream(
        reader(schema_of(true), vec![batch(&test_rows(), true)]),
        3,
        Some(&short),
        true,
    )
    .marshal_err()
    {
        MarshalError::Data(m) => {
            assert!(m.contains('2') && m.contains('3'), "should give both counts; got {m:?}")
        }
        e => panic!("expected Data, got {e:?}"),
    }
}

#[test]
fn a_sample_with_no_entries_keeps_its_place() {
    // This is why n_samples is an input. Sample 1 contributes no rows, and if
    // its empty span were dropped, sample 2's entries would be attributed to it.
    let rows = vec![(0i64, 3i64, 1.0f64), (2, 6, 1.0)];
    let got = table_from_stream(reader(schema_of(false), vec![batch(&rows, false)]), 3, None, false)
        .unwrap();
    assert_eq!(got.n_samples, 3);
    assert_eq!(got.colptr, vec![0, 1, 1, 2], "sample 1 must keep an empty span");
    assert_eq!(got.node, vec![3, 6]);
}

#[test]
fn empty_batches_are_skipped() {
    let rows = test_rows();
    let batches = vec![
        batch(&[], true),
        batch(&rows[..4], true),
        batch(&[], true),
        batch(&rows[4..], true),
        batch(&[], true),
    ];
    let got =
        table_from_stream(reader(schema_of(true), batches), 3, Some(&weighted_sums()), true).unwrap();
    assert_same_table(&got, &test_table());
}

#[test]
fn a_stream_with_no_batches_yields_a_table_with_no_entries() {
    // Not an error: the C API reports "fewer than two samples" as an empty
    // result, and that decision belongs one layer up, not here.
    let got = table_from_stream(reader(schema_of(false), vec![]), 3, None, false).unwrap();
    assert_eq!(got.n_samples, 3);
    assert_eq!(got.colptr, vec![0, 0, 0, 0]);
    assert!(got.node.is_empty());
}

#[test]
fn an_error_from_the_callers_reader_is_propagated() {
    let schema = schema_of(false);
    let items: Vec<Result<RecordBatch, ArrowError>> = vec![
        Ok(batch(&test_rows()[..2], false)),
        Err(ArrowError::ComputeError("the scan gave up".into())),
    ];
    let r = RecordBatchIterator::new(items.into_iter(), schema);
    match table_from_stream(r, 3, None, false).marshal_err() {
        MarshalError::Stream(m) => assert!(m.contains("the scan gave up"), "got {m:?}"),
        e => panic!("expected Stream, got {e:?}"),
    }
}

#[test]
fn a_float64_value_column_is_required_not_float32() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("sample_idx", DataType::Int64, false),
        Field::new("node_idx", DataType::Int64, false),
        Field::new("value", DataType::Float32, false),
    ]));
    let cols: Vec<ArrayRef> = vec![
        Arc::new(Int64Array::from(vec![0i64])),
        Arc::new(Int64Array::from(vec![3i64])),
        Arc::new(Float32Array::from(vec![1.0f32])),
    ];
    let b = RecordBatch::try_new(schema.clone(), cols).unwrap();
    match table_from_stream(reader(schema, vec![b]), 3, Some(&weighted_sums()), true).marshal_err() {
        MarshalError::Schema(m) => assert!(m.contains("value"), "should name it; got {m:?}"),
        e => panic!("expected Schema, got {e:?}"),
    }
}

#[test]
fn the_advertised_input_schema_is_the_one_that_is_accepted() {
    // `input_schema` is what a caller builds against; if it ever drifts from
    // what the reader actually requires, every caller breaks at once.
    let sums = weighted_sums();
    for weighted in [false, true] {
        let cs = if weighted { Some(&sums[..]) } else { None };
        let advertised = Arc::new(dartunifrac_arrow::input_schema(weighted));
        let b = batch_with_schema(advertised.clone(), &test_rows(), weighted);
        assert!(
            table_from_stream(reader(advertised, vec![b]), 3, cs, weighted).is_ok(),
            "weighted={weighted}"
        );
    }
}

#[test]
fn duplicate_entries_for_one_node_are_ordered_canonically_too() {
    // Sorting by node id alone leaves ties in arrival order, which is exactly
    // the non-determinism the sort exists to remove: two rows for the same
    // (sample, node) would still accumulate in whichever order the scan
    // produced them. The tie-break is the weight, so the result is canonical
    // even when a caller's feature ids collapse onto one tree node.
    let a = vec![(0i64, 3i64, 1.0f64), (0, 3, 2.0), (0, 3, 0.5), (1, 6, 1.0)];
    let b = vec![(0i64, 3i64, 0.5f64), (0, 3, 1.0), (1, 6, 1.0), (0, 3, 2.0)];
    let sums = [3.5, 1.0, 0.0];

    let of = |rows: &[(i64, i64, f64)]| {
        table_from_stream(reader(schema_of(true), vec![batch(rows, true)]), 3, Some(&sums), true)
            .unwrap()
    };
    let (x, y) = (of(&a), of(&b));
    assert_eq!(x.node, y.node);
    assert_eq!(x.value, vec![0.5, 1.0, 2.0, 1.0], "ties must sort by weight, not arrival");
    assert_eq!(x.value, y.value);
}
