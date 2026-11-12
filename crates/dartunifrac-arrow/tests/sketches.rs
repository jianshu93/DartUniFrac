//! Contracts of `sketches_reader` and `sketches_from_stream`: a sketch set as
//! the rows a caller stores, and back.
//!
//! The first group is the reason the module exists: a set written out and read
//! back is the set, bit for bit, however the store orders, batches or retypes
//! the rows. The second group is what a store can hand back that must be
//! refused, each naming the row, since a row the caller cannot find is a row it
//! cannot fix.

mod common;

use std::sync::Arc;

use arrow_array::builder::{FixedSizeListBuilder, GenericListBuilder, PrimitiveBuilder};
use arrow_array::types::{
    ArrowPrimitiveType, Int16Type, Int64Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    ArrayRef, BinaryArray, FixedSizeBinaryArray, Int32Array, Int64Array, LargeBinaryArray,
    OffsetSizeTrait, RecordBatch, RecordBatchIterator, RecordBatchReader, StringArray, UInt16Array,
};
use arrow_buffer::{Buffer, OffsetBuffer, ScalarBuffer};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use common::*;
use dartunifrac_arrow::{
    sketches_from_stream, sketches_reader, sketches_schema, trim_to_bbits, MarshalError, Sketches,
    FINGERPRINT, SAMPLE_IDX, SKETCH,
};

/// Values per sketch. Not a power of two, so a k mixed up with anything else
/// shows.
const K: usize = 24;
const WIDTHS: [u8; 3] = [16, 32, 64];

/// The caller's sample indices. Gapped and starting past zero's neighbours, as
/// the kept ids of a real set are, so a reader that numbered rows instead of
/// carrying ids would be caught.
fn ids() -> Vec<i64> {
    vec![0, 2, 3, 7, 9, 12, 40]
}

/// A fingerprint whose bytes all differ, so a reversed or shifted copy is not
/// equal to it.
fn fp() -> [u8; 32] {
    std::array::from_fn(|i| (i as u8).wrapping_mul(7).wrapping_add(1))
}

/// The same fingerprint with its last bit flipped: another key.
fn other() -> [u8; 32] {
    let mut f = fp();
    f[31] ^= 1;
    f
}

/// Seven sketches over the full 64-bit range, narrowed to `bbits`. The full
/// range is what makes the 64-bit round trip mean something: a narrowing or a
/// sign bug shows only in the high bits.
fn set(bbits: u8) -> Sketches {
    trim_to_bbits(fake_sketches(ids().len(), K, u64::MAX), bbits).expect("a supported width")
}

/// A set's values widened to u64, with its width, so any two compare alike.
fn widen(s: &Sketches) -> (u8, Vec<Vec<u64>>) {
    fn up<T: Copy + Into<u64>>(rows: &[Vec<T>]) -> Vec<Vec<u64>> {
        rows.iter().map(|r| r.iter().map(|&x| x.into()).collect()).collect()
    }
    match s {
        Sketches::U16(r) => (16, up(r)),
        Sketches::U32(r) => (32, up(r)),
        Sketches::U64(r) => (64, up(r)),
    }
}

fn uint(bbits: u8) -> DataType {
    match bbits {
        16 => DataType::UInt16,
        32 => DataType::UInt32,
        64 => DataType::UInt64,
        other => panic!("no width {other}"),
    }
}

/// The schema the export is documented to write, spelled out here rather than
/// taken from `sketches_schema`, so the two are checked against each other.
fn documented(bbits: u8) -> Schema {
    Schema::new(vec![
        Field::new("sample_idx", DataType::Int64, false),
        Field::new("fingerprint", DataType::FixedSizeBinary(32), false),
        Field::new(
            "sketch",
            DataType::FixedSizeList(Arc::new(Field::new("item", uint(bbits), false)), K as i32),
            false,
        ),
    ])
}

fn export(sk: Sketches, ids: Vec<i64>, rows_per_batch: usize) -> Vec<RecordBatch> {
    drain(sketches_reader(sk, ids, K, fp(), rows_per_batch).expect("a consistent set"))
}

fn import(
    batches: Vec<RecordBatch>,
    k: usize,
    bbits: u8,
) -> Result<(Sketches, Vec<i64>), MarshalError> {
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => Arc::new(documented(bbits)),
    };
    sketches_from_stream(reader(schema, batches), k, bbits, &fp())
}

fn schema_err<T>(r: Result<T, MarshalError>) -> String {
    match r.marshal_err() {
        MarshalError::Schema(m) => m,
        other => panic!("expected MarshalError::Schema, got {other:?}"),
    }
}

fn data_err<T>(r: Result<T, MarshalError>) -> String {
    match r.marshal_err() {
        MarshalError::Data(m) => m,
        other => panic!("expected MarshalError::Data, got {other:?}"),
    }
}

fn sketch_err<T>(r: Result<T, MarshalError>) -> String {
    match r.marshal_err() {
        MarshalError::Sketch(m) => m,
        other => panic!("expected MarshalError::Sketch, got {other:?}"),
    }
}

// ---------------------------------------------------------------- stored rows

/// One stored row, every part optional so a test can null any of them.
#[derive(Clone)]
struct Row {
    id: Option<i64>,
    fp: Option<Vec<u8>>,
    sketch: Option<Vec<Option<u64>>>,
}

/// `sk` as the rows a store holds, one per id, each carrying [`fp`].
fn rows_of_set(sk: &Sketches, ids: &[i64]) -> Vec<Row> {
    let (_, values) = widen(sk);
    assert_eq!(values.len(), ids.len());
    ids.iter()
        .zip(values)
        .map(|(&id, v)| Row {
            id: Some(id),
            fp: Some(fp().to_vec()),
            sketch: Some(v.into_iter().map(Some).collect()),
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
enum Lists {
    Fixed,
    List,
    Large,
}

#[derive(Clone, Copy, Debug)]
enum Bytes {
    Fixed,
    Binary,
    Large,
}

/// How a store hands the rows back: the column types, the element type, and
/// the lengths the fixed-size types declare.
#[derive(Clone)]
struct Shape {
    lists: Lists,
    bytes: Bytes,
    elem: DataType,
    k: i32,
    fp_len: i32,
}

/// The shape the export writes.
fn fixed(bbits: u8) -> Shape {
    Shape {
        lists: Lists::Fixed,
        bytes: Bytes::Fixed,
        elem: uint(bbits),
        k: K as i32,
        fp_len: 32,
    }
}

fn fixed_lists<P: ArrowPrimitiveType>(
    rows: &[Row],
    k: i32,
    conv: impl Fn(u64) -> P::Native,
) -> ArrayRef {
    let child = Field::new("element", P::DATA_TYPE, true);
    let mut b = FixedSizeListBuilder::new(PrimitiveBuilder::<P>::new(), k).with_field(child);
    for r in rows {
        match &r.sketch {
            Some(s) => {
                for v in s {
                    b.values().append_option(v.map(&conv));
                }
                b.append(true);
            }
            None => {
                b.values().append_nulls(k as usize);
                b.append(false);
            }
        }
    }
    Arc::new(b.finish())
}

fn var_lists<O: OffsetSizeTrait, P: ArrowPrimitiveType>(
    rows: &[Row],
    conv: impl Fn(u64) -> P::Native,
) -> ArrayRef {
    let child = Field::new("element", P::DATA_TYPE, true);
    let mut b = GenericListBuilder::<O, _>::new(PrimitiveBuilder::<P>::new()).with_field(child);
    for r in rows {
        if let Some(s) = &r.sketch {
            for v in s {
                b.values().append_option(v.map(&conv));
            }
        }
        b.append(r.sketch.is_some());
    }
    Arc::new(b.finish())
}

fn sketch_column(rows: &[Row], shape: &Shape) -> ArrayRef {
    macro_rules! of {
        ($p:ty, $native:ty) => {{
            let conv = |x: u64| x as $native;
            match shape.lists {
                Lists::Fixed => fixed_lists::<$p>(rows, shape.k, conv),
                Lists::List => var_lists::<i32, $p>(rows, conv),
                Lists::Large => var_lists::<i64, $p>(rows, conv),
            }
        }};
    }
    match &shape.elem {
        DataType::UInt16 => of!(UInt16Type, u16),
        DataType::UInt32 => of!(UInt32Type, u32),
        DataType::UInt64 => of!(UInt64Type, u64),
        DataType::Int16 => of!(Int16Type, i16),
        DataType::Int64 => of!(Int64Type, i64),
        other => panic!("no fixture for {other:?}"),
    }
}

fn fingerprint_column(rows: &[Row], shape: &Shape) -> ArrayRef {
    let values = rows.iter().map(|r| r.fp.clone());
    match shape.bytes {
        Bytes::Fixed => Arc::new(
            FixedSizeBinaryArray::try_from_sparse_iter_with_size(values, shape.fp_len)
                .expect("fixture fingerprints match the declared length"),
        ),
        Bytes::Binary => Arc::new(values.collect::<BinaryArray>()),
        Bytes::Large => Arc::new(values.collect::<LargeBinaryArray>()),
    }
}

/// `rows` as one batch in `shape`, every field nullable and the list's child
/// named `element`: the form DuckDB and Parquet hand rows back in, rather than
/// the form the export writes.
fn stored(rows: &[Row], shape: &Shape) -> RecordBatch {
    let ids: ArrayRef = Arc::new(Int64Array::from(rows.iter().map(|r| r.id).collect::<Vec<_>>()));
    RecordBatch::try_from_iter_with_nullable(vec![
        (SAMPLE_IDX, ids, true),
        (FINGERPRINT, fingerprint_column(rows, shape), true),
        (SKETCH, sketch_column(rows, shape), true),
    ])
    .expect("fixture batch is well formed")
}

/// `batch` with the column `name` swapped for `col`.
fn replace(batch: &RecordBatch, name: &str, col: ArrayRef) -> RecordBatch {
    let schema = batch.schema();
    let cols = schema.fields().iter().zip(batch.columns()).map(|(f, c)| {
        let c = if f.name() == name { col.clone() } else { c.clone() };
        (f.name().clone(), c, true)
    });
    RecordBatch::try_from_iter_with_nullable(cols).expect("replacement has the batch's length")
}

// ----------------------------------------------------------------- round trip

/// The property the module exists for: a set written as rows and read back is
/// the set, bit for bit, at every width the distance kernel compares.
#[test]
fn a_set_survives_export_and_import_at_every_width() {
    for bbits in WIDTHS {
        let want = widen(&set(bbits));
        assert!(want.1.windows(2).all(|w| w[0] != w[1]), "fixture rows must differ");
        let (sk, got) = import(export(set(bbits), ids(), 2), K, bbits).unwrap();
        assert_eq!(got, ids(), "bbits {bbits}");
        assert_eq!(widen(&sk), want, "bbits {bbits}");
    }
}

/// What a caller's store receives, read back here independently: the
/// documented schema exactly, one row per sample in the order given, and the
/// set's fingerprint on every row.
#[test]
fn export_writes_the_documented_schema_and_one_row_per_sample() {
    for bbits in WIDTHS {
        assert_eq!(sketches_schema(K, bbits).unwrap(), documented(bbits), "bbits {bbits}");
        let r = sketches_reader(set(bbits), ids(), K, fp(), 3).unwrap();
        assert_eq!(*r.schema(), documented(bbits), "bbits {bbits}");
        let batches = drain(r);
        for b in &batches {
            assert_eq!(*b.schema(), documented(bbits), "bbits {bbits}");
        }
        let rows = sketch_rows(&batches);
        let want = widen(&set(bbits)).1;
        assert_eq!(rows.len(), ids().len(), "bbits {bbits}");
        for (r, (id, f, values)) in rows.into_iter().enumerate() {
            assert_eq!(id, ids()[r], "bbits {bbits}, row {r}");
            assert_eq!(f, fp(), "bbits {bbits}, row {r}");
            assert_eq!(values, want[r], "bbits {bbits}, row {r}");
        }
    }
}

/// `rows_per_batch` bounds every batch, and only the last may be short. Zero
/// is clamped to one rather than producing empty batches forever, as the
/// distance readers clamp it.
#[test]
fn export_batches_hold_at_most_rows_per_batch() {
    for (rows_per_batch, sizes) in [
        (0, vec![1; 7]),
        (1, vec![1; 7]),
        (3, vec![3, 3, 1]),
        (7, vec![7]),
        (100, vec![7]),
    ] {
        let got: Vec<usize> = export(set(16), ids(), rows_per_batch)
            .iter()
            .map(|b| b.num_rows())
            .collect();
        assert_eq!(got, sizes, "rows_per_batch {rows_per_batch}");
    }
}

/// An empty set still says what it would hold. Its schema carries k and the
/// width, so a store created from it accepts the rows of a later set.
#[test]
fn an_empty_set_exports_as_no_rows_under_the_full_schema() {
    for bbits in WIDTHS {
        let empty = trim_to_bbits(Vec::new(), bbits).unwrap();
        let r = sketches_reader(empty, Vec::new(), K, fp(), 3).unwrap();
        assert_eq!(*r.schema(), documented(bbits), "bbits {bbits}");
        assert_eq!(drain(r).iter().map(|b| b.num_rows()).sum::<usize>(), 0);
    }
}

/// Zero rows is an empty set, not an error, as fewer than two samples is for
/// `df_sketch`: a query that matches nothing in a store must still import. Its
/// width follows `bbits`, so it combines like any other set.
#[test]
fn an_empty_stream_is_an_empty_set_of_the_requested_width() {
    for bbits in WIDTHS {
        let schema = Arc::new(documented(bbits));
        for batches in [vec![], vec![RecordBatch::new_empty(schema.clone())]] {
            let (sk, got) =
                sketches_from_stream(reader(schema.clone(), batches), K, bbits, &fp()).unwrap();
            assert!(got.is_empty(), "bbits {bbits}");
            assert_eq!(widen(&sk), (bbits, vec![]), "bbits {bbits}");
        }
    }
}

/// A store returns rows in whatever order its scan produces, batched however it
/// likes, and neither may show in the set. The batches here are slices of one,
/// so they carry offsets, as a reader's batches often do.
#[test]
fn import_is_invariant_to_row_order_and_batching() {
    let rows = shuffled(&rows_of_set(&set(32), &ids()));
    let order: Vec<i64> = rows.iter().map(|r| r.id.unwrap()).collect();
    assert!(order.windows(2).any(|w| w[0] > w[1]), "the shuffle left {order:?} sorted");
    let one = stored(&rows, &fixed(32));
    let batches = vec![one.slice(0, 1), one.slice(1, 4), one.slice(5, 2)];
    let (sk, got) = import(batches, K, 32).unwrap();
    assert_eq!(got, ids());
    assert_eq!(widen(&sk), widen(&set(32)));
}

/// Decision 3, measured with DuckDB 1.3.2 and pyarrow 15: DuckDB hands a
/// `fixed_size_list` back as one, but through Parquet it comes back as `list`,
/// and a `fixed_size_binary` comes back as `binary`, with every field nullable
/// and the list's child renamed. Each must import as the original. The large
/// variants are what a producer with 64-bit offsets sends.
#[test]
fn import_accepts_the_types_duckdb_and_parquet_hand_back() {
    for bbits in WIDTHS {
        for lists in [Lists::Fixed, Lists::List, Lists::Large] {
            for bytes in [Bytes::Fixed, Bytes::Binary, Bytes::Large] {
                let shape = Shape { lists, bytes, ..fixed(bbits) };
                let batch = stored(&rows_of_set(&set(bbits), &ids()), &shape);
                let label = format!("{lists:?} of UInt{bbits}, {bytes:?} fingerprint");
                let (sk, got) =
                    import(vec![batch], K, bbits).unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(got, ids(), "{label}");
                assert_eq!(widen(&sk), widen(&set(bbits)), "{label}");
            }
        }
    }
}

/// Columns are found by name, so a store's own columns (a study, a timestamp)
/// may ride along and are ignored, as the table reader ignores them.
#[test]
fn extra_columns_are_ignored() {
    let batch = stored(&rows_of_set(&set(16), &ids()), &fixed(16));
    let study: ArrayRef = Arc::new(StringArray::from(vec!["emp"; ids().len()]));
    let mut cols: Vec<(String, ArrayRef, bool)> = vec![("study".into(), study, true)];
    for (f, c) in batch.schema().fields().iter().zip(batch.columns()) {
        cols.push((f.name().clone(), c.clone(), true));
    }
    let batch = RecordBatch::try_from_iter_with_nullable(cols).unwrap();
    let (sk, got) = import(vec![batch], K, 16).unwrap();
    assert_eq!(got, ids());
    assert_eq!(widen(&sk), widen(&set(16)));
}

// ----------------------------------------------------------------- rejections

/// A projection that drops or misnames a column is told which one.
#[test]
fn a_missing_column_is_named() {
    let full = stored(&rows_of_set(&set(16), &ids()), &fixed(16));
    for missing in [SAMPLE_IDX, FINGERPRINT, SKETCH] {
        let keep: Vec<usize> = (0..3)
            .filter(|&i| full.schema().field(i).name() != missing)
            .collect();
        let m = schema_err(import(vec![full.project(&keep).unwrap()], K, 16));
        assert!(m.contains(missing), "should name {missing:?}; got {m:?}");
    }
}

/// A sketch read at another width is another sketch. The kernel compares the
/// low `bbits` of each value, so u32 values read for bbits 16, or a SMALLINT
/// column read as USMALLINT, would compare numbers no sketcher produced. Only
/// exactly `UInt{bbits}` is accepted.
#[test]
fn a_sketch_of_another_element_type_is_rejected() {
    for (bbits, elem) in [
        (16, DataType::Int16),
        (16, DataType::UInt32),
        (32, DataType::UInt16),
        (32, DataType::UInt64),
        (64, DataType::Int64),
        (64, DataType::UInt32),
    ] {
        let shape = Shape { elem: elem.clone(), ..fixed(bbits) };
        let batch = stored(&rows_of_set(&set(bbits), &ids()), &shape);
        let m = schema_err(import(vec![batch], K, bbits));
        let want = format!("{:?}", uint(bbits));
        assert!(
            m.contains(SKETCH) && m.contains(&want),
            "{elem:?} for bbits {bbits} should name {SKETCH:?} and {want}; got {m:?}"
        );
    }
}

/// Each column must be of a kind that can hold what it names: an integer index,
/// bytes, and a list.
#[test]
fn a_column_of_another_kind_is_rejected_and_named() {
    let batch = stored(&rows_of_set(&set(16), &ids()), &fixed(16));
    let n = ids().len();
    let cases: [(&str, ArrayRef); 3] = [
        (SAMPLE_IDX, Arc::new(Int32Array::from(vec![0; n]))),
        (FINGERPRINT, Arc::new(StringArray::from(vec!["x"; n]))),
        (SKETCH, Arc::new(UInt16Array::from(vec![0; n]))),
    ];
    for (name, col) in cases {
        let m = schema_err(import(vec![replace(&batch, name, col)], K, 16));
        assert!(m.contains(name), "should name {name:?}; got {m:?}");
    }
}

/// A `fixed_size_list` declares its length in its type, so a store of another
/// k is refused whole, naming both lengths.
#[test]
fn a_fixed_size_list_of_another_length_is_rejected() {
    let longer = trim_to_bbits(fake_sketches(ids().len(), K + 1, u64::MAX), 16).unwrap();
    let shape = Shape { k: K as i32 + 1, ..fixed(16) };
    let batch = stored(&rows_of_set(&longer, &ids()), &shape);
    let m = schema_err(import(vec![batch], K, 16));
    assert!(m.contains("25") && m.contains("24"), "should name both lengths; got {m:?}");
}

/// A variable-length list can hold a row of any length. One that is not k is a
/// truncated or foreign sketch, and its row is named so it can be found.
#[test]
fn a_list_row_of_another_length_is_rejected_naming_the_row() {
    for lists in [Lists::List, Lists::Large] {
        let mut rows = rows_of_set(&set(16), &ids());
        rows[3].sketch.as_mut().unwrap().pop();
        let batch = stored(&rows, &Shape { lists, ..fixed(16) });
        let m = data_err(import(vec![batch], K, 16));
        let sample = format!("sample_idx {}", ids()[3]);
        assert!(
            m.contains("row 3") && m.contains(&sample) && m.contains("23"),
            "{lists:?}: should name row 3, its {sample} and its length; got {m:?}"
        );
    }
}

/// Every part of a row is required. A null is refused naming its row, counted
/// across batches, so a hole in a store is found rather than read as zeros.
/// Row 3 is the second batch's second row here, so a per-batch count would say
/// row 1.
#[test]
fn a_null_anywhere_in_a_row_is_rejected_naming_the_row() {
    type Spoil = fn(&mut Row);
    let cases: [(&str, &str, Spoil); 4] = [
        ("a null sample_idx", SAMPLE_IDX, |r| r.id = None),
        ("a null fingerprint", FINGERPRINT, |r| r.fp = None),
        ("a null sketch", SKETCH, |r| r.sketch = None),
        ("a null element", SKETCH, |r| r.sketch.as_mut().unwrap()[5] = None),
    ];
    for lists in [Lists::Fixed, Lists::List] {
        for (label, column, spoil) in cases {
            let mut rows = rows_of_set(&set(16), &ids());
            spoil(&mut rows[3]);
            let batch = stored(&rows, &Shape { lists, ..fixed(16) });
            let m = schema_err(import(vec![batch.slice(0, 2), batch.slice(2, 5)], K, 16));
            assert!(
                m.contains(column) && m.contains("row 3"),
                "{label} in a {lists:?}: should name {column:?} and row 3; got {m:?}"
            );
        }
    }
}

/// A sample index is the caller's own, and no caller numbers samples below 0.
#[test]
fn a_negative_sample_idx_is_rejected_naming_the_row() {
    let mut rows = rows_of_set(&set(16), &ids());
    rows[2].id = Some(-5);
    let m = data_err(import(vec![stored(&rows, &fixed(16))], K, 16));
    assert!(m.contains("row 2") && m.contains("-5"), "{m:?}");
}

/// Two rows for one sample would make the set depend on which one the store
/// returned last. An exact copy, which merging two overlapping stores makes, is
/// refused too: the overlap is the caller's bug either way. With three rows for
/// one sample, the first two are named, so the message does not depend on how
/// the sort happened to order them.
#[test]
fn a_repeated_sample_idx_is_rejected_naming_it() {
    for exact in [false, true] {
        let mut rows = rows_of_set(&set(16), &ids());
        let mut extra = if exact { rows[1].clone() } else { rows[4].clone() };
        extra.id = Some(ids()[1]);
        rows.push(extra.clone());
        rows.push(extra);
        let batch = stored(&rows, &fixed(16));
        let m = data_err(import(vec![batch.slice(0, 4), batch.slice(4, 5)], K, 16));
        let sample = format!("sample_idx {}", ids()[1]);
        assert!(
            m.contains(&sample) && m.contains("rows 1 and 7"),
            "exact copy {exact}: should name {sample} at rows 1 and 7; got {m:?}"
        );
    }
}

/// A fingerprint has one length, whatever the store's column type allows.
#[test]
fn a_fingerprint_of_another_length_is_rejected() {
    let mut rows = rows_of_set(&set(16), &ids());
    rows[1].fp.as_mut().unwrap().pop();
    let batch = stored(&rows, &Shape { bytes: Bytes::Binary, ..fixed(16) });
    let m = data_err(import(vec![batch], K, 16));
    assert!(m.contains("row 1") && m.contains("31"), "{m:?}");

    for r in &mut rows {
        r.fp = Some(fp()[..31].to_vec());
    }
    let batch = stored(&rows, &Shape { fp_len: 31, ..fixed(16) });
    let m = schema_err(import(vec![batch], K, 16));
    assert!(m.contains(FINGERPRINT) && m.contains("31"), "{m:?}");
}

/// arrow-rs reads a byte array's values through its offsets unchecked, and
/// imports a C producer's arrays without validating them, sizing a binary's data
/// by its last offset alone. So a producer whose offsets start past the end and
/// step by 32 passes every length check and points each read outside the data.
/// That column must be refused before any value is read, not compared, and not
/// printed into an error.
#[test]
fn a_byte_fingerprint_column_with_offsets_outside_its_data_is_refused_unread() {
    fn bad<O: arrow_array::OffsetSizeTrait>(n: usize) -> ArrayRef {
        let data = fp().repeat(n);
        let mut offsets: Vec<O> = (0..n).map(|r| O::usize_as(4096 + 32 * r)).collect();
        offsets.push(O::usize_as(data.len()));
        let offsets = unsafe { OffsetBuffer::new_unchecked(ScalarBuffer::from(offsets)) };
        Arc::new(unsafe {
            arrow_array::GenericBinaryArray::<O>::new_unchecked(offsets, Buffer::from_vec(data), None)
        })
    }
    let batch = stored(&rows_of_set(&set(16), &ids()), &Shape { bytes: Bytes::Binary, ..fixed(16) });
    for (label, col) in [("binary", bad::<i32>(ids().len())), ("large_binary", bad::<i64>(ids().len()))] {
        let m = data_err(import(vec![replace(&batch, FINGERPRINT, col)], K, 16));
        assert!(
            m.contains(FINGERPRINT) && m.contains("invalid offsets"),
            "{label}: should be refused for its offsets; got {m:?}"
        );
    }
}

/// The point of the column. A row sketched under another key is refused,
/// naming the row and its sample, rather than compared and turned into
/// plausible, wrong distances.
#[test]
fn a_row_of_another_fingerprint_is_rejected_naming_it() {
    let mut rows = rows_of_set(&set(16), &ids());
    rows[5].fp = Some(other().to_vec());
    let m = data_err(import(vec![stored(&rows, &fixed(16))], K, 16));
    let sample = format!("sample_idx {}", ids()[5]);
    assert!(
        m.contains("row 5") && m.contains(&sample) && m.contains(FINGERPRINT),
        "should name row 5, its {sample} and the fingerprint; got {m:?}"
    );
}

/// The fingerprint is checked before the sketch's shape. Rows of another k or
/// width were made under another key, and saying so names the cause; "expected
/// UInt16" would send the caller to their column types instead.
#[test]
fn another_keys_rows_are_reported_as_such_before_their_shape() {
    let longer = trim_to_bbits(fake_sketches(ids().len(), K + 1, u64::MAX), 32).unwrap();
    let mut rows = rows_of_set(&longer, &ids());
    for r in &mut rows {
        r.fp = Some(other().to_vec());
    }
    let batch = stored(&rows, &Shape { k: K as i32 + 1, ..fixed(32) });
    let m = data_err(import(vec![batch], K, 16));
    assert!(m.contains(FINGERPRINT) && m.contains("row 0"), "{m:?}");
}

/// The store's own failure is reported as the store's, text intact.
#[test]
fn a_failing_stream_is_a_stream_error_with_its_text() {
    let batch = stored(&rows_of_set(&set(16), &ids()), &fixed(16));
    let items = vec![Ok(batch.clone()), Err(ArrowError::ComputeError("the scan gave up".into()))];
    let r = RecordBatchIterator::new(items, batch.schema());
    match sketches_from_stream(r, K, 16, &fp()).marshal_err() {
        MarshalError::Stream(m) => assert!(m.contains("the scan gave up"), "{m:?}"),
        other => panic!("expected MarshalError::Stream, got {other:?}"),
    }
}

// ------------------------------------------------------------------ arguments

/// The arguments are the caller's sketching parameters. A width the kernel
/// does not compare, or a sketch of no values, is refused: every distance from
/// such a set would be meaningless.
#[test]
fn an_unsupported_width_or_a_zero_k_is_refused() {
    let none = || reader(Arc::new(documented(16)), vec![]);
    sketch_err(sketches_from_stream(none(), K, 8, &fp()));
    sketch_err(sketches_from_stream(none(), 0, 16, &fp()));
    sketch_err(sketches_schema(K, 8));
    sketch_err(sketches_schema(0, 16));
}

/// A `fixed_size_list` length is an i32 in the Arrow format, so a k past
/// i32::MAX cannot be described. It is refused rather than wrapped.
#[test]
fn a_k_beyond_i32_cannot_be_exported() {
    let k = i32::MAX as usize + 1;
    let m = sketch_err(sketches_schema(k, 16));
    assert!(m.contains("2147483648"), "{m:?}");
    sketch_err(sketches_reader(Sketches::U16(vec![]), vec![], k, fp(), 1));
}

/// The reader sits beneath the C API as the distance reader does, and refuses
/// the same inconsistent sets, rather than writing rows no import would accept.
#[test]
fn the_reader_refuses_a_set_its_ids_or_k_do_not_describe() {
    let short: Vec<i64> = ids()[..6].to_vec();
    sketch_err(sketches_reader(set(16), short, K, fp(), 3));
    sketch_err(sketches_reader(set(16), ids(), K + 1, fp(), 3));
}
