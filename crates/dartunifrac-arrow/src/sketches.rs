//! A sketch set as Arrow rows, out and back in, so a caller can store one.
//!
//! One row per sample: its `sample_idx`, the set's fingerprint and its sketch.
//! The fingerprint is opaque bytes here; what it covers is the C API's business.
//! It is a column rather than schema metadata because DuckDB drops Arrow schema
//! metadata, and a stored row has to carry its own key.

use std::ops::Range;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, UInt16Type, UInt32Type, UInt64Type};
use arrow_array::{
    Array, ArrayAccessor, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray,
    GenericListArray, Int64Array, OffsetSizeTrait, PrimitiveArray, RecordBatch, RecordBatchReader,
};
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Schema, SchemaRef};

use crate::table::{INDEX_TYPE, SAMPLE_IDX};
use crate::{MarshalError, Sketches};

/// Column holding each row's fingerprint.
pub const FINGERPRINT: &str = "fingerprint";
/// Column holding each row's sketch.
pub const SKETCH: &str = "sketch";
/// Bytes in a fingerprint.
pub const FINGERPRINT_LEN: usize = 32;

/// The schema [`sketches_reader`] produces: `sample_idx int64`, `fingerprint
/// fixed_size_binary(32)` and `sketch fixed_size_list<uint{bbits}>[k]`, none of
/// them nullable.
///
/// The sketch is a `fixed_size_list` because its length is part of what the set
/// is: a store that keeps the type keeps k with it. `bbits` must be 16, 32 or
/// 64, and `k` from 1 to `i32::MAX`, the longest list the Arrow format can
/// declare.
pub fn sketches_schema(k: usize, bbits: u8) -> Result<Schema, MarshalError> {
    let (item, k) = sketch_item(k, bbits)?;
    Ok(schema_of(item, k))
}

fn schema_of(item: FieldRef, k: i32) -> Schema {
    Schema::new(vec![
        Field::new(SAMPLE_IDX, INDEX_TYPE, false),
        Field::new(
            FINGERPRINT,
            DataType::FixedSizeBinary(FINGERPRINT_LEN as i32),
            false,
        ),
        Field::new(SKETCH, DataType::FixedSizeList(item, k), false),
    ])
}

/// The sketch list's element field, and k as the `i32` Arrow declares it in.
fn sketch_item(k: usize, bbits: u8) -> Result<(FieldRef, i32), MarshalError> {
    let element = element_type(bbits)?;
    nonzero(k)?;
    let k = i32::try_from(k).map_err(|_| {
        MarshalError::Sketch(format!(
            "k = {k} is more than {}, the longest fixed_size_list Arrow can declare",
            i32::MAX
        ))
    })?;
    Ok((Arc::new(Field::new_list_field(element, false)), k))
}

/// The element type a sketch of `bbits` is stored as.
fn element_type(bbits: u8) -> Result<DataType, MarshalError> {
    match bbits {
        16 => Ok(DataType::UInt16),
        32 => Ok(DataType::UInt32),
        64 => Ok(DataType::UInt64),
        other => Err(unsupported(other)),
    }
}

fn width(sketches: &Sketches) -> u8 {
    match sketches {
        Sketches::U16(_) => 16,
        Sketches::U32(_) => 32,
        Sketches::U64(_) => 64,
    }
}

/// How many rows there are, and the first one whose length is not `k`.
fn shape<T>(rows: &[Vec<T>], k: usize) -> (usize, Option<(usize, usize)>) {
    let misfit = rows
        .iter()
        .enumerate()
        .find(|(_, r)| r.len() != k)
        .map(|(s, r)| (s, r.len()));
    (rows.len(), misfit)
}

/// Stream `sketches` as rows under [`sketches_schema`], in the order given, at
/// most `rows_per_batch` rows per batch. Every row carries `fingerprint`.
///
/// `ids` are the caller's `sample_idx` for each sketch, and `k` is the length
/// every sketch must have. It is passed rather than measured because an empty
/// set has no sketch to measure, and its schema must still say k. A set its
/// `ids` or `k` do not describe is refused here, as [`crate::distances_reader`]
/// refuses one, rather than written as rows no import would accept. An empty
/// set gives the schema and no batches.
///
/// `rows_per_batch` only frames the output, so 0 is clamped to 1 rather than
/// refused, as the distance readers clamp it. `sketches` is shared, not copied,
/// when passed as an `Arc`, so the stream stays valid after the C API's handle
/// is freed.
pub fn sketches_reader(
    sketches: impl Into<Arc<Sketches>>,
    ids: Vec<i64>,
    k: usize,
    fingerprint: [u8; FINGERPRINT_LEN],
    rows_per_batch: usize,
) -> Result<Box<dyn RecordBatchReader + Send>, MarshalError> {
    let sketches = sketches.into();
    let (item, list_len) = sketch_item(k, width(&sketches))?;
    let (count, misfit) = match &*sketches {
        Sketches::U16(rows) => shape(rows, k),
        Sketches::U32(rows) => shape(rows, k),
        Sketches::U64(rows) => shape(rows, k),
    };
    if ids.len() != count {
        return Err(MarshalError::Sketch(format!(
            "{} ids were given for {count} sketches; they index the same samples",
            ids.len()
        )));
    }
    if let Some((s, len)) = misfit {
        return Err(MarshalError::Sketch(format!(
            "sketch {s} holds {len} values, but k is {k}"
        )));
    }
    let schema = Arc::new(schema_of(item.clone(), list_len));
    Ok(Box::new(SketchesReader {
        sketches,
        ids,
        fingerprint,
        item,
        k: list_len,
        rows_per_batch: rows_per_batch.max(1),
        next_row: 0,
        schema,
    }))
}

struct SketchesReader {
    sketches: Arc<Sketches>,
    ids: Vec<i64>,
    fingerprint: [u8; FINGERPRINT_LEN],
    item: FieldRef,
    k: i32,
    rows_per_batch: usize,
    next_row: usize,
    schema: SchemaRef,
}

impl SketchesReader {
    fn batch(&self, rows: Range<usize>) -> Result<RecordBatch, ArrowError> {
        let n = rows.len();
        let ids = Int64Array::from(self.ids[rows.clone()].to_vec());
        let fingerprints = FixedSizeBinaryArray::try_new(
            FINGERPRINT_LEN as i32,
            self.fingerprint.repeat(n).into(),
            None,
        )?;
        let sketches = match &*self.sketches {
            Sketches::U16(r) => sketch_column::<UInt16Type>(&r[rows], self.k, &self.item),
            Sketches::U32(r) => sketch_column::<UInt32Type>(&r[rows], self.k, &self.item),
            Sketches::U64(r) => sketch_column::<UInt64Type>(&r[rows], self.k, &self.item),
        }?;
        RecordBatch::try_new(
            self.schema.clone(),
            vec![Arc::new(ids), Arc::new(fingerprints), sketches],
        )
    }
}

/// `rows` as one `fixed_size_list` column: their values laid end to end.
fn sketch_column<T: ArrowPrimitiveType>(
    rows: &[Vec<T::Native>],
    k: i32,
    item: &FieldRef,
) -> Result<ArrayRef, ArrowError> {
    let mut values = Vec::with_capacity(rows.len() * k as usize);
    for r in rows {
        values.extend_from_slice(r);
    }
    let values = PrimitiveArray::<T>::try_new(values.into(), None)?;
    Ok(Arc::new(FixedSizeListArray::try_new(
        item.clone(),
        k,
        Arc::new(values),
        None,
    )?))
}

impl Iterator for SketchesReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let n = self.ids.len();
        if self.next_row >= n {
            return None;
        }
        let rows = self.next_row..self.next_row.saturating_add(self.rows_per_batch).min(n);
        self.next_row = rows.end;
        Some(self.batch(rows))
    }
}

impl RecordBatchReader for SketchesReader {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

/// Rebuild a sketch set from rows in any order and any batching, as a store
/// hands them back. Returns the sketches and their `sample_idx`, ascending, so
/// an export read back is the set that was exported, and two exports read back
/// as one are the set a joint run would have made.
///
/// Columns are matched by name and others are ignored. Besides the types
/// [`sketches_schema`] writes, a `sketch` column of `list` or `large_list` and a
/// `fingerprint` column of `binary` or `large_binary` are accepted, with any
/// nullability and any child name: Parquet hands a `fixed_size_list` back as a
/// `list`, and DuckDB a `fixed_size_binary` as a `binary`. The elements must be
/// exactly `uint{bbits}`, since a sketch read at another width is another
/// sketch, and every sketch must hold `k` values.
///
/// Every row's fingerprint must equal `fingerprint`, and it is checked before
/// the sketch column, so rows made with another k or width are reported as made
/// under another key, which is the cause. A null anywhere, a negative or
/// repeated `sample_idx`, and a fingerprint of another length are refused too.
/// Each error names its row, counted from 0 across the whole stream. Within a
/// batch the columns are checked in turn (`sample_idx`, then `fingerprint`,
/// then `sketch`), so the error reported is the first in that order, not
/// necessarily the earliest row. A batch with no rows is skipped unread, so
/// zero rows give an empty set of the requested width whatever the stream's
/// schema.
pub fn sketches_from_stream<R: RecordBatchReader>(
    reader: R,
    k: usize,
    bbits: u8,
    fingerprint: &[u8; FINGERPRINT_LEN],
) -> Result<(Sketches, Vec<i64>), MarshalError> {
    nonzero(k)?;
    Ok(match bbits {
        16 => {
            let (rows, ids) = read::<UInt16Type>(reader, k, fingerprint)?;
            (Sketches::U16(rows), ids)
        }
        32 => {
            let (rows, ids) = read::<UInt32Type>(reader, k, fingerprint)?;
            (Sketches::U32(rows), ids)
        }
        64 => {
            let (rows, ids) = read::<UInt64Type>(reader, k, fingerprint)?;
            (Sketches::U64(rows), ids)
        }
        other => return Err(unsupported(other)),
    })
}

/// One sketch per row, as [`Sketches`] holds them.
type Rows<T> = Vec<Vec<T>>;

/// Drain `reader` into sketches of `T`, then put them in `sample_idx` order.
fn read<T: ArrowPrimitiveType>(
    reader: impl RecordBatchReader,
    k: usize,
    fingerprint: &[u8; FINGERPRINT_LEN],
) -> Result<(Rows<T::Native>, Vec<i64>), MarshalError> {
    let mut ids: Vec<i64> = Vec::new();
    let mut rows: Rows<T::Native> = Vec::new();
    let mut first = 0usize; // the stream's row number of this batch's row 0
    for batch in reader {
        let batch = batch.map_err(|e| MarshalError::Stream(e.to_string()))?;
        if batch.num_rows() == 0 {
            continue;
        }
        let batch_ids = id_column(&batch, first)?;
        check_fingerprints(&batch, first, batch_ids, fingerprint)?;
        read_sketches::<T>(&batch, first, batch_ids, k, &mut rows)?;
        ids.extend_from_slice(batch_ids.values());
        first += batch.num_rows();
    }
    ascending(ids, rows)
}

fn nonzero(k: usize) -> Result<(), MarshalError> {
    if k == 0 {
        return Err(MarshalError::Sketch(
            "k is 0; a sketch needs at least one value".into(),
        ));
    }
    Ok(())
}

fn unsupported(bbits: u8) -> MarshalError {
    MarshalError::Sketch(format!("bbits {bbits} is not supported; use 16, 32 or 64"))
}

fn column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a ArrayRef, MarshalError> {
    batch.column_by_name(name).ok_or_else(|| {
        MarshalError::Schema(format!(
            "required column {name:?} is missing; columns are matched by name"
        ))
    })
}

/// The first null in `a`, if it has one.
fn first_null(a: &dyn Array) -> Option<usize> {
    if a.null_count() == 0 {
        return None;
    }
    (0..a.len()).find(|&i| a.is_null(i))
}

/// The batch's `sample_idx` column, with no nulls and no negative index.
fn id_column(batch: &RecordBatch, first: usize) -> Result<&Int64Array, MarshalError> {
    let col = column(batch, SAMPLE_IDX)?;
    let ids = col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        MarshalError::Schema(format!(
            "column {SAMPLE_IDX:?} is {:?}, expected {INDEX_TYPE:?}",
            col.data_type()
        ))
    })?;
    if let Some(r) = first_null(ids) {
        return Err(MarshalError::Schema(format!(
            "row {}: {SAMPLE_IDX} is null; every row must carry its sample's index",
            first + r
        )));
    }
    if let Some(r) = ids.values().iter().position(|&s| s < 0) {
        return Err(MarshalError::Data(format!(
            "row {}: {SAMPLE_IDX} {} is negative",
            first + r,
            ids.value(r)
        )));
    }
    Ok(ids)
}

/// Check that every row of the batch carries `want`.
fn check_fingerprints(
    batch: &RecordBatch,
    first: usize,
    ids: &Int64Array,
    want: &[u8; FINGERPRINT_LEN],
) -> Result<(), MarshalError> {
    let col = column(batch, FINGERPRINT)?;
    match col.data_type() {
        DataType::FixedSizeBinary(n) if *n as usize == FINGERPRINT_LEN => {
            each_fingerprint(col.as_fixed_size_binary(), first, ids, want)
        }
        DataType::FixedSizeBinary(n) => Err(MarshalError::Schema(format!(
            "column {FINGERPRINT:?} is FixedSizeBinary({n}); a fingerprint is \
             {FINGERPRINT_LEN} bytes"
        ))),
        DataType::Binary | DataType::LargeBinary => {
            // A byte array's values are read through its offsets unchecked, and
            // arrow-rs imports a C producer's arrays without validating them. So
            // the offsets are checked before any value is read: a bad one would
            // otherwise read outside the data, and print what it read into the
            // error below.
            col.to_data().validate_full().map_err(|e| {
                MarshalError::Data(format!("column {FINGERPRINT:?} has invalid offsets: {e}"))
            })?;
            match col.data_type() {
                DataType::Binary => each_fingerprint(col.as_binary::<i32>(), first, ids, want),
                _ => each_fingerprint(col.as_binary::<i64>(), first, ids, want),
            }
        }
        other => Err(MarshalError::Schema(format!(
            "column {FINGERPRINT:?} is {other:?}, expected \
             FixedSizeBinary({FINGERPRINT_LEN}), Binary or LargeBinary"
        ))),
    }
}

fn each_fingerprint<'a>(
    fps: impl ArrayAccessor<Item = &'a [u8]>,
    first: usize,
    ids: &Int64Array,
    want: &[u8; FINGERPRINT_LEN],
) -> Result<(), MarshalError> {
    if let Some(r) = first_null(&fps) {
        return Err(MarshalError::Schema(format!(
            "row {}: {FINGERPRINT} is null; every row must carry its set's fingerprint",
            first + r
        )));
    }
    for r in 0..fps.len() {
        let got = fps.value(r);
        if got.len() != FINGERPRINT_LEN {
            return Err(MarshalError::Data(format!(
                "row {}: {FINGERPRINT} holds {} bytes; a fingerprint is {FINGERPRINT_LEN}",
                first + r,
                got.len()
            )));
        }
        if got != want {
            return Err(MarshalError::Data(format!(
                "row {} ({SAMPLE_IDX} {}): {FINGERPRINT} {} is not this set's, {}. The row \
                 was sketched against another tree, with other parameters or by another \
                 build, so its sketch cannot be compared with this set's",
                first + r,
                ids.value(r),
                hex(got),
                hex(want)
            )));
        }
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Append the batch's sketches to `out`, from whichever list type carries them.
fn read_sketches<T: ArrowPrimitiveType>(
    batch: &RecordBatch,
    first: usize,
    ids: &Int64Array,
    k: usize,
    out: &mut Rows<T::Native>,
) -> Result<(), MarshalError> {
    let col = column(batch, SKETCH)?;
    match col.data_type() {
        DataType::FixedSizeList(item, n) => {
            check_elements::<T>(item)?;
            if usize::try_from(*n) != Ok(k) {
                return Err(MarshalError::Schema(format!(
                    "column {SKETCH:?} is a fixed_size_list of {n} values, but k is {k}"
                )));
            }
            let list = col.as_fixed_size_list();
            let spans = (0..list.len()).map(|r| (r * k, (r + 1) * k));
            take_rows::<T>(list, list.values(), spans, first, ids, k, out)
        }
        DataType::List(item) => {
            check_elements::<T>(item)?;
            let list = col.as_list::<i32>();
            take_rows::<T>(list, list.values(), offsets(list), first, ids, k, out)
        }
        DataType::LargeList(item) => {
            check_elements::<T>(item)?;
            let list = col.as_list::<i64>();
            take_rows::<T>(list, list.values(), offsets(list), first, ids, k, out)
        }
        other => Err(MarshalError::Schema(format!(
            "column {SKETCH:?} is {other:?}, expected a FixedSizeList, List or LargeList of {:?}",
            T::DATA_TYPE
        ))),
    }
}

/// Each row's span of a variable-length list's values.
fn offsets<O: OffsetSizeTrait>(
    list: &GenericListArray<O>,
) -> impl Iterator<Item = (usize, usize)> + '_ {
    list.value_offsets()
        .windows(2)
        .map(|w| (w[0].as_usize(), w[1].as_usize()))
}

fn check_elements<T: ArrowPrimitiveType>(item: &Field) -> Result<(), MarshalError> {
    if *item.data_type() != T::DATA_TYPE {
        return Err(MarshalError::Schema(format!(
            "column {SKETCH:?} holds {:?} values, expected {:?}: a sketch is read at \
             exactly the width it was made at",
            item.data_type(),
            T::DATA_TYPE
        )));
    }
    Ok(())
}

/// Copy out each row's span of `values`, refusing a null row, a null value or a
/// row that does not hold `k` values.
fn take_rows<T: ArrowPrimitiveType>(
    list: &dyn Array,
    values: &ArrayRef,
    spans: impl Iterator<Item = (usize, usize)>,
    first: usize,
    ids: &Int64Array,
    k: usize,
    out: &mut Rows<T::Native>,
) -> Result<(), MarshalError> {
    // Checked against the list's declared element type already, so this is
    // only reachable from an array that contradicts its own schema.
    let values = values.as_primitive_opt::<T>().ok_or_else(|| {
        MarshalError::Schema(format!(
            "column {SKETCH:?} holds {:?} values, expected {:?}",
            values.data_type(),
            T::DATA_TYPE
        ))
    })?;
    for (r, (start, end)) in spans.enumerate() {
        let row = first + r;
        if list.is_null(r) {
            return Err(MarshalError::Schema(format!(
                "row {row}: {SKETCH} is null; every row must carry a sketch"
            )));
        }
        // Saturating: offsets that arrive over FFI are not re-validated, and a
        // producer whose offsets decrease must get an error, not an underflow.
        let len = end.saturating_sub(start);
        if len != k {
            return Err(MarshalError::Data(format!(
                "row {row} ({SAMPLE_IDX} {}): {SKETCH} holds {len} values, but k is {k}",
                ids.value(r)
            )));
        }
        if values.null_count() > 0 && (start..end).any(|i| values.is_null(i)) {
            return Err(MarshalError::Schema(format!(
                "row {row}: {SKETCH} holds a null value; every value of a sketch is required"
            )));
        }
        out.push(values.values()[start..end].to_vec());
    }
    Ok(())
}

/// Put the rows in `sample_idx` order, refusing a repeated one, and naming the
/// first two rows that hold it. The index of a row here is its row number in
/// the stream.
fn ascending<T>(ids: Vec<i64>, mut rows: Rows<T>) -> Result<(Rows<T>, Vec<i64>), MarshalError> {
    let mut order: Vec<usize> = (0..ids.len()).collect();
    // The row breaks ties, so a repeat is reported at its first two rows.
    order.sort_unstable_by_key(|&i| (ids[i], i));
    if let Some(w) = order.windows(2).find(|w| ids[w[0]] == ids[w[1]]) {
        let (a, b) = (w[0], w[1]);
        return Err(MarshalError::Data(format!(
            "rows {a} and {b} are both {SAMPLE_IDX} {}; a set holds one sketch per sample",
            ids[a]
        )));
    }
    let sorted = order.iter().map(|&i| std::mem::take(&mut rows[i])).collect();
    Ok((sorted, order.iter().map(|&i| ids[i]).collect()))
}
