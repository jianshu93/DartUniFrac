use std::sync::Arc;

use arrow_array::{Float32Array, Int64Array, RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};

use crate::table::INDEX_TYPE;

/// Row index column of the COO output.
pub const I: &str = "i";
/// Column index column of the COO output.
pub const J: &str = "j";
/// The distance itself.
pub const DISTANCE: &str = "distance";

/// The schema [`coo_reader`] produces.
///
/// `i` and `j` carry [`INDEX_TYPE`] — the same type the input's `sample_idx`
/// column has — so the output joins straight back against the caller's sample
/// table with no cast. `distance` is `Float32` because that is what core
/// computes and what the binary writes; widening it would invent precision.
pub fn output_schema() -> Schema {
    Schema::new(vec![
        Field::new(I, INDEX_TYPE, false),
        Field::new(J, INDEX_TYPE, false),
        Field::new(DISTANCE, DataType::Float32, false),
    ])
}

/// Assemble one COO batch. The only place either reader builds an array, so the
/// column order and the array types cannot drift away from [`output_schema`].
pub(crate) fn coo_batch(
    schema: &SchemaRef,
    i: Vec<i64>,
    j: Vec<i64>,
    d: Vec<f32>,
) -> Result<RecordBatch, ArrowError> {
    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(i)),
            Arc::new(Int64Array::from(j)),
            Arc::new(Float32Array::from(d)),
        ],
    )
}

/// Batch `(i, j, distance)` triples into record batches of `batch_rows` rows.
///
/// `pairs` are indices into `ids`, which is the sketch set's `kept` vector: the
/// caller's own sample indices, in order. The emitted `i`/`j` are `ids[i]` and
/// `ids[j]`, not the compacted positions, so a caller never has to join through
/// a second result to learn which samples a distance belongs to.
pub fn coo_reader(
    pairs: impl Iterator<Item = (u32, u32, f32)> + Send + 'static,
    ids: Vec<i64>,
    batch_rows: usize,
) -> Box<dyn RecordBatchReader + Send> {
    Box::new(CooReader {
        pairs,
        ids,
        // A zero would busy-loop producing empty batches forever.
        batch_rows: batch_rows.max(1),
        schema: Arc::new(output_schema()),
        done: false,
    })
}

struct CooReader<I> {
    pairs: I,
    ids: Vec<i64>,
    batch_rows: usize,
    schema: SchemaRef,
    done: bool,
}

impl<I: Iterator<Item = (u32, u32, f32)>> CooReader<I> {
    fn id(&self, which: &str, idx: u32) -> Result<i64, ArrowError> {
        self.ids.get(idx as usize).copied().ok_or_else(|| {
            // Only reachable from a bug in the layer that produced the pairs.
            // This crate sits beneath an FFI boundary, where a recoverable
            // error beats a panic even for an internal invariant.
            ArrowError::InvalidArgumentError(format!(
                "{which} index {idx} has no entry in the id table, which holds {} samples",
                self.ids.len()
            ))
        })
    }

    fn next_batch(&mut self) -> Result<Option<RecordBatch>, ArrowError> {
        let mut i = Vec::with_capacity(self.batch_rows);
        let mut j = Vec::with_capacity(self.batch_rows);
        let mut d = Vec::with_capacity(self.batch_rows);

        while i.len() < self.batch_rows {
            let Some((a, b, dist)) = self.pairs.next() else {
                self.done = true;
                break;
            };
            i.push(self.id(I, a)?);
            j.push(self.id(J, b)?);
            d.push(dist);
        }

        if i.is_empty() {
            return Ok(None);
        }
        coo_batch(&self.schema, i, j, d).map(Some)
    }
}

impl<I: Iterator<Item = (u32, u32, f32)>> Iterator for CooReader<I> {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        match self.next_batch() {
            Ok(Some(b)) => Some(Ok(b)),
            Ok(None) => None,
            Err(e) => {
                // A failed batch ends the stream: continuing would hand the
                // caller a hole in the middle of a distance matrix.
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

impl<I: Iterator<Item = (u32, u32, f32)>> RecordBatchReader for CooReader<I> {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}
