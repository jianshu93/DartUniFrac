//! Shared fixtures. Deliberately the *same* fixture `dartunifrac-core`'s
//! `golden.rs` pins exact sketch values against — `data/test.nwk` with
//! `data/test_OTU_table.txt` — so a table that survives this crate intact can be
//! checked against numbers captured from the pre-extraction binary, not just
//! against itself.
#![allow(dead_code)]

use std::sync::Arc;

use arrow_array::{Float64Array, Int64Array, RecordBatch, RecordBatchIterator};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use dartunifrac_core::{Method, SketchParams, Table, Tree, NO_PARENT};

/// `data/test.nwk` as `build_parent_and_lens_simple` numbers it. Node 0 is the
/// `newick` crate's unused slot: no parent, no length, no edge.
pub fn test_tree() -> Tree {
    Tree {
        parent: vec![NO_PARENT, NO_PARENT, 1, 2, 2, 4, 4, 1, 7, 7, 9, 9, 9],
        lens: vec![
            0.0,
            0.0,
            0.30000001192092896,
            0.10000000149011612,
            0.019999999552965164,
            0.05000000074505806,
            0.05000000074505806,
            0.4000000059604645,
            0.20000000298023224,
            0.05000000074505806,
            0.10000000149011612,
            0.15000000596046448,
            0.20000000298023224,
        ],
    }
}

/// `data/test_OTU_table.txt` resolved to node ids, as the dense TSV reader emits
/// it. Every sample's node ids happen to be ascending already, which is what
/// lets a shuffled stream be compared against this byte for byte.
pub fn test_table() -> Table {
    Table {
        n_samples: 3,
        colptr: vec![0, 3, 6, 9],
        node: vec![3, 6, 8, 5, 10, 11, 3, 6, 12],
        value: vec![2.0, 5.0, 3.0, 3.0, 9.0, 6.0, 4.0, 7.0, 3.0],
        col_sums: vec![10.0, 18.0, 14.0],
    }
}

/// [`test_table`] flattened back into the `(sample_idx, node_idx, value)` rows a
/// caller would stream. Derived from the table rather than written out again, so
/// the two cannot drift apart.
pub fn test_rows() -> Vec<(i64, i64, f64)> {
    let t = test_table();
    let mut rows = Vec::new();
    for s in 0..t.n_samples {
        for k in t.colptr[s]..t.colptr[s + 1] {
            rows.push((s as i64, t.node[k] as i64, t.value[k]));
        }
    }
    rows
}

pub fn params(weighted: bool) -> SketchParams {
    SketchParams {
        k: 16,
        method: Method::Dmh,
        ers_l: 2048,
        seed: 1337,
        weighted,
        raw_counts: false,
        portable: false,
    }
}

/// Pack rows into one batch under [`dartunifrac_arrow::input_schema`].
pub fn batch(rows: &[(i64, i64, f64)], weighted: bool) -> RecordBatch {
    let schema = Arc::new(dartunifrac_arrow::input_schema(weighted));
    batch_with_schema(schema, rows, weighted)
}

pub fn batch_with_schema(schema: SchemaRef, rows: &[(i64, i64, f64)], weighted: bool) -> RecordBatch {
    let samples = Int64Array::from(rows.iter().map(|r| r.0).collect::<Vec<_>>());
    let nodes = Int64Array::from(rows.iter().map(|r| r.1).collect::<Vec<_>>());
    let mut cols: Vec<arrow_array::ArrayRef> = vec![Arc::new(samples), Arc::new(nodes)];
    if weighted {
        cols.push(Arc::new(Float64Array::from(
            rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )));
    }
    RecordBatch::try_new(schema, cols).expect("fixture batch is well formed")
}

/// A reader over `batches`, carrying `schema` even when there are no batches.
pub fn reader(
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
) -> RecordBatchIterator<std::vec::IntoIter<Result<RecordBatch, ArrowError>>> {
    RecordBatchIterator::new(batches.into_iter().map(Ok).collect::<Vec<_>>().into_iter(), schema)
}

/// Rows split into batches of at most `n` rows each.
pub fn batched(rows: &[(i64, i64, f64)], n: usize, weighted: bool) -> Vec<RecordBatch> {
    rows.chunks(n).map(|c| batch(c, weighted)).collect()
}

/// A deterministic shuffle. `Table` order is what is under test, so the
/// permutation must be fixed rather than random — a flaky ordering would make a
/// failure impossible to reproduce.
///
/// Reversing before the stride walk is not decoration. A coprime stride alone
/// mixes the samples together but can leave a given sample's own entries in
/// their original relative order, and on this fixture it did: of three samples
/// it scrambled exactly one. Reversing first makes every sample's sub-order
/// descending, so the stride cannot restore any of them. Callers should still
/// assert the property with [`assert_no_sample_arrives_sorted`] rather than
/// trust the arithmetic.
pub fn shuffled<T: Clone>(rows: &[T]) -> Vec<T> {
    let n = rows.len();
    let stride = 5;
    assert!(n > 1 && n % stride != 0, "stride must not divide the length");
    let reversed: Vec<T> = rows.iter().rev().cloned().collect();
    (0..n).map(|i| reversed[(i * stride) % n].clone()).collect()
}

/// Assert that no sample's entries arrive in ascending node order.
///
/// The point of a shuffled fixture is to exercise the per-sample sort on *every*
/// span, not whichever one the permutation happened to disturb. A fixture that
/// left some samples already in order would still let the order-independence
/// tests pass while saying nothing about those spans — so the property is
/// checked, not assumed.
pub fn assert_no_sample_arrives_sorted(rows: &[(i64, i64, f64)], n_samples: usize) {
    for s in 0..n_samples as i64 {
        let nodes: Vec<i64> = rows.iter().filter(|r| r.0 == s).map(|r| r.1).collect();
        assert!(nodes.len() > 1, "sample {s} has too few entries to be out of order");
        assert!(
            nodes.windows(2).any(|w| w[0] > w[1]),
            "sample {s} arrives ascending ({nodes:?}); the sort is not exercised on its span"
        );
    }
}

pub fn schema_of(weighted: bool) -> SchemaRef {
    Arc::new(dartunifrac_arrow::input_schema(weighted))
}

pub fn empty_schema() -> SchemaRef {
    Arc::new(Schema::empty())
}

/// `Result::unwrap_err` and `expect_err` both require `T: Debug`, and core's
/// `Tree` and `Table` deliberately have no `Debug` impl. This is the same thing
/// without the bound, so error assertions read normally.
pub trait UnwrapErrNoDebug<E> {
    fn marshal_err(self) -> E;
}

impl<T, E> UnwrapErrNoDebug<E> for Result<T, E> {
    fn marshal_err(self) -> E {
        match self {
            Ok(_) => panic!("expected an error, got a value"),
            Err(e) => e,
        }
    }
}
