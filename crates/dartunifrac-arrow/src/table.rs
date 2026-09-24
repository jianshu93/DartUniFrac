use arrow_array::{Array, Float64Array, Int64Array, RecordBatch, RecordBatchReader};
use arrow_schema::{DataType, Field, Schema};
use dartunifrac_core::Table;
use rayon::prelude::*;

use crate::MarshalError;

/// Column this crate reads a sample index from.
pub const SAMPLE_IDX: &str = "sample_idx";
/// Column this crate reads a tree node index from.
pub const NODE_IDX: &str = "node_idx";
/// Column this crate reads a weight from; weighted runs only.
pub const VALUE: &str = "value";

/// The index type the input carries and the output therefore echoes.
///
/// Only `Int64` is accepted today. It is named once, here, so that widening the
/// input later cannot leave [`crate::coo`]'s output type behind.
pub const INDEX_TYPE: DataType = DataType::Int64;

/// The schema `table_from_stream` accepts, for callers that want to build one.
///
/// Columns are matched by *name*, not position, so a caller's projection may
/// order them however it likes and may carry extra columns; this is only the
/// minimum. `value` is present iff `weighted`.
pub fn input_schema(weighted: bool) -> Schema {
    let mut fields = vec![
        Field::new(SAMPLE_IDX, INDEX_TYPE, false),
        Field::new(NODE_IDX, INDEX_TYPE, false),
    ];
    if weighted {
        fields.push(Field::new(VALUE, DataType::Float64, false));
    }
    Schema::new(fields)
}

/// Drain a stream of `(sample_idx, node_idx, value)` rows into a [`Table`].
///
/// Rows may arrive in any order and split across batches however the producer
/// likes. Within each sample the entries are put into ascending `node_idx`
/// order before the table is handed over, which is what makes a sketch a pure
/// function of the *logical* table.
///
/// Core accumulates weights in `f32`, so a different order changes the low bits
/// of the weighted set. What that does to the sketch depends on the method, and
/// it was measured rather than assumed: under ERS a single-ulp change moves a
/// rejection-sampling decision, desynchronising the whole RNG stream and
/// changing **every** slot — 256/256, for tip counts from 3 to 128. DartMinHash
/// and TreeMinHash absorbed perturbations that small in every case tried. So
/// the sort is not "otherwise every sketch moves"; it is that one of the three
/// shipped methods is catastrophically order-sensitive, the other two are not
/// guaranteed to be immune, and a caller whose rows arrive from a parallel scan
/// has no control over the order in the first place.
///
/// `n_samples` and `col_sums` are inputs because neither can be recovered from
/// the stream. A sample with no entries never appears in it, and `col_sums` is
/// summed over the caller's *whole* table including features that resolve to no
/// tip — rows this crate never sees. `col_sums` is required iff `weighted`.
pub fn table_from_stream<R: RecordBatchReader>(
    reader: R,
    n_samples: usize,
    col_sums: Option<&[f64]>,
    weighted: bool,
) -> Result<Table, MarshalError> {
    // Core wants `col_sums.len() == n_samples` whether or not it reads them, so
    // presence runs get zeros rather than an awkward Option inside `Table`.
    let col_sums = match (weighted, col_sums) {
        (true, None) => {
            return Err(MarshalError::Data(
                "weighted runs need col_sums: they are summed over the caller's whole \
                 table, including features that resolve to no tip, and those rows never \
                 reach this crate"
                    .into(),
            ))
        }
        (true, Some(cs)) if cs.len() != n_samples => {
            return Err(MarshalError::Data(format!(
                "col_sums has {} entries, expected n_samples = {n_samples}",
                cs.len()
            )))
        }
        (true, Some(cs)) => {
            // The numerator of `value / denom` is checked below; leaving the
            // denominator unchecked would guard nothing. NaN slips past core's
            // `denom == 0.0` skip and poisons the sample's accumulator, and an
            // infinity divides every entry to zero, which core's `inc == 0.0`
            // fast path then discards -- the sample silently loses all its mass.
            if let Some((s, v)) = cs.iter().enumerate().find(|(_, v)| !v.is_finite()) {
                return Err(MarshalError::Data(format!(
                    "col_sums[{s}] is {v}, which is not finite"
                )));
            }
            cs.to_vec()
        }
        (false, _) => vec![0.0; n_samples],
    };

    // Pass one: drain the stream, keeping rows grouped by nothing in particular
    // and counting per sample so the CSC offsets can be built without a second
    // pass over a stream that cannot be rewound.
    let mut row_sample: Vec<u32> = Vec::new();
    let mut row_node: Vec<usize> = Vec::new();
    let mut row_value: Vec<f64> = Vec::new();
    // Staged sample ids are u32 to keep the per-row cost down; this is the one
    // place that assumption has to be stated rather than assumed.
    if n_samples > u32::MAX as usize {
        return Err(MarshalError::Data(format!(
            "n_samples = {n_samples} exceeds {}, the widest sample index this crate stages",
            u32::MAX
        )));
    }
    let mut counts = vec![0usize; n_samples];
    let mut first_row_of_batch = 0usize;

    for batch in reader {
        let batch = batch.map_err(|e| MarshalError::Stream(e.to_string()))?;
        if batch.num_rows() == 0 {
            continue;
        }
        let samples = index_column(&batch, SAMPLE_IDX)?;
        let nodes = index_column(&batch, NODE_IDX)?;
        let values = if weighted { Some(weight_column(&batch)?) } else { None };

        for r in 0..batch.num_rows() {
            let row = first_row_of_batch + r;

            // Compare in i64 and narrow afterwards, never the other way round.
            // `s as usize` on a 32-bit target -- wasm32-unknown-emscripten is a
            // declared target of this crate -- discards the high word before the
            // comparison, so 2^32 would read as 0 and pass. That would silently
            // file a row under the wrong sample, which is precisely the class of
            // failure this layer exists to stop.
            let s = samples.value(r);
            if s < 0 || s >= n_samples as i64 {
                return Err(MarshalError::Data(format!(
                    "row {row}: {SAMPLE_IDX} {s} is outside 0..{n_samples}"
                )));
            }
            let s = s as usize; // in range by the check above, so exact everywhere

            let v = nodes.value(r);
            if v < 0 {
                return Err(MarshalError::Data(format!(
                    "row {row}: {NODE_IDX} {v} is negative"
                )));
            }
            // Core owns the upper bound, but it can only check the value it is
            // given: a truncating cast here would hand it a different number.
            let v = usize::try_from(v).map_err(|_| {
                MarshalError::Data(format!(
                    "row {row}: {NODE_IDX} {v} does not fit a {}-bit usize",
                    usize::BITS
                ))
            })?;
            if let Some(vals) = values {
                let w = vals.value(r);
                if !w.is_finite() {
                    return Err(MarshalError::Data(format!(
                        "row {row}: {VALUE} {w} is not finite"
                    )));
                }
                row_value.push(w);
            }

            counts[s] += 1;
            row_sample.push(s as u32);
            row_node.push(v);
        }
        first_row_of_batch += batch.num_rows();
    }

    // Offsets, then scatter each row into its sample's span.
    let mut colptr = Vec::with_capacity(n_samples + 1);
    let mut running = 0usize;
    colptr.push(0);
    for &c in &counts {
        running += c;
        colptr.push(running);
    }

    let nnz = row_node.len();
    let mut node = vec![0usize; nnz];
    let mut value = if weighted { vec![0.0f64; nnz] } else { Vec::new() };
    let mut cursor: Vec<usize> = colptr[..n_samples].to_vec();
    for (r, &s) in row_sample.iter().enumerate() {
        let dst = &mut cursor[s as usize];
        node[*dst] = row_node[r];
        if weighted {
            value[*dst] = row_value[r];
        }
        *dst += 1;
    }

    // The staging arrays are dead from here, but locals live to end of scope --
    // roughly 20 bytes per entry sitting alongside the arrays that still matter.
    drop(row_sample);
    drop(row_node);
    drop(row_value);
    drop(cursor);

    order_entries_by_node(&colptr, &mut node, &mut value);

    Ok(Table { n_samples, colptr, node, value, col_sums })
}

/// Put every sample's entries into ascending node order, in place.
///
/// Ties are broken by the weight, not by arrival position, so a caller that
/// sends two rows for the same `(sample, node)` still gets one canonical order.
/// Breaking ties by arrival would leave exactly the non-determinism this sort
/// exists to remove.
fn order_entries_by_node(colptr: &[usize], node: &mut [usize], value: &mut [f64]) {
    let lens: Vec<usize> = colptr.windows(2).map(|w| w[1] - w[0]).collect();

    if value.is_empty() {
        // Presence-only: node ids are the whole entry, so equal ids are
        // interchangeable and a plain sort is already canonical.
        let mut rest: &mut [usize] = node;
        let mut spans: Vec<&mut [usize]> = Vec::with_capacity(lens.len());
        for &len in &lens {
            let (head, tail) = rest.split_at_mut(len);
            spans.push(head);
            rest = tail;
        }
        spans.par_iter_mut().for_each(|s| s.sort_unstable());
        return;
    }

    let mut rest_n: &mut [usize] = node;
    let mut rest_v: &mut [f64] = value;
    let mut spans: Vec<(&mut [usize], &mut [f64])> = Vec::with_capacity(lens.len());
    for &len in &lens {
        let (hn, tn) = rest_n.split_at_mut(len);
        let (hv, tv) = rest_v.split_at_mut(len);
        rest_n = tn;
        rest_v = tv;
        spans.push((hn, hv));
    }

    // Scratch is one sample wide, not one table wide, and rayon hands each
    // worker its own — the same shape core uses for its accumulator.
    spans.par_iter_mut().for_each_init(
        || (Vec::<usize>::new(), Vec::<usize>::new(), Vec::<f64>::new()),
        |(perm, sorted_n, sorted_v), (ns, vs)| {
            if ns.len() < 2 {
                return;
            }
            perm.clear();
            perm.extend(0..ns.len());
            perm.sort_unstable_by(|&a, &b| {
                ns[a].cmp(&ns[b]).then_with(|| vs[a].total_cmp(&vs[b]))
            });
            sorted_n.clear();
            sorted_n.extend(perm.iter().map(|&i| ns[i]));
            sorted_v.clear();
            sorted_v.extend(perm.iter().map(|&i| vs[i]));
            ns.copy_from_slice(sorted_n);
            vs.copy_from_slice(sorted_v);
        },
    );
}

/// Fetch a required `Int64` column by name, rejecting nulls.
fn index_column<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int64Array, MarshalError> {
    let col = batch.column_by_name(name).ok_or_else(|| {
        MarshalError::Schema(format!(
            "required column {name:?} is missing. Columns are matched by name: \
             {SAMPLE_IDX:?} and {NODE_IDX:?} are always required, {VALUE:?} when weighted"
        ))
    })?;
    if col.null_count() > 0 {
        return Err(MarshalError::Schema(format!(
            "column {name:?} contains nulls; every row must carry an index"
        )));
    }
    col.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
        MarshalError::Schema(format!(
            "column {name:?} is {:?}, expected {INDEX_TYPE:?}",
            col.data_type()
        ))
    })
}

/// Fetch the required `Float64` weight column, rejecting nulls.
fn weight_column(batch: &RecordBatch) -> Result<&Float64Array, MarshalError> {
    let col = batch.column_by_name(VALUE).ok_or_else(|| {
        MarshalError::Schema(format!(
            "required column {VALUE:?} is missing; a weighted run needs one weight per row"
        ))
    })?;
    if col.null_count() > 0 {
        return Err(MarshalError::Schema(format!(
            "column {VALUE:?} contains nulls; a weighted run needs one weight per row"
        )));
    }
    col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        MarshalError::Schema(format!(
            "column {VALUE:?} is {:?}, expected {:?}",
            col.data_type(),
            DataType::Float64
        ))
    })
}
