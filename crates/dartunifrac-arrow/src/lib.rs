//! Marshaling between a caller's data and [`dartunifrac_core`]'s types.
//!
//! The C API hands core two things core cannot build for itself: a tree, as
//! parent pointers and branch lengths, and a feature table already resolved to
//! tree node ids. This crate is that conversion, and the reverse for the
//! distances that come back out.
//!
//! It contains **no engine and no C ABI**. The sketching is core's; the opaque
//! handles, the rayon pool and the `catch_unwind` barrier belong to the crate
//! above this one. The seam is deliberate: everything here is exercised by
//! `cargo test` with no C toolchain, which is where the test loop wants to be.
//! That crate converts to and from `FFI_ArrowArrayStream`; this one speaks in
//! `RecordBatchReader`, on both sides.
//!
//! The name is a mild misnomer — [`tree_from_arrays`] takes plain slices,
//! because the C API receives the tree as `const int64_t *` / `const double *`
//! and wrapping two dense arrays in Arrow would buy nothing. Only the feature
//! table and the distances are streamed, and those are the parts that are
//! genuinely Arrow.

mod coo;
mod error;
mod table;
mod tree;

pub use coo::{coo_reader, output_schema, DISTANCE, I, J};
pub use error::MarshalError;
pub use table::{input_schema, table_from_stream, INDEX_TYPE, NODE_IDX, SAMPLE_IDX, VALUE};
pub use tree::{tree_from_arrays, NO_PARENT_IN};
