//! The adapter between a distance reader and the stream handed to C.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::{ArrowError, SchemaRef};
use rayon::ThreadPool;

use crate::guard::{panic_message, sanitise};
use crate::run_in_pool;

/// What `df_distances` exports: the distance reader, run in its context's pool,
/// behind a panic barrier.
///
/// The barrier is needed because arrow-rs's exported `get_next` is an
/// `extern "C"` fn with none of its own, so a panic in the reader would abort the
/// host. For the same reason no error may carry a NUL: arrow-rs turns the
/// message into a `CString` with `expect`, and that panic aborts as well.
///
/// The pool is held by `Arc`, so the stream keeps working after the context
/// that made it is freed.
pub(crate) struct PoolReader {
    pub(crate) inner: Box<dyn RecordBatchReader + Send>,
    pub(crate) pool: Option<Arc<ThreadPool>>,
    /// Set by a panic: the reader's state is unknown after one, so the stream
    /// ends rather than read on.
    pub(crate) done: bool,
}

impl Iterator for PoolReader {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let (inner, pool) = (&mut self.inner, self.pool.as_deref());
        match catch_unwind(AssertUnwindSafe(|| run_in_pool(pool, || inner.next()))) {
            Ok(Some(Err(e))) => {
                let message = e.to_string();
                Some(Err(if message.contains('\0') {
                    ArrowError::ComputeError(sanitise(&message))
                } else {
                    e
                }))
            }
            Ok(item) => item,
            Err(payload) => {
                self.done = true;
                let message = format!("panic: {}", panic_message(payload));
                Some(Err(ArrowError::ComputeError(sanitise(&message))))
            }
        }
    }
}

impl RecordBatchReader for PoolReader {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}
