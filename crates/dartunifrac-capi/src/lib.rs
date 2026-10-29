//! `dartunifrac-capi`: the reentrant C ABI over [`dartunifrac_core`] and
//! [`dartunifrac_arrow`].
//!
//! This crate is **policy only**. It owns the opaque handles, the per-context
//! rayon pool, the `catch_unwind` barrier, the status codes and the last-error
//! storage. Every conversion between the caller's data and core's types is
//! `dartunifrac_arrow`'s; all of the sketching is core's.
//!
//! The contract every entry point shares is written once, in `cbindgen.toml`'s
//! header block, because that is the copy a C caller reads. In short: NULL is
//! checked and becomes `DF_ERR_NULL_POINTER`; handle out-params are NULL on
//! failure; errors are recorded on the context, or per thread when there is
//! none; and a panic becomes `DF_ERR_PANIC` instead of unwinding into C.
#![allow(non_camel_case_types)]
// The shared safety contract above covers every entry point; per-function
// `# Safety` sections would repeat it verbatim.
#![allow(clippy::missing_safety_doc)]

mod guard;

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::ptr;
use std::sync::Arc;

use dartunifrac_arrow::tree_from_arrays;
use dartunifrac_core::Tree;
use rayon::ThreadPool;

use guard::{ApiError, guard, guard_cstr, guard_void, slice_or_empty};

// ---------------------------------------------------------------------------
// Status codes and constants
// ---------------------------------------------------------------------------

/// Returned by every fallible `df_*` function: `DF_OK`, or a negative class.
/// Plain `int32_t` constants rather than a C enum, so new classes can be
/// appended without changing the type's size or invalidating a caller's switch.
pub type df_status_t = i32;

/// Success.
pub const DF_OK: df_status_t = 0;
/// A parameter, tree, table or sketch set was rejected, or a requested thread
/// pool could not be built; the message says which.
pub const DF_ERR_INVALID_INPUT: df_status_t = -1;
/// A required pointer was NULL; the message names it.
pub const DF_ERR_NULL_POINTER: df_status_t = -2;
/// The caller's Arrow stream failed, or had already been released.
pub const DF_ERR_ARROW: df_status_t = -3;
/// A panic was caught at the boundary: a bug in DartUniFrac, or an Arrow
/// producer that broke the C Data Interface contract badly enough to trip an
/// assertion while being read.
pub const DF_ERR_PANIC: df_status_t = -99;

/// The `batch_rows` a `df_config_t` of 0 selects.
pub const DF_DEFAULT_BATCH_ROWS: u64 = 65536;
/// The largest accepted `batch_rows`. An output batch is 20 bytes per row, so
/// this is about 1.3 GB; past it an allocation could fail, and an allocation
/// failure aborts the process rather than returning an error.
pub const DF_MAX_BATCH_ROWS: u64 = 1 << 26;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Context configuration. A NULL `df_config_t *` means every field is 0.
#[repr(C)]
pub struct df_config_t {
    /// 0: no dedicated pool; parallel work runs on rayon's global pool, which is
    /// also the only option on a wasm build without threads. 1 up to rayon's
    /// limit (65535 on 64-bit targets): a dedicated pool of exactly that many
    /// threads. Anything else is DF_ERR_INVALID_INPUT.
    pub n_threads: i32,
    /// Rows per output batch. 0 selects DF_DEFAULT_BATCH_ROWS; above
    /// DF_MAX_BATCH_ROWS is DF_ERR_INVALID_INPUT.
    pub batch_rows: u64,
}

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

/// A context: the thread pool, the output batch size and the last error. Use
/// it from one thread at a time.
//
// Single-threaded use is why `last_error` needs only a `RefCell`. That also
// makes `&df_ctx_t` `!Send`, so a closure handed to the pool must capture copies
// of what it needs, never the context.
pub struct df_ctx_t {
    // `Arc` so that an exported stream (M5b) can keep the pool alive after the
    // context is freed.
    #[allow(dead_code)] // read by df_sketch and df_distances (M5b)
    pool: Option<Arc<ThreadPool>>,
    #[allow(dead_code)] // read by df_distances (M5b)
    batch_rows: usize,
    last_error: RefCell<Option<CString>>,
}

/// A tree whose parents and branch lengths have been checked. A cycle is not
/// detected until a non-empty table is sketched against it. Immutable once
/// built, so it may be shared across threads and contexts.
pub struct df_tree_t {
    #[allow(dead_code)] // read by df_sketch (M5b)
    tree: Tree,
}

/// Run `f` on `pool` when there is one. Otherwise run it on the calling thread,
/// where any parallelism inside it uses rayon's global pool.
#[allow(dead_code)] // used by df_sketch and df_distances (M5b)
fn run_in_pool<R: Send>(pool: Option<&ThreadPool>, f: impl FnOnce() -> R + Send) -> R {
    match pool {
        Some(pool) => pool.install(f),
        None => f(),
    }
}

/// The pointer C sees for a last-error slot. It aliases the stored `CString`,
/// which lives until the next error replaces it or its owner is dropped.
fn message_ptr(slot: &RefCell<Option<CString>>) -> *const c_char {
    match &*slot.borrow() {
        Some(message) => message.as_ptr(),
        None => ptr::null(),
    }
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// The library's version. It equals the header's DF_VERSION when the header
/// and the library come from the same build, so comparing the two detects a
/// mismatch. Static; do not free.
#[unsafe(no_mangle)]
pub extern "C" fn df_version() -> *const c_char {
    // `concat!` supplies the terminating NUL, and the bytes are static.
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

/// Create a context. `cfg` may be NULL for all defaults. Errors are recorded
/// per thread, since there is no context yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_ctx_new(
    cfg: *const df_config_t,
    out: *mut *mut df_ctx_t,
) -> df_status_t {
    guard(ptr::null(), || {
        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        // SAFETY (every dereference here): the API contract.
        unsafe { out.write(ptr::null_mut()) };
        let (n_threads, batch_rows) = match unsafe { cfg.as_ref() } {
            Some(cfg) => (cfg.n_threads, cfg.batch_rows),
            None => (0, 0),
        };

        // Bounded before narrowing: on a 32-bit target `as usize` would wrap.
        let batch_rows = match batch_rows {
            0 => DF_DEFAULT_BATCH_ROWS,
            b if b > DF_MAX_BATCH_ROWS => {
                return Err(ApiError::invalid(format!(
                    "batch_rows is {b}; the maximum is DF_MAX_BATCH_ROWS ({DF_MAX_BATCH_ROWS})"
                )));
            }
            b => b,
        } as usize;

        let pool = match n_threads {
            0 => None,
            n => {
                // rayon would silently cap a larger count, so it is refused instead.
                let max = rayon::max_num_threads();
                let threads = usize::try_from(n)
                    .ok()
                    .filter(|&t| t <= max)
                    .ok_or_else(|| {
                        ApiError::invalid(format!(
                            "n_threads is {n}; it must be 0 (no dedicated pool) or 1 to {max}"
                        ))
                    })?;
                // Building a pool traps there, and a trap cannot be caught.
                if cfg!(all(target_family = "wasm", not(target_feature = "atomics"))) {
                    return Err(ApiError::invalid(format!(
                        "n_threads is {n}, but this wasm build has no threads; pass 0"
                    )));
                }
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .map_err(|e| ApiError::invalid(format!("a pool of {threads} threads: {e}")))?;
                Some(Arc::new(pool))
            }
        };

        let ctx = df_ctx_t {
            pool,
            batch_rows,
            last_error: RefCell::new(None),
        };
        unsafe { out.write(Box::into_raw(Box::new(ctx))) };
        Ok(())
    })
}

/// Free a context. NULL is a no-op.
///
/// A dedicated pool's threads are told to exit but are not joined: they finish
/// shortly after this returns. Do not unload the library while they may still
/// be running.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_ctx_free(ctx: *mut df_ctx_t) {
    guard_void(|| {
        if !ctx.is_null() {
            // SAFETY: a non-NULL ctx came from df_ctx_new, and is freed once.
            drop(unsafe { Box::from_raw(ctx) });
        }
    });
}

/// The last error recorded on `ctx`, or NULL if there has been none (or `ctx`
/// is NULL). Valid until the next error on `ctx` or `df_ctx_free(ctx)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_ctx_last_error(ctx: *const df_ctx_t) -> *const c_char {
    // SAFETY: the API contract.
    guard_cstr(|| match unsafe { ctx.as_ref() } {
        Some(ctx) => message_ptr(&ctx.last_error),
        None => ptr::null(),
    })
}

/// The last error of a call on this thread that had no context to record it
/// on: `df_ctx_new`, or any call passed a NULL `ctx`. NULL if there has been
/// none. Read it on the thread that made the failing call, before that thread
/// makes another. Valid until the next such error on this thread, or until the
/// thread exits.
#[unsafe(no_mangle)]
pub extern "C" fn df_thread_last_error() -> *const c_char {
    guard_cstr(|| guard::THREAD_LAST_ERROR.with(message_ptr))
}

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

/// Build a tree from `n_nodes` parent indices and branch lengths, both indexed
/// by node id. Both arrays are copied, so the caller may free them on return,
/// and the tree does not depend on `ctx`, which is used only to report an error.
///
/// A root has parent -1, and there may be more than one. Any other parent
/// outside 0..n_nodes is an error, as is a branch length that is not finite.
/// Only branches of positive length enter a sketch: a zero or negative length is
/// accepted and never sketched, and at least one branch must be positive.
///
/// Node numbering is part of every sketch made against this tree: two trees
/// that number the same topology differently produce sketches that must never
/// be compared.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_tree_new(
    ctx: *mut df_ctx_t,
    parent: *const i64,
    branch_length: *const f64,
    n_nodes: i64,
    out: *mut *mut df_tree_t,
) -> df_status_t {
    guard(ctx, || {
        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        // SAFETY (every dereference here): the API contract.
        unsafe { out.write(ptr::null_mut()) };
        if ctx.is_null() {
            return Err(ApiError::null("ctx"));
        }

        let n = usize::try_from(n_nodes).map_err(|_| {
            ApiError::invalid(format!(
                "n_nodes is {n_nodes}; it must be 0 or more and fit this platform's address space"
            ))
        })?;
        let parent = unsafe { slice_or_empty(parent, n, "parent") }?;
        let branch_length = unsafe { slice_or_empty(branch_length, n, "branch_length") }?;
        let tree = tree_from_arrays(parent, branch_length)?;
        // Core's edge filter is `len > 0.0`, so this is exactly the tree core would
        // find nothing to sketch on. Saying so now beats an error at sketch time.
        if !tree.lens.iter().any(|&l| l > 0.0) {
            return Err(ApiError::invalid(format!(
                "none of the tree's {n} nodes has a positive branch length, so every \
                 sample's sketch would be empty"
            )));
        }

        unsafe { out.write(Box::into_raw(Box::new(df_tree_t { tree }))) };
        Ok(())
    })
}

/// Free a tree. NULL is a no-op.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_tree_free(tree: *mut df_tree_t) {
    guard_void(|| {
        if !tree.is_null() {
            // SAFETY: a non-NULL tree came from df_tree_new, and is freed once.
            drop(unsafe { Box::from_raw(tree) });
        }
    });
}

#[cfg(test)]
mod tests;
