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

mod fingerprint;
mod guard;
mod stream;

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::ptr;
use std::sync::Arc;

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema, to_ffi};
use arrow_array::ffi_stream::{FFI_ArrowArrayStream, RecordBatchStreamReader};
use arrow_array::{Array, Int64Array};
use dartunifrac_arrow::{
    Sketches, distances_reader, sketches_from_stream, sketches_reader, table_from_stream,
    tree_from_arrays, trim_to_bbits,
};
use dartunifrac_core::{CoreError, Method, NO_PARENT, SketchParams, Tree, build_sketches};
use rayon::ThreadPool;

use fingerprint::{Key, tree_digest};
use guard::{ApiError, guard, guard_cstr, guard_void, slice_or_empty};
use stream::PoolReader;

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
/// The caller's Arrow stream failed or had already been released, or an Arrow
/// export failed.
pub const DF_ERR_ARROW: df_status_t = -3;
/// A panic was caught at the boundary: a bug in DartUniFrac, or an Arrow
/// producer that broke the C Data Interface contract badly enough to trip an
/// assertion while being read.
pub const DF_ERR_PANIC: df_status_t = -99;

/// DartMinHash, the method the CLI uses by default.
pub const DF_METHOD_DMH: u32 = 0;
/// TreeMinHash.
pub const DF_METHOD_TMH: u32 = 1;

/// The `batch_rows` a `df_config_t` of 0 selects.
pub const DF_DEFAULT_BATCH_ROWS: u64 = 65536;
/// The largest accepted `batch_rows`. An output batch is 20 bytes per row, so
/// this is about 1.3 GB; past it an allocation could fail, and an allocation
/// failure aborts the process rather than returning an error.
pub const DF_MAX_BATCH_ROWS: u64 = 1 << 26;

/// Bytes in a sketch set's fingerprint: a sha256.
pub const DF_FINGERPRINT_LEN: usize = 32;
// A literal above, for cbindgen; the column it is stored in must agree.
const _: () = assert!(DF_FINGERPRINT_LEN == dartunifrac_arrow::FINGERPRINT_LEN);

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

/// How to sketch. Every field is explicit: none has a zero-means-default, so a
/// struct the caller forgot to fill is rejected rather than quietly accepted.
/// The CLI's defaults, for reference, are DMH, k = 2048, seed = 1337 and
/// bbits = 16.
#[repr(C)]
pub struct df_params_t {
    /// DF_METHOD_DMH or DF_METHOD_TMH. ERS, which the CLI also offers, is not
    /// available: every sketch from this API is portable, and the CLI refuses
    /// ERS for portable sketches too.
    pub method: u32,
    /// Values per sketch; at least 1.
    pub k: u32,
    pub seed: u64,
    /// Bits of each value the distances compare: 16, 32 or 64.
    pub bbits: u32,
    /// Weighted normalized UniFrac rather than unweighted.
    pub weighted: bool,
    /// Weighted only: weight entries by raw counts rather than by each
    /// sample's relative abundance.
    pub raw_counts: bool,
    /// The caller's own identity for the data it sketches, such as a
    /// rarefaction's depth and seed. It changes no sketch value, but it is part
    /// of the fingerprint, so sets made from different data versions are never
    /// mixed. Opaque here; 0 is an ordinary value.
    pub data_version: u64,
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
    // `Arc` so that an exported stream can keep the pool alive after the
    // context is freed.
    pool: Option<Arc<ThreadPool>>,
    batch_rows: usize,
    last_error: RefCell<Option<CString>>,
}

/// A tree whose parents and branch lengths have been checked. A cycle is not
/// detected until a non-empty table is sketched against it. Immutable once
/// built, so it may be shared across threads and contexts.
pub struct df_tree_t {
    tree: Tree,
    // Whether each node has no children. A table entry must name a tip.
    is_tip: Vec<bool>,
    // The sha256 of the caller's arrays: the tree's term in every fingerprint.
    digest: [u8; 32],
}

/// One sketch per sample that had any branch to sketch, in the tree's portable
/// id space: made by df_sketch, or read back by df_sketches_import. Immutable
/// once built, so it may be shared across threads and contexts.
pub struct df_sketches_t {
    // `Arc` so that every stream drawn from this handle shares the one sketch
    // set, and outlives the handle.
    sketches: Arc<Sketches>,
    // The caller's sample_idx for each sketch, ascending.
    ids: Vec<i64>,
    weighted: bool,
    // Values per sketch, kept because an empty set has no sketch to measure.
    k: u32,
    // The compatibility key's sha256: which sets this one may be compared with.
    fingerprint: [u8; DF_FINGERPRINT_LEN],
}

/// Run `f` on `pool` when there is one. Otherwise run it on the calling thread,
/// where any parallelism inside it uses rayon's global pool.
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
                // rayon cannot start a thread there; on wasip1, building a pool
                // fails with "Not supported". Refusing first names the cause.
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

        // A tip is a node no other node names as its parent. Every parent was
        // range-checked by tree_from_arrays.
        let mut is_tip = vec![true; n];
        for &p in &tree.parent {
            if p != NO_PARENT {
                is_tip[p] = false;
            }
        }

        // From the caller's arrays, not `tree`: its `usize` parents are a different
        // width on wasm32, and the digest must not be.
        let digest = tree_digest(parent, branch_length);
        let tree = df_tree_t {
            tree,
            is_tip,
            digest,
        };
        unsafe { out.write(Box::into_raw(Box::new(tree))) };
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

// ---------------------------------------------------------------------------
// Sketches
// ---------------------------------------------------------------------------

/// Sketch every sample in a feature table streamed as `(sample_idx int64,
/// node_idx int64[, value float64])` rows; `value` is required when weighted.
///
/// `input` is consumed on every status (see the header's contract). Rows may
/// arrive in any order and in any number of batches. Each `node_idx` must be a
/// tip of `tree`, and each `sample_idx` must be in 0..n_samples. A row that
/// repeats a (sample_idx, node_idx) pair is used as well: when weighted, the
/// values add. `col_sums` holds n_samples per-sample totals over the caller's
/// whole table, and is required when weighted (raw_counts included) and ignored
/// otherwise.
///
/// Samples whose entries reach no branch of positive length get no sketch.
/// With fewer than two sketched samples there is nothing to compare: the result
/// is DF_OK and an empty sketch set, not an error.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_sketch(
    ctx: *mut df_ctx_t,
    tree: *const df_tree_t,
    params: *const df_params_t,
    input: *mut FFI_ArrowArrayStream,
    n_samples: i64,
    col_sums: *const f64,
    out: *mut *mut df_sketches_t,
) -> df_status_t {
    guard(ctx, || {
        // SAFETY (every dereference here): the API contract.
        if !out.is_null() {
            unsafe { out.write(ptr::null_mut()) };
        }
        // Take the input stream before anything can fail. It runs no callback,
        // and from here dropping `stream` releases it, on every path out.
        let stream = (!input.is_null()).then(|| unsafe { FFI_ArrowArrayStream::from_raw(input) });

        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        let ctx = unsafe { ctx.as_ref() }.ok_or_else(|| ApiError::null("ctx"))?;
        let tree = unsafe { tree.as_ref() }.ok_or_else(|| ApiError::null("tree"))?;
        let params = unsafe { params.as_ref() }.ok_or_else(|| ApiError::null("params"))?;
        let stream = stream.ok_or_else(|| ApiError::null("input"))?;
        let (sketch_params, bbits) = sketch_params(params)?;
        // The table allocates per sample before reading a row, and stages
        // sample indices as u32, so the bound is checked here first.
        let n = usize::try_from(n_samples)
            .ok()
            .filter(|&n| n <= u32::MAX as usize)
            .ok_or_else(|| {
                ApiError::invalid(format!(
                    "n_samples is {n_samples}; it must be between 0 and {}",
                    u32::MAX
                ))
            })?;
        let col_sums = match params.weighted {
            true => Some(unsafe { slice_or_empty(col_sums, n, "col_sums") }?),
            false => None,
        };
        // From the params just validated, before the sketching: the fingerprint
        // must describe what was sketched, whatever the caller does meanwhile.
        // In the pool, because the process's first fingerprint runs the
        // self-test, which sketches with rayon, and the caller's thread budget
        // covers that too.
        let (weighted, k) = (params.weighted, params.k);
        let pool = ctx.pool.clone();
        let fingerprint = run_in_pool(pool.as_deref(), || Key::new(tree.digest, params).digest());

        let (sketches, ids) = run_in_pool(pool.as_deref(), move || {
            sketch_in_pool(tree, stream, n, col_sums, &sketch_params, bbits)
        })?;
        let set = df_sketches_t {
            sketches: Arc::new(sketches),
            ids,
            weighted,
            k,
            fingerprint,
        };
        unsafe { out.write(Box::into_raw(Box::new(set))) };
        Ok(())
    })
}

/// Check every field of `p`, and translate it into core's parameters and the
/// sketch width. Every sketch from this API is portable.
fn sketch_params(p: &df_params_t) -> Result<(SketchParams, u8), ApiError> {
    let method = match p.method {
        DF_METHOD_DMH => Method::Dmh,
        DF_METHOD_TMH => Method::Tmh,
        // ERS's place in the CLI's list. Say why it is missing rather than
        // calling it unknown.
        2 => {
            return Err(ApiError::invalid(
                "method 2 (ERS) is not available: every sketch from this API is portable, \
                 and the CLI refuses ERS for portable sketches (weighted ERS takes its caps \
                 from the run's own samples, and unweighted ERS over every branch of the \
                 tree risks extra bias). Use DF_METHOD_DMH (0) or DF_METHOD_TMH (1)",
            ));
        }
        m => {
            return Err(ApiError::invalid(format!(
                "method is {m}; use DF_METHOD_DMH (0) or DF_METHOD_TMH (1)"
            )));
        }
    };
    if p.k == 0 {
        return Err(ApiError::invalid(
            "k is 0; a sketch needs at least one value",
        ));
    }
    // Matched as u32: narrowed to a byte first, 272 would pass as 16.
    let bbits = match p.bbits {
        16 | 32 | 64 => p.bbits as u8,
        b => return Err(ApiError::invalid(format!("bbits is {b}; use 16, 32 or 64"))),
    };
    if p.raw_counts && !p.weighted {
        return Err(ApiError::invalid(
            "raw_counts applies only to weighted UniFrac; set weighted, or clear raw_counts",
        ));
    }
    let params = SketchParams {
        k: p.k as usize,
        method,
        ers_l: 0, // read only by ERS, which is refused above
        seed: p.seed,
        weighted: p.weighted,
        raw_counts: p.raw_counts,
        portable: true,
    };
    Ok((params, bbits))
}

/// Everything in df_sketch that touches the caller's data, run in the
/// context's pool: reading the stream (so the producer's callbacks run there
/// too), building the table, and sketching. Returns the sketches and the
/// caller's sample_idx of each.
fn sketch_in_pool(
    tree: &df_tree_t,
    stream: FFI_ArrowArrayStream,
    n_samples: usize,
    col_sums: Option<&[f64]>,
    params: &SketchParams,
    bbits: u8,
) -> Result<(Sketches, Vec<i64>), ApiError> {
    let empty = || -> Result<(Sketches, Vec<i64>), ApiError> {
        Ok((trim_to_bbits(Vec::new(), bbits)?, Vec::new()))
    };

    let reader = RecordBatchStreamReader::try_new(stream)
        .map_err(|e| ApiError::new(DF_ERR_ARROW, format!("stream: {e}")))?;
    let table = table_from_stream(reader, n_samples, col_sums, params.weighted)?;
    // No entries at all: core would refuse a weighted table with no weights as
    // malformed, but this is simply fewer than two samples.
    if table.node.is_empty() {
        return empty();
    }
    // An id past the tree's end is not a tip either, but that is core's range
    // check to report, so only in-range internal nodes are caught here.
    if let Some(k) = table
        .node
        .iter()
        .position(|&v| tree.is_tip.get(v) == Some(&false))
    {
        let sample = table.colptr.partition_point(|&c| c <= k) - 1;
        return Err(ApiError::invalid(format!(
            "sample {sample}: node_idx {} is an internal node, not a tip. A table entry \
             must name a tip of the tree, the node a feature resolves to",
            table.node[k]
        )));
    }

    let set = match build_sketches(&tree.tree, &table, params, "") {
        Ok(set) => set,
        Err(CoreError::FewerThanTwoNonEmptySamples) => return empty(),
        Err(CoreError::NoActiveEdges) => {
            return Err(ApiError::invalid(
                "no branch of positive length is in scope, so nothing can be sketched",
            ));
        }
        Err(CoreError::MalformedTable(m)) => return Err(ApiError::invalid(m)),
    };
    // Within 0..n_samples, which is at most u32::MAX: exact in i64.
    let ids = set.kept.iter().map(|&s| s as i64).collect();
    Ok((trim_to_bbits(set.sketches, bbits)?, ids))
}

/// Free a sketch set. NULL is a no-op. Streams already drawn from it stay valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_sketches_free(sketches: *mut df_sketches_t) {
    guard_void(|| {
        if !sketches.is_null() {
            // SAFETY: a non-NULL handle came from df_sketch or df_sketches_import,
            // and is freed once.
            drop(unsafe { Box::from_raw(sketches) });
        }
    });
}

/// The caller's sample_idx of each sketch, ascending, as an int64 array: the
/// i and j that df_distances reports. Errors are recorded per thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_sketch_samples(
    sketches: *const df_sketches_t,
    out: *mut FFI_ArrowArray,
    out_schema: *mut FFI_ArrowSchema,
) -> df_status_t {
    guard(ptr::null(), || {
        // SAFETY (every dereference here): the API contract. `write`, never `*out
        // =`, because the caller's memory is uninitialised and must not be dropped.
        if !out.is_null() {
            unsafe { out.write(FFI_ArrowArray::empty()) };
        }
        if !out_schema.is_null() {
            unsafe { out_schema.write(FFI_ArrowSchema::empty()) };
        }
        let sketches = unsafe { sketches.as_ref() }.ok_or_else(|| ApiError::null("sketches"))?;
        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        if out_schema.is_null() {
            return Err(ApiError::null("out_schema"));
        }
        let ids = Int64Array::from(sketches.ids.clone()).into_data();
        let (array, schema) =
            to_ffi(&ids).map_err(|e| ApiError::new(DF_ERR_ARROW, e.to_string()))?;
        unsafe {
            out.write(array);
            out_schema.write(schema);
        }
        Ok(())
    })
}

/// Write the set's fingerprint, DF_FINGERPRINT_LEN bytes, to `out`: a sha256 over
/// everything that must match for two sketch sets to be comparable. That is the
/// tree (its exact parent and branch length arrays), every df_params_t field,
/// the DartMinHash release, this library's version, the target platform and its
/// libm. It depends on nothing about the samples, so sets sketched in separate
/// calls from the same tree and parameters share it. On failure `out` is left
/// as it was. Errors are recorded per thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_sketches_fingerprint(
    sketches: *const df_sketches_t,
    out: *mut u8,
) -> df_status_t {
    guard(ptr::null(), || {
        // SAFETY (every dereference here): the API contract.
        let sketches = unsafe { sketches.as_ref() }.ok_or_else(|| ApiError::null("sketches"))?;
        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        // `out` holds DF_FINGERPRINT_LEN bytes of the caller's; u8 needs no alignment.
        unsafe {
            ptr::copy_nonoverlapping(sketches.fingerprint.as_ptr(), out, DF_FINGERPRINT_LEN)
        };
        Ok(())
    })
}

/// Stream the set as rows a caller can store, one per sample: `(sample_idx
/// int64, fingerprint fixed_size_binary(DF_FINGERPRINT_LEN), sketch
/// fixed_size_list<uintN>[k])`, where N is the set's bbits, in ascending
/// sample_idx. Batches hold at most max(1, batch_rows / k) rows, so batch_rows
/// bounds the values in a batch. An empty set gives a stream with that schema
/// and no rows. df_sketches_import reads the rows back.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_sketches_export(
    ctx: *mut df_ctx_t,
    sketches: *const df_sketches_t,
    out: *mut FFI_ArrowArrayStream,
) -> df_status_t {
    guard(ctx, || {
        // SAFETY (every dereference here): the API contract.
        if !out.is_null() {
            unsafe { out.write(FFI_ArrowArrayStream::empty()) };
        }
        let ctx = unsafe { ctx.as_ref() }.ok_or_else(|| ApiError::null("ctx"))?;
        let sketches = unsafe { sketches.as_ref() }.ok_or_else(|| ApiError::null("sketches"))?;
        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        // A handle's k is at least 1: df_sketch and df_sketches_import validate it.
        let k = sketches.k as usize;
        let inner = sketches_reader(
            sketches.sketches.clone(),
            sketches.ids.clone(),
            k,
            sketches.fingerprint,
            (ctx.batch_rows / k).max(1),
        )?;
        let reader = PoolReader {
            inner,
            pool: ctx.pool.clone(),
            done: false,
        };
        unsafe { out.write(FFI_ArrowArrayStream::new(Box::new(reader))) };
        Ok(())
    })
}

/// Rebuild a sketch set from rows that df_sketches_export wrote, in any order
/// and any batching; columns are matched by name, and others are ignored. The
/// rows may have been through DuckDB or Parquet: a sketch column of list or
/// large_list, and a fingerprint column of binary or large_binary, are accepted
/// too, as long as the elements are exactly uintN for params->bbits and every
/// sketch holds params->k values.
///
/// Every row's fingerprint must be the one `tree` and `params` give, so the
/// rows were sketched against this tree with these parameters by this build;
/// otherwise the call fails naming the row and its sample_idx. A repeated, null
/// or negative sample_idx is an error too, and every rejection names its row,
/// counted from 0 across the stream. `input` is consumed on every status (see
/// the header's contract). Zero rows give DF_OK and an empty set, whatever the
/// stream's schema; one row gives a set of one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_sketches_import(
    ctx: *mut df_ctx_t,
    tree: *const df_tree_t,
    params: *const df_params_t,
    input: *mut FFI_ArrowArrayStream,
    out: *mut *mut df_sketches_t,
) -> df_status_t {
    guard(ctx, || {
        // SAFETY (every dereference here): the API contract.
        if !out.is_null() {
            unsafe { out.write(ptr::null_mut()) };
        }
        // Take the input stream before anything can fail, as df_sketch does: from
        // here dropping `stream` releases it, on every path out.
        let stream = (!input.is_null()).then(|| unsafe { FFI_ArrowArrayStream::from_raw(input) });

        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        let ctx = unsafe { ctx.as_ref() }.ok_or_else(|| ApiError::null("ctx"))?;
        let tree = unsafe { tree.as_ref() }.ok_or_else(|| ApiError::null("tree"))?;
        let params = unsafe { params.as_ref() }.ok_or_else(|| ApiError::null("params"))?;
        let stream = stream.ok_or_else(|| ApiError::null("input"))?;
        // Nothing is sketched, but the parameters define the key, so they are
        // held to what df_sketch accepts.
        let (_, bbits) = sketch_params(params)?;
        let (weighted, k) = (params.weighted, params.k);
        // In the pool, for the reasons df_sketch gives.
        let pool = ctx.pool.clone();
        let fingerprint = run_in_pool(pool.as_deref(), || Key::new(tree.digest, params).digest());

        let (sketches, ids) = run_in_pool(pool.as_deref(), move || {
            let reader = RecordBatchStreamReader::try_new(stream)
                .map_err(|e| ApiError::new(DF_ERR_ARROW, format!("stream: {e}")))?;
            Ok::<_, ApiError>(sketches_from_stream(reader, k as usize, bbits, &fingerprint)?)
        })?;
        let set = df_sketches_t {
            sketches: Arc::new(sketches),
            ids,
            weighted,
            k,
            fingerprint,
        };
        unsafe { out.write(Box::into_raw(Box::new(set))) };
        Ok(())
    })
}

/// Stream the condensed upper triangle of the distance matrix: one row per
/// pair, `(i int64, j int64, distance float32)` with i < j, where i and j are
/// the caller's sample_idx. Batches hold at most the context's batch_rows rows.
/// Nothing is computed until the stream's get_next is called.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn df_distances(
    ctx: *mut df_ctx_t,
    sketches: *const df_sketches_t,
    out: *mut FFI_ArrowArrayStream,
) -> df_status_t {
    guard(ctx, || {
        // SAFETY (every dereference here): the API contract.
        if !out.is_null() {
            unsafe { out.write(FFI_ArrowArrayStream::empty()) };
        }
        let ctx = unsafe { ctx.as_ref() }.ok_or_else(|| ApiError::null("ctx"))?;
        let sketches = unsafe { sketches.as_ref() }.ok_or_else(|| ApiError::null("sketches"))?;
        if out.is_null() {
            return Err(ApiError::null("out"));
        }
        // Validates only; nothing is computed until the caller's first get_next.
        let inner = distances_reader(
            sketches.sketches.clone(),
            sketches.ids.clone(),
            sketches.weighted,
            ctx.batch_rows,
        )?;
        let reader = PoolReader {
            inner,
            pool: ctx.pool.clone(),
            done: false,
        };
        unsafe { out.write(FFI_ArrowArrayStream::new(Box::new(reader))) };
        Ok(())
    })
}

#[cfg(test)]
mod tests;
