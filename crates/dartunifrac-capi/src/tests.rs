//! The C ABI, exercised from Rust through the exported functions themselves.
//! In-crate so a test can look inside a handle (which pool a context holds,
//! what tree a handle carries) without the ABI having to expose it.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::mem::{MaybeUninit, align_of, offset_of, size_of};
use std::ptr;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{Arc, Mutex};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi};
use arrow_array::ffi_stream::{FFI_ArrowArrayStream, RecordBatchStreamReader};
use arrow_array::{Int64Array, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{ArrowError, DataType, Field, Schema, SchemaRef};
use dartunifrac_arrow::{Sketches, distances_reader, table_from_stream, trim_to_bbits};
use dartunifrac_core::{Method, NO_PARENT, SketchParams, build_sketches};

use crate::guard::{ApiError, THREAD_LAST_ERROR, guard, guard_cstr, guard_void, record_error};
use crate::stream::PoolReader;
use crate::*;

// The fixtures dartunifrac-arrow's tests use, so both crates test against one
// tree instead of two copies that could drift. The path leaves this package, so
// these tests build only inside the workspace, not from a packaged tarball.
#[path = "../../dartunifrac-arrow/tests/common/mod.rs"]
mod common;

/// The fixture tree as a C caller passes it, with -1 for a root.
fn fixture_arrays() -> (Vec<i64>, Vec<f64>) {
    let t = common::test_tree();
    let parent = t
        .parent
        .iter()
        .map(|&p| if p == NO_PARENT { -1 } else { p as i64 })
        .collect();
    (parent, t.lens)
}

fn text(p: *const c_char) -> Option<String> {
    (!p.is_null()).then(|| unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned())
}

fn ctx_error(ctx: *const df_ctx_t) -> Option<String> {
    text(unsafe { df_ctx_last_error(ctx) })
}

fn thread_error() -> Option<String> {
    text(df_thread_last_error())
}

/// Forget any error an earlier test left on this thread, so an assertion about
/// the thread's slot reads what this test put there.
fn clear_thread_error() {
    THREAD_LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// A non-NULL pointer no test dereferences. Written into an out-param before a
/// call, it tells "reset to NULL" apart from "never touched".
fn sentinel<T>() -> *mut T {
    ptr::NonNull::dangling().as_ptr()
}

/// A context with default settings, for tests about something else.
fn new_ctx() -> *mut df_ctx_t {
    let mut ctx = ptr::null_mut();
    assert_eq!(unsafe { df_ctx_new(ptr::null(), &mut ctx) }, DF_OK);
    ctx
}

fn ctx_with(n_threads: i32, batch_rows: u64) -> (df_status_t, *mut df_ctx_t) {
    let cfg = df_config_t {
        n_threads,
        batch_rows,
    };
    let mut ctx = sentinel();
    let status = unsafe { df_ctx_new(&cfg, &mut ctx) };
    (status, ctx)
}

fn tree_new(ctx: *mut df_ctx_t, parent: &[i64], lens: &[f64]) -> (df_status_t, *mut df_tree_t) {
    assert_eq!(parent.len(), lens.len());
    let mut tree = sentinel();
    let n = parent.len() as i64;
    let status = unsafe { df_tree_new(ctx, parent.as_ptr(), lens.as_ptr(), n, &mut tree) };
    (status, tree)
}

// ---------------------------------------------------------------------------
// Version and configuration
// ---------------------------------------------------------------------------

/// The header's DF_VERSION and the library's df_version() come from the same
/// Cargo version, so a caller can detect a header built against one library and
/// linked against another. The C harness checks the header's side.
#[test]
fn version_matches_the_crate_version() {
    assert_eq!(
        text(df_version()).as_deref(),
        Some(env!("CARGO_PKG_VERSION"))
    );
}

/// A NULL config and an all-zero one both mean no dedicated pool: the wasm
/// mode, and the one where the host's own rayon setup governs. Work then runs
/// on the calling thread, not on some pool's worker. On wasm the two cannot be
/// told apart: once anything has used rayon's global pool there, rayon makes
/// the calling thread that pool's only worker, so it has an index too.
#[test]
fn null_config_means_no_dedicated_pool_and_default_batch_rows() {
    let null = new_ctx();
    let (status, zero) = ctx_with(0, 0);
    assert_eq!(status, DF_OK);
    for ctx in [null, zero] {
        let c = unsafe { &*ctx };
        assert!(c.pool.is_none());
        assert_eq!(c.batch_rows, DF_DEFAULT_BATCH_ROWS as usize);
        if cfg!(not(target_family = "wasm")) {
            assert_eq!(
                run_in_pool(c.pool.as_deref(), rayon::current_thread_index),
                None
            );
        }
        unsafe { df_ctx_free(ctx) };
    }
}

/// The thread count is a promise to the host, which budgets its own threads
/// around it. So the pool must have exactly that many, and work handed to
/// run_in_pool must run on it. One more than the global pool's size makes the
/// two pools impossible to confuse; 1 is the smallest count accepted.
#[test]
#[cfg_attr(target_family = "wasm", ignore = "rayon cannot spawn threads on wasm")]
fn dedicated_pool_has_exactly_the_requested_threads() {
    for n in [1, rayon::current_num_threads() + 1] {
        let (status, ctx) = ctx_with(n as i32, 0);
        assert_eq!(status, DF_OK, "n_threads = {n}: {:?}", thread_error());
        let pool = unsafe { &*ctx }.pool.clone();
        let (threads, index) = run_in_pool(pool.as_deref(), || {
            (rayon::current_num_threads(), rayon::current_thread_index())
        });
        assert_eq!(threads, n);
        assert!(
            index.is_some(),
            "ran on the calling thread, not on a pool worker"
        );
        unsafe { df_ctx_free(ctx) };
    }
}

/// A negative count means nothing, and rayon silently caps one above its limit,
/// which would quietly break the promise above. Both are refused, and named.
#[test]
fn negative_or_excessive_n_threads_is_rejected_and_named() {
    let too_many = i32::try_from(rayon::max_num_threads() + 1).unwrap();
    for n in [-1, i32::MIN, too_many] {
        clear_thread_error();
        let (status, ctx) = ctx_with(n, 0);
        assert_eq!(status, DF_ERR_INVALID_INPUT, "n_threads = {n}");
        assert!(ctx.is_null());
        let msg = thread_error().unwrap();
        assert!(
            msg.contains("n_threads") && msg.contains(&n.to_string()),
            "{msg}"
        );
    }
}

/// A wasm build without threads cannot have a dedicated pool. rayon's own
/// error there says only "Not supported", so the request is refused first,
/// with a message that names the cause and the fix.
#[test]
#[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
fn a_dedicated_pool_on_wasm_without_threads_is_rejected_and_named() {
    clear_thread_error();
    let (status, ctx) = ctx_with(1, 0);
    assert_eq!(status, DF_ERR_INVALID_INPUT);
    assert!(ctx.is_null());
    assert!(thread_error().unwrap().contains("wasm"));
}

/// batch_rows sizes an allocation, and a failed allocation aborts instead of
/// returning, so the bound is checked up front. 2^32 + 1 proves the check comes
/// before any narrowing: on a 32-bit target it would otherwise become 1.
#[test]
fn batch_rows_above_the_maximum_is_rejected_and_the_maximum_is_accepted() {
    for b in [DF_MAX_BATCH_ROWS + 1, (1 << 32) + 1, u64::MAX] {
        clear_thread_error();
        let (status, ctx) = ctx_with(0, b);
        assert_eq!(status, DF_ERR_INVALID_INPUT, "batch_rows = {b}");
        assert!(ctx.is_null());
        let msg = thread_error().unwrap();
        assert!(
            msg.contains("batch_rows") && msg.contains(&b.to_string()),
            "{msg}"
        );
    }
    let (status, ctx) = ctx_with(0, DF_MAX_BATCH_ROWS);
    assert_eq!(status, DF_OK);
    assert_eq!(unsafe { &*ctx }.batch_rows, DF_MAX_BATCH_ROWS as usize);
    unsafe { df_ctx_free(ctx) };
}

// ---------------------------------------------------------------------------
// Errors and the panic barrier
// ---------------------------------------------------------------------------

/// With no context to hold it, an error must still be readable, so it goes to
/// the calling thread. With a context, it goes there and only there.
#[test]
fn a_null_out_is_a_null_pointer_error_recorded_thread_locally() {
    clear_thread_error();
    assert_eq!(
        unsafe { df_ctx_new(ptr::null(), ptr::null_mut()) },
        DF_ERR_NULL_POINTER
    );
    assert!(thread_error().unwrap().contains("out"));

    // A call that takes a context but is given NULL has the same problem.
    clear_thread_error();
    let (parent, lens) = fixture_arrays();
    let (status, tree) = tree_new(ptr::null_mut(), &parent, &lens);
    assert_eq!(status, DF_ERR_NULL_POINTER);
    assert!(tree.is_null());
    assert!(thread_error().unwrap().contains("ctx"));

    clear_thread_error();
    let ctx = new_ctx();
    let n = parent.len() as i64;
    let status = unsafe { df_tree_new(ctx, parent.as_ptr(), lens.as_ptr(), n, ptr::null_mut()) };
    assert_eq!(status, DF_ERR_NULL_POINTER);
    assert!(ctx_error(ctx).unwrap().contains("out"));
    assert_eq!(thread_error(), None);
    unsafe { df_ctx_free(ctx) };
}

/// Two contexts are independent, so one's error never shows on the other. The
/// message outlives later successful calls, because a C caller typically reads
/// it only after checking a status, and a later error replaces it.
#[test]
fn error_text_is_per_context_and_survives_until_the_next_error() {
    clear_thread_error();
    let (a, b) = (new_ctx(), new_ctx());
    let (parent, lens) = fixture_arrays();
    let mut bad = parent.clone();

    bad[5] = 99;
    assert_eq!(tree_new(a, &bad, &lens).0, DF_ERR_INVALID_INPUT);
    assert!(ctx_error(a).unwrap().contains("99"));
    assert_eq!(ctx_error(b), None);

    let (status, tree) = tree_new(a, &parent, &lens);
    assert_eq!(status, DF_OK);
    assert!(
        ctx_error(a).unwrap().contains("99"),
        "a success cleared the error"
    );

    bad[5] = 77;
    assert_eq!(tree_new(a, &bad, &lens).0, DF_ERR_INVALID_INPUT);
    let msg = ctx_error(a).unwrap();
    assert!(msg.contains("77") && !msg.contains("99"), "{msg}");

    assert_eq!(
        thread_error(),
        None,
        "an error with a context also reached the thread"
    );
    unsafe {
        df_tree_free(tree);
        df_ctx_free(a);
        df_ctx_free(b);
    }
}

/// A C string ends at its first NUL, so an embedded one would silently cut a
/// message short, and CString::new refuses one outright. Messages carry
/// caller-supplied text, so a NUL is replaced visibly instead, in both places
/// a message can be stored.
#[test]
fn nul_bytes_in_messages_are_sanitised() {
    let ctx = new_ctx();
    record_error(ctx, "before\0after");
    assert_eq!(ctx_error(ctx).as_deref(), Some("before\\0after"));

    clear_thread_error();
    record_error(ptr::null(), "before\0after");
    assert_eq!(thread_error().as_deref(), Some("before\\0after"));

    // And through the guard, which is how every entry point reports.
    let status = guard(ctx, || Err(ApiError::new(DF_ERR_INVALID_INPUT, "x\0y")));
    assert_eq!(status, DF_ERR_INVALID_INPUT);
    assert_eq!(ctx_error(ctx).as_deref(), Some("x\\0y"));
    unsafe { df_ctx_free(ctx) };
}

/// A panic unwinding into C aborts the host (Rust 1.81 and later), so every
/// entry point turns one into a status. The payload becomes the message,
/// because it is the only clue to the bug.
#[test]
#[cfg_attr(
    not(panic = "unwind"),
    ignore = "panic=abort: catch_unwind cannot catch"
)]
fn guard_turns_a_panic_into_the_panic_status_with_its_payload() {
    let ctx = new_ctx();
    assert_eq!(guard(ctx, || panic!("formatted {}", 42)), DF_ERR_PANIC);
    assert_eq!(ctx_error(ctx).as_deref(), Some("panic: formatted 42"));
    assert_eq!(guard(ctx, || panic!("static")), DF_ERR_PANIC);
    assert_eq!(ctx_error(ctx).as_deref(), Some("panic: static"));

    // The void and string barriers have no status to return, so they swallow it.
    let mut ran = false;
    guard_void(|| {
        ran = true;
        panic!("in a free function");
    });
    assert!(ran);
    assert!(guard_cstr(|| panic!("in a last_error function")).is_null());
    unsafe { df_ctx_free(ctx) };
}

/// C cleanup code frees unconditionally on its error paths, often with handles
/// still NULL. That must be a no-op, as free(NULL) is.
#[test]
fn free_functions_accept_null() {
    unsafe {
        df_ctx_free(ptr::null_mut());
        df_tree_free(ptr::null_mut());
        assert!(df_ctx_last_error(ptr::null()).is_null());
    }
}

// ---------------------------------------------------------------------------
// Trees
// ---------------------------------------------------------------------------

/// The fixture is the tree dartunifrac-arrow's and core's golden tests use, so
/// a handle that carries it unchanged can be held to their numbers.
#[test]
fn tree_new_accepts_the_fixture() {
    let ctx = new_ctx();
    let (parent, lens) = fixture_arrays();
    let (status, tree) = tree_new(ctx, &parent, &lens);
    assert_eq!(status, DF_OK, "{:?}", ctx_error(ctx));
    let expected = common::test_tree();
    let got = &unsafe { &*tree }.tree;
    assert_eq!(got.parent, expected.parent);
    assert_eq!(got.lens, expected.lens);
    unsafe {
        df_tree_free(tree);
        df_ctx_free(ctx);
    }
}

/// A parent outside the tree would index out of bounds in core. It is refused
/// and named, and the out-param is left NULL rather than half-built.
#[test]
fn tree_new_rejects_an_out_of_range_parent_naming_it() {
    let ctx = new_ctx();
    let (mut parent, lens) = fixture_arrays();
    parent[5] = 99;
    let (status, tree) = tree_new(ctx, &parent, &lens);
    assert_eq!(status, DF_ERR_INVALID_INPUT);
    assert!(tree.is_null());
    assert!(ctx_error(ctx).unwrap().contains("99"));
    unsafe { df_ctx_free(ctx) };
}

/// n_nodes sizes two slices over the caller's memory. A negative one means
/// nothing, and one too large for the address space would make the slice itself
/// undefined behaviour. Both are refused before either pointer is read.
#[test]
fn tree_new_rejects_negative_or_huge_n_nodes() {
    let ctx = new_ctx();
    let (parent, lens) = fixture_arrays();
    for n in [-1, i64::MIN, i64::MAX] {
        let mut tree = sentinel();
        let status = unsafe { df_tree_new(ctx, parent.as_ptr(), lens.as_ptr(), n, &mut tree) };
        assert_eq!(status, DF_ERR_INVALID_INPUT, "n_nodes = {n}");
        assert!(tree.is_null());
        let msg = ctx_error(ctx).unwrap();
        assert!(msg.contains(&n.to_string()), "{msg}");
    }
    unsafe { df_ctx_free(ctx) };
}

/// An empty tree has no branches, so nothing could ever be sketched against it.
/// The pointers, which C may legitimately leave NULL for an empty array, are
/// never read: the answer is the tree's emptiness, not a NULL-pointer error.
#[test]
fn tree_new_with_zero_nodes_is_rejected_without_reading_pointers() {
    let ctx = new_ctx();
    let mut tree = sentinel();
    let status = unsafe { df_tree_new(ctx, ptr::null(), ptr::null(), 0, &mut tree) };
    assert_eq!(status, DF_ERR_INVALID_INPUT, "{:?}", ctx_error(ctx));
    assert!(tree.is_null());
    assert!(ctx_error(ctx).unwrap().contains("positive"));
    unsafe { df_ctx_free(ctx) };
}

/// Core sketches only branches of positive length. With none, every sample's
/// sketch would be empty, and core would fail later with less to go on. Zero and
/// negative lengths are both "not positive", matching core's own `> 0.0` filter;
/// one positive branch is enough.
#[test]
fn tree_new_rejects_a_tree_with_no_positive_branch() {
    let ctx = new_ctx();
    let (status, tree) = tree_new(ctx, &[-1, 0, 0], &[0.0, 0.0, -1.0]);
    assert_eq!(status, DF_ERR_INVALID_INPUT);
    assert!(tree.is_null());
    assert!(ctx_error(ctx).unwrap().contains("positive"));

    let (status, tree) = tree_new(ctx, &[-1, 0, 0], &[0.0, 0.5, -1.0]);
    assert_eq!(status, DF_OK, "{:?}", ctx_error(ctx));
    unsafe {
        df_tree_free(tree);
        df_ctx_free(ctx);
    }
}

/// A NULL array with a nonzero length is a caller bug that would otherwise be a
/// NULL dereference. It is reported with the argument's name.
#[test]
fn tree_new_null_parent_with_nonzero_n_is_a_null_pointer_error() {
    let ctx = new_ctx();
    let (parent, lens) = fixture_arrays();
    let n = parent.len() as i64;
    let cases = [
        (ptr::null(), lens.as_ptr(), "parent"),
        (parent.as_ptr(), ptr::null(), "branch_length"),
    ];
    for (p, l, name) in cases {
        let mut tree = sentinel();
        let status = unsafe { df_tree_new(ctx, p, l, n, &mut tree) };
        assert_eq!(status, DF_ERR_NULL_POINTER, "{name}");
        assert!(tree.is_null());
        assert!(ctx_error(ctx).unwrap().contains(name));
    }
    unsafe { df_ctx_free(ctx) };
}

/// Reading through a misaligned pointer is undefined behaviour, and debug builds
/// abort on it. Refusing it turns a crash into a status.
#[test]
fn tree_new_rejects_a_misaligned_array() {
    let ctx = new_ctx();
    let (parent, lens) = fixture_arrays();
    // Off by one byte, and never read: the point is that it is refused first.
    let misaligned = parent.as_ptr().cast::<u8>().wrapping_add(1).cast::<i64>();
    let mut tree = sentinel();
    let n = parent.len() as i64 - 1;
    let status = unsafe { df_tree_new(ctx, misaligned, lens.as_ptr(), n, &mut tree) };
    assert_eq!(status, DF_ERR_INVALID_INPUT);
    assert!(tree.is_null());
    let msg = ctx_error(ctx).unwrap();
    assert!(msg.contains("parent") && msg.contains("aligned"), "{msg}");
    unsafe { df_ctx_free(ctx) };
}

// ---------------------------------------------------------------------------
// ABI
// ---------------------------------------------------------------------------

/// The C harness asserts the same layout with _Static_assert. Between them, a
/// field reordered or retyped on either side fails a build instead of
/// corrupting a call. The numbers hold on every supported target (x86-64,
/// aarch64, wasm32); i686, where u64 aligns to 4, is not one.
#[test]
fn abi_struct_sizes() {
    assert_eq!(
        (size_of::<df_config_t>(), align_of::<df_config_t>()),
        (16, 8)
    );
    assert_eq!(offset_of!(df_config_t, n_threads), 0);
    assert_eq!(offset_of!(df_config_t, batch_rows), 8);

    assert_eq!(
        (size_of::<df_params_t>(), align_of::<df_params_t>()),
        (24, 8)
    );
    assert_eq!(offset_of!(df_params_t, method), 0);
    assert_eq!(offset_of!(df_params_t, k), 4);
    assert_eq!(offset_of!(df_params_t, seed), 8);
    assert_eq!(offset_of!(df_params_t, bbits), 16);
    assert_eq!(offset_of!(df_params_t, weighted), 20);
    assert_eq!(offset_of!(df_params_t, raw_counts), 21);

    // Four callbacks and a pointer, on 64-bit and 32-bit targets alike.
    assert_eq!(size_of::<FFI_ArrowArrayStream>(), 5 * size_of::<usize>());
    // The test producer below writes this mirror where arrow-rs expects its own.
    assert_eq!(size_of::<CSchema>(), size_of::<FFI_ArrowSchema>());
}

// ===========================================================================
// Sketches and distances (M5b)
// ===========================================================================

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/// A context and the fixture tree, freed together.
struct World {
    ctx: *mut df_ctx_t,
    tree: *mut df_tree_t,
}

impl World {
    fn new() -> Self {
        Self::with_threads(0)
    }

    fn with_threads(n_threads: i32) -> Self {
        Self::with(n_threads, 0)
    }

    fn with(n_threads: i32, batch_rows: u64) -> Self {
        let (status, ctx) = ctx_with(n_threads, batch_rows);
        assert_eq!(status, DF_OK, "{:?}", thread_error());
        let (parent, lens) = fixture_arrays();
        let (status, tree) = tree_new(ctx, &parent, &lens);
        assert_eq!(status, DF_OK, "{:?}", ctx_error(ctx));
        World { ctx, tree }
    }

    fn error(&self) -> String {
        ctx_error(self.ctx).unwrap_or_default()
    }
}

impl Drop for World {
    fn drop(&mut self) {
        unsafe {
            df_tree_free(self.tree);
            df_ctx_free(self.ctx);
        }
    }
}

/// One past the global pool's size: a count no other pool in the process has.
fn distinct_thread_count() -> i32 {
    rayon::current_num_threads() as i32 + 1
}

/// Where a reader's `next` ran: rayon's thread count there, and the worker index
/// if it ran on a pool thread.
type Seen = Arc<Mutex<Vec<(usize, Option<usize>)>>>;

/// A reader that records where each `next` ran, and counts its own drops.
/// arrow-rs's exported stream drops its reader in `release`, so the drop count
/// is how many times the producer was released.
struct Probe {
    inner: Box<dyn RecordBatchReader + Send>,
    drops: Arc<AtomicUsize>,
    seen: Seen,
}

impl Iterator for Probe {
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        let here = (rayon::current_num_threads(), rayon::current_thread_index());
        self.seen.lock().unwrap().push(here);
        self.inner.next()
    }
}

impl RecordBatchReader for Probe {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        self.drops.fetch_add(1, SeqCst);
    }
}

/// A caller's input stream, exported through arrow-rs as a C++ producer would
/// hand it over.
struct Input {
    stream: FFI_ArrowArrayStream,
    drops: Arc<AtomicUsize>,
    seen: Seen,
}

fn input_from(schema: SchemaRef, items: Vec<Result<RecordBatch, ArrowError>>) -> Input {
    let drops = Arc::new(AtomicUsize::new(0));
    let seen = Seen::default();
    let probe = Probe {
        inner: Box::new(RecordBatchIterator::new(items, schema)),
        drops: drops.clone(),
        seen: seen.clone(),
    };
    Input {
        stream: FFI_ArrowArrayStream::new(Box::new(probe)),
        drops,
        seen,
    }
}

/// `rows` as one batch under the input schema.
fn input(rows: &[(i64, i64, f64)], weighted: bool) -> Input {
    input_from(
        common::schema_of(weighted),
        vec![Ok(common::batch(rows, weighted))],
    )
}

fn params(method: u32, weighted: bool, raw_counts: bool, bbits: u32) -> df_params_t {
    df_params_t {
        method,
        k: 16,
        seed: 1337,
        bbits,
        weighted,
        raw_counts,
    }
}

fn dmh(weighted: bool) -> df_params_t {
    params(DF_METHOD_DMH, weighted, false, 16)
}

/// df_sketch with the fixture's col_sums, which only a weighted run reads.
fn sketch(
    w: &World,
    p: &df_params_t,
    inp: &mut Input,
    n_samples: i64,
) -> (df_status_t, *mut df_sketches_t) {
    let col_sums = common::test_table().col_sums;
    let mut out = sentinel();
    let status = unsafe {
        df_sketch(
            w.ctx,
            w.tree,
            p,
            &mut inp.stream,
            n_samples,
            col_sums.as_ptr(),
            &mut out,
        )
    };
    (status, out)
}

/// df_sketch that must succeed.
fn sketch_ok(
    w: &World,
    p: &df_params_t,
    rows: &[(i64, i64, f64)],
    n_samples: i64,
) -> *mut df_sketches_t {
    let (status, s) = sketch(w, p, &mut input(rows, p.weighted), n_samples);
    assert_eq!(status, DF_OK, "{}", w.error());
    s
}

/// The fixture with sample 1 left out. Samples 0 and 2 leave positive branches
/// 5, 10 and 11 untouched, so the portable id space (every positive branch)
/// differs from the run-local one (only the touched ones), and a test that
/// mixed the two up could see it. The full fixture touches every branch.
fn samples_0_and_2() -> Vec<(i64, i64, f64)> {
    common::test_rows()
        .into_iter()
        .filter(|r| r.0 != 1)
        .collect()
}

/// A sketch set's values widened to u64, with its width, so that any two sets
/// compare alike.
fn sketch_values(s: &Sketches) -> (u32, Vec<Vec<u64>>) {
    fn widen<T: Copy + Into<u64>>(rows: &[Vec<T>]) -> Vec<Vec<u64>> {
        rows.iter()
            .map(|r| r.iter().map(|&x| x.into()).collect())
            .collect()
    }
    match s {
        Sketches::U16(r) => (16, widen(r)),
        Sketches::U32(r) => (32, widen(r)),
        Sketches::U64(r) => (64, widen(r)),
    }
}

fn handle(s: *const df_sketches_t) -> &'static df_sketches_t {
    unsafe { &*s }
}

/// Drain a stream through arrow-rs's importer, the way a C++ consumer reads one.
fn drain_stream(stream: FFI_ArrowArrayStream) -> Vec<RecordBatch> {
    let reader = RecordBatchStreamReader::try_new(stream).expect("the stream has a schema");
    reader.map(|b| b.expect("no batch should fail")).collect()
}

fn distances(w: &World, s: *const df_sketches_t) -> Vec<RecordBatch> {
    let mut out = FFI_ArrowArrayStream::empty();
    assert_eq!(
        unsafe { df_distances(w.ctx, s, &mut out) },
        DF_OK,
        "{}",
        w.error()
    );
    drain_stream(out)
}

/// `(i, j, distance)` with the distance as bits, so equality is exact.
fn bits(batches: &[RecordBatch]) -> Vec<(i64, i64, u32)> {
    common::triples(batches)
        .into_iter()
        .map(|(i, j, d)| (i, j, d.to_bits()))
        .collect()
}

fn samples_of(s: *const df_sketches_t) -> Vec<i64> {
    let (mut array, mut schema) = (FFI_ArrowArray::empty(), FFI_ArrowSchema::empty());
    assert_eq!(
        unsafe { df_sketch_samples(s, &mut array, &mut schema) },
        DF_OK,
        "{:?}",
        thread_error()
    );
    let data = unsafe { from_ffi(array, &schema) }.expect("a valid int64 array");
    Int64Array::from(data).values().to_vec()
}

/// What the API produced: the kept sample ids, the sketches and the distances.
type Outcome = (Vec<i64>, (u32, Vec<Vec<u64>>), Vec<(i64, i64, u32)>);

fn through_api(w: &World, p: &df_params_t, rows: &[(i64, i64, f64)]) -> Outcome {
    let s = sketch_ok(w, p, rows, 3);
    let out = (
        handle(s).ids.clone(),
        sketch_values(&handle(s).sketches),
        bits(&distances(w, s)),
    );
    unsafe { df_sketches_free(s) };
    out
}

/// The same run with arrow and core called directly, from parameters written
/// out as literals rather than translated by the code under test.
fn direct(
    w: &World,
    method: Method,
    p: &df_params_t,
    rows: &[(i64, i64, f64)],
    portable: bool,
) -> Outcome {
    let col_sums = common::test_table().col_sums;
    let pool = unsafe { &*w.ctx }.pool.clone();
    run_in_pool(pool.as_deref(), || {
        let reader = common::reader(
            common::schema_of(p.weighted),
            vec![common::batch(rows, p.weighted)],
        );
        let table =
            table_from_stream(reader, 3, p.weighted.then_some(&col_sums[..]), p.weighted).unwrap();
        let sp = SketchParams {
            k: p.k as usize,
            method,
            ers_l: 0,
            seed: p.seed,
            weighted: p.weighted,
            raw_counts: p.raw_counts,
            portable,
        };
        let set = build_sketches(&common::test_tree(), &table, &sp, "").unwrap();
        let ids: Vec<i64> = set.kept.iter().map(|&k| k as i64).collect();
        let sketches = trim_to_bbits(set.sketches, p.bbits as u8).unwrap();
        let values = sketch_values(&sketches);
        let reader = distances_reader(
            sketches,
            ids.clone(),
            p.weighted,
            DF_DEFAULT_BATCH_ROWS as usize,
        )
        .unwrap();
        (ids, values, bits(&common::drain(reader)))
    })
}

const METHODS: [(Method, u32); 2] = [(Method::Dmh, DF_METHOD_DMH), (Method::Tmh, DF_METHOD_TMH)];
/// (weighted, raw_counts): unweighted, weighted, and weighted on raw counts.
const WEIGHTINGS: [(bool, bool); 3] = [(false, false), (true, false), (true, true)];

// ---------------------------------------------------------------------------
// Producers a C caller could write, and arrow-rs's exporter never would
// ---------------------------------------------------------------------------

/// The C layout of `struct ArrowSchema`. arrow-rs keeps its fields private, and
/// a test producer needs to write one arrow-rs would never build.
#[repr(C)]
struct CSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut CSchema,
    dictionary: *mut CSchema,
    release: Option<unsafe extern "C" fn(*mut CSchema)>,
    private_data: *mut c_void,
}

unsafe extern "C" fn release_c_schema(s: *mut CSchema) {
    unsafe { (*s).release = None };
}

/// A schema whose format string is not UTF-8. arrow-rs's importer `expect`s
/// UTF-8 there, so reading this panics.
unsafe extern "C" fn non_utf8_schema(
    _: *mut FFI_ArrowArrayStream,
    out: *mut FFI_ArrowSchema,
) -> c_int {
    let schema = CSchema {
        format: c"\xff".as_ptr(),
        name: ptr::null(),
        metadata: ptr::null(),
        flags: 0,
        n_children: 0,
        children: ptr::null_mut(),
        dictionary: ptr::null_mut(),
        release: Some(release_c_schema),
        private_data: ptr::null_mut(),
    };
    unsafe { out.cast::<CSchema>().write(schema) };
    0
}

unsafe extern "C" fn unweighted_schema(
    _: *mut FFI_ArrowArrayStream,
    out: *mut FFI_ArrowSchema,
) -> c_int {
    let schema = FFI_ArrowSchema::try_from(&dartunifrac_arrow::input_schema(false)).unwrap();
    unsafe { out.write(schema) };
    0
}

/// EIO, with no message: the C Stream Interface lets get_last_error return NULL.
unsafe extern "C" fn fail_next(_: *mut FFI_ArrowArrayStream, _: *mut FFI_ArrowArray) -> c_int {
    5
}

unsafe extern "C" fn no_message(_: *mut FFI_ArrowArrayStream) -> *const c_char {
    ptr::null()
}

/// Counts its calls in the `Arc<AtomicUsize>` that `private_data` holds.
unsafe extern "C" fn count_release(s: *mut FFI_ArrowArrayStream) {
    let s = unsafe { &mut *s };
    let released = unsafe { Arc::from_raw(s.private_data().cast::<AtomicUsize>()) };
    released.fetch_add(1, SeqCst);
    unsafe { s.set_release(None) };
}

fn raw_stream(
    get_schema: unsafe extern "C" fn(*mut FFI_ArrowArrayStream, *mut FFI_ArrowSchema) -> c_int,
) -> (FFI_ArrowArrayStream, Arc<AtomicUsize>) {
    let released = Arc::new(AtomicUsize::new(0));
    let private = Arc::into_raw(released.clone()).cast_mut().cast::<c_void>();
    let stream = unsafe {
        FFI_ArrowArrayStream::new_unchecked(
            Some(get_schema),
            Some(fail_next),
            Some(no_message),
            Some(count_release),
            private,
        )
    };
    (stream, released)
}

// ---------------------------------------------------------------------------
// Parameters and bounds
// ---------------------------------------------------------------------------

/// Every field is validated before anything runs, and the rejection names the
/// field, because the caller has no other way to find which one. bbits 272 is
/// 16 once narrowed to a byte, so it proves the check runs on the full value.
#[test]
fn params_rejections_name_the_field() {
    let w = World::new();
    let rows = common::test_rows();
    let base = dmh(false);
    let cases: [(&str, df_params_t, &[&str]); 6] = [
        (
            "method 2",
            df_params_t { method: 2, ..base },
            &["ERS", "portable"],
        ),
        (
            "method 99",
            df_params_t { method: 99, ..base },
            &["method", "99"],
        ),
        ("k 0", df_params_t { k: 0, ..base }, &["k is 0"]),
        ("bbits 8", df_params_t { bbits: 8, ..base }, &["bbits", "8"]),
        (
            "bbits 272",
            df_params_t { bbits: 272, ..base },
            &["bbits", "272"],
        ),
        (
            "raw unweighted",
            df_params_t {
                raw_counts: true,
                ..base
            },
            &["raw_counts", "weighted"],
        ),
    ];
    for (label, p, needles) in cases {
        let (status, s) = sketch(&w, &p, &mut input(&rows, false), 3);
        assert_eq!(status, DF_ERR_INVALID_INPUT, "{label}");
        assert!(s.is_null(), "{label}");
        let msg = w.error();
        for needle in needles {
            assert!(msg.contains(needle), "{label}: {msg:?} lacks {needle:?}");
        }
    }
}

/// n_samples sizes an allocation before a single row is read. u32::MAX + 1 is
/// past the widest sample index the table stages, so it is refused up front.
#[test]
fn n_samples_negative_or_above_u32_max_is_rejected_and_the_stream_still_consumed() {
    let w = World::new();
    for n in [-1, i64::MIN, u32::MAX as i64 + 1] {
        let mut inp = input(&common::test_rows(), false);
        let (status, s) = sketch(&w, &dmh(false), &mut inp, n);
        assert_eq!(status, DF_ERR_INVALID_INPUT, "n_samples = {n}");
        assert!(s.is_null());
        assert!(w.error().contains(&n.to_string()), "{}", w.error());
        assert_eq!(inp.drops.load(SeqCst), 1, "n_samples = {n}");
    }
}

// ---------------------------------------------------------------------------
// Stream ownership
// ---------------------------------------------------------------------------

/// The caller cannot tell from a status whether its stream was read, so the
/// rule is that it is always consumed: released exactly once, whatever happens.
/// A stream released twice is a double free in the producer; never released,
/// a leak.
#[test]
fn input_stream_is_consumed_exactly_once_on_every_status() {
    let w = World::new();
    let rows = common::test_rows();
    let col_sums = common::test_table().col_sums;
    let one_bad_batch = || {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(vec![1]))])
            .unwrap();
        input_from(schema, vec![Ok(batch)])
    };
    let next_fails = || {
        input_from(
            common::schema_of(false),
            vec![Err(ArrowError::ComputeError("broke".into()))],
        )
    };
    type Call<'a> =
        Box<dyn Fn(*mut FFI_ArrowArrayStream, *mut *mut df_sketches_t) -> df_status_t + 'a>;
    let (good, bad) = (dmh(false), df_params_t { k: 0, ..dmh(false) });
    let weighted = dmh(true);
    let with = |ctx, tree, p: *const df_params_t, cs: *const f64, n| -> Call {
        Box::new(move |inp, out| unsafe { df_sketch(ctx, tree, p, inp, n, cs, out) })
    };
    let ok = || with(w.ctx, w.tree, &good, col_sums.as_ptr(), 3);
    let cases: Vec<(&str, Input, df_status_t, Call)> = vec![
        (
            "out NULL",
            input(&rows, false),
            DF_ERR_NULL_POINTER,
            Box::new(|inp, _| ok()(inp, ptr::null_mut())),
        ),
        (
            "ctx NULL",
            input(&rows, false),
            DF_ERR_NULL_POINTER,
            with(ptr::null_mut(), w.tree, &good, ptr::null(), 3),
        ),
        (
            "tree NULL",
            input(&rows, false),
            DF_ERR_NULL_POINTER,
            with(w.ctx, ptr::null(), &good, ptr::null(), 3),
        ),
        (
            "params NULL",
            input(&rows, false),
            DF_ERR_NULL_POINTER,
            with(w.ctx, w.tree, ptr::null(), ptr::null(), 3),
        ),
        (
            "bad params",
            input(&rows, false),
            DF_ERR_INVALID_INPUT,
            with(w.ctx, w.tree, &bad, ptr::null(), 3),
        ),
        (
            "n_samples < 0",
            input(&rows, false),
            DF_ERR_INVALID_INPUT,
            with(w.ctx, w.tree, &good, ptr::null(), -1),
        ),
        (
            "weighted, col_sums NULL",
            input(&rows, true),
            DF_ERR_NULL_POINTER,
            with(w.ctx, w.tree, &weighted, ptr::null(), 3),
        ),
        (
            "bad schema, one batch",
            one_bad_batch(),
            DF_ERR_INVALID_INPUT,
            ok(),
        ),
        ("get_next fails", next_fails(), DF_ERR_ARROW, ok()),
        ("success", input(&rows, false), DF_OK, ok()),
    ];
    for (label, mut inp, expected, call) in cases {
        let mut out = sentinel();
        let status = call(&mut inp.stream, &mut out);
        assert_eq!(status, expected, "{label}: {}", w.error());
        assert!(
            inp.stream.release().is_none(),
            "{label}: the caller's stream still has a release"
        );
        assert_eq!(
            inp.drops.load(SeqCst),
            1,
            "{label}: released {} times",
            inp.drops.load(SeqCst)
        );
        if status == DF_OK {
            unsafe { df_sketches_free(out) };
        }
    }

    // Already released: refused, and its release is not called a second time.
    let mut inp = input(&rows, false);
    drop(std::mem::replace(
        &mut inp.stream,
        FFI_ArrowArrayStream::empty(),
    ));
    assert_eq!(inp.drops.load(SeqCst), 1);
    let mut out = sentinel();
    assert_eq!(ok()(&mut inp.stream, &mut out), DF_ERR_ARROW);
    assert!(out.is_null());
    assert_eq!(inp.drops.load(SeqCst), 1);
}

/// The C Stream Interface lets a producer fail without saying why. arrow 60
/// reports that as an ordinary error (59 panicked on it), so it is the Arrow
/// status, and the stream is still released exactly once.
#[test]
fn producer_error_without_a_message_is_the_arrow_status_and_released_once() {
    let w = World::new();
    let (mut stream, released) = raw_stream(unweighted_schema);
    let mut out = sentinel();
    let col_sums = common::test_table().col_sums;
    let status = unsafe {
        df_sketch(
            w.ctx,
            w.tree,
            &dmh(false),
            &mut stream,
            3,
            col_sums.as_ptr(),
            &mut out,
        )
    };
    assert_eq!(status, DF_ERR_ARROW, "{}", w.error());
    assert!(out.is_null());
    assert_eq!(released.load(SeqCst), 1);
    assert!(stream.release().is_none());
}

/// A producer can break the C Data Interface badly enough that arrow-rs's
/// importer panics: here, with a schema format that is not UTF-8. That must
/// come back as a status, not abort the host, and the stream must still be
/// released exactly once while the panic unwinds. With a dedicated pool the
/// panic happens on a worker and crosses back through rayon, so both are run.
#[test]
#[cfg_attr(
    not(panic = "unwind"),
    ignore = "panic=abort: catch_unwind cannot catch"
)]
fn a_malformed_producer_schema_is_the_panic_status_and_released_once() {
    let col_sums = common::test_table().col_sums;
    for threads in [0, 2] {
        let w = World::with_threads(threads);
        let (mut stream, released) = raw_stream(non_utf8_schema);
        let mut out = sentinel();
        let status = unsafe {
            df_sketch(
                w.ctx,
                w.tree,
                &dmh(false),
                &mut stream,
                3,
                col_sums.as_ptr(),
                &mut out,
            )
        };
        assert_eq!(status, DF_ERR_PANIC, "threads={threads}: {}", w.error());
        assert!(w.error().starts_with("panic: "), "{}", w.error());
        assert!(out.is_null());
        assert_eq!(released.load(SeqCst), 1, "threads={threads}");
    }
}

/// A failing producer's own message is the most useful thing to hand back, so
/// it reaches the caller intact.
#[test]
fn producer_get_next_error_is_the_arrow_status_with_its_text() {
    let w = World::new();
    let mut inp = input_from(
        common::schema_of(false),
        vec![Err(ArrowError::ComputeError("the producer broke".into()))],
    );
    let (status, s) = sketch(&w, &dmh(false), &mut inp, 3);
    assert_eq!(status, DF_ERR_ARROW);
    assert!(s.is_null());
    assert!(w.error().contains("the producer broke"), "{}", w.error());
}

// ---------------------------------------------------------------------------
// Gate B: the API changes nothing
// ---------------------------------------------------------------------------

/// The C API is packaging, so its sketches and distances must be exactly what
/// core and arrow produce when called directly with the same, portable,
/// parameters: for both methods, all three weightings and every width. Run on
/// samples 0 and 2, where the portable id space differs from the run-local
/// one, and on all three samples, for more than one pair.
#[test]
fn sketches_and_distances_match_core_bit_for_bit() {
    let w = World::new();
    let fixtures = [
        (samples_0_and_2(), vec![0, 2]),
        (common::test_rows(), vec![0, 1, 2]),
    ];
    let mut distinct = std::collections::BTreeSet::new();
    for (rows, kept) in &fixtures {
        for (method, code) in METHODS {
            for (weighted, raw) in WEIGHTINGS {
                for bbits in [16, 32, 64] {
                    let p = params(code, weighted, raw, bbits);
                    let label = format!(
                        "{method:?} weighted={weighted} raw={raw} bbits={bbits} kept={kept:?}"
                    );
                    let api = through_api(&w, &p, rows);
                    assert_eq!(api, direct(&w, method, &p, rows, true), "{label}");
                    // Not vacuous: one sketch of k values per kept sample, at the
                    // asked width, and every pair, each a real distance.
                    let n = kept.len();
                    assert_eq!(&api.0, kept, "{label}");
                    assert_eq!(
                        (api.1.0, api.1.1.len(), api.1.1[0].len()),
                        (bbits, n, 16),
                        "{label}"
                    );
                    assert_eq!(api.2.len(), n * (n - 1) / 2, "{label}");
                    for &(_, _, d) in &api.2 {
                        let d = f32::from_bits(d);
                        assert!((0.0..=1.0).contains(&d), "{label}: distance {d}");
                        distinct.insert(d.to_bits());
                    }
                }
            }
        }
    }
    // Equal constants on both sides would also be "identical".
    assert!(distinct.len() >= 5, "distances barely vary: {distinct:?}");
}

/// Gate B could pass vacuously if the fixture could not see the id space or the
/// weighting. It can: the run-local id space gives different sketches from the
/// API's portable one, and raw counts give different weighted sketches.
#[test]
fn gate_b_controls() {
    let w = World::new();
    let rows = samples_0_and_2();
    for (method, code) in METHODS {
        for (weighted, raw) in WEIGHTINGS {
            let p = params(code, weighted, raw, 64);
            let api = through_api(&w, &p, &rows);
            let run_local = direct(&w, method, &p, &rows, false);
            assert_ne!(
                api.1, run_local.1,
                "{method:?} weighted={weighted} raw={raw}"
            );
        }
        let relative = through_api(&w, &params(code, true, false, 64), &rows);
        let raw = through_api(&w, &params(code, true, true, 64), &rows);
        assert_ne!(relative.1, raw.1, "{method:?}: raw_counts changed nothing");
    }
}

// ---------------------------------------------------------------------------
// Sketch semantics
// ---------------------------------------------------------------------------

/// A query engine streams rows in whatever order and batching it likes, so
/// neither may change a sketch or a distance.
#[test]
fn multi_batch_shuffled_input_sketches_identically() {
    let w = World::new();
    let rows = common::test_rows();
    let shuffled = common::shuffled(&rows);
    common::assert_no_sample_arrives_sorted(&shuffled, 3);
    for weighted in [false, true] {
        let p = dmh(weighted);
        let one = through_api(&w, &p, &rows);
        let batches = common::batched(&shuffled, 2, weighted)
            .into_iter()
            .map(Ok)
            .collect();
        let mut inp = input_from(common::schema_of(weighted), batches);
        let (status, s) = sketch(&w, &p, &mut inp, 3);
        assert_eq!(status, DF_OK, "{}", w.error());
        let many = (
            handle(s).ids.clone(),
            sketch_values(&handle(s).sketches),
            bits(&distances(&w, s)),
        );
        unsafe { df_sketches_free(s) };
        assert_eq!(many, one, "weighted={weighted}");
        assert_eq!(one.2.len(), 3);
    }
}

/// With fewer than two sketched samples there is nothing to compare. For a
/// query over a filtered table that is an ordinary outcome, not a failure: the
/// result is DF_OK, no samples, and a distance stream with a schema and no rows.
#[test]
fn fewer_than_two_nonempty_samples_is_ok_and_empty() {
    let w = World::new();
    let sample_0: Vec<_> = common::test_rows()
        .into_iter()
        .filter(|r| r.0 == 0)
        .collect();
    for weighted in [false, true] {
        let cases = [
            (
                "no samples",
                input_from(common::schema_of(weighted), vec![]),
                0,
            ),
            (
                "no batches",
                input_from(common::schema_of(weighted), vec![]),
                3,
            ),
            ("one sample", input(&sample_0, weighted), 3),
        ];
        for (label, mut inp, n) in cases {
            let (status, s) = sketch(&w, &dmh(weighted), &mut inp, n);
            assert_eq!(status, DF_OK, "{label} weighted={weighted}: {}", w.error());
            assert_eq!(samples_of(s), Vec::<i64>::new(), "{label}");
            assert_eq!(distances(&w, s).len(), 0, "{label}");
            unsafe { df_sketches_free(s) };
        }
    }
}

/// The caller resolves feature names to tree nodes, and a feature is a tip. An
/// entry on an internal node means that resolution went wrong, and sketching it
/// anyway would quietly credit a whole clade, so it is refused and named.
#[test]
fn a_non_tip_node_index_is_rejected_naming_it() {
    let w = World::new();
    let mut rows = common::test_rows();
    rows.push((1, 2, 1.0)); // node 2 is the parent of nodes 3 and 4
    for weighted in [false, true] {
        let (status, s) = sketch(&w, &dmh(weighted), &mut input(&rows, weighted), 3);
        assert_eq!(status, DF_ERR_INVALID_INPUT, "weighted={weighted}");
        assert!(s.is_null());
        let msg = w.error();
        assert!(
            msg.contains("node_idx 2") && msg.contains("sample 1") && msg.contains("tip"),
            "{msg}"
        );
    }
}

/// The tip check reads the tree by node id, so an id past the tree's end must
/// be an error from the range check, not an out-of-bounds panic.
#[test]
fn a_node_index_outside_the_tree_is_rejected_not_a_panic() {
    let w = World::new();
    let mut rows = common::test_rows();
    rows.push((0, 99, 1.0));
    for weighted in [false, true] {
        let (status, s) = sketch(&w, &dmh(weighted), &mut input(&rows, weighted), 3);
        assert_eq!(
            status,
            DF_ERR_INVALID_INPUT,
            "weighted={weighted}: {}",
            w.error()
        );
        assert!(s.is_null());
        assert!(w.error().contains("99"), "{}", w.error());
    }
}

/// A sample with nothing to sketch gets no sketch, so the caller needs the list
/// of who survived. Sample 1 has an entry, but on node 0, a tip with no branch
/// of positive length above it; samples 0 and 2 are kept.
#[test]
fn samples_report_kept_ids_and_skip_a_dropped_sample() {
    let w = World::new();
    let mut rows = samples_0_and_2();
    rows.push((1, 0, 5.0));
    for weighted in [false, true] {
        let s = sketch_ok(&w, &dmh(weighted), &rows, 3);
        assert_eq!(samples_of(s), vec![0, 2], "weighted={weighted}");
        unsafe { df_sketches_free(s) };
    }
}

/// Distances name samples by the caller's own sample_idx, not by position in
/// the sketch set, so the caller needs no second join to read them.
#[test]
fn distances_ids_are_the_callers_sample_idx() {
    let w = World::new();
    let rows: Vec<_> = samples_0_and_2()
        .into_iter()
        .map(|(s, n, v)| (if s == 0 { 1 } else { 4 }, n, v))
        .collect();
    let s = sketch_ok(&w, &dmh(false), &rows, 5);
    let pairs: Vec<_> = bits(&distances(&w, s))
        .into_iter()
        .map(|(i, j, _)| (i, j))
        .collect();
    assert_eq!(pairs, vec![(1, 4)]);
    unsafe { df_sketches_free(s) };
}

/// batch_rows bounds every output batch, which is how the caller bounds memory,
/// and moves only the batch boundaries, never a distance.
#[test]
fn batches_respect_config_batch_rows() {
    let rows = common::test_rows();
    let expected = {
        let w = World::new();
        let s = sketch_ok(&w, &dmh(false), &rows, 3);
        let d = bits(&distances(&w, s));
        unsafe { df_sketches_free(s) };
        d
    };
    assert_eq!(expected.len(), 3);
    for batch_rows in [1u64, 2, 3, 4] {
        let w = World::with(0, batch_rows);
        let s = sketch_ok(&w, &dmh(false), &rows, 3);
        let batches = distances(&w, s);
        assert!(
            batches.iter().all(|b| b.num_rows() as u64 <= batch_rows),
            "batch_rows = {batch_rows}"
        );
        assert_eq!(bits(&batches), expected, "batch_rows = {batch_rows}");
        if batch_rows == 1 {
            assert_eq!(batches.len(), 3);
        }
        unsafe { df_sketches_free(s) };
    }
}

// ---------------------------------------------------------------------------
// Lifetime and threads
// ---------------------------------------------------------------------------

/// df_distances itself must hand its stream the context's pool: every result
/// test would still pass if it ran on rayon's global pool instead, since
/// results do not depend on the pool, but the caller's thread budget would be
/// silently ignored. The stream holds the pool from creation until release.
#[test]
#[cfg_attr(target_family = "wasm", ignore = "rayon cannot spawn threads on wasm")]
fn df_distances_gives_its_stream_the_ctx_pool() {
    let w = World::with_threads(2);
    let s = sketch_ok(&w, &dmh(false), &common::test_rows(), 3);
    let holders = || Arc::strong_count(unsafe { &*w.ctx }.pool.as_ref().unwrap());
    assert_eq!(holders(), 1);
    let mut stream = FFI_ArrowArrayStream::empty();
    assert_eq!(unsafe { df_distances(w.ctx, s, &mut stream) }, DF_OK);
    assert_eq!(holders(), 2, "the stream does not hold the context's pool");
    drop(stream);
    assert_eq!(
        holders(),
        1,
        "releasing the stream did not let go of the pool"
    );
    unsafe { df_sketches_free(s) };
}

/// A query engine releases a result stream long after the call that made it,
/// often after the objects it came from are gone. The stream must not depend
/// on them, including on the context's pool.
#[test]
fn a_stream_outlives_its_ctx_and_sketches() {
    let threads = if cfg!(target_family = "wasm") { 0 } else { 2 };
    let w = World::with_threads(threads);
    let s = sketch_ok(&w, &dmh(true), &common::test_rows(), 3);
    let expected = bits(&distances(&w, s));
    let mut stream = FFI_ArrowArrayStream::empty();
    assert_eq!(unsafe { df_distances(w.ctx, s, &mut stream) }, DF_OK);
    unsafe { df_sketches_free(s) };
    drop(w);
    let got = bits(&drain_stream(stream));
    assert_eq!(got, expected);
    assert_eq!(got.len(), 3);
}

/// The thread count is a resource choice, so it must never show in a result.
#[test]
#[cfg_attr(target_family = "wasm", ignore = "rayon cannot spawn threads on wasm")]
fn output_is_identical_whatever_the_thread_count() {
    let (one, many) = (
        World::with_threads(1),
        World::with_threads(distinct_thread_count()),
    );
    for weighted in [false, true] {
        let p = dmh(weighted);
        let rows = common::test_rows();
        assert_eq!(
            through_api(&one, &p, &rows),
            through_api(&many, &p, &rows),
            "weighted={weighted}"
        );
    }
}

/// Decision 2: the caller's stream is drained inside the context's pool, so its
/// thread budget covers the reading as well as the sketching.
#[test]
#[cfg_attr(target_family = "wasm", ignore = "rayon cannot spawn threads on wasm")]
fn input_get_next_runs_inside_the_ctx_pool() {
    let n = distinct_thread_count();
    let w = World::with_threads(n);
    let mut inp = input(&common::test_rows(), false);
    let (status, s) = sketch(&w, &dmh(false), &mut inp, 3);
    assert_eq!(status, DF_OK, "{}", w.error());
    let seen = inp.seen.lock().unwrap().clone();
    assert!(seen.len() >= 2, "{seen:?}"); // one batch, then the end
    for (threads, index) in seen {
        assert_eq!(threads, n as usize);
        assert!(index.is_some(), "get_next ran on the calling thread");
    }
    unsafe { df_sketches_free(s) };
}

/// A distance stream computes in its context's pool whichever thread calls
/// get_next, and still does after the context is freed, since the stream holds
/// the pool itself. Checked through the exported callback a C consumer calls.
#[test]
#[cfg_attr(target_family = "wasm", ignore = "rayon cannot spawn threads on wasm")]
fn exported_stream_work_runs_inside_the_ctx_pool() {
    let n = distinct_thread_count();
    let w = World::with_threads(n);
    let seen = Seen::default();
    let probe = Probe {
        inner: Box::new(common::reader(
            Arc::new(dartunifrac_arrow::output_schema()),
            vec![],
        )),
        drops: Arc::default(),
        seen: seen.clone(),
    };
    let pool = unsafe { &*w.ctx }.pool.clone();
    let mut stream = FFI_ArrowArrayStream::new(Box::new(PoolReader {
        inner: Box::new(probe),
        pool,
        done: false,
    }));
    drop(w);
    let (rc, array) = c_get_next(&mut stream);
    assert_eq!((rc, array.is_none()), (0, true));
    let seen = seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].0, n as usize);
    assert!(seen[0].1.is_some(), "get_next ran on the calling thread");
}

// ---------------------------------------------------------------------------
// Failures inside an exported stream
// ---------------------------------------------------------------------------

/// Fails on its first batch: by panicking, or with an error whose text has a NUL.
struct Failing {
    panic: bool,
}

impl Iterator for Failing {
    type Item = Result<RecordBatch, ArrowError>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.panic {
            panic!("boom\0tail");
        }
        Some(Err(ArrowError::ComputeError("bad\0text".into())))
    }
}

impl RecordBatchReader for Failing {
    fn schema(&self) -> SchemaRef {
        Arc::new(dartunifrac_arrow::output_schema())
    }
}

fn exported_failing(panic: bool, pool: Option<Arc<ThreadPool>>) -> FFI_ArrowArrayStream {
    let inner = Box::new(Failing { panic });
    FFI_ArrowArrayStream::new(Box::new(PoolReader {
        inner,
        pool,
        done: false,
    }))
}

/// No pool, and (where threads exist) a dedicated one, where a panic happens on
/// a worker and crosses back through rayon before the barrier sees it.
fn pools() -> Vec<Option<Arc<ThreadPool>>> {
    let mut pools = vec![None];
    if cfg!(not(target_family = "wasm")) {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        pools.push(Some(Arc::new(pool)));
    }
    pools
}

/// The stream's own get_next, called the way a C consumer calls it.
fn c_get_next(s: &mut FFI_ArrowArrayStream) -> (c_int, Option<FFI_ArrowArray>) {
    let mut array = FFI_ArrowArray::empty();
    let rc = unsafe { s.get_next().unwrap()(s, &mut array) };
    (rc, (!array.is_released()).then_some(array))
}

fn c_last_error(s: &mut FFI_ArrowArrayStream) -> String {
    let p = unsafe { s.get_last_error().unwrap()(s) };
    text(p).expect("a failed get_next leaves a message")
}

/// arrow-rs's exported get_next is extern "C" with no panic barrier, so a panic
/// in our reader would abort the host. It becomes a stream error instead, with
/// the NUL in its payload spelled out (arrow-rs aborts on a NUL in an error
/// message too), and the stream then ends.
#[test]
#[cfg_attr(
    not(panic = "unwind"),
    ignore = "panic=abort: catch_unwind cannot catch"
)]
fn a_panic_inside_the_exported_stream_becomes_an_error_then_the_end() {
    for pool in pools() {
        let label = if pool.is_some() {
            "dedicated pool"
        } else {
            "no pool"
        };
        let mut stream = exported_failing(true, pool);
        let (rc, array) = c_get_next(&mut stream);
        assert!(rc != 0 && array.is_none(), "{label}");
        let msg = c_last_error(&mut stream);
        assert!(msg.contains("panic: boom\\0tail"), "{label}: {msg}");
        let (rc, array) = c_get_next(&mut stream);
        assert_eq!(
            (rc, array.is_none()),
            (0, true),
            "{label}: the stream did not end"
        );
    }
}

/// The same abort waits for an ordinary error whose text holds a NUL.
#[test]
fn an_error_with_a_nul_inside_the_exported_stream_is_sanitised_not_an_abort() {
    for pool in pools() {
        let mut stream = exported_failing(false, pool);
        let (rc, array) = c_get_next(&mut stream);
        assert!(rc != 0 && array.is_none());
        let msg = c_last_error(&mut stream);
        assert!(msg.contains("bad\\0text"), "{msg}");
    }
}

// ---------------------------------------------------------------------------
// Null pointers and reset outputs
// ---------------------------------------------------------------------------

/// Memory as a C caller hands over an out-param: uninitialised. Filled here
/// with bytes that are not a released struct, so a callee that fails without
/// resetting it is caught.
fn garbage<T>() -> MaybeUninit<T> {
    let mut m = MaybeUninit::uninit();
    unsafe { ptr::write_bytes(m.as_mut_ptr(), 0xAB, 1) };
    m
}

/// Whether an out-param came back released. Never drops one that did not:
/// its release would be a garbage pointer.
fn released_array(m: MaybeUninit<FFI_ArrowArray>) -> bool {
    let a = unsafe { m.assume_init() };
    let released = a.is_released();
    if !released {
        std::mem::forget(a);
    }
    released
}

fn released_schema(m: MaybeUninit<FFI_ArrowSchema>) -> bool {
    let s = unsafe { m.assume_init() };
    let released = s.release().is_none();
    if !released {
        std::mem::forget(s);
    }
    released
}

fn released_stream(m: MaybeUninit<FFI_ArrowArrayStream>) -> bool {
    let s = unsafe { m.assume_init() };
    let released = s.release().is_none();
    if !released {
        std::mem::forget(s);
    }
    released
}

/// A NULL is a status, never a crash, and every Arrow out-param the call was
/// given is left released, so a caller's cleanup path can always release it.
#[test]
fn null_pointers_on_every_new_entry_point_return_a_status_and_reset_outputs() {
    let w = World::new();
    let s = sketch_ok(&w, &dmh(false), &common::test_rows(), 3);
    let col_sums = common::test_table().col_sums;

    let mut out = sentinel();
    let status = unsafe {
        df_sketch(
            w.ctx,
            w.tree,
            &dmh(false),
            ptr::null_mut(),
            3,
            col_sums.as_ptr(),
            &mut out,
        )
    };
    assert_eq!(status, DF_ERR_NULL_POINTER);
    assert!(out.is_null());
    assert_eq!(w.error(), "null pointer: input");

    for (label, sk, with_array, with_schema) in [
        ("sketches", ptr::null(), true, true),
        ("out", s.cast_const(), false, true),
        ("out_schema", s.cast_const(), true, false),
    ] {
        clear_thread_error();
        let (mut array, mut schema) = (garbage::<FFI_ArrowArray>(), garbage::<FFI_ArrowSchema>());
        let a = if with_array {
            array.as_mut_ptr()
        } else {
            ptr::null_mut()
        };
        let sc = if with_schema {
            schema.as_mut_ptr()
        } else {
            ptr::null_mut()
        };
        assert_eq!(
            unsafe { df_sketch_samples(sk, a, sc) },
            DF_ERR_NULL_POINTER,
            "{label}"
        );
        assert_eq!(thread_error(), Some(format!("null pointer: {label}")));
        if with_array {
            assert!(released_array(array), "{label}: array not reset");
        }
        if with_schema {
            assert!(released_schema(schema), "{label}: schema not reset");
        }
    }

    // A NULL ctx reports on the thread; otherwise on the ctx.
    for (label, ctx, sk) in [
        ("ctx", ptr::null_mut(), s.cast_const()),
        ("sketches", w.ctx, ptr::null()),
    ] {
        clear_thread_error();
        let mut stream = garbage::<FFI_ArrowArrayStream>();
        assert_eq!(
            unsafe { df_distances(ctx, sk, stream.as_mut_ptr()) },
            DF_ERR_NULL_POINTER,
            "{label}"
        );
        let msg = if ctx.is_null() {
            thread_error()
        } else {
            ctx_error(ctx)
        };
        assert_eq!(msg, Some(format!("null pointer: {label}")));
        assert!(released_stream(stream), "{label}: stream not reset");
    }
    assert_eq!(
        unsafe { df_distances(w.ctx, s, ptr::null_mut()) },
        DF_ERR_NULL_POINTER
    );
    assert_eq!(w.error(), "null pointer: out");

    unsafe {
        df_sketches_free(s);
        df_sketches_free(ptr::null_mut());
    }
}
