//! The C ABI, exercised from Rust through the exported functions themselves.
//! In-crate so a test can look inside a handle (which pool a context holds,
//! what tree a handle carries) without the ABI having to expose it.

use std::ffi::{CStr, c_char};
use std::mem::{align_of, offset_of, size_of};
use std::ptr;

use dartunifrac_core::NO_PARENT;

use crate::guard::{ApiError, THREAD_LAST_ERROR, guard, guard_cstr, guard_void, record_error};
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
/// on the calling thread, not on some pool's worker.
#[test]
fn null_config_means_no_dedicated_pool_and_default_batch_rows() {
    let null = new_ctx();
    let (status, zero) = ctx_with(0, 0);
    assert_eq!(status, DF_OK);
    for ctx in [null, zero] {
        let c = unsafe { &*ctx };
        assert!(c.pool.is_none());
        assert_eq!(c.batch_rows, DF_DEFAULT_BATCH_ROWS as usize);
        assert_eq!(
            run_in_pool(c.pool.as_deref(), rayon::current_thread_index),
            None
        );
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

/// Building a pool traps on a wasm build without threads. A trap cannot be
/// caught, so the request is refused before rayon is asked.
#[test]
#[cfg(all(target_family = "wasm", not(target_feature = "atomics")))]
fn a_dedicated_pool_on_wasm_without_threads_is_rejected_not_a_trap() {
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
}
