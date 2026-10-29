//! The panic barrier, and the error plumbing every entry point goes through.

use std::any::Any;
use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use dartunifrac_arrow::MarshalError;

use crate::{
    DF_ERR_ARROW, DF_ERR_INVALID_INPUT, DF_ERR_NULL_POINTER, DF_ERR_PANIC, DF_OK, df_ctx_t,
    df_status_t,
};

/// An error on its way out of an entry point: a status class and the message
/// [`guard`] records for the caller.
pub(crate) struct ApiError {
    pub(crate) code: df_status_t,
    pub(crate) message: String,
}

impl ApiError {
    pub(crate) fn new(code: df_status_t, message: impl Into<String>) -> Self {
        ApiError {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn null(what: &str) -> Self {
        ApiError::new(DF_ERR_NULL_POINTER, format!("null pointer: {what}"))
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        ApiError::new(DF_ERR_INVALID_INPUT, message)
    }
}

impl From<MarshalError> for ApiError {
    fn from(e: MarshalError) -> Self {
        let code = match e {
            // The caller's own stream failed; nothing of theirs was rejected.
            MarshalError::Stream(_) => DF_ERR_ARROW,
            MarshalError::Schema(_)
            | MarshalError::Data(_)
            | MarshalError::Tree(_)
            | MarshalError::Sketch(_) => DF_ERR_INVALID_INPUT,
        };
        ApiError::new(code, e.to_string())
    }
}

thread_local! {
    /// The last error of a call on this thread that had no context to record it
    /// on; read through `df_thread_last_error`.
    pub(crate) static THREAD_LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Make `message` safe to hand to C: an embedded NUL would end the C string
/// early, so it is spelled out instead.
pub(crate) fn sanitise(message: &str) -> String {
    message.replace('\0', "\\0")
}

/// Record `message` as `ctx`'s last error or, when `ctx` is NULL, as the
/// calling thread's.
///
/// Runs outside the panic barrier, so it must not panic: the slots are reached
/// with `try_with` and `try_borrow_mut`, which fail quietly where `with` and
/// `borrow_mut` would panic (during thread teardown, or under a reentrant
/// borrow the API contract already forbids). Losing a message is better than
/// aborting the host.
pub(crate) fn record_error(ctx: *const df_ctx_t, message: &str) {
    // Cannot fail: sanitise removed every NUL.
    let message = CString::new(sanitise(message)).unwrap_or_default();
    let store = |slot: &RefCell<Option<CString>>| {
        if let Ok(mut slot) = slot.try_borrow_mut() {
            *slot = Some(message);
        }
    };
    // SAFETY: a non-NULL ctx is a live context, by the API contract.
    match unsafe { ctx.as_ref() } {
        Some(ctx) => store(&ctx.last_error),
        None => {
            let _ = THREAD_LAST_ERROR.try_with(store);
        }
    }
}

pub(crate) fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_owned()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_owned()
    }
}

/// Run `f` behind the panic barrier and map its outcome to a status, recording
/// any error on `ctx` (or on the thread, when `ctx` is NULL). A panic becomes
/// `DF_ERR_PANIC`; it never unwinds into C.
pub(crate) fn guard<F>(ctx: *const df_ctx_t, f: F) -> df_status_t
where
    F: FnOnce() -> Result<(), ApiError>,
{
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => DF_OK,
        Ok(Err(e)) => {
            record_error(ctx, &e.message);
            e.code
        }
        Err(payload) => {
            record_error(ctx, &format!("panic: {}", panic_message(payload)));
            DF_ERR_PANIC
        }
    }
}

/// The barrier for the `void` free functions, which have no status to return:
/// a panic is swallowed.
pub(crate) fn guard_void<F: FnOnce()>(f: F) {
    let _ = catch_unwind(AssertUnwindSafe(f));
}

/// The barrier for the functions that return a message: a panic returns NULL.
pub(crate) fn guard_cstr<F: FnOnce() -> *const c_char>(f: F) -> *const c_char {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(std::ptr::null())
}

/// Borrow `n` elements of a C array.
///
/// Length 0 is an empty slice and `p` is never read, since C may pass NULL for
/// an empty array. Otherwise `p` must be non-NULL and aligned, and the span must
/// fit the address space; `slice::from_raw_parts` makes each of those undefined
/// behaviour, and its debug check aborts rather than unwinds.
///
/// # Safety
/// A non-NULL, aligned `p` must point to `n` initialised elements that outlive
/// the returned slice.
pub(crate) unsafe fn slice_or_empty<'a, T>(
    p: *const T,
    n: usize,
    what: &str,
) -> Result<&'a [T], ApiError> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(ApiError::null(what));
    }
    if !p.is_aligned() {
        return Err(ApiError::invalid(format!(
            "{what} is not aligned for its element type"
        )));
    }
    if n > isize::MAX as usize / size_of::<T>().max(1) {
        return Err(ApiError::invalid(format!(
            "{what} has {n} elements, more than this platform can address"
        )));
    }
    // SAFETY: checked above, and the caller's contract for the rest.
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}
