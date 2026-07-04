use crate::dartunifrac::{
    DartUniFracDistanceMatrix, DartUniFracOptions, compute_dartunifrac_matrix,
    init_logging_from_env, run_dartunifrac,
};
use std::{
    cell::RefCell,
    ffi::{CStr, CString},
    os::raw::{c_char, c_int},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

pub const DARTUNIFRAC_OK: c_int = 0;
pub const DARTUNIFRAC_ERROR: c_int = -1;
pub const DARTUNIFRAC_NULL_POINTER: c_int = -2;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DartUniFracConfig {
    pub tree_path: *const c_char,
    pub input_tsv_path: *const c_char,
    pub biom_path: *const c_char,
    pub output_path: *const c_char,
    pub method: *const c_char,
    pub sketch_size: usize,
    pub ers_length: u64,
    pub seed: u64,
    pub bbits: u8,
    pub weighted: u8,
    pub raw_counts: u8,
    pub succ: u8,
    pub compress: u8,
    pub pcoa: u8,
    pub streaming: u8,
    pub block_rows: usize,
    pub threads: usize,
}

#[repr(C)]
pub struct DartUniFracMatrix {
    _private: [u8; 0],
}

struct MatrixHandle {
    sample_names: Vec<CString>,
    sample_name_ptrs: Vec<*const c_char>,
    distances: Vec<f32>,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn cstring_sanitize(s: impl AsRef<str>) -> CString {
    let replaced = s.as_ref().replace('\0', "\\0");
    CString::new(replaced).expect("interior NUL bytes were replaced")
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

fn set_last_error(message: impl AsRef<str>) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(cstring_sanitize(message));
    });
}

fn ffi_result<F>(f: F) -> c_int
where
    F: FnOnce() -> Result<c_int, String>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(code)) => code,
        Ok(Err(err)) => {
            set_last_error(err);
            DARTUNIFRAC_ERROR
        }
        Err(_) => {
            set_last_error("panic crossed DartUniFrac C API boundary");
            DARTUNIFRAC_ERROR
        }
    }
}

unsafe fn optional_string(ptr: *const c_char, field: &str) -> Result<Option<String>, String> {
    if ptr.is_null() {
        return Ok(None);
    }

    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str()
        .map(|s| Some(s.to_owned()))
        .map_err(|err| format!("{field} must be valid UTF-8: {err}"))
}

unsafe fn required_string(ptr: *const c_char, field: &str) -> Result<String, String> {
    unsafe { optional_string(ptr, field) }?.ok_or_else(|| format!("{field} is required"))
}

unsafe fn options_from_config(
    config: *const DartUniFracConfig,
) -> Result<DartUniFracOptions, String> {
    let cfg = unsafe { config.as_ref() }.ok_or_else(|| "config pointer is null".to_owned())?;
    let tree_file = unsafe { required_string(cfg.tree_path, "tree_path") }?;
    let input_tsv = unsafe { optional_string(cfg.input_tsv_path, "input_tsv_path") }?;
    let biom_h5 = unsafe { optional_string(cfg.biom_path, "biom_path") }?;
    let output_file = unsafe { optional_string(cfg.output_path, "output_path") }?
        .unwrap_or_else(|| "unifrac.tsv".to_owned());
    let method =
        unsafe { optional_string(cfg.method, "method") }?.unwrap_or_else(|| "dmh".to_owned());

    Ok(DartUniFracOptions {
        tree_file,
        input_tsv,
        biom_h5,
        output_file,
        sketch_size: if cfg.sketch_size == 0 {
            2048
        } else {
            cfg.sketch_size
        },
        method,
        bbits: if cfg.bbits == 0 { 16 } else { cfg.bbits },
        ers_length: if cfg.ers_length == 0 {
            2048
        } else {
            cfg.ers_length
        },
        seed: cfg.seed,
        weighted: cfg.weighted != 0,
        raw_counts: cfg.raw_counts != 0,
        succ: cfg.succ != 0,
        compress: cfg.compress != 0,
        pcoa: cfg.pcoa != 0,
        streaming: cfg.streaming != 0,
        streaming_block_size: (cfg.block_rows != 0).then_some(cfg.block_rows),
        threads: (cfg.threads != 0).then_some(cfg.threads),
    })
}

fn matrix_to_handle(matrix: DartUniFracDistanceMatrix) -> Result<*mut DartUniFracMatrix, String> {
    let sample_names: Vec<CString> = matrix
        .sample_names
        .into_iter()
        .map(|name| {
            CString::new(name).map_err(|_| "sample name contains an interior NUL byte".to_owned())
        })
        .collect::<Result<_, _>>()?;
    let sample_name_ptrs = sample_names.iter().map(|name| name.as_ptr()).collect();
    let handle = Box::new(MatrixHandle {
        sample_names,
        sample_name_ptrs,
        distances: matrix.distances,
    });
    Ok(Box::into_raw(handle) as *mut DartUniFracMatrix)
}

unsafe fn matrix_ref<'a>(matrix: *const DartUniFracMatrix) -> Option<&'a MatrixHandle> {
    unsafe { (matrix as *const MatrixHandle).as_ref() }
}

#[unsafe(no_mangle)]
pub extern "C" fn dartunifrac_config_default() -> DartUniFracConfig {
    DartUniFracConfig {
        tree_path: ptr::null(),
        input_tsv_path: ptr::null(),
        biom_path: ptr::null(),
        output_path: ptr::null(),
        method: c"dmh".as_ptr(),
        sketch_size: 2048,
        ers_length: 2048,
        seed: 1337,
        bbits: 16,
        weighted: 0,
        raw_counts: 0,
        succ: 0,
        compress: 0,
        pcoa: 0,
        streaming: 0,
        block_rows: 0,
        threads: 0,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dartunifrac_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

#[unsafe(no_mangle)]
pub extern "C" fn dartunifrac_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|msg| msg.as_ptr())
            .unwrap_or(ptr::null())
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn dartunifrac_status_message(code: c_int) -> *const c_char {
    match code {
        DARTUNIFRAC_OK => c"ok".as_ptr(),
        DARTUNIFRAC_NULL_POINTER => c"null pointer".as_ptr(),
        _ => c"error".as_ptr(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dartunifrac_run(config: *const DartUniFracConfig) -> c_int {
    ffi_result(|| {
        let options = unsafe { options_from_config(config) }?;
        init_logging_from_env();
        run_dartunifrac(&options).map_err(|err| err.to_string())?;
        Ok(DARTUNIFRAC_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dartunifrac_compute_matrix(
    config: *const DartUniFracConfig,
    out_matrix: *mut *mut DartUniFracMatrix,
) -> c_int {
    ffi_result(|| {
        if out_matrix.is_null() {
            set_last_error("out_matrix pointer is null");
            return Ok(DARTUNIFRAC_NULL_POINTER);
        }
        let options = unsafe { options_from_config(config) }?;
        init_logging_from_env();
        let matrix = compute_dartunifrac_matrix(&options).map_err(|err| err.to_string())?;
        let handle = matrix_to_handle(matrix)?;
        unsafe {
            *out_matrix = handle;
        }
        Ok(DARTUNIFRAC_OK)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dartunifrac_matrix_sample_count(
    matrix: *const DartUniFracMatrix,
) -> usize {
    unsafe { matrix_ref(matrix) }
        .map(|handle| handle.sample_names.len())
        .unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dartunifrac_matrix_sample_name(
    matrix: *const DartUniFracMatrix,
    index: usize,
) -> *const c_char {
    unsafe { matrix_ref(matrix) }
        .and_then(|handle| handle.sample_name_ptrs.get(index).copied())
        .unwrap_or(ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dartunifrac_matrix_sample_names(
    matrix: *const DartUniFracMatrix,
) -> *const *const c_char {
    unsafe { matrix_ref(matrix) }
        .map(|handle| handle.sample_name_ptrs.as_ptr())
        .unwrap_or(ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dartunifrac_matrix_distances(
    matrix: *const DartUniFracMatrix,
) -> *const f32 {
    unsafe { matrix_ref(matrix) }
        .map(|handle| handle.distances.as_ptr())
        .unwrap_or(ptr::null())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn dartunifrac_free_matrix(matrix: *mut DartUniFracMatrix) {
    if !matrix.is_null() {
        unsafe {
            drop(Box::from_raw(matrix as *mut MatrixHandle));
        }
    }
}
