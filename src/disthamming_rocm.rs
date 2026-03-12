//! ROCm/ROCm-rs backend for pairwise Hamming distances on AMD GPUs (e.g. MI300A).
//!
//! Kernels are simple row-wise Hamming:
//!   - One kernel launch computes distances d(i, j) for fixed row i and all j.
//!   - We provide kernels for u16 / u32 / u64 sketches.
//!
//! Host side:
//!   - In-memory pairwise: multi-GPU, row-parallel, then CPU symmetrization.
//!   - Streaming writer: multi-GPU, each GPU computes subsets of rows and
//!     sends finished lines to a writer thread via a channel.
//!
//! Public API (mirrors CUDA backend):
//!   - device_count()
//!   - pairwise_hamming_multi_gpu_u16/u32/u64
//!   - write_matrix_streaming_gpu_auto_u16/u32/u64
//!

use anyhow::{bail, Context, Result};
use log::info;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::{atomic::AtomicUsize, atomic::Ordering, Arc};
use std::time::Instant;
use std::thread;

// ROCm / HIP host-side API
use rocm_rs::hip::*;
use rocm_rs::hip::kernel::AsKernelArg;

// Low-level HIP FFI for device count & error code
use rocm_rs::rocsparse::ffi::hipGetDeviceCount;
// Rust GPU kernels via rocm_kernel_macros
use rocm_kernel_macros::{amdgpu_global, amdgpu_kernel_finalize, amdgpu_kernel_init};

// Initialize kernel compilation context.
amdgpu_kernel_init!();

/// Row-wise Hamming kernel for u16 sketches.
///
/// Arguments:
/// - `sketches`: [n*k] u16, row-major
/// - `indices`:  [n] u32  (0..n-1), used only to get work-item id via read_by_workitem_id_x
/// - `out`:      [n] f32, distances d(i, j) for fixed i and all j
/// - `cfg`:      [3] u32: [n, k, i]
// Row-wise Hamming kernel for u16 sketches.
// One workitem computes distance from fixed row_i to column j = indices[workitem_x].
#[amdgpu_global]
fn hamming_row_kernel_u16(
    sketches: *const u16,
    indices: *const u32,
    out: *mut f32,
    cfg: *const u32,
) {
    // cfg layout: [0] = n, [1] = k, [2] = row_i
    let n = unsafe { *cfg.add(0) as usize };
    let k = unsafe { *cfg.add(1) as usize };
    let row_i = unsafe { *cfg.add(2) as usize };

    // Provided by rocm_rs::hip::kernel
    let j_u32 = read_by_workitem_id_x(indices);
    let j = j_u32 as usize;
    if j >= n {
        return;
    }

    let base_i = row_i * k;
    let base_j = j * k;

    let mut diff: u32 = 0;
    let mut t = 0usize;
    while t < k {
        unsafe {
            if *sketches.add(base_i + t) != *sketches.add(base_j + t) {
                diff += 1;
            }
        }
        t += 1;
    }

    let d = if row_i == j {
        0.0f32
    } else {
        diff as f32 / k as f32
    };

    write_by_workitem_id_x(out, d);
}

// Row-wise Hamming kernel for u32 sketches.
#[amdgpu_global]
fn hamming_row_kernel_u32(
    sketches: *const u32,
    indices: *const u32,
    out: *mut f32,
    cfg: *const u32,
) {
    let n = unsafe { *cfg.add(0) as usize };
    let k = unsafe { *cfg.add(1) as usize };
    let row_i = unsafe { *cfg.add(2) as usize };

    let j_u32 = read_by_workitem_id_x(indices);
    let j = j_u32 as usize;
    if j >= n {
        return;
    }

    let base_i = row_i * k;
    let base_j = j * k;

    let mut diff: u32 = 0;
    let mut t = 0usize;
    while t < k {
        unsafe {
            if *sketches.add(base_i + t) != *sketches.add(base_j + t) {
                diff += 1;
            }
        }
        t += 1;
    }

    let d = if row_i == j {
        0.0f32
    } else {
        diff as f32 / k as f32
    };

    write_by_workitem_id_x(out, d);
}

// Row-wise Hamming kernel for u64 sketches.
#[amdgpu_global]
fn hamming_row_kernel_u64(
    sketches: *const u64,
    indices: *const u32,
    out: *mut f32,
    cfg: *const u32,
) {
    let n = unsafe { *cfg.add(0) as usize };
    let k = unsafe { *cfg.add(1) as usize };
    let row_i = unsafe { *cfg.add(2) as usize };

    let j_u32 = read_by_workitem_id_x(indices);
    let j = j_u32 as usize;
    if j >= n {
        return;
    }

    let base_i = row_i * k;
    let base_j = j * k;

    let mut diff: u32 = 0;
    let mut t = 0usize;
    while t < k {
        unsafe {
            if *sketches.add(base_i + t) != *sketches.add(base_j + t) {
                diff += 1;
            }
        }
        t += 1;
    }

    let d = if row_i == j {
        0.0f32
    } else {
        diff as f32 / k as f32
    };

    write_by_workitem_id_x(out, d);
}

// Finalize kernel compilation and get path to AMDGPU binary (HSACO).
const AMDGPU_KERNEL_BINARY_PATH: &str = amdgpu_kernel_finalize!();

#[inline]
fn mib(x: usize) -> f64 {
    (x as f64) / (1024.0 * 1024.0)
}
#[inline]
fn gib(x: usize) -> f64 {
    (x as f64) / (1024.0 * 1024.0 * 1024.0)
}

/// Wrapper for raw output pointer so we can mark it Send/Sync.
///
/// Safety: we guarantee disjoint row writes via `next_row` atomics.
#[derive(Clone, Copy)]
struct OutPtr(*mut f32);
unsafe impl Send for OutPtr {}
unsafe impl Sync for OutPtr {}

/// How many HIP devices (GPUs) are visible.
pub fn device_count() -> Result<usize> {
    unsafe {
        let mut n = 0i32;
        let err = hipGetDeviceCount(&mut n as *mut i32);

        // HIP success is always error code 0
        if err != 0 {
            bail!(
                "hipGetDeviceCount failed: HIP error code = {} (did you load the ROCm module and set LD_LIBRARY_PATH?)",
                err
            );
        }

        Ok(n.max(0) as usize)
    }
}

/// Internal: per-device row worker for in-memory pairwise Hamming.
///
/// Each worker:
/// - Loads module, function.
/// - Uploads sketches + indices once.
/// - Repeatedly grabs the next row index `i` from `next_row`, runs the kernel,
///   copies the row back, and writes into `out_row_major[i, j]` for all j.
///
/// NOTE: This only writes the [i, j] entries, not [j, i]. Symmetrization and
/// optional weighted normalization are done afterwards on CPU.
fn pairwise_hamming_worker<T: Copy + Send + Sync + 'static>(
    device_id: i32,
    kernel_name: &'static str,
    sketches: Arc<Vec<T>>,
    n: usize,
    k: usize,
    out_ptr: OutPtr,
    next_row: Arc<AtomicUsize>,
) -> Result<()> {
    let device = Device::new(device_id)?;
    device.set_current()?;

    let kernel_path = PathBuf::from(AMDGPU_KERNEL_BINARY_PATH);
    let module = Module::load(kernel_path)?;
    let function = module
        .get_function(kernel_name)
        .with_context(|| format!("load ROCm kernel '{}'", kernel_name))?;

    // Upload sketches once
    let mut d_sketches: DeviceMemory<T> = DeviceMemory::new(n * k)?;
    d_sketches.copy_from_host(&sketches[..])?;

    // Indices buffer: 0..n-1, used to map work_item_id_x -> column index j
    let indices_host: Vec<u32> = (0..n as u32).collect();
    let mut d_indices: DeviceMemory<u32> = DeviceMemory::new(n)?;
    d_indices.copy_from_host(&indices_host)?;

    // Config buffer [n, k, row_i]
    let mut d_cfg: DeviceMemory<u32> = DeviceMemory::new(3)?;

    // Output buffer: one row of distances
    let d_row: DeviceMemory<f32> = DeviceMemory::new(n)?;
    let mut row_host = vec![0.0f32; n];

    let block_x: u32 = 256.min(n as u32).max(1);
    let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
    let block_dim = Dim3 {
        x: block_x,
        y: 1,
        z: 1,
    };
    let grid_dim = Dim3 {
        x: grid_x,
        y: 1,
        z: 1,
    };

    info!(
        "ROCm worker dev {}: kernel={} grid=({},1,1) block=({},1,1) row_scratch≈{:.2} MiB",
        device_id,
        kernel_name,
        grid_x,
        block_x,
        mib(n * std::mem::size_of::<f32>()),
    );

    loop {
        let i = next_row.fetch_add(1, Ordering::Relaxed);
        if i >= n {
            break;
        }

        let cfg_host = [n as u32, k as u32, i as u32];
        d_cfg.copy_from_host(&cfg_host)?;

        let mut kernel_args = [
            d_sketches.as_kernel_arg(),
            d_indices.as_kernel_arg(),
            d_row.as_kernel_arg(),
            d_cfg.as_kernel_arg(),
        ];

        unsafe {
            function.launch(
                grid_dim,
                block_dim,
                0,    // shared_mem_bytes
                None, // default stream
                &mut kernel_args,
            )?;
        }

        d_row.copy_to_host(&mut row_host)?;

        // Write row into output matrix: out[i, j] for all j
        unsafe {
            let out_row = out_ptr.0.add(i * n);
            std::ptr::copy_nonoverlapping(row_host.as_ptr(), out_row, n);
        }
    }

    Ok(())
}

fn pairwise_hamming_multi_gpu_impl<T: Copy + Send + Sync + 'static>(
    sketches: &[T],
    n: usize,
    k: usize,
    out: &mut [f32],
    _block_rows: usize, // not used in this implementation
    weighted_normalized: bool,
    kernel_name: &'static str,
) -> Result<()> {
    if sketches.len() != n * k {
        bail!(
            "sketches length mismatch: got {}, expected {}",
            sketches.len(),
            n * k
        );
    }
    if out.len() != n * n {
        bail!(
            "out length mismatch: got {}, expected {}",
            out.len(),
            n * n
        );
    }

    let ng = device_count().unwrap_or(0);
    if ng == 0 {
        bail!("ROCm backend: no HIP devices available");
    }

    let sketches_arc = Arc::new(sketches.to_vec());
    let next_row = Arc::new(AtomicUsize::new(0));
    let out_ptr = OutPtr(out.as_mut_ptr());

    info!(
        "ROCm pairwise_hamming_multi_gpu_impl: n={} k={} devices={} sketches={:.2} GiB out={:.2} GiB",
        n,
        k,
        ng,
        gib(n * k * std::mem::size_of::<T>()),
        gib(n * n * std::mem::size_of::<f32>()),
    );

    // Multi-GPU row-parallel execution.
    thread::scope(|scope| {
        for dev_id in 0..(ng as i32) {
            let sketches_arc = Arc::clone(&sketches_arc);
            let next_row = Arc::clone(&next_row);
            let out_ptr = out_ptr;

            scope.spawn(move || {
                let inner = || -> Result<()> {
                    pairwise_hamming_worker::<T>(
                        dev_id,
                        kernel_name,
                        sketches_arc,
                        n,
                        k,
                        out_ptr,
                        next_row,
                    )
                };
                if let Err(e) = inner() {
                    panic!("ROCm worker device {} failed: {e:?}", dev_id);
                }
            });
        }
    });

    // CPU symmetrization + optional weighted normalization
    for i in 0..n {
        out[i * n + i] = 0.0;
        for j in (i + 1)..n {
            let mut d = out[i * n + j];
            if weighted_normalized {
                d = if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 };
            }
            out[i * n + j] = d;
            out[j * n + i] = d;
        }
    }

    Ok(())
}

pub fn pairwise_hamming_multi_gpu_u16(
    sketches: &[u16],
    n: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    weighted: bool,
) -> Result<()> {
    pairwise_hamming_multi_gpu_impl::<u16>(
        sketches,
        n,
        k,
        out,
        block_rows,
        weighted,
        "hamming_row_kernel_u16",
    )
}

pub fn pairwise_hamming_multi_gpu_u32(
    sketches: &[u32],
    n: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    weighted: bool,
) -> Result<()> {
    pairwise_hamming_multi_gpu_impl::<u32>(
        sketches,
        n,
        k,
        out,
        block_rows,
        weighted,
        "hamming_row_kernel_u32",
    )
}

pub fn pairwise_hamming_multi_gpu_u64(
    sketches: &[u64],
    n: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    weighted: bool,
) -> Result<()> {
    pairwise_hamming_multi_gpu_impl::<u64>(
        sketches,
        n,
        k,
        out,
        block_rows,
        weighted,
        "hamming_row_kernel_u64",
    )
}

fn write_matrix_streaming_gpu_single<T: Copy>(
    names: &[String],
    sketches_flat: &[T],
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted_normalized: bool,
    device_id: i32,
    kernel_name: &'static str,
) -> Result<()> {
    if sketches_flat.len() != n * k {
        bail!(
            "sketches_flat length mismatch: got {}, expected {}",
            sketches_flat.len(),
            n * k
        );
    }

    let device = Device::new(device_id)?;
    device.set_current()?;

    let kernel_path = PathBuf::from(AMDGPU_KERNEL_BINARY_PATH);
    let module = Module::load(kernel_path)?;
    let function = module
        .get_function(kernel_name)
        .with_context(|| format!("load ROCm kernel '{}'", kernel_name))?;

    // Upload sketches & indices once
    let mut d_sketches: DeviceMemory<T> = DeviceMemory::new(n * k)?;
    d_sketches.copy_from_host(sketches_flat)?;

    let indices_host: Vec<u32> = (0..n as u32).collect();
    let mut d_indices: DeviceMemory<u32> = DeviceMemory::new(n)?;
    d_indices.copy_from_host(&indices_host)?;

    let mut d_cfg: DeviceMemory<u32> = DeviceMemory::new(3)?;
    let d_row: DeviceMemory<f32> = DeviceMemory::new(n)?;
    let mut row_host = vec![0.0f32; n];

    let block_x: u32 = 256.min(n as u32).max(1);
    let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
    let block_dim = Dim3 {
        x: block_x,
        y: 1,
        z: 1,
    };
    let grid_dim = Dim3 {
        x: grid_x,
        y: 1,
        z: 1,
    };

    info!(
        "ROCm streaming(single): dev={} n={} k={} row_scratch≈{:.2} MiB kernel={}",
        device_id,
        n,
        k,
        mib(n * std::mem::size_of::<f32>()),
        kernel_name
    );

    // Writer (optional zstd)
    let mut writer: Box<dyn std::io::Write> = if compress {
        let file = std::fs::File::create(path)?;
        let mut enc = zstd::Encoder::new(file, 0)?;
        let zstd_threads = rayon::current_num_threads() as u32;
        if zstd_threads > 1 {
            enc.multithread(zstd_threads)?;
        }
        Box::new(std::io::BufWriter::with_capacity(16 << 20, enc.auto_finish()))
    } else {
        Box::new(std::io::BufWriter::with_capacity(
            16 << 20,
            std::fs::File::create(path)?,
        ))
    };

    // Header
    writer.write_all(b"")?;
    for name in names {
        writer.write_all(b"\t")?;
        writer.write_all(name.as_bytes())?;
    }
    writer.write_all(b"\n")?;

    let mut fmt = ryu::Buffer::new();

    for i in 0..n {
        let cfg_host = [n as u32, k as u32, i as u32];
        d_cfg.copy_from_host(&cfg_host)?;

        let mut kernel_args = [
            d_sketches.as_kernel_arg(),
            d_indices.as_kernel_arg(),
            d_row.as_kernel_arg(),
            d_cfg.as_kernel_arg(),
        ];

        let block_start = Instant::now();

        unsafe {
            function.launch(
                grid_dim,
                block_dim,
                0,    // shared_mem_bytes
                None, // default stream
                &mut kernel_args,
            )?;
        }

        d_row.copy_to_host(&mut row_host)?;

        let compute_ms = block_start.elapsed().as_millis();
        info!(
            "ROCm streaming(single): row {} compute done in {} ms",
            i, compute_ms
        );

        // Build line: <name_i> \t d(i,0) \t d(i,1) ...
        let mut line = String::with_capacity(64 + 12 * n);
        line.push_str(&names[i]);

        for j in 0..n {
            let mut d = if i == j { 0.0f32 } else { row_host[j] };
            if weighted_normalized {
                d = if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 };
            }
            line.push('\t');
            line.push_str(fmt.format_finite(d));
        }

        writer.write_all(line.as_bytes())?;
        writer.write_all(b"\n")?;
    }

    writer.flush()?;
    Ok(())
}

fn write_matrix_streaming_gpu_multi_impl<T: Copy + Send + Sync + 'static>(
    names: &[String],
    sketches_flat: &[T],
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted: bool,
    tile_cols: usize,
    tile_rows: usize,
    kernel_name: &'static str,
    devices: &[usize],
) -> Result<()> {
    if sketches_flat.len() != n * k {
        bail!(
            "sketches_flat length mismatch: got {}, expected {}",
            sketches_flat.len(),
            n * k
        );
    }

    let ng = devices.len().max(1);

    let names_arc = Arc::new(names.to_vec());
    let sketches_arc: Arc<Vec<T>> = Arc::new(sketches_flat.to_vec());

    // Channel: workers send (row_index, line) to writer
    let (tx, rx) = mpsc::sync_channel::<(usize, String)>(ng * 2);

    let path_str = path.to_string();
    let names_for_writer = Arc::clone(&names_arc);
    let writer_handle = thread::spawn(move || -> Result<()> {
        let mut writer: Box<dyn std::io::Write> = if compress {
            let file = std::fs::File::create(&path_str)?;
            let mut enc = zstd::Encoder::new(file, 0)?;
            let zstd_threads = rayon::current_num_threads() as u32;
            if zstd_threads > 1 {
                enc.multithread(zstd_threads)?;
            }
            Box::new(std::io::BufWriter::with_capacity(16 << 20, enc.auto_finish()))
        } else {
            Box::new(std::io::BufWriter::with_capacity(
                16 << 20,
                std::fs::File::create(&path_str)?,
            ))
        };

        // Header
        writer.write_all(b"")?;
        for name in &*names_for_writer {
            writer.write_all(b"\t")?;
            writer.write_all(name.as_bytes())?;
        }
        writer.write_all(b"\n")?;

        let t_writer_start = Instant::now();

        let mut next = 0usize;
        let mut stash: BTreeMap<usize, String> = BTreeMap::new();
        while next < names_for_writer.len() {
            let (i, line) = rx.recv().expect("worker channel closed unexpectedly");
            stash.insert(i, line);
            while let Some(line) = stash.remove(&next) {
                writer.write_all(line.as_bytes())?;
                next += 1;
            }
        }
        writer.flush()?;

        let writer_ms = t_writer_start.elapsed().as_millis();
        info!(
            "ROCm streaming(multi): writer finished all {} rows in {} ms",
            names_for_writer.len(),
            writer_ms
        );
        Ok(())
    });

    // GPU workers
    thread::scope(|scope| {
        for (widx, &dev_usize) in devices.iter().enumerate() {
            let tx = tx.clone();
            let names = Arc::clone(&names_arc);
            let sketches = Arc::clone(&sketches_arc);
            let devs = devices.len().max(1);
            let device_id = dev_usize as i32;

            scope.spawn(move || {
                let inner = || -> Result<()> {
                    let device = Device::new(device_id)?;
                    device.set_current()?;

                    let kernel_path = PathBuf::from(AMDGPU_KERNEL_BINARY_PATH);
                    let module = Module::load(kernel_path)?;
                    let function = module
                        .get_function(kernel_name)
                        .with_context(|| format!("load ROCm kernel '{}'", kernel_name))?;

                    let mut d_sketches: DeviceMemory<T> = DeviceMemory::new(n * k)?;
                    d_sketches.copy_from_host(&sketches[..])?;

                    let indices_host: Vec<u32> = (0..n as u32).collect();
                    let mut d_indices: DeviceMemory<u32> = DeviceMemory::new(n)?;
                    d_indices.copy_from_host(&indices_host)?;

                    let mut d_cfg: DeviceMemory<u32> = DeviceMemory::new(3)?;
                    let d_row: DeviceMemory<f32> = DeviceMemory::new(n)?;
                    let mut row_host = vec![0.0f32; n];

                    let block_x: u32 = 256.min(n as u32).max(1);
                    let grid_x: u32 = ((n as u32) + block_x - 1) / block_x;
                    let block_dim = Dim3 {
                        x: block_x,
                        y: 1,
                        z: 1,
                    };
                    let grid_dim = Dim3 {
                        x: grid_x,
                        y: 1,
                        z: 1,
                    };

                    let mut fmt = ryu::Buffer::new();

                    info!(
                        "ROCm streaming(multi): dev {} starting, n={} k={} tile_rows={} tile_cols={} row_scratch≈{:.2} MiB kernel={}",
                        device_id,
                        n,
                        k,
                        tile_rows,
                        tile_cols,
                        mib(n * std::mem::size_of::<f32>()),
                        kernel_name
                    );

                    let mut i0 = widx * tile_rows.max(1);
                    while i0 < n {
                        let bw = (n - i0).min(tile_rows.max(1));

                        let block_start = Instant::now();
                        info!(
                            "ROCm streaming(multi): dev {} row-block i0={}..{} (bw={}) compute start",
                            device_id,
                            i0,
                            i0 + bw,
                            bw
                        );

                        for local in 0..bw {
                            let i = i0 + local;
                            if i >= n {
                                break;
                            }

                            let cfg_host = [n as u32, k as u32, i as u32];
                            d_cfg.copy_from_host(&cfg_host)?;

                            let mut kernel_args = [
                                d_sketches.as_kernel_arg(),
                                d_indices.as_kernel_arg(),
                                d_row.as_kernel_arg(),
                                d_cfg.as_kernel_arg(),
                            ];

                            unsafe {
                                function.launch(
                                    grid_dim,
                                    block_dim,
                                    0,    // shared_mem_bytes
                                    None, // default stream
                                    &mut kernel_args,
                                )?;
                            }

                            d_row.copy_to_host(&mut row_host)?;

                            // Build line for row i
                            let mut line = String::with_capacity(64 + 12 * n);
                            line.push_str(&names[i]);

                            for j in 0..n {
                                let mut d = if i == j { 0.0f32 } else { row_host[j] };
                                if weighted {
                                    d = if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 };
                                }
                                line.push('\t');
                                line.push_str(fmt.format_finite(d));
                            }
                            line.push('\n');
                            tx.send((i, line)).expect("send row to writer");
                        }

                        let compute_ms = block_start.elapsed().as_millis();
                        info!(
                            "ROCm streaming(multi): dev {} row-block i0={}..{} compute+enqueue done in {} ms",
                            device_id,
                            i0,
                            i0 + bw,
                            compute_ms
                        );

                        i0 = i0.saturating_add(tile_rows.max(1) * devs);
                    }

                    Ok(())
                };

                if let Err(e) = inner() {
                    panic!("ROCm multi-GPU streaming worker on device {device_id} failed: {e:?}");
                }
            });
        }
    });

    drop(tx);
    writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("ROCm writer thread panicked"))??;

    Ok(())
}

// ----- AUTO STREAMING ENTRYPOINTS (now mirror CUDA logic) -------------------

pub fn write_matrix_streaming_gpu_auto_u16(
    names: &[String],
    sketches: &[u16],
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted: bool,
    tile_cols: usize,
    tile_rows: usize,
) -> Result<()> {
    let ng = device_count().unwrap_or(0);
    if ng == 0 {
        bail!("ROCm backend: no HIP devices available for streaming");
    } else if ng == 1 {
        // Single-GPU: use the simpler single-streaming path (no writer thread).
        write_matrix_streaming_gpu_single::<u16>(
            names,
            sketches,
            n,
            k,
            path,
            compress,
            weighted,
            0, // device 0
            "hamming_row_kernel_u16",
        )
    } else {
        // Multi-GPU streaming with writer thread and work sharing, as in CUDA backend.
        let devices: Vec<usize> = (0..ng).collect();
        write_matrix_streaming_gpu_multi_impl::<u16>(
            names,
            sketches,
            n,
            k,
            path,
            compress,
            weighted,
            tile_cols,
            tile_rows,
            "hamming_row_kernel_u16",
            &devices,
        )
    }
}

pub fn write_matrix_streaming_gpu_auto_u32(
    names: &[String],
    sketches: &[u32],
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted: bool,
    tile_cols: usize,
    tile_rows: usize,
) -> Result<()> {
    let ng = device_count().unwrap_or(0);
    if ng == 0 {
        bail!("ROCm backend: no HIP devices available for streaming");
    } else if ng == 1 {
        write_matrix_streaming_gpu_single::<u32>(
            names,
            sketches,
            n,
            k,
            path,
            compress,
            weighted,
            0,
            "hamming_row_kernel_u32",
        )
    } else {
        let devices: Vec<usize> = (0..ng).collect();
        write_matrix_streaming_gpu_multi_impl::<u32>(
            names,
            sketches,
            n,
            k,
            path,
            compress,
            weighted,
            tile_cols,
            tile_rows,
            "hamming_row_kernel_u32",
            &devices,
        )
    }
}

pub fn write_matrix_streaming_gpu_auto_u64(
    names: &[String],
    sketches: &[u64],
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted: bool,
    tile_cols: usize,
    tile_rows: usize,
) -> Result<()> {
    let ng = device_count().unwrap_or(0);
    if ng == 0 {
        bail!("ROCm backend: no HIP devices available for streaming");
    } else if ng == 1 {
        write_matrix_streaming_gpu_single::<u64>(
            names,
            sketches,
            n,
            k,
            path,
            compress,
            weighted,
            0,
            "hamming_row_kernel_u64",
        )
    } else {
        let devices: Vec<usize> = (0..ng).collect();
        write_matrix_streaming_gpu_multi_impl::<u64>(
            names,
            sketches,
            n,
            k,
            path,
            compress,
            weighted,
            tile_cols,
            tile_rows,
            "hamming_row_kernel_u64",
            &devices,
        )
    }
}