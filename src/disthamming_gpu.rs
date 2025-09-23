//! CUDA: in-memory pairwise (single/multi GPU) hamming & automatic streaming writer.

use anyhow::{Context, Result, bail};
use std::{sync::Arc, thread};

use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use log::debug;
use log::info;
use log::warn;

/// Kernel: computes a (bw×bh) tile of normalized Hamming distances
/// sketches: [n*k] row-major u64 IDs
/// Kernel: shared-memory–tiled Hamming for u64 IDs
const KERNEL_SRC: &str = r#"
#ifndef BK
#define BK 64   // k-slab per iteration (tune: 32, 64, 128)
#endif

extern "C" __global__
void hamming_tile_u64(
    const unsigned long long* __restrict__ sketches, // [n*k], row-major
    int n, int k,
    int i0, int j0,
    int bw, int bh,
    double* __restrict__ out, // [bw*bh], row-major, ldo = bh
    int only_upper // 1 => only write j>i, 0 => write full tile
){
    // Thread's local coords within the logical tile
    const int jj = blockIdx.x * blockDim.x + threadIdx.x; // 0..bh-1
    const int ii = blockIdx.y * blockDim.y + threadIdx.y; // 0..bw-1

    const int i = i0 + ii;
    const int j = j0 + jj;

    // Dynamic shared memory layout:
    //   As: blockDim.y rows × BK
    //   Bs: blockDim.x cols × BK
    extern __shared__ unsigned long long smem[];
    unsigned long long* As = smem;
    unsigned long long* Bs = As + (size_t)blockDim.y * (size_t)BK;

    unsigned int diff = 0u;

    // Flattened thread id in the block for cooperative loads
    const int tpb = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int t0 = 0; t0 < k; t0 += BK) {
        const int bk = min(BK, k - t0);

        // --- Load A-slab: (blockDim.y × bk) ---
        const int totalA = blockDim.y * bk;
        for (int idx = tid; idx < totalA; idx += tpb) {
            const int r   = idx / bk;   // local row in [0, blockDim.y)
            const int t   = idx - r*bk; // 0..bk-1
            const int gi  = i0 + r;
            unsigned long long val = 0ULL;
            if (r < bw && (t0 + t) < k && gi < n) {
                val = sketches[(size_t)gi * (size_t)k + (size_t)(t0 + t)];
            }
            As[(size_t)r * (size_t)BK + (size_t)t] = val;
        }

        // --- Load B-slab: (blockDim.x × bk) ---
        const int totalB = blockDim.x * bk;
        for (int idx = tid; idx < totalB; idx += tpb) {
            const int c   = idx / bk;   // local col in [0, blockDim.x)
            const int t   = idx - c*bk; // 0..bk-1
            const int gj  = j0 + c;
            unsigned long long val = 0ULL;
            if (c < bh && (t0 + t) < k && gj < n) {
                val = sketches[(size_t)gj * (size_t)k + (size_t)(t0 + t)];
            }
            Bs[(size_t)c * (size_t)BK + (size_t)t] = val;
        }

        __syncthreads();

        // --- Consume this k-slab for our (ii,jj) if valid ---
        if (ii < bw && jj < bh) {
            if (!(i == j || (only_upper && j <= i))) {
                // rows map along blockDim.y; cols along blockDim.x
                const size_t arow = (size_t)threadIdx.y * (size_t)BK;
                const size_t brow = (size_t)threadIdx.x * (size_t)BK;
                #pragma unroll
                for (int t = 0; t < bk; ++t) {
                    diff += (As[arow + (size_t)t] != Bs[brow + (size_t)t]);
                }
            }
        }

        __syncthreads();
    }

    // Write result (normalize in host if you need weighted transform)
    if (ii < bw && jj < bh) {
        if (!(i == j || (only_upper && j <= i))) {
            out[(size_t)ii * (size_t)bh + (size_t)jj] = (double)diff / (double)k;
        }
    }
}
"#;

/// How many CUDA devices are visible.
pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

// In-memory GPU computation

#[inline]
fn mib(x: usize) -> f64 {
    (x as f64) / (1024.0 * 1024.0)
}
#[inline]
fn gib(x: usize) -> f64 {
    (x as f64) / (1024.0 * 1024.0 * 1024.0)
}

/// Single-GPU, produces full n×n matrix in host CPU memory.
pub fn pairwise_hamming_single_gpu(
    sketches_flat_u64: &[u64],
    n: usize,
    k: usize,
    out_upper_tri: &mut [f64],
    mut block_rows: usize,
    weighted_normalized: bool,
) -> Result<()> {
    // Sanity
    if sketches_flat_u64.len() != n * k {
        bail!(
            "sketches_flat_u64 length mismatch: got {}, expected {}",
            sketches_flat_u64.len(),
            n * k
        );
    }
    if out_upper_tri.len() != n * n {
        bail!(
            "out_upper_tri length mismatch: got {}, expected {}",
            out_upper_tri.len(),
            n * n
        );
    }

    // Optional safety cap so a too-large block_rows doesn’t explode memory on a “tight” node
    let cap = 4096usize.min(n);
    if block_rows > cap {
        warn!(
            "single-GPU: capping block_rows from {} → {} for stability",
            block_rows, cap
        );
        block_rows = cap;
    }

    info!(
        "single-GPU: n={} k={} block_rows={} sketches={} GiB host_out={} GiB",
        n,
        k,
        block_rows,
        gib(n * k * 8),
        gib(n * n * 8),
    );

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(KERNEL_SRC)?;
    let module = ctx.load_module(ptx)?;
    let func = module
        .load_function("hamming_tile_u64")
        .context("load function 'hamming_tile_u64'")?;

    // Upload sketches
    let d_sketches: CudaSlice<u64> = stream.memcpy_stod(sketches_flat_u64)?;
    info!("single-GPU: uploaded sketches: {} MiB", mib(n * k * 8));

    // Reusable scratch (block_rows × block_rows)
    let max_t = block_rows.min(n);
    let scratch_elems = max_t * max_t;
    let mut d_tile: CudaSlice<f64> = stream
        .alloc_zeros(scratch_elems)
        .with_context(|| format!("alloc d_tile: {} MiB", mib(scratch_elems * 8)))?;
    let mut h_tile = vec![0.0f64; scratch_elems];
    info!(
        "single-GPU: scratch allocated: elems={} ({:.2} MiB)",
        scratch_elems,
        mib(scratch_elems * 8)
    );

    // Launch cfg & consts
    let n_i32 = n as i32;
    let k_i32 = k as i32;
    let only_upper_i32 = 1i32;

    let nb = (n + block_rows - 1) / block_rows;
    for bi in 0..nb {
        let i0 = bi * block_rows;
        let i1 = (i0 + block_rows).min(n);
        let bw = i1 - i0;

        for bj in bi..nb {
            let j0 = bj * block_rows;
            let j1 = (j0 + block_rows).min(n);
            let bh = j1 - j0;

            let blk_x = 64usize; // threads along columns (j)
            let blk_y = 8usize;  // threads along rows (i)
            let bk    = 128usize; // must match BK in kernel (see #define above)

            // dynamic shared memory: (BK * (blk_y + blk_x)) * sizeof(u64)
            let smem_bytes = ((bk * (blk_y + blk_x)) * std::mem::size_of::<u64>()) as u32;

            let cfg = LaunchConfig {
                grid_dim: (
                    ((bh + blk_x - 1) / blk_x) as u32,
                    ((bw + blk_y - 1) / blk_y) as u32,
                    1,
                ),
                block_dim: (blk_x as u32, blk_y as u32, 1),
                shared_mem_bytes: smem_bytes,
            };

            let i0_i32 = i0 as i32;
            let j0_i32 = j0 as i32;
            let bw_i32 = bw as i32;
            let bh_i32 = bh as i32;

            debug!(
                "single-GPU: tile bi={} bj={} i0={} j0={} bw={} bh={} (tile {:.2} MiB)",
                bi,
                bj,
                i0,
                j0,
                bw,
                bh,
                mib(bw * bh * 8)
            );

            let mut launch = stream.launch_builder(&func);
            launch.arg(&d_sketches);
            launch.arg(&n_i32);
            launch.arg(&k_i32);
            launch.arg(&i0_i32);
            launch.arg(&j0_i32);
            launch.arg(&bw_i32);
            launch.arg(&bh_i32);
            launch.arg(&mut d_tile);
            launch.arg(&only_upper_i32);
            debug!("launch tile bi={} bj={} …", bi, bj);
            unsafe { launch.launch(cfg) }?;
            debug!("kernel launched");
            stream.synchronize()?; // kernel done
            debug!("sync ok");

            // Copy back only bw*bh
            let need = bw * bh;

            // Safety checks & trace
            debug_assert!(need <= h_tile.len());
            debug!(
                "dtoh: copying full scratch {} elems ({} MiB), need={}",
                h_tile.len(),
                (h_tile.len() * 8) as f64 / (1024.0 * 1024.0),
                need
            );

            // Copy the entire device slice into the full host buffer.
            stream.memcpy_dtoh(&d_tile, &mut h_tile)?;
            debug!("dtoh ok ({} MiB)", (need * 8) as f64 / (1024.0 * 1024.0));
            // Scatter into final matrix (upper + mirror lower)
            let base_ptr = out_upper_tri.as_mut_ptr();
            unsafe {
                for ii in 0..bw {
                    let i = i0 + ii;
                    for jj in 0..bh {
                        let j = j0 + jj;
                        if j <= i {
                            continue;
                        }
                        let mut d = h_tile[ii * bh + jj];
                        if weighted_normalized {
                            d = if d < 2.0 { d / (2.0 - d) } else { 1.0 };
                        }
                        *base_ptr.add(i * n + j) = d;
                        *base_ptr.add(j * n + i) = d;
                    }
                }
            }
        }
    }
    Ok(())
}

/// Multi-GPU in-memory, n*n matrix in host CPU memory.
pub fn pairwise_hamming_multi_gpu(
    sketches_flat_u64: &[u64],
    n: usize,
    k: usize,
    out_upper_tri: &mut [f64],
    block_rows: usize,
    weighted_normalized: bool,
) -> Result<()> {
    if sketches_flat_u64.len() != n * k {
        bail!(
            "sketches_flat_u64 length mismatch: got {}, expected {}",
            sketches_flat_u64.len(),
            n * k
        );
    }
    if out_upper_tri.len() != n * n {
        bail!(
            "out_upper_tri length mismatch: got {}, expected {}",
            out_upper_tri.len(),
            n * n
        );
    }

    let ng = CudaContext::device_count()? as usize;
    if ng == 0 {
        bail!("No CUDA devices available");
    }
    if ng == 1 {
        return pairwise_hamming_single_gpu(
            sketches_flat_u64,
            n,
            k,
            out_upper_tri,
            block_rows,
            weighted_normalized,
        );
    }

    // upper-triangular tiles
    let nb = (n + block_rows - 1) / block_rows;
    let mut tiles = Vec::<(usize, usize)>::new();
    tiles.reserve(nb * nb / 2 + nb);
    for bi in 0..nb {
        for bj in bi..nb {
            tiles.push((bi, bj));
        }
    }

    let tiles = Arc::new(tiles);
    let sk_arc = Arc::new(sketches_flat_u64.to_vec());
    // pass output buffer address as usize (Send) to avoid Send bound on *mut T
    let out_addr: usize = out_upper_tri.as_mut_ptr() as usize;
    let n_arc = n;
    let k_arc = k;
    let br_arc = block_rows;
    let weighted = weighted_normalized;

    thread::scope(|scope| {
        for dev_id in 0..ng {
            let tiles = Arc::clone(&tiles);
            let sk = Arc::clone(&sk_arc);
            scope.spawn(move || {
                let inner = || -> Result<()> {
                    let ctx = CudaContext::new(dev_id)?;
                    let stream = ctx.default_stream();

                    let ptx = compile_ptx(KERNEL_SRC)?;
                    let module = ctx.load_module(ptx)?;
                    let func = module.load_function("hamming_tile_u64")?;

                    // HtoD
                    let d_sketches: CudaSlice<u64> = stream.memcpy_stod(&sk[..])?;

                    let max_t = br_arc.min(n_arc);
                    let mut d_tile: CudaSlice<f64> = stream.alloc_zeros(max_t * max_t)?;
                    let mut h_tile = vec![0.0f64; max_t * max_t];

                    // kernel constants as locals
                    let n_i32 = n_arc as i32;
                    let k_i32 = k_arc as i32;
                    let only_upper_i32 = 1i32;

                    for (tix, &(bi, bj)) in tiles.iter().enumerate() {
                        if tix % ng != dev_id {
                            continue;
                        }

                        let i0 = bi * br_arc;
                        let i1 = (i0 + br_arc).min(n_arc);
                        let bw = i1 - i0;

                        let j0 = bj * br_arc;
                        let j1 = (j0 + br_arc).min(n_arc);
                        let bh = j1 - j0;

                        let blk_x = 64usize; // threads along columns (j)
                        let blk_y = 8usize;  // threads along rows (i)
                        let bk    = 128usize; // must match BK in kernel (see #define above)

                        // dynamic shared memory: (BK * (blk_y + blk_x)) * sizeof(u64)
                        let smem_bytes = ((bk * (blk_y + blk_x)) * std::mem::size_of::<u64>()) as u32;

                        let cfg = LaunchConfig {
                            grid_dim: (
                                ((bh + blk_x - 1) / blk_x) as u32,
                                ((bw + blk_y - 1) / blk_y) as u32,
                                1,
                            ),
                            block_dim: (blk_x as u32, blk_y as u32, 1),
                            shared_mem_bytes: smem_bytes,
                        };

                        let i0_i32 = i0 as i32;
                        let j0_i32 = j0 as i32;
                        let bw_i32 = bw as i32;
                        let bh_i32 = bh as i32;

                        let mut launch = stream.launch_builder(&func);
                        launch.arg(&d_sketches);
                        launch.arg(&n_i32);
                        launch.arg(&k_i32);
                        launch.arg(&i0_i32);
                        launch.arg(&j0_i32);
                        launch.arg(&bw_i32);
                        launch.arg(&bh_i32);
                        launch.arg(&mut d_tile);
                        launch.arg(&only_upper_i32);

                        unsafe { launch.launch(cfg) }?;
                        stream.synchronize()?;

                        // D2H of the used part
                        stream.memcpy_dtoh(&d_tile, &mut h_tile)?;

                        // host write-back (upper + mirror lower)
                        let base_ptr = out_addr as *mut f64;
                        unsafe {
                            for ii in 0..bw {
                                let i = i0 + ii;
                                for jj in 0..bh {
                                    let j = j0 + jj;
                                    if j <= i {
                                        continue;
                                    }
                                    let mut d = h_tile[ii * bh + jj];
                                    if weighted {
                                        d = if d < 2.0 { d / (2.0 - d) } else { 1.0 };
                                    }
                                    *base_ptr.add(i * n_arc + j) = d;
                                    *base_ptr.add(j * n_arc + i) = d;
                                }
                            }
                        }
                    }
                    Ok(())
                };
                if let Err(e) = inner() {
                    panic!("GPU worker {} failed: {e:?}", dev_id);
                }
            });
        }
    });
    Ok(())
}

/// Single-GPU computing with streaming writer: writes full matrix to disk row-by-row (no n×n host memory).
/// Single-GPU compute with streaming writer: writes full matrix to disk row-by-row.
/// Computes many rows per kernel launch to reduce launch/D2H overhead.
fn write_matrix_streaming_gpu_single(
    names: &[String],
    sketches_flat_u64: &[u64], // [n*k], row-major
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted_normalized: bool,
    tile_cols: usize, // e.g. 24576
    gpu_id: usize,    // device index
) -> Result<()> {
    use std::io::Write;

    let ctx = CudaContext::new(gpu_id)?;
    let stream = ctx.default_stream();

    let ptx = compile_ptx(KERNEL_SRC)?;
    let module = ctx.load_module(ptx)?;
    let func = module.load_function("hamming_tile_u64")?;

    // Upload sketches once
    let d_sketches: CudaSlice<u64> = stream.memcpy_stod(sketches_flat_u64)?;

    // Writer (optional zstd)
    let mut writer: Box<dyn std::io::Write> = if compress {
        let file = std::fs::File::create(path)?;
        let mut enc = zstd::Encoder::new(file, 0)?;
        let zstd_threads = rayon::current_num_threads() as u32;
        if zstd_threads > 1 {
            enc.multithread(zstd_threads)?;
        }
        Box::new(std::io::BufWriter::with_capacity(
            16 << 20,
            enc.auto_finish(),
        ))
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

    // Kernel consts
    let n_i32 = n as i32;
    let k_i32 = k as i32;
    let only_upper_i32 = 0i32; // compute full strip

    // Choose tile_rows dynamically (~512 MiB device scratch target)
    let target_bytes: usize = 512 << 20; // 512 MiB
    let mut tile_rows = ((target_bytes / 8).saturating_div(tile_cols.max(1))).max(1);
    tile_rows = tile_rows.min(4096).min(n); // safety cap similar to in-memory
    let bh_max = tile_cols.min(n);

    // Device/host scratch for a bw × bh_max tile
    // NOTE: device writes only the first (bw*bh) entries; we copy bw*bh_max then use needed part.
    let mut d_tile: CudaSlice<f64> = stream.alloc_zeros(tile_rows * bh_max)?;
    let mut h_tile = vec![0.0f64; tile_rows * bh_max];

    // For formatting
    let mut fmt = ryu::Buffer::new();

    // Process by row blocks (bw rows at a time)
    let mut i0 = 0usize;
    while i0 < n {
        let bw = (n - i0).min(tile_rows);

        // Prepare per-row line buffers (prefix with sample name once)
        let mut lines: Vec<String> = (0..bw)
            .map(|ii| {
                let mut s = String::with_capacity(8 + n * 12);
                s.push_str(&names[i0 + ii]);
                s
            })
            .collect();

        // Sweep columns in stripes
        let mut j0 = 0usize;
        while j0 < n {
            let bh = (n - j0).min(tile_cols);

            let blk_x = 64usize; // threads along columns (j)
            let blk_y = 8usize;  // threads along rows (i)
            let bk    = 128usize; // must match BK in kernel (see #define above)

            // dynamic shared memory: (BK * (blk_y + blk_x)) * sizeof(u64)
            let smem_bytes = ((bk * (blk_y + blk_x)) * std::mem::size_of::<u64>()) as u32;

            let cfg = LaunchConfig {
                grid_dim: (
                    ((bh + blk_x - 1) / blk_x) as u32,
                    ((bw + blk_y - 1) / blk_y) as u32,
                    1,
                ),
                block_dim: (blk_x as u32, blk_y as u32, 1),
                shared_mem_bytes: smem_bytes,
            };

            let i0_i32 = i0 as i32;
            let j0_i32 = j0 as i32;
            let bw_i32 = bw as i32;
            let bh_i32 = bh as i32;

            let mut launch = stream.launch_builder(&func);
            launch.arg(&d_sketches);
            launch.arg(&n_i32);
            launch.arg(&k_i32);
            launch.arg(&i0_i32);
            launch.arg(&j0_i32);
            launch.arg(&bw_i32);
            launch.arg(&bh_i32);
            launch.arg(&mut d_tile);
            launch.arg(&only_upper_i32);

            unsafe { launch.launch(cfg) }?;
            stream.synchronize()?; // kernel done

            // Copy full scratch (bw * bh_max) to satisfy cudarc's size check
            stream.memcpy_dtoh(&d_tile, &mut h_tile)?;

            // Append this stripe to each row's line
            for ii in 0..bw {
                let row_off = ii * bh; // kernel packed with ldo=bh
                let line = &mut lines[ii];
                for jj in 0..bh {
                    let mut d = h_tile[row_off + jj];
                    if weighted_normalized {
                        d = if d < 2.0 { d / (2.0 - d) } else { 1.0 };
                    }
                    line.push('\t');
                    line.push_str(fmt.format_finite(d));
                }
            }

            j0 += bh;
        }

        // Flush bw completed rows
        for s in lines {
            writer.write_all(s.as_bytes())?;
            writer.write_all(b"\n")?;
        }

        i0 += bw;
    }

    writer.flush()?;
    Ok(())
}

/// Multi-GPU compute & streaming writer: GPU workers compute rows; writer outputs in order via CPU.
/// Multi-GPU compute & streaming writer:
/// Each GPU processes blocks of `rows_per_launch` rows in a block-strided fashion
/// and sends completed lines to a single writer thread that writes in-order.
fn write_matrix_streaming_gpu_multi(
    names: &[String],
    sketches_flat_u64: &[u64], // [n*k], row-major
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted_normalized: bool,
    tile_cols: usize,
    devices: &[usize],
) -> Result<()> {
    use std::collections::BTreeMap;
    use std::io::Write;
    use std::sync::mpsc;

    let (tx, rx) = mpsc::sync_channel::<(usize, String)>(devices.len() * 2);

    // Writer thread: drains rows in-order
    let names_vec = names.to_vec();
    let path_str = path.to_string();
    let writer_handle = std::thread::spawn(move || -> Result<()> {
        let mut writer: Box<dyn std::io::Write> = if compress {
            let file = std::fs::File::create(&path_str)?;
            let mut enc = zstd::Encoder::new(file, 0)?;
            let zstd_threads = rayon::current_num_threads() as u32;
            if zstd_threads > 1 {
                enc.multithread(zstd_threads)?;
            }
            Box::new(std::io::BufWriter::with_capacity(
                16 << 20,
                enc.auto_finish(),
            ))
        } else {
            Box::new(std::io::BufWriter::with_capacity(
                16 << 20,
                std::fs::File::create(&path_str)?,
            ))
        };

        // header
        writer.write_all(b"")?;
        for name in &names_vec {
            writer.write_all(b"\t")?;
            writer.write_all(name.as_bytes())?;
        }
        writer.write_all(b"\n")?;

        let mut next = 0usize;
        let mut stash: BTreeMap<usize, String> = BTreeMap::new();
        while next < names_vec.len() {
            let (i, line) = rx.recv().expect("worker channel closed unexpectedly");
            stash.insert(i, line);
            while let Some(line) = stash.remove(&next) {
                writer.write_all(line.as_bytes())?;
                next += 1;
            }
        }
        writer.flush()?;
        Ok(())
    });

    // GPU workers: assign row-blocks in block-stride by #devices
    std::thread::scope(|scope| {
        for (widx, &dev_id) in devices.iter().enumerate() {
            let tx = tx.clone();
            let names = names.to_vec();
            let sketches = sketches_flat_u64.to_vec();

            scope.spawn(move || {
                let inner = || -> Result<()> {
                    let ctx = CudaContext::new(dev_id)?;
                    let stream = ctx.default_stream();

                    let ptx = compile_ptx(KERNEL_SRC)?;
                    let module = ctx.load_module(ptx)?;
                    let func = module.load_function("hamming_tile_u64")?;

                    // Upload sketches once per device
                    let d_sketches: CudaSlice<u64> = stream.memcpy_stod(&sketches[..])?;

                    // Kernel consts
                    let n_i32 = n as i32;
                    let k_i32 = k as i32;
                    let only_upper_i32 = 0i32;

                    // Pick tile_rows (~512 MiB scratch)
                    let target_bytes: usize = 512 << 20;
                    let mut tile_rows =
                        ((target_bytes / 8).saturating_div(tile_cols.max(1))).max(1);
                    tile_rows = tile_rows.min(4096).min(n);
                    let bh_max = tile_cols.min(n);

                    // Scratch for bw × bh_max
                    let mut d_tile: CudaSlice<f64> = stream.alloc_zeros(tile_rows * bh_max)?;
                    let mut h_tile = vec![0.0f64; tile_rows * bh_max];

                    let devs = devices.len();
                    let mut fmt = ryu::Buffer::new();

                    // Block-stride over row blocks
                    let mut i0 = widx * tile_rows;
                    while i0 < n {
                        let bw = (n - i0).min(tile_rows);

                        // per-row string buffers for this block
                        let mut lines: Vec<String> = (0..bw)
                            .map(|ii| {
                                let mut s = String::with_capacity(8 + n * 12);
                                s.push_str(&names[i0 + ii]);
                                s
                            })
                            .collect();

                        // sweep columns
                        let mut j0 = 0usize;
                        while j0 < n {
                            let bh = (n - j0).min(tile_cols);

                            let blk_x = 64usize; // threads along columns (j)
                            let blk_y = 8usize;  // threads along rows (i)
                            let bk    = 128usize; // must match BK in kernel (see #define above)

                            // dynamic shared memory: (BK * (blk_y + blk_x)) * sizeof(u64)
                            let smem_bytes = ((bk * (blk_y + blk_x)) * std::mem::size_of::<u64>()) as u32;

                            let cfg = LaunchConfig {
                                grid_dim: (
                                    ((bh + blk_x - 1) / blk_x) as u32,
                                    ((bw + blk_y - 1) / blk_y) as u32,
                                    1,
                                ),
                                block_dim: (blk_x as u32, blk_y as u32, 1),
                                shared_mem_bytes: smem_bytes,
                            };

                            let i0_i32 = i0 as i32;
                            let j0_i32 = j0 as i32;
                            let bw_i32 = bw as i32;
                            let bh_i32 = bh as i32;

                            let mut launch = stream.launch_builder(&func);
                            launch.arg(&d_sketches);
                            launch.arg(&n_i32);
                            launch.arg(&k_i32);
                            launch.arg(&i0_i32);
                            launch.arg(&j0_i32);
                            launch.arg(&bw_i32);
                            launch.arg(&bh_i32);
                            launch.arg(&mut d_tile);
                            launch.arg(&only_upper_i32);

                            unsafe { launch.launch(cfg) }?;
                            stream.synchronize()?;
                            stream.memcpy_dtoh(&d_tile, &mut h_tile)?; // copy bw*bh_max

                            // append stripe
                            for ii in 0..bw {
                                let row_off = ii * bh; // kernel packed with ldo=bh
                                let line = &mut lines[ii];
                                for jj in 0..bh {
                                    let mut d = h_tile[row_off + jj];
                                    if weighted_normalized {
                                        d = if d < 2.0 { d / (2.0 - d) } else { 1.0 };
                                    }
                                    line.push('\t');
                                    line.push_str(fmt.format_finite(d));
                                }
                            }

                            j0 += bh;
                        }

                        // emit completed rows
                        for ii in 0..bw {
                            let i = i0 + ii;
                            let mut s = lines[ii].clone();
                            s.push('\n');
                            tx.send((i, s)).expect("send row to writer");
                        }

                        // next block for this device
                        i0 = i0.saturating_add(tile_rows * devs);
                    }

                    Ok(())
                };
                if let Err(e) = inner() {
                    panic!("multi-GPU streaming worker on device {dev_id} failed: {e:?}");
                }
            });
        }
    });

    drop(tx);
    writer_handle
        .join()
        .map_err(|_| anyhow::anyhow!("writer thread panicked"))??;
    Ok(())
}
/// GPU compute and streaming write:
/// - if >=2 GPUs detected: multi-GPU streaming across all devices
/// - if 1 GPU: single-GPU streaming on device 0
pub fn write_matrix_streaming_gpu_auto(
    names: &[String],
    sketches_flat_u64: &[u64],
    n: usize,
    k: usize,
    path: &str,
    compress: bool,
    weighted_normalized: bool,
    tile_cols: usize,
) -> Result<()> {
    let ng = device_count()?; // may be 0 if CUDA not present
    if ng == 0 {
        bail!("CUDA streaming requested but no CUDA devices are available");
    } else if ng == 1 {
        write_matrix_streaming_gpu_single(
            names,
            sketches_flat_u64,
            n,
            k,
            path,
            compress,
            weighted_normalized,
            tile_cols,
            0,
        )
    } else {
        let devices: Vec<usize> = (0..ng).collect();
        write_matrix_streaming_gpu_multi(
            names,
            sketches_flat_u64,
            n,
            k,
            path,
            compress,
            weighted_normalized,
            tile_cols,
            &devices,
        )
    }
}
