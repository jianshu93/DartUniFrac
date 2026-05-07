//! stripe_cu.rs
//! CUDA offload for Striped UniFrac (unweighted + weighted normalized only).
//!
//! This version:
//! - Keeps the existing unweighted tile kernel.
//! - Uses a weighted normalized CUDA path based on diagonal distance stripes and branch batches.
//! - Avoids sparse atomics and avoids per-rectangular-tile row-map rebuild/copy.
//! - Uses the min-sum identity:
//!     den(i,j) = sample_sum[i] + sample_sum[j]
//!     d(i,j) = 1 - 2 * sum_v(len[v] * min(p[v,i], p[v,j])) / den(i,j)
//!
//! NOTE:
//! - Requires `--features cuda` and `cudarc`.
//! - Weighted GPU assumes normalized weighted UniFrac (alpha=1).

use anyhow::{bail, Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use cudarc::driver::{CudaContext, CudaSlice, LaunchConfig};
use cudarc::driver::PushKernelArg;
use cudarc::nvrtc::compile_ptx;
use log::info;
use rayon::prelude::*;
use std::ptr::NonNull;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

// Small helper for raw output ptr

#[derive(Clone, Copy)]
struct DistPtr(NonNull<f64>);

impl DistPtr {
    #[inline]
    fn as_mut_ptr(self) -> *mut f64 {
        self.0.as_ptr()
    }
}

// Safety: we guarantee each GPU worker writes disjoint (i,j) regions.
unsafe impl Send for DistPtr {}
unsafe impl Sync for DistPtr {}

// Options / Inputs 

#[derive(Clone, Debug)]
pub struct GpuOptions {
    pub devices: Vec<usize>,
    pub block_rows: usize,
    pub block_dim_x: u32,
    pub block_dim_y: u32,
}

impl Default for GpuOptions {
    fn default() -> Self {
        Self {
            devices: Vec::new(),
            block_rows: 1024,
            block_dim_x: 16,
            block_dim_y: 16,
        }
    }
}

pub fn device_count() -> Result<usize> {
    Ok(CudaContext::device_count()? as usize)
}

pub enum InputTable<'a> {
    DenseCounts(&'a [Vec<f64>]), // rows x nsamp
    Csr {
        indptr: &'a [u32],
        indices: &'a [u32],
        data: &'a [f64],
    },
}

// Public API: Unweighted GPU

pub fn unifrac_striped_unweighted_gpu(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    masks: Vec<BitVec<u8, Lsb0>>,
    opts: GpuOptions,
) -> Result<Vec<f64>> {
    let t_all = Instant::now();
    let nsamp = masks.len();
    if nsamp == 0 {
        return Ok(Vec::new());
    }

    let (node_bits, active_per_strip, blk) =
        build_unweighted_node_bits_and_active(post, kids, lens, leaf_ids, masks)?;

    let nblk = (nsamp + blk - 1) / blk;
    let pairs: Vec<(usize, usize)> = (0..nblk)
        .flat_map(|bi| (bi..nblk).map(move |bj| (bi, bj)))
        .collect();

    let dist = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let out_ptr: DistPtr = DistPtr(unsafe { NonNull::new_unchecked(dist.as_ptr() as *mut f64) });

    let devices = pick_devices(&opts, pairs.len())?;
    if devices.is_empty() {
        bail!("GPU requested but no CUDA devices available");
    }

    info!(
        "GPU(unweighted): nsamp={} blk={} nblk={} tiles={} visible_gpus={} using_gpus={:?}",
        nsamp,
        blk,
        nblk,
        pairs.len(),
        device_count().unwrap_or(0),
        devices
    );

    let ptx = Arc::new(compile_ptx(KERNEL_SRC).context("nvrtc compile PTX")?);

    let node_bits = Arc::new(node_bits);
    let active_per_strip = Arc::new(active_per_strip);
    let lens_arc = Arc::new(lens.to_vec());
    let pairs = Arc::new(pairs);

    let ng = devices.len();

    thread::scope(|scope| {
        for (widx, &dev_id) in devices.iter().enumerate() {
            let pairs = Arc::clone(&pairs);
            let node_bits = Arc::clone(&node_bits);
            let active_per_strip = Arc::clone(&active_per_strip);
            let lens_arc = Arc::clone(&lens_arc);
            let ptx = Arc::clone(&ptx);

            scope.spawn(move || {
                let inner = || -> Result<()> {
                    let ctx = CudaContext::new(dev_id)?;
                    let stream = ctx.default_stream();

                    let module = ctx.load_module((*ptx).clone())?;
                    let f_unw = module
                        .load_function("unifrac_unweighted_tile_u64")
                        .context("load kernel unifrac_unweighted_tile_u64")?;

                    // Output tile buffer: blk*blk (max)
                    let max_elems = blk * blk;
                    let mut d_out: CudaSlice<f32> = stream.alloc_zeros(max_elems)?;
                    let mut h_out = vec![0.0f32; max_elems];

                    let n_i32 = nsamp as i32;

                    for (tix, &(bi, bj)) in pairs.iter().enumerate() {
                        if tix % ng != widx {
                            continue;
                        }

                        let i0 = bi * blk;
                        let i1 = ((bi + 1) * blk).min(nsamp);
                        let j0 = bj * blk;
                        let j1 = ((bj + 1) * blk).min(nsamp);

                        let bw = i1 - i0;
                        let bh = j1 - j0;
                        if bw == 0 || bh == 0 {
                            continue;
                        }

                        let nodes_u =
                            merge_union_sorted_usize(&active_per_strip[bi], &active_per_strip[bj]);
                        if nodes_u.is_empty() {
                            continue;
                        }

                        let words_a = (bw + 63) >> 6;
                        let words_b = (bh + 63) >> 6;

                        let mut bits_a = vec![0u64; nodes_u.len() * words_a];
                        let mut bits_b = vec![0u64; nodes_u.len() * words_b];
                        let mut lens_v = vec![0f32; nodes_u.len()];

                        for (ni, &v) in nodes_u.iter().enumerate() {
                            lens_v[ni] = lens_arc[v];
                            let raw = node_bits[v].as_raw_slice();
                            extract_words_into(
                                raw,
                                i0,
                                bw,
                                &mut bits_a[ni * words_a..(ni + 1) * words_a],
                            );
                            extract_words_into(
                                raw,
                                j0,
                                bh,
                                &mut bits_b[ni * words_b..(ni + 1) * words_b],
                            );
                        }

                        // Per-tile allocs here are still present (bits_a/b + lens_v),
                        // but these are host-side; device allocs are via clone_htod.
                        // (If you want, we can also reuse d_bits buffers similarly.)
                        let d_bits_a: CudaSlice<u64> = stream.clone_htod(bits_a.as_slice())?;
                        let d_bits_b: CudaSlice<u64> = stream.clone_htod(bits_b.as_slice())?;
                        let d_lens: CudaSlice<f32> = stream.clone_htod(lens_v.as_slice())?;

                        let num_nodes_i32 = nodes_u.len() as i32;
                        let words_a_i32 = words_a as i32;
                        let words_b_i32 = words_b as i32;
                        let i0_i32 = i0 as i32;
                        let j0_i32 = j0 as i32;
                        let bw_i32 = bw as i32;
                        let bh_i32 = bh as i32;
                        let only_upper_i32 = if bi == bj { 1i32 } else { 0i32 };

                        let cfg = LaunchConfig {
                            grid_dim: (
                                ((bh as u32 + opts.block_dim_x - 1) / opts.block_dim_x),
                                ((bw as u32 + opts.block_dim_y - 1) / opts.block_dim_y),
                                1,
                            ),
                            block_dim: (opts.block_dim_x, opts.block_dim_y, 1),
                            shared_mem_bytes: 0,
                        };

                        let mut launch = stream.launch_builder(&f_unw);
                        launch.arg(&d_bits_a);
                        launch.arg(&d_bits_b);
                        launch.arg(&d_lens);
                        launch.arg(&num_nodes_i32);
                        launch.arg(&words_a_i32);
                        launch.arg(&words_b_i32);
                        launch.arg(&n_i32);
                        launch.arg(&i0_i32);
                        launch.arg(&j0_i32);
                        launch.arg(&bw_i32);
                        launch.arg(&bh_i32);
                        launch.arg(&only_upper_i32);
                        launch.arg(&mut d_out);

                        unsafe { launch.launch(cfg) }?;
                        stream.memcpy_dtoh(&d_out, &mut h_out)?;

                        unsafe {
                            see_scatter_tile_to_host(out_ptr, &h_out, nsamp, i0, j0, bw, bh);
                        }
                    }

                    Ok(())
                };

                if let Err(e) = inner() {
                    panic!("GPU worker dev={} failed: {e:?}", dev_id);
                }
            });
        }
    });

    info!(
        "GPU(unweighted): total wall time {} ms",
        t_all.elapsed().as_millis()
    );

    Ok(Arc::try_unwrap(dist).unwrap())
}

// Public API: Weighted GPU (normalized only)
//
// This path is intentionally NOT the sparse-atomic prototype.  It follows the
// same algebra used by fast weighted UniFrac implementations:
//
//   den(i,j) = sample_sum[i] + sample_sum[j]
//   num(i,j) = den(i,j) - 2 * Σ_v len[v] * min(p[v,i], p[v,j])
//   d(i,j)   = num / den
//
// where sample_sum[s] = Σ_v len[v] * p[v,s].  For each tile, the CUDA kernel
// only loops over branches that are present in both sample stripes.  This keeps
// the GPU path dense/thread-per-pair, avoids global atomics, and avoids scanning
// branch rows that can only contribute through the precomputed denominator.

pub fn unifrac_striped_weighted_gpu(
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    row2leaf: &[Option<usize>],
    table: InputTable<'_>,
    nsamp: usize,
    col_sums: &[f64],
    opts: GpuOptions,
) -> Result<Vec<f64>> {
    let t_all = Instant::now();
    if nsamp == 0 {
        return Ok(Vec::new());
    }
    if col_sums.len() != nsamp {
        bail!(
            "col_sums length mismatch: got {}, expected {}",
            col_sums.len(),
            nsamp
        );
    }

    let total = lens.len();

    // parent[]
    let parent: Vec<usize> = {
        let mut p = vec![usize::MAX; total];
        for v in 0..total {
            for &c in &kids[v] {
                p[c] = v;
            }
        }
        p
    };

    // This block size is only used for the one-time branch embedding build.
    // The weighted CUDA compute below is diagonal-stripe based, not rectangular-tile based.
    let blk = opts
        .block_rows
        .max(1)
        .min(nsamp)
        .next_power_of_two()
        .clamp(64, 4096);
    let nblk = (nsamp + blk - 1) / blk;

    let devices = pick_devices(&opts, nsamp.saturating_sub(1))?;
    if devices.is_empty() {
        bail!("GPU requested but no CUDA devices available");
    }

    info!(
        "GPU(weighted diagonal): nsamp={} embed_blk={} nblk={} diag_stripes={} visible_gpus={} using_gpus={:?}",
        nsamp,
        blk,
        nblk,
        nsamp.saturating_sub(1),
        device_count().unwrap_or(0),
        devices
    );

    // Compile PTX once.
    let ptx = Arc::new(compile_ptx(KERNEL_SRC).context("nvrtc compile PTX")?);

    // Share table.
    let dense_counts: Option<Arc<Vec<Vec<f64>>>> = match table {
        InputTable::DenseCounts(c) => Some(Arc::new(c.to_vec())),
        _ => None,
    };
    let csr_pack: Option<(Arc<Vec<u32>>, Arc<Vec<u32>>, Arc<Vec<f64>>)> = match table {
        InputTable::Csr {
            indptr,
            indices,
            data,
        } => Some((
            Arc::new(indptr.to_vec()),
            Arc::new(indices.to_vec()),
            Arc::new(data.to_vec()),
        )),
        _ => None,
    };
    if dense_counts.is_none() && csr_pack.is_none() {
        bail!("invalid table mode");
    }

    let parent = Arc::new(parent);
    let leaf_ids = Arc::new(leaf_ids.to_vec());
    let row2leaf = Arc::new(row2leaf.to_vec());
    let col_sums = Arc::new(col_sums.to_vec());
    let lens_f32 = Arc::new(lens.to_vec());

    // Build sample stripes once on CPU.  These are used only to assemble branch batches;
    // the GPU compute itself does not revisit rectangular stripe pairs.
    let t_stripes = Instant::now();
    let stripes: Vec<Stripe> = (0..nblk)
        .into_par_iter()
        .map(|bi| {
            let s0 = bi * blk;
            let s1 = ((bi + 1) * blk).min(nsamp);

            match (&dense_counts, &csr_pack) {
                (Some(c), _) => build_stripe_dense_compact(
                    c,
                    &row2leaf,
                    &leaf_ids,
                    &parent,
                    &col_sums,
                    s0,
                    s1,
                    total,
                ),
                (_, Some((ip, idx, dat))) => build_stripe_csr_compact(
                    ip,
                    idx,
                    dat,
                    &row2leaf,
                    &leaf_ids,
                    &parent,
                    &col_sums,
                    s0,
                    s1,
                    total,
                ),
                _ => unreachable!(),
            }
        })
        .collect();

    info!(
        "GPU(weighted diagonal): precomputed {} embedding stripes in {} ms",
        nblk,
        t_stripes.elapsed().as_millis()
    );

    // Active positive-length nodes, and sample_sums[s] = Σ_v len[v] * p[v,s].
    // The final denominator is den(i,j) = sample_sums[i] + sample_sums[j].
    let mut active = vec![false; total];
    let mut sample_sums = vec![0.0f32; nsamp];
    for (bi, stripe) in stripes.iter().enumerate() {
        let s0 = bi * blk;
        let width = stripe.width;
        for (ri, &nid) in stripe.nodes.iter().enumerate() {
            let v = nid as usize;
            let len = lens_f32[v];
            if len <= 0.0 {
                continue;
            }
            active[v] = true;
            let row0 = ri * width;
            for c in 0..width {
                let x = stripe.rows[row0 + c];
                if x != 0.0 {
                    sample_sums[s0 + c] += len * x;
                }
            }
        }
    }

    let active_nodes: Vec<u32> = active
        .iter()
        .enumerate()
        .filter_map(|(v, &a)| {
            if a && lens_f32[v] > 0.0 {
                Some(v as u32)
            } else {
                None
            }
        })
        .collect();

    let branch_batch: usize = std::env::var("UNIFRAC_CUDA_BRANCH_BATCH")
        .ok()
        .and_then(|x| x.parse::<usize>().ok())
        .unwrap_or(2048)
        .clamp(64, 16384);

    info!(
        "GPU(weighted diagonal): active positive branches={} branch_batch={}",
        active_nodes.len(),
        branch_batch
    );

    let stripes = Arc::new(stripes);
    let active_nodes = Arc::new(active_nodes);
    let sample_sums = Arc::new(sample_sums);
    let lens_f32 = Arc::clone(&lens_f32);

    // One full float matrix on device accumulates shared min-sums over branch batches.
    // This intentionally trades GPU memory for avoiding per-tile branch/row-map copies.
    // For 25,145 samples this is ~2.53 GB as f32.
    let dist = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let out_ptr = DistPtr(unsafe { NonNull::new_unchecked(dist.as_ptr() as *mut f64) });

    // Current implementation uses one GPU for the full resident matrix.  Multiple GPUs can be
    // added later by splitting diagonal ranges, but this version avoids inter-GPU reduction.
    let dev_id = devices[0];
    let ctx = CudaContext::new(dev_id)?;
    let stream = ctx.default_stream();
    let module = ctx.load_module((*ptx).clone())?;
    let f_accum = module
        .load_function("unifrac_weighted_diag_accum_minsum_f32")
        .context("load kernel unifrac_weighted_diag_accum_minsum_f32")?;
    let f_norm = module
        .load_function("unifrac_weighted_diag_normalize_f32")
        .context("load kernel unifrac_weighted_diag_normalize_f32")?;

    let matrix_elems = nsamp
        .checked_mul(nsamp)
        .context("nsamp*nsamp overflow for weighted GPU matrix")?;
    let mut d_shared: CudaSlice<f32> = stream.alloc_zeros(matrix_elems)?;
    let d_sample_sums: CudaSlice<f32> = stream.clone_htod(sample_sums.as_slice())?;

    let n_i32 = nsamp as i32;
    let block_x = opts.block_dim_x.max(1);
    let block_y = opts.block_dim_y.max(1);
    let diag_count = nsamp.saturating_sub(1) as u32;

    // Process branch batches.  Each batch is materialized as dense branch-major
    // [batch_nodes x nsamp] only once, uploaded, and then used by a diagonal-pair kernel.
    let t_batches = Instant::now();
    let nbatches = (active_nodes.len() + branch_batch - 1) / branch_batch;

    for b0 in (0..active_nodes.len()).step_by(branch_batch) {
        let b1 = (b0 + branch_batch).min(active_nodes.len());
        let cur = b1 - b0;
        let batch_id = b0 / branch_batch + 1;

        let mut h_emb = vec![0.0f32; cur * nsamp];
        let mut h_lens = vec![0.0f32; cur];
        let mut h_sample_has_batch = vec![0u8; nsamp];

        // Map global node id -> local batch row.
        let mut local_of = vec![u32::MAX; total];
        for (local, &nid) in active_nodes[b0..b1].iter().enumerate() {
            let v = nid as usize;
            local_of[v] = local as u32;
            h_lens[local] = lens_f32[v];
        }

        // Assemble dense embedded branch batch from the one-time CPU stripes.
        for (bi, stripe) in stripes.iter().enumerate() {
            let s0 = bi * blk;
            let width = stripe.width;
            for (ri, &nid) in stripe.nodes.iter().enumerate() {
                let local = local_of[nid as usize];
                if local == u32::MAX {
                    continue;
                }
                let local = local as usize;
                let src0 = ri * width;
                let dst0 = local * nsamp + s0;
                h_emb[dst0..dst0 + width].copy_from_slice(&stripe.rows[src0..src0 + width]);
                for c in 0..width {
                    if stripe.rows[src0 + c] > 0.0 {
                        h_sample_has_batch[s0 + c] = 1;
                    }
                }
            }
        }

        let d_emb: CudaSlice<f32> = stream.clone_htod(h_emb.as_slice())?;
        let d_lens_batch: CudaSlice<f32> = stream.clone_htod(h_lens.as_slice())?;
        let d_sample_has_batch: CudaSlice<u8> = stream.clone_htod(h_sample_has_batch.as_slice())?;

        let cur_i32 = cur as i32;
        let cfg = LaunchConfig {
            grid_dim: (
                ((nsamp as u32 + block_x - 1) / block_x),
                ((diag_count + block_y - 1) / block_y),
                1,
            ),
            block_dim: (block_x, block_y, 1),
            shared_mem_bytes: 0,
        };

        let mut launch = stream.launch_builder(&f_accum);
        launch.arg(&d_emb);
        launch.arg(&d_lens_batch);
        launch.arg(&d_sample_has_batch);
        launch.arg(&cur_i32);
        launch.arg(&n_i32);
        launch.arg(&mut d_shared);
        unsafe { launch.launch(cfg) }?;

        if batch_id == 1 || batch_id == nbatches || (batch_id & 7) == 0 {
            info!(
                "GPU(weighted diagonal): processed branch batch {}/{} ({} branches)",
                batch_id,
                nbatches,
                cur
            );
        }
    }

    info!(
        "GPU(weighted diagonal): accumulated branch batches in {} ms",
        t_batches.elapsed().as_millis()
    );

    // Convert shared min-sums into distances in-place in d_shared upper triangle.
    let cfg_norm = LaunchConfig {
        grid_dim: (
            ((nsamp as u32 + block_x - 1) / block_x),
            ((diag_count + block_y - 1) / block_y),
            1,
        ),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(&f_norm);
    launch.arg(&d_sample_sums);
    launch.arg(&n_i32);
    launch.arg(&mut d_shared);
    unsafe { launch.launch(cfg_norm) }?;

    // Copy the upper-triangular f32 result back and scatter to symmetric f64 host output.
    let t_copy = Instant::now();
    let mut h_mat = vec![0.0f32; matrix_elems];
    stream.memcpy_dtoh(&d_shared, &mut h_mat)?;
    unsafe {
        scatter_upper_matrix_to_host(out_ptr, &h_mat, nsamp);
    }
    drop(h_mat);

    info!(
        "GPU(weighted diagonal): copied/scattered output in {} ms",
        t_copy.elapsed().as_millis()
    );
    info!(
        "GPU(weighted diagonal): total wall time {} ms",
        t_all.elapsed().as_millis()
    );

    Ok(Arc::try_unwrap(dist).unwrap())
}

// CUDA kernels 

const KERNEL_SRC: &str = r#"
extern "C" __global__
void unifrac_unweighted_tile_u64(
    const unsigned long long* __restrict__ bitsA,
    const unsigned long long* __restrict__ bitsB,
    const float* __restrict__ lens,
    int num_nodes,
    int wordsA,
    int wordsB,
    int n,
    int i0,
    int j0,
    int bw,
    int bh,
    int only_upper,
    float* __restrict__ out
){
    int jj = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ii = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (ii >= bw || jj >= bh) return;

    int gi = i0 + ii;
    int gj = j0 + jj;
    if (gi >= n || gj >= n) return;

    if (only_upper && gj <= gi) return;
    if (gi == gj) return;

    int wi = ii >> 6;
    int wj = jj >> 6;
    unsigned long long mi = 1ULL << (ii & 63);
    unsigned long long mj = 1ULL << (jj & 63);

    float u = 0.0f;
    float s = 0.0f;

    for (int v = 0; v < num_nodes; ++v) {
        float len = lens[v];
        if (len <= 0.0f) continue;

        unsigned long long a = bitsA[(size_t)v * (size_t)wordsA + (size_t)wi];
        unsigned long long b = bitsB[(size_t)v * (size_t)wordsB + (size_t)wj];

        int a_set = ((a & mi) != 0ULL);
        int b_set = ((b & mj) != 0ULL);

        if (a_set | b_set) {
            u += len;
            if (a_set & b_set) s += len;
        }
    }

    float d = 0.0f;
    if (u > 0.0f) d = 1.0f - (s / u);
    out[(size_t)ii * (size_t)bh + (size_t)jj] = d;
}

extern "C" __global__
void unifrac_weighted_tile_minsum_f32(
    const float* __restrict__ rowsA,
    const float* __restrict__ rowsB,
    const float* __restrict__ lens_all,
    const float* __restrict__ sample_sums,
    const unsigned int* __restrict__ common_nodes,
    const int* __restrict__ mapA,
    const int* __restrict__ mapB,
    int num_common,
    int bw,
    int bh,
    int i0,
    int j0,
    int only_upper,
    float* __restrict__ out
){
    int jj = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int ii = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    if (ii >= bw || jj >= bh) return;

    int gi = i0 + ii;
    int gj = j0 + jj;

    if (only_upper && gj <= gi) return;
    if (gi == gj) return;

    float den = sample_sums[(size_t)gi] + sample_sums[(size_t)gj];
    float minsum = 0.0f;

    for (int k = 0; k < num_common; ++k) {
        unsigned int nid = common_nodes[k];
        float len = lens_all[(size_t)nid];
        if (len <= 0.0f) continue;

        int ra = mapA[k];
        int rb = mapB[k];

        float a = rowsA[(size_t)ra * (size_t)bw + (size_t)ii];
        float b = rowsB[(size_t)rb * (size_t)bh + (size_t)jj];
        float m = a < b ? a : b;
        if (m > 0.0f) minsum += len * m;
    }

    float d = 0.0f;
    if (den > 0.0f) {
        d = 1.0f - (2.0f * minsum / den);
        if (d < 0.0f) d = 0.0f;
        if (d > 1.0f) d = 1.0f;
    }
    out[(size_t)ii * (size_t)bh + (size_t)jj] = d;
}


extern "C" __global__
void unifrac_weighted_diag_accum_minsum_f32(
    const float* __restrict__ emb,          // [num_nodes * n], branch-major
    const float* __restrict__ lens_batch,   // [num_nodes]
    const unsigned char* __restrict__ sample_has_batch, // [n], 1 if sample has any nonzero in this branch batch
    int num_nodes,
    int n,
    float* __restrict__ shared              // [n * n], upper triangle accumulated min-sum
){
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int d = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 1;
    if (k >= n || d >= n) return;

    int l = k + d;
    if (l >= n) return;
    if (sample_has_batch[(size_t)k] == 0 || sample_has_batch[(size_t)l] == 0) return;

    float acc = 0.0f;
    for (int r = 0; r < num_nodes; ++r) {
        float len = lens_batch[(size_t)r];
        if (len <= 0.0f) continue;
        size_t base = (size_t)r * (size_t)n;
        float a = emb[base + (size_t)k];
        float b = emb[base + (size_t)l];
        float m = a < b ? a : b;
        if (m > 0.0f) acc += len * m;
    }

    shared[(size_t)k * (size_t)n + (size_t)l] += acc;
}

extern "C" __global__
void unifrac_weighted_diag_normalize_f32(
    const float* __restrict__ sample_sums,
    int n,
    float* __restrict__ shared_as_dist      // input: shared min-sum, output: distance
){
    int k = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    int d = (int)(blockIdx.y * blockDim.y + threadIdx.y) + 1;
    if (k >= n || d >= n) return;

    int l = k + d;
    if (l >= n) return;

    size_t idx = (size_t)k * (size_t)n + (size_t)l;
    float den = sample_sums[(size_t)k] + sample_sums[(size_t)l];
    float val = 0.0f;
    if (den > 0.0f) {
        val = 1.0f - (2.0f * shared_as_dist[idx] / den);
        if (val < 0.0f) val = 0.0f;
        if (val > 1.0f) val = 1.0f;
    }
    shared_as_dist[idx] = val;
}
"#;

// GPU selection

fn auto_gpu_count(visible: usize, tiles: usize) -> usize {
    if visible == 0 {
        return 0;
    }
    if tiles <= 1 {
        return 1;
    }
    const MIN_TILES_PER_GPU: usize = 64;
    let mut want = (tiles + MIN_TILES_PER_GPU - 1) / MIN_TILES_PER_GPU;
    want = want.max(1).min(tiles).min(visible);
    want
}

fn pick_devices(opts: &GpuOptions, tiles: usize) -> Result<Vec<usize>> {
    let visible = device_count()?;
    if visible == 0 {
        return Ok(Vec::new());
    }

    if !opts.devices.is_empty() {
        for &d in &opts.devices {
            if d >= visible {
                bail!(
                    "requested CUDA device {} but only {} devices available",
                    d,
                    visible
                );
            }
        }
        return Ok(opts.devices.clone());
    }

    let use_n = auto_gpu_count(visible, tiles);
    Ok((0..use_n).collect())
}

// Scatter helpers

#[inline(always)]
unsafe fn see_scatter_tile_to_host(
    out_ptr: DistPtr,
    h_out: &[f32],
    nsamp: usize,
    i0: usize,
    j0: usize,
    bw: usize,
    bh: usize,
) {
    let base = out_ptr.as_mut_ptr();
    for ii in 0..bw {
        let gi = i0 + ii;
        for jj in 0..bh {
            let gj = j0 + jj;
            if gj <= gi {
                continue;
            }
            let d = h_out[ii * bh + jj] as f64;
            unsafe {
                *base.add(gi * nsamp + gj) = d;
                *base.add(gj * nsamp + gi) = d;
            }
        }
    }
}


#[inline(always)]
unsafe fn scatter_upper_matrix_to_host(
    out_ptr: DistPtr,
    h_mat: &[f32],
    nsamp: usize,
) {
    let base = out_ptr.as_mut_ptr();
    for i in 0..nsamp {
        let row0 = i * nsamp;
        for j in (i + 1)..nsamp {
            let d = h_mat[row0 + j] as f64;
            unsafe {
                *base.add(i * nsamp + j) = d;
                *base.add(j * nsamp + i) = d;
            }
        }
    }
}

// Bit helpers (unweighted) 

fn merge_union_sorted_usize(a: &[usize], b: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let mut ia = 0usize;
    let mut ib = 0usize;
    while ia < a.len() || ib < b.len() {
        match (a.get(ia), b.get(ib)) {
            (Some(&va), Some(&vb)) => {
                if va < vb {
                    out.push(va);
                    ia += 1;
                } else if vb < va {
                    out.push(vb);
                    ib += 1;
                } else {
                    out.push(va);
                    ia += 1;
                    ib += 1;
                }
            }
            (Some(&va), None) => {
                out.push(va);
                ia += 1;
            }
            (None, Some(&vb)) => {
                out.push(vb);
                ib += 1;
            }
            _ => break,
        }
    }
    out
}

fn extract_words_into(raw: &[u64], start_bit: usize, len_bits: usize, dst: &mut [u64]) {
    if len_bits == 0 {
        return;
    }
    let words = (len_bits + 63) >> 6;
    debug_assert_eq!(dst.len(), words);

    let bit_off = start_bit & 63;
    let w0 = start_bit >> 6;

    for w in 0..words {
        let idx = w0 + w;
        let mut v = if idx < raw.len() { raw[idx] } else { 0u64 };
        if bit_off != 0 {
            v >>= bit_off;
            if idx + 1 < raw.len() {
                v |= raw[idx + 1] << (64 - bit_off);
            }
        }
        dst[w] = v;
    }

    let tail = len_bits & 63;
    if tail != 0 {
        let mask = (1u64 << tail) - 1;
        dst[words - 1] &= mask;
    }
}

// Unweighted CPU phase 1/2 

fn build_unweighted_node_bits_and_active(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    mut masks: Vec<BitVec<u8, Lsb0>>,
) -> Result<(Vec<BitVec<u64, Lsb0>>, Vec<Vec<usize>>, usize)> {
    let nsamp = masks.len();
    let total = lens.len();
    let n_threads = rayon::current_num_threads().max(1);

    let stripe = (nsamp + n_threads - 1) / n_threads;
    let words_str = (stripe + 63) >> 6;

    let mut node_masks: Vec<Vec<Vec<u64>>> = (0..n_threads)
        .map(|_| vec![vec![0u64; words_str]; total])
        .collect();

    let t0 = Instant::now();
    rayon::scope(|scope| {
        for (tid, node_masks_t) in node_masks.iter_mut().enumerate() {
            let stripe_start = tid * stripe;
            if stripe_start >= nsamp {
                break;
            }
            let stripe_end = (stripe_start + stripe).min(nsamp);

            let masks_slice = &masks[stripe_start..stripe_end];
            let leaf = leaf_ids;
            let kids = kids;
            let post = post;

            scope.spawn(move |_| {
                for (local_s, sm) in masks_slice.iter().enumerate() {
                    for pos in sm.iter_ones() {
                        let v = leaf[pos];
                        let w = local_s >> 6;
                        let b = local_s & 63;
                        node_masks_t[v][w] |= 1u64 << b;
                    }
                }
                for &v in post {
                    for &c in &kids[v] {
                        for w in 0..words_str {
                            node_masks_t[v][w] |= node_masks_t[c][w];
                        }
                    }
                }
            });
        }
    });
    info!(
        "GPU(unweighted): phase-1 masks built {} ms",
        t0.elapsed().as_millis()
    );

    masks.clear();
    masks.shrink_to_fit();

    let mut node_bits: Vec<BitVec<u64, Lsb0>> =
        (0..total).map(|_| BitVec::repeat(false, nsamp)).collect();

    node_bits
        .par_iter_mut()
        .enumerate()
        .for_each(|(v, bv)| {
            let dst_words = bv.as_raw_mut_slice();
            for tid in 0..n_threads {
                let stripe_start = tid * stripe;
                let stripe_end = (stripe_start + stripe).min(nsamp);
                if stripe_start >= stripe_end {
                    break;
                }

                let src_words = &node_masks[tid][v];
                let word_off = stripe_start >> 6;
                let bit_off = (stripe_start & 63) as u32;

                for w in 0..src_words.len() {
                    let mut val = src_words[w];
                    if w == src_words.len() - 1 {
                        let tail_bits = (stripe_end - stripe_start) & 63;
                        if tail_bits != 0 {
                            val &= (1u64 << tail_bits) - 1;
                        }
                    }
                    if val == 0 {
                        continue;
                    }
                    dst_words[word_off + w] |= val << bit_off;
                    if bit_off != 0 && word_off + w + 1 < dst_words.len() {
                        dst_words[word_off + w + 1] |= val >> (64 - bit_off);
                    }
                }
            }
        });

    drop(node_masks);

    let n_threads2 = rayon::current_num_threads().max(1);
    let est_blk = ((nsamp as f64 / (2.0 * n_threads2 as f64)).sqrt()) as usize;
    let blk = est_blk.clamp(64, 512).next_power_of_two();
    let nblk = (nsamp + blk - 1) / blk;

    let mut active_per_strip: Vec<Vec<usize>> = vec![Vec::new(); nblk];
    for v in 0..total {
        if lens[v] <= 0.0 {
            continue;
        }
        let raw = node_bits[v].as_raw_slice();
        for bi in 0..nblk {
            let i0 = bi * blk;
            let i1 = ((bi + 1) * blk).min(nsamp);
            let w0 = i0 >> 6;
            let w1 = (i1 + 63) >> 6;
            if raw[w0..w1].iter().any(|&w| w != 0) {
                active_per_strip[bi].push(v);
            }
        }
    }

    for lst in &mut active_per_strip {
        lst.sort_unstable();
        lst.dedup();
    }

    info!(
        "GPU(unweighted): phase-2 active lists built (blk={}, nblk={})",
        blk, nblk
    );

    Ok((node_bits, active_per_strip, blk))
}

// Weighted stripe building (compact) 

#[derive(Clone)]
struct Stripe {
    nodes: Vec<u32>, // sorted node ids
    rows: Vec<f32>,  // row-major [nrows * width]
    index: Vec<u32>, // len=total, maps node_id -> row index or u32::MAX
    width: usize,    // stripe width
}

#[inline]
fn ensure_row_slot_compact(
    v: usize,
    idx_of: &mut [u32],
    nodes: &mut Vec<u32>,
    rows: &mut Vec<f32>,
    width: usize,
) -> usize {
    let idx = idx_of[v];
    if idx != u32::MAX {
        return idx as usize;
    }
    let new_idx = nodes.len() as u32;
    idx_of[v] = new_idx;
    nodes.push(v as u32);
    rows.resize(rows.len() + width, 0.0f32);
    new_idx as usize
}

fn build_stripe_dense_compact(
    counts: &[Vec<f64>],
    row2leaf: &[Option<usize>],
    leaf_ids: &[usize],
    parent: &[usize],
    col_sums: &[f64],
    s0: usize,
    s1: usize,
    total: usize,
) -> Stripe {
    let width = s1 - s0;
    let mut idx_of = vec![u32::MAX; total];
    let mut nodes: Vec<u32> = Vec::new();
    let mut rows: Vec<f32> = Vec::new();

    for (r, lopt) in row2leaf.iter().enumerate() {
        let Some(lp) = *lopt else { continue };
        let v_leaf = leaf_ids[lp];

        let row = &counts[r];
        for s in s0..s1 {
            let denom = col_sums[s];
            if denom <= 0.0 {
                continue;
            }
            let val = row[s];
            if val <= 0.0 {
                continue;
            }
            let inc = (val / denom) as f32;
            let col = s - s0;

            let mut v = v_leaf;
            loop {
                let ri = ensure_row_slot_compact(v, &mut idx_of, &mut nodes, &mut rows, width);
                rows[ri * width + col] += inc;

                let p = parent[v];
                if p == usize::MAX {
                    break;
                }
                v = p;
            }
        }
    }

    sort_stripe_compact(nodes, rows, idx_of, width)
}

fn build_stripe_csr_compact(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    row2leaf: &[Option<usize>],
    leaf_ids: &[usize],
    parent: &[usize],
    col_sums: &[f64],
    s0: usize,
    s1: usize,
    total: usize,
) -> Stripe {
    let width = s1 - s0;
    let mut idx_of = vec![u32::MAX; total];
    let mut nodes: Vec<u32> = Vec::new();
    let mut rows: Vec<f32> = Vec::new();

    for r in 0..row2leaf.len() {
        let Some(lp) = row2leaf[r] else { continue };
        let v_leaf = leaf_ids[lp];

        let a = indptr[r] as usize;
        let b = indptr[r + 1] as usize;
        for k in a..b {
            let s = indices[k] as usize;
            if s < s0 || s >= s1 {
                continue;
            }
            let denom = col_sums[s];
            if denom <= 0.0 {
                continue;
            }
            let val = data[k];
            if val <= 0.0 {
                continue;
            }
            let inc = (val / denom) as f32;
            let col = s - s0;

            let mut v = v_leaf;
            loop {
                let ri = ensure_row_slot_compact(v, &mut idx_of, &mut nodes, &mut rows, width);
                rows[ri * width + col] += inc;

                let p = parent[v];
                if p == usize::MAX {
                    break;
                }
                v = p;
            }
        }
    }

    sort_stripe_compact(nodes, rows, idx_of, width)
}

fn sort_stripe_compact(
    mut nodes: Vec<u32>,
    mut rows: Vec<f32>,
    mut idx_of: Vec<u32>,
    width: usize,
) -> Stripe {
    if nodes.len() > 1 {
        let mut order: Vec<usize> = (0..nodes.len()).collect();
        order.sort_unstable_by_key(|&i| nodes[i]);

        let mut nodes_sorted = vec![0u32; nodes.len()];
        let mut rows_sorted = vec![0f32; rows.len()];

        for (new_i, &old_i) in order.iter().enumerate() {
            nodes_sorted[new_i] = nodes[old_i];
            let src0 = old_i * width;
            let dst0 = new_i * width;
            rows_sorted[dst0..dst0 + width].copy_from_slice(&rows[src0..src0 + width]);
        }

        idx_of.fill(u32::MAX);
        for (i, &nid) in nodes_sorted.iter().enumerate() {
            idx_of[nid as usize] = i as u32;
        }

        nodes = nodes_sorted;
        rows = rows_sorted;
    }

    Stripe {
        nodes,
        rows,
        index: idx_of,
        width,
    }
}

// union builder into existing Vec (no alloc)

#[inline]
fn merge_union_u32_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(a.len().saturating_add(b.len()));

    let mut ia = 0usize;
    let mut ib = 0usize;
    while ia < a.len() || ib < b.len() {
        match (a.get(ia), b.get(ib)) {
            (Some(&va), Some(&vb)) => {
                if va < vb {
                    out.push(va);
                    ia += 1;
                } else if vb < va {
                    out.push(vb);
                    ib += 1;
                } else {
                    out.push(va);
                    ia += 1;
                    ib += 1;
                }
            }
            (Some(&va), None) => {
                out.push(va);
                ia += 1;
            }
            (None, Some(&vb)) => {
                out.push(vb);
                ib += 1;
            }
            _ => break,
        }
    }
}
// intersection builder into existing Vec (no alloc)
#[inline]
fn merge_intersection_u32_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
    out.clear();
    out.reserve(a.len().min(b.len()));

    let mut ia = 0usize;
    let mut ib = 0usize;
    while ia < a.len() && ib < b.len() {
        let va = a[ia];
        let vb = b[ib];
        if va < vb {
            ia += 1;
        } else if vb < va {
            ib += 1;
        } else {
            out.push(va);
            ia += 1;
            ib += 1;
        }
    }
}
