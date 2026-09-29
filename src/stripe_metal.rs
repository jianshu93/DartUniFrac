//! metal backend for striped unifrac (apple silicon), unweighted and weighted.
//!
//! faithful translation of `unifrac_unweighted_tile_u64` in `stripe_cu.rs`: same
//! blocking, the same per-tile bitset extraction, and the same host scatter, all of
//! which come from `stripe_common`. the cuda kernel stages nothing in shared memory
//! and neither does this one yet, so the two should be directly comparable.

use anyhow::{Context, Result, anyhow, bail};
use bitvec::{order::Lsb0, vec::BitVec};
use log::{debug, info};
use metal::objc::rc::autoreleasepool;
use metal::{
    Buffer, CompileOptions, ComputePipelineState, Device, MTLDispatchType, MTLResourceOptions,
    MTLSize, NSUInteger,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::time::Instant;

use rayon::prelude::*;

use crate::stripe_common::{
    DistPtr, InputTable, Stripe, build_stripe_csr_compact, build_stripe_dense_compact,
    build_unweighted_node_bits_and_active, extract_words_into, merge_union_sorted_usize,
    scatter_upper_matrix_to_host, see_scatter_tile_to_host,
};

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    int num_nodes, words_a, words_b, n;
    int i0, j0, bw, bh, only_upper;
};

kernel void unifrac_unweighted_tile(
    device const ulong *bitsA [[buffer(0)]],
    device const ulong *bitsB [[buffer(1)]],
    device const float *lens  [[buffer(2)]],
    constant Params &P        [[buffer(3)]],
    device float *out         [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int jj = int(gid.x);
    const int ii = int(gid.y);
    if (ii >= P.bw || jj >= P.bh) return;

    const int gi = P.i0 + ii;
    const int gj = P.j0 + jj;
    if (gi >= P.n || gj >= P.n) return;
    if (P.only_upper && gj <= gi) return;
    if (gi == gj) return;

    const int wi = ii >> 6;
    const int wj = jj >> 6;
    const ulong mi = ulong(1) << (ii & 63);
    const ulong mj = ulong(1) << (jj & 63);

    // union and shared branch length over every node present in either sample
    float u = 0.0f;
    float s = 0.0f;

    for (int v = 0; v < P.num_nodes; ++v) {
        const float len = lens[v];
        if (len <= 0.0f) continue;

        const ulong a = bitsA[(uint)v * (uint)P.words_a + (uint)wi];
        const ulong b = bitsB[(uint)v * (uint)P.words_b + (uint)wj];

        const bool a_set = (a & mi) != 0;
        const bool b_set = (b & mj) != 0;

        if (a_set || b_set) {
            u += len;
            if (a_set && b_set) s += len;
        }
    }

    out[(uint)ii * (uint)P.bh + (uint)jj] = (u > 0.0f) ? (1.0f - s / u) : 0.0f;
}

struct WParams {
    int num_nodes, words_per_sample, n;
};

// accumulates the min-sum numerator over one batch of branches. each (k,l) pair is
// touched by exactly one thread, so the += needs no atomic, but the batches must run
// in order against each other.
kernel void unifrac_weighted_diag_accum(
    device const float *emb        [[buffer(0)]],
    device const float *lens_batch [[buffer(1)]],
    device const ulong *masks      [[buffer(2)]],
    constant WParams &P            [[buffer(3)]],
    device float *shared_acc       [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int k = int(gid.x);
    const int d = int(gid.y) + 1;
    if (k >= P.n || d >= P.n) return;
    const int l = k + d;
    if (l >= P.n) return;

    device const ulong *mk = masks + (uint)k * (uint)P.words_per_sample;
    device const ulong *ml = masks + (uint)l * (uint)P.words_per_sample;

    float acc = 0.0f;
    for (int w = 0; w < P.words_per_sample; ++w) {
        ulong common = mk[w] & ml[w];
        // only branches present in both samples can contribute a nonzero min
        while (common != 0) {
            const int r = (w << 6) + int(ctz(common));   // nonzero by the loop guard
            if (r < P.num_nodes) {
                const float len = lens_batch[r];
                if (len > 0.0f) {
                    const uint base = (uint)r * (uint)P.n;
                    const float a = emb[base + (uint)k];
                    const float b = emb[base + (uint)l];
                    const float m = a < b ? a : b;
                    if (m > 0.0f) acc += len * m;
                }
            }
            common &= (common - 1);
        }
    }

    shared_acc[(ulong)k * (ulong)P.n + (ulong)l] += acc;
}

kernel void unifrac_weighted_diag_normalize(
    device const float *sample_sums [[buffer(0)]],
    constant WParams &P             [[buffer(1)]],
    device float *shared_as_dist    [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    const int k = int(gid.x);
    const int d = int(gid.y) + 1;
    if (k >= P.n || d >= P.n) return;
    const int l = k + d;
    if (l >= P.n) return;

    const ulong idx = (ulong)k * (ulong)P.n + (ulong)l;
    const float den = sample_sums[k] + sample_sums[l];
    float val = 0.0f;
    if (den > 0.0f) {
        val = 1.0f - (2.0f * shared_as_dist[idx] / den);
        val = clamp(val, 0.0f, 1.0f);
    }
    shared_as_dist[idx] = val;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct WParams {
    num_nodes: i32,
    words_per_sample: i32,
    n: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    num_nodes: i32,
    words_a: i32,
    words_b: i32,
    n: i32,
    i0: i32,
    j0: i32,
    bw: i32,
    bh: i32,
    only_upper: i32,
}

/// same occupancy rule as the hamming backend: half the thread budget, with x tracking
/// the simd width so a bundle covers a contiguous run of columns.
fn thread_shape(pso: &ComputePipelineState) -> Result<(usize, usize)> {
    let width = pso.thread_execution_width() as usize;
    let max_threads = pso.max_total_threads_per_threadgroup() as usize;
    if width == 0 || max_threads == 0 {
        bail!("metal reported a zero threadgroup limit (width={width}, max={max_threads})");
    }
    let threads = (max_threads / 2).max(width);
    let x = width.min(threads);
    Ok((x, (threads / x).max(1)))
}

/// true if this machine has a metal device we can use.
pub fn is_available() -> bool {
    Device::system_default().is_some()
}

pub fn unifrac_striped_unweighted_metal(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    masks: Vec<BitVec<u8, Lsb0>>,
) -> Result<Vec<f64>> {
    let t_all = Instant::now();
    let nsamp = masks.len();
    if nsamp == 0 {
        return Ok(Vec::new());
    }

    let (node_bits, active_per_strip, blk) =
        build_unweighted_node_bits_and_active(post, kids, lens, leaf_ids, masks)?;
    let nblk = nsamp.div_ceil(blk);

    let mut dist = vec![0.0f64; nsamp * nsamp];
    let out_ptr = DistPtr(unsafe { NonNull::new_unchecked(dist.as_mut_ptr()) });

    let device = Device::system_default().context("no metal device found")?;
    let queue = device.new_command_queue();

    let copts = CompileOptions::new();
    copts.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(SHADER, &copts)
        .map_err(|e| anyhow!("metal shader compile failed: {e}"))?;
    let function = library
        .get_function("unifrac_unweighted_tile", None)
        .map_err(|e| anyhow!("metal function not found: {e}"))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow!("metal pipeline failed: {e}"))?;

    let (tx, ty) = thread_shape(&pso)?;
    // no threadgroup staging here: every thread of a group reads the same bitsA/bitsB
    // word, which the hardware already broadcasts, so staging only adds barriers
    info!(
        "metal(unweighted): {} | nsamp={} blk={} tiles={} threads {}x{}",
        device.name(),
        nsamp,
        blk,
        nblk * (nblk + 1) / 2,
        tx,
        ty
    );

    // a 64x64 tile is far too small to pay for a command buffer on its own, so many
    // tiles share one submission and write into their own slot of a batch buffer.
    let tile_elems = blk * blk;
    let max_slots = ((64 << 20) / (tile_elems * 4)).clamp(1, 512);
    let b_out = device.new_buffer(
        (max_slots * tile_elems * std::mem::size_of::<f32>()) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );

    let all_tiles: Vec<(usize, usize)> = (0..nblk)
        .flat_map(|bi| (bi..nblk).map(move |bj| (bi, bj)))
        .collect();
    info!(
        "metal(unweighted): {} tiles, {} per submission",
        all_tiles.len(),
        max_slots
    );

    let t_gpu = Instant::now();
    let mut tiles = 0usize;

    for chunk in all_tiles.chunks(max_slots) {
        autoreleasepool(|| {
            let cb = queue.new_command_buffer();
            // concurrent: the tiles touch disjoint inputs and outputs, and each one is
            // too small to fill the gpu by itself
            let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
            enc.set_compute_pipeline_state(&pso);

            let mut alive: Vec<(Buffer, Buffer, Buffer)> = Vec::with_capacity(chunk.len());
            let mut placed: Vec<(usize, usize, usize, usize, usize)> =
                Vec::with_capacity(chunk.len());

            for (slot, &(bi, bj)) in chunk.iter().enumerate() {
                let i0 = bi * blk;
                let bw = ((bi + 1) * blk).min(nsamp) - i0;
                let j0 = bj * blk;
                let bh = ((bj + 1) * blk).min(nsamp) - j0;
                if bw == 0 || bh == 0 {
                    continue;
                }

                let nodes_u =
                    merge_union_sorted_usize(&active_per_strip[bi], &active_per_strip[bj]);
                if nodes_u.is_empty() {
                    continue;
                }

                let words_a = bw.div_ceil(64);
                let words_b = bh.div_ceil(64);
                let mut bits_a = vec![0u64; nodes_u.len() * words_a];
                let mut bits_b = vec![0u64; nodes_u.len() * words_b];
                let mut lens_v = vec![0f32; nodes_u.len()];

                for (ni, &v) in nodes_u.iter().enumerate() {
                    lens_v[ni] = lens[v];
                    let raw = node_bits[v].as_raw_slice();
                    extract_words_into(raw, i0, bw, &mut bits_a[ni * words_a..(ni + 1) * words_a]);
                    extract_words_into(raw, j0, bh, &mut bits_b[ni * words_b..(ni + 1) * words_b]);
                }

                let params = Params {
                    num_nodes: nodes_u.len() as i32,
                    words_a: words_a as i32,
                    words_b: words_b as i32,
                    n: nsamp as i32,
                    i0: i0 as i32,
                    j0: j0 as i32,
                    bw: bw as i32,
                    bh: bh as i32,
                    only_upper: i32::from(bi == bj),
                };

                let b_a = device.new_buffer_with_data(
                    bits_a.as_ptr() as *const c_void,
                    std::mem::size_of_val(&bits_a[..]) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                );
                let b_b = device.new_buffer_with_data(
                    bits_b.as_ptr() as *const c_void,
                    std::mem::size_of_val(&bits_b[..]) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                );
                let b_len = device.new_buffer_with_data(
                    lens_v.as_ptr() as *const c_void,
                    std::mem::size_of_val(&lens_v[..]) as NSUInteger,
                    MTLResourceOptions::StorageModeShared,
                );

                enc.set_buffer(0, Some(&b_a), 0);
                enc.set_buffer(1, Some(&b_b), 0);
                enc.set_buffer(2, Some(&b_len), 0);
                enc.set_bytes(
                    3,
                    std::mem::size_of::<Params>() as NSUInteger,
                    &params as *const Params as *const c_void,
                );
                enc.set_buffer(4, Some(&b_out), (slot * tile_elems * 4) as NSUInteger);
                enc.dispatch_threads(
                    MTLSize::new(bh as NSUInteger, bw as NSUInteger, 1),
                    MTLSize::new(tx as NSUInteger, ty as NSUInteger, 1),
                );

                alive.push((b_a, b_b, b_len));
                placed.push((slot, i0, j0, bw, bh));
            }

            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();

            for &(slot, i0, j0, bw, bh) in &placed {
                // safety: this slot holds bw*bh f32 and the submission has completed
                let h_out: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        (b_out.contents() as *const f32).add(slot * tile_elems),
                        bw * bh,
                    )
                };
                unsafe {
                    see_scatter_tile_to_host(out_ptr, h_out, nsamp, i0, j0, bw, bh);
                }
            }
            tiles += placed.len();
            drop(alive);
        });
    }

    info!(
        "metal(unweighted): {} tiles in {} ms, total wall {} ms",
        tiles,
        t_gpu.elapsed().as_millis(),
        t_all.elapsed().as_millis()
    );

    Ok(dist)
}

/// branch-embedding stripe width and branches per accumulation batch. both mirror the
/// cuda defaults so the two backends do the same amount of work per batch.
const EMBED_BLK: usize = 1024;
const BRANCH_BATCH: usize = 2048;

pub fn unifrac_striped_weighted_metal(
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    row2leaf: &[Option<usize>],
    table: InputTable<'_>,
    nsamp: usize,
    col_sums: &[f64],
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

    let parent: Vec<usize> = {
        let mut p = vec![usize::MAX; total];
        for v in 0..total {
            for &c in &kids[v] {
                p[c] = v;
            }
        }
        p
    };

    let blk = EMBED_BLK.min(nsamp).next_power_of_two().clamp(64, 4096);
    let nblk = nsamp.div_ceil(blk);

    let t_stripes = Instant::now();
    let stripes: Vec<Stripe> = (0..nblk)
        .into_par_iter()
        .map(|bi| {
            let s0 = bi * blk;
            let s1 = ((bi + 1) * blk).min(nsamp);
            match &table {
                InputTable::DenseCounts(c) => build_stripe_dense_compact(
                    c, row2leaf, leaf_ids, &parent, col_sums, s0, s1, total,
                ),
                InputTable::Csr {
                    indptr,
                    indices,
                    data,
                } => build_stripe_csr_compact(
                    indptr, indices, data, row2leaf, leaf_ids, &parent, col_sums, s0, s1, total,
                ),
            }
        })
        .collect();
    debug!(
        "metal(weighted): {} embedding stripes in {} ms",
        nblk,
        t_stripes.elapsed().as_millis()
    );

    // sample_sums[s] = sum_v len[v] * p[v,s]; active_counts[v] = samples where p[v,s] > 0
    let mut active_counts = vec![0u32; total];
    let mut sample_sums = vec![0.0f32; nsamp];
    for (bi, stripe) in stripes.iter().enumerate() {
        let s0 = bi * blk;
        let width = stripe.width;
        for (ri, &nid) in stripe.nodes.iter().enumerate() {
            let v = nid as usize;
            let len = lens[v];
            if len <= 0.0 {
                continue;
            }
            let row0 = ri * width;
            for c in 0..width {
                let x = stripe.rows[row0 + c];
                if x > 0.0 {
                    active_counts[v] += 1;
                    sample_sums[s0 + c] += len * x;
                }
            }
        }
    }

    // a branch in only one sample has min(p_i, p_j) == 0 for every pair, so it can be
    // dropped from the numerator; its denominator contribution is already in sample_sums
    let active_nodes: Vec<u32> = active_counts
        .iter()
        .enumerate()
        .filter_map(|(v, &cnt)| (cnt >= 2 && lens[v] > 0.0).then_some(v as u32))
        .collect();

    let device = Device::system_default().context("no metal device found")?;
    let queue = device.new_command_queue();

    let copts = CompileOptions::new();
    copts.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(SHADER, &copts)
        .map_err(|e| anyhow!("metal shader compile failed: {e}"))?;
    let make = |name: &str| -> Result<ComputePipelineState> {
        let f = library
            .get_function(name, None)
            .map_err(|e| anyhow!("metal function '{name}' not found: {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| anyhow!("metal pipeline '{name}' failed: {e}"))
    };
    let pso_accum = make("unifrac_weighted_diag_accum")?;
    let pso_norm = make("unifrac_weighted_diag_normalize")?;

    let (tx, ty) = thread_shape(&pso_accum)?;
    let nbatches = active_nodes.len().div_ceil(BRANCH_BATCH);
    info!(
        "metal(weighted): {} | nsamp={} stripes={} shared-min branches={} batches={} threads {}x{}",
        device.name(),
        nsamp,
        nblk,
        active_nodes.len(),
        nbatches,
        tx,
        ty
    );

    let matrix_elems = nsamp
        .checked_mul(nsamp)
        .context("nsamp*nsamp overflow for the weighted matrix")?;
    let b_shared = device.new_buffer(
        (matrix_elems * std::mem::size_of::<f32>()) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );
    // metal does not promise zeroed buffers, and the kernel accumulates into this
    unsafe {
        std::ptr::write_bytes(b_shared.contents() as *mut u8, 0, matrix_elems * 4);
    }
    let b_sums = device.new_buffer_with_data(
        sample_sums.as_ptr() as *const c_void,
        std::mem::size_of_val(&sample_sums[..]) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );

    let diag_count = nsamp.saturating_sub(1);
    let grid = MTLSize::new(nsamp as NSUInteger, diag_count as NSUInteger, 1);
    let tgroup = MTLSize::new(tx as NSUInteger, ty as NSUInteger, 1);

    let t_batches = Instant::now();
    for (batch_id, chunk) in active_nodes.chunks(BRANCH_BATCH).enumerate() {
        let cur = chunk.len();
        let words_per_sample = cur.div_ceil(64);

        let mut h_emb = vec![0.0f32; cur * nsamp];
        let mut h_masks = vec![0u64; nsamp * words_per_sample];
        let mut h_lens = vec![0.0f32; cur];

        let mut local_of = vec![u32::MAX; total];
        for (local, &nid) in chunk.iter().enumerate() {
            local_of[nid as usize] = local as u32;
            h_lens[local] = lens[nid as usize];
        }

        for (bi, stripe) in stripes.iter().enumerate() {
            let s0 = bi * blk;
            let width = stripe.width;
            for (ri, &nid) in stripe.nodes.iter().enumerate() {
                let local = local_of[nid as usize];
                if local == u32::MAX {
                    continue;
                }
                let local = local as usize;
                let word = local >> 6;
                let bit_mask = 1u64 << (local & 63);
                let src0 = ri * width;
                let dst0 = local * nsamp + s0;
                h_emb[dst0..dst0 + width].copy_from_slice(&stripe.rows[src0..src0 + width]);
                for c in 0..width {
                    if stripe.rows[src0 + c] > 0.0 {
                        h_masks[(s0 + c) * words_per_sample + word] |= bit_mask;
                    }
                }
            }
        }

        let params = WParams {
            num_nodes: cur as i32,
            words_per_sample: words_per_sample as i32,
            n: nsamp as i32,
        };

        autoreleasepool(|| {
            let b_emb = device.new_buffer_with_data(
                h_emb.as_ptr() as *const c_void,
                std::mem::size_of_val(&h_emb[..]) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            let b_lens = device.new_buffer_with_data(
                h_lens.as_ptr() as *const c_void,
                std::mem::size_of_val(&h_lens[..]) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );
            let b_masks = device.new_buffer_with_data(
                h_masks.as_ptr() as *const c_void,
                std::mem::size_of_val(&h_masks[..]) as NSUInteger,
                MTLResourceOptions::StorageModeShared,
            );

            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pso_accum);
            enc.set_buffer(0, Some(&b_emb), 0);
            enc.set_buffer(1, Some(&b_lens), 0);
            enc.set_buffer(2, Some(&b_masks), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<WParams>() as NSUInteger,
                &params as *const WParams as *const c_void,
            );
            enc.set_buffer(4, Some(&b_shared), 0);
            enc.dispatch_threads(grid, tgroup);
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
        });

        if batch_id == 0 || batch_id + 1 == nbatches {
            debug!(
                "metal(weighted): batch {}/{} ({cur} branches, {words_per_sample} words/sample)",
                batch_id + 1,
                nbatches
            );
        }
    }
    debug!(
        "metal(weighted): accumulated {} batches in {} ms",
        nbatches,
        t_batches.elapsed().as_millis()
    );

    let norm_params = WParams {
        num_nodes: 0,
        words_per_sample: 0,
        n: nsamp as i32,
    };
    autoreleasepool(|| {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso_norm);
        enc.set_buffer(0, Some(&b_sums), 0);
        enc.set_bytes(
            1,
            std::mem::size_of::<WParams>() as NSUInteger,
            &norm_params as *const WParams as *const c_void,
        );
        enc.set_buffer(2, Some(&b_shared), 0);
        enc.dispatch_threads(grid, tgroup);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    });

    let mut dist = vec![0.0f64; matrix_elems];
    let out_ptr = DistPtr(unsafe { NonNull::new_unchecked(dist.as_mut_ptr()) });
    // safety: b_shared holds nsamp*nsamp f32 and every dispatch above has completed
    let h_mat: &[f32] =
        unsafe { std::slice::from_raw_parts(b_shared.contents() as *const f32, matrix_elems) };
    unsafe {
        scatter_upper_matrix_to_host(out_ptr, h_mat, nsamp);
    }

    info!(
        "metal(weighted): total wall {} ms",
        t_all.elapsed().as_millis()
    );
    Ok(dist)
}
