//! metal backend for the pairwise hamming step (apple silicon).
//!
//! same tiling idea as `disthamming_gpu.rs`: a threadgroup stages a slab of rows and
//! columns into threadgroup memory once, then every thread in the group reuses it.
//! differences from the cuda path:
//!   - one gpu, so none of the multi-device work splitting
//!   - unified memory, so the kernel writes straight into the result matrix
//!   - tile geometry is derived from device limits instead of being hardcoded

use anyhow::{Context, Result, anyhow, bail};
use log::{debug, info};
use metal::objc::rc::autoreleasepool;
use metal::{
    CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize, NSUInteger,
};
use rayon::prelude::*;
use std::ffi::c_void;
use std::time::Instant;

/// every staged slot is 8 bytes regardless of element width, so the threadgroup
/// memory arithmetic is the same for u16/u32/u64.
const SLOT_BYTES: usize = 8;

const SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Params {
    int n, k;
    int i0, j0, bw, bh;
    int only_upper, weighted;
    int bk, stride;
};

inline uint lane_diff(ushort4 a, ushort4 b) {
    ushort4 ne = select(ushort4(0), ushort4(1), a != b);
    return uint(ne.x) + uint(ne.y) + uint(ne.z) + uint(ne.w);
}
inline uint lane_diff(uint2 a, uint2 b) {
    return uint(a.x != b.x) + uint(a.y != b.y);
}
inline uint lane_diff(ulong a, ulong b) {
    return uint(a != b);
}

inline ushort4 load_slot(device const ushort *p) { return ushort4(p[0], p[1], p[2], p[3]); }
inline uint2   load_slot(device const uint   *p) { return uint2(p[0], p[1]); }
inline ulong   load_slot(device const ulong  *p) { return p[0]; }

template <typename SLOT, typename ELEM, int LANES>
inline void hamming_impl(device const ELEM *sketches,
                         constant Params &P,
                         device float *out,
                         threadgroup SLOT *smem,
                         uint2 tgpos, uint2 tpt, uint2 tgdim)
{
    const int n = P.n, k = P.k, stride = P.stride, BK = P.bk;

    const int jj = int(tgpos.x * tgdim.x + tpt.x);
    const int ii = int(tgpos.y * tgdim.y + tpt.y);
    const int i = P.i0 + ii;
    const int j = P.j0 + jj;

    const int nslot = k / LANES;
    const int krem  = k - nslot * LANES;

    threadgroup SLOT *As = smem;
    threadgroup SLOT *Bs = As + int(tgdim.y) * stride;

    const int tpb = int(tgdim.x * tgdim.y);
    const int tid = int(tpt.y * tgdim.x + tpt.x);

    const bool inrange = (ii < P.bw) && (jj < P.bh);
    const bool active  = inrange && !(P.only_upper && j <= i) && (i != j);

    uint diff = 0u;

    for (int t0 = 0; t0 < nslot; t0 += BK) {
        const int bk = min(BK, nslot - t0);

        const int rowbase = P.i0 + int(tgpos.y * tgdim.y);
        for (int idx = tid; idx < int(tgdim.y) * bk; idx += tpb) {
            const int r = idx / bk, t = idx - r * bk;
            const int gi = rowbase + r;
            SLOT v = SLOT(0);
            if (gi < n && (gi - P.i0) < P.bw)
                v = load_slot(sketches + (uint)gi * (uint)k + (uint)((t0 + t) * LANES));
            As[r * stride + t] = v;
        }

        const int colbase = P.j0 + int(tgpos.x * tgdim.x);
        for (int idx = tid; idx < int(tgdim.x) * bk; idx += tpb) {
            const int c = idx / bk, t = idx - c * bk;
            const int gj = colbase + c;
            SLOT v = SLOT(0);
            if (gj < n && (gj - P.j0) < P.bh)
                v = load_slot(sketches + (uint)gj * (uint)k + (uint)((t0 + t) * LANES));
            Bs[c * stride + t] = v;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
            const int arow = int(tpt.y) * stride;
            const int brow = int(tpt.x) * stride;
            for (int t = 0; t < bk; ++t)
                diff += lane_diff(As[arow + t], Bs[brow + t]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // tail when k is not a multiple of LANES, read straight from device memory
    if (active && krem > 0) {
        device const ELEM *a = sketches + (uint)i * (uint)k + (uint)(nslot * LANES);
        device const ELEM *b = sketches + (uint)j * (uint)k + (uint)(nslot * LANES);
        for (int t = 0; t < krem; ++t) diff += uint(a[t] != b[t]);
    }

    if (inrange && !(P.only_upper && j <= i)) {
        float d = (i == j) ? 0.0f : (float(diff) / float(k));
        if (P.weighted) d = (d < 2.0f) ? (d / (2.0f - d)) : 1.0f;
        out[(uint)i * (uint)n + (uint)j] = d;
    }
}

#define HAMMING_KERNEL(NAME, SLOT, ELEM, LANES)                      \
kernel void NAME(device const ELEM *sketches [[buffer(0)]],          \
                 constant Params &P           [[buffer(1)]],         \
                 device float *out            [[buffer(2)]],         \
                 threadgroup SLOT *smem       [[threadgroup(0)]],    \
                 uint2 tgpos [[threadgroup_position_in_grid]],       \
                 uint2 tpt   [[thread_position_in_threadgroup]],     \
                 uint2 tgdim [[threads_per_threadgroup]])            \
{ hamming_impl<SLOT, ELEM, LANES>(sketches, P, out, smem, tgpos, tpt, tgdim); }

HAMMING_KERNEL(hamming_u16, ushort4, ushort, 4)
HAMMING_KERNEL(hamming_u32, uint2,   uint,   2)
HAMMING_KERNEL(hamming_u64, ulong,   ulong,  1)
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    n: i32,
    k: i32,
    i0: i32,
    j0: i32,
    bw: i32,
    bh: i32,
    only_upper: i32,
    weighted: i32,
    bk: i32,
    stride: i32,
}

trait MetalElem: Copy + Send + Sync {
    const KERNEL: &'static str;
}
impl MetalElem for u16 {
    const KERNEL: &'static str = "hamming_u16";
}
impl MetalElem for u32 {
    const KERNEL: &'static str = "hamming_u32";
}
impl MetalElem for u64 {
    const KERNEL: &'static str = "hamming_u64";
}

#[derive(Clone, Copy, Debug)]
struct Tile {
    x: usize,
    y: usize,
    bk: usize,
    stride: usize,
    smem: usize,
}

/// pick the tile from what the device reports, so this adapts to other apple gpus.
///
/// two rules, both measured on an m4 max: spend only half the thread budget and half
/// the threadgroup memory, so a second threadgroup stays resident to hide the first
/// one's memory stalls. maximising per-tile reuse instead loses ~8% -- occupancy wins.
fn choose_tile(pso: &ComputePipelineState, max_smem: usize) -> Result<Tile> {
    let width = pso.thread_execution_width() as usize;
    let max_threads = pso.max_total_threads_per_threadgroup() as usize;
    if width == 0 || max_threads == 0 {
        bail!("metal reported a zero threadgroup limit (width={width}, max={max_threads})");
    }

    let threads = (max_threads / 2).max(width);
    // one simd group spans a contiguous run of columns, so x tracks the simd width
    let x = width.min(threads);
    let y = (threads / x).max(1);

    let budget = max_smem / 2;
    let mut bk = 1usize;
    while bk < 256 && (2 * bk + 1) * (x + y) * SLOT_BYTES <= budget {
        bk *= 2;
    }

    let stride = bk + 1;
    let smem = stride * (x + y) * SLOT_BYTES;
    if smem > max_smem {
        bail!("tile needs {smem} B of threadgroup memory, device allows {max_smem} B");
    }
    Ok(Tile { x, y, bk, stride, smem })
}

/// true if this machine has a metal device we can use.
pub fn is_available() -> bool {
    Device::system_default().is_some()
}

/// name of the metal device, for logging.
pub fn device_name() -> Option<String> {
    Device::system_default().map(|d| d.name().to_string())
}

fn pairwise_hamming_metal<E: MetalElem>(
    sketches: &[E],
    n: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    weighted: bool,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    if sketches.len() != n * k {
        bail!("sketches has {} elements, expected n*k = {}", sketches.len(), n * k);
    }
    if out.len() != n * n {
        bail!("out has {} elements, expected n*n = {}", out.len(), n * n);
    }

    let device = Device::system_default().context("no metal device found")?;
    let queue = device.new_command_queue();

    let t_compile = Instant::now();
    // fast math is on by default and makes division an approximate reciprocal, which
    // perturbs the weighted transform. correctness here means matching the cpu path
    // bit for bit, and the kernel is integer-bound anyway, so turn it off.
    let opts = CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let library = device
        .new_library_with_source(SHADER, &opts)
        .map_err(|e| anyhow!("metal shader compile failed: {e}"))?;
    let function = library
        .get_function(E::KERNEL, None)
        .map_err(|e| anyhow!("metal function '{}' not found: {e}", E::KERNEL))?;
    let pso = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow!("metal pipeline for '{}' failed: {e}", E::KERNEL))?;

    let max_smem = device.max_threadgroup_memory_length() as usize;
    let tile = choose_tile(&pso, max_smem)?;
    info!(
        "metal: {} | tile {}x{} ({} threads), slab {}, threadgroup mem {}/{} B | pipeline ready in {} ms",
        device.name(),
        tile.x,
        tile.y,
        tile.x * tile.y,
        tile.bk,
        tile.smem,
        max_smem,
        t_compile.elapsed().as_millis()
    );

    // sketches are copied in; the result buffer is shared, so the kernel writes
    // into memory the cpu reads directly afterwards.
    let sk_bytes = std::mem::size_of_val(sketches) as NSUInteger;
    let b_sketches = device.new_buffer_with_data(
        sketches.as_ptr() as *const c_void,
        sk_bytes,
        MTLResourceOptions::StorageModeShared,
    );
    let b_out = device.new_buffer(
        (n * n * std::mem::size_of::<f32>()) as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );

    let block_rows = block_rows.max(1).min(n);
    let nb = n.div_ceil(block_rows);

    let t_gpu = Instant::now();
    // one command buffer per tile keeps any single gpu submission short; macos will
    // kill a command buffer that runs too long.
    for bi in 0..nb {
        let i0 = bi * block_rows;
        let bw = (i0 + block_rows).min(n) - i0;

        for bj in bi..nb {
            let j0 = bj * block_rows;
            let bh = (j0 + block_rows).min(n) - j0;

            let params = Params {
                n: n as i32,
                k: k as i32,
                i0: i0 as i32,
                j0: j0 as i32,
                bw: bw as i32,
                bh: bh as i32,
                only_upper: 1,
                weighted: weighted as i32,
                bk: tile.bk as i32,
                stride: tile.stride as i32,
            };

            autoreleasepool(|| {
                let cb = queue.new_command_buffer();
                let enc = cb.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&pso);
                enc.set_buffer(0, Some(&b_sketches), 0);
                enc.set_bytes(
                    1,
                    std::mem::size_of::<Params>() as NSUInteger,
                    &params as *const Params as *const c_void,
                );
                enc.set_buffer(2, Some(&b_out), 0);
                enc.set_threadgroup_memory_length(0, tile.smem as NSUInteger);
                enc.dispatch_thread_groups(
                    MTLSize::new(
                        bh.div_ceil(tile.x) as NSUInteger,
                        bw.div_ceil(tile.y) as NSUInteger,
                        1,
                    ),
                    MTLSize::new(tile.x as NSUInteger, tile.y as NSUInteger, 1),
                );
                enc.end_encoding();
                cb.commit();
                cb.wait_until_completed();
            });

            debug!("metal: tile bi={bi} bj={bj} i0={i0} j0={j0} bw={bw} bh={bh}");
        }
    }
    info!("metal: kernels done in {} ms", t_gpu.elapsed().as_millis());

    // the kernel filled the upper triangle; mirror it into the caller's matrix.
    // safety: b_out is n*n f32 and the gpu work above has completed.
    let upper: &[f32] =
        unsafe { std::slice::from_raw_parts(b_out.contents() as *const f32, n * n) };

    // two regions, so neither loop branches per element: the upper half of a row is
    // contiguous in `upper` and copies flat, while the lower half reads down a column
    // and is blocked so each cache line gets used more than once.
    const MB: usize = 64;
    let t_mirror = Instant::now();
    out.par_chunks_mut(n * MB).enumerate().for_each(|(band, rows)| {
        let i0 = band * MB;
        let nrows = rows.len() / n;

        for (ii, row) in rows.chunks_mut(n).enumerate() {
            let i = i0 + ii;
            row[i] = 0.0;
            if i + 1 < n {
                row[i + 1..].copy_from_slice(&upper[i * n + i + 1..(i + 1) * n]);
            }
        }

        for j0 in (0..(i0 + nrows).min(n)).step_by(MB) {
            let j1 = (j0 + MB).min(n);
            for (ii, row) in rows.chunks_mut(n).enumerate() {
                let i = i0 + ii;
                for j in j0..j1.min(i) {
                    row[j] = upper[j * n + i];
                }
            }
        }
    });
    debug!("metal: mirrored in {} ms", t_mirror.elapsed().as_millis());

    Ok(())
}

pub fn pairwise_hamming_metal_u16(
    sketches: &[u16],
    n: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    weighted: bool,
) -> Result<()> {
    pairwise_hamming_metal::<u16>(sketches, n, k, out, block_rows, weighted)
}

pub fn pairwise_hamming_metal_u32(
    sketches: &[u32],
    n: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    weighted: bool,
) -> Result<()> {
    pairwise_hamming_metal::<u32>(sketches, n, k, out, block_rows, weighted)
}

pub fn pairwise_hamming_metal_u64(
    sketches: &[u64],
    n: usize,
    k: usize,
    out: &mut [f32],
    block_rows: usize,
    weighted: bool,
) -> Result<()> {
    pairwise_hamming_metal::<u64>(sketches, n, k, out, block_rows, weighted)
}
