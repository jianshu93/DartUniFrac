//! backend-agnostic pieces of the striped unifrac gpu paths.
//!
//! the bitset construction and the tile scatter are pure cpu work with no device api
//! in them, so the cuda and metal backends share them instead of keeping two copies
//! that can drift apart.

use anyhow::Result;
use bitvec::{order::Lsb0, vec::BitVec};
use log::info;
use rayon::prelude::*;
use std::ptr::NonNull;
use std::time::Instant;

// Small helper for raw output ptr

#[derive(Clone, Copy)]
pub(crate) struct DistPtr(pub(crate) NonNull<f64>);

impl DistPtr {
    #[inline]
    pub(crate) fn as_mut_ptr(self) -> *mut f64 {
        self.0.as_ptr()
    }
}

// Safety: we guarantee each GPU worker writes disjoint (i,j) regions.
unsafe impl Send for DistPtr {}
unsafe impl Sync for DistPtr {}

#[inline(always)]
pub(crate) unsafe fn see_scatter_tile_to_host(
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

// Bit helpers (unweighted) 

pub(crate) fn merge_union_sorted_usize(a: &[usize], b: &[usize]) -> Vec<usize> {
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

pub(crate) fn extract_words_into(raw: &[u64], start_bit: usize, len_bits: usize, dst: &mut [u64]) {
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

pub(crate) fn build_unweighted_node_bits_and_active(
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

pub(crate) enum InputTable<'a> {
    DenseCounts(&'a [Vec<f64>]), // rows x nsamp
    Csr {
        indptr: &'a [u32],
        indices: &'a [u32],
        data: &'a [f64],
    },
}

// Scatter helpers



#[inline(always)]
pub(crate) unsafe fn scatter_upper_matrix_to_host(
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


// Weighted stripe building (compact) 

#[derive(Clone)]
pub(crate) struct Stripe {
    pub(crate) nodes: Vec<u32>, // sorted node ids
    pub(crate) rows: Vec<f32>,  // row-major [nrows * width]
    pub(crate) index: Vec<u32>, // len=total, maps node_id -> row index or u32::MAX
    pub(crate) width: usize,    // stripe width
}

#[inline]
pub(crate) fn ensure_row_slot_compact(
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

pub(crate) fn build_stripe_dense_compact(
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

pub(crate) fn build_stripe_csr_compact(
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

pub(crate) fn sort_stripe_compact(
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
pub(crate) fn merge_union_u32_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
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
pub(crate) fn merge_intersection_u32_into(a: &[u32], b: &[u32], out: &mut Vec<u32>) {
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
