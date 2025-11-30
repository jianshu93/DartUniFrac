#![cfg_attr(feature = "stdsimd", feature(portable_simd))]

//! BSD 3-Clause License
//!
//! Copyright (c) 2016-2025, UniFrac development team.
//! All rights reserved.
//!
//! See LICENSE file for more details

//! succinct-BP UniFrac
//!  * --striped – single post-order pass, **parallel blocks**
//!      (works for tens-of-thousands samples)

use anyhow::{Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use clap::ArgAction;
use clap::ArgGroup;
use clap::{Arg, Command};
use env_logger;
use hdf5::{types::VarLenUnicode, File as H5File};
use log::info;
use newick::{one_from_string, Newick, NodeID};
use rayon::prelude::*;
use std::ptr::NonNull;
use std::time::Instant;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    sync::Arc,
};
use succparen::tree::Node;
use succparen::{
    bitwise::{ops::NndOne, SparseOneNnd},
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec,
    },
};

#[cfg(feature = "stdsimd")]
use std::simd::{LaneCount, Simd, SupportedLaneCount};


// Plain new-type – automatically `Copy`.
#[derive(Clone, Copy)]
struct DistPtr(NonNull<f64>);

// We guarantee in the algorithm that every thread writes a
// *disjoint* rectangle of the matrix.  No two threads touch the same
// element → pointer can be shared.
unsafe impl Send for DistPtr {}
unsafe impl Sync for DistPtr {}

type NwkTree = newick::NewickTree;

// Newick sanitization: drop internal labels and comments
fn sanitize_newick_drop_internal_labels_and_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'[' => {
                // Skip bracket comments (tolerate nested)
                i += 1;
                let mut depth = 1;
                while i < bytes.len() && depth > 0 {
                    match bytes[i] {
                        b'[' => depth += 1,
                        b']' => depth -= 1,
                        _ => {}
                    }
                    i += 1;
                }
            }
            b')' => {
                out.push(')');
                i += 1;
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }
                // Optional internal label after ')'
                if i < bytes.len() && bytes[i] == b'\'' {
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' && i + 1 < bytes.len() {
                            i += 2;
                            continue;
                        }
                        if bytes[i] == b'\'' {
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                } else {
                    while i < bytes.len() {
                        let c = bytes[i];
                        if c.is_ascii_whitespace()
                            || matches!(c, b':' | b',' | b')' | b'(' | b';' | b'[')
                        {
                            break;
                        }
                        i += 1;
                    }
                }
            }
            _ => {
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    out
}

// Succinct traversal to collect branch lengths into `lens`
struct SuccTrav<'a> {
    t: &'a NwkTree,
    stack: Vec<(NodeID, usize, usize)>,
    lens: &'a mut Vec<f32>,
}

impl<'a> SuccTrav<'a> {
    fn new(t: &'a NwkTree, lens: &'a mut Vec<f32>) -> Self {
        Self {
            t,
            stack: vec![(t.root(), 0, 0)],
            lens,
        }
    }
}

impl<'a> DepthFirstTraverse for SuccTrav<'a> {
    type Label = ();

    fn next(&mut self) -> Option<VisitNode<Self::Label>> {
        let (id, lvl, nth) = self.stack.pop()?;
        let n_children = self.t[id].children().len();
        for (k, &c) in self.t[id].children().iter().enumerate().rev() {
            let nth = n_children - 1 - k;
            self.stack.push((c, lvl + 1, nth));
        }
        if self.lens.len() <= id {
            self.lens.resize(id + 1, 0.0);
        }
        self.lens[id] = self.t[id].branch().copied().unwrap_or(0.0);
        Some(VisitNode::new((), lvl, nth))
    }
}

fn collect_children<N: NndOne>(
    node: &BpNode<LabelVec<()>, N, &BalancedParensTree<LabelVec<()>, N>>,
    kids: &mut [Vec<usize>],
    post: &mut Vec<usize>,
) {
    let pid = node.id() as usize;
    for edge in node.children() {
        let cid = edge.node.id() as usize;
        kids[pid].push(cid);
        collect_children(&edge.node, kids, post);
    }
    post.push(pid);
}

// TSV readers
fn read_table(p: &str) -> Result<(Vec<String>, Vec<String>, Vec<Vec<f64>>)> {
    let f = File::open(p)?;
    let mut lines = BufReader::new(f).lines();
    let hdr = lines.next().context("empty table")??;
    let mut it = hdr.split('\t');
    it.next();
    let samples = it.map(|s| s.to_owned()).collect();

    let mut taxa = Vec::new();
    let mut mat = Vec::new();
    for l in lines {
        let row = l?;
        let mut p = row.split('\t');
        let tax = p.next().unwrap().to_owned();
        let vals = p
            .map(|v| if v.parse::<f64>().unwrap_or(0.0) > 0.0 { 1.0 } else { 0.0 })
            .collect();
        taxa.push(tax);
        mat.push(vals);
    }
    Ok((taxa, samples, mat))
}
fn read_table_counts(p: &str) -> Result<(Vec<String>, Vec<String>, Vec<Vec<f64>>)> {
    let f = File::open(p)?;
    let mut lines = BufReader::new(f).lines();
    let hdr = lines.next().context("empty table")??;
    let mut it = hdr.split('\t');
    it.next();
    let samples: Vec<String> = it.map(|s| s.to_owned()).collect();

    let mut taxa = Vec::new();
    let mut mat = Vec::new();
    for l in lines {
        let row = l?;
        let mut p = row.split('\t');
        let tax = p.next().unwrap().to_owned();
        let vals = p.map(|v| v.parse::<f64>().unwrap_or(0.0)).collect::<Vec<f64>>();
        taxa.push(tax);
        mat.push(vals);
    }
    Ok((taxa, samples, mat))
}

// Output writer
fn write_matrix(names: &[String], d: &[f64], n: usize, path: &str) -> Result<()> {
    // build header
    let header = {
        let mut s = String::with_capacity(n * 16);
        s.push_str("Sample");
        for name in names {
            s.push('\t');
            s.push_str(name);
        }
        s.push('\n');
        s
    };

    // rows in parallel
    let mut rows: Vec<String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut line = String::with_capacity(n * 12);
            line.push_str(&names[i]);
            let base = i * n;
            for j in 0..n {
                let val = unsafe { *d.get_unchecked(base + j) };
                line.push('\t');
                line.push_str(ryu::Buffer::new().format_finite(val));
            }
            line.push('\n');
            line
        })
        .collect();

    let mut out = BufWriter::with_capacity(16 << 20, File::create(path)?);
    out.write_all(header.as_bytes())?;
    for line in &mut rows {
        out.write_all(line.as_bytes())?;
        line.clear();
    }
    out.flush()?;
    Ok(())
}

// Optional instrumentation (unweighted)
fn log_relevant_branch_counts_from_bits(
    node_bits: &[bitvec::vec::BitVec<u64, Lsb0>],
    lens: &[f32],
) {
    if node_bits.is_empty() {
        return;
    }
    let total = lens.len();
    let nsamp = node_bits[0].len();
    let total_branches = lens.iter().filter(|&&l| l > 0.0).count();
    let mut rel_counts = vec![0u32; nsamp];

    for v in 0..total {
        if lens[v] <= 0.0 {
            continue;
        }
        let words = node_bits[v].as_raw_slice();
        for (wi, &w0) in words.iter().enumerate() {
            let mut w = w0;
            while w != 0 {
                let b = w.trailing_zeros() as usize;
                let s = (wi << 6) + b;
                if s < nsamp {
                    rel_counts[s] += 1;
                }
                w &= w - 1;
            }
        }
    }
    for s in 0..nsamp {
        let cnt = rel_counts[s] as usize;
        let frac = (cnt as f64) / (total_branches as f64);
        log::info!(
            "sample {}: relevant branches = {} / {} = {}",
            s,
            rel_counts[s],
            total_branches,
            frac
        );
    }
}

// Unweighted striped pass
fn unifrac_striped_par(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    mut masks: Vec<BitVec<u8, Lsb0>>, // take ownership
) -> Vec<f64> {
    // constants
    let nsamp = masks.len();
    let total = lens.len();
    let n_threads = rayon::current_num_threads().max(1);

    // Partition samples into stripes by thread
    let stripe = (nsamp + n_threads - 1) / n_threads;
    let words_str = (stripe + 63) >> 6;

    // node_masks[tid][node][word]
    let mut node_masks: Vec<Vec<Vec<u64>>> = (0..n_threads)
        .map(|_| vec![vec![0u64; words_str]; total])
        .collect();

    // Phase 1 : build stripes
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
                // scatter leaf bits
                for (local_s, sm) in masks_slice.iter().enumerate() {
                    for pos in sm.iter_ones() {
                        let v = leaf[pos];
                        let w = local_s >> 6;
                        let b = local_s & 63;
                        node_masks_t[v][w] |= 1u64 << b;
                    }
                }
                // bottom-up OR inside the stripe
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
    log::info!("phase-1 masks built {:>6} ms", t0.elapsed().as_millis());

    masks.clear();
    masks.shrink_to_fit();

    // Merge stripes into one BitVec per node (node_bits)
    let mut node_bits: Vec<BitVec<u64, Lsb0>> =
        (0..total).map(|_| BitVec::repeat(false, nsamp)).collect();

    node_bits.par_iter_mut().enumerate().for_each(|(v, bv)| {
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

            let n_src_words = src_words.len();
            for w in 0..n_src_words {
                let mut val = src_words[w];
                if w == n_src_words - 1 {
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

    if log::log_enabled!(log::Level::Info) {
        log_relevant_branch_counts_from_bits(&node_bits, lens);
    }

    // Phase 2 : active nodes per matrix strip
    let n_threads = rayon::current_num_threads().max(1);
    let est_blk = ((nsamp as f64 / (2.0 * n_threads as f64)).sqrt()) as usize;
    let blk = est_blk.clamp(64, 512).next_power_of_two();
    let nblk = (nsamp + blk - 1) / blk;

    let mut active_per_strip: Vec<Vec<usize>> = vec![Vec::new(); nblk];

    for v in 0..total {
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
    log::info!("phase-2 sparse lists built ({} strips)", nblk);

    // Phase 3 : block sweep (upper triangle)
    let dist = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let ptr = DistPtr(unsafe { NonNull::new_unchecked(dist.as_ptr() as *mut f64) });

    let pairs: Vec<(usize, usize)> = (0..nblk)
        .flat_map(|bi| (bi..nblk).map(move |bj| (bi, bj)))
        .collect();

    let t3 = Instant::now();
    pairs.into_par_iter().for_each(|(bi, bj)| {
        let ptr = ptr;
        let (i0, i1) = (bi * blk, ((bi + 1) * blk).min(nsamp));
        let (j0, j1) = (bj * blk, ((bj + 1) * blk).min(nsamp));
        let bw = i1 - i0;
        let bh = j1 - j0;

        let mut union = vec![0.0f64; bw * bh];
        let mut shared = vec![0.0f64; bw * bh];

        let list_a = &active_per_strip[bi];
        let list_b = &active_per_strip[bj];
        let mut ia = 0;
        let mut ib = 0;

        while ia < list_a.len() || ib < list_b.len() {
            let v = match (list_a.get(ia), list_b.get(ib)) {
                (Some(&va), Some(&vb)) => {
                    if va < vb {
                        ia += 1;
                        va
                    } else if vb < va {
                        ib += 1;
                        vb
                    } else {
                        ia += 1;
                        ib += 1;
                        va
                    }
                }
                (Some(&va), None) => {
                    ia += 1;
                    va
                }
                (None, Some(&vb)) => {
                    ib += 1;
                    vb
                }
                _ => unreachable!(),
            };

            let len = lens[v] as f64;
            let words = node_bits[v].as_raw_slice();

            for ii in 0..bw {
                let samp_i = i0 + ii;
                let word_i = words[samp_i >> 6];
                let bit_i = 1u64 << (samp_i & 63);
                let a_set = (word_i & bit_i) != 0;

                for jj in 0..bh {
                    let samp_j = j0 + jj;
                    if samp_j <= samp_i {
                        continue;
                    }
                    let word_j = words[samp_j >> 6];
                    let bit_j = 1u64 << (samp_j & 63);
                    let b_set = (word_j & bit_j) != 0;
                    if !a_set && !b_set {
                        continue;
                    }
                    let idx = ii * bh + jj;
                    union[idx] += len;
                    if a_set && b_set {
                        shared[idx] += len;
                    }
                }
            }
        }

        unsafe {
            let base = ptr.0.as_ptr();
            for ii in 0..bw {
                let i = i0 + ii;
                for jj in 0..bh {
                    let j = j0 + jj;
                    if j <= i {
                        continue;
                    }
                    let idx = ii * bh + jj;
                    let u = union[idx];
                    if u == 0.0 {
                        continue;
                    }
                    let d = 1.0 - shared[idx] / u;
                    *base.add(i * nsamp + j) = d;
                    *base.add(j * nsamp + i) = d;
                }
            }
        }
    });

    drop(active_per_strip);
    drop(node_bits);
    info!("phase-3 block pass {:>6} ms", t3.elapsed().as_millis());

    Arc::try_unwrap(dist).unwrap()
}

// Weighted / Generalized infrastructure

#[derive(Clone, Copy)]
enum WeightedMode<'a> {
    Dense { counts: &'a [Vec<f64>] }, // rows x nsamp
    Csr {
        indptr: &'a [u32],
        indices: &'a [u32],
        data: &'a [f64],
    }, // BIOM CSR
}

/// Sparse stripe: only nodes that are non-zero in [s0..s1)
/// `nz` lists local non-zero columns per row.
struct Stripe {
    nodes: Vec<usize>,
    rows: Vec<Vec<f32>>,   // rows[k] length == bw
    index: Vec<u32>,       // node-id -> row index (u32::MAX if absent)
    nz: Vec<Vec<usize>>,   // rows[k] -> sorted list of local jj with >0
}

fn ensure_row_slot<'a>(
    v: usize,
    idx_of: &mut [u32],
    nodes: &mut Vec<usize>,
    rows: &'a mut Vec<Vec<f32>>,
    nz: &mut Vec<Vec<usize>>,      // <- NOT &'a
    bw: usize,
) -> (usize, &'a mut [f32]) {      // <- return (row_idx, row_slice)
    let idx = idx_of[v];
    if idx != u32::MAX {
        let i = idx as usize;
        return (i, rows[i].as_mut_slice());
    }
    let new_idx = rows.len() as u32;
    idx_of[v] = new_idx;
    nodes.push(v);
    rows.push(vec![0f32; bw]);
    nz.push(Vec::new());
    let i = new_idx as usize;
    (i, rows[i].as_mut_slice())
}

fn build_stripe_dense(
    counts: &[Vec<f64>],
    row2leaf: &[Option<usize>],
    leaf_ids: &[usize],
    parent: &[usize],
    col_sums: &[f64],
    s0: usize,
    s1: usize,
    total: usize,
    raw_counts: bool,
) -> Stripe {
    let bw = s1 - s0;
    let mut idx_of = vec![u32::MAX; total];
    let mut nodes: Vec<usize> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut nz: Vec<Vec<usize>> = Vec::new();

    for (r, lopt) in row2leaf.iter().enumerate() {
        let Some(lp) = *lopt else { continue };
        let v_leaf = leaf_ids[lp];

        for s in s0..s1 {
            let val = counts[r][s];
            if val <= 0.0 {
                continue;
            }

            // choose between raw counts and normalized
            let inc: f32 = if raw_counts {
                val as f32
            } else {
                let denom = col_sums[s];
                if denom <= 0.0 {
                    continue;
                }
                (val / denom) as f32
            };

            // leaf to root accumulation
            let mut v = v_leaf;
            loop {
                let (row_idx, row) =
                    ensure_row_slot(v, &mut idx_of, &mut nodes, &mut rows, &mut nz, bw);
                let off = s - s0;
                if row[off] == 0.0 {
                    nz[row_idx].push(off);
                }
                row[off] += inc;
                let p = parent[v];
                if p == usize::MAX {
                    break;
                }
                v = p;
            }
        }
    }
    Stripe { nodes, rows, index: idx_of, nz }
}

fn build_stripe_csr(
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
    raw_counts: bool,
) -> Stripe {
    let bw = s1 - s0;
    let mut idx_of = vec![u32::MAX; total];
    let mut nodes: Vec<usize> = Vec::new();
    let mut rows: Vec<Vec<f32>> = Vec::new();
    let mut nz: Vec<Vec<usize>> = Vec::new();

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

            let val = data[k];
            if val <= 0.0 {
                continue;
            }

            // choose between raw counts and normalized
            let inc: f32 = if raw_counts {
                val as f32
            } else {
                let denom = col_sums[s];
                if denom <= 0.0 {
                    continue;
                }
                (val / denom) as f32
            };

            let mut v = v_leaf;
            loop {
                let (row_idx, row) =
                    ensure_row_slot(v, &mut idx_of, &mut nodes, &mut rows, &mut nz, bw);
                let off = s - s0;
                if row[off] == 0.0 {
                    nz[row_idx].push(off);
                }
                row[off] += inc;
                let p = parent[v];
                if p == usize::MAX {
                    break;
                }
                v = p;
            }
        }
    }
    Stripe { nodes, rows, index: idx_of, nz }
}

#[cfg(feature = "stdsimd")]
#[inline]
fn add_const_to_row_simd<const LANES: usize>(buf: &mut [f64], add: f64, start: usize)
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let addv = Simd::<f64, LANES>::splat(add);
    let mut j = start;
    while j + LANES <= buf.len() {
        let mut v = Simd::<f64, LANES>::from_slice(&buf[j..j + LANES]);
        v += addv;
        // stable portable_simd: store via to_array + copy_from_slice
        buf[j..j + LANES].copy_from_slice(&v.to_array());
        j += LANES;
    }
    while j < buf.len() {
        buf[j] += add;
        j += 1;
    }
}

#[cfg(not(feature = "stdsimd"))]
#[inline]
fn add_const_to_row_simd<const LANES: usize>(buf: &mut [f64], add: f64, start: usize) {
    for j in start..buf.len() {
        buf[j] += add;
    }
}

fn unifrac_striped_par_weighted(
    _post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    row2leaf: &[Option<usize>],
    mode: WeightedMode,
    nsamp: usize,
    col_sums: &[f64],
    raw_counts: bool,
) -> Vec<f64> {
    // parent[]
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

    // block geometry
    let n_threads = rayon::current_num_threads().max(1);
    let est_blk = ((nsamp as f64 / (2.0 * n_threads as f64)).sqrt()) as usize;
    let blk = est_blk.clamp(64, 512).next_power_of_two();
    let nblk = (nsamp + blk - 1) / blk;

    let dist = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let base_addr: usize = dist.as_ptr() as usize;

    for bi in 0..nblk {
        let i0 = bi * blk;
        let i1 = (i0 + blk).min(nsamp);
        let bw = i1 - i0;

        let stripe_i = match &mode {
            WeightedMode::Dense { counts } => build_stripe_dense(
                counts, row2leaf, leaf_ids, &parent, col_sums, i0, i1, total, raw_counts // NEW
            ),
            WeightedMode::Csr { indptr, indices, data } => build_stripe_csr(
                indptr, indices, data, row2leaf, leaf_ids, &parent, col_sums, i0, i1, total, raw_counts // NEW
            ),
        };

        let mode_c = mode;
        let parent_ref = &parent;
        let row2leaf_ref = row2leaf;
        let leaf_ids_ref = leaf_ids;
        let col_sums_ref = col_sums;
        let stripe_i_ref = &stripe_i;

        (bi..nblk).into_par_iter().for_each(move |bj| {
            let (j0, j1) = (bj * blk, ((bj + 1) * blk).min(nsamp));
            let bh = j1 - j0;
            let diagonal_block = bj == bi;

            let stripe_j = if diagonal_block {
                Stripe {
                    nodes: stripe_i_ref.nodes.clone(),
                    rows: stripe_i_ref.rows.clone(),
                    index: stripe_i_ref.index.clone(),
                    nz: stripe_i_ref.nz.clone(),
                }
            } else {
                match mode_c {
                    WeightedMode::Dense { counts } => build_stripe_dense(
                        counts, row2leaf_ref, leaf_ids_ref, parent_ref, col_sums_ref, j0, j1, total, raw_counts
                    ),
                    WeightedMode::Csr { indptr, indices, data } => build_stripe_csr(
                        indptr, indices, data, row2leaf_ref, leaf_ids_ref, parent_ref, col_sums_ref, j0, j1, total, raw_counts
                    ),
                }
            };

            // num = Σ_i ℓ_i |a_i - b_i|,  den = Σ_i ℓ_i (a_i + b_i)
            let mut num = vec![0.0f64; bw * bh];
            let mut den = vec![0.0f64; bw * bh];

            // (A) row baseline: add len*ai across row (to both num & den)
            for &v in &stripe_i_ref.nodes {
                let len = lens[v] as f64;
                if len <= 0.0 { continue; }
                let ri = stripe_i_ref.index[v] as usize;
                let ai = &stripe_i_ref.rows[ri];
                let nz_i = &stripe_i_ref.nz[ri];

                for &ii in nz_i {
                    let a = len * (ai[ii] as f64);
                    if a == 0.0 { continue; }
                    let start_j = if diagonal_block { ii + 1 } else { 0 };
                    let row_n = &mut num[ii * bh..(ii + 1) * bh];
                    let row_d = &mut den[ii * bh..(ii + 1) * bh];
                    add_const_to_row_simd::<8>(row_n, a, start_j);
                    add_const_to_row_simd::<8>(row_d, a, start_j);
                }
            }

            // (B) column baseline: add len*aj down column (to both num & den)
            for &v in &stripe_j.nodes {
                let len = lens[v] as f64;
                if len <= 0.0 { continue; }
                let rj = stripe_j.index[v] as usize;
                let aj = &stripe_j.rows[rj];
                let nz_j = &stripe_j.nz[rj];

                for &jj in nz_j {
                    let add = len * (aj[jj] as f64);
                    if add == 0.0 { continue; }
                    let ii_end = if diagonal_block { ((j0 + jj).saturating_sub(i0)).min(bw) } else { bw };
                    let mut ii = 0usize;
                    while ii < ii_end {
                        let idx = ii * bh + jj;
                        num[idx] += add;
                        den[idx] += add;
                        ii += 1;
                    }
                }
            }

            // (C) intersection correction: num -= 2*len*min(ai,aj); den unchanged
            for &v in &stripe_i_ref.nodes {
                let j_idx = stripe_j.index[v];
                if j_idx == u32::MAX { continue; }
                let len2 = 2.0 * (lens[v] as f64);
                if len2 == 0.0 { continue; }
                let ri = stripe_i_ref.index[v] as usize;
                let rj = j_idx as usize;
                let ai = &stripe_i_ref.rows[ri];
                let aj = &stripe_j.rows[rj];
                let nz_i = &stripe_i_ref.nz[ri];
                let nz_j = &stripe_j.nz[rj];

                for &ii in nz_i {
                    let gi = i0 + ii;
                    for &jj in nz_j {
                        let gj = j0 + jj;
                        if diagonal_block && gj <= gi { continue; }
                        let m = (ai[ii] as f64).min(aj[jj] as f64);
                        num[ii * bh + jj] -= len2 * m;
                    }
                }
            }

            // write back normalized value
            unsafe {
                let base = base_addr as *mut f64;
                for ii in 0..bw {
                    let i = i0 + ii;
                    for jj in 0..bh {
                        let j = j0 + jj;
                        if j <= i { continue; }
                        let idx = ii * bh + jj;
                        let d = if den[idx] > 0.0 { num[idx] / den[idx] } else { 0.0 };
                        *base.add(i * nsamp + j) = d;
                        *base.add(j * nsamp + i) = d;
                    }
                }
            }
        });
    }

    Arc::try_unwrap(dist).unwrap()
}

fn unifrac_striped_par_generalized(
    _post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    row2leaf: &[Option<usize>],
    mode: WeightedMode,
    nsamp: usize,
    col_sums: &[f64],
    alpha: f64,
) -> Vec<f64> {
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

    let n_threads = rayon::current_num_threads().max(1);
    let est_blk = ((nsamp as f64 / (2.0 * n_threads as f64)).sqrt()) as usize;
    let blk = est_blk.clamp(64, 512).next_power_of_two();
    let nblk = (nsamp + blk - 1) / blk;

    let dist = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let base_addr: usize = dist.as_ptr() as usize;

    // α == 0 → denominator is sum of positive branch lengths (constant)
    let lsum_alpha0: f64 = if alpha.abs() <= 1e-12 {
        lens.iter().filter(|&&l| l > 0.0).map(|&l| l as f64).sum()
    } else {
        0.0
    };

    for bi in 0..nblk {
        let i0 = bi * blk;
        let i1 = (i0 + blk).min(nsamp);
        let bw = i1 - i0;

        let stripe_i = match &mode {
            WeightedMode::Dense { counts } => {
                build_stripe_dense(counts, row2leaf, leaf_ids, &parent, col_sums, i0, i1, total, false)
            }
            WeightedMode::Csr { indptr, indices, data } => build_stripe_csr(
                indptr, indices, data, row2leaf, leaf_ids, &parent, col_sums, i0, i1, total, false
            ),
        };

        let mode_c = mode;
        let parent_ref = &parent;
        let row2leaf_ref = row2leaf;
        let leaf_ids_ref = leaf_ids;
        let col_sums_ref = col_sums;
        let stripe_i_ref = &stripe_i;

        (bi..nblk).into_par_iter().for_each(move |bj| {
            let (j0, j1) = (bj * blk, ((bj + 1) * blk).min(nsamp));
            let bh = j1 - j0;
            let diagonal_block = bj == bi;

            let stripe_j = if diagonal_block {
                Stripe {
                    nodes: stripe_i_ref.nodes.clone(),
                    rows: stripe_i_ref.rows.clone(),
                    index: stripe_i_ref.index.clone(),
                    nz: stripe_i_ref.nz.clone(),
                }
            } else {
                match mode_c {
                    WeightedMode::Dense { counts } => build_stripe_dense(
                        counts, row2leaf_ref, leaf_ids_ref, parent_ref, col_sums_ref, j0, j1, total, false
                    ),
                    WeightedMode::Csr { indptr, indices, data } => build_stripe_csr(
                        indptr, indices, data, row2leaf_ref, leaf_ids_ref, parent_ref, col_sums_ref, j0, j1, total, false
                    ),
                }
            };

            let mut num = vec![0.0f64; bw * bh];
            let mut den = vec![0.0f64; bw * bh];

            #[inline(always)]
            fn pow_alpha(x: f64, a: f64) -> f64 {
                if x <= 0.0 { 0.0 }
                else if (a - 1.0).abs() <= 1e-12 { x }
                else if a.abs() <= 1e-12 { 1.0 }
                else if (a - 0.5).abs() <= 1e-12 { x.sqrt() }
                else { x.powf(a) }
            }
            #[inline(always)]
            fn pow_alpha_minus1(x: f64, a: f64) -> f64 {
                if x <= 0.0 { 0.0 }
                else if (a - 1.0).abs() <= 1e-12 { 1.0 }
                else if a.abs() <= 1e-12 { 1.0 / x }
                else if (a - 0.5).abs() <= 1e-12 { 1.0 / x.sqrt() }
                else { x.powf(a - 1.0) }
            }

            // (A) add len * ai^α across row (to both num & den for α≠0)
            for &v in &stripe_i_ref.nodes {
                let len = lens[v] as f64;
                if len <= 0.0 { continue; }
                let ri = stripe_i_ref.index[v] as usize;
                let ai = &stripe_i_ref.rows[ri];
                let nz_i = &stripe_i_ref.nz[ri];

                for &ii in nz_i {
                    let a = ai[ii] as f64;
                    if a <= 0.0 { continue; }
                    let base = len * pow_alpha(a, alpha);
                    let start_j = if diagonal_block { ii + 1 } else { 0 };
                    add_const_to_row_simd::<8>(&mut num[ii * bh..(ii + 1) * bh], base, start_j);
                    if alpha.abs() > 1e-12 {
                        add_const_to_row_simd::<8>(&mut den[ii * bh..(ii + 1) * bh], base, start_j);
                    }
                }
            }

            // (B) add len * aj^α down column (to both num & den for α≠0)
            for &v in &stripe_j.nodes {
                let len = lens[v] as f64;
                if len <= 0.0 { continue; }
                let rj = stripe_j.index[v] as usize;
                let aj = &stripe_j.rows[rj];
                let nz_j = &stripe_j.nz[rj];

                for &jj in nz_j {
                    let a = aj[jj] as f64;
                    if a <= 0.0 { continue; }
                    let add = len * pow_alpha(a, alpha);
                    let ii_end = if diagonal_block { ((j0 + jj).saturating_sub(i0)).min(bw) } else { bw };
                    let mut ii = 0usize;
                    while ii < ii_end {
                        let idx = ii * bh + jj;
                        num[idx] += add;
                        if alpha.abs() > 1e-12 {
                            den[idx] += add;
                        }
                        ii += 1;
                    }
                }
            }

            // (C) intersection correction
            // num += len * [ s^(α-1)|ai-aj| - (ai^α + aj^α) ]
            // den += len * [ s^α          - (ai^α + aj^α) ]   (only if α≠0)
            for &v in &stripe_i_ref.nodes {
                let j_idx = stripe_j.index[v];
                if j_idx == u32::MAX { continue; }
                let len = lens[v] as f64;
                if len <= 0.0 { continue; }
                let ri = stripe_i_ref.index[v] as usize;
                let rj = j_idx as usize;
                let ai = &stripe_i_ref.rows[ri];
                let aj = &stripe_j.rows[rj];
                let nz_i = &stripe_i_ref.nz[ri];
                let nz_j = &stripe_j.nz[rj];

                for &ii in nz_i {
                    let gi = i0 + ii;
                    let a = ai[ii] as f64;
                    for &jj in nz_j {
                        let gj = j0 + jj;
                        if diagonal_block && gj <= gi { continue; }
                        let b = aj[jj] as f64;
                        let s = a + b;
                        if s <= 0.0 { continue; }

                        let s_alpha  = pow_alpha(s, alpha);
                        let s_am1    = pow_alpha_minus1(s, alpha);
                        let a_alpha  = pow_alpha(a, alpha);
                        let b_alpha  = pow_alpha(b, alpha);
                        let idx = ii * bh + jj;

                        num[idx] += len * (s_am1 * (a - b).abs() - (a_alpha + b_alpha));
                        if alpha.abs() > 1e-12 {
                            den[idx] += len * (s_alpha - (a_alpha + b_alpha));
                        }
                    }
                }
            }

            // write back (normalized if α≠0; α==0 uses constant denom)
            unsafe {
                let base = base_addr as *mut f64;
                for ii in 0..bw {
                    let i = i0 + ii;
                    for jj in 0..bh {
                        let j = j0 + jj;
                        if j <= i { continue; }
                        let idx = ii * bh + jj;
                        let d = if alpha.abs() <= 1e-12 {
                            if lsum_alpha0 > 0.0 { num[idx] / lsum_alpha0 } else { 0.0 }
                        } else if den[idx] > 0.0 {
                            num[idx] / den[idx]
                        } else {
                            0.0
                        };
                        *base.add(i * nsamp + j) = d;
                        *base.add(j * nsamp + i) = d;
                    }
                }
            }
        });
    }

    Arc::try_unwrap(dist).unwrap()
}

// BIOM readers
fn read_biom_csr(p: &str) -> Result<(Vec<String>, Vec<String>, Vec<u32>, Vec<u32>)> {
    let f = H5File::open(p).with_context(|| format!("open BIOM file {p}"))?;

    fn read_utf8(f: &H5File, path: &str) -> Result<Vec<String>> {
        Ok(f.dataset(path)?
            .read_1d::<VarLenUnicode>()?
            .into_iter()
            .map(|v| v.as_str().to_owned())
            .collect())
    }
    fn read_u32(f: &H5File, path: &str) -> Result<Vec<u32>> {
        Ok(f.dataset(path)?.read_raw::<u32>()?.to_vec())
    }

    let taxa = read_utf8(&f, "observation/ids").context("missing observation/ids")?;
    let samples = read_utf8(&f, "sample/ids").context("missing sample/ids")?;

    let try_paths = |name: &str| -> Result<Vec<u32>> {
        read_u32(&f, &format!("observation/matrix/{name}"))
            .or_else(|_| read_u32(&f, &format!("observation/{name}")))
            .with_context(|| format!("missing observation/**/{name}"))
    };

    let indptr = try_paths("indptr")?;
    let indices = try_paths("indices")?;
    Ok((taxa, samples, indptr, indices))
}

fn read_biom_csr_values(
    p: &str,
) -> Result<(Vec<String>, Vec<String>, Vec<u32>, Vec<u32>, Vec<f64>)> {
    let f = H5File::open(p).with_context(|| format!("open BIOM file {p}"))?;

    fn read_utf8(f: &H5File, path: &str) -> Result<Vec<String>> {
        Ok(f.dataset(path)?
            .read_1d::<VarLenUnicode>()?
            .into_iter()
            .map(|v| v.as_str().to_owned())
            .collect())
    }
    fn read_u32(f: &H5File, path: &str) -> Result<Vec<u32>> {
        Ok(f.dataset(path)?.read_raw::<u32>()?.to_vec())
    }
    fn read_f64_flex(f: &H5File, path: &str) -> Result<Vec<f64>> {
        if let Ok(v) = f.dataset(path)?.read_raw::<f64>() {
            Ok(v.to_vec())
        } else {
            let v32 = f.dataset(path)?.read_raw::<f32>()?;
            Ok(v32.iter().map(|&x| x as f64).collect())
        }
    }

    let taxa = read_utf8(&f, "observation/ids").context("missing observation/ids")?;
    let samples = read_utf8(&f, "sample/ids").context("missing sample/ids")?;

    let try_paths_u32 = |name: &str| -> Result<Vec<u32>> {
        read_u32(&f, &format!("observation/matrix/{name}"))
            .or_else(|_| read_u32(&f, &format!("observation/{name}")))
            .with_context(|| format!("missing observation/**/{name}"))
    };
    let try_paths_f64 = |name: &str| -> Result<Vec<f64>> {
        read_f64_flex(&f, &format!("observation/matrix/{name}"))
            .or_else(|_| read_f64_flex(&f, &format!("observation/{name}")))
            .with_context(|| format!("missing observation/**/{name}"))
    };

    let indptr = try_paths_u32("indptr")?;
    let indices = try_paths_u32("indices")?;
    let data = try_paths_f64("data")?;
    Ok((taxa, samples, indptr, indices, data))
}

fn main() -> Result<()> {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");

    let m = Command::new("unifrac-rs")
        .version("0.2.7")
        .about("Striped UniFrac via Optimal Balanced Parenthesis")
        .arg(
            Arg::new("tree")
                .short('t')
                .long("tree")
                .help("Input tree in Newick format")
                .required(true),
        )
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .help("OTU/Feature table in TSV format")
                .required(false),
        )
        .arg(
            Arg::new("biom")
                .short('m')
                .long("biom")
                .help("OTU/Feature table in BIOM (HDF5) format")
                .required(false),
        )
        .group(ArgGroup::new("table").args(["input", "biom"]).required(true))
        .arg(
            Arg::new("weighted")
                .long("weighted")
                .help("Weighted UniFrac (normalized). Per-sample relative abundances will be used")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("raw_sample_counts")
                .long("raw-sample-counts")
                .help("Weighted UniFrac using raw sample counts. Applies only with --weighted. Ignored for --generalized and unweighted.")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .short('T')
                .help("Number of threads, default all logical cores")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output distance matrix in TSV format")
                .default_value("unifrac.tsv"),
        )
        .arg(
            Arg::new("generalized")
                .long("generalized")
                .help("Generalized UniFrac")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("alpha")
                .long("alpha")
                .help("Alpha parameter for Generalized UniFrac. Only used with --generalized.")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.5"),
        )
        .group(
            ArgGroup::new("metric")
                .args(["weighted", "generalized"])
                .required(false)
                .multiple(false),
        )
        .get_matches();

    let tree_file = m.get_one::<String>("tree").unwrap();
    let out_file = m.get_one::<String>("output").unwrap();
    let generalized = *m.get_one::<bool>("generalized").unwrap_or(&false);
    let alpha = *m.get_one::<f64>("alpha").unwrap_or(&0.5);
    let weighted = *m.get_one::<bool>("weighted").unwrap_or(&false);
    let raw_counts = *m.get_one::<bool>("raw_sample_counts").unwrap_or(&false);

    let threads = m
        .get_one::<usize>("threads")
        .copied()
        .unwrap_or_else(|| num_cpus::get());

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build_global()
        .unwrap();

    // load tree
    let raw = std::fs::read_to_string(tree_file).context("read newick")?;
    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
    let t: NwkTree = one_from_string(&sanitized).context("parse newick (sanitized)")?;
    let mut lens = Vec::<f32>::new();
    let trav = SuccTrav::new(&t, &mut lens);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    // leaves → mapping taxon-name → leaf-index
    let mut leaf_ids = Vec::<usize>::new();
    let mut leaf_nm = Vec::<String>::new();
    for n in t.nodes() {
        if t[n].is_leaf() {
            leaf_ids.push(n);
            leaf_nm.push(t.name(n).map(ToOwned::to_owned).unwrap_or_else(|| format!("L{n}")));
        }
    }
    let t2leaf: HashMap<&str, usize> = leaf_nm
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    drop(t);

    // children & post-order
    let total = bp.len() + 1;
    lens.resize(total, 0.0);
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);
    log::info!(
        "Total branches with positive length: {}",
        lens.iter().filter(|&&l| l > 0.0).count()
    );

    let mut node2leaf = vec![usize::MAX; total];
    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        node2leaf[nid] = leaf_pos;
    }
    drop(node2leaf);

    // Read table (TSV or BIOM)
    log::info!("Start parsing input.");

    let samples: Vec<String>;
    let mut taxa: Vec<String>;

    // Data containers
    let mut pres: Vec<Vec<f64>> = Vec::new(); // TSV presence for unweighted
    let mut counts: Vec<Vec<f64>> = Vec::new(); // TSV counts for weighted/generalized
    let mut indptr: Vec<u32> = Vec::new(); // BIOM (both modes)
    let mut indices: Vec<u32> = Vec::new(); // BIOM (both modes)
    let mut data: Vec<f64> = Vec::new(); // BIOM values for weighted/generalized
    let mut pres_dense = false; // true => TSV mode

    if let Some(tsv) = m.get_one::<String>("input") {
        pres_dense = true;
        if weighted || generalized {
            let (t, s, mat) = read_table_counts(tsv)?;
            taxa = t;
            samples = s;
            counts = mat;
        } else {
            let (t, s, mat) = read_table(tsv)?;
            taxa = t;
            samples = s;
            pres = mat;
        }
    } else {
        let biom = m.get_one::<String>("biom").unwrap();
        if weighted || generalized {
            let (t, s, ip, idx, vals) = read_biom_csr_values(biom)?;
            taxa = t;
            samples = s;
            indptr = ip;
            indices = idx;
            data = vals;
        } else {
            let (t, s, ip, idx) = read_biom_csr(biom)?;
            taxa = t;
            samples = s;
            indptr = ip;
            indices = idx;
        }
    }
    let nsamp = samples.len();

    // Map each input row (taxon) to a leaf position (if present)
    let row2leaf: Vec<Option<usize>> = taxa
        .iter()
        .map(|name| t2leaf.get(name.as_str()).copied())
        .collect();

    // Dispatch
    let dist: Vec<f64> = if generalized {
        // per-sample column sums for relative abundances
        let mut col_sums = vec![0.0f64; nsamp];
        if pres_dense {
            for r in 0..counts.len() {
                for s in 0..nsamp {
                    col_sums[s] += counts[r][s];
                }
            }
            if col_sums.iter().all(|&x| x == 0.0) {
                log::warn!("All column sums are zero in TSV generalized run; check counts.");
            }
            if (alpha - 1.0).abs() <= 1e-12 {
                unifrac_striped_par_weighted(
                    &post,
                    &kids,
                    &lens,
                    &leaf_ids,
                    &row2leaf,
                    WeightedMode::Dense { counts: &counts },
                    nsamp,
                    &col_sums,
                    raw_counts,
                )
            } else {
                unifrac_striped_par_generalized(
                    &post,
                    &kids,
                    &lens,
                    &leaf_ids,
                    &row2leaf,
                    WeightedMode::Dense { counts: &counts },
                    nsamp,
                    &col_sums,
                    alpha,
                )
            }
        } else {
            for r in 0..taxa.len() {
                let start = indptr[r] as usize;
                let stop = indptr[r + 1] as usize;
                for k in start..stop {
                    let s = indices[k] as usize;
                    col_sums[s] += data[k];
                }
            }
            if col_sums.iter().all(|&x| x == 0.0) {
                log::warn!("All column sums are zero in BIOM generalized run; check BIOM data.");
            }
            if (alpha - 1.0).abs() <= 1e-12 {
                unifrac_striped_par_weighted(
                    &post,
                    &kids,
                    &lens,
                    &leaf_ids,
                    &row2leaf,
                    WeightedMode::Csr {
                        indptr: &indptr,
                        indices: &indices,
                        data: &data,
                    },
                    nsamp,
                    &col_sums,
                    raw_counts,
                )
            } else {
                unifrac_striped_par_generalized(
                    &post,
                    &kids,
                    &lens,
                    &leaf_ids,
                    &row2leaf,
                    WeightedMode::Csr {
                        indptr: &indptr,
                        indices: &indices,
                        data: &data,
                    },
                    nsamp,
                    &col_sums,
                    alpha,
                )
            }
        }
    } else if weighted {
        let _col_sums = vec![0.0f64; nsamp];
        if pres_dense {
            let col_sums = if raw_counts {
                vec![1.0f64; nsamp] // dummy; unused when raw_counts==true
            } else {
                let mut v = vec![0.0f64; nsamp];
                for r in 0..counts.len() {
                    for s in 0..nsamp { v[s] += counts[r][s]; }
                }
                v
            };

            unifrac_striped_par_weighted(
                &post,
                &kids,
                &lens,
                &leaf_ids,
                &row2leaf,
                WeightedMode::Dense { counts: &counts },
                nsamp,
                &col_sums,
                raw_counts,
            )
        } else {
            let col_sums = if raw_counts {
                vec![1.0f64; nsamp] // dummy; unused when raw_counts==true
            } else {
                let mut v = vec![0.0f64; nsamp];
                for r in 0..taxa.len() {
                    let a = indptr[r] as usize;
                    let b = indptr[r + 1] as usize;
                    for k in a..b { v[indices[k] as usize] += data[k]; }
                }
                v
            };

            unifrac_striped_par_weighted(
                &post,
                &kids,
                &lens,
                &leaf_ids,
                &row2leaf,
                WeightedMode::Csr {
                    indptr: &indptr,
                    indices: &indices,
                    data: &data,
                },
                nsamp,
                &col_sums,
                raw_counts,
            )
        }
    } else {
        // Unweighted
        let mut masks: Vec<BitVec<u8, Lsb0>> =
            (0..nsamp).map(|_| BitVec::repeat(false, leaf_ids.len())).collect();

        if pres_dense {
            for (r, lopt) in row2leaf.iter().enumerate() {
                if let Some(leaf_pos) = lopt {
                    let leaf = *leaf_pos;
                    for (s, bits) in masks.iter_mut().enumerate() {
                        if pres[r][s] > 0.0 {
                            bits.set(leaf, true);
                        }
                    }
                }
            }
        } else {
            for r in 0..taxa.len() {
                if let Some(leaf_pos) = row2leaf[r] {
                    let start = indptr[r] as usize;
                    let stop = indptr[r + 1] as usize;
                    for k in start..stop {
                        let s = indices[k] as usize;
                        masks[s].set(leaf_pos, true);
                    }
                }
            }
        }
        unifrac_striped_par(&post, &kids, &lens, &leaf_ids, masks)
    };

    // Free big structures early
    drop(t2leaf);
    leaf_nm.clear();
    leaf_nm.shrink_to_fit();
    taxa.clear();
    taxa.shrink_to_fit();
    pres.clear();
    pres.shrink_to_fit();
    counts.clear();
    counts.shrink_to_fit();
    indptr.clear();
    indptr.shrink_to_fit();
    indices.clear();
    indices.shrink_to_fit();
    data.clear();
    data.shrink_to_fit();
    drop(kids);
    drop(post);
    drop(leaf_ids);
    lens.clear();
    lens.shrink_to_fit();

    log::info!("Start writing output.");
    write_matrix(&samples, &dist, nsamp, out_file)
}
