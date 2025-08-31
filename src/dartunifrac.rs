//! BSD 3-Clause License
//!
//! Copyright (c) 2016-2025, UniFrac development team.
//! All rights reserved.
//!
//! See LICENSE file for more details

//! DartUniFrac: Approximate UniFrac via Weighted MinHash
//! DartMinHash or ERS (Efficient Rejection Sampling) can be used as the underlying algorithm
//! Tree parsing via optimal balanced parenthesis:
//! With constant-time rank/select primitives (rank₁(i) = # of 1-bits up to i, select₁(k) = position of the k-th 1-bit) you get parent, k-th child, next sibling, sub-tree size, depth, all in O(1). every node knows its opening index i. parent(i) = select₁(rank₁(i) - 1), next_sibling(i) = find_close(i) + 1 (where find_close is the matching 0), etc. Those functions are just pointer-arithmetic on the backing Vec<u64>.
//! Both unweighted and weighted UniFrac (normalized) are supported
//! Input: TSV or BIOM (HDF5) feature tables. BIOM can be used for very sparse dataset to save space
//! Output: TSV distance matrix and pcoa results (optional)

use std::path::{Path, PathBuf};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};

use anyhow::{Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use clap::{Arg, ArgGroup, Command};
use env_logger;
use log::{info, warn};
use rayon::prelude::*;

use anndists::dist::{DistHamming, Distance};
use dartminhash::{DartMinHash, ErsWmh, rng_utils::mt_from_seed};
use hdf5::{File as H5File, types::VarLenUnicode};
use newick::{Newick, NodeID, one_from_string};
use succparen::{
    bitwise::{SparseOneNnd, ops::NndOne},
    tree::Node,
    tree::{
        LabelVec,
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
    },
};

use fpcoa::{FpcoaOptions, pcoa_randomized};
use ndarray::{Array1, Array2};

type NwkTree = newick::NewickTree;

// Tree traversal to collect branch lengths
fn sanitize_newick_drop_internal_labels_and_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'[' => {
                // Skip bracket comments (and tolerate nested just in case)
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
                // Emit ')'
                out.push(')');
                i += 1;

                // Skip whitespace
                while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                    i += 1;
                }

                // Optional internal label right after ')': quoted or unquoted.
                if i < bytes.len() && bytes[i] == b'\'' {
                    // Quoted label — skip it
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
                    // (comments after this will be removed by the '[' arm next loop)
                } else {
                    // Unquoted run until a delimiter
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
                // Don’t consume delimiters like ':' — they’ll be handled in the main loop.
            }
            _ => {
                // Normal char — copy
                out.push(bytes[i] as char);
                i += 1;
            }
        }
    }
    out
}

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

// TSV unweighted
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
            .map(|v| {
                if v.parse::<f64>().unwrap_or(0.0) > 0.0 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        taxa.push(tax);
        mat.push(vals);
    }
    Ok((taxa, samples, mat))
}
// TSV weighted
fn read_table_counts(p: &str) -> Result<(Vec<String>, Vec<String>, Vec<Vec<f64>>)> {
    let f = File::open(p)?;
    let mut lines = BufReader::new(f).lines();
    let hdr = lines.next().context("empty table")??;
    let mut it = hdr.split('\t');
    it.next();
    let samples = it.map(|s| s.to_owned()).collect::<Vec<_>>();

    let mut taxa = Vec::new();
    let mut mat = Vec::new();
    for l in lines {
        let row = l?;
        let mut p = row.split('\t');
        let tax = p.next().unwrap().to_owned();
        let vals = p
            .map(|v| v.parse::<f64>().unwrap_or(0.0))
            .collect::<Vec<f64>>();
        taxa.push(tax);
        mat.push(vals);
    }
    Ok((taxa, samples, mat))
}

// Uweighted BIOM CSR
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

// Weighted BIOM CSR with values
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

    let try_u32 = |name: &str| -> Result<Vec<u32>> {
        read_u32(&f, &format!("observation/matrix/{name}"))
            .or_else(|_| read_u32(&f, &format!("observation/{name}")))
            .with_context(|| format!("missing observation/**/{name}"))
    };
    let try_f64 = |name: &str| -> Result<Vec<f64>> {
        read_f64_flex(&f, &format!("observation/matrix/{name}"))
            .or_else(|_| read_f64_flex(&f, &format!("observation/{name}")))
            .with_context(|| format!("missing observation/**/{name}"))
    };

    let indptr = try_u32("indptr")?;
    let indices = try_u32("indices")?;
    let data = try_f64("data")?;
    Ok((taxa, samples, indptr, indices, data))
}

// Write TSV matrix (fast, reusing ryu buffer per row)
fn write_matrix(names: &[String], d: &[f64], n: usize, path: &str) -> Result<()> {
    // Header
    let header = {
        let mut s = String::with_capacity(n * 16);
        s.push_str("");
        for name in names {
            s.push('\t');
            s.push_str(name);
        }
        s.push('\n');
        s
    };

    // Build all rows in parallel; reuse a single ryu buffer per row
    let mut rows: Vec<String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut line = String::with_capacity(8 + n * 12);
            line.push_str(&names[i]);
            let base = i * n;
            // Adams, U., 2018, June. Ryū: fast float-to-string conversion. In Proceedings of the 39th ACM SIGPLAN Conference on Programming Language Design and Implementation (pp. 270-282).
            let mut buf = ryu::Buffer::new();
            for j in 0..n {
                let val = unsafe { *d.get_unchecked(base + j) };
                line.push('\t');
                line.push_str(buf.format_finite(val));
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

fn write_matrix_zstd(names: &[String], d: &[f64], n: usize, path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    // zstd multi-threading
    let file = File::create(path)?;
    let mut enc = zstd::Encoder::new(file, 0)?;
    let zstd_threads = rayon::current_num_threads() as u32;
    if zstd_threads > 1 {
        enc.multithread(zstd_threads)?;
    }
    let mut out = BufWriter::with_capacity(16 << 20, enc.auto_finish());

    // Header
    let header = {
        let mut s = String::with_capacity(n * 16);
        s.push_str("");
        for name in names {
            s.push('\t');
            s.push_str(name);
        }
        s.push('\n');
        s
    };
    out.write_all(header.as_bytes())?;

    let mut rows: Vec<String> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut line = String::with_capacity(8 + n * 12);
            line.push_str(&names[i]);
            let base = i * n;
            let mut buf = ryu::Buffer::new();
            for j in 0..n {
                let val = unsafe { *d.get_unchecked(base + j) };
                line.push('\t');
                line.push_str(buf.format_finite(val));
            }
            line.push('\n');
            line
        })
        .collect();

    for line in &mut rows {
        out.write_all(line.as_bytes())?;
        line.clear();
    }
    out.flush()?;

    Ok(())
}

fn write_matrix_streaming_zstd(
    names: &[String],
    sketches: &[Vec<u64>],
    path: &str,
    block_size_opt: Option<usize>,
    weighted_normalized: bool,
) -> Result<()> {
    let n = names.len();
    assert_eq!(sketches.len(), n);

    // zstd multi-threaded encoder + big buffer
    let file = File::create(path)?;
    let mut enc = zstd::Encoder::new(file, 0)?;
    let zstd_threads = rayon::current_num_threads() as u32;
    if zstd_threads > 1 {
        enc.multithread(zstd_threads)?;
    }
    let mut w = BufWriter::with_capacity(16 << 20, enc.auto_finish());

    // Header: "", <names...>
    w.write_all(b"")?;
    for name in names {
        w.write_all(b"\t")?;
        w.write_all(name.as_bytes())?;
    }
    w.write_all(b"\n")?;

    // Block size: default = floor(sqrt(n))
    let default_bs = ((n as f64).sqrt() as usize).max(1);
    let bs = block_size_opt.unwrap_or(default_bs);
    info!("streaming block-size = {} (n = {})", bs, n);

    // Column-major block buffer: n × bs (each column j is a contiguous slice of length bs)
    let mut block = vec![0.0f64; n * bs];
    let dh = DistHamming;

    let mut i0 = 0usize;
    while i0 < n {
        let h = (n - i0).min(bs);

        // Fill block in parallel over columns; each `col` is a disjoint &mut [f64] (length = bs)
        block.par_chunks_mut(bs).enumerate().for_each(|(j, col)| {
            for bi in 0..h {
                let i = i0 + bi;
                let mut d = if i == j {
                    0.0
                } else {
                    dh.eval(&sketches[i], &sketches[j]) as f64
                };
                // d is an unbiased estimate of d_J = 1 - Jw
                if weighted_normalized {
                    // Bray–Curtis / normalized weighted UniFrac:
                    // D = (1 - Jw) / (1 + Jw) = d_J / (2 - d_J)
                    d = if d < 2.0 { d / (2.0 - d) } else { 1.0 };
                }
                col[bi] = d; // write into column-major slot (j, bi)
            }
        });

        // Write the block (single writer, amortized I/O)
        let mut lines: Vec<String> = (0..h)
            .into_par_iter()
            .map(|bi| {
                let i = i0 + bi;
                // Pre-size roughly: tab + ~12 chars per number
                let mut line = String::with_capacity(8 + n * 12);
                line.push_str(&names[i]);

                // Each worker needs its own Ryu buffer
                let mut fmt = ryu::Buffer::new();
                for j in 0..n {
                    line.push('\t');
                    let v = block[j * bs + bi]; // column-major index (j, bi)
                    line.push_str(fmt.format_finite(v));
                }
                line.push('\n');
                line
            })
            .collect();

        // Single writer, amortized by BufWriter + zstd MT compression
        for line in &mut lines {
            w.write_all(line.as_bytes())?;
            line.clear(); // allow capacity reuse across iterations (if the allocator keeps it)
        }
        w.flush()?;
        i0 += h;
    }

    Ok(())
}

// Build per-node sample bitsets (presence under node)
fn build_node_bits(
    post: &[usize],
    kids: &[Vec<usize>],
    leaf_ids: &[usize],
    masks: &[bitvec::vec::BitVec<u8, Lsb0>],
    total_nodes: usize,
) -> Vec<bitvec::vec::BitVec<u64, Lsb0>> {
    let nsamp = masks.len();
    let n_threads = rayon::current_num_threads().max(1);
    let stripe = (nsamp + n_threads - 1) / n_threads; // ceil
    let words_str = (stripe + 63) >> 6; // u64 words per stripe

    // node_masks[tid][node][word] – per-thread stripes
    let mut node_masks: Vec<Vec<Vec<u64>>> = (0..n_threads)
        .map(|_| vec![vec![0u64; words_str]; total_nodes])
        .collect();

    // Phase 1: build per-thread stripes bottom-up
    rayon::scope(|scope| {
        for (tid, node_masks_t) in node_masks.iter_mut().enumerate() {
            let stripe_start = tid * stripe;
            if stripe_start >= nsamp {
                break;
            }
            let stripe_end = (stripe_start + stripe).min(nsamp);
            let masks_slice = &masks[stripe_start..stripe_end];

            scope.spawn(move |_| {
                // scatter leaves
                for (local_s, sm) in masks_slice.iter().enumerate() {
                    for pos in sm.iter_ones() {
                        let v = leaf_ids[pos];
                        let w = local_s >> 6;
                        let b = local_s & 63;
                        node_masks_t[v][w] |= 1u64 << b;
                    }
                }
                // bottom-up OR within stripe
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

    // Phase 2: merge stripes to a single BitVec per node
    let mut node_bits: Vec<bitvec::vec::BitVec<u64, Lsb0>> = (0..total_nodes)
        .map(|_| bitvec::vec::BitVec::repeat(false, nsamp))
        .collect();

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

    node_bits
}

// Weighted mode switch for dense/CSR input
enum WeightedMode<'a> {
    Dense {
        counts: &'a [Vec<f64>],
        col_sums: &'a [f64],
    }, // rows x nsamp
    Csr {
        indptr: &'a [u32],
        indices: &'a [u32],
        data: &'a [f64],
        col_sums: &'a [f64],
    }, // BIOM
}

// Weighted, compute per-node per-sample relative masses A_v[s] using stripes
// Returns:
//  - node_sums[tid][v][local_sample]
//  - has_mass[v]  (OR across all stripes)
//  - edges_touched_per_tid[tid] : list of node ids v that had any mass in that stripe
fn build_node_sums_weighted(
    post: &[usize],
    kids: &[Vec<usize>],
    leaf_ids: &[usize],
    row2leaf: &[Option<usize>],
    mode: WeightedMode<'_>,
    nsamp: usize,
) -> (Vec<Vec<Vec<f64>>>, Vec<bool>, Vec<Vec<usize>>) {
    let total = kids.len();
    let n_threads = rayon::current_num_threads().max(1);
    let stripe = (nsamp + n_threads - 1) / n_threads;

    // node_sums[tid][node][local_sample]
    let mut node_sums: Vec<Vec<Vec<f64>>> = (0..n_threads)
        .map(|tid| {
            let start = tid * stripe;
            if start >= nsamp {
                Vec::new()
            } else {
                let end = (start + stripe).min(nsamp);
                vec![vec![0.0f64; end - start]; total]
            }
        })
        .collect();

    // any_mass_per_tid[tid][node] — node had any positive mass in this stripe
    let mut any_mass_per_tid: Vec<Vec<bool>> = (0..n_threads)
        .map(|tid| {
            let start = tid * stripe;
            if start >= nsamp { Vec::new() } else { vec![false; total] }
        })
        .collect();

    // IMPORTANT: consume the flags slice one-by-one to give each task a unique &mut
    let mut flags_slice: &mut [Vec<bool>] = any_mass_per_tid.as_mut_slice();

    rayon::scope(|scope| {
        for (tid, sums_t) in node_sums.iter_mut().enumerate() {
            let stripe_start = tid * stripe;
            if stripe_start >= nsamp { break; }
            let stripe_end = (stripe_start + stripe).min(nsamp);
            let wloc = stripe_end - stripe_start;

            // unique &mut to this tid's flags (fixes E0499)
            let (touched_t, rest) = flags_slice.split_first_mut().expect("flags_slice underflow");
            flags_slice = rest;

            match mode {
                WeightedMode::Dense { counts, col_sums } => {
                    scope.spawn(move |_| {
                        // scatter leaves (relative abundances)
                        for (r, lopt) in row2leaf.iter().enumerate() {
                            if let Some(leaf_pos) = lopt {
                                let v = leaf_ids[*leaf_pos];
                                let sv = &mut sums_t[v];
                                for s in stripe_start..stripe_end {
                                    let denom = col_sums[s];
                                    if denom > 0.0 {
                                        let val = counts[r][s] / denom;
                                        if val > 0.0 {
                                            sv[s - stripe_start] += val;
                                            touched_t[v] = true;
                                        }
                                    }
                                }
                            }
                        }
                        // bottom-up aggregate within stripe and flag propagation
                        for &v in post {
                            for &c in &kids[v] {
                                let (left, right) =
                                    if c <= v { sums_t.split_at_mut(v) } else { sums_t.split_at_mut(c) };
                                let (sv, sc) = if c <= v { (&mut right[0], &left[c]) }
                                               else        { (&mut left[v],  &right[0]) };
                                for k in 0..wloc { sv[k] += sc[k]; }
                                if touched_t[c] { touched_t[v] = true; }
                            }
                        }
                    });
                }
                WeightedMode::Csr { indptr, indices, data, col_sums } => {
                    scope.spawn(move |_| {
                        // scatter leaves (CSR to relative abundances)
                        for r in 0..row2leaf.len() {
                            if let Some(leaf_pos) = row2leaf[r] {
                                let v = leaf_ids[leaf_pos];
                                let sv = &mut sums_t[v];
                                let start = indptr[r] as usize;
                                let stop  = indptr[r + 1] as usize;
                                for k in start..stop {
                                    let s = indices[k] as usize;
                                    if s < stripe_start || s >= stripe_end { continue; }
                                    let denom = col_sums[s];
                                    if denom > 0.0 {
                                        let val = data[k] / denom;
                                        if val > 0.0 {
                                            sv[s - stripe_start] += val;
                                            touched_t[v] = true;
                                        }
                                    }
                                }
                            }
                        }
                        // bottom-up aggregate within stripe + flag propagation
                        for &v in post {
                            for &c in &kids[v] {
                                let (left, right) =
                                    if c <= v { sums_t.split_at_mut(v) } else { sums_t.split_at_mut(c) };
                                let (sv, sc) = if c <= v { (&mut right[0], &left[c]) }
                                               else        { (&mut left[v],  &right[0]) };
                                for k in 0..wloc { sv[k] += sc[k]; }
                                if touched_t[c] { touched_t[v] = true; }
                            }
                        }
                    });
                }
            }
        }
    });

    // OR-reduce to global has_mass
    let mut has_mass = vec![false; total];
    for flags in &any_mass_per_tid {
        for (v, &f) in flags.iter().enumerate() {
            if f { has_mass[v] = true; }
        }
    }

    // Compact per-tid flags → lists of nodes touched in that stripe
    let mut edges_touched_per_tid: Vec<Vec<usize>> = Vec::with_capacity(n_threads);
    for flags in any_mass_per_tid.into_iter() {
        if flags.is_empty() {
            edges_touched_per_tid.push(Vec::new());
            continue;
        }
        // Heuristic reserve: ~total/64
        let mut lst = Vec::with_capacity(total >> 6);
        for (v, f) in flags.into_iter().enumerate() {
            if f { lst.push(v); }
        }
        edges_touched_per_tid.push(lst);
    }

    (node_sums, has_mass, edges_touched_per_tid)
}

fn build_sketches(
    tree_file: &str,
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    k: usize,
    method: &str,
    ers_l: u64,
    seed: u64,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    // Load tree
    let raw = std::fs::read_to_string(tree_file).context("read newick")?;
    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
    let t: NwkTree = one_from_string(&sanitized).context("parse newick (sanitized)")?;
    let mut lens_f32 = Vec::<f32>::new();
    let trav = SuccTrav::new(&t, &mut lens_f32);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    // Leaves and mapping name to leaf index
    let mut leaf_ids = Vec::<usize>::new();
    let mut leaf_nm = Vec::<String>::new();
    for n in t.nodes() {
        if t[n].is_leaf() {
            leaf_ids.push(n);
            leaf_nm.push(
                t.name(n)
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| format!("L{n}")),
            );
        }
    }
    let t2leaf: HashMap<&str, usize> = leaf_nm
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    // children & postorder
    let total = bp.len() + 1;
    lens_f32.resize(total, 0.0);
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    // Read table/biom and build leaf masks
    let (_taxa, mut samples, masks) = if let Some(tsv) = input_tsv {
        let (taxa, samples, mat) = read_table(tsv)?;
        let nsamp = samples.len();
        let mut masks: Vec<BitVec<u8, Lsb0>> = (0..nsamp)
            .map(|_| BitVec::repeat(false, leaf_ids.len()))
            .collect();
        for (ti, tax) in taxa.iter().enumerate() {
            if let Some(&leaf) = t2leaf.get(tax.as_str()) {
                for (s, bits) in masks.iter_mut().enumerate() {
                    if mat[ti][s] > 0.0 {
                        bits.set(leaf, true);
                    }
                }
            }
        }
        (taxa, samples, masks)
    } else {
        let biom = biom_h5.expect("biom path required when TSV not provided");
        let (taxa, samples, indptr, indices) = read_biom_csr(biom)?;
        let nsamp = samples.len();
        let mut masks: Vec<BitVec<u8, Lsb0>> = (0..nsamp)
            .map(|_| BitVec::repeat(false, leaf_ids.len()))
            .collect();
        for row in 0..taxa.len() {
            if let Some(&leaf) = t2leaf.get(taxa[row].as_str()) {
                let start = indptr[row] as usize;
                let stop = indptr[row + 1] as usize;
                for k in start..stop {
                    let s = indices[k] as usize;
                    masks[s].set(leaf, true);
                }
            }
        }
        (taxa, samples, masks)
    };

    let nsamp0 = samples.len();
    info!(
        "nodes = {}  leaves = {}  samples = {}",
        total,
        leaf_ids.len(),
        nsamp0
    );

    // node_bits[v]: bitset over samples
    let t0 = Instant::now();
    let node_bits = build_node_bits(&post, &kids, &leaf_ids, &masks, total);
    info!("node_bits built in {} ms", t0.elapsed().as_millis());
    // masks is not need after obtaining node_bits
    drop(masks);
    // Positive-length edges and weights
    let lens: Vec<f64> = lens_f32.iter().map(|&x| x as f64).collect();

    // Keep only edges with ℓ_e > 0 and present in at least one sample
    let active_edges: Vec<usize> = (0..total)
        .filter(|&v| lens[v] > 0.0 && node_bits[v].any())
        .collect();

    if active_edges.is_empty() {
        anyhow::bail!("No active edges: no positive-length branch is present in any sample.");
    }

    // Dense remap old edge id -> [0..active_edges.len())
    let mut id_map = vec![usize::MAX; total];
    for (new_id, &v) in active_edges.iter().enumerate() {
        id_map[v] = new_id;
    }
    info!(
        "active edges = {} (from {} total, {} leaves)",
        active_edges.len(),
        total,
        leaf_ids.len()
    );

    // Build weighted sets per sample on active edges
    let t1 = Instant::now();
    let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); samples.len()];
    for &v in &active_edges {
        let w = lens[v];
        let words = node_bits[v].as_raw_slice();
        for (wi, &w0) in words.iter().enumerate() {
            let mut word = w0;
            while word != 0 {
                let b = word.trailing_zeros() as usize;
                let s = (wi << 6) + b;
                if s < wsets.len() {
                    wsets[s].push((id_map[v] as u64, w));
                }
                word &= word - 1;
            }
        }
    }
    info!(
        "built per-sample weighted sets in {} ms",
        t1.elapsed().as_millis()
    );
    // keep memory tight
    drop(node_bits);
    drop(kids);
    drop(post);
    drop(leaf_ids);
    drop(t2leaf);
    drop(id_map);

    // Drop empty samples
    let empty_idx: Vec<usize> = wsets
        .iter()
        .enumerate()
        .filter_map(|(i, ws)| if ws.is_empty() { Some(i) } else { None })
        .collect();

    if !empty_idx.is_empty() {
        let show = empty_idx.len().min(20);
        for &s in empty_idx.iter().take(show) {
            warn!(
                "Dropping sample '{}' (no covered branches; all zeros).",
                samples[s]
            );
        }
        if empty_idx.len() > show {
            warn!("... and {} more empty samples.", empty_idx.len() - show);
        }
        // Filter wsets and sample names in lockstep
        let mut kept_ws = Vec::with_capacity(wsets.len() - empty_idx.len());
        let mut kept_names = Vec::with_capacity(samples.len() - empty_idx.len());
        for (i, ws) in wsets.into_iter().enumerate() {
            if !ws.is_empty() {
                kept_ws.push(ws);
                kept_names.push(samples[i].clone());
            }
        }
        wsets = kept_ws;
        samples = kept_names;
        info!("Kept {} non-empty samples.", samples.len());
        if samples.len() < 2 {
            anyhow::bail!("Fewer than 2 non-empty samples after filtering; nothing to compare.");
        }
    }
    info!("sketching starting...");
    // Sketch per sample in parallel
    let mut rng = mt_from_seed(seed);
    let sketches_u64: Vec<Vec<u64>> = if method == "dmh" {
        let dmh = DartMinHash::new_mt(&mut rng, k as u64);
        wsets
            .par_iter()
            .map(|ws| {
                dmh.sketch(ws)
                    .into_iter()
                    .map(|(id, _rank)| id)
                    .collect::<Vec<u64>>()
            })
            .collect()
    } else {
        let mut m_per_dim = vec![0u32; active_edges.len()];
        for (new_id, &v) in active_edges.iter().enumerate() {
            let mut cap = lens[v].ceil();
            if cap < 1.0 {
                cap = 1.0;
            }
            m_per_dim[new_id] = cap.min(u32::MAX as f64) as u32;
        }
        let ers = ErsWmh::new_mt(&mut rng, &m_per_dim, k as u64);
        wsets
            .par_iter()
            .map(|ws| {
                let sk = ers.sketch(ws, Some(ers_l));
                sk.into_iter().map(|(id, _rank)| id).collect::<Vec<u64>>()
            })
            .collect()
    };
    info!("sketching done.");
    // Everything heavy (tree structures, masks, node_bits, wsets, etc.) drops here.
    Ok((samples, sketches_u64))
}

// Weighted, build sketches for normalized weighted UniFrac
fn build_sketches_weighted(
    tree_file: &str,
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    k: usize,
    method: &str,
    ers_l: u64,
    seed: u64,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    // Load tree/topology (same as unweighted)
    let raw = std::fs::read_to_string(tree_file).context("read newick")?;
    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
    let t: NwkTree = one_from_string(&sanitized).context("parse newick (sanitized)")?;
    let mut lens_f32 = Vec::<f32>::new();
    let trav = SuccTrav::new(&t, &mut lens_f32);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    // Leaves & name→leaf map
    let mut leaf_ids = Vec::<usize>::new();
    let mut leaf_nm = Vec::<String>::new();
    for n in t.nodes() {
        if t[n].is_leaf() {
            leaf_ids.push(n);
            leaf_nm.push(
                t.name(n)
                    .map(|s| s.to_owned())
                    .unwrap_or_else(|| format!("L{n}")),
            );
        }
    }
    let t2leaf: HashMap<&str, usize> = leaf_nm
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    // children & postorder
    let total = bp.len() + 1;
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    lens_f32.resize(total, 0.0);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    // Read table/biom for counts/values
    let (_taxa, samples, _row2leaf, node_sums, has_mass, edges_touched_per_tid) = if let Some(tsv) = input_tsv {
        let (taxa, samples, counts) = read_table_counts(tsv)?;
        let nsamp = samples.len();
        let mut col_sums = vec![0.0f64; nsamp];
        for r in 0..taxa.len() {
            for s in 0..nsamp {
                col_sums[s] += counts[r][s];
            }
        }
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();

        let t0 = Instant::now();
        let (ns, has_mass, edges_touched_per_tid) = build_node_sums_weighted(
            &post, &kids, &leaf_ids, &row2leaf,
            WeightedMode::Dense { counts: &counts, col_sums: &col_sums },
            nsamp
        );
        info!("node_sums built in {} ms", t0.elapsed().as_millis());
        (taxa, samples, row2leaf, ns, has_mass, edges_touched_per_tid)
    } else {
        let biom = biom_h5.expect("biom path required when TSV not provided");
        let (taxa, samples, indptr, indices, data) = read_biom_csr_values(biom)?;
        let nsamp = samples.len();
        let mut col_sums = vec![0.0f64; nsamp];
        for r in 0..taxa.len() {
            let a = indptr[r] as usize;
            let b = indptr[r + 1] as usize;
            for k2 in a..b {
                col_sums[indices[k2] as usize] += data[k2];
            }
        }
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();
        let t0 = Instant::now();
        let (ns, has_mass, edges_touched_per_tid) = build_node_sums_weighted(
            &post, &kids, &leaf_ids, &row2leaf,
            WeightedMode::Csr { indptr: &indptr, indices: &indices, data: &data, col_sums: &col_sums },
            nsamp
        );
        info!("node_sums built in {} ms", t0.elapsed().as_millis());
        (taxa, samples, row2leaf, ns, has_mass, edges_touched_per_tid)
    };

    let nsamp = samples.len();
    let lens: Vec<f64> = lens_f32.iter().map(|&x| x as f64).collect();
    info!("nodes = {}  leaves = {}  samples = {}", total, leaf_ids.len(), nsamp);

    // has_mass already computed in build_node_sums_weighted
    let active_edges: Vec<usize> =
        (0..total).filter(|&v| lens[v] > 0.0 && has_mass[v]).collect();

    if active_edges.is_empty() {
        anyhow::bail!("No active edges for weighted case.");
    }

    // Dense ID remap  old v -> [0..active)
    let mut id_map = vec![usize::MAX; total];
    for (new_id, &v) in active_edges.iter().enumerate() {
        id_map[v] = new_id;
    }
    info!(
        "active edges = {} (from {} total, {} leaves)",
        active_edges.len(), total, leaf_ids.len()
    );
    // Build weighted sets per sample: z_v(s) = ℓ_v * A_v[s]
    let n_threads = rayon::current_num_threads().max(1);
    let stripe = (nsamp + n_threads - 1) / n_threads;
    let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];

    let t1 = Instant::now();

    wsets
        .as_mut_slice()
        .par_chunks_mut(stripe)
        .enumerate()
        .for_each(|(tid, target)| {
            if tid >= node_sums.len() || node_sums[tid].is_empty() { return; }
            let wloc = target.len(); // local samples in this stripe
            let touched = &edges_touched_per_tid[tid];

            // optional pre-reserve to reduce reallocations (cheap since touched is small)
            let mut counts = vec![0usize; wloc];
            for &v in touched {
                let id_new = id_map[v];
                if id_new == usize::MAX { continue; } // not active (ℓ==0 or never mass globally)
                let row = &node_sums[tid][v];
                debug_assert_eq!(row.len(), wloc);
                for (off, &a) in row.iter().enumerate() {
                    if a > 0.0 { counts[off] += 1; }
                }
            }
            for (off, c) in counts.into_iter().enumerate() {
                if c != 0 { target[off].reserve(c); }
            }

            // push only edges that had mass in this stripe
            for &v in touched {
                let id_new = id_map[v];
                if id_new == usize::MAX { continue; } // filter to active_edges
                let lw = lens[v];
                let id = id_new as u64;
                let row = &node_sums[tid][v];
                for (off, &a) in row.iter().enumerate() {
                    if a > 0.0 {
                        target[off].push((id, lw * a));
                    }
                }
            }
        });
    info!("built per-sample weighted sets in {} ms", t1.elapsed().as_millis());

    drop(node_sums);
    drop(kids);
    drop(post);
    drop(leaf_ids);
    drop(t2leaf);
    drop(id_map);

    // Drop empty samples (no covered branches)
    let mut kept_ws = Vec::with_capacity(nsamp);
    let mut kept_names = Vec::with_capacity(nsamp);
    for (i, ws) in wsets.into_iter().enumerate() {
        if !ws.is_empty() {
            kept_ws.push(ws);
            kept_names.push(samples[i].clone());
        }
    }
    let wsets = kept_ws;
    let samples = kept_names;
    if samples.len() < 2 {
        anyhow::bail!("Fewer than 2 non-empty samples; nothing to compare.");
    }
    info!("sketching starting...");
    // Sketch (DMH or ERS).  IMPORTANT: ERS `m_per_dim` must align with *dense* id space.
    let mut rng = mt_from_seed(seed);
    let sketches_u64: Vec<Vec<u64>> = if method == "dmh" {
        let dmh = DartMinHash::new_mt(&mut rng, k as u64);
        wsets
            .par_iter()
            .map(|ws| dmh.sketch(ws).into_iter().map(|(id, _rank)| id).collect())
            .collect()
    } else {
        let mut m_per_dim = vec![0u32; active_edges.len()];
        for (&v, new_id) in active_edges.iter().zip(0..) {
            let mut cap = lens[v].ceil();
            if cap < 1.0 {
                cap = 1.0;
            }
            m_per_dim[new_id] = cap.min(u32::MAX as f64) as u32;
        }
        let ers = ErsWmh::new_mt(&mut rng, &m_per_dim, k as u64);
        wsets
            .par_iter()
            .map(|ws| {
                ers.sketch(ws, Some(ers_l))
                    .into_iter()
                    .map(|(id, _rank)| id)
                    .collect()
            })
            .collect()
    };
    info!("sketching done.");
    Ok((samples, sketches_u64))
}

fn write_pcoa(
    sample_names: &[String],
    coords: &ndarray::Array2<f64>,
    prop_explained: &ndarray::Array1<f64>,
    path: &str,
) -> Result<()> {
    use std::io::Write;

    let n = coords.nrows();
    let k = coords.ncols();
    assert_eq!(sample_names.len(), n);

    let mut out = BufWriter::with_capacity(16 << 20, File::create(path)?);

    // Header row: "", PC1..PCk
    out.write_all(b"")?;
    for pc in 1..=k {
        out.write_all(b"\t")?;
        out.write_all(format!("PC{pc}").as_bytes())?;
    }
    out.write_all(b"\n")?;

    // Rows: sample_name, then coordinates
    let mut buf = ryu::Buffer::new();
    for i in 0..n {
        out.write_all(sample_names[i].as_bytes())?;
        for j in 0..k {
            out.write_all(b"\t")?;
            out.write_all(buf.format_finite(coords[[i, j]]).as_bytes())?;
        }
        out.write_all(b"\n")?;
    }

    // Blank line
    out.write_all(b"\n")?;

    // Header again for the rates (to match your request)
    out.write_all(b"")?;
    for pc in 1..=k {
        out.write_all(b"\t")?;
        out.write_all(format!("PC{pc}").as_bytes())?;
    }
    out.write_all(b"\n")?;

    // One row of 10 (or k) explanation rates
    out.write_all(b"proportion_explained")?;
    for j in 0..k {
        out.write_all(b"\t")?;
        out.write_all(buf.format_finite(prop_explained[j]).as_bytes())?;
    }
    out.write_all(b"\n")?;

    out.flush()?;
    Ok(())
}

fn write_pcoa_ordination(
    sample_names: &[String],
    coords: &Array2<f64>,
    eigenvalues: &Array1<f64>,
    proportion_explained: &Array1<f64>,
    path: &str,
) -> anyhow::Result<()> {
    use std::io::Write;

    let n = coords.nrows();
    let k = eigenvalues.len();
    assert_eq!(sample_names.len(), n, "sample_names length mismatch");
    assert_eq!(
        coords.ncols(),
        k,
        "coords.ncols() must equal eigenvalues.len()"
    );
    assert_eq!(
        proportion_explained.len(),
        k,
        "proportion_explained length mismatch"
    );

    let mut out = std::io::BufWriter::with_capacity(16 << 20, std::fs::File::create(path)?);
    let mut buf = ryu::Buffer::new();

    // Eigvals
    writeln!(out, "Eigvals\t{}", k)?;
    for j in 0..k {
        if j > 0 {
            out.write_all(b"\t")?;
        }
        out.write_all(buf.format_finite(eigenvalues[j]).as_bytes())?;
    }
    out.write_all(b"\n\n")?;

    // Proportion explained
    writeln!(out, "Proportion explained\t{}", k)?;
    for j in 0..k {
        if j > 0 {
            out.write_all(b"\t")?;
        }
        out.write_all(buf.format_finite(proportion_explained[j]).as_bytes())?;
    }
    out.write_all(b"\n\n")?;

    // Species
    writeln!(out, "Species\t0\t0")?;
    out.write_all(b"\n")?;

    // Site
    writeln!(out, "Site\t{}\t{}", n, k)?;
    for i in 0..n {
        out.write_all(sample_names[i].as_bytes())?;
        for j in 0..k {
            out.write_all(b"\t")?;
            out.write_all(buf.format_finite(coords[[i, j]]).as_bytes())?;
        }
        out.write_all(b"\n")?;
    }
    out.write_all(b"\n")?;

    // Biplot & Site constraints
    writeln!(out, "Biplot\t0\t0")?;
    out.write_all(b"\n")?;
    writeln!(out, "Site constraints\t0\t0")?;

    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("Logger initialized from default environment");

    let dart = emojis::get_by_shortcode("dart")
        .map(|e| e.as_str())
        .unwrap_or("🎯");

    let m = Command::new("dartunifrac")
        .version("0.2.2")
        .about(format!("DartUniFrac: Approximate UniFrac via Weighted MinHash {dart}{dart}{dart}"))
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
                .help("OTU/Feature table in TSV format"),
        )
        .arg(
            Arg::new("biom")
                .short('b')
                .long("biom")
                .help("OTU/Feature table in BIOM (HDF5) format"),
        )
        .group(
            ArgGroup::new("table").
            args(["input", "biom"]).
            required(true))
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .help("Output distance matrix in TSV format")
                .default_value("unifrac.tsv"),
        )
        .arg(
            Arg::new("weighted")
                .long("weighted")
                .help("Weighted UniFrac (normalized)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("sketch-size")
                .short('s')
                .long("sketch")
                .help("Sketch size for Weighted MinHash (DartMinHash or ERS)")
                .value_parser(clap::value_parser!(usize))
                .default_value("2048"),
        )
        .arg(
            Arg::new("method")
                .long("method")
                .short('m')
                .help("Sketching method: dmh (DartMinHash) or ers (Efficient Rejection Sampling)")
                .value_parser(["dmh", "ers"])
                .default_value("dmh"),
        )
        .arg(
            Arg::new("seq-length")
                .long("length")
                .short('l')
                .help("Per-hash independent random sequence length for ERS, must be >= 1024")
                .value_parser(clap::value_parser!(u64))
                // See Li and Li 2021 AAAI paper Figure 2. Large L has smaller bias and will be unbiased when L is unlimited (Rejection Sampling)
                // L should be determined by the sparsity of relevant branches for each sample
                .default_value("16384"),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .short('T')
                .help("Number of threads, default all logical cores")
                .value_parser(clap::value_parser!(usize)),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .help("Random seed for reproducibility")
                .value_parser(clap::value_parser!(u64))
                .default_value("1337"),
        )
        .arg(
            Arg::new("compress")
                .long("compress")
                .help("Compress output with zstd, .zst suffix will be added to the output file name")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("pcoa")
                .long("pcoa")
                .help("Fast Principle Coordinate Analysis based on Randomized SVD (subspace iteration), output saved to pcoa.txt/ordination.txt")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("streaming")
                .long("streaming")
                .help("Streaming the distance matrix while computing (zstd-compressed)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("block")
                .long("block")
                .help("Number of rows per chunk, streaming mode only")
                .value_parser(clap::value_parser!(usize)),
        )
        .get_matches();

    let tree_file = m.get_one::<String>("tree").unwrap();
    let input_tsv = m.get_one::<String>("input").map(|s| s.as_str());
    let biom_path = m.get_one::<String>("biom").map(|s| s.as_str());
    let out_file = m.get_one::<String>("output").unwrap();
    let k = *m.get_one::<usize>("sketch-size").unwrap();
    let weighted = m.get_flag("weighted");
    let method = m.get_one::<String>("method").unwrap().as_str();
    let ers_l = *m.get_one::<u64>("seq-length").unwrap();
    let seed = *m.get_one::<u64>("seed").unwrap();
    let compress = m.get_flag("compress");
    let pcoa = m.get_flag("pcoa");
    let stream = m.get_flag("streaming");
    let block = m.get_one::<usize>("block").copied();

    let threads = m
        .get_one::<usize>("threads")
        .copied()
        .unwrap_or_else(|| num_cpus::get());

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build_global()
        .unwrap();

    info!("{} threads will be used ", rayon::current_num_threads());

    info!("method={method}   k={k}   seed={seed}");
    if method == "ers" {
        info!("ERS L={ers_l}");
    }
    if weighted {
        info!("Weighted mode");
    } else {
        info!("Unweighted mode");
    };
    let (samples, sketches_u64) = if weighted {
        build_sketches_weighted(tree_file, input_tsv, biom_path, k, method, ers_l, seed)?
    } else {
        build_sketches(tree_file, input_tsv, biom_path, k, method, ers_l, seed)?
    };
    let nsamp = samples.len();
    if stream {
        if pcoa {
            warn!("--pcoa is incompatible with --stream; skipping PCoA in streaming mode.");
        }
        if compress {
            warn!(
                "--compress is ignored with --stream; streaming output is already zstd-compressed."
            );
        }
        let out_path_stream: PathBuf = if stream {
            let p_stream = Path::new(out_file);
            match p_stream.extension().and_then(|e| e.to_str()) {
                Some("zst") => p_stream.to_path_buf(),
                _ => PathBuf::from(format!("{out_file}.zst")),
            }
        } else {
            PathBuf::from(out_file)
        };

        // Convert to &str for your existing functions
        let out_path_stream_str = out_path_stream.to_string_lossy();

        info!(
            "Streaming zstd-compressed distance matrix → {}",
            out_path_stream_str
        );
        write_matrix_streaming_zstd(
            &samples,
            &sketches_u64,
            &out_path_stream_str,
            block,
            weighted,
        )?;
        info!("Done → {}", out_path_stream_str);
        return Ok(());
    }
    // Pairwise UniFrac (≈ 1 - Jaccard) via normalized Hamming on ID arrays.
    let t2 = Instant::now();
    let dist = {
        let n = nsamp;
        let dh = DistHamming;
        let mut out = vec![0.0f64; n * n];
        // this is the most computational expensive part (N^2/2 hamming similarity computation)
        out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            row[i] = 0.0;
            for j in (i + 1)..n {
                // DistHamming<u64> returns (# !=)/k ∈ [0,1]
                let mut d = dh.eval(&sketches_u64[i], &sketches_u64[j]) as f64; // d_J ≈ 1 - Jw
                if weighted {
                    // normalized weighted UniFrac = Bray–Curtis transform
                    d = if d < 2.0 { d / (2.0 - d) } else { 1.0 };
                }
                row[j] = d; // (i,j)
            }
        });

        // Mirror upper to lower.
        for i in 0..n {
            for j in (i + 1)..n {
                let v = out[i * n + j];
                out[j * n + i] = v;
            }
        }
        out
    };
    info!("pairwise distances in {} ms", t2.elapsed().as_millis());

    // Write output (fast ryu formatting) with compression (.zst)
    let out_path: PathBuf = if compress {
        let p = Path::new(out_file);
        match p.extension().and_then(|e| e.to_str()) {
            Some("zst") => p.to_path_buf(),
            _ => PathBuf::from(format!("{out_file}.zst")),
        }
    } else {
        PathBuf::from(out_file)
    };

    // Convert to &str for your existing functions
    let out_path_str = out_path.to_string_lossy();

    if compress {
        info!("Writing compressed (zstd) output → {}", out_path_str);
        write_matrix_zstd(&samples, &dist, nsamp, &out_path_str)?;
    } else {
        info!("Writing uncompressed output → {}", out_path_str);
        write_matrix(&samples, &dist, nsamp, &out_path_str)?;
    }
    info!("Done → {}", out_path_str);

    if pcoa {
        let n = nsamp;
        // take the ownership of dist to avoid copy, dist wrote to disk already
        let dm = Array2::from_shape_vec((n, n), dist).expect("distance matrix shape");

        let opts = FpcoaOptions {
            k: 10,
            oversample: 8,
            nbiter: 2,
            symmetrize_input: false,
        };

        info!(
            "Running randomized PCoA: k={}, oversample={}, iters={}",
            opts.k, opts.oversample, opts.nbiter
        );
        let t_pcoa = Instant::now();
        let res = pcoa_randomized(dm.view(), opts);
        info!("PCoA done in {} ms", t_pcoa.elapsed().as_millis());

        // Write ordination in simple format
        let pcoa_path = {
            let p_pcoa = std::path::Path::new(out_file);
            let mut pb_poca = p_pcoa.to_path_buf();
            pb_poca.set_file_name("pcoa.txt");
            pb_poca
        };
        // Write ordination in ordination format
        let ord_path = {
            let p = std::path::Path::new(out_file);
            let mut pb = p.to_path_buf();
            pb.set_file_name("ordination.txt");
            pb
        };
        write_pcoa(
            &samples,
            &res.coordinates,
            &res.proportion_explained,
            pcoa_path.to_str().unwrap(),
        )?;
        info!(
            "Writing pcoa and ordination results → {}",
            ord_path.display()
        );
        // ordination results
        write_pcoa_ordination(
            &samples,
            &res.coordinates,
            &res.eigenvalues,
            &res.proportion_explained,
            ord_path.to_str().unwrap(),
        )?;
    }

    Ok(())
}
