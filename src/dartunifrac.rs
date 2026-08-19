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
use clap::{Arg, ArgGroup, Command};
use env_logger;
use log::{info, warn};
use rayon::prelude::*;

use anndists::dist::{DistHamming, Distance};
// The engine. `Tree`/`Table` are renamed on import purely for readability: this
// file also handles the newick tree and the raw parsed table, so unqualified
// `Tree`/`Table` would not say which is which at the call site.
use dartunifrac_core::{
    CoreError, Method, SketchParams, Table as CoreTable, Tree as CoreTree,
};
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

use fpcoa::{FpcoaOptions, pcoa_randomized_inplace_f32};
use ndarray::{Array1, Array2};


const UNIFRAC_CITATIONS: &str = r#"
Citations:
  For DartUniFrac, please see:
    Zhao, J., McDonald, D., Sfiligoi, I., Lladser, M.E., Patel, L., Weng, Y., Khatib, L., Degregori, S., Gonzalez, A., Lozupone, C. and Knight, R., 2026. Enabling Megascale Microbiome Analysis with DartUniFrac. bioRxiv, pp.2026-03. doi: https://doi.org/10.64898/2026.03.01.708916
"#;

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

fn build_parent_and_lens_simple(t: &NwkTree) -> (Vec<usize>, Vec<f64>) {
    // find max node id so we can size vectors (NodeID is indexable)
    let mut max_id = 0usize;
    for v in t.nodes() {
        if v > max_id {
            max_id = v;
        }
    }
    let total = max_id + 1;

    // lengths: store branch length on the child node id (edge parent->child)
    let mut lens = vec![0.0f64; total];
    for v in t.nodes() {
        lens[v] = t[v].branch().copied().unwrap_or(0.0) as f64;
    }

    // parents: compute by rooted traversal to avoid parent-overwrite / undirected adjacency issues
    let mut parent = vec![usize::MAX; total];
    let mut visited = vec![false; total];

    let root = t.root();
    visited[root] = true;
    parent[root] = usize::MAX;

    let mut stack = vec![root];
    while let Some(v) = stack.pop() {
        for &c in t[v].children() {
            // If the underlying representation ever includes back-edges or shared adjacency,
            // this prevents assigning the wrong direction / overwriting an existing parent.
            if visited[c] {
                continue;
            }
            visited[c] = true;
            parent[c] = v;
            stack.push(c);
        }
    }

    // sanity check: everything reachable from root should be visited.
    // If this trips, the tree is disconnected or `children()` isn't the right neighbor API.
    debug_assert!(
        t.nodes().all(|v| visited[v]),
        "Tree traversal did not visit all nodes from root; parent pointers would be incomplete."
    );

    // Root has no incoming edge in UniFrac; prevent accidental inclusion if root has a branch length
    // lens[root] = 0.0;

    (parent, lens)
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

// Write TSV matrix (parallel formatting, block-wise, d is f32)
fn write_matrix(names: &[String], d: &[f32], n: usize, path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let file = File::create(path)?;
    let mut out = BufWriter::with_capacity(16 << 20, file);

    // Header: "", <names...>
    // Build in a single String to minimize write calls.
    let mut header = String::new();
    header.push_str(""); // first empty cell
    for name in names {
        header.push('\t');
        header.push_str(name);
    }
    header.push('\n');
    out.write_all(header.as_bytes())?;

    let nn = n;

    // Block size: default ~ sqrt(n) rows at a time.
    let block_rows = ((n as f64).sqrt() as usize).max(1);
    log::info!(
        "write_matrix: parallel block-wise writing with block_rows = {} (n = {})",
        block_rows,
        n
    );

    let mut i0 = 0usize;
    while i0 < nn {
        let h = (nn - i0).min(block_rows);

        // Build h rows in parallel, each as its own String.
        let lines: Vec<String> = (0..h)
            .into_par_iter()
            .map(|bi| {
                let i = i0 + bi;
                let mut line =
                    String::with_capacity(names[i].len() + 1 + nn * 12); // rough capacity

                // row label
                line.push_str(&names[i]);

                // row values
                let base = i * nn;
                let mut fmt = ryu::Buffer::new();
                for j in 0..nn {
                    line.push('\t');
                    let val: f32 = unsafe { *d.get_unchecked(base + j) };
                    line.push_str(fmt.format_finite(val));
                }
                line.push('\n');
                line
            })
            .collect();

        // Single writer to the buffered file.
        for line in &lines {
            out.write_all(line.as_bytes())?;
        }

        i0 += h;
    }

    out.flush()?;
    Ok(())
}

fn write_matrix_zstd(names: &[String], d: &[f32], n: usize, path: &str) -> Result<()> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    // zstd encoder (multi-threaded) + big buffer
    let file = File::create(path)?;
    let mut enc = zstd::Encoder::new(file, 0)?;
    let zstd_threads = rayon::current_num_threads() as u32;
    if zstd_threads > 1 {
        enc.multithread(zstd_threads)?;
    }
    let mut out = BufWriter::with_capacity(16 << 20, enc.auto_finish());

    // Header: "", <names...>
    let mut header = String::new();
    header.push_str(""); // first empty cell
    for name in names {
        header.push('\t');
        header.push_str(name);
    }
    header.push('\n');
    out.write_all(header.as_bytes())?;

    let nn = n;

    // Block size: default ~ sqrt(n) rows at a time.
    let block_rows = ((n as f64).sqrt() as usize).max(1);
    log::info!(
        "write_matrix_zstd: parallel block-wise writing with block_rows = {} (n = {})",
        block_rows,
        n
    );

    let mut i0 = 0usize;
    while i0 < nn {
        let h = (nn - i0).min(block_rows);

        // Build h rows in parallel, each as its own String.
        let lines: Vec<String> = (0..h)
            .into_par_iter()
            .map(|bi| {
                let i = i0 + bi;
                let mut line =
                    String::with_capacity(names[i].len() + 1 + nn * 12); // rough capacity

                // row label
                line.push_str(&names[i]);

                // row values
                let base = i * nn;
                let mut fmt = ryu::Buffer::new();
                for j in 0..nn {
                    line.push('\t');
                    let val: f32 = unsafe { *d.get_unchecked(base + j) };
                    line.push_str(fmt.format_finite(val));
                }
                line.push('\n');
                line
            })
            .collect();

        // Single writer to the compressed stream.
        for line in &lines {
            out.write_all(line.as_bytes())?;
        }

        i0 += h;
    }

    out.flush()?;
    Ok(())
}


// Streaming distances directly from sketches (for --streaming mode)
fn write_matrix_streaming_zstd_u64(
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
    let mut block = vec![0.0f32; n * bs];
    let dh = DistHamming;

    let mut i0 = 0usize;
    while i0 < n {
        let h = (n - i0).min(bs);

        // Fill block in parallel over columns; each `col` is a disjoint &mut [f32] (length = bs)
        block.par_chunks_mut(bs).enumerate().for_each(|(j, col)| {
            for bi in 0..h {
                let i = i0 + bi;
                let mut d: f32 = if i == j {
                    0.0f32
                } else {
                    dh.eval(&sketches[i], &sketches[j]) as f32
                };
                // d is an unbiased estimate of d_J = 1 - Jw
                if weighted_normalized {
                    // Bray–Curtis / normalized weighted UniFrac:
                    // D = (1 - Jw) / (1 + Jw) = d_J / (2 - d_J)
                    d = if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 };
                }
                col[bi] = d; // write into column-major slot (j, bi)
            }
        });

        // Write the block (single writer, amortized I/O)
        let mut lines: Vec<String> = (0..h)
            .into_par_iter()
            .map(|bi| {
                let i = i0 + bi;
                let mut line = String::with_capacity(8 + n * 12);
                line.push_str(&names[i]);

                let mut fmt = ryu::Buffer::new();
                for j in 0..n {
                    line.push('\t');
                    let v: f32 = block[j * bs + bi]; // column-major index (j, bi)
                    line.push_str(fmt.format_finite(v));
                }
                line.push('\n');
                line
            })
            .collect();

        for line in &mut lines {
            w.write_all(line.as_bytes())?;
            line.clear();
        }
        w.flush()?;
        i0 += h;
    }

    Ok(())
}

fn write_matrix_streaming_zstd_u16(
    names: &[String],
    sketches: &[Vec<u16>],
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
    let mut block = vec![0.0f32; n * bs];
    let dh = DistHamming;

    let mut i0 = 0usize;
    while i0 < n {
        let h = (n - i0).min(bs);

        // Fill block in parallel over columns; each `col` is a disjoint &mut [f32] (length = bs)
        block.par_chunks_mut(bs).enumerate().for_each(|(j, col)| {
            for bi in 0..h {
                let i = i0 + bi;
                let mut d: f32 = if i == j {
                    0.0f32
                } else {
                    dh.eval(&sketches[i], &sketches[j]) as f32
                };
                // d is an unbiased estimate of d_J = 1 - Jw
                if weighted_normalized {
                    // Bray–Curtis / normalized weighted UniFrac:
                    // D = (1 - Jw) / (1 + Jw) = d_J / (2 - d_J)
                    d = if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 };
                }
                col[bi] = d; // write into column-major slot (j, bi)
            }
        });

        // Write the block (single writer, amortized I/O)
        let mut lines: Vec<String> = (0..h)
            .into_par_iter()
            .map(|bi| {
                let i = i0 + bi;
                let mut line = String::with_capacity(8 + n * 12);
                line.push_str(&names[i]);

                let mut fmt = ryu::Buffer::new();
                for j in 0..n {
                    line.push('\t');
                    let v: f32 = block[j * bs + bi]; // column-major index (j, bi)
                    line.push_str(fmt.format_finite(v));
                }
                line.push('\n');
                line
            })
            .collect();

        for line in &mut lines {
            w.write_all(line.as_bytes())?;
            line.clear();
        }
        w.flush()?;
        i0 += h;
    }

    Ok(())
}

fn write_matrix_streaming_zstd_u32(
    names: &[String],
    sketches: &[Vec<u32>],
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
    let mut block = vec![0.0f32; n * bs];
    let dh = DistHamming;

    let mut i0 = 0usize;
    while i0 < n {
        let h = (n - i0).min(bs);

        // Fill block in parallel over columns; each `col` is a disjoint &mut [f32] (length = bs)
        block.par_chunks_mut(bs).enumerate().for_each(|(j, col)| {
            for bi in 0..h {
                let i = i0 + bi;
                let mut d: f32 = if i == j {
                    0.0f32
                } else {
                    dh.eval(&sketches[i], &sketches[j]) as f32
                };
                // d is an unbiased estimate of d_J = 1 - Jw
                if weighted_normalized {
                    // Bray–Curtis / normalized weighted UniFrac:
                    // D = (1 - Jw) / (1 + Jw) = d_J / (2 - d_J)
                    d = if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 };
                }
                col[bi] = d; // write into column-major slot (j, bi)
            }
        });

        // Write the block (single writer, amortized I/O)
        let mut lines: Vec<String> = (0..h)
            .into_par_iter()
            .map(|bi| {
                let i = i0 + bi;
                let mut line = String::with_capacity(8 + n * 12);
                line.push_str(&names[i]);

                let mut fmt = ryu::Buffer::new();
                for j in 0..n {
                    line.push('\t');
                    let v: f32 = block[j * bs + bi]; // column-major index (j, bi)
                    line.push_str(fmt.format_finite(v));
                }
                line.push('\n');
                line
            })
            .collect();

        for line in &mut lines {
            w.write_all(line.as_bytes())?;
            line.clear();
        }
        w.flush()?;
        i0 += h;
    }

    Ok(())
}




// ---------------------------------------------------------------------------
// Tree loading and table marshaling for `dartunifrac-core`.
//
// The engine takes a tree as parent pointers plus branch lengths and a feature
// table already resolved to node ids, so everything that touches a file, a
// feature name, or an input format lives here. The four `build_sketches*`
// entry points below keep their original signatures and are now thin adapters.
// ---------------------------------------------------------------------------

/// Leaf node ids and their names, in the `newick` crate's node order.
///
/// An unnamed leaf gets the synthetic name `L{id}`, which cannot match a real
/// feature id — such a tip is simply never populated.
fn leaf_index(t: &NwkTree) -> (Vec<usize>, Vec<String>) {
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
    (leaf_ids, leaf_nm)
}

fn parse_tree(tree_file: &str) -> Result<NwkTree> {
    let raw = std::fs::read_to_string(tree_file).context("read newick")?;
    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
    one_from_string(&sanitized).context("parse newick (sanitized)")
}

/// Parent pointers and branch lengths via the succinct balanced-parens tree.
///
/// Note the two id spaces in play: `SuccTrav` fills `lens` against the `newick`
/// crate's node ids, while `collect_children` builds the child lists against
/// `BalancedParensTree` ids. They coincide because both derive from the same DFS
/// order, and the branch lengths round through f32 on the way — both are relied
/// upon for bit-identical output, so neither may be "cleaned up".
fn tree_arrays_succ(t: &NwkTree) -> CoreTree {
    let mut lens_f32 = Vec::<f32>::new();
    let trav = SuccTrav::new(t, &mut lens_f32);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    let total = bp.len() + 1;
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    lens_f32.resize(total, 0.0);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    CoreTree {
        parent: compute_parent(total, &kids),
        lens: lens_f32.iter().map(|&x| x as f64).collect(),
    }
}

/// Parent pointers and branch lengths straight from the parsed Newick tree.
fn tree_arrays_simple(t: &NwkTree) -> CoreTree {
    let (parent, lens) = build_parent_and_lens_simple(t);
    CoreTree { parent, lens }
}

/// Map feature rows onto leaf ordinals; `None` for a feature absent from the tree.
fn row_to_leaf(taxa: &[String], leaf_nm: &[String]) -> Vec<Option<usize>> {
    let t2leaf: HashMap<&str, usize> = leaf_nm
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();
    taxa.iter()
        .map(|n| t2leaf.get(n.as_str()).copied())
        .collect()
}

/// Dense TSV to CSC over node ids.
///
/// Walks features in ascending row order within each sample, which is the order
/// the accumulation used before the engine moved out, and drops non-positive
/// values — a filter the dense reader applies and the BIOM reader does not.
/// `col_sums`, when weighted, is summed over *every* row including features the
/// tree does not contain.
fn table_from_tsv(
    taxa: &[String],
    counts: &[Vec<f64>],
    row2leaf: &[Option<usize>],
    leaf_ids: &[usize],
    nsamp: usize,
    weighted: bool,
) -> CoreTable {
    let mut col_sums = vec![0.0f64; nsamp];
    if weighted {
        for r in 0..taxa.len() {
            for s in 0..nsamp {
                col_sums[s] += counts[r][s];
            }
        }
    }

    let mut colptr = vec![0usize; nsamp + 1];
    let mut node = Vec::<usize>::new();
    let mut value = Vec::<f64>::new();
    for s in 0..nsamp {
        for (r, lopt) in row2leaf.iter().enumerate() {
            let lp = match lopt {
                Some(v) => *v,
                None => continue,
            };
            let val = counts[r][s];
            if val <= 0.0 {
                continue;
            }
            node.push(leaf_ids[lp]);
            value.push(val);
        }
        colptr[s + 1] = node.len();
    }

    CoreTable {
        n_samples: nsamp,
        colptr,
        node,
        value,
        col_sums,
    }
}

/// BIOM CSR to CSC over node ids.
///
/// Every stored entry counts, with no value filter — matching the BIOM path as
/// it was, which means a stored zero marks a feature present and a negative
/// weight survives into the accumulation.
fn table_from_biom(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    row2leaf: &[Option<usize>],
    leaf_ids: &[usize],
    nsamp: usize,
    weighted: bool,
    log_label: &str,
) -> CoreTable {
    let mut col_sums = vec![0.0f64; nsamp];
    if weighted {
        for r in 0..(indptr.len() - 1) {
            for k in indptr[r] as usize..indptr[r + 1] as usize {
                col_sums[indices[k] as usize] += data[k];
            }
        }
    }

    info!("{log_label}transposing BIOM CSR→CSC …");
    let (csc_ptr, rowind, vals) = csr_to_csc(indptr, indices, data, nsamp);

    let mut colptr = vec![0usize; nsamp + 1];
    let mut node = Vec::<usize>::new();
    let mut value = Vec::<f64>::new();
    for s in 0..nsamp {
        for kk in csc_ptr[s]..csc_ptr[s + 1] {
            let lp = match row2leaf[rowind[kk]] {
                Some(v) => v,
                None => continue,
            };
            node.push(leaf_ids[lp]);
            value.push(vals[kk]);
        }
        colptr[s + 1] = node.len();
    }

    CoreTable {
        n_samples: nsamp,
        colptr,
        node,
        value,
        col_sums,
    }
}

/// Translate an engine error into this binary's historical wording.
///
/// The engine reports "fewer than two survivors" as a value because the C API
/// must turn it into an empty result rather than a failure; here it stays fatal.
fn core_err(e: CoreError, few_msg: &str, no_edges_msg: &str) -> anyhow::Error {
    match e {
        CoreError::FewerThanTwoNonEmptySamples => anyhow::anyhow!("{few_msg}"),
        CoreError::NoActiveEdges => anyhow::anyhow!("{no_edges_msg}"),
        CoreError::MalformedTable(m) => anyhow::anyhow!("malformed input: {m}"),
    }
}

fn method_from_str(method: &str) -> Result<Method> {
    match method {
        "dmh" => Ok(Method::Dmh),
        "tmh" => Ok(Method::Tmh),
        "ers" => Ok(Method::Ers),
        other => anyhow::bail!("unsupported sketching method: {other}"),
    }
}

/// Read the requested table and marshal it for the engine.
fn load_table(
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    leaf_ids: &[usize],
    leaf_nm: &[String],
    weighted: bool,
    log_label: &str,
) -> Result<(Vec<String>, CoreTable)> {
    if let Some(tsv) = input_tsv {
        let (taxa, samples, counts) = if weighted {
            read_table_counts(tsv)?
        } else {
            read_table(tsv)?
        };
        let row2leaf = row_to_leaf(&taxa, leaf_nm);
        let nsamp = samples.len();
        let table = table_from_tsv(&taxa, &counts, &row2leaf, leaf_ids, nsamp, weighted);
        Ok((samples, table))
    } else {
        let biom = biom_h5.expect("biom path required when TSV not provided");
        let (taxa, samples, indptr, indices, data) = if weighted {
            read_biom_csr_values(biom)?
        } else {
            let (taxa, samples, indptr, indices) = read_biom_csr(biom)?;
            // Presence only, so the values are synthesized rather than read.
            let ones = vec![1.0; indices.len()];
            (taxa, samples, indptr, indices, ones)
        };
        let row2leaf = row_to_leaf(&taxa, leaf_nm);
        let nsamp = samples.len();
        let table = table_from_biom(
            &indptr, &indices, &data, &row2leaf, leaf_ids, nsamp, weighted, log_label,
        );
        Ok((samples, table))
    }
}

/// Shared body of the four entry points: load, marshal, sketch, name.
fn sketch_via_core(
    tree: CoreTree,
    samples: Vec<String>,
    table: CoreTable,
    params: &SketchParams,
    log_label: &str,
    few_msg: &str,
    no_edges_msg: &str,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    if samples.len() < 2 {
        anyhow::bail!("Fewer than 2 samples; nothing to compare.");
    }

    let out = dartunifrac_core::build_sketches(&tree, &table, params, log_label)
        .map_err(|e| core_err(e, few_msg, no_edges_msg))?;

    let names = out.kept.iter().map(|&i| samples[i].clone()).collect();
    Ok((names, out.sketches))
}

// Unweighted, succinct tree.
fn build_sketches(
    tree_file: &str,
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    k: usize,
    method: &str,
    ers_l: u64,
    seed: u64,
    portable: bool,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    let t = parse_tree(tree_file)?;
    let (leaf_ids, leaf_nm) = leaf_index(&t);
    let tree = tree_arrays_succ(&t);
    let (samples, table) = load_table(input_tsv, biom_h5, &leaf_ids, &leaf_nm, false, "")?;
    let params = SketchParams {
        k,
        method: method_from_str(method)?,
        ers_l,
        seed,
        weighted: false,
        raw_counts: false,
        portable,
    };
    sketch_via_core(
        tree,
        samples,
        table,
        &params,
        "",
        "Fewer than 2 non-empty samples after filtering; nothing to compare.",
        "No active edges after presence accumulation.",
    )
}

/// Build parent pointers from children lists. Root will have usize::MAX.
fn compute_parent(total: usize, kids: &[Vec<usize>]) -> Vec<usize> {
    let mut parent = vec![usize::MAX; total];
    for v in 0..total {
        for &c in &kids[v] {
            parent[c] = v;
        }
    }
    parent
}

/// CSR (rows=features, cols=samples) to CSC (cols=samples) for fast per-sample scans.
/// Returns (colptr, rowind, vals) with colptr.len()==nsamp+1, rowind/vals.len()==nnz.
fn csr_to_csc(
    indptr: &[u32],
    indices: &[u32],
    data: &[f64],
    nsamp: usize,
) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let nnz = data.len();
    let mut col_counts = vec![0usize; nsamp];
    for &sidx in indices {
        col_counts[sidx as usize] += 1;
    }

    let mut colptr = vec![0usize; nsamp + 1];
    for i in 0..nsamp {
        colptr[i + 1] = colptr[i] + col_counts[i];
    }

    let mut cur = colptr.clone();
    let mut rowind = vec![0usize; nnz];
    let mut vals = vec![0f64; nnz];

    for r in 0..(indptr.len() - 1) {
        let a = indptr[r] as usize;
        let b = indptr[r + 1] as usize;
        for k in a..b {
            let s = indices[k] as usize;
            let dst = cur[s];
            rowind[dst] = r;
            vals[dst] = data[k];
            cur[s] += 1;
        }
    }
    (colptr, rowind, vals)
}

// Weighted, succinct tree.
//
// By default the table values are converted to relative abundance per sample.
// With `raw_counts`, raw counts go into the branch accumulation directly.
fn build_sketches_weighted(
    tree_file: &str,
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    k: usize,
    method: &str,
    ers_l: u64,
    seed: u64,
    raw_counts: bool,
    portable: bool,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    let t = parse_tree(tree_file)?;
    let (leaf_ids, leaf_nm) = leaf_index(&t);
    let tree = tree_arrays_succ(&t);
    info!("nodes = {}  leaves = {}", tree.parent.len(), leaf_ids.len());
    let (samples, table) = load_table(input_tsv, biom_h5, &leaf_ids, &leaf_nm, true, "")?;
    let params = SketchParams {
        k,
        method: method_from_str(method)?,
        ers_l,
        seed,
        weighted: true,
        raw_counts,
        portable,
    };
    sketch_via_core(
        tree,
        samples,
        table,
        &params,
        "",
        "Fewer than 2 non-empty samples; nothing to compare.",
        "No active edges for weighted case.",
    )
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

    // Header again for the rates
    out.write_all(b"")?;
    for pc in 1..=k {
        out.write_all(b"\t")?;
        out.write_all(format!("PC{pc}").as_bytes())?;
    }
    out.write_all(b"\n")?;

    // One row of explanation rates
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

// Unweighted, plain Newick tree.
fn build_sketches_simple(
    tree_file: &str,
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    k: usize,
    method: &str,
    ers_l: u64,
    seed: u64,
    portable: bool,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    let t = parse_tree(tree_file)?;
    let (leaf_ids, leaf_nm) = leaf_index(&t);
    let tree = tree_arrays_simple(&t);
    let (samples, table) =
        load_table(input_tsv, biom_h5, &leaf_ids, &leaf_nm, false, "(simple) ")?;
    let params = SketchParams {
        k,
        method: method_from_str(method)?,
        ers_l,
        seed,
        weighted: false,
        raw_counts: false,
        portable,
    };
    sketch_via_core(
        tree,
        samples,
        table,
        &params,
        "(simple) ",
        "Fewer than 2 non-empty samples after filtering; nothing to compare.",
        "No active edges after presence accumulation.",
    )
}

// Weighted, plain Newick tree.
//
// By default the table values are converted to relative abundance per sample.
// With `raw_counts`, raw counts go into the branch accumulation directly.
fn build_sketches_weighted_simple(
    tree_file: &str,
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    k: usize,
    method: &str,
    ers_l: u64,
    seed: u64,
    raw_counts: bool,
    portable: bool,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    let t = parse_tree(tree_file)?;
    let (leaf_ids, leaf_nm) = leaf_index(&t);
    let tree = tree_arrays_simple(&t);
    info!("(simple) nodes = {}  leaves = {}", tree.parent.len(), leaf_ids.len());
    let (samples, table) =
        load_table(input_tsv, biom_h5, &leaf_ids, &leaf_nm, true, "(simple) ")?;
    let params = SketchParams {
        k,
        method: method_from_str(method)?,
        ers_l,
        seed,
        weighted: true,
        raw_counts,
        portable,
    };
    sketch_via_core(
        tree,
        samples,
        table,
        &params,
        "(simple) ",
        "Fewer than 2 non-empty samples; nothing to compare.",
        "No active edges for weighted case.",
    )
}

enum Sketches {
    U16(Vec<Vec<u16>>),
    U32(Vec<Vec<u32>>),
    U64(Vec<Vec<u64>>),
}

fn trim_sketches_to_bbits(sk: Vec<Vec<u64>>, bbits: u8) -> Sketches {
    match bbits {
        16 => {
            let out: Vec<Vec<u16>> = sk
                .into_iter()
                .map(|row| row.into_iter().map(|x| x as u16).collect())
                .collect();
            Sketches::U16(out)
        }
        32 => {
            let out: Vec<Vec<u32>> = sk
                .into_iter()
                .map(|row| row.into_iter().map(|x| x as u32).collect())
                .collect();
            Sketches::U32(out)
        }
        64 => Sketches::U64(sk),
        other => {
            // Should not happen if main normalizes, but keep it robust.
            warn!("trim_sketches_to_bbits: bbits={} not supported; using 16.", other);
            let out: Vec<Vec<u16>> = sk
                .into_iter()
                .map(|row| row.into_iter().map(|x| x as u16).collect())
                .collect();
            Sketches::U16(out)
        }
    }
}

fn main() -> Result<()> {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("Logger initialized from default environment");

    let dart = emojis::get_by_shortcode("dart")
        .map(|e| e.as_str())
        .unwrap_or("🎯");

    let m = Command::new("dartunifrac")
        .version("0.3.2")
        .about(format!(
            "DartUniFrac: Approximate UniFrac via Weighted MinHash {dart}{dart}{dart}"
        ))
        .after_help(UNIFRAC_CITATIONS)
        .after_long_help(UNIFRAC_CITATIONS)
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
        .group(ArgGroup::new("table").args(["input", "biom"]).required(true))
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
                .help("Weighted UniFrac (normalized by default)")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("raw-counts")
                .long("raw-counts")
                .help("Weighted mode only: pass raw table counts into branch accumulation instead of converting each sample to relative abundance")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("succ")
                .long("succ")
                .help("Use succparen balanced-parentheses tree representation")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("sketch-size")
                .short('s')
                .long("sketch")
                .help("Sketch size for Weighted MinHash (DartMinHash, TreeMinHash, or ERS)")
                .value_parser(clap::value_parser!(usize))
                .default_value("2048"),
        )
        .arg(
            Arg::new("method")
                .long("method")
                .short('m')
                .help("Sketching method: dmh (DartMinHash), tmh (TreeMinHash), or ers (Efficient Rejection Sampling)")
                .value_parser(["dmh", "tmh", "ers"])
                .default_value("dmh"),
        )
        .arg(
            Arg::new("bbits")
                .long("bbits")
                .help("Extracting lower bits from hashes. Supported: 16 (default), 32, 64.")
                .value_parser(clap::value_parser!(u8))
                .default_value("16"),
        )
        .arg(
            Arg::new("portable-sketches")
                .long("portable-sketches")
                .help(
                    "Derive the branch id space from the tree alone instead of from the samples in this run, \
                     so sketches of the same sample are identical across runs and can be compared or merged. \
                     Changes the distances (equivalent to a different --seed); off by default",
                )
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("seq-length")
                .long("length")
                .short('l')
                .help("Per-hash independent random sequence length for ERS, must be >= 512")
                .value_parser(clap::value_parser!(u64))
                // See Li and Li 2021 AAAI paper Figure 2. Large L has smaller bias and will be unbiased when L goes unlimited (Rejection Sampling)
                // L should be determined by the sparsity of relevant branches for each sample
                .default_value("2048"),
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
                .help("Fast Principal Coordinate Analysis based on Randomized SVD (subspace iteration), output saved to pcoa.txt/ordination.txt")
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
    let raw_counts = m.get_flag("raw-counts");
    let method = m.get_one::<String>("method").unwrap().as_str();
    let ers_l = *m.get_one::<u64>("seq-length").unwrap();
    let seed = *m.get_one::<u64>("seed").unwrap();
    let compress = m.get_flag("compress");
    let pcoa = m.get_flag("pcoa");
    let stream = m.get_flag("streaming");
    let block = m.get_one::<usize>("block").copied();
    let succ = m.get_flag("succ");
    let bbits_in = *m.get_one::<u8>("bbits").unwrap();
    let bbits: u8 = match bbits_in {
        16 | 32 | 64 => bbits_in,
        other => {
            warn!("--bbits={} not supported; using 16 (supported: 16/32/64).", other);
            16
        }
    };
    info!("bbits = {}", bbits);

    let portable = m.get_flag("portable-sketches");
    // ERS is refused under portable sketches, for two separate reasons.
    //
    // Weighted ERS caps each branch by the largest weight seen among the run's
    // samples, so its sketches stay run-dependent however branches are numbered
    // -- portability is simply unattainable.
    //
    // Unweighted ERS caps by branch length, which does come from the tree, so it
    // would be portable. But a tree-derived id space spans every positive-length
    // branch rather than only the touched ones, which inflates the cap total and
    // so lowers the acceptance rate; for a sparse batch that risks additional
    // finite-L bias. Rather than ship a combination whose bias depends on how
    // much of the tree a batch happens to cover, refuse it.
    if portable && method == "ers" {
        anyhow::bail!(
            "--portable-sketches cannot be combined with -m ers. Weighted ERS caps each branch by \
             the largest weight among the run's samples, so its sketches are not comparable across \
             runs. Unweighted ERS would be comparable, but a tree-derived id space spans every \
             positive-length branch, inflating the cap total and lowering the acceptance rate, \
             which can add finite-L bias for batches covering little of the tree. \
             Use -m dmh or -m tmh for portable sketches."
        );
    }
    if portable {
        info!("Portable sketches: branch id space derived from the tree, not from this sample set");
    }

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
    if method == "tmh" {
        info!("TreeMinHash enabled");
    }
    if method == "ers" {
        info!("ERS L={ers_l}");
    }
    if weighted {
        if raw_counts {
            info!("Weighted mode with raw counts: no per-sample relative-abundance normalization before branch accumulation");
        } else {
            info!("Weighted mode with relative abundance normalization");
        }
    } else {
        if raw_counts {
            warn!("--raw-counts was set but ignored because --weighted was not set");
        }
        info!("Unweighted mode");
    };
    if succ {
        info!("Using succparen balanced-parentheses tree representation (--succ)");
    } else {
        info!("Using simple Newick tree parsing (default)");
    }

    let (samples, sketches_u64): (Vec<String>, Vec<Vec<u64>>) =
    if weighted {
        if succ {
            build_sketches_weighted(tree_file, input_tsv, biom_path, k, method, ers_l, seed, raw_counts, portable)?
        } else {
            build_sketches_weighted_simple(tree_file, input_tsv, biom_path, k, method, ers_l, seed, raw_counts, portable)?
        }
    } else {
        if succ {
            build_sketches(tree_file, input_tsv, biom_path, k, method, ers_l, seed, portable)?
        } else {
            build_sketches_simple(tree_file, input_tsv, biom_path, k, method, ers_l, seed, portable)?
        }
    };
    let sketches = trim_sketches_to_bbits(sketches_u64, bbits);
    let nsamp = samples.len();

    // Streaming mode: compute Hamming on the fly from sketches and stream to disk
    if stream {
        if pcoa {
            warn!("--pcoa is incompatible with --stream; skipping PCoA in streaming mode.");
        }
        if compress {
            warn!(
                "--compress is ignored with --stream; streaming output is already zstd-compressed."
            );
        }
        let out_path_stream: PathBuf = {
            let p_stream = Path::new(out_file);
            match p_stream.extension().and_then(|e| e.to_str()) {
                Some("zst") => p_stream.to_path_buf(),
                _ => PathBuf::from(format!("{out_file}.zst")),
            }
        };

        let out_path_stream_str = out_path_stream.to_string_lossy();

        info!(
            "Streaming zstd-compressed distance matrix → {}",
            out_path_stream_str
        );
        match &sketches {
            Sketches::U16(s) => write_matrix_streaming_zstd_u16(&samples, s, &out_path_stream_str, block, weighted)?,
            Sketches::U32(s) => write_matrix_streaming_zstd_u32(&samples, s, &out_path_stream_str, block, weighted)?,
            Sketches::U64(s) => write_matrix_streaming_zstd_u64(&samples, s, &out_path_stream_str, block, weighted)?,
        }
        info!("Done → {}", out_path_stream_str);
        return Ok(());
    }

    macro_rules! compute_pairwise_from_sketches {
        ($sketches:expr, $weighted:expr) => {{
            let n = $sketches.len();
            let dh = DistHamming;
            let mut out = vec![0.0f32; n * n];

            out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                row[i] = 0.0f32;
                for j in (i + 1)..n {
                    let mut d: f32 = dh.eval(&$sketches[i], &$sketches[j]) as f32; // d_J ≈ 1 - Jw
                    if $weighted {
                        d = if d < 2.0f32 { d / (2.0f32 - d) } else { 1.0f32 };
                    }
                    row[j] = d;
                }
            });

            for i in 0..n {
                for j in (i + 1)..n {
                    let v = out[i * n + j];
                    out[j * n + i] = v;
                }
            }
            out
        }};
    }

    // Pairwise UniFrac (≈ 1 - Jaccard) via normalized Hamming on ID arrays (full N×N in f32)
    let t2 = Instant::now();
    let dist: Vec<f32> = match &sketches {
        Sketches::U16(s) => compute_pairwise_from_sketches!(s, weighted),
        Sketches::U32(s) => compute_pairwise_from_sketches!(s, weighted),
        Sketches::U64(s) => compute_pairwise_from_sketches!(s, weighted),
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
        let mut dm_f32 = Array2::from_shape_vec((n, n), dist).expect("distance matrix shape");

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
        let res = pcoa_randomized_inplace_f32(&mut dm_f32, opts);
        info!("PCoA done in {} ms", t_pcoa.elapsed().as_millis());

        let pcoa_path = {
            let p_pcoa = std::path::Path::new(out_file);
            let mut pb_pcoa = p_pcoa.to_path_buf();
            pb_pcoa.set_file_name("pcoa.txt");
            pb_pcoa
        };
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

#[cfg(test)]
mod tests {
    use super::*;

    const TREE: &str = "data/ASVs_aligned.tre";
    const TABLE: &str = "data/ASVs_counts.txt";
    const K: usize = 2048;
    const SEED: u64 = 1337;

    /// Copy the fixture table keeping only its first `n_samples` columns, so the
    /// retained samples carry byte-identical counts in both the full and the
    /// subset table. Any change in their sketches is then attributable to the
    /// sample set alone.
    fn subset_table(n_samples: usize, tag: &str) -> PathBuf {
        let src = std::fs::read_to_string(TABLE).expect("read fixture table");
        let mut out = String::new();
        for line in src.lines() {
            let cols: Vec<&str> = line.split('\t').collect();
            out.push_str(&cols[..=n_samples].join("\t"));
            out.push('\n');
        }
        let dest = std::env::temp_dir().join(format!(
            "dartunifrac_{tag}_{n_samples}_{}.tsv",
            std::process::id()
        ));
        std::fs::write(&dest, out).expect("write subset table");
        dest
    }

    fn sketch(table: &Path, weighted: bool, portable: bool) -> (Vec<String>, Vec<Vec<u64>>) {
        let table = table.to_str().unwrap();
        if weighted {
            build_sketches_weighted_simple(
                TREE, Some(table), None, K, "dmh", 2048, SEED, false, portable,
            )
        } else {
            build_sketches_simple(TREE, Some(table), None, K, "dmh", 2048, SEED, portable)
        }
        .expect("build sketches")
    }

    /// Sketches for the samples common to both runs, keyed by sample id.
    fn shared(
        a: &(Vec<String>, Vec<Vec<u64>>),
        b: &(Vec<String>, Vec<Vec<u64>>),
    ) -> Vec<(String, Vec<u64>, Vec<u64>)> {
        a.0.iter()
            .enumerate()
            .filter_map(|(i, name)| {
                b.0.iter()
                    .position(|other| other == name)
                    .map(|j| (name.clone(), a.1[i].clone(), b.1[j].clone()))
            })
            .collect()
    }

    /// With --portable-sketches the branch id space comes from the tree, so a
    /// sample's sketch must not depend on which other samples were in the run.
    /// This is what makes sketches from separate runs comparable and mergeable.
    #[test]
    fn portable_sketches_do_not_depend_on_the_sample_set() {
        for weighted in [false, true] {
            let subset = subset_table(3, "portable");
            let few = sketch(&subset, weighted, true);
            let all = sketch(Path::new(TABLE), weighted, true);
            let common = shared(&few, &all);
            assert_eq!(common.len(), 3, "expected the 3 subset samples in both runs");
            for (name, from_subset, from_full) in common {
                assert_eq!(
                    from_subset, from_full,
                    "weighted={weighted}: sketch for {name} changed with the sample set"
                );
            }
            let _ = std::fs::remove_file(&subset);
        }
    }

    /// The default id space is compacted over the branches touched by the run's
    /// own samples, so the same sample sketches differently depending on its
    /// company. This is the behaviour --portable-sketches exists to opt out of;
    /// pinning it here keeps the flag honest.
    #[test]
    fn default_sketches_depend_on_the_sample_set() {
        let subset = subset_table(3, "default");
        let few = sketch(&subset, true, false);
        let all = sketch(Path::new(TABLE), true, false);
        let common = shared(&few, &all);
        assert_eq!(common.len(), 3);
        assert!(
            common.iter().any(|(_, a, b)| a != b),
            "expected the default id space to make sketches run-dependent"
        );
        let _ = std::fs::remove_file(&subset);
    }

    /// Turning the flag on must not disturb a run whose samples already touch
    /// every positive-length branch: the two id spaces coincide there.
    #[test]
    fn portable_matches_default_when_samples_cover_the_tree() {
        let all_default = sketch(Path::new(TABLE), true, false);
        let all_portable = sketch(Path::new(TABLE), true, true);
        assert_eq!(all_default.0, all_portable.0);
        assert_eq!(all_default.1, all_portable.1);
    }
}
