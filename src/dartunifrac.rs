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

use fpcoa::{FpcoaOptions, pcoa_randomized_inplace_f32};
use ndarray::{Array1, Array2};

#[cfg(feature = "cuda")]
mod disthamming_gpu;


const UNIFRAC_CITATIONS: &str = r#"
Citations:
  For DartUniFrac, please see:
  Zhao J. et.al., 2025, DOI:
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

    // branch lengths indexed by bp node id
    lens_bp: &'a mut Vec<f32>,

    // map: newick node id -> bp node id
    nwk2bp: &'a mut Vec<usize>,

    // next bp id assigned in traversal order
    bp_next: usize,
}

impl<'a> SuccTrav<'a> {
    fn new(t: &'a NwkTree, lens_bp: &'a mut Vec<f32>, nwk2bp: &'a mut Vec<usize>) -> Self {
        Self {
            t,
            stack: vec![(t.root(), 0, 0)],
            lens_bp,
            nwk2bp,
            bp_next: 0,
        }
    }
}

impl<'a> DepthFirstTraverse for SuccTrav<'a> {
    type Label = ();

    fn next(&mut self) -> Option<VisitNode<Self::Label>> {
        let (id, lvl, nth) = self.stack.pop()?;

        // Push children in reverse so pop() visits them left-to-right,
        // but keep nth_child as the ORIGINAL index.
        for (k, &c) in self.t[id].children().iter().enumerate().rev() {
            self.stack.push((c, lvl + 1, k));
        }

        let bp_id = self.bp_next;
        self.bp_next += 1;

        // record mapping newick->bp
        let nid = id as usize;
        if self.nwk2bp.len() <= nid {
            self.nwk2bp.resize(nid + 1, usize::MAX);
        }
        self.nwk2bp[nid] = bp_id;

        // store branch length by bp-id
        if self.lens_bp.len() <= bp_id {
            self.lens_bp.resize(bp_id + 1, 0.0);
        }
        self.lens_bp[bp_id] = self.t[id].branch().copied().unwrap_or(0.0);

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

// for recording per-worker accumulators
struct WorkerAccum {
    acc: Vec<f32>,
    touched: Vec<usize>,
}

/// Simple Newick-based parent + branch-length extractor.
///
/// - `parent[v]` = parent node id of v, or usize::MAX if v is root.
/// - `lens[v]`   = branch length from parent[v] to v (0.0 for root if absent).
fn build_parent_and_lens_simple(t: &NwkTree) -> (Vec<usize>, Vec<f64>) {
    // First pass: find max node id so we can size the vectors
    let mut max_id = 0usize;
    for v in t.nodes() {
        if v > max_id {
            max_id = v;
        }
    }
    let total = max_id + 1;

    let mut parent = vec![usize::MAX; total];
    let mut lens = vec![0.0f64; total];

    // Second pass: fill parent + branch length
    for v in t.nodes() {
        lens[v] = t[v].branch().copied().unwrap_or(0.0) as f64;
        for &c in t[v].children() {
            parent[c] = v;
        }
    }

    (parent, lens)
}

// Unweighted, build sketches using simple Newick tree parsing (no succparen)
fn build_sketches_simple(
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

    // Parent pointers + branch lengths directly from Newick
    let (parent, lens) = build_parent_and_lens_simple(&t);
    let total = parent.len();

    // Leaves & mapping name→leaf
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

    // Build per-sample presence sets (emit (edge_id, ℓ_v) when present)
    let (samples, wsets_by_vid): (Vec<String>, Vec<Vec<(u64, f64)>>) = if let Some(tsv) = input_tsv
    {
        // TSV dense (presence/absence)
        let (taxa, samples, mat) = read_table(tsv)?;
        let nsamp = samples.len();
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids;

        info!("(simple) building per-sample presence sets from TSV …");
        let t0 = Instant::now();

        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets.par_iter_mut().enumerate().for_each(|(s, out)| {
            // per-sample scratch: 0/1 presence over nodes + list of touched nodes
            let mut seen = vec![0u8; total_usize];
            let mut touched: Vec<usize> = Vec::new();

            // scan taxa rows for this sample
            for (r, lopt) in row2leaf.iter().enumerate() {
                if mat[r][s] <= 0.0 {
                    continue;
                }
                let lp = match lopt {
                    Some(v) => *v,
                    None => continue,
                };
                let mut v = leaf_ids_ref[lp];

                // walk to root until we hit an already-seen node
                loop {
                    if seen[v] != 0 {
                        break;
                    }
                    seen[v] = 1;
                    touched.push(v);
                    let p = parent[v];
                    if p == usize::MAX {
                        break;
                    }
                    v = p;
                }
            }

            out.reserve(touched.len());
            for &v in &touched {
                let lw = lens_ref[v];
                if lw > 0.0 {
                    out.push((v as u64, lw));
                }
            }
        });

        info!("(simple) built presence sets in {} ms", t0.elapsed().as_millis());
        (samples, wsets)
    } else {
        // BIOM (CSR) → CSC for fast per-sample traversal
        let biom = biom_h5.expect("biom path required when TSV not provided");
        let (taxa, samples, indptr, indices) = read_biom_csr(biom)?;
        let nsamp = samples.len();
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();

        // synthesize a data[] of ones (presence) and transpose
        info!("(simple) transposing BIOM CSR→CSC …");
        let ones: Vec<f64> = vec![1.0; indices.len()];
        let (colptr, rowind, _vals) = csr_to_csc(&indptr, &indices, &ones, nsamp);

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids;

        info!("(simple) building per-sample presence sets from BIOM (CSC) …");
        let t0 = Instant::now();

        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets.par_iter_mut().enumerate().for_each(|(s, out)| {
            let mut seen = vec![0u8; total_usize];
            let mut touched: Vec<usize> = Vec::new();

            // iterate only the nonzeros in column s
            for kk in colptr[s]..colptr[s + 1] {
                let r = rowind[kk];
                let lp = match row2leaf[r] {
                    Some(v) => v,
                    None => continue,
                };
                let mut v = leaf_ids_ref[lp];

                loop {
                    if seen[v] != 0 {
                        break;
                    }
                    seen[v] = 1;
                    touched.push(v);
                    let p = parent[v];
                    if p == usize::MAX {
                        break;
                    }
                    v = p;
                }
            }

            out.reserve(touched.len());
            for &v in &touched {
                let lw = lens_ref[v];
                if lw > 0.0 {
                    out.push((v as u64, lw));
                }
            }
        });

        info!(
            "(simple) built presence sets from BIOM in {} ms",
            t0.elapsed().as_millis()
        );
        (samples, wsets)
    };

    let nsamp = samples.len();
    if nsamp < 2 {
        anyhow::bail!("Fewer than 2 samples; nothing to compare.");
    }

    // Drop empty samples
    let mut kept_ws = Vec::with_capacity(nsamp);
    let mut kept_names = Vec::with_capacity(nsamp);
    for (i, ws) in wsets_by_vid.into_iter().enumerate() {
        if !ws.is_empty() {
            kept_ws.push(ws);
            kept_names.push(samples[i].clone());
        }
    }
    let mut wsets = kept_ws;
    let samples = kept_names;
    if samples.len() < 2 {
        anyhow::bail!("Fewer than 2 non-empty samples after filtering; nothing to compare.");
    }

    // Compact to active edge id space
    let mut used = vec![false; total];
    for ws in &wsets {
        for &(vid, _) in ws {
            used[vid as usize] = true;
        }
    }
    let active_edges: Vec<usize> = (0..total).filter(|&v| used[v] && lens[v] > 0.0).collect();
    if active_edges.is_empty() {
        anyhow::bail!("No active edges after presence accumulation.");
    }

    let mut id_map = vec![usize::MAX; total];
    for (new_id, &v) in active_edges.iter().enumerate() {
        id_map[v] = new_id;
    }
    for ws in &mut wsets {
        for (id, _) in ws.iter_mut() {
            *id = id_map[*id as usize] as u64;
        }
    }

    info!(
        "(simple) active edges = {} (from {} total, {} leaves)",
        active_edges.len(),
        total,
        leaf_ids.len()
    );

    // Sketch (DMH or ERS with tight f64 caps)
    info!("(simple) sketching starting...");
    let mut rng = mt_from_seed(seed);
    let sketches_u64: Vec<Vec<u64>> = if method == "dmh" {
        let dmh = DartMinHash::new_mt(&mut rng, k as u64);
        wsets
            .par_iter()
            .map(|ws| dmh.sketch(ws).into_iter().map(|(id, _rank)| id).collect())
            .collect()
    } else {
        // For unweighted presence, per-dim max weight is exactly ℓ_v
        let caps: Vec<f64> = active_edges.iter().map(|&v| lens[v]).collect();
        let ers = ErsWmh::new_mt(&mut rng, &caps, k as u64);
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
    info!("(simple) sketching done.");

    Ok((samples, sketches_u64))
}


// Weighted, build sketches for normalized weighted UniFrac using simple Newick tree parsing
fn build_sketches_weighted_simple(
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

    // Parent pointers + branch lengths directly from Newick
    let (parent, lens) = build_parent_and_lens_simple(&t);
    let total = parent.len();

    // Leaves & mapping name→leaf-ordinal, and leaf node ids
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

    info!(
        "(simple) nodes = {}  leaves = {}",
        total,
        leaf_ids.len()
    );

    // Build per-sample weighted sets
    let (samples, wsets_by_vid): (Vec<String>, Vec<Vec<(u64, f64)>>) = if let Some(tsv) = input_tsv
    {
        // TSV (dense)
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

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids;
        info!("(simple) building per-sample weighted sets from TSV (dense) …");
        let t0 = Instant::now();

        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets
            .par_iter_mut()
            .enumerate()
            .for_each_init(
                || WorkerAccum {
                    acc: vec![0f32; total_usize],
                    touched: Vec::new(),
                },
                |state, (s, out)| {
                    let denom = col_sums[s];
                    if denom == 0.0 {
                        return;
                    }

                    // reset only touched entries
                    for &v in &state.touched {
                        state.acc[v] = 0.0;
                    }
                    state.touched.clear();
                    out.clear();

                    let acc = &mut state.acc;
                    let touched = &mut state.touched;

                    for (r, lopt) in row2leaf.iter().enumerate() {
                        let lp = match lopt {
                            Some(v) => *v,
                            None => continue,
                        };
                        let val = counts[r][s];
                        if val <= 0.0 {
                            continue;
                        }
                        let inc = (val / denom) as f32;
                        if inc == 0.0 {
                            continue;
                        }

                        let mut v = leaf_ids_ref[lp];
                        loop {
                            if acc[v] == 0.0 {
                                touched.push(v);
                            }
                            acc[v] += inc;
                            let p = parent[v];
                            if p == usize::MAX {
                                break;
                            }
                            v = p;
                        }
                    }

                    out.reserve(touched.len());
                    for &v in touched.iter() {
                        let a = acc[v] as f64;
                        if a > 0.0 {
                            let lw = lens_ref[v];
                            if lw > 0.0 {
                                out.push((v as u64, lw * a));
                            }
                        }
                    }
                },
            );

        info!(
            "(simple) built weighted sets from TSV in {} ms",
            t0.elapsed().as_millis()
        );
        (samples, wsets)
    } else {
        // BIOM (CSR) to CSC then per-sample scatter
        let biom = biom_h5.expect("biom path required when TSV not provided");
        let (taxa, samples, indptr, indices, data) = read_biom_csr_values(biom)?;
        let nsamp = samples.len();

        // column sums
        let mut col_sums = vec![0.0f64; nsamp];
        for r in 0..taxa.len() {
            let a = indptr[r] as usize;
            let b = indptr[r + 1] as usize;
            for k in a..b {
                col_sums[indices[k] as usize] += data[k];
            }
        }
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();

        // transpose to CSC for fast per-sample scans
        info!("(simple) transposing BIOM CSR→CSC …");
        let (colptr, rowind, vals) = csr_to_csc(&indptr, &indices, &data, nsamp);

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids;

        info!("(simple) building per-sample weighted sets from BIOM (CSC) …");
        let t0 = Instant::now();

        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets
            .par_iter_mut()
            .enumerate()
            .for_each_init(
                || WorkerAccum {
                    acc: vec![0f32; total_usize],
                    touched: Vec::new(),
                },
                |state, (s, out)| {
                    let denom = col_sums[s];
                    if denom == 0.0 {
                        // nothing for this sample
                        return;
                    }

                    // Reset only previously-touched entries in this worker's acc buffer.
                    for &v in &state.touched {
                        state.acc[v] = 0.0;
                    }
                    state.touched.clear();
                    out.clear();

                    let acc = &mut state.acc;
                    let touched = &mut state.touched;

                    // iterate only nnz in column s
                    for kk in colptr[s]..colptr[s + 1] {
                        let r = rowind[kk];
                        let lp = match row2leaf[r] {
                            Some(v) => v,
                            None => continue,
                        };
                        let mut v = leaf_ids_ref[lp];

                        let inc = (vals[kk] / denom) as f32;
                        if inc == 0.0 {
                            continue;
                        }

                        loop {
                            if acc[v] == 0.0 {
                                touched.push(v);
                            }
                            acc[v] += inc;
                            let p = parent[v];
                            if p == usize::MAX {
                                break;
                            }
                            v = p;
                        }
                    }

                    out.reserve(touched.len());
                    for &v in touched.iter() {
                        let a = acc[v] as f64;
                        if a > 0.0 {
                            let lw = lens_ref[v];
                            if lw > 0.0 {
                                out.push((v as u64, lw * a));
                            }
                        }
                    }
                },
            );

        info!(
            "(simple) built weighted sets from BIOM in {} ms",
            t0.elapsed().as_millis()
        );
        (samples, wsets)
    };

    let nsamp = samples.len();
    if nsamp < 2 {
        anyhow::bail!("Fewer than 2 samples; nothing to compare.");
    }

    // Drop empty samples
    let mut kept_ws = Vec::with_capacity(nsamp);
    let mut kept_names = Vec::with_capacity(nsamp);
    for (i, ws) in wsets_by_vid.into_iter().enumerate() {
        if !ws.is_empty() {
            kept_ws.push(ws);
            kept_names.push(samples[i].clone());
        }
    }
    let mut wsets = kept_ws;
    let samples = kept_names;
    if samples.len() < 2 {
        anyhow::bail!("Fewer than 2 non-empty samples; nothing to compare.");
    }

    // Compact to active edge id space
    let mut used = vec![false; total];
    for ws in &wsets {
        for &(vid, _) in ws {
            used[vid as usize] = true;
        }
    }
    let active_edges: Vec<usize> = (0..total).filter(|&v| used[v] && lens[v] > 0.0).collect();

    if active_edges.is_empty() {
        anyhow::bail!("No active edges for weighted case.");
    }

    let mut id_map = vec![usize::MAX; total];
    for (new_id, &v) in active_edges.iter().enumerate() {
        id_map[v] = new_id;
    }
    for ws in &mut wsets {
        for (id, _) in ws.iter_mut() {
            *id = id_map[*id as usize] as u64;
        }
    }

    info!(
        "(simple) active edges = {} (from {} total, {} leaves)",
        active_edges.len(),
        total,
        leaf_ids.len()
    );

    // Sketch (DMH or ERS)
    info!("(simple) sketching starting...");
    let mut rng = mt_from_seed(seed);
    let sketches_u64: Vec<Vec<u64>> = if method == "dmh" {
        let dmh = DartMinHash::new_mt(&mut rng, k as u64);
        wsets
            .par_iter()
            .map(|ws| dmh.sketch(ws).into_iter().map(|(id, _rank)| id).collect())
            .collect()
    } else {
        // tight caps: m_i = max_s (ℓ_v * A_v[s]) over samples for that edge
        let t_caps = Instant::now();
        let d = active_edges.len();

        let caps: Vec<f64> = wsets
            .par_iter()
            .fold(
                || vec![0.0f64; d],
                |mut local, ws| {
                    for &(id, w) in ws {
                        let idx = id as usize;
                        if w > local[idx] {
                            local[idx] = w;
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0f64; d],
                |mut a, b| {
                    for i in 0..d {
                        if b[i] > a[i] {
                            a[i] = b[i];
                        }
                    }
                    a
                },
            );

        info!(
            "(simple) ERS: caps(max_w) built in {} ms",
            t_caps.elapsed().as_millis()
        );

        let ers = ErsWmh::new_mt(&mut rng, &caps, k as u64);

        let t_ers = Instant::now();
        let sketches: Vec<Vec<u64>> = wsets
            .par_iter()
            .map(|ws| {
                ers.sketch(ws, Some(ers_l))
                    .into_iter()
                    .map(|(id, _rank)| id)
                    .collect()
            })
            .collect();

        info!(
            "(simple) ERS: sketching in {} ms",
            t_ers.elapsed().as_millis()
        );

        sketches
    };
    info!("(simple) sketching done.");

    Ok((samples, sketches_u64))
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
    let mut block = vec![0.0f32; n * bs];
    let dh = DistHamming;

    let mut i0 = 0usize;
    while i0 < n {
        let h = (n - i0).min(bs);

        // Fill block in parallel over columns; each `col` is a disjoint &mut [f64] (length = bs)
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
                // Pre-size roughly: tab + ~12 chars per number
                let mut line = String::with_capacity(8 + n * 12);
                line.push_str(&names[i]);

                // Each worker needs its own Ryu buffer
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

// Unweighted, build sketches without node_bits (leaf→root OR per sample)
fn build_sketches(
    tree_file: &str,
    input_tsv: Option<&str>,
    biom_h5: Option<&str>,
    k: usize,
    method: &str,
    ers_l: u64,
    seed: u64,
) -> Result<(Vec<String>, Vec<Vec<u64>>)> {
    // Load tree & balanced parens
    let raw = std::fs::read_to_string(tree_file).context("read newick")?;
    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
    let t: NwkTree = one_from_string(&sanitized).context("parse newick (sanitized)")?;
    let mut lens_f32 = Vec::<f32>::new();
    let mut nwk2bp = Vec::<usize>::new();

    let trav = SuccTrav::new(&t, &mut lens_f32, &mut nwk2bp);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    // Leaves & mapping name→leaf
    let mut leaf_ids_bp = Vec::<usize>::new();
    let mut leaf_nm = Vec::<String>::new();

    for n in t.nodes() {
        if t[n].is_leaf() {
            let nid = n as usize;
            let bid = nwk2bp[nid];
            debug_assert!(bid != usize::MAX, "leaf newick id {nid} missing bp mapping");

            leaf_ids_bp.push(bid);
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

    // children, post (unused here but harmless), parents, lengths
    let total = bp.len() + 1;
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    lens_f32.resize(total, 0.0);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);
    let parent = compute_parent(total, &kids);
    let lens: Vec<f64> = lens_f32.iter().map(|&x| x as f64).collect();

    // Build per-sample presence sets (emit (edge_id, ℓ_v) when present)
    let (samples, wsets_by_vid): (Vec<String>, Vec<Vec<(u64, f64)>>) = if let Some(tsv) = input_tsv
    {
        // TSV dense (presence/absence)
        let (taxa, samples, mat) = read_table(tsv)?;
        let nsamp = samples.len();
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids_bp;

        info!("building per-sample presence sets from TSV …");
        let t0 = Instant::now();

        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets.par_iter_mut().enumerate().for_each(|(s, out)| {
            // per-sample scratch: 0/1 presence over nodes + list of touched nodes
            let mut seen = vec![0u8; total_usize];
            let mut touched: Vec<usize> = Vec::new();

            // scan taxa rows for this sample
            for (r, lopt) in row2leaf.iter().enumerate() {
                if mat[r][s] <= 0.0 {
                    continue;
                }
                let lp = match lopt {
                    Some(v) => *v,
                    None => continue,
                };
                let mut v = leaf_ids_ref[lp];

                // walk to root until we hit an already-seen node
                loop {
                    if seen[v] != 0 {
                        break;
                    }
                    seen[v] = 1;
                    touched.push(v);
                    let p = parent[v];
                    if p == usize::MAX {
                        break;
                    }
                    v = p;
                }
            }

            out.reserve(touched.len());
            for &v in &touched {
                let lw = lens_ref[v];
                if lw > 0.0 {
                    out.push((v as u64, lw));
                } // unweighted: weight = branch length
            }
        });

        info!("built presence sets in {} ms", t0.elapsed().as_millis());
        (samples, wsets)
    } else {
        // BIOM (CSR) → CSC for fast per-sample traversal
        let biom = biom_h5.expect("biom path required when TSV not provided");
        let (taxa, samples, indptr, indices) = read_biom_csr(biom)?;
        let nsamp = samples.len();
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();

        // synthesize a data[] of ones (presence) and transpose
        info!("transposing BIOM CSR→CSC …");
        let ones: Vec<f64> = vec![1.0; indices.len()];
        let (colptr, rowind, _vals) = csr_to_csc(&indptr, &indices, &ones, nsamp);

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids_bp;

        info!("building per-sample presence sets from BIOM (CSC) …");
        let t0 = Instant::now();

        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets.par_iter_mut().enumerate().for_each(|(s, out)| {
            let mut seen = vec![0u8; total_usize];
            let mut touched: Vec<usize> = Vec::new();

            // iterate only the nonzeros in column s
            for kk in colptr[s]..colptr[s + 1] {
                let r = rowind[kk];
                let lp = match row2leaf[r] {
                    Some(v) => v,
                    None => continue,
                };
                let mut v = leaf_ids_ref[lp];

                loop {
                    if seen[v] != 0 {
                        break;
                    }
                    seen[v] = 1;
                    touched.push(v);
                    let p = parent[v];
                    if p == usize::MAX {
                        break;
                    }
                    v = p;
                }
            }

            out.reserve(touched.len());
            for &v in &touched {
                let lw = lens_ref[v];
                if lw > 0.0 {
                    out.push((v as u64, lw));
                }
            }
        });

        info!("built presence sets in {} ms", t0.elapsed().as_millis());
        (samples, wsets)
    };

    let nsamp = samples.len();
    if nsamp < 2 {
        anyhow::bail!("Fewer than 2 samples; nothing to compare.");
    }

    // Drop empty samples
    let mut kept_ws = Vec::with_capacity(nsamp);
    let mut kept_names = Vec::with_capacity(nsamp);
    for (i, ws) in wsets_by_vid.into_iter().enumerate() {
        if !ws.is_empty() {
            kept_ws.push(ws);
            kept_names.push(samples[i].clone());
        }
    }
    let mut wsets = kept_ws;
    let samples = kept_names;
    if samples.len() < 2 {
        anyhow::bail!("Fewer than 2 non-empty samples after filtering; nothing to compare.");
    }

    // Compact to active edge id space
    let mut used = vec![false; total];
    for ws in &wsets {
        for &(vid, _) in ws {
            used[vid as usize] = true;
        }
    }
    let active_edges: Vec<usize> = (0..total).filter(|&v| used[v] && lens[v] > 0.0).collect();
    if active_edges.is_empty() {
        anyhow::bail!("No active edges after presence accumulation.");
    }

    let mut id_map = vec![usize::MAX; total];
    for (new_id, &v) in active_edges.iter().enumerate() {
        id_map[v] = new_id;
    }
    for ws in &mut wsets {
        for (id, _) in ws.iter_mut() {
            *id = id_map[*id as usize] as u64;
        }
    }

    info!(
        "active edges = {} (from {} total, {} leaves)",
        active_edges.len(),
        total,
        leaf_ids_bp.len()
    );

    // Sketch (DMH or ERS with tight f64 caps)
    info!("sketching starting...");
    let mut rng = mt_from_seed(seed);
    let sketches_u64: Vec<Vec<u64>> = if method == "dmh" {
        let dmh = DartMinHash::new_mt(&mut rng, k as u64);
        wsets
            .par_iter()
            .map(|ws| dmh.sketch(ws).into_iter().map(|(id, _rank)| id).collect())
            .collect()
    } else {
        // For unweighted presence, per-dim max weight is exactly ℓ_v
        let caps: Vec<f64> = active_edges.iter().map(|&v| lens[v]).collect();
        let ers = ErsWmh::new_mt(&mut rng, &caps, k as u64);
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

/// CSR (rows=features, cols=samples)to CSC (cols=samples) for fast per-sample scans.
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
    // Load tree & balanced parens
    let raw = std::fs::read_to_string(tree_file).context("read newick")?;
    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);
    let t: NwkTree = one_from_string(&sanitized).context("parse newick (sanitized)")?;
    let mut lens_f32 = Vec::<f32>::new();
    let mut nwk2bp = Vec::<usize>::new();

    let trav = SuccTrav::new(&t, &mut lens_f32, &mut nwk2bp);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    // Leaves & mapping name→leaf-ordinal, and leaf node ids
    let mut leaf_ids_bp = Vec::<usize>::new();
    let mut leaf_nm = Vec::<String>::new();

    for n in t.nodes() {
        if t[n].is_leaf() {
            let nid = n as usize;
            let bid = nwk2bp[nid];
            debug_assert!(bid != usize::MAX, "leaf newick id {nid} missing bp mapping");

            leaf_ids_bp.push(bid);
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

    // parent pointers for leaf→root accumulation
    let parent = compute_parent(total, &kids);
    let lens: Vec<f64> = lens_f32.iter().map(|&x| x as f64).collect();
    info!("nodes = {}  leaves = {}", total, leaf_ids_bp.len());

    // Build per-sample weighted sets
    let (samples, wsets_by_vid): (Vec<String>, Vec<Vec<(u64, f64)>>) = if let Some(tsv) = input_tsv
    {
        // TSV (dense)
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

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids_bp;
        info!("building per-sample weighted sets from TSV (dense) …");
        let t0 = Instant::now();

        // One sparse weighted set per sample
        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets
            .par_iter_mut()
            .enumerate()
            .for_each_init(
                || WorkerAccum {
                    acc: vec![0f32; total_usize],
                    touched: Vec::new(),
                },
                |state, (s, out)| {
                    let denom = col_sums[s];
                    if denom == 0.0 {
                        // nothing for this sample
                        return;
                    }

                    // Reset only previously-touched entries in this worker's acc buffer.
                    for &v in &state.touched {
                        state.acc[v] = 0.0;
                    }
                    state.touched.clear();
                    out.clear();

                    let acc = &mut state.acc;
                    let touched = &mut state.touched;

                    // iterate only nnz in column s
                    for kk in colptr[s]..colptr[s + 1] {
                        let r = rowind[kk];
                        let lp = match row2leaf[r] {
                            Some(v) => v,
                            None => continue,
                        };
                        let mut v = leaf_ids_ref[lp];

                        let inc = (vals[kk] / denom) as f32;
                        if inc == 0.0 {
                            continue;
                        }

                        loop {
                            if acc[v] == 0.0 {
                                touched.push(v);
                            }
                            acc[v] += inc;
                            let p = parent[v];
                            if p == usize::MAX {
                                break;
                            }
                            v = p;
                        }
                    }

                    out.reserve(touched.len());
                    for &v in touched.iter() {
                        let a = acc[v] as f64;
                        if a > 0.0 {
                            let lw = lens_ref[v];
                            if lw > 0.0 {
                                out.push((v as u64, lw * a));
                            }
                        }
                    }
                },
            );

        info!(
            "(simple) built weighted sets from BIOM in {} ms",
            t0.elapsed().as_millis()
        );
        (samples, wsets)
    } else {
        // BIOM (CSR) to CSC then per-sample scatter
        let biom = biom_h5.expect("biom path required when TSV not provided");
        let (taxa, samples, indptr, indices, data) = read_biom_csr_values(biom)?;
        let nsamp = samples.len();

        // column sums
        let mut col_sums = vec![0.0f64; nsamp];
        for r in 0..taxa.len() {
            let a = indptr[r] as usize;
            let b = indptr[r + 1] as usize;
            for k in a..b {
                col_sums[indices[k] as usize] += data[k];
            }
        }
        // row to leaf-ordinal
        let row2leaf: Vec<Option<usize>> = taxa
            .iter()
            .map(|n| t2leaf.get(n.as_str()).copied())
            .collect();

        // transpose to CSC for fast per-sample scans
        info!("transposing BIOM CSR→CSC …");
        let (colptr, rowind, vals) = csr_to_csc(&indptr, &indices, &data, nsamp);

        let total_usize = total;
        let lens_ref = &lens;
        let leaf_ids_ref = &leaf_ids_bp;

        info!("building per-sample weighted sets from BIOM (CSC) …");
        let t0 = Instant::now();

        let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
        wsets
            .par_iter_mut()
            .enumerate()
            .for_each_init(
                || WorkerAccum {
                    acc: vec![0f32; total_usize],
                    touched: Vec::new(),
                },
                |state, (s, out)| {
                    let denom = col_sums[s];
                    if denom == 0.0 {
                        // nothing for this sample
                        return;
                    }

                    // Reset only previously-touched entries in this worker's acc buffer.
                    for &v in &state.touched {
                        state.acc[v] = 0.0;
                    }
                    state.touched.clear();
                    out.clear();

                    let acc = &mut state.acc;
                    let touched = &mut state.touched;

                    // iterate only nnz in column s
                    for kk in colptr[s]..colptr[s + 1] {
                        let r = rowind[kk];
                        let lp = match row2leaf[r] {
                            Some(v) => v,
                            None => continue,
                        };
                        let mut v = leaf_ids_ref[lp];

                        let inc = (vals[kk] / denom) as f32;
                        if inc == 0.0 {
                            continue;
                        }

                        loop {
                            if acc[v] == 0.0 {
                                touched.push(v);
                            }
                            acc[v] += inc;
                            let p = parent[v];
                            if p == usize::MAX {
                                break;
                            }
                            v = p;
                        }
                    }

                    out.reserve(touched.len());
                    for &v in touched.iter() {
                        let a = acc[v] as f64;
                        if a > 0.0 {
                            let lw = lens_ref[v];
                            if lw > 0.0 {
                                out.push((v as u64, lw * a));
                            }
                        }
                    }
                },
            );

        info!(
            "(simple) built weighted sets from BIOM in {} ms",
            t0.elapsed().as_millis()
        );
        (samples, wsets)
    };

    let nsamp = samples.len();
    if nsamp < 2 {
        anyhow::bail!("Fewer than 2 samples; nothing to compare.");
    }

    // Drop empty samples
    let mut kept_ws = Vec::with_capacity(nsamp);
    let mut kept_names = Vec::with_capacity(nsamp);
    for (i, ws) in wsets_by_vid.into_iter().enumerate() {
        if !ws.is_empty() {
            kept_ws.push(ws);
            kept_names.push(samples[i].clone());
        }
    }
    let mut wsets = kept_ws;
    let samples = kept_names;
    if samples.len() < 2 {
        anyhow::bail!("Fewer than 2 non-empty samples; nothing to compare.");
    }

    // Compact to active edge id space
    let mut used = vec![false; total];
    for ws in &wsets {
        for &(vid, _) in ws {
            used[vid as usize] = true;
        }
    }
    let active_edges: Vec<usize> = (0..total).filter(|&v| used[v] && lens[v] > 0.0).collect();

    if active_edges.is_empty() {
        anyhow::bail!("No active edges for weighted case.");
    }

    let mut id_map = vec![usize::MAX; total];
    for (new_id, &v) in active_edges.iter().enumerate() {
        id_map[v] = new_id;
    }
    for ws in &mut wsets {
        for (id, _) in ws.iter_mut() {
            *id = id_map[*id as usize] as u64;
        }
    }

    info!(
        "active edges = {} (from {} total, {} leaves)",
        active_edges.len(),
        total,
        leaf_ids_bp.len()
    );

    // Sketch (DMH or ERS)
    info!("sketching starting...");
    let mut rng = mt_from_seed(seed);
    let sketches_u64: Vec<Vec<u64>> = if method == "dmh" {
        let dmh = DartMinHash::new_mt(&mut rng, k as u64);
        wsets
            .par_iter()
            .map(|ws| dmh.sketch(ws).into_iter().map(|(id, _rank)| id).collect())
            .collect()
    } else {
        // tight caps: m_i = max_s (ℓ_v * A_v[s]) over samples for that edge
        let t_caps = Instant::now();
        let d = active_edges.len();

        // Parallel caps build: thread-local max vectors + reduce by max
        let caps: Vec<f64> = wsets
            .par_iter()
            .fold(
                || vec![0.0f64; d], // thread-local maxima
                |mut local, ws| {
                    for &(id, w) in ws {
                        let idx = id as usize;
                        if w > local[idx] {
                            local[idx] = w;
                        }
                    }
                    local
                },
            )
            .reduce(
                || vec![0.0f64; d], // identity
                |mut a, b| {
                    // elementwise max
                    for i in 0..d {
                        if b[i] > a[i] {
                            a[i] = b[i];
                        }
                    }
                    a
                },
            );

        info!("ERS: caps(max_w) built in {} ms", t_caps.elapsed().as_millis());

        // Build ERS with caps
        let ers = ErsWmh::new_mt(&mut rng, &caps, k as u64);

        // Sketch all samples
        let t_ers = Instant::now();
        let sketches: Vec<Vec<u64>> = wsets
            .par_iter()
            .map(|ws| {
                ers.sketch(ws, Some(ers_l))
                    .into_iter()
                    .map(|(id, _rank)| id)
                    .collect()
            })
            .collect();

        info!(
            "ERS: sketching (ers.sketch over all samples) in {} ms",
            t_ers.elapsed().as_millis()
        );

        sketches
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

    let mut cmd = Command::new("dartunifrac")
        .version("0.2.7")
        .about(format!("DartUniFrac: Approximate UniFrac via Weighted MinHash {dart}{dart}{dart}"))
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
        .group(
            ArgGroup::new("table")
                .args(["input", "biom"])
                .required(true),
        )
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
            Arg::new("succ")
                .long("succ")
                .help("Use succparen balanced-parentheses tree representation")
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
                .help("Per-hash independent random sequence length for ERS, must be >= 512")
                .value_parser(clap::value_parser!(u64))
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
        );
    #[cfg(feature = "cuda")]
    {
        cmd = cmd
            .arg(
                Arg::new("gpu-streaming")
                    .long("gpu-streaming")
                    .help("Streaming the distance matrix to disk block by block (multi-GPU support); available only with the 'cuda' feature")
                    .action(clap::ArgAction::SetTrue),
            )
            .arg(
                Arg::new("tile-cols")
                    .long("tile-cols")
                    .help("Number of columns per GPU tile in gpu-streaming mode")
                    .value_parser(clap::value_parser!(usize))
                    .default_value("32768"),
            )
            .arg(
                Arg::new("tile-rows")
                    .long("tile-rows")
                    .help("Number of rows per GPU tile in gpu-streaming mode")
                    .value_parser(clap::value_parser!(usize))
                    // 1028 rows: safe for 256G RAM, easy to bump up
                    .default_value("1028"),
            );
    }

    let m = cmd.get_matches();

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
    let succ = m.get_flag("succ");

    #[cfg(feature = "cuda")]
    let gpu_streaming = m.get_flag("gpu-streaming");

    #[cfg(feature = "cuda")]
    let tile_cols = *m.get_one::<usize>("tile-cols").unwrap();

    #[cfg(feature = "cuda")]
    let tile_rows = *m.get_one::<usize>("tile-rows").unwrap();


    let threads = m
        .get_one::<usize>("threads")
        .copied()
        .unwrap_or_else(|| num_cpus::get());

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build_global()
        .unwrap();

    info!("{} threads will be used ", rayon::current_num_threads());

    if succ {
        info!("Using succparen balanced-parentheses tree representation (--succ)");
    } else {
        info!("Using simple Newick tree parsing (default)");
    }

    info!("method={method}   k={k}   seed={seed}");
    if method == "ers" {
        info!("ERS L={ers_l}");
    }
    if weighted {
        info!("Weighted mode");
    } else {
        info!("Unweighted mode");
    };

    let (samples, sketches_u64): (Vec<String>, Vec<Vec<u64>>) =
        if weighted {
            if succ {
                build_sketches_weighted(tree_file, input_tsv, biom_path, k, method, ers_l, seed)?
            } else {
                build_sketches_weighted_simple(tree_file, input_tsv, biom_path, k, method, ers_l, seed)?
            }
        } else {
            if succ {
                build_sketches(tree_file, input_tsv, biom_path, k, method, ers_l, seed)?
            } else {
                build_sketches_simple(tree_file, input_tsv, biom_path, k, method, ers_l, seed)?
            }
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
    // CPU streaming block
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

    // GPU streaming block
    #[cfg(feature = "cuda")]
    if gpu_streaming {
        if pcoa {
            warn!("--pcoa is incompatible with --gpu-streaming; skipping PCoA.");
        }
        if compress {
            warn!("--compress is ignored with --gpu-streaming; output is written as zstd already.");
        }

        let ng = disthamming_gpu::device_count().unwrap_or(0);
        if ng == 0 {
            warn!(
                "--gpu-streaming requested but no CUDA device found; falling back to normal path."
            );
        } else {
            // pick/path with .zst suffix if none provided
            let out_path_stream: PathBuf = {
                let p = Path::new(out_file);
                match p.extension().and_then(|e| e.to_str()) {
                    Some("zst") => p.to_path_buf(),
                    _ => PathBuf::from(format!("{out_file}.zst")),
                }
            };
            let out_path_stream_str = out_path_stream.to_string_lossy();

            // flatten sketches to row-major [n*k]
            let n = nsamp;
            let ksk = sketches_u64[0].len();
            let mut flat: Vec<u64> = Vec::with_capacity(n * ksk);
            for row in &sketches_u64 {
                debug_assert_eq!(row.len(), ksk);
                flat.extend_from_slice(row);
            }

            info!(
                "GPU streaming with {} GPU{} → {} (tile_cols={})",
                ng,
                if ng > 1 { "s" } else { "" },
                out_path_stream_str,
                tile_cols
            );

            // This function should *internally* use all GPUs if ng>1, otherwise the single GPU,
            // and stream rows to a zstd writer so host RAM stays small.
            disthamming_gpu::write_matrix_streaming_gpu_auto(
                &samples,             // names
                &flat,                // sketches_flat_u64
                n,                    // n
                ksk,                  // k
                &out_path_stream_str, // path
                true,                 // compress (zstd)
                weighted,             // weighted_normalized
                tile_cols,            // tile width
                tile_rows,            // tile height (rows per block)
            )?;

            info!("Done → {}", out_path_stream_str);
            return Ok(());
        }
    }

    // Pairwise UniFrac (≈ 1 - Jaccard) via normalized Hamming on ID arrays.
    let dist = {
        let n = nsamp;
        #[cfg(feature = "cuda")]
        {
            let k = sketches_u64[0].len();
            match disthamming_gpu::device_count() {
                Ok(ng) if ng >= 1 => {
                    log::info!(
                        "CUDA detected ({} device{}). Computing pairwise distances on GPU{} …",
                        ng,
                        if ng > 1 { "s" } else { "" },
                        if ng > 1 { "s" } else { "" }
                    );
                    let t2 = Instant::now();

                    // Flatten sketches: row-major [n*k]
                    let mut flat: Vec<u64> = Vec::with_capacity(n * k);
                    for row in &sketches_u64 {
                        debug_assert_eq!(row.len(), k);
                        flat.extend_from_slice(row);
                    }

                    let mut out = vec![0.0f32; n * n];
                    // 8192 works well; feel free to tune (4096..16384)
                    disthamming_gpu::pairwise_hamming_multi_gpu(
                        &flat, n, k, &mut out, 8192, weighted,
                    )?;
                    log::info!(
                        "pairwise distances (GPU) in {} ms",
                        t2.elapsed().as_millis()
                    );
                    out
                }
                _ => {
                    log::info!("CUDA not available (or not enabled). Falling back to CPU.");
                    // CPU fallback path
                    let t2 = Instant::now();
                    let dh = DistHamming;
                    let mut out = vec![0.0f32; n * n];
                    out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                        row[i] = 0.0f32;
                        for j in (i + 1)..n {
                            let mut d: f32 = dh.eval(&sketches_u64[i], &sketches_u64[j]) as f32;
                            if weighted {
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
                    log::info!(
                        "pairwise distances (CPU) in {} ms",
                        t2.elapsed().as_millis()
                    );
                    out
                }
            }
        }

        // If compiled without the "cuda" feature, this block is the only one:
        #[cfg(not(feature = "cuda"))]
        {
            let t2 = Instant::now();
            let dh = DistHamming;
            let mut out = vec![0.0f32; n * n];
            out.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                row[i] = 0.0f32;
                for j in (i + 1)..n {
                    let mut d: f32 = dh.eval(&sketches_u64[i], &sketches_u64[j]) as f32;
                    if weighted {
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
            log::info!(
                "pairwise distances (CPU) in {} ms",
                t2.elapsed().as_millis()
            );
            out
        }
    };

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