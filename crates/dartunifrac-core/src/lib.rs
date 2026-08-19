//! BSD 3-Clause License
//!
//! Copyright (c) 2016-2025, UniFrac development team.
//! All rights reserved.
//!
//! See LICENSE file for more details

//! Core engine for DartUniFrac: per-sample weighted set accumulation over a
//! phylogeny, then weighted MinHash sketching of those sets.
//!
//! This crate is deliberately narrow. It does not read files, resolve feature
//! names, compute distances, or run PCoA — it takes a tree as parent pointers
//! plus branch lengths, a feature table already resolved to node ids, and
//! returns one sketch per surviving sample. That keeps it free of HDF5, BLAS
//! and nightly Rust, so it builds on stable and can be compiled to wasm.
//!
//! The `dartunifrac` binary is the first caller; the C API is the second.

use std::time::Instant;

use dartminhash::{DartMinHash, ErsWmh, TreeMinHash, rng_utils::mt_from_seed};
use log::info;
use rayon::prelude::*;

/// A phylogeny reduced to what sketching needs.
///
/// Both vectors are indexed by node id and must be the same length. `lens[v]`
/// is the length of the edge *into* `v`, so the root's entry is never read.
/// Node ids need not be contiguous over real nodes — an id that no edge reaches
/// simply has `parent[v] == NO_PARENT` and `lens[v] == 0.0`, which is what the
/// `newick` crate produces for its unused slot 0.
///
/// **Node numbering is part of the sketch.** Two callers that number the same
/// tree differently produce different — though statistically equivalent —
/// sketches, and those sketches must never be compared. See the caller's
/// compatibility key.
pub struct Tree {
    pub parent: Vec<usize>,
    pub lens: Vec<f64>,
}

/// Sentinel stored in `Tree::parent` for a node with no parent.
pub const NO_PARENT: usize = usize::MAX;

/// Which weighted-MinHash algorithm to sketch with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Method {
    /// DartMinHash.
    Dmh,
    /// TreeMinHash.
    Tmh,
    /// Efficient Rejection Sampling.
    Ers,
}

/// Unlike [`Table`] and [`Tree`], these fields are **not** validated: `k == 0`,
/// an absurd `ers_l`, or a non-finite weight in `Table::value` will produce
/// nonsense or panic rather than a [`CoreError`]. That matches the binary, whose
/// clap parsers set no bounds either. Worth revisiting when the C API starts
/// accepting parameters from across the FFI boundary.
pub struct SketchParams {
    /// Sketch size; every returned sketch has exactly this length.
    pub k: usize,
    pub method: Method,
    /// Rejection budget for [`Method::Ers`]; ignored otherwise.
    pub ers_l: u64,
    pub seed: u64,
    /// Weighted normalized UniFrac rather than unweighted.
    pub weighted: bool,
    /// Weighted only: use raw counts instead of per-sample relative abundance.
    pub raw_counts: bool,
    /// Derive the edge id space from the tree rather than from this run's
    /// samples, making sketches comparable across calls.
    pub portable: bool,
}

/// A feature table in CSC, already resolved to tree node ids.
///
/// Sample `s` owns entries `colptr[s]..colptr[s + 1]` of `node` and `value`.
/// Entries that do not resolve to a tip are the caller's to drop before
/// building this, and so are any format-specific filters (the TSV reader drops
/// non-positive values; the BIOM reader does not).
///
/// **Entry order within a sample is significant.** Weighted accumulation sums
/// in f32, so reordering a sample's entries changes the low bits of the result.
/// Callers must preserve their source order — both of the binary's readers
/// naturally emit ascending feature rows.
pub struct Table {
    pub n_samples: usize,
    pub colptr: Vec<usize>,
    pub node: Vec<usize>,
    pub value: Vec<f64>,
    /// Per-sample denominators for weighted mode.
    ///
    /// These must be summed over the caller's **entire** table, including
    /// features that resolve to no tip — that is what the binary does, and on a
    /// sheared tree it is most of the table. This crate cannot recompute them
    /// from the fields above, which hold resolved entries only.
    pub col_sums: Vec<f64>,
}

/// One sketch per surviving sample.
#[derive(Debug, PartialEq, Eq)]
pub struct SketchSet {
    /// Indices into the input `Table`'s samples, ascending. Samples whose
    /// weighted set came out empty are absent, so this is how a caller
    /// re-associates sketches with its own sample identifiers.
    pub kept: Vec<usize>,
    /// `kept.len()` sketches, each of length `params.k`. Values are full 64-bit
    /// hashes; truncating them to fewer bits is the caller's concern.
    pub sketches: Vec<Vec<u64>>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CoreError {
    /// Fewer than two samples had a non-empty weighted set, so there is nothing
    /// to compare. Callers differ on whether this is fatal: the CLI errors, the
    /// C API returns an empty result.
    FewerThanTwoNonEmptySamples,
    /// No edge with positive length was in scope, so nothing can be sketched.
    NoActiveEdges,
    /// `Table` fields disagree — wrong `colptr` length, `node`/`value` length
    /// mismatch, `col_sums` length mismatch, or a node id outside the tree.
    MalformedTable(String),
}

/// Choose the edge id space that the weighted-MinHash hashes over.
///
/// By default the space is compacted over the edges touched by the samples in
/// *this* run (`used`), which is compact but run-dependent: the same sample
/// sketched alongside a different set of samples lands in a different id space
/// and therefore produces a different sketch. Sketches are then only comparable
/// against others produced by the same invocation.
///
/// With `portable`, the space is derived from the tree alone -- every edge with
/// a positive branch length, whether or not any sample touches it. A sample's
/// sketch then depends only on that sample, the tree, and the sketching
/// parameters, so sketches built by separate runs are directly comparable and
/// can be merged or stored for later reuse.
///
/// Not public: it indexes `lens`/`used` without bounds checks, relying on
/// [`build_sketches`] having validated their lengths first. Exposing a
/// panic-on-short-slice function from a crate destined for an FFI boundary would
/// be an invitation to misuse.
pub(crate) fn active_edge_ids(
    total: usize,
    lens: &[f64],
    used: &[bool],
    portable: bool,
) -> Vec<usize> {
    (0..total)
        .filter(|&v| lens[v] > 0.0 && (portable || used[v]))
        .collect()
}

/// Per-rayon-worker scratch, reused across every sample that worker handles.
///
/// `acc` is tree-sized, so allocating one per sample would dominate the run at
/// any realistic tree size. It is reset by replaying `touched` rather than by
/// zeroing, making reset cost proportional to the previous sample's footprint
/// instead of to the tree.
struct WorkerAccum {
    /// Unweighted: a 0.0/1.0 `seen` flag. Weighted: a running abundance sum.
    acc: Vec<f32>,
    /// Node ids written since the last reset.
    touched: Vec<usize>,
}

/// Walk each present feature from its tip to the root, accumulating branch
/// weights, and emit one sparse weighted set per sample.
///
/// Weights are `ℓ_v` when unweighted and `ℓ_v × accumulated_mass` when weighted.
/// Ids are tree node ids at this point; stage 5 renumbers them.
fn accumulate_weighted_sets(
    tree: &Tree,
    table: &Table,
    params: &SketchParams,
) -> Vec<Vec<(u64, f64)>> {
    let total = tree.parent.len();
    let parent = &tree.parent;
    let lens = &tree.lens;
    let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); table.n_samples];

    wsets.par_iter_mut().enumerate().for_each_init(
        || WorkerAccum {
            acc: vec![0f32; total],
            touched: Vec::new(),
        },
        |state, (s, out)| {
            let denom = table.col_sums[s];
            if params.weighted && !params.raw_counts && denom == 0.0 {
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
            let span = table.colptr[s]..table.colptr[s + 1];

            if params.weighted {
                for kk in span {
                    // The cast to f32 happens before accumulation, exactly as
                    // the pre-extraction engine did it. Summing in f64 would be
                    // more accurate and would change every published distance.
                    let inc = if params.raw_counts {
                        table.value[kk]
                    } else {
                        table.value[kk] / denom
                    } as f32;
                    if inc == 0.0 {
                        continue;
                    }

                    let mut v = table.node[kk];
                    loop {
                        if acc[v] == 0.0 {
                            touched.push(v);
                        }
                        acc[v] += inc;

                        let p = parent[v];
                        if p == NO_PARENT {
                            break;
                        }
                        v = p;
                    }
                }

                out.reserve(touched.len());
                for &v in touched.iter() {
                    let a = acc[v] as f64;
                    if a > 0.0 {
                        let lw = lens[v];
                        if lw > 0.0 {
                            out.push((v as u64, lw * a));
                        }
                    }
                }
            } else {
                // Presence only: stop climbing as soon as the path is already
                // marked, so each node is visited at most once per sample.
                for kk in span {
                    let mut v = table.node[kk];
                    loop {
                        if acc[v] != 0.0 {
                            break;
                        }
                        acc[v] = 1.0;
                        touched.push(v);

                        let p = parent[v];
                        if p == NO_PARENT {
                            break;
                        }
                        v = p;
                    }
                }

                out.reserve(touched.len());
                for &v in touched.iter() {
                    let lw = lens[v];
                    if lw > 0.0 {
                        out.push((v as u64, lw));
                    }
                }
            }
        },
    );

    wsets
}

/// Renumber node ids densely over the active edge id space, in place.
///
/// The returned vector maps a compacted id back to its node id; its length is
/// the size of the space the sketcher hashes over.
fn compact_ids(
    tree: &Tree,
    wsets: &mut [Vec<(u64, f64)>],
    portable: bool,
) -> Result<Vec<usize>, CoreError> {
    let total = tree.parent.len();
    let mut used = vec![false; total];
    for ws in wsets.iter() {
        for &(vid, _) in ws {
            used[vid as usize] = true;
        }
    }
    let active_edges = active_edge_ids(total, &tree.lens, &used, portable);
    if active_edges.is_empty() {
        return Err(CoreError::NoActiveEdges);
    }

    let mut id_map = vec![usize::MAX; total];
    for (new_id, &v) in active_edges.iter().enumerate() {
        id_map[v] = new_id;
    }
    for ws in wsets.iter_mut() {
        for (id, _) in ws.iter_mut() {
            *id = id_map[*id as usize] as u64;
        }
    }
    Ok(active_edges)
}

fn sketch_all(
    wsets: &[Vec<(u64, f64)>],
    active_edges: &[usize],
    lens: &[f64],
    params: &SketchParams,
    log_label: &str,
) -> Vec<Vec<u64>> {
    let mut rng = mt_from_seed(params.seed);
    match params.method {
        Method::Dmh => {
            let dmh = DartMinHash::new_mt(&mut rng, params.k as u64);
            wsets
                .par_iter()
                .map(|ws| dmh.sketch(ws).into_iter().map(|(id, _rank)| id).collect())
                .collect()
        }
        Method::Tmh => {
            let tmh = TreeMinHash::new_mt(&mut rng, params.k as u64);
            wsets
                .par_iter()
                .map(|ws| tmh.sketch(ws).into_iter().map(|(id, _rank)| id).collect())
                .collect()
        }
        Method::Ers => {
            // Unweighted caps come from the tree; weighted caps are the largest
            // weight any of *this run's* samples put on the edge, which is why
            // weighted ERS sketches can never be portable.
            let caps: Vec<f64> = if params.weighted {
                let t_caps = Instant::now();
                let d = active_edges.len();
                let caps = wsets
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
                    "{log_label}ERS: caps(max_w) built in {} ms",
                    t_caps.elapsed().as_millis()
                );
                caps
            } else {
                active_edges.iter().map(|&v| lens[v]).collect()
            };

            let ers = ErsWmh::new_mt(&mut rng, &caps, params.k as u64);
            let t_ers = Instant::now();
            let sketches: Vec<Vec<u64>> = wsets
                .par_iter()
                .map(|ws| {
                    ers.sketch(ws, Some(params.ers_l))
                        .into_iter()
                        .map(|(id, _rank)| id)
                        .collect()
                })
                .collect();
            info!(
                "{log_label}ERS: sketching in {} ms",
                t_ers.elapsed().as_millis()
            );
            sketches
        }
    }
}

fn validate(tree: &Tree, table: &Table) -> Result<(), CoreError> {
    let malformed = |m: String| Err(CoreError::MalformedTable(m));
    if tree.parent.len() != tree.lens.len() {
        return malformed(format!(
            "tree parent[] has {} entries but branch lengths have {}",
            tree.parent.len(),
            tree.lens.len()
        ));
    }
    if tree.parent.is_empty() {
        return malformed("tree is empty".to_string());
    }
    if table.colptr.len() != table.n_samples + 1 {
        return malformed(format!(
            "colptr has {} entries, expected n_samples + 1 = {}",
            table.colptr.len(),
            table.n_samples + 1
        ));
    }
    if table.node.len() != table.value.len() {
        return malformed(format!(
            "node[] has {} entries but value[] has {}",
            table.node.len(),
            table.value.len()
        ));
    }
    if table.col_sums.len() != table.n_samples {
        return malformed(format!(
            "col_sums has {} entries, expected n_samples = {}",
            table.col_sums.len(),
            table.n_samples
        ));
    }
    if table.colptr[0] != 0 {
        return malformed(format!("colptr must start at 0, got {}", table.colptr[0]));
    }
    if *table.colptr.last().expect("checked non-empty above") != table.node.len() {
        return malformed(format!(
            "colptr ends at {} but there are {} entries",
            table.colptr.last().unwrap(),
            table.node.len()
        ));
    }
    for s in 0..table.n_samples {
        if table.colptr[s] > table.colptr[s + 1] {
            return malformed(format!("colptr decreases at sample {s}"));
        }
    }
    let total = tree.parent.len();
    if let Some(&bad) = table.node.iter().find(|&&v| v >= total) {
        return malformed(format!(
            "node id {bad} is outside the tree, which has {total} nodes"
        ));
    }

    // Every parent pointer must land inside the tree, and following them must
    // terminate. Both matter more than they look: the leaf-to-root climb indexes
    // `acc[v]` straight off the pointer, so an out-of-range parent panics, and
    // the weighted climb has no revisit guard, so a cycle spins forever rather
    // than failing. The unweighted climb would stop on its own, which makes the
    // hang mode depend on the weighting -- worse than a plain error either way.
    //
    // One pass over the forest, three-colouring each node, so this stays O(n)
    // rather than re-walking every chain to the root.
    const UNVISITED: u8 = 0;
    const ON_PATH: u8 = 1;
    const SETTLED: u8 = 2;
    let mut state = vec![UNVISITED; total];
    let mut path = Vec::new();
    for start in 0..total {
        if state[start] != UNVISITED {
            continue;
        }
        let mut v = start;
        loop {
            if state[v] == ON_PATH {
                return malformed(format!("parent pointers form a cycle through node {v}"));
            }
            if state[v] == SETTLED {
                break;
            }
            state[v] = ON_PATH;
            path.push(v);

            let p = tree.parent[v];
            if p == NO_PARENT {
                break;
            }
            if p >= total {
                return malformed(format!(
                    "node {v} has parent {p}, outside the tree's {total} nodes"
                ));
            }
            v = p;
        }
        for &v in &path {
            state[v] = SETTLED;
        }
        path.clear();
    }
    Ok(())
}

/// Accumulate each sample's weighted branch set and sketch it.
///
/// `log_label` prefixes this crate's progress messages so a caller running more
/// than one configuration can tell them apart; pass `""` for none.
///
/// Samples whose weighted set comes out empty are dropped; [`SketchSet::kept`]
/// reports which input samples survived.
pub fn build_sketches(
    tree: &Tree,
    table: &Table,
    params: &SketchParams,
    log_label: &str,
) -> Result<SketchSet, CoreError> {
    validate(tree, table)?;

    let what = if params.weighted { "weighted" } else { "presence" };
    info!("{log_label}building per-sample {what} sets …");
    let t0 = Instant::now();
    let wsets_by_vid = accumulate_weighted_sets(tree, table, params);
    info!(
        "{log_label}built {what} sets in {} ms",
        t0.elapsed().as_millis()
    );

    // Drop empty samples, remembering which inputs survived so the caller can
    // re-associate its own sample identifiers.
    let mut kept = Vec::with_capacity(table.n_samples);
    let mut wsets = Vec::with_capacity(table.n_samples);
    for (i, ws) in wsets_by_vid.into_iter().enumerate() {
        if !ws.is_empty() {
            kept.push(i);
            wsets.push(ws);
        }
    }
    if kept.len() < 2 {
        return Err(CoreError::FewerThanTwoNonEmptySamples);
    }

    let active_edges = compact_ids(tree, &mut wsets, params.portable)?;
    info!(
        "{log_label}active edges = {} (from {} total)",
        active_edges.len(),
        tree.parent.len()
    );

    info!("{log_label}sketching starting...");
    let sketches = sketch_all(&wsets, &active_edges, &tree.lens, params, log_label);
    info!("{log_label}sketching done.");

    Ok(SketchSet { kept, sketches })
}

#[cfg(test)]
mod tests {
    use super::*;

    // lens[0] is a zero-length edge, lens[3] belongs to an untouched branch.
    fn lens() -> Vec<f64> {
        vec![0.0, 0.5, 0.25, 0.75]
    }

    #[test]
    fn run_local_space_covers_only_touched_positive_edges() {
        let used = vec![true, true, true, false];
        assert_eq!(active_edge_ids(4, &lens(), &used, false), vec![1, 2]);
    }

    #[test]
    fn portable_space_covers_every_positive_edge_regardless_of_use() {
        let used = vec![true, true, true, false];
        assert_eq!(active_edge_ids(4, &lens(), &used, true), vec![1, 2, 3]);
    }

    /// A zero-length branch carries no UniFrac mass, so it is excluded from the
    /// id space in both modes — including when a sample touches it.
    #[test]
    fn zero_length_edges_are_never_active() {
        let used = vec![true, true, true, true];
        assert!(!active_edge_ids(4, &lens(), &used, false).contains(&0));
        assert!(!active_edge_ids(4, &lens(), &used, true).contains(&0));
    }

    /// The compaction must be order-preserving and dense, because the position
    /// of an edge in this vector *is* the id that gets hashed.
    #[test]
    fn active_edges_are_ascending_so_ids_are_reproducible() {
        let lens = vec![1.0; 6];
        let used = vec![false, true, false, true, true, false];
        assert_eq!(active_edge_ids(6, &lens, &used, false), vec![1, 3, 4]);
    }
}
