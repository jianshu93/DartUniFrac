//! succinct-BP UniFrac  (unweighted)
//!  * --striped – single post-order pass, **parallel blocks**
//!      (works for tens-of-thousands samples)
use std::time::Instant;
use log::info;
use anyhow::{Context, Result};
use bitvec::{order::Lsb0, vec::BitVec};
use clap::{Arg, Command};
use env_logger;
use rayon::prelude::*;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader, Write, BufWriter},
    sync::Arc,
};
use clap::ArgGroup;
use newick::{one_from_string, Newick, NodeID};
use succparen::{
    bitwise::{ops::NndOne, SparseOneNnd},
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec,
    },
};
use succparen::tree::Node;
use hdf5::{File as H5File, types::VarLenUnicode}; 
use std::ptr::NonNull;

// Plain new-type – automatically `Copy`.
#[derive(Clone, Copy)]
struct DistPtr(NonNull<f64>);

//  we guarantee in the algorithm that every thread writes a
//  *disjoint* rectangle of the matrix.  No two threads touch the same
//  element → pointer can be shared.
unsafe impl Send for DistPtr {}
unsafe impl Sync for DistPtr {}

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
                while i < bytes.len() && bytes[i].is_ascii_whitespace() { i += 1; }

                // Optional internal label right after ')': quoted or unquoted.
                if i < bytes.len() && bytes[i] == b'\'' {
                    // Quoted label — skip it
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' && i + 1 < bytes.len() { i += 2; continue; }
                        if bytes[i] == b'\'' { i += 1; break; }
                        i += 1;
                    }
                    // (comments after this will be removed by the '[' arm next loop)
                } else {
                    // Unquoted run until a delimiter
                    while i < bytes.len() {
                        let c = bytes[i];
                        if c.is_ascii_whitespace() || matches!(c, b':'|b','|b')'|b'('|b';'|b'[') { break; }
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
        Self { t, stack: vec![(t.root(), 0, 0)], lens }
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

fn read_table(p: &str) -> Result<(Vec<String>, Vec<String>, Vec<Vec<f64>>)> {
    let f = File::open(p)?;
    let mut lines = BufReader::new(f).lines();
    let hdr = lines.next().context("empty table")??;
    let mut it = hdr.split('\t');
    it.next();
    let samples = it.map(|s| s.to_owned()).collect();

    let mut taxa = Vec::new();
    let mut mat  = Vec::new();
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

fn write_matrix(names: &[String], d: &[f64], n: usize, path: &str) -> Result<()> {
    // build every line in parallel
    let header = {
        let mut s = String::with_capacity(n * 16);
        s.push_str("Sample");
        for name in names { s.push('\t'); s.push_str(name); }
        s.push('\n');
        s
    };

    // one String per row; share slices of `d`
    let mut rows: Vec<String> = (0..n).into_par_iter().map(|i| {
        let mut line = String::with_capacity(n * 12);
        line.push_str(&names[i]);
        let base = i * n;
        for j in 0..n {
            // SAFETY: j in 0..n
            let val = unsafe { *d.get_unchecked(base + j) };
            line.push('\t');
            // fastest stable float→string in std (ryu); no allocation
            // Ulf Adams. 2018. Ryū: fast float-to-string conversion. In Proceedings of the 39th ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI 2018). Association for Computing Machinery, New York, NY, USA, 270–282. https://doi.org/10.1145/3192366.3192369
            line.push_str(ryu::Buffer::new().format_finite(val));
        }
        line.push('\n');
        line
    }).collect();

    // single, large write
    let mut out = BufWriter::with_capacity(16 << 20, File::create(path)?); // 16 MiB buffer
    out.write_all(header.as_bytes())?;
    for line in &mut rows {
        out.write_all(line.as_bytes())?;
        // free the row buffer early
        line.clear();
    }
    out.flush()?;
    Ok(())
}

// help function in unifrac_par_stripd_par to count relevant branches
fn log_relevant_branch_counts_from_bits(node_bits: &[bitvec::vec::BitVec<u64, Lsb0>], lens: &[f32]) {
    // assumes caller checked log level
    let total = lens.len();
    if node_bits.is_empty() { return; }
    let nsamp = node_bits[0].len();

    // only positive-length edges count as “branches”
    let total_branches = lens.iter().filter(|&&l| l > 0.0).count();
    let mut rel_counts = vec![0u32; nsamp];

    for v in 0..total {
        if lens[v] <= 0.0 { continue; }
        let words = node_bits[v].as_raw_slice(); // &[u64], bit s set sample s covers node v
        for (wi, &w0) in words.iter().enumerate() {
            let mut w = w0;
            while w != 0 {
                let b = w.trailing_zeros() as usize;    // next set bit in this word
                let s = (wi << 6) + b;                  // sample index
                if s < nsamp { rel_counts[s] += 1; }
                w &= w - 1;                              // clear lowest set bit
            }
        }
    }

    for s in 0..nsamp {
        log::info!("sample {}: relevant branches = {} / {}",
                   s, rel_counts[s], total_branches);
    }
}
/// Striped UniFrac (unweighted)
fn unifrac_striped_par(
    post: &[usize],
    kids: &[Vec<usize>],
    lens: &[f32],
    leaf_ids: &[usize],
    masks: &[BitVec<u8, Lsb0>],
) -> Vec<f64>
{
    // constants
    let nsamp = masks.len();
    let total = lens.len();
    let n_threads = rayon::current_num_threads().max(1);

    let stripe = (nsamp + n_threads - 1) / n_threads;   // ceil
    let words_str = (stripe + 63) >> 6;                    // u64 / stripe

    // node_masks[tid][node][word]
    let mut node_masks: Vec<Vec<Vec<u64>>> =
        (0..n_threads)
            .map(|_| vec![vec![0u64; words_str]; total])
            .collect();

    // Phase-1 : build stripes
    let t0 = Instant::now();
    rayon::scope(|scope| {
        for (tid, node_masks_t) in node_masks.iter_mut().enumerate() {
            let stripe_start = tid * stripe;
            if stripe_start >= nsamp { break; }          // no samples left
            let stripe_end = (stripe_start + stripe).min(nsamp);
            let masks_slice = &masks[stripe_start .. stripe_end];
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

    // Merge stripes to one BitVec per node 
    let mut node_bits: Vec<BitVec<u64, Lsb0>> =
    (0..total).map(|_| BitVec::repeat(false, nsamp)).collect();

    node_bits
        .par_iter_mut()
        .enumerate()
        .for_each(|(v, bv)| {
            let dst_words = bv.as_raw_mut_slice();          // &mut [u64]
            for tid in 0..n_threads {
                let stripe_start = tid * stripe;
                let stripe_end   = (stripe_start + stripe).min(nsamp);
                if stripe_start >= stripe_end { break; }

                let src_words = &node_masks[tid][v];
                let word_off  = stripe_start >> 6;
                let bit_off   = (stripe_start & 63) as u32; // 0-63

                let n_src_words = src_words.len();
                for w in 0..n_src_words {
                    // mask tail bits (last word of the stripe)
                    let mut val = src_words[w];
                    if w == n_src_words - 1 {
                        let tail_bits = (stripe_end - stripe_start) & 63;
                        if tail_bits != 0 {
                            val &= (1u64 << tail_bits) - 1;
                        }
                    }
                    if val == 0 { continue; }

                    // low part
                    dst_words[word_off + w] |= val << bit_off;
                    // carry into the next word if mis-aligned
                    if bit_off != 0 && word_off + w + 1 < dst_words.len() {
                        dst_words[word_off + w + 1] |= val >> (64 - bit_off);
                    }
                }
            }
        });
    // extract relevant branches and count how many for testing purposes
    if log::log_enabled!(log::Level::Info) {
        log_relevant_branch_counts_from_bits(&node_bits, lens);
    }
    // Phase-2 : active nodes per matrix strip 
    let est_blk = ((nsamp as f64 / (2.0 * n_threads as f64)).sqrt()) as usize;
    let blk = est_blk.clamp(64, 512).next_power_of_two();
    let nblk = (nsamp + blk - 1) / blk;

    let mut active_per_strip: Vec<Vec<usize>> = vec![Vec::new(); nblk];

    for v in 0..total {
        let raw = node_bits[v].as_raw_slice();   // &[u64]
        for bi in 0..nblk {
            let i0 = bi * blk;
            let i1 = ((bi + 1) * blk).min(nsamp);
            let w0 =  i0 >> 6;
            let w1 = (i1 + 63) >> 6;
            if raw[w0..w1].iter().any(|&w| w != 0) {
                active_per_strip[bi].push(v);
            }
        }
    }
    log::info!("phase-2 sparse lists built ({} strips)", nblk);

    // Phase-3 : original simple block sweep
    let dist  = Arc::new(vec![0.0f64; nsamp * nsamp]);
    let ptr   = DistPtr(unsafe { NonNull::new_unchecked(dist.as_ptr() as *mut f64) });

    let pairs: Vec<(usize, usize)> =
        (0..nblk).flat_map(|bi| (bi..nblk).map(move |bj| (bi, bj))).collect();

    let t3 = Instant::now();
    pairs.into_par_iter().for_each(|(bi, bj)| {
        let ptr  = ptr;
        let (i0, i1) = (bi * blk, ((bi + 1) * blk).min(nsamp));
        let (j0, j1) = (bj * blk, ((bj + 1) * blk).min(nsamp));

        let bw = i1 - i0;
        let bh = j1 - j0;
        // let words_per_row = (nsamp + 63) >> 6;

        let mut union = vec![0.0f64; bw * bh];
        let mut shared = vec![0.0f64; bw * bh];

        let list_a = &active_per_strip[bi];
        let list_b = &active_per_strip[bj];
        let mut ia = 0;
        let mut ib = 0;

        while ia < list_a.len() || ib < list_b.len() {
            let v = match (list_a.get(ia), list_b.get(ib)) {
                (Some(&va), Some(&vb)) => {
                    if va < vb { ia += 1; va }
                    else if vb < va { ib += 1; vb }
                    else { ia += 1; ib += 1; va }
                }
                (Some(&va), None) => { ia += 1; va }
                (None,   Some(&vb)) => { ib += 1; vb }
                _ => unreachable!(),
            };

            let len = lens[v] as f64;
            let words = node_bits[v].as_raw_slice();          // &[u64]

            for ii in 0..bw {
                let samp_i = i0 + ii;
                let word_i = words[samp_i >> 6];
                let bit_i = 1u64 << (samp_i & 63);
                let a_set = (word_i & bit_i) != 0;

                for jj in 0..bh {
                    let samp_j = j0 + jj;
                    if samp_j <= samp_i { continue; }

                    let word_j = words[samp_j >> 6];
                    let bit_j = 1u64 << (samp_j & 63);
                    let b_set = (word_j & bit_j) != 0;

                    if !a_set && !b_set { continue; }          // nothing here

                    let idx = ii * bh + jj;
                    union [idx] += len;                         // a ∨ b
                    if  a_set &&  b_set { shared[idx] += len; } // a ∧ b
                }
            }
        }

        // write-back 
        unsafe {
            let base = ptr.0.as_ptr();
            for ii in 0..bw {
                let i = i0 + ii;
                for jj in 0..bh {
                    let j = j0 + jj;
                    if j <= i { continue; }
                    let idx = ii * bh + jj;
                    let u = union[idx];
                    if u == 0.0 { continue; }
                    let d = 1.0 - shared[idx] / u;
                    *base.add(i * nsamp + j) = d;
                    *base.add(j * nsamp + i) = d;
                }
            }
        }
    });
    info!("phase-3 block pass {:>6} ms", t3.elapsed().as_millis());

    Arc::try_unwrap(dist).unwrap()
}

fn read_biom_csr(p: &str)
    -> Result<(Vec<String>, Vec<String>, Vec<u32>, Vec<u32>)>
{
    let f = H5File::open(p)
        .with_context(|| format!("open BIOM file {p}"))?;

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

    // required datasets
    let taxa = read_utf8(&f, "observation/ids")
        .context("missing observation/ids")?;
    let samples = read_utf8(&f, "sample/ids")
        .context("missing sample/ids")?;

    // CSR arrays may be directly under observation/  (old BIOM 2.0)
    // or under observation/matrix/  (current spec).  Try the new path
    // first, fall back to the old one.
    let try_paths = |name: &str| -> Result<Vec<u32>> {
        read_u32(&f, &format!("observation/matrix/{name}"))
            .or_else(|_| read_u32(&f, &format!("observation/{name}")))
            .with_context(|| format!("missing observation/**/{name}"))
    };

    let indptr = try_paths("indptr")?;
    let indices = try_paths("indices")?;

    Ok((taxa, samples, indptr, indices))
}

fn main() -> Result<()> {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");
    let m = Command::new("unifrac-rs")
        .version("0.2.0")
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
        .get_matches();

    let tree_file = m.get_one::<String>("tree").unwrap();
    let out_file = m.get_one::<String>("output").unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
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
            leaf_nm.push(
                t.name(n).map(ToOwned::to_owned).unwrap_or_else(|| format!("L{n}")),
            );
        }
    }
    let t2leaf: HashMap<&str, usize> = leaf_nm
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    // children & post-order
    let total = bp.len() + 1;
    lens.resize(total, 0.0);
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);
    log::info!("Total branches with positive length: {}", lens.iter().filter(|&&l| l > 0.0).count());

    let mut node2leaf = vec![usize::MAX; total];
    for (leaf_pos, &nid) in leaf_ids.iter().enumerate() {
        node2leaf[nid] = leaf_pos;
    }
    // Read table (TSV or BIOM)
    let (taxa, samples, pres_dense);
    let mut pres = Vec::<Vec<f64>>::new();     // only for TSV
    let mut indptr = Vec::<u32>::new();         // only for BIOM
    let mut indices = Vec::<u32>::new();         // only for BIOM
    log::info!("Start parsing input.");
    if let Some(tsv) = m.get_one::<String>("input") {
        let (t,s,mat) = read_table(tsv)?;
        taxa = t;
        samples = s;
        pres = mat;
        pres_dense = true;
    } else {
        let biom = m.get_one::<String>("biom").unwrap();
        let (t,s,ip,idx) = read_biom_csr(biom)?;
        taxa = t; 
        samples = s; 
        indptr = ip;
        indices = idx;
        pres_dense = false;
    }
    

    let nsamp = samples.len();
    let mut masks: Vec<BitVec<u8, Lsb0>> =
        (0..nsamp).map(|_| BitVec::repeat(false, leaf_ids.len())).collect();

    // Build masks
    if pres_dense {
        for (ti, tax) in taxa.iter().enumerate() {
            if let Some(&leaf) = t2leaf.get(tax.as_str()) {
                for (s, bits) in masks.iter_mut().enumerate() {
                    if pres[ti][s] > 0.0 {
                        bits.set(leaf, true);
                    }
                }
            }
        }
    } else {
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
    }
    
    // Compute UniFrac
    let dist = unifrac_striped_par(&post, &kids, &lens, &leaf_ids, &masks);

    log::info!("Start writing output.");
    // Write output 
    write_matrix(&samples, &dist, nsamp, out_file)
}