//! DartUniFrac (approximate unweighted UniFrac via Weighted MinHash)
//! Methods: DartMinHash or ERS (Efficient Rejection Sampling)
//! Tree parsing via succinct-BP (balanced parenthesis)
//! Input: TSV or BIOM (HDF5) feature tables
//! Output: TSV distance matrix

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

use newick::{one_from_filename, Newick, NodeID};
use succparen::{
    bitwise::{ops::NndOne, SparseOneNnd},
    tree::{
        balanced_parens::{BalancedParensTree, Node as BpNode},
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec,
    },
    tree::Node,
};
use hdf5::{types::VarLenUnicode, File as H5File};

// Weighted MinHash backends
use dartminhash::{rng_utils::mt_from_seed, DartMinHash, ErsWmh};

// SIMD Hamming (normalized) over u64
use anndists::dist::{Distance, DistHamming};

type NwkTree = newick::NewickTree;

// ---------------- Tree traversal to collect branch lengths ----------------

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

// ---------------- TSV / BIOM readers (presence/absence) ----------------

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

// ---------------- Write TSV matrix ----------------

fn write_matrix(names: &[String], d: &[f64], n: usize, path: &str) -> Result<()> {
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

// -------------- Build per-node sample bitsets (presence under node) --------------

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
    let words_str = (stripe + 63) >> 6;               // u64 words per stripe

    // node_masks[tid][node][word] – per-thread stripes
    let mut node_masks: Vec<Vec<Vec<u64>>> =
        (0..n_threads).map(|_| vec![vec![0u64; words_str]; total_nodes]).collect();

    // Phase 1: build per-thread stripes bottom-up
    rayon::scope(|scope| {
        for (tid, node_masks_t) in node_masks.iter_mut().enumerate() {
            let stripe_start = tid * stripe;
            if stripe_start >= nsamp { break; }
            let stripe_end = (stripe_start + stripe).min(nsamp);
            let masks_slice = &masks[stripe_start..stripe_end];

            scope.spawn(move |_| {
                // scatter leaves for this stripe
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
    let mut node_bits: Vec<bitvec::vec::BitVec<u64, Lsb0>> =
        (0..total_nodes).map(|_| bitvec::vec::BitVec::repeat(false, nsamp)).collect();

    node_bits.par_iter_mut().enumerate().for_each(|(v, bv)| {
        let dst_words = bv.as_raw_mut_slice();
        for tid in 0..n_threads {
            let stripe_start = tid * stripe;
            let stripe_end   = (stripe_start + stripe).min(nsamp);
            if stripe_start >= stripe_end { break; }

            let src_words = &node_masks[tid][v];
            let word_off  = stripe_start >> 6;
            let bit_off   = (stripe_start & 63) as u32;

            let n_src_words = src_words.len();
            for w in 0..n_src_words {
                let mut val = src_words[w];
                if w == n_src_words - 1 {
                    let tail_bits = (stripe_end - stripe_start) & 63;
                    if tail_bits != 0 {
                        val &= (1u64 << tail_bits) - 1;
                    }
                }
                if val == 0 { continue; }
                dst_words[word_off + w] |= val << bit_off;
                if bit_off != 0 && word_off + w + 1 < dst_words.len() {
                    dst_words[word_off + w + 1] |= val >> (64 - bit_off);
                }
            }
        }
    });

    node_bits
}

fn main() -> Result<()> {
    println!("\n ************** initializing logger *****************\n");
    env_logger::Builder::from_default_env().init();
    log::info!("logger initialized from default environment");

    let m = Command::new("dartunifrac")
        .version("0.1.0")
        .about("Approximate unweighted UniFrac via Weighted MinHash")
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
            Arg::new("sketch-size")
                .short('s')
                .long("sketch")
                .help("Sketch size for Weighted MinHash (DartMinHash or ERS)")
                .value_parser(clap::value_parser!(usize))
                .default_value("1024"),
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
                .help("Per-hash independent random sequence length L for ERS")
                .value_parser(clap::value_parser!(u64))
                .default_value("1024"),
        )
        .arg(
            Arg::new("seed")
                .long("seed")
                .help("Random seed for reproducibility")
                .value_parser(clap::value_parser!(u64))
                .default_value("1337"),
        )
        .get_matches();

    let tree_file = m.get_one::<String>("tree").unwrap();
    let out_file = m.get_one::<String>("output").unwrap();
    let k = *m.get_one::<usize>("sketch-size").unwrap();
    let method = m.get_one::<String>("method").unwrap().as_str(); // "dmh" | "ers"
    let ers_l = *m.get_one::<u64>("seq-length").unwrap();
    let seed = *m.get_one::<u64>("seed").unwrap();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();

    info!("method={method}   k={k}   ers-L={ers_l}   seed={seed}");

    // Load tree
    let t: NwkTree = one_from_filename(tree_file).context("parse newick")?;
    let mut lens_f32 = Vec::<f32>::new();
    let trav = SuccTrav::new(&t, &mut lens_f32);
    let bp: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new_builder(trav, LabelVec::<()>::new()).build_all();

    // Leaves and mapping name -> leaf index
    let mut leaf_ids = Vec::<usize>::new();
    let mut leaf_nm = Vec::<String>::new();
    for n in t.nodes() {
        if t[n].is_leaf() {
            leaf_ids.push(n);
            leaf_nm.push(t.name(n).map(|s| s.to_owned()).unwrap_or_else(|| format!("L{n}")));
        }
    }
    let t2leaf: HashMap<&str, usize> =
        leaf_nm.iter().enumerate().map(|(i, n)| (n.as_str(), i)).collect();

    // children & postorder
    let total = bp.len() + 1;
    lens_f32.resize(total, 0.0);
    let mut kids = vec![Vec::<usize>::new(); total];
    let mut post = Vec::<usize>::with_capacity(total);
    collect_children::<SparseOneNnd>(&bp.root(), &mut kids, &mut post);

    // Read table and build leaf masks
    let (_taxa, samples, masks) = if let Some(tsv) = m.get_one::<String>("input") {
        let (taxa, samples, mat) = read_table(tsv)?;
        let nsamp = samples.len();
        let mut masks: Vec<BitVec<u8, Lsb0>> =
            (0..nsamp).map(|_| BitVec::repeat(false, leaf_ids.len())).collect();
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
        let biom = m.get_one::<String>("biom").unwrap();
        let (taxa, samples, indptr, indices) = read_biom_csr(biom)?;
        let nsamp = samples.len();
        let mut masks: Vec<BitVec<u8, Lsb0>> =
            (0..nsamp).map(|_| BitVec::repeat(false, leaf_ids.len())).collect();
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

    let nsamp = samples.len();
    info!(
        "nodes = {}  leaves = {}  samples = {}",
        total,
        leaf_ids.len(),
        nsamp
    );

    // node_bits[v]: bitset over samples that have presence under node v
    let t0 = Instant::now();
    let node_bits = build_node_bits(&post, &kids, &leaf_ids, &masks, total);
    info!("node_bits built in {} ms", t0.elapsed().as_millis());

    // Positive-length edges and weights (raw, no normalization)
    let lens: Vec<f64> = lens_f32.iter().map(|&x| x as f64).collect();
    let pos_edges: Vec<usize> = (0..total).filter(|&v| lens[v] > 0.0).collect();
    if pos_edges.is_empty() {
        warn!("No positive-length edges found.");
    }

    // Build weighted sets per sample: if sample S has presence under edge v, include (v, ℓ_v)
    let t1 = Instant::now();
    let mut wsets: Vec<Vec<(u64, f64)>> = vec![Vec::new(); nsamp];
    for &v in &pos_edges {
        let w = lens[v];
        let words = node_bits[v].as_raw_slice();
        for (wi, &w0) in words.iter().enumerate() {
            let mut word = w0;
            while word != 0 {
                let b = word.trailing_zeros() as usize;
                let s = (wi << 6) + b;
                if s < nsamp {
                    wsets[s].push((v as u64, w));
                }
                word &= word - 1;
            }
        }
    }
    info!(
        "built per-sample weighted sets in {} ms",
        t1.elapsed().as_millis()
    );

    // Sketch per sample (parallel).
    // We keep only the per-bucket **ID** in the signature.
    // Collisions are (idA[j] == idB[j]). DistHamming over u64 gives 1 - collision rate.
    let mut rng = mt_from_seed(seed);
    let sketches_u64: Vec<Vec<u64>> = if method == "dmh" {
        // DartMinHash: reuse one instance so bucket hash & dart stream are shared.
        let dmh = DartMinHash::new_mt(&mut rng, k as u64);
        wsets
            .par_iter()
            .map(|ws| {
                dmh.sketch(ws)
                    .into_iter()
                    .map(|(id, _rank)| id) // keep ID only
                    .collect::<Vec<u64>>()
            })
            .collect()
    } else {
        // ERS caps per dimension: m_i = ceil(ℓ_e), at least 1 (valid for all samples).
        let mut m_per_dim = vec![0u32; total];
        for &v in &pos_edges {
            let mut cap = lens[v].ceil();
            if cap < 1.0 {
                cap = 1.0;
            }
            let cap_u32 = if cap.is_finite() {
                cap.min(u32::MAX as f64) as u32
            } else {
                u32::MAX
            };
            m_per_dim[v] = cap_u32;
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

    // Pairwise UniFrac (≈ 1 - Jaccard) via normalized Hamming (or simply hamming similarity) on ID arrays.
    let t2 = Instant::now();
    let dist = {
        let n = nsamp;
        let dh = DistHamming;
        let mut out = vec![0.0f64; n * n];

        // Fill upper triangle (including diagonal) in parallel, per-row.
        out.par_chunks_mut(n)
            .enumerate()
            .for_each(|(i, row)| {
                row[i] = 0.0;
                for j in (i + 1)..n {
                    // DistHamming<u64> returns (# !=)/k ∈ [0,1]
                    let d = dh.eval(&sketches_u64[i], &sketches_u64[j]) as f64;
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

    // Write output
    write_matrix(&samples, &dist, nsamp, out_file)?;
    info!("Done → {}", out_file);

    Ok(())
}