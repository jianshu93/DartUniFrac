use anyhow::{bail, Context, Result};
use clap::{Arg, Command};
use hdf5::{types::VarLenUnicode, File as H5File};
use log::{info, warn};
use newick::{one_from_string, Newick, NodeID};
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    time::Instant,
};
use succparen::{
    bitwise::SparseOneNnd,
    tree::{
        balanced_parens::BalancedParensTree,
        traversal::{DepthFirstTraverse, VisitNode},
        LabelVec,
    },
};

type NwkTree = newick::NewickTree;

#[derive(Debug)]
struct FeatureTable {
    taxa_order: Vec<String>,
    sample_names: Vec<String>,

    // sample_columns[sample_idx] = [(taxon_idx, normalized_value)]
    sample_columns: Vec<Vec<(usize, f64)>>,
}

#[derive(Debug)]
struct TreeIndex {
    // Compact observed-subtree postorder arrays.
    tint: Vec<usize>,                 // postorder node -> parent postorder node
    lint: Vec<f64>,                   // postorder node -> branch length to parent
    depth: Vec<f64>,                  // postorder node -> root-to-node distance
    node_name: Vec<Option<String>>,   // postorder node -> name
    is_leaf: Vec<bool>,               // postorder node -> compact leaf?
    leaf_map: HashMap<String, usize>, // observed leaf name -> compact postorder node
    num_nodes: usize,
    master_l_total: f64,
    full_tree_nodes_seen: usize,
    full_tree_leaves_seen: usize,
    matched_taxa: usize,
    missing_taxa: usize,
}

#[derive(Debug)]
struct PairResult {
    i: usize,
    j: usize,
    distance: f64,
}

#[derive(Debug)]
struct ContributionRecord {
    sample_i: String,
    sample_j: String,
    taxon: String,
    edge_length: f64,
    signed_mass_difference: f64,
    contribution: f64,
}

struct SuccTrav<'a> {
    tree: &'a NwkTree,
    stack: Vec<(NodeID, usize, usize)>,
}

impl<'a> SuccTrav<'a> {
    fn new(tree: &'a NwkTree) -> Self {
        Self {
            tree,
            stack: vec![(tree.root(), 0, 0)],
        }
    }
}

impl<'a> DepthFirstTraverse for SuccTrav<'a> {
    type Label = ();

    fn next(&mut self) -> Option<VisitNode<Self::Label>> {
        let (id, level, nth) = self.stack.pop()?;

        let children = self.tree[id].children();
        let n_children = children.len();

        for (k, &child) in children.iter().enumerate().rev() {
            let child_nth = n_children - 1 - k;
            self.stack.push((child, level + 1, child_nth));
        }

        Some(VisitNode::new((), level, nth))
    }
}

fn main() -> Result<()> {
    env_logger::init();

    let matches = Command::new("EMDUniFrac")
        .version("0.3.1")
        .about("Fast EMD UniFrac using newick + succparen with observed-subtree pruning")
        .arg(
            Arg::new("tree")
                .short('t')
                .long("tree")
                .value_name("TREE_FILE")
                .help("Input Newick format tree file")
                .required(true),
        )
        .arg(
            Arg::new("table")
                .short('i')
                .long("input")
                .value_name("TABLE_FILE")
                .help("Input feature table: TSV or BIOM HDF5")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("OUTPUT_FILE")
                .help("Output file for distance matrix")
                .required(true),
        )
        .arg(
            Arg::new("weighted")
                .long("weighted")
                .action(clap::ArgAction::SetTrue)
                .help("Weighted EMDUniFrac. Values are normalized per sample."),
        )
        .arg(
            Arg::new("input-format")
                .long("input-format")
                .value_parser(["auto", "tsv", "biom"])
                .default_value("auto")
                .help("Input table format"),
        )
        .arg(
            Arg::new("contrib-output")
                .long("contrib-output")
                .value_name("CONTRIB_TSV")
                .help("Optional sparse TSV of per-taxon leaf-edge contributions"),
        )
        .arg(
            Arg::new("norm-output")
                .long("norm-output")
                .value_name("NORMALIZATION_TSV")
                .help(
                    "Optional TSV of per-sample weighted-normalized UniFrac normalization factors. \
                     For pair (a,b), denominator = factor[a] + factor[b].",
                ),
        )
        .get_matches();

    let tree_file = matches.get_one::<String>("tree").unwrap();
    let table_file = matches.get_one::<String>("table").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();
    let weighted = matches.get_flag("weighted");
    let input_format = matches.get_one::<String>("input-format").unwrap();
    let contrib_output = matches.get_one::<String>("contrib-output").cloned();
    let norm_output = matches.get_one::<String>("norm-output").cloned();

    let total_start = Instant::now();

    info!("tree: {tree_file}");
    info!("table: {table_file}");
    info!("input format: {input_format}");
    info!("output distance matrix: {output_file}");
    info!("weighted: {weighted}");
    info!("rayon threads: {}", rayon::current_num_threads());

    let table_start = Instant::now();
    let feature_table = read_feature_table(table_file, input_format, weighted)?;
    info!(
        "table loaded in {:.2?}: taxa={}, samples={}",
        table_start.elapsed(),
        feature_table.taxa_order.len(),
        feature_table.sample_names.len()
    );

    let tree_start = Instant::now();
    let nwk_tree = read_newick_tree(tree_file)?;
    info!("Newick parse completed in {:.2?}", tree_start.elapsed());

    let succ_start = Instant::now();
    let bp_tree: BalancedParensTree<LabelVec<()>, SparseOneNnd> =
        BalancedParensTree::new(SuccTrav::new(&nwk_tree));
    info!(
        "succparen BalancedParensTree built in {:.2?}; bp nodes reported: {}",
        succ_start.elapsed(),
        bp_tree.len()
    );

    let index_start = Instant::now();
    let tree_index = build_observed_subtree_index(&nwk_tree, &feature_table.taxa_order)?;
    info!(
        "observed-subtree index built in {:.2?}: compact_nodes={}, compact_leaves={}, full_nodes_seen={}, full_leaves_seen={}, matched_taxa={}, missing_taxa={}, total_compact_branch_length={:.6}",
        index_start.elapsed(),
        tree_index.num_nodes,
        tree_index.leaf_map.len(),
        tree_index.full_tree_nodes_seen,
        tree_index.full_tree_leaves_seen,
        tree_index.matched_taxa,
        tree_index.missing_taxa,
        tree_index.master_l_total
    );

    if tree_index.matched_taxa == 0 {
        let examples_table: Vec<_> = feature_table.taxa_order.iter().take(20).cloned().collect();
        bail!(
            "zero table taxa matched tree leaves. Example table taxa: {:?}",
            examples_table
        );
    }

    let taxa_leaf_index = build_taxa_leaf_index(&feature_table.taxa_order, &tree_index.leaf_map);

    if let Some(norm_path) = norm_output {
        let norm_start = Instant::now();

        let sample_norm_factors = compute_sample_normalization_factors(
            &tree_index,
            &taxa_leaf_index,
            &feature_table.sample_columns,
        );

        write_sample_normalization_factors(
            &norm_path,
            &feature_table.sample_names,
            &sample_norm_factors,
        )?;

        info!(
            "sample normalization factors written in {:.2?}: {}",
            norm_start.elapsed(),
            norm_path
        );
    }

    let n_samples = feature_table.sample_names.len();

    let pairs: Vec<(usize, usize)> = (0..n_samples)
        .flat_map(|i| (i + 1..n_samples).map(move |j| (i, j)))
        .collect();

    info!(
        "computing {} pairwise distances over compact observed subtree with {} nodes",
        pairs.len(),
        tree_index.num_nodes
    );

    let dist_start = Instant::now();

    let updates: Vec<PairResult> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let distance = compute_unifrac_for_pair(
                &tree_index,
                &taxa_leaf_index,
                &feature_table.sample_columns,
                i,
                j,
            );

            PairResult { i, j, distance }
        })
        .collect();

    info!("pairwise distances completed in {:.2?}", dist_start.elapsed());

    let mut dist_matrix = vec![0.0_f64; n_samples * n_samples];

    for result in updates {
        dist_matrix[result.i * n_samples + result.j] = result.distance;
        dist_matrix[result.j * n_samples + result.i] = result.distance;
    }

    let write_start = Instant::now();
    write_matrix(
        &feature_table.sample_names,
        &dist_matrix,
        n_samples,
        output_file,
    )?;
    info!("distance matrix written in {:.2?}", write_start.elapsed());

    if let Some(contrib_path) = contrib_output {
        let contrib_start = Instant::now();

        write_taxa_contributions(
            &contrib_path,
            &tree_index,
            &feature_table.taxa_order,
            &taxa_leaf_index,
            &feature_table.sample_names,
            &feature_table.sample_columns,
        )?;

        info!(
            "taxon contribution file written in {:.2?}: {}",
            contrib_start.elapsed(),
            contrib_path
        );
    }

    info!("total elapsed time: {:.2?}", total_start.elapsed());

    Ok(())
}

fn read_newick_tree(path: &str) -> Result<NwkTree> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read Newick tree: {path}"))?;

    info!("raw Newick bytes: {}", raw.len());

    let sanitized = sanitize_newick_drop_internal_labels_and_comments(&raw);

    info!(
        "sanitized Newick bytes after removing comments/internal labels: {}",
        sanitized.len()
    );

    let tree: NwkTree = one_from_string(&sanitized)
        .with_context(|| format!("failed to parse sanitized Newick tree: {path}"))?;

    info!("Newick tree root node id: {}", tree.root());

    Ok(tree)
}

/// Remove bracket annotations/comments and internal labels after ')'.
/// Leaf labels and branch lengths are preserved.
fn sanitize_newick_drop_internal_labels_and_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'[' => {
                i += 1;
                let mut depth = 1usize;

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

/// Build compact observed-subtree arrays.
///
/// This is the important speedup:
/// - only table-observed leaves are matched
/// - only those leaves and their ancestors are retained
/// - pairwise EMD only walks retained compact nodes
///
/// Branch lengths are preserved on retained original edges. Unary ancestors are
/// kept rather than compressed, which preserves your tested full-tree logic
/// exactly while avoiding unrelated side branches.
fn build_observed_subtree_index(tree: &NwkTree, taxa_order: &[String]) -> Result<TreeIndex> {
    let root = tree.root();

    let taxa_set: HashSet<&str> = taxa_order.iter().map(|s| s.as_str()).collect();

    let mut retained = HashSet::<NodeID>::new();
    let mut matched_leaf_node = HashMap::<String, NodeID>::new();

    let mut full_tree_nodes_seen = 0usize;
    let mut full_tree_leaves_seen = 0usize;
    let mut duplicate_matched_taxa = 0usize;

    let mut path = Vec::<NodeID>::new();

    mark_observed_paths_dfs(
        tree,
        root,
        &taxa_set,
        &mut path,
        &mut retained,
        &mut matched_leaf_node,
        &mut full_tree_nodes_seen,
        &mut full_tree_leaves_seen,
        &mut duplicate_matched_taxa,
    );

    let matched_taxa = matched_leaf_node.len();
    let missing_taxa = taxa_order.len().saturating_sub(matched_taxa);

    if duplicate_matched_taxa > 0 {
        warn!("duplicate matched leaf names in tree: {duplicate_matched_taxa}");
    }

    if matched_taxa == 0 {
        return Ok(TreeIndex {
            tint: Vec::new(),
            lint: Vec::new(),
            depth: Vec::new(),
            node_name: Vec::new(),
            is_leaf: Vec::new(),
            leaf_map: HashMap::new(),
            num_nodes: 0,
            master_l_total: 0.0,
            full_tree_nodes_seen,
            full_tree_leaves_seen,
            matched_taxa,
            missing_taxa,
        });
    }

    retained.insert(root);

    info!(
        "observed-subtree marking: retained_nodes={}, matched_taxa={}, missing_taxa={}",
        retained.len(),
        matched_taxa,
        missing_taxa
    );

    let mut postorder = Vec::<NodeID>::with_capacity(retained.len());
    let mut parent_map = HashMap::<NodeID, NodeID>::with_capacity(retained.len());

    collect_retained_postorder(tree, root, &retained, &mut postorder, &mut parent_map)?;

    if postorder.is_empty() {
        bail!("observed subtree postorder is empty despite matched taxa");
    }

    let n = postorder.len();

    let mut retained_depth_by_node = HashMap::<NodeID, f64>::with_capacity(n);
    assign_retained_depths(tree, root, &retained, 0.0, &mut retained_depth_by_node)?;

    let mut node_to_post = HashMap::<NodeID, usize>::with_capacity(n);

    for (pos, &node_id) in postorder.iter().enumerate() {
        node_to_post.insert(node_id, pos);
    }

    let root_pos = *node_to_post
        .get(&root)
        .context("root missing from compact observed-subtree postorder")?;

    let mut tint = vec![0usize; n];
    let mut lint = vec![0.0_f64; n];
    let mut depth = vec![0.0_f64; n];
    let mut is_leaf = vec![false; n];
    let mut node_name = vec![None::<String>; n];

    for &node_id in &postorder {
        let pos = node_to_post[&node_id];

        node_name[pos] = tree.name(node_id).cloned();

        depth[pos] = *retained_depth_by_node
            .get(&node_id)
            .with_context(|| format!("missing retained depth for node {node_id}"))?;

        let retained_child_count = tree[node_id]
            .children()
            .iter()
            .filter(|&&child| retained.contains(&child))
            .count();

        is_leaf[pos] = retained_child_count == 0;

        if node_id == root {
            tint[pos] = root_pos;
            lint[pos] = 0.0;
        } else {
            let parent_id = *parent_map
                .get(&node_id)
                .with_context(|| format!("retained non-root node {node_id} has no parent"))?;

            tint[pos] = node_to_post[&parent_id];
            lint[pos] = tree[node_id].branch().copied().unwrap_or(0.0) as f64;
        }
    }

    let mut leaf_map = HashMap::<String, usize>::with_capacity(matched_leaf_node.len());

    for (taxon, node_id) in matched_leaf_node {
        let Some(&post_pos) = node_to_post.get(&node_id) else {
            bail!("matched leaf {taxon} was not present in compact observed-subtree postorder");
        };

        leaf_map.insert(taxon, post_pos);
    }

    let master_l_total: f64 = lint.iter().sum();

    Ok(TreeIndex {
        tint,
        lint,
        depth,
        node_name,
        is_leaf,
        leaf_map,
        num_nodes: n,
        master_l_total,
        full_tree_nodes_seen,
        full_tree_leaves_seen,
        matched_taxa,
        missing_taxa,
    })
}

fn mark_observed_paths_dfs(
    tree: &NwkTree,
    node_id: NodeID,
    taxa_set: &HashSet<&str>,
    path: &mut Vec<NodeID>,
    retained: &mut HashSet<NodeID>,
    matched_leaf_node: &mut HashMap<String, NodeID>,
    full_tree_nodes_seen: &mut usize,
    full_tree_leaves_seen: &mut usize,
    duplicate_matched_taxa: &mut usize,
) {
    *full_tree_nodes_seen += 1;
    path.push(node_id);

    let children = tree[node_id].children();

    if children.is_empty() {
        *full_tree_leaves_seen += 1;

        let leaf_name = tree
            .name(node_id)
            .map(|s| s.to_owned())
            .unwrap_or_else(|| format!("L{node_id}"));

        if taxa_set.contains(leaf_name.as_str()) {
            for &ancestor in path.iter() {
                retained.insert(ancestor);
            }

            if matched_leaf_node.insert(leaf_name, node_id).is_some() {
                *duplicate_matched_taxa += 1;
            }
        }
    } else {
        for &child in children {
            mark_observed_paths_dfs(
                tree,
                child,
                taxa_set,
                path,
                retained,
                matched_leaf_node,
                full_tree_nodes_seen,
                full_tree_leaves_seen,
                duplicate_matched_taxa,
            );
        }
    }

    path.pop();
}

fn collect_retained_postorder(
    tree: &NwkTree,
    node_id: NodeID,
    retained: &HashSet<NodeID>,
    out: &mut Vec<NodeID>,
    parent_map: &mut HashMap<NodeID, NodeID>,
) -> Result<()> {
    if !retained.contains(&node_id) {
        return Ok(());
    }

    for &child in tree[node_id].children() {
        if retained.contains(&child) {
            parent_map.insert(child, node_id);
            collect_retained_postorder(tree, child, retained, out, parent_map)?;
        }
    }

    out.push(node_id);

    Ok(())
}

fn assign_retained_depths(
    tree: &NwkTree,
    node_id: NodeID,
    retained: &HashSet<NodeID>,
    current_depth: f64,
    depth_by_node: &mut HashMap<NodeID, f64>,
) -> Result<()> {
    if !retained.contains(&node_id) {
        return Ok(());
    }

    depth_by_node.insert(node_id, current_depth);

    for &child in tree[node_id].children() {
        if retained.contains(&child) {
            let child_branch = tree[child].branch().copied().unwrap_or(0.0) as f64;
            assign_retained_depths(
                tree,
                child,
                retained,
                current_depth + child_branch,
                depth_by_node,
            )?;
        }
    }

    Ok(())
}

fn read_feature_table(path: &str, input_format: &str, weighted: bool) -> Result<FeatureTable> {
    let fmt = match input_format {
        "tsv" => "tsv",
        "biom" => "biom",
        "auto" => {
            let lower = path.to_ascii_lowercase();

            if lower.ends_with(".biom") || lower.ends_with(".h5") || lower.ends_with(".hdf5") {
                "biom"
            } else {
                "tsv"
            }
        }
        other => bail!("unsupported input format: {other}"),
    };

    match fmt {
        "tsv" => read_sample_table_tsv(path, weighted),
        "biom" => read_sample_table_biom(path, weighted),
        _ => unreachable!(),
    }
}

fn read_sample_table_tsv(filename: &str, weighted: bool) -> Result<FeatureTable> {
    let f = File::open(filename).with_context(|| format!("failed to open table: {filename}"))?;
    let mut lines = BufReader::new(f).lines();

    let header = lines.next().context("No header in table")??;
    let mut hdr_split = header.split('\t');

    hdr_split.next();

    let sample_names: Vec<String> = hdr_split.map(|s| s.to_string()).collect();

    if sample_names.is_empty() {
        bail!("table has zero samples in header");
    }

    let n_samples = sample_names.len();

    let mut taxa_order = Vec::new();
    let mut sample_columns = vec![Vec::<(usize, f64)>::new(); n_samples];
    let mut sample_sums = vec![0.0_f64; n_samples];

    for (line_no, line) in lines.enumerate() {
        let line = line?;
        let mut parts = line.split('\t');

        let taxon = parts
            .next()
            .with_context(|| format!("taxon missing on line {}", line_no + 2))?
            .to_string();

        let taxon_idx = taxa_order.len();
        taxa_order.push(taxon);

        let mut values_seen = 0usize;

        for (sample_idx, x) in parts.enumerate() {
            values_seen += 1;

            if sample_idx >= n_samples {
                bail!(
                    "line {} has more values than header samples ({})",
                    line_no + 2,
                    n_samples
                );
            }

            let raw_val: f64 = x.parse().unwrap_or(0.0);

            let val = if weighted {
                raw_val
            } else if raw_val > 0.0 {
                1.0
            } else {
                0.0
            };

            if val != 0.0 {
                sample_columns[sample_idx].push((taxon_idx, val));
                sample_sums[sample_idx] += val;
            }
        }

        if values_seen != n_samples {
            bail!(
                "line {} has {} values, expected {}",
                line_no + 2,
                values_seen,
                n_samples
            );
        }
    }

    if taxa_order.is_empty() {
        bail!("table has zero taxa/features");
    }

    normalize_sample_columns(&mut sample_columns, &sample_sums);

    let nnz: usize = sample_columns.iter().map(|c| c.len()).sum();

    info!(
        "TSV table loaded: taxa={}, samples={}, nonzero entries={}",
        taxa_order.len(),
        sample_names.len(),
        nnz
    );

    Ok(FeatureTable {
        taxa_order,
        sample_names,
        sample_columns,
    })
}

fn read_sample_table_biom(filename: &str, weighted: bool) -> Result<FeatureTable> {
    let f = H5File::open(filename).with_context(|| format!("failed to open BIOM: {filename}"))?;

    let taxa_order =
        read_utf8_dataset(&f, "observation/ids").context("BIOM missing observation/ids")?;
    let sample_names = read_utf8_dataset(&f, "sample/ids").context("BIOM missing sample/ids")?;

    let n_obs = taxa_order.len();
    let n_samples = sample_names.len();

    if n_obs == 0 {
        bail!("BIOM has zero observations/features");
    }

    if n_samples == 0 {
        bail!("BIOM has zero samples");
    }

    info!(
        "reading BIOM observation/matrix as CSR: observations/features={}, samples={}",
        n_obs, n_samples
    );

    let indptr =
        read_u64_dataset_flexible(&f, &["observation/matrix/indptr", "observation/indptr"])?;
    let indices =
        read_u64_dataset_flexible(&f, &["observation/matrix/indices", "observation/indices"])?;
    let data = read_f64_dataset_flexible(&f, &["observation/matrix/data", "observation/data"])?;

    validate_observation_csr(n_obs, n_samples, &indptr, &indices, &data)?;

    let mut sample_columns = vec![Vec::<(usize, f64)>::new(); n_samples];
    let mut sample_sums = vec![0.0_f64; n_samples];

    for taxon_idx in 0..n_obs {
        let start = indptr[taxon_idx] as usize;
        let end = indptr[taxon_idx + 1] as usize;

        for p in start..end {
            let sample_idx = indices[p] as usize;
            let raw_val = data[p];

            let val = if weighted {
                raw_val
            } else if raw_val > 0.0 {
                1.0
            } else {
                0.0
            };

            if val != 0.0 {
                sample_columns[sample_idx].push((taxon_idx, val));
                sample_sums[sample_idx] += val;
            }
        }
    }

    normalize_sample_columns(&mut sample_columns, &sample_sums);

    let nnz: usize = sample_columns.iter().map(|c| c.len()).sum();

    info!(
        "BIOM table loaded: taxa={}, samples={}, nonzero entries={}",
        taxa_order.len(),
        sample_names.len(),
        nnz
    );

    Ok(FeatureTable {
        taxa_order,
        sample_names,
        sample_columns,
    })
}

fn normalize_sample_columns(sample_columns: &mut [Vec<(usize, f64)>], sample_sums: &[f64]) {
    for (sample_idx, col) in sample_columns.iter_mut().enumerate() {
        let sum = sample_sums[sample_idx];

        if sum <= 0.0 {
            continue;
        }

        let inv_sum = 1.0 / sum;

        for (_, val) in col.iter_mut() {
            *val *= inv_sum;
        }
    }
}

fn validate_observation_csr(
    n_obs: usize,
    n_samples: usize,
    indptr: &[u64],
    indices: &[u64],
    data: &[f64],
) -> Result<()> {
    if indptr.len() != n_obs + 1 {
        bail!(
            "invalid observation CSR: indptr length {} != observation count {} + 1",
            indptr.len(),
            n_obs
        );
    }

    if indices.len() != data.len() {
        bail!(
            "invalid observation CSR: indices length {} != data length {}",
            indices.len(),
            data.len()
        );
    }

    if indptr[0] != 0 {
        bail!("invalid observation CSR: indptr[0] is {}, expected 0", indptr[0]);
    }

    if indptr[n_obs] as usize != data.len() {
        bail!(
            "invalid observation CSR: final indptr {} != nnz {}",
            indptr[n_obs],
            data.len()
        );
    }

    for row in 0..n_obs {
        if indptr[row] > indptr[row + 1] {
            bail!(
                "invalid observation CSR: indptr decreases at row {}: {} > {}",
                row,
                indptr[row],
                indptr[row + 1]
            );
        }
    }

    for &sample_idx in indices {
        if sample_idx as usize >= n_samples {
            bail!(
                "invalid observation CSR: sample index {} out of bounds for {} samples",
                sample_idx,
                n_samples
            );
        }
    }

    Ok(())
}

fn read_utf8_dataset(f: &H5File, path: &str) -> Result<Vec<String>> {
    Ok(f.dataset(path)
        .with_context(|| format!("missing string dataset: {path}"))?
        .read_1d::<VarLenUnicode>()
        .with_context(|| format!("failed reading UTF-8 dataset: {path}"))?
        .into_iter()
        .map(|v| v.as_str().to_owned())
        .collect())
}

fn try_read_u64_dataset_flexible(f: &H5File, paths: &[&str]) -> Result<Option<Vec<u64>>> {
    for path in paths {
        if let Ok(ds) = f.dataset(path) {
            if let Ok(v) = ds.read_raw::<u64>() {
                return Ok(Some(v.to_vec()));
            }

            if let Ok(v) = ds.read_raw::<i64>() {
                return v
                    .into_iter()
                    .map(|x| {
                        if x < 0 {
                            bail!("negative value found in integer dataset {path}: {x}");
                        }

                        Ok(x as u64)
                    })
                    .collect::<Result<Vec<u64>>>()
                    .map(Some);
            }

            if let Ok(v) = ds.read_raw::<u32>() {
                return Ok(Some(v.into_iter().map(|x| x as u64).collect()));
            }

            if let Ok(v) = ds.read_raw::<i32>() {
                return v
                    .into_iter()
                    .map(|x| {
                        if x < 0 {
                            bail!("negative value found in integer dataset {path}: {x}");
                        }

                        Ok(x as u64)
                    })
                    .collect::<Result<Vec<u64>>>()
                    .map(Some);
            }

            bail!("found dataset {path}, but could not read it as an integer vector");
        }
    }

    Ok(None)
}

fn read_u64_dataset_flexible(f: &H5File, paths: &[&str]) -> Result<Vec<u64>> {
    match try_read_u64_dataset_flexible(f, paths)? {
        Some(v) => Ok(v),
        None => bail!("could not find/read any integer datasets: {:?}", paths),
    }
}

fn read_f64_dataset_flexible(f: &H5File, paths: &[&str]) -> Result<Vec<f64>> {
    for path in paths {
        if let Ok(ds) = f.dataset(path) {
            if let Ok(v) = ds.read_raw::<f64>() {
                return Ok(v.to_vec());
            }

            if let Ok(v) = ds.read_raw::<f32>() {
                return Ok(v.into_iter().map(|x| x as f64).collect());
            }

            if let Ok(v) = ds.read_raw::<u64>() {
                return Ok(v.into_iter().map(|x| x as f64).collect());
            }

            if let Ok(v) = ds.read_raw::<i64>() {
                return Ok(v.into_iter().map(|x| x as f64).collect());
            }

            if let Ok(v) = ds.read_raw::<u32>() {
                return Ok(v.into_iter().map(|x| x as f64).collect());
            }

            if let Ok(v) = ds.read_raw::<i32>() {
                return Ok(v.into_iter().map(|x| x as f64).collect());
            }

            bail!("found dataset {path}, but could not read it as a numeric vector");
        }
    }

    bail!("could not find/read any numeric datasets: {:?}", paths)
}

fn build_taxa_leaf_index(
    taxa_order: &[String],
    leaf_map: &HashMap<String, usize>,
) -> Vec<Option<usize>> {
    taxa_order
        .iter()
        .map(|taxon| leaf_map.get(taxon).copied())
        .collect()
}

fn compute_sample_normalization_factors(
    tree_index: &TreeIndex,
    taxa_leaf_index: &[Option<usize>],
    sample_columns: &[Vec<(usize, f64)>],
) -> Vec<f64> {
    sample_columns
        .iter()
        .map(|col| {
            col.iter()
                .filter_map(|&(taxon_idx, value)| {
                    taxa_leaf_index[taxon_idx].map(|leaf_idx| tree_index.depth[leaf_idx] * value)
                })
                .sum()
        })
        .collect()
}

fn write_sample_normalization_factors(
    output_path: &str,
    sample_names: &[String],
    factors: &[f64],
) -> Result<()> {
    if sample_names.len() != factors.len() {
        bail!(
            "sample normalization factor length mismatch: samples={}, factors={}",
            sample_names.len(),
            factors.len()
        );
    }

    let file = File::create(output_path)
        .with_context(|| format!("failed to create normalization output: {output_path}"))?;
    let mut writer = BufWriter::with_capacity(1 << 20, file);

    writeln!(writer, "sample\tnormalization_factor")?;

    for (sample, factor) in sample_names.iter().zip(factors.iter()) {
        writeln!(writer, "{}\t{:.12}", sample, factor)?;
    }

    writer.flush()?;

    Ok(())
}

fn compute_unifrac_for_pair(
    tree_index: &TreeIndex,
    taxa_leaf_index: &[Option<usize>],
    sample_columns: &[Vec<(usize, f64)>],
    i: usize,
    j: usize,
) -> f64 {
    let num_nodes = tree_index.num_nodes;
    let mut partial_sums = vec![0.0_f64; num_nodes];

    for &(taxon_idx, value) in &sample_columns[i] {
        if let Some(leaf_idx) = taxa_leaf_index[taxon_idx] {
            partial_sums[leaf_idx] += value;
        }
    }

    for &(taxon_idx, value) in &sample_columns[j] {
        if let Some(leaf_idx) = taxa_leaf_index[taxon_idx] {
            partial_sums[leaf_idx] -= value;
        }
    }

    let mut z = 0.0_f64;

    for node_pos in 0..num_nodes {
        let parent = tree_index.tint[node_pos];

        if parent == node_pos {
            continue;
        }

        let val = partial_sums[node_pos];

        partial_sums[parent] += val;
        z += tree_index.lint[node_pos] * val.abs();
    }

    z
}

/// Taxon contribution is leaf-edge contribution only:
/// edge_length(taxon leaf) * abs(P_taxon - Q_taxon).
fn compute_taxa_contributions_for_pair(
    tree_index: &TreeIndex,
    taxa_order: &[String],
    taxa_leaf_index: &[Option<usize>],
    sample_columns: &[Vec<(usize, f64)>],
    sample_names: &[String],
    i: usize,
    j: usize,
) -> Vec<ContributionRecord> {
    let mut diffs =
        Vec::<(usize, f64)>::with_capacity(sample_columns[i].len() + sample_columns[j].len());

    for &(taxon_idx, value) in &sample_columns[i] {
        diffs.push((taxon_idx, value));
    }

    for &(taxon_idx, value) in &sample_columns[j] {
        diffs.push((taxon_idx, -value));
    }

    diffs.sort_unstable_by_key(|x| x.0);

    let mut records = Vec::<ContributionRecord>::new();

    let mut k = 0usize;

    while k < diffs.len() {
        let taxon_idx = diffs[k].0;
        let mut diff = 0.0_f64;

        while k < diffs.len() && diffs[k].0 == taxon_idx {
            diff += diffs[k].1;
            k += 1;
        }

        if diff.abs() <= 1e-14 {
            continue;
        }

        let Some(leaf_idx) = taxa_leaf_index[taxon_idx] else {
            continue;
        };

        let edge_length = tree_index.lint[leaf_idx];
        let contribution = edge_length * diff.abs();

        if contribution <= 1e-14 {
            continue;
        }

        records.push(ContributionRecord {
            sample_i: sample_names[i].clone(),
            sample_j: sample_names[j].clone(),
            taxon: taxa_order[taxon_idx].clone(),
            edge_length,
            signed_mass_difference: diff,
            contribution,
        });
    }

    records
}

fn write_taxa_contributions(
    output_path: &str,
    tree_index: &TreeIndex,
    taxa_order: &[String],
    taxa_leaf_index: &[Option<usize>],
    sample_names: &[String],
    sample_columns: &[Vec<(usize, f64)>],
) -> Result<()> {
    let n_samples = sample_names.len();

    let pairs: Vec<(usize, usize)> = (0..n_samples)
        .flat_map(|i| (i + 1..n_samples).map(move |j| (i, j)))
        .collect();

    let file = File::create(output_path)
        .with_context(|| format!("failed to create contribution output: {output_path}"))?;
    let mut writer = BufWriter::with_capacity(16 << 20, file);

    writeln!(
        writer,
        "sample_i\tsample_j\ttaxon\tedge_length\tsigned_mass_difference\tcontribution"
    )?;

    for chunk in pairs.chunks(256) {
        let chunk_records: Vec<Vec<ContributionRecord>> = chunk
            .par_iter()
            .map(|&(i, j)| {
                compute_taxa_contributions_for_pair(
                    tree_index,
                    taxa_order,
                    taxa_leaf_index,
                    sample_columns,
                    sample_names,
                    i,
                    j,
                )
            })
            .collect();

        for records in chunk_records {
            for r in records {
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{:.12}\t{:.12}\t{:.12}",
                    r.sample_i,
                    r.sample_j,
                    r.taxon,
                    r.edge_length,
                    r.signed_mass_difference,
                    r.contribution
                )?;
            }
        }
    }

    writer.flush()?;

    Ok(())
}

fn write_matrix(
    sample_names: &[String],
    dist_matrix: &[f64],
    n: usize,
    output_file: &str,
) -> Result<()> {
    let file =
        File::create(output_file).with_context(|| format!("failed to create {output_file}"))?;
    let mut writer = BufWriter::with_capacity(16 << 20, file);

    write!(writer, "Sample")?;

    for sn in sample_names {
        write!(writer, "\t{}", sn)?;
    }

    writeln!(writer)?;

    for i in 0..n {
        write!(writer, "{}", sample_names[i])?;

        for j in 0..n {
            write!(writer, "\t{:.6}", dist_matrix[i * n + j])?;
        }

        writeln!(writer)?;
    }

    writer.flush()?;

    Ok(())
}
