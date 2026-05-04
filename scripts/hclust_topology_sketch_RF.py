#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

# Plot settings

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20,
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "grid.color": "0.7",
        "grid.linestyle": "--",
        "grid.linewidth": 0.1,
    }
)

# User settings
TREE_FILE = "./GWMC_rep_seqs_all.tre"
BIOM_FILE = "./GWMC_16S_otutab.biom"

# Exact / truth UniFrac distance matrix
TRUTH_DISTANCE_FILE = "./GWMC_16S_C++_dist.tsv"

SKETCH_SIZES = [
    1024,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
    4608,
    5120,
    5632,
    6144,
    6656,
]

DARTUNIFRAC_SEEDS = [
    1337,
    1999,
    314159,
    271828,
    8675309,
    42,
    2024,
    777,
    10007,
    65537,
]

DARTUNIFRAC_EXE = "dartunifrac"

# Hierarchical clustering linkage method.
# "average" is a good default for distance matrices.
LINKAGE_METHOD = "average"

# If False, existing distance matrices are reused.
RERUN_DARTUNIFRAC = True

# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run DartUniFrac hierarchical-clustering topology benchmark "
            "using Robinson-Foulds-style distances."
        )
    )

    parser.add_argument(
        "--method",
        choices=["dmh", "ers"],
        required=True,
        help="DartUniFrac sketch method. Possible values: dmh, ers.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, uses method-specific default.",
    )

    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing distance matrices instead of rerunning DartUniFrac.",
    )

    return parser.parse_args()


def default_output_dir(method):
    if method == "dmh":
        return "./dartunifrac_dmh_rf_topology"
    if method == "ers":
        return "./dartunifrac_ers_rf_topology"

    raise ValueError(f"Unsupported method: {method}")


# DartUniFrac runner


def run_dartunifrac(method, sketch_size, dart_seed, output_path):
    cmd = [
        DARTUNIFRAC_EXE,
        "-t",
        TREE_FILE,
        "-b",
        BIOM_FILE,
        "--weighted",
        "-m",
        method,
        "--seed",
        str(dart_seed),
        "-s",
        str(sketch_size),
        "-o",
        str(output_path),
    ]

    print(
        f"\nRunning dartunifrac method {method} "
        f"with sketch size {sketch_size}, seed {dart_seed}"
    )
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


# Distance matrix helpers
def read_distance_matrix_df(distance_file):
    df = pd.read_csv(distance_file, sep="\t", index_col=0)

    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    common_ids = df.index.intersection(df.columns)

    if len(common_ids) == 0:
        raise ValueError(
            f"No shared sample IDs between rows and columns in {distance_file}"
        )

    df = df.loc[common_ids, common_ids].copy()
    df = df.apply(pd.to_numeric, errors="coerce")

    if df.isna().any().any():
        bad_rows, bad_cols = np.where(df.isna().values)
        examples = [
            f"{df.index[i]} x {df.columns[j]}"
            for i, j in zip(bad_rows[:10], bad_cols[:10])
        ]
        raise ValueError(
            f"Non-numeric or missing values found in distance matrix {distance_file}. "
            f"Examples: {examples}"
        )

    data = np.ascontiguousarray(df.to_numpy(dtype=np.float64))

    if not np.allclose(data, data.T, atol=1e-8):
        raise ValueError(f"Distance matrix is not symmetric: {distance_file}")

    if not np.allclose(np.diag(data), 0, atol=1e-8):
        raise ValueError(f"Distance matrix diagonal is not zero: {distance_file}")

    return pd.DataFrame(
        data,
        index=df.index.tolist(),
        columns=df.columns.tolist(),
    )


def align_distance_matrices(truth_df, approx_df):
    common_ids = truth_df.index.intersection(approx_df.index)
    common_ids = common_ids.intersection(truth_df.columns)
    common_ids = common_ids.intersection(approx_df.columns)

    if len(common_ids) < 4:
        raise ValueError(
            "Too few shared samples between truth and approximate distance matrices."
        )

    truth_aligned = truth_df.loc[common_ids, common_ids].copy()
    approx_aligned = approx_df.loc[common_ids, common_ids].copy()

    return truth_aligned, approx_aligned


def distance_df_to_condensed(distance_df):
    data = distance_df.to_numpy(dtype=np.float64)
    data = np.ascontiguousarray(data)

    if not np.allclose(data, data.T, atol=1e-8):
        raise ValueError("Distance matrix is not symmetric.")

    if not np.allclose(np.diag(data), 0, atol=1e-8):
        raise ValueError("Distance matrix diagonal is not zero.")

    return squareform(data, checks=False)


def compute_linkage(distance_df, linkage_method):
    condensed_dist = distance_df_to_condensed(distance_df)

    Z = linkage(
        condensed_dist,
        method=linkage_method,
        optimal_ordering=False,
    )

    return Z


# Robinson-Foulds-style topology metrics


def linkage_to_internal_clusters(Z, n_leaves):
    """
    Convert a scipy linkage matrix into internal clusters.

    Leaves are indexed 0..n_leaves-1.
    Each internal cluster is represented as a frozenset of leaf indices.
    The final full-root cluster is included here and filtered later.
    """
    clusters = {i: frozenset([i]) for i in range(n_leaves)}
    internal_clusters = []

    for row_idx, row in enumerate(Z):
        left = int(row[0])
        right = int(row[1])

        merged = clusters[left] | clusters[right]

        new_cluster_id = n_leaves + row_idx
        clusters[new_cluster_id] = merged

        internal_clusters.append(merged)

    return internal_clusters


def rooted_clades_from_linkage(Z, n_leaves):
    """
    Rooted RF-style clades.

    Excludes:
    - singleton leaves
    - full root cluster

    For a fully bifurcating rooted tree with n leaves,
    there are n - 2 nontrivial rooted clades.
    """
    internal_clusters = linkage_to_internal_clusters(Z, n_leaves)

    clades = {cluster for cluster in internal_clusters if 1 < len(cluster) < n_leaves}

    return clades


def canonical_split(cluster, all_leaves):
    """
    Convert a cluster into an unrooted bipartition split.

    The smaller side is used as the canonical representation.
    If both sides have equal size, use lexical order for determinism.
    """
    cluster = frozenset(cluster)
    complement = frozenset(all_leaves - cluster)

    if len(cluster) < len(complement):
        return cluster

    if len(complement) < len(cluster):
        return complement

    # Tie: deterministic lexical comparison
    cluster_tuple = tuple(sorted(cluster))
    complement_tuple = tuple(sorted(complement))

    if cluster_tuple <= complement_tuple:
        return cluster

    return complement


def unrooted_splits_from_linkage(Z, n_leaves):
    """
    Unrooted RF-style bipartitions.

    Excludes:
    - singleton splits
    - full/empty splits

    Complementary splits collapse to the same canonical split.
    For a fully bifurcating unrooted tree with n leaves,
    there are n - 3 nontrivial splits.
    """
    all_leaves = frozenset(range(n_leaves))
    internal_clusters = linkage_to_internal_clusters(Z, n_leaves)

    splits = set()

    for cluster in internal_clusters:
        if len(cluster) == n_leaves:
            continue

        split = canonical_split(cluster, all_leaves)

        if 1 < len(split) < n_leaves - 1:
            splits.add(split)

    return splits


def rf_distance(set_a, set_b):
    """
    Robinson-Foulds symmetric-difference distance.
    """
    return len(set_a - set_b) + len(set_b - set_a)


def normalized_rf_distance(set_a, set_b):
    """
    Normalized RF distance using the maximum possible symmetric difference
    for the two split/clade sets being compared.
    """
    denom = len(set_a) + len(set_b)

    if denom == 0:
        return np.nan

    return rf_distance(set_a, set_b) / denom


def compare_rf_topology_to_truth(approx_df, truth_df, linkage_method):
    truth_aligned_df, approx_aligned_df = align_distance_matrices(
        truth_df=truth_df,
        approx_df=approx_df,
    )

    sample_ids = truth_aligned_df.index.tolist()
    n_leaves = len(sample_ids)

    truth_Z = compute_linkage(
        distance_df=truth_aligned_df,
        linkage_method=linkage_method,
    )

    approx_Z = compute_linkage(
        distance_df=approx_aligned_df,
        linkage_method=linkage_method,
    )

    truth_rooted = rooted_clades_from_linkage(
        Z=truth_Z,
        n_leaves=n_leaves,
    )

    approx_rooted = rooted_clades_from_linkage(
        Z=approx_Z,
        n_leaves=n_leaves,
    )

    truth_unrooted = unrooted_splits_from_linkage(
        Z=truth_Z,
        n_leaves=n_leaves,
    )

    approx_unrooted = unrooted_splits_from_linkage(
        Z=approx_Z,
        n_leaves=n_leaves,
    )

    rooted_rf = rf_distance(approx_rooted, truth_rooted)
    rooted_nrf = normalized_rf_distance(approx_rooted, truth_rooted)
    rooted_similarity = 1.0 - rooted_nrf

    unrooted_rf = rf_distance(approx_unrooted, truth_unrooted)
    unrooted_nrf = normalized_rf_distance(approx_unrooted, truth_unrooted)
    unrooted_similarity = 1.0 - unrooted_nrf

    return {
        "n_samples_compared": n_leaves,
        "truth_rooted_clades": len(truth_rooted),
        "approx_rooted_clades": len(approx_rooted),
        "rooted_rf": rooted_rf,
        "rooted_normalized_rf": rooted_nrf,
        "rooted_topology_similarity": rooted_similarity,
        "truth_unrooted_splits": len(truth_unrooted),
        "approx_unrooted_splits": len(approx_unrooted),
        "unrooted_rf": unrooted_rf,
        "unrooted_normalized_rf": unrooted_nrf,
        "unrooted_topology_similarity": unrooted_similarity,
    }


# -----------------------------
# Summarization
# -----------------------------


def summarize_replicates(replicate_results_df, method):
    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    summary_df = replicate_results_df.groupby("sketch_size", as_index=False).agg(
        rooted_rf_mean=("rooted_rf", "mean"),
        rooted_rf_sd=("rooted_rf", "std"),
        rooted_rf_se=("rooted_rf", se),
        rooted_rf_min=("rooted_rf", "min"),
        rooted_rf_max=("rooted_rf", "max"),
        rooted_nrf_mean=("rooted_normalized_rf", "mean"),
        rooted_nrf_sd=("rooted_normalized_rf", "std"),
        rooted_nrf_se=("rooted_normalized_rf", se),
        rooted_nrf_min=("rooted_normalized_rf", "min"),
        rooted_nrf_max=("rooted_normalized_rf", "max"),
        rooted_similarity_mean=("rooted_topology_similarity", "mean"),
        rooted_similarity_sd=("rooted_topology_similarity", "std"),
        rooted_similarity_se=("rooted_topology_similarity", se),
        rooted_similarity_min=("rooted_topology_similarity", "min"),
        rooted_similarity_max=("rooted_topology_similarity", "max"),
        unrooted_rf_mean=("unrooted_rf", "mean"),
        unrooted_rf_sd=("unrooted_rf", "std"),
        unrooted_rf_se=("unrooted_rf", se),
        unrooted_rf_min=("unrooted_rf", "min"),
        unrooted_rf_max=("unrooted_rf", "max"),
        unrooted_nrf_mean=("unrooted_normalized_rf", "mean"),
        unrooted_nrf_sd=("unrooted_normalized_rf", "std"),
        unrooted_nrf_se=("unrooted_normalized_rf", se),
        unrooted_nrf_min=("unrooted_normalized_rf", "min"),
        unrooted_nrf_max=("unrooted_normalized_rf", "max"),
        unrooted_similarity_mean=("unrooted_topology_similarity", "mean"),
        unrooted_similarity_sd=("unrooted_topology_similarity", "std"),
        unrooted_similarity_se=("unrooted_topology_similarity", se),
        unrooted_similarity_min=("unrooted_topology_similarity", "min"),
        unrooted_similarity_max=("unrooted_topology_similarity", "max"),
        n_samples_compared=("n_samples_compared", "first"),
        truth_rooted_clades=("truth_rooted_clades", "first"),
        approx_rooted_clades=("approx_rooted_clades", "first"),
        truth_unrooted_splits=("truth_unrooted_splits", "first"),
        approx_unrooted_splits=("approx_unrooted_splits", "first"),
        n_replicates=("rooted_rf", "count"),
    )

    fill_zero_cols = [
        "rooted_rf_sd",
        "rooted_rf_se",
        "rooted_nrf_sd",
        "rooted_nrf_se",
        "rooted_similarity_sd",
        "rooted_similarity_se",
        "unrooted_rf_sd",
        "unrooted_rf_se",
        "unrooted_nrf_sd",
        "unrooted_nrf_se",
        "unrooted_similarity_sd",
        "unrooted_similarity_se",
    ]

    for col in fill_zero_cols:
        summary_df[col] = summary_df[col].fillna(0)

    summary_df["method"] = method
    summary_df["linkage_method"] = LINKAGE_METHOD

    return summary_df


# Plotting


def plot_rf_results(summary_df, output_dir, method):
    output_dir = Path(output_dir)

    nrf_plot = (
        output_dir / f"rf_normalized_distance_{method}_mean_sd_vs_sketch_size.pdf"
    )
    nrf_png = output_dir / f"rf_normalized_distance_{method}_mean_sd_vs_sketch_size.png"

    sim_plot = (
        output_dir / f"rf_topology_similarity_{method}_mean_sd_vs_sketch_size.pdf"
    )
    sim_png = output_dir / f"rf_topology_similarity_{method}_mean_sd_vs_sketch_size.png"

    raw_rf_plot = output_dir / f"rf_raw_distance_{method}_mean_sd_vs_sketch_size.pdf"
    raw_rf_png = output_dir / f"rf_raw_distance_{method}_mean_sd_vs_sketch_size.png"

    summary_df = summary_df.copy()
    summary_df["sketch_size"] = summary_df["sketch_size"].astype(int)
    summary_df = summary_df.sort_values("sketch_size")

    x_min = summary_df["sketch_size"].min()
    x_max = summary_df["sketch_size"].max()

    # Normalized RF distance
    # Lower is better

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["rooted_nrf_mean"],
        yerr=summary_df["rooted_nrf_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label="Rooted RF mean ± SD",
    )

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["unrooted_nrf_mean"],
        yerr=summary_df["unrooted_nrf_sd"],
        marker="s",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        linestyle="--",
        color="0.35",
        label="Unrooted RF mean ± SD",
    )

    ax.hlines(
        y=0.0,
        xmin=x_min,
        xmax=x_max,
        linewidth=2,
        linestyle=":",
        color="black",
        label="Perfect agreement",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("Normalized RF distance")
    ax.set_title(f"{method.upper()} RF topology distance")

    ax.set_ylim(-0.02, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(nrf_plot, bbox_inches="tight")
    fig.savefig(nrf_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # RF topology similarity
    # Higher is better

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["rooted_similarity_mean"],
        yerr=summary_df["rooted_similarity_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label="Rooted similarity mean ± SD",
    )

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["unrooted_similarity_mean"],
        yerr=summary_df["unrooted_similarity_sd"],
        marker="s",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        linestyle="--",
        color="0.35",
        label="Unrooted similarity mean ± SD",
    )

    ax.hlines(
        y=1.0,
        xmin=x_min,
        xmax=x_max,
        linewidth=2,
        linestyle=":",
        color="black",
        label="Perfect agreement",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("RF topology similarity")
    ax.set_title(f"{method.upper()} RF topology similarity")

    ax.set_ylim(-0.02, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(sim_plot, bbox_inches="tight")
    fig.savefig(sim_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Raw RF distance
    # Lower is better

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["rooted_rf_mean"],
        yerr=summary_df["rooted_rf_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label="Rooted RF mean ± SD",
    )

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["unrooted_rf_mean"],
        yerr=summary_df["unrooted_rf_sd"],
        marker="s",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        linestyle="--",
        color="0.35",
        label="Unrooted RF mean ± SD",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("Raw RF distance")
    ax.set_title(f"{method.upper()} raw RF distance")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(raw_rf_plot, bbox_inches="tight")
    fig.savefig(raw_rf_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {nrf_plot}")
    print(f"Saved plot: {nrf_png}")
    print(f"Saved plot: {sim_plot}")
    print(f"Saved plot: {sim_png}")
    print(f"Saved plot: {raw_rf_plot}")
    print(f"Saved plot: {raw_rf_png}")


# Main workflow


def main():
    args = parse_args()

    method = args.method
    rerun_dartunifrac = not args.reuse

    output_dir = Path(args.output_dir or default_output_dir(method))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading truth distance matrix: {TRUTH_DISTANCE_FILE}")
    truth_df = read_distance_matrix_df(TRUTH_DISTANCE_FILE)

    replicate_rows = []

    for sketch_size in SKETCH_SIZES:
        for replicate_index, dart_seed in enumerate(DARTUNIFRAC_SEEDS, start=1):
            dm_file = output_dir / (
                f"unifrac_weighted_{method}_"
                f"s{sketch_size}_rep{replicate_index}_seed{dart_seed}.tsv"
            )

            if rerun_dartunifrac or not dm_file.exists():
                run_dartunifrac(
                    method=method,
                    sketch_size=sketch_size,
                    dart_seed=dart_seed,
                    output_path=dm_file,
                )
            else:
                print(f"\nReusing existing distance matrix: {dm_file}")

            approx_df = read_distance_matrix_df(dm_file)

            rf_result = compare_rf_topology_to_truth(
                approx_df=approx_df,
                truth_df=truth_df,
                linkage_method=LINKAGE_METHOD,
            )

            result_row = {
                "sketch_size": sketch_size,
                "method": method,
                "linkage_method": LINKAGE_METHOD,
                "replicate": replicate_index,
                "dartunifrac_seed": dart_seed,
                "distance_matrix": str(dm_file),
            }

            result_row.update(rf_result)
            replicate_rows.append(result_row)

            print(
                f"Method {method}, "
                f"sketch size {sketch_size}, "
                f"replicate {replicate_index}, "
                f"seed {dart_seed}: "
                f"rooted NRF = {rf_result['rooted_normalized_rf']:.6f}, "
                f"rooted similarity = {rf_result['rooted_topology_similarity']:.6f}, "
                f"unrooted NRF = {rf_result['unrooted_normalized_rf']:.6f}, "
                f"unrooted similarity = {rf_result['unrooted_topology_similarity']:.6f}"
            )

    replicate_results_df = pd.DataFrame(replicate_rows)

    replicate_results_file = output_dir / (f"rf_topology_{method}_replicates.tsv")

    replicate_results_df.to_csv(
        replicate_results_file,
        sep="\t",
        index=False,
    )

    print(f"\nSaved replicate-level RF topology results: {replicate_results_file}")

    summary_df = summarize_replicates(
        replicate_results_df=replicate_results_df,
        method=method,
    )

    summary_file = output_dir / (f"rf_topology_{method}_mean_sd.tsv")

    summary_df.to_csv(
        summary_file,
        sep="\t",
        index=False,
    )

    print(f"Saved mean/SD RF topology summary: {summary_file}")

    plot_rf_results(
        summary_df=summary_df,
        output_dir=output_dir,
        method=method,
    )

    print("\nRF topology summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
