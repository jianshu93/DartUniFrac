#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
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


# Input

TREE_FILE = "./GWMC_rep_seqs_all.tre"
BIOM_FILE = "./GWMC_16S_otutab.biom"

# Metadata must be tab-delimited with columns:
# SampleID    Country
METADATA_FILE = "./GWMC_metadata.txt"
METADATA_GROUP_COLUMN = "Country"

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
# Common choices: "average", "complete", "single", "ward"
# For distance matrices, "average" is usually a safe default.
LINKAGE_METHOD = "average"

# Dendrogram plotting.
# Full labeled dendrograms for ~1000+ samples are large but editable.
PLOT_DENDROGRAMS = True

# Plot dendrograms only for this replicate index for each sketch size.
# This avoids creating 10 dendrograms per sketch size.
DENDROGRAM_REPLICATE_INDEX = 1

DENDROGRAM_LABEL_FONT_SIZE = 2.5
DENDROGRAM_FIG_WIDTH = 28
DENDROGRAM_FIG_HEIGHT = 10

# If False, existing distance matrices are reused.
RERUN_DARTUNIFRAC = True


# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run DartUniFrac hierarchical-clustering benchmark "
            "using cophenetic correlation coefficient."
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

    parser.add_argument(
        "--skip-dendrograms",
        action="store_true",
        help="Skip dendrogram PDF generation.",
    )

    return parser.parse_args()


def default_output_dir(method):
    if method == "dmh":
        return "./dartunifrac_dmh_hclust_topology"
    if method == "ers":
        return "./dartunifrac_ers_hclust_topology"

    raise ValueError(f"Unsupported method: {method}")


# Metadata/color helpers


def load_metadata(metadata_file, group_column):
    metadata = pd.read_csv(metadata_file, sep="\t")

    required_cols = {"SampleID", group_column}
    missing = required_cols - set(metadata.columns)

    if missing:
        raise ValueError(f"Metadata file is missing columns: {missing}")

    metadata["SampleID"] = metadata["SampleID"].astype(str)
    metadata[group_column] = metadata[group_column].astype(str)

    metadata = metadata.set_index("SampleID")

    return metadata


def make_group_color_map(metadata, group_column):
    groups = sorted(metadata[group_column].dropna().astype(str).unique())

    # Fixed discrete categorical palette.
    # This avoids continuous colormap gradients.
    discrete_colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
        "#aec7e8",  # light blue
        "#ffbb78",  # light orange
        "#98df8a",  # light green
        "#ff9896",  # light red
        "#c5b0d5",  # light purple
        "#c49c94",  # light brown
        "#f7b6d2",  # light pink
        "#c7c7c7",  # light gray
        "#dbdb8d",  # light olive
        "#9edae5",  # light cyan
        "#393b79",
        "#637939",
        "#8c6d31",
        "#843c39",
        "#7b4173",
        "#3182bd",
        "#31a354",
        "#756bb1",
        "#636363",
        "#e6550d",
    ]

    group_color_map = {}

    for i, group in enumerate(groups):
        group_color_map[group] = discrete_colors[i % len(discrete_colors)]

    return group_color_map


def make_sample_color_map(metadata, group_column, group_color_map):
    sample_color_map = {}

    for sample_id, row in metadata.iterrows():
        group = str(row[group_column])
        sample_color_map[str(sample_id)] = group_color_map.get(group, "black")

    return sample_color_map


def save_group_color_key(group_color_map, output_dir):
    color_key_df = pd.DataFrame(
        [{"group": group, "color": color} for group, color in group_color_map.items()]
    )

    color_key_file = Path(output_dir) / f"{METADATA_GROUP_COLUMN}_color_key.tsv"
    color_key_df.to_csv(color_key_file, sep="\t", index=False)

    print(f"Saved group color key: {color_key_file}")


# Helper functions


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

    return pd.DataFrame(data, index=df.index.tolist(), columns=df.columns.tolist())


def align_distance_matrices(truth_df, approx_df):
    common_ids = truth_df.index.intersection(approx_df.index)
    common_ids = common_ids.intersection(truth_df.columns)
    common_ids = common_ids.intersection(approx_df.columns)

    if len(common_ids) < 3:
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


def compute_linkage_and_cophenetic(distance_df, linkage_method):
    condensed_dist = distance_df_to_condensed(distance_df)

    Z = linkage(
        condensed_dist,
        method=linkage_method,
        optimal_ordering=False,
    )

    coph_corr_against_input, coph_dist = cophenet(Z, condensed_dist)

    return {
        "linkage": Z,
        "input_condensed": condensed_dist,
        "cophenetic_condensed": coph_dist,
        "self_cophenetic_corr": float(coph_corr_against_input),
    }


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.shape != y.shape:
        raise ValueError("Vectors must have the same shape for correlation.")

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    return float(
        pd.Series(x).corr(
            pd.Series(y),
            method="spearman",
        )
    )


def compare_hclust_topology_to_truth(
    approx_df,
    truth_df,
    truth_cophenetic,
    linkage_method,
):
    truth_aligned_df, approx_aligned_df = align_distance_matrices(
        truth_df=truth_df,
        approx_df=approx_df,
    )

    # Recompute truth cophenetic if alignment changed.
    # Usually this will not happen, but it protects against sample mismatches.
    if list(truth_aligned_df.index) != list(truth_df.index):
        truth_info = compute_linkage_and_cophenetic(
            distance_df=truth_aligned_df,
            linkage_method=linkage_method,
        )
        truth_coph = truth_info["cophenetic_condensed"]
    else:
        truth_coph = truth_cophenetic

    approx_info = compute_linkage_and_cophenetic(
        distance_df=approx_aligned_df,
        linkage_method=linkage_method,
    )

    approx_coph = approx_info["cophenetic_condensed"]

    ccc_pearson = pearson_corr(approx_coph, truth_coph)
    ccc_spearman = spearman_corr(approx_coph, truth_coph)

    coph_diff = approx_coph - truth_coph

    coph_rmse = float(np.sqrt(np.mean(coph_diff**2)))
    coph_mae = float(np.mean(np.abs(coph_diff)))
    coph_bias = float(np.mean(coph_diff))
    coph_max_abs_error = float(np.max(np.abs(coph_diff)))

    return {
        "n_samples_compared": len(truth_aligned_df.index),
        "approx_self_cophenetic_corr": approx_info["self_cophenetic_corr"],
        "ccc_pearson_vs_truth": ccc_pearson,
        "ccc_spearman_vs_truth": ccc_spearman,
        "cophenetic_rmse_vs_truth": coph_rmse,
        "cophenetic_mae_vs_truth": coph_mae,
        "cophenetic_bias_vs_truth": coph_bias,
        "cophenetic_max_abs_error_vs_truth": coph_max_abs_error,
        "approx_linkage": approx_info["linkage"],
        "sample_ids": truth_aligned_df.index.tolist(),
    }


def summarize_replicates(replicate_results_df, method):
    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    def cv(x):
        mean = x.mean()
        if mean == 0:
            return np.nan
        return x.std(ddof=1) / mean

    summary_df = replicate_results_df.groupby("sketch_size", as_index=False).agg(
        ccc_pearson_mean=("ccc_pearson_vs_truth", "mean"),
        ccc_pearson_sd=("ccc_pearson_vs_truth", "std"),
        ccc_pearson_se=("ccc_pearson_vs_truth", se),
        ccc_pearson_min=("ccc_pearson_vs_truth", "min"),
        ccc_pearson_max=("ccc_pearson_vs_truth", "max"),
        ccc_pearson_cv=("ccc_pearson_vs_truth", cv),
        ccc_spearman_mean=("ccc_spearman_vs_truth", "mean"),
        ccc_spearman_sd=("ccc_spearman_vs_truth", "std"),
        ccc_spearman_se=("ccc_spearman_vs_truth", se),
        ccc_spearman_min=("ccc_spearman_vs_truth", "min"),
        ccc_spearman_max=("ccc_spearman_vs_truth", "max"),
        ccc_spearman_cv=("ccc_spearman_vs_truth", cv),
        cophenetic_rmse_mean=("cophenetic_rmse_vs_truth", "mean"),
        cophenetic_rmse_sd=("cophenetic_rmse_vs_truth", "std"),
        cophenetic_mae_mean=("cophenetic_mae_vs_truth", "mean"),
        cophenetic_mae_sd=("cophenetic_mae_vs_truth", "std"),
        cophenetic_bias_mean=("cophenetic_bias_vs_truth", "mean"),
        cophenetic_bias_sd=("cophenetic_bias_vs_truth", "std"),
        cophenetic_max_abs_error_mean=("cophenetic_max_abs_error_vs_truth", "mean"),
        cophenetic_max_abs_error_sd=("cophenetic_max_abs_error_vs_truth", "std"),
        approx_self_cophenetic_corr_mean=("approx_self_cophenetic_corr", "mean"),
        approx_self_cophenetic_corr_sd=("approx_self_cophenetic_corr", "std"),
        n_samples_compared=("n_samples_compared", "first"),
        n_replicates=("ccc_pearson_vs_truth", "count"),
    )

    fill_zero_cols = [
        "ccc_pearson_sd",
        "ccc_pearson_se",
        "ccc_pearson_cv",
        "ccc_spearman_sd",
        "ccc_spearman_se",
        "ccc_spearman_cv",
        "cophenetic_rmse_sd",
        "cophenetic_mae_sd",
        "cophenetic_bias_sd",
        "cophenetic_max_abs_error_sd",
        "approx_self_cophenetic_corr_sd",
    ]

    for col in fill_zero_cols:
        summary_df[col] = summary_df[col].fillna(0)

    summary_df["method"] = method
    summary_df["linkage_method"] = LINKAGE_METHOD

    return summary_df


def plot_ccc(summary_df, output_dir, method):
    output_dir = Path(output_dir)

    ccc_plot = output_dir / f"hclust_ccc_{method}_mean_sd_vs_sketch_size.pdf"
    ccc_png = output_dir / f"hclust_ccc_{method}_mean_sd_vs_sketch_size.png"

    rmse_plot = (
        output_dir / f"hclust_cophenetic_rmse_{method}_mean_sd_vs_sketch_size.pdf"
    )
    rmse_png = (
        output_dir / f"hclust_cophenetic_rmse_{method}_mean_sd_vs_sketch_size.png"
    )

    summary_df = summary_df.copy()
    summary_df["sketch_size"] = summary_df["sketch_size"].astype(int)
    summary_df = summary_df.sort_values("sketch_size")

    # CCC plot: Pearson and Spearman

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["ccc_pearson_mean"],
        yerr=summary_df["ccc_pearson_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label="Pearson CCC mean ± SD",
    )

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["ccc_spearman_mean"],
        yerr=summary_df["ccc_spearman_sd"],
        marker="s",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        linestyle="--",
        color="0.35",
        label="Spearman CCC mean ± SD",
    )

    ax.hlines(
        y=1.0,
        xmin=summary_df["sketch_size"].min(),
        xmax=summary_df["sketch_size"].max(),
        linewidth=2,
        linestyle=":",
        color="black",
        label="Perfect topology agreement",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("Cophenetic correlation vs truth")
    ax.set_title(f"{method.upper()} hierarchical topology vs sketch size")

    ax.set_ylim(0, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(ccc_plot, bbox_inches="tight")
    fig.savefig(ccc_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Cophenetic RMSE plot

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["cophenetic_rmse_mean"],
        yerr=summary_df["cophenetic_rmse_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label=f"DartUniFrac {method.upper()} mean ± SD",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("Cophenetic RMSE vs truth")
    ax.set_title(f"{method.upper()} cophenetic error vs sketch size")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(rmse_plot, bbox_inches="tight")
    fig.savefig(rmse_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {ccc_plot}")
    print(f"Saved plot: {ccc_png}")
    print(f"Saved plot: {rmse_plot}")
    print(f"Saved plot: {rmse_png}")


def plot_dendrogram_pdf(
    linkage_matrix,
    sample_ids,
    output_file,
    title,
    metadata,
    group_column,
    sample_color_map,
    group_color_map,
):
    fig, ax = plt.subplots(figsize=(DENDROGRAM_FIG_WIDTH, DENDROGRAM_FIG_HEIGHT))

    dendro = dendrogram(
        linkage_matrix,
        labels=sample_ids,
        leaf_rotation=90,
        leaf_font_size=DENDROGRAM_LABEL_FONT_SIZE,
        color_threshold=None,
        above_threshold_color="black",
        link_color_func=lambda k: "black",
        ax=ax,
    )

    # Color terminal tip labels by metadata group.
    for tick_label in ax.get_xmajorticklabels():
        sample_id = tick_label.get_text()
        tick_label.set_color(sample_color_map.get(sample_id, "black"))

    ax.set_title(title)
    ax.set_ylabel("Distance")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add a compact legend for groups. With many countries, this can be long.
    legend_handles = [
        mpl.lines.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=color,
            markeredgecolor=color,
            label=group,
        )
        for group, color in group_color_map.items()
    ]

    ax.legend(
        handles=legend_handles,
        title=group_column,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=10,
        title_fontsize=12,
        ncol=1,
    )

    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved dendrogram: {output_file}")

    # Save leaf order and group/color annotation.
    leaf_order = dendro["ivl"]
    leaf_rows = []

    for i, sample_id in enumerate(leaf_order, start=1):
        if sample_id in metadata.index:
            group = str(metadata.loc[sample_id, group_column])
        else:
            group = "NA"

        leaf_rows.append(
            {
                "leaf_order": i,
                "SampleID": sample_id,
                group_column: group,
                "color": sample_color_map.get(sample_id, "black"),
            }
        )

    leaf_order_file = Path(output_file).with_suffix(".leaf_order.tsv")
    pd.DataFrame(leaf_rows).to_csv(leaf_order_file, sep="\t", index=False)

    print(f"Saved dendrogram leaf order: {leaf_order_file}")


# Main workflow
def main():
    args = parse_args()

    method = args.method
    rerun_dartunifrac = not args.reuse
    plot_dendrograms = PLOT_DENDROGRAMS and not args.skip_dendrograms

    output_dir = Path(args.output_dir or default_output_dir(method))
    output_dir.mkdir(parents=True, exist_ok=True)

    dendrogram_dir = output_dir / "dendrograms"
    if plot_dendrograms:
        dendrogram_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(
        metadata_file=METADATA_FILE,
        group_column=METADATA_GROUP_COLUMN,
    )

    group_color_map = make_group_color_map(
        metadata=metadata,
        group_column=METADATA_GROUP_COLUMN,
    )

    sample_color_map = make_sample_color_map(
        metadata=metadata,
        group_column=METADATA_GROUP_COLUMN,
        group_color_map=group_color_map,
    )

    save_group_color_key(group_color_map, output_dir)

    print(f"\nReading truth distance matrix: {TRUTH_DISTANCE_FILE}")
    truth_df = read_distance_matrix_df(TRUTH_DISTANCE_FILE)

    missing_metadata_samples = [
        sample_id for sample_id in truth_df.index if sample_id not in metadata.index
    ]

    if missing_metadata_samples:
        print(
            "\nWarning: Some samples in the truth distance matrix are missing "
            f"from metadata. Missing count = {len(missing_metadata_samples)}"
        )
        print("First missing samples:")
        print("\n".join(missing_metadata_samples[:20]))

    print(
        f"Computing truth hierarchical clustering "
        f"using linkage method: {LINKAGE_METHOD}"
    )

    truth_info = compute_linkage_and_cophenetic(
        distance_df=truth_df,
        linkage_method=LINKAGE_METHOD,
    )

    truth_linkage = truth_info["linkage"]
    truth_cophenetic = truth_info["cophenetic_condensed"]
    truth_self_coph_corr = truth_info["self_cophenetic_corr"]

    print(f"Truth self cophenetic correlation: {truth_self_coph_corr:.6f}")

    if plot_dendrograms:
        truth_dendrogram_file = dendrogram_dir / (
            f"truth_Cpp_hclust_{LINKAGE_METHOD}_SampleID_labels_colored_by_"
            f"{METADATA_GROUP_COLUMN}.pdf"
        )

        plot_dendrogram_pdf(
            linkage_matrix=truth_linkage,
            sample_ids=truth_df.index.tolist(),
            output_file=truth_dendrogram_file,
            title=(
                f"Truth C++ hierarchical clustering "
                f"({LINKAGE_METHOD}; tips colored by {METADATA_GROUP_COLUMN})"
            ),
            metadata=metadata,
            group_column=METADATA_GROUP_COLUMN,
            sample_color_map=sample_color_map,
            group_color_map=group_color_map,
        )

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

            topo_result = compare_hclust_topology_to_truth(
                approx_df=approx_df,
                truth_df=truth_df,
                truth_cophenetic=truth_cophenetic,
                linkage_method=LINKAGE_METHOD,
            )

            approx_linkage = topo_result.pop("approx_linkage")
            sample_ids = topo_result.pop("sample_ids")

            result_row = {
                "sketch_size": sketch_size,
                "method": method,
                "linkage_method": LINKAGE_METHOD,
                "replicate": replicate_index,
                "dartunifrac_seed": dart_seed,
                "distance_matrix": str(dm_file),
            }

            result_row.update(topo_result)
            replicate_rows.append(result_row)

            print(
                f"Method {method}, "
                f"sketch size {sketch_size}, "
                f"replicate {replicate_index}, "
                f"seed {dart_seed}: "
                f"Pearson CCC = {topo_result['ccc_pearson_vs_truth']:.6f}, "
                f"Spearman CCC = {topo_result['ccc_spearman_vs_truth']:.6f}, "
                f"cophenetic RMSE = {topo_result['cophenetic_rmse_vs_truth']:.6g}"
            )

            if plot_dendrograms and replicate_index == DENDROGRAM_REPLICATE_INDEX:
                dendrogram_file = dendrogram_dir / (
                    f"hclust_{method}_{LINKAGE_METHOD}_"
                    f"s{sketch_size}_rep{replicate_index}_"
                    f"seed{dart_seed}_SampleID_labels_colored_by_"
                    f"{METADATA_GROUP_COLUMN}.pdf"
                )

                plot_dendrogram_pdf(
                    linkage_matrix=approx_linkage,
                    sample_ids=sample_ids,
                    output_file=dendrogram_file,
                    title=(
                        f"{method.upper()} sketch {sketch_size}, "
                        f"seed {dart_seed} ({LINKAGE_METHOD}; "
                        f"tips colored by {METADATA_GROUP_COLUMN})"
                    ),
                    metadata=metadata,
                    group_column=METADATA_GROUP_COLUMN,
                    sample_color_map=sample_color_map,
                    group_color_map=group_color_map,
                )

    replicate_results_df = pd.DataFrame(replicate_rows)

    replicate_results_file = output_dir / (f"hclust_topology_{method}_replicates.tsv")
    replicate_results_df.to_csv(replicate_results_file, sep="\t", index=False)

    print(f"\nSaved replicate-level topology results: {replicate_results_file}")

    summary_df = summarize_replicates(
        replicate_results_df=replicate_results_df,
        method=method,
    )

    summary_file = output_dir / (f"hclust_topology_{method}_mean_sd.tsv")
    summary_df.to_csv(summary_file, sep="\t", index=False)

    print(f"Saved mean/SD topology summary: {summary_file}")

    plot_ccc(
        summary_df=summary_df,
        output_dir=output_dir,
        method=method,
    )

    print("\nTopology summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
