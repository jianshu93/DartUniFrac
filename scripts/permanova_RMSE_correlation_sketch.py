#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skbio import DistanceMatrix
from skbio.stats.distance import permanova

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

# Metadata must be tab-delimited with columns:
# SampleID    Climate
METADATA_FILE = "./GWMC_metadata.txt"

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

# Same seed list for every sketch size.
# This gives paired stochastic comparisons across sketch sizes.
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

PERMUTATIONS = 999

DARTUNIFRAC_EXE = "dartunifrac"

# PERMANOVA permutation seed
RANDOM_SEED = 123

# If False, existing distance matrices are reused.
RERUN_DARTUNIFRAC = True

# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DartUniFrac PERMANOVA benchmark across sketch sizes."
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
        return "./dartunifrac_sketch_permanova"
    if method == "ers":
        return "./dartunifrac_ers_sketch_permanova"

    raise ValueError(f"Unsupported method: {method}")


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


def read_distance_matrix(distance_file):
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

    # scikit-bio PERMANOVA Cython code requires C-contiguous memory.
    data = np.ascontiguousarray(df.to_numpy(dtype=np.float64))

    if not np.allclose(data, data.T, atol=1e-8):
        raise ValueError(f"Distance matrix is not symmetric: {distance_file}")

    if not np.allclose(np.diag(data), 0, atol=1e-8):
        raise ValueError(f"Distance matrix diagonal is not zero: {distance_file}")

    return DistanceMatrix(data, ids=common_ids.tolist())


def load_metadata(metadata_file):
    metadata = pd.read_csv(metadata_file, sep="\t")

    required_cols = {"SampleID", "Climate"}
    missing = required_cols - set(metadata.columns)

    if missing:
        raise ValueError(f"Metadata file is missing columns: {missing}")

    metadata["SampleID"] = metadata["SampleID"].astype(str)
    metadata["Climate"] = metadata["Climate"].astype(str)

    metadata = metadata.set_index("SampleID")

    return metadata


def run_permanova_for_dm(dm, metadata):
    dm_ids = list(dm.ids)

    missing_metadata = [sid for sid in dm_ids if sid not in metadata.index]

    if missing_metadata:
        raise ValueError(
            "These samples are in the distance matrix but missing from metadata:\n"
            + "\n".join(missing_metadata[:20])
            + ("\n..." if len(missing_metadata) > 20 else "")
        )

    aligned_metadata = metadata.loc[dm_ids]
    grouping = aligned_metadata["Climate"].astype(str).to_numpy()

    result = permanova(
        distance_matrix=dm,
        grouping=grouping,
        permutations=PERMUTATIONS,
        seed=RANDOM_SEED,
    )

    pseudo_f = float(result["test statistic"])
    p_value = float(result["p-value"])

    n = int(result["sample size"])
    g = int(result["number of groups"])

    df_between = g - 1
    df_within = n - g

    # R² from PERMANOVA pseudo-F:
    #
    # F = (SS_between / df_between) / (SS_within / df_within)
    # R² = SS_between / (SS_between + SS_within)
    #
    # R² = (F * df_between) / (F * df_between + df_within)
    r_squared = (pseudo_f * df_between) / (pseudo_f * df_between + df_within)

    return {
        "n_samples": n,
        "n_groups": g,
        "pseudo_F": pseudo_f,
        "R2": r_squared,
        "p_value": p_value,
        "permutations": PERMUTATIONS,
    }


def compare_dm_to_truth(approx_dm, truth_dm):
    approx_df = approx_dm.to_data_frame()
    truth_df = truth_dm.to_data_frame()

    common_ids = approx_df.index.intersection(truth_df.index)

    if len(common_ids) < 3:
        raise ValueError(
            "Too few shared samples between approximate and truth distance matrices."
        )

    approx_df = approx_df.loc[common_ids, common_ids]
    truth_df = truth_df.loc[common_ids, common_ids]

    approx_data = approx_df.to_numpy(dtype=np.float64)
    truth_data = truth_df.to_numpy(dtype=np.float64)

    tri = np.triu_indices_from(approx_data, k=1)

    approx_vec = approx_data[tri]
    truth_vec = truth_data[tri]

    diff = approx_vec - truth_vec

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs_error = float(np.max(np.abs(diff)))
    bias = float(np.mean(diff))

    if np.std(approx_vec) == 0 or np.std(truth_vec) == 0:
        pearson_r = np.nan
    else:
        pearson_r = float(np.corrcoef(approx_vec, truth_vec)[0, 1])

    # Spearman rho without requiring scipy.
    spearman_rho = float(
        pd.Series(approx_vec).corr(pd.Series(truth_vec), method="spearman")
    )

    return {
        "dm_n_samples_compared": len(common_ids),
        "dm_rmse_vs_truth": rmse,
        "dm_mae_vs_truth": mae,
        "dm_max_abs_error_vs_truth": max_abs_error,
        "dm_bias_vs_truth": bias,
        "dm_pearson_r_vs_truth": pearson_r,
        "dm_spearman_rho_vs_truth": spearman_rho,
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
        pseudo_F_mean=("pseudo_F", "mean"),
        pseudo_F_sd=("pseudo_F", "std"),
        pseudo_F_se=("pseudo_F", se),
        pseudo_F_min=("pseudo_F", "min"),
        pseudo_F_max=("pseudo_F", "max"),
        pseudo_F_cv=("pseudo_F", cv),
        R2_mean=("R2", "mean"),
        R2_sd=("R2", "std"),
        R2_se=("R2", se),
        R2_min=("R2", "min"),
        R2_max=("R2", "max"),
        R2_cv=("R2", cv),
        p_value_mean=("p_value", "mean"),
        p_value_sd=("p_value", "std"),
        dm_rmse_mean=("dm_rmse_vs_truth", "mean"),
        dm_rmse_sd=("dm_rmse_vs_truth", "std"),
        dm_mae_mean=("dm_mae_vs_truth", "mean"),
        dm_mae_sd=("dm_mae_vs_truth", "std"),
        dm_max_abs_error_mean=("dm_max_abs_error_vs_truth", "mean"),
        dm_max_abs_error_sd=("dm_max_abs_error_vs_truth", "std"),
        dm_bias_mean=("dm_bias_vs_truth", "mean"),
        dm_bias_sd=("dm_bias_vs_truth", "std"),
        dm_pearson_r_mean=("dm_pearson_r_vs_truth", "mean"),
        dm_pearson_r_sd=("dm_pearson_r_vs_truth", "std"),
        dm_spearman_rho_mean=("dm_spearman_rho_vs_truth", "mean"),
        dm_spearman_rho_sd=("dm_spearman_rho_vs_truth", "std"),
        n_samples=("n_samples", "first"),
        n_groups=("n_groups", "first"),
        dm_n_samples_compared=("dm_n_samples_compared", "first"),
        permutations=("permutations", "first"),
        n_replicates=("pseudo_F", "count"),
    )

    fill_zero_cols = [
        "pseudo_F_sd",
        "pseudo_F_se",
        "pseudo_F_cv",
        "R2_sd",
        "R2_se",
        "R2_cv",
        "p_value_sd",
        "dm_rmse_sd",
        "dm_mae_sd",
        "dm_max_abs_error_sd",
        "dm_bias_sd",
        "dm_pearson_r_sd",
        "dm_spearman_rho_sd",
    ]

    for col in fill_zero_cols:
        summary_df[col] = summary_df[col].fillna(0)

    summary_df["method"] = method

    return summary_df


def add_truth_to_summary(summary_df, truth_result, truth_distance_file):
    truth_row = {
        "sketch_size": "truth",
        "method": "truth",
        "pseudo_F_mean": truth_result["pseudo_F"],
        "pseudo_F_sd": 0.0,
        "pseudo_F_se": 0.0,
        "pseudo_F_min": truth_result["pseudo_F"],
        "pseudo_F_max": truth_result["pseudo_F"],
        "pseudo_F_cv": 0.0,
        "R2_mean": truth_result["R2"],
        "R2_sd": 0.0,
        "R2_se": 0.0,
        "R2_min": truth_result["R2"],
        "R2_max": truth_result["R2"],
        "R2_cv": 0.0,
        "p_value_mean": truth_result["p_value"],
        "p_value_sd": 0.0,
        "dm_rmse_mean": 0.0,
        "dm_rmse_sd": 0.0,
        "dm_mae_mean": 0.0,
        "dm_mae_sd": 0.0,
        "dm_max_abs_error_mean": 0.0,
        "dm_max_abs_error_sd": 0.0,
        "dm_bias_mean": 0.0,
        "dm_bias_sd": 0.0,
        "dm_pearson_r_mean": 1.0,
        "dm_pearson_r_sd": 0.0,
        "dm_spearman_rho_mean": 1.0,
        "dm_spearman_rho_sd": 0.0,
        "n_samples": truth_result["n_samples"],
        "n_groups": truth_result["n_groups"],
        "dm_n_samples_compared": truth_result["n_samples"],
        "permutations": truth_result["permutations"],
        "n_replicates": 1,
        "distance_matrix": str(truth_distance_file),
    }

    truth_df = pd.DataFrame([truth_row])
    return pd.concat([summary_df, truth_df], ignore_index=True)


def add_truth_to_replicates(replicate_results_df, truth_result, truth_distance_file):
    truth_row = {
        "sketch_size": "truth",
        "method": "truth",
        "replicate": "truth",
        "dartunifrac_seed": "truth",
        "pseudo_F": truth_result["pseudo_F"],
        "R2": truth_result["R2"],
        "p_value": truth_result["p_value"],
        "n_samples": truth_result["n_samples"],
        "n_groups": truth_result["n_groups"],
        "permutations": truth_result["permutations"],
        "dm_n_samples_compared": truth_result["n_samples"],
        "dm_rmse_vs_truth": 0.0,
        "dm_mae_vs_truth": 0.0,
        "dm_max_abs_error_vs_truth": 0.0,
        "dm_bias_vs_truth": 0.0,
        "dm_pearson_r_vs_truth": 1.0,
        "dm_spearman_rho_vs_truth": 1.0,
        "distance_matrix": str(truth_distance_file),
    }

    truth_df = pd.DataFrame([truth_row])
    return pd.concat([replicate_results_df, truth_df], ignore_index=True)


def plot_results(summary_df, truth_result, output_dir, method):
    output_dir = Path(output_dir)

    f_plot = output_dir / f"permanova_{method}_pseudoF_mean_sd_vs_sketch_size.pdf"
    r2_plot = output_dir / f"permanova_{method}_R2_mean_sd_vs_sketch_size.pdf"
    rmse_plot = output_dir / f"distance_rmse_{method}_mean_sd_vs_sketch_size.pdf"
    corr_plot = output_dir / f"distance_correlation_{method}_mean_sd_vs_sketch_size.pdf"

    sketch_summary_df = summary_df.copy()
    sketch_summary_df["sketch_size"] = sketch_summary_df["sketch_size"].astype(int)
    sketch_summary_df = sketch_summary_df.sort_values("sketch_size")

    x = sketch_summary_df["sketch_size"].to_numpy()
    x_min = x.min()
    x_max = x.max()

    truth_f = truth_result["pseudo_F"]
    truth_r2 = truth_result["R2"]

    # PERMANOVA pseudo-F mean ± SD

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        sketch_summary_df["sketch_size"],
        sketch_summary_df["pseudo_F_mean"],
        yerr=sketch_summary_df["pseudo_F_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label=f"DartUniFrac {method.upper()} mean ± SD",
    )

    ax.hlines(
        y=truth_f,
        xmin=x_min,
        xmax=x_max,
        linewidth=2,
        linestyle="--",
        color="black",
        label="Truth C++",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("PERMANOVA pseudo-F")
    ax.set_title(f"{method.upper()} pseudo-F vs sketch size")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(f_plot, bbox_inches="tight")
    plt.close(fig)

    # PERMANOVA R² mean ± SD

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        sketch_summary_df["sketch_size"],
        sketch_summary_df["R2_mean"],
        yerr=sketch_summary_df["R2_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label=f"DartUniFrac {method.upper()} mean ± SD",
    )

    ax.hlines(
        y=truth_r2,
        xmin=x_min,
        xmax=x_max,
        linewidth=2,
        linestyle="--",
        color="black",
        label="Truth C++",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel(r"PERMANOVA $R^2$")
    ax.set_title(rf"{method.upper()} $R^2$ vs sketch size")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(r2_plot, bbox_inches="tight")
    plt.close(fig)

    # Distance RMSE versus truth

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        sketch_summary_df["sketch_size"],
        sketch_summary_df["dm_rmse_mean"],
        yerr=sketch_summary_df["dm_rmse_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label=f"DartUniFrac {method.upper()} mean ± SD",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("Distance RMSE vs truth")
    ax.set_title(f"{method.upper()} distance error vs sketch size")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(rmse_plot, bbox_inches="tight")
    plt.close(fig)

    # Pearson r and Spearman rho versus truth

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        sketch_summary_df["sketch_size"],
        sketch_summary_df["dm_pearson_r_mean"],
        yerr=sketch_summary_df["dm_pearson_r_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label="Pearson r mean ± SD",
    )

    ax.errorbar(
        sketch_summary_df["sketch_size"],
        sketch_summary_df["dm_spearman_rho_mean"],
        yerr=sketch_summary_df["dm_spearman_rho_sd"],
        marker="s",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        linestyle="--",
        color="0.35",
        label="Spearman rho mean ± SD",
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
    ax.set_ylabel("Distance correlation vs truth")
    ax.set_title(f"{method.upper()} distance correlation vs truth")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(corr_plot, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {f_plot}")
    print(f"Saved plot: {r2_plot}")
    print(f"Saved plot: {rmse_plot}")
    print(f"Saved plot: {corr_plot}")


# Main workflow
def main():
    args = parse_args()

    method = args.method
    rerun_dartunifrac = not args.reuse

    output_dir = Path(args.output_dir or default_output_dir(method))
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(METADATA_FILE)

    print(f"\nReading truth distance matrix: {TRUTH_DISTANCE_FILE}")
    truth_dm = read_distance_matrix(TRUTH_DISTANCE_FILE)

    truth_result = run_permanova_for_dm(
        dm=truth_dm,
        metadata=metadata,
    )

    print(
        f"Truth C++: "
        f"F = {truth_result['pseudo_F']:.6f}, "
        f"R² = {truth_result['R2']:.6f}, "
        f"p = {truth_result['p_value']:.6f}"
    )

    all_results = []

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

            dm = read_distance_matrix(dm_file)

            permanova_result = run_permanova_for_dm(
                dm=dm,
                metadata=metadata,
            )

            dm_error_result = compare_dm_to_truth(
                approx_dm=dm,
                truth_dm=truth_dm,
            )

            result_row = {
                "sketch_size": sketch_size,
                "method": method,
                "replicate": replicate_index,
                "dartunifrac_seed": dart_seed,
                "distance_matrix": str(dm_file),
            }

            result_row.update(permanova_result)
            result_row.update(dm_error_result)

            all_results.append(result_row)

            print(
                f"Method {method}, "
                f"sketch size {sketch_size}, "
                f"replicate {replicate_index}, "
                f"seed {dart_seed}: "
                f"F = {permanova_result['pseudo_F']:.6f}, "
                f"R² = {permanova_result['R2']:.6f}, "
                f"RMSE = {dm_error_result['dm_rmse_vs_truth']:.6g}, "
                f"Pearson r = {dm_error_result['dm_pearson_r_vs_truth']:.6f}, "
                f"Spearman rho = {dm_error_result['dm_spearman_rho_vs_truth']:.6f}"
            )

    replicate_results_df = pd.DataFrame(all_results)

    replicate_results_file = output_dir / "permanova_by_sketch_size_replicates.tsv"
    replicate_results_df.to_csv(replicate_results_file, sep="\t", index=False)

    print(f"\nSaved replicate-level PERMANOVA results: {replicate_results_file}")

    replicate_results_with_truth_df = add_truth_to_replicates(
        replicate_results_df=replicate_results_df,
        truth_result=truth_result,
        truth_distance_file=TRUTH_DISTANCE_FILE,
    )

    replicate_results_with_truth_file = (
        output_dir / "permanova_by_sketch_size_replicates_with_truth.tsv"
    )

    replicate_results_with_truth_df.to_csv(
        replicate_results_with_truth_file,
        sep="\t",
        index=False,
    )

    print(
        f"Saved replicate-level PERMANOVA results with truth: "
        f"{replicate_results_with_truth_file}"
    )

    summary_df = summarize_replicates(
        replicate_results_df=replicate_results_df,
        method=method,
    )

    summary_file = output_dir / "permanova_by_sketch_size_mean_sd.tsv"
    summary_df.to_csv(summary_file, sep="\t", index=False)

    print(f"Saved mean/SD summary: {summary_file}")

    summary_with_truth_df = add_truth_to_summary(
        summary_df=summary_df,
        truth_result=truth_result,
        truth_distance_file=TRUTH_DISTANCE_FILE,
    )

    summary_with_truth_file = (
        output_dir / "permanova_by_sketch_size_mean_sd_with_truth.tsv"
    )

    summary_with_truth_df.to_csv(
        summary_with_truth_file,
        sep="\t",
        index=False,
    )

    print(f"Saved mean/SD summary with truth: {summary_with_truth_file}")

    plot_results(
        summary_df=summary_df,
        truth_result=truth_result,
        output_dir=output_dir,
        method=method,
    )

    print("\nMean/SD summary:")
    print(summary_with_truth_df.to_string(index=False))


if __name__ == "__main__":
    main()
