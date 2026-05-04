#!/usr/bin/env python3

import argparse
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

# Same seed list for every sketch size.
# Each seed also defines one query/database split.
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

# Query/database setup
QUERY_FRACTION = 0.05
TOP_K = 10

# If False, existing distance matrices are reused.
RERUN_DARTUNIFRAC = True

# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DartUniFrac nearest-neighbor recall benchmark across sketch sizes."
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
        return "./dartunifrac_dmh_nn_recall"
    if method == "ers":
        return "./dartunifrac_ers_nn_recall"

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


def read_distance_matrix_as_df(distance_file):
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
            "Too few shared samples between truth and approximate matrices."
        )

    truth_aligned = truth_df.loc[common_ids, common_ids].copy()
    approx_aligned = approx_df.loc[common_ids, common_ids].copy()

    return truth_aligned, approx_aligned


def make_query_database_split(sample_ids, split_seed, query_fraction):
    rng = np.random.default_rng(split_seed)

    sample_ids = np.array(list(sample_ids), dtype=str)
    n_samples = len(sample_ids)

    n_query = int(round(query_fraction * n_samples))
    n_query = max(1, min(n_query, n_samples - 1))

    query_ids = rng.choice(sample_ids, size=n_query, replace=False)
    query_set = set(query_ids.tolist())

    db_ids = np.array([sid for sid in sample_ids if sid not in query_set], dtype=str)

    return query_ids.tolist(), db_ids.tolist()


def top_k_neighbors(distance_df, query_id, db_ids, k):
    distances = distance_df.loc[query_id, db_ids].to_numpy(dtype=np.float64)

    # Stable sort makes ties deterministic.
    order = np.argsort(distances, kind="mergesort")

    k_eff = min(k, len(db_ids))
    top_indices = order[:k_eff]

    return [db_ids[i] for i in top_indices]


def compute_recall_at_k(truth_df, approx_df, query_ids, db_ids, k):
    per_query_rows = []

    k_eff = min(k, len(db_ids))

    for query_id in query_ids:
        true_neighbors = top_k_neighbors(
            distance_df=truth_df,
            query_id=query_id,
            db_ids=db_ids,
            k=k_eff,
        )

        approx_neighbors = top_k_neighbors(
            distance_df=approx_df,
            query_id=query_id,
            db_ids=db_ids,
            k=k_eff,
        )

        true_set = set(true_neighbors)
        approx_set = set(approx_neighbors)

        n_overlap = len(true_set.intersection(approx_set))
        recall = n_overlap / k_eff

        per_query_rows.append(
            {
                "query_id": query_id,
                "top_k": k_eff,
                "n_overlap": n_overlap,
                "recall_at_k": recall,
                "true_neighbors": ";".join(true_neighbors),
                "approx_neighbors": ";".join(approx_neighbors),
            }
        )

    per_query_df = pd.DataFrame(per_query_rows)

    mean_recall = float(per_query_df["recall_at_k"].mean())
    sd_recall_across_queries = float(per_query_df["recall_at_k"].std(ddof=1))
    median_recall = float(per_query_df["recall_at_k"].median())
    min_recall = float(per_query_df["recall_at_k"].min())
    max_recall = float(per_query_df["recall_at_k"].max())

    return {
        "mean_recall_at_k": mean_recall,
        "sd_recall_across_queries": sd_recall_across_queries,
        "median_recall_at_k": median_recall,
        "min_recall_at_k": min_recall,
        "max_recall_at_k": max_recall,
        "n_queries": len(query_ids),
        "n_database": len(db_ids),
        "top_k": k_eff,
        "per_query_df": per_query_df,
    }


def summarize_replicates(replicate_results_df, method):
    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    summary_df = replicate_results_df.groupby("sketch_size", as_index=False).agg(
        recall_mean=("mean_recall_at_k", "mean"),
        recall_sd=("mean_recall_at_k", "std"),
        recall_se=("mean_recall_at_k", se),
        recall_min=("mean_recall_at_k", "min"),
        recall_max=("mean_recall_at_k", "max"),
        recall_median=("mean_recall_at_k", "median"),
        query_recall_sd_mean=("sd_recall_across_queries", "mean"),
        query_recall_sd_sd=("sd_recall_across_queries", "std"),
        n_queries=("n_queries", "first"),
        n_database=("n_database", "first"),
        top_k=("top_k", "first"),
        n_replicates=("mean_recall_at_k", "count"),
    )

    fill_zero_cols = [
        "recall_sd",
        "recall_se",
        "query_recall_sd_sd",
    ]

    for col in fill_zero_cols:
        summary_df[col] = summary_df[col].fillna(0)

    summary_df["method"] = method

    return summary_df


def plot_recall(summary_df, output_dir, method):
    output_dir = Path(output_dir)

    pdf_plot = output_dir / f"nn_recall_top{TOP_K}_{method}_mean_sd_vs_sketch_size.pdf"
    png_plot = output_dir / f"nn_recall_top{TOP_K}_{method}_mean_sd_vs_sketch_size.png"

    summary_df = summary_df.copy()
    summary_df["sketch_size"] = summary_df["sketch_size"].astype(int)
    summary_df = summary_df.sort_values("sketch_size")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["recall_mean"],
        yerr=summary_df["recall_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label=f"DartUniFrac {method.upper()} mean ± SD",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel(f"Top-{TOP_K} nearest-neighbor recall")
    ax.set_title(f"{method.upper()} top-{TOP_K} recall vs sketch size")

    ax.set_ylim(0, 1.02)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(pdf_plot, bbox_inches="tight")
    fig.savefig(png_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {pdf_plot}")
    print(f"Saved plot: {png_plot}")


def save_split_map(split_map, output_dir):
    rows = []

    for split_index, split_info in split_map.items():
        query_ids = split_info["query_ids"]
        db_ids = split_info["db_ids"]

        for sid in query_ids:
            rows.append(
                {
                    "split_index": split_index,
                    "sample_id": sid,
                    "role": "query",
                }
            )

        for sid in db_ids:
            rows.append(
                {
                    "split_index": split_index,
                    "sample_id": sid,
                    "role": "database",
                }
            )

    split_df = pd.DataFrame(rows)
    split_file = Path(output_dir) / "query_database_splits.tsv"
    split_df.to_csv(split_file, sep="\t", index=False)

    print(f"Saved query/database split map: {split_file}")


# Main workflow


def main():
    args = parse_args()

    method = args.method
    rerun_dartunifrac = not args.reuse

    output_dir = Path(args.output_dir or default_output_dir(method))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nReading truth distance matrix: {TRUTH_DISTANCE_FILE}")
    truth_df = read_distance_matrix_as_df(TRUTH_DISTANCE_FILE)

    sample_ids = truth_df.index.tolist()

    n_samples = len(sample_ids)
    n_query = max(1, min(int(round(QUERY_FRACTION * n_samples)), n_samples - 1))

    print(f"Number of samples in truth matrix: {n_samples}")
    print(f"Query fraction: {QUERY_FRACTION}")
    print(f"Queries per split: {n_query}")
    print(f"Database samples per split: {n_samples - n_query}")
    print(f"Top-k recall: {TOP_K}")
    print(f"Number of splits / DartUniFrac seeds: {len(DARTUNIFRAC_SEEDS)}")

    # Same split seeds as DartUniFrac seeds.
    # This makes each replicate paired across sketch sizes.
    split_map = {}

    for split_index, split_seed in enumerate(DARTUNIFRAC_SEEDS, start=1):
        query_ids, db_ids = make_query_database_split(
            sample_ids=sample_ids,
            split_seed=split_seed,
            query_fraction=QUERY_FRACTION,
        )

        split_map[split_index] = {
            "split_seed": split_seed,
            "query_ids": query_ids,
            "db_ids": db_ids,
        }

    save_split_map(split_map, output_dir)

    replicate_rows = []
    per_query_all = []

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

            approx_df = read_distance_matrix_as_df(dm_file)

            truth_aligned_df, approx_aligned_df = align_distance_matrices(
                truth_df=truth_df,
                approx_df=approx_df,
            )

            aligned_ids = truth_aligned_df.index.tolist()

            # Reuse the same split, but restrict to IDs present in both matrices.
            split_info = split_map[replicate_index]
            query_ids = [sid for sid in split_info["query_ids"] if sid in aligned_ids]
            db_ids = [sid for sid in split_info["db_ids"] if sid in aligned_ids]

            if len(query_ids) == 0:
                raise ValueError(
                    f"No query IDs remain after alignment for sketch size {sketch_size}, "
                    f"replicate {replicate_index}."
                )

            if len(db_ids) < TOP_K:
                raise ValueError(
                    f"Database has fewer than TOP_K samples after alignment: "
                    f"n_database={len(db_ids)}, TOP_K={TOP_K}"
                )

            recall_result = compute_recall_at_k(
                truth_df=truth_aligned_df,
                approx_df=approx_aligned_df,
                query_ids=query_ids,
                db_ids=db_ids,
                k=TOP_K,
            )

            replicate_row = {
                "sketch_size": sketch_size,
                "method": method,
                "replicate": replicate_index,
                "dartunifrac_seed": dart_seed,
                "split_seed": split_info["split_seed"],
                "query_fraction": QUERY_FRACTION,
                "distance_matrix": str(dm_file),
                "mean_recall_at_k": recall_result["mean_recall_at_k"],
                "sd_recall_across_queries": recall_result["sd_recall_across_queries"],
                "median_recall_at_k": recall_result["median_recall_at_k"],
                "min_recall_at_k": recall_result["min_recall_at_k"],
                "max_recall_at_k": recall_result["max_recall_at_k"],
                "n_queries": recall_result["n_queries"],
                "n_database": recall_result["n_database"],
                "top_k": recall_result["top_k"],
            }

            replicate_rows.append(replicate_row)

            per_query_df = recall_result["per_query_df"].copy()
            per_query_df.insert(0, "sketch_size", sketch_size)
            per_query_df.insert(1, "method", method)
            per_query_df.insert(2, "replicate", replicate_index)
            per_query_df.insert(3, "dartunifrac_seed", dart_seed)
            per_query_df.insert(4, "split_seed", split_info["split_seed"])
            per_query_df.insert(5, "distance_matrix", str(dm_file))

            per_query_all.append(per_query_df)

            print(
                f"Method {method}, "
                f"sketch size {sketch_size}, "
                f"replicate {replicate_index}, "
                f"seed {dart_seed}: "
                f"top-{TOP_K} recall = {recall_result['mean_recall_at_k']:.6f}, "
                f"query SD = {recall_result['sd_recall_across_queries']:.6f}, "
                f"n_queries = {recall_result['n_queries']}, "
                f"n_database = {recall_result['n_database']}"
            )

    replicate_results_df = pd.DataFrame(replicate_rows)

    replicate_results_file = output_dir / f"nn_recall_top{TOP_K}_replicates.tsv"
    replicate_results_df.to_csv(replicate_results_file, sep="\t", index=False)

    print(f"\nSaved replicate-level recall results: {replicate_results_file}")

    per_query_results_df = pd.concat(per_query_all, ignore_index=True)

    per_query_results_file = output_dir / f"nn_recall_top{TOP_K}_per_query.tsv"
    per_query_results_df.to_csv(per_query_results_file, sep="\t", index=False)

    print(f"Saved per-query recall results: {per_query_results_file}")

    summary_df = summarize_replicates(
        replicate_results_df=replicate_results_df,
        method=method,
    )

    summary_file = output_dir / f"nn_recall_top{TOP_K}_mean_sd.tsv"
    summary_df.to_csv(summary_file, sep="\t", index=False)

    print(f"Saved mean/SD recall summary: {summary_file}")

    plot_recall(
        summary_df=summary_df,
        output_dir=output_dir,
        method=method,
    )

    print("\nRecall summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
