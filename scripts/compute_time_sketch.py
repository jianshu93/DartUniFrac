#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
from datetime import datetime
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

TREE_FILE = "./2024.09.phylogeny.md5.nwk"
BIOM_FILE = "./AGP.raw.rarefied3000.biom"

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

# 5 replicates per sketch size
DARTUNIFRAC_SEEDS = [
    1337,
    1999,
    314159,
    271828,
    8675309,
]

DARTUNIFRAC_EXE = "dartunifrac"

# If False, existing log files are reused when present.
RERUN_DARTUNIFRAC = True

# Argument parsing


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark DartUniFrac compute time across sketch sizes."
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
        help="Output directory. If omitted, uses AGP method-specific default.",
    )

    parser.add_argument(
        "--reuse",
        action="store_true",
        help="Reuse existing output/log files instead of rerunning DartUniFrac.",
    )

    return parser.parse_args()


def default_output_dir(method):
    if method == "dmh":
        return "./agp_dartunifrac_dmh_timing"
    if method == "ers":
        return "./agp_dartunifrac_ers_timing"

    raise ValueError(f"Unsupported method: {method}")


# Log parsing helpers

TIMESTAMP_RE = re.compile(
    r"^\[(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)"
)


def parse_rust_timestamp(line):
    """
    Parse timestamps like:
    [2026-05-01T07:46:45Z INFO  dartunifrac] ...
    """
    match = TIMESTAMP_RE.search(line)

    if match is None:
        return None

    ts = match.group("timestamp")

    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"

    return datetime.fromisoformat(ts)


def parse_compute_time_from_log_text(log_text):
    """
    Compute time is defined as the timestamp difference from the line containing
    'nodes =' to the line containing 'pairwise distances'.

    Example start line:
    [2026-05-01T07:46:45Z INFO  dartunifrac] nodes = ...

    Example end line:
    [2026-05-01T07:46:45Z INFO  dartunifrac] pairwise distances in 90 ms
    """

    start_time = None
    end_time = None

    start_line = None
    end_line = None

    for line in log_text.splitlines():
        if "nodes =" in line and start_time is None:
            start_time = parse_rust_timestamp(line)
            start_line = line

        if "pairwise distances" in line:
            end_time = parse_rust_timestamp(line)
            end_line = line

    if start_time is None:
        raise ValueError(
            "Could not find start timing line containing 'nodes =' in DartUniFrac log."
        )

    if end_time is None:
        raise ValueError(
            "Could not find end timing line containing 'pairwise distances' in DartUniFrac log."
        )

    compute_seconds = (end_time - start_time).total_seconds()

    # Optional parse of reported pairwise distance time, e.g. "pairwise distances in 90 ms"
    pairwise_ms = np.nan
    ms_match = re.search(r"pairwise distances in\s+([0-9.]+)\s*ms", end_line)
    if ms_match is not None:
        pairwise_ms = float(ms_match.group(1))

    return {
        "compute_time_seconds": compute_seconds,
        "compute_start_timestamp": start_time.isoformat(),
        "compute_end_timestamp": end_time.isoformat(),
        "compute_start_line": start_line,
        "compute_end_line": end_line,
        "reported_pairwise_ms": pairwise_ms,
    }


def parse_compute_time_from_log_file(log_file):
    log_text = Path(log_file).read_text()
    return parse_compute_time_from_log_text(log_text)


# DartUniFrac runner


def run_dartunifrac_with_log(method, sketch_size, dart_seed, output_path, log_path):
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

    env = os.environ.copy()
    env["RUST_LOG"] = "info"

    print(
        f"\nRunning DartUniFrac method {method} "
        f"with sketch size {sketch_size}, seed {dart_seed}"
    )
    print("RUST_LOG=info " + " ".join(cmd))

    completed = subprocess.run(
        cmd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    log_text = completed.stdout + "\n" + completed.stderr

    Path(log_path).write_text(log_text)

    timing = parse_compute_time_from_log_text(log_text)

    return timing


# Summary and plotting


def summarize_timing_replicates(timing_df, method):
    def se(x):
        return x.std(ddof=1) / np.sqrt(len(x))

    def cv(x):
        mean = x.mean()
        if mean == 0:
            return np.nan
        return x.std(ddof=1) / mean

    summary_df = timing_df.groupby("sketch_size", as_index=False).agg(
        compute_time_mean=("compute_time_seconds", "mean"),
        compute_time_sd=("compute_time_seconds", "std"),
        compute_time_se=("compute_time_seconds", se),
        compute_time_min=("compute_time_seconds", "min"),
        compute_time_max=("compute_time_seconds", "max"),
        compute_time_median=("compute_time_seconds", "median"),
        compute_time_cv=("compute_time_seconds", cv),
        reported_pairwise_ms_mean=("reported_pairwise_ms", "mean"),
        reported_pairwise_ms_sd=("reported_pairwise_ms", "std"),
        n_replicates=("compute_time_seconds", "count"),
    )

    fill_zero_cols = [
        "compute_time_sd",
        "compute_time_se",
        "compute_time_cv",
        "reported_pairwise_ms_sd",
    ]

    for col in fill_zero_cols:
        summary_df[col] = summary_df[col].fillna(0)

    summary_df["method"] = method

    return summary_df


def plot_timing(summary_df, output_dir, method):
    output_dir = Path(output_dir)

    pdf_plot = output_dir / f"agp_compute_time_{method}_mean_sd_vs_sketch_size.pdf"
    png_plot = output_dir / f"agp_compute_time_{method}_mean_sd_vs_sketch_size.png"

    summary_df = summary_df.copy()
    summary_df["sketch_size"] = summary_df["sketch_size"].astype(int)
    summary_df = summary_df.sort_values("sketch_size")

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.errorbar(
        summary_df["sketch_size"],
        summary_df["compute_time_mean"],
        yerr=summary_df["compute_time_sd"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        capsize=5,
        color="black",
        label=f"DartUniFrac {method.upper()} mean ± SD",
    )

    ax.set_xlabel("Sketch size")
    ax.set_ylabel("Compute time, seconds")
    ax.set_title(f"AGP {method.upper()} compute time vs sketch size")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(pdf_plot, bbox_inches="tight")
    fig.savefig(png_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot: {pdf_plot}")
    print(f"Saved plot: {png_plot}")


# Main workflow


def main():
    args = parse_args()

    method = args.method
    rerun_dartunifrac = not args.reuse

    output_dir = Path(args.output_dir or default_output_dir(method))
    output_dir.mkdir(parents=True, exist_ok=True)

    timing_rows = []

    for sketch_size in SKETCH_SIZES:
        for replicate_index, dart_seed in enumerate(DARTUNIFRAC_SEEDS, start=1):
            dm_file = output_dir / (
                f"agp_unifrac_weighted_{method}_"
                f"s{sketch_size}_rep{replicate_index}_seed{dart_seed}.tsv"
            )

            log_file = output_dir / (
                f"agp_unifrac_weighted_{method}_"
                f"s{sketch_size}_rep{replicate_index}_seed{dart_seed}.log"
            )

            if rerun_dartunifrac or not log_file.exists() or not dm_file.exists():
                timing = run_dartunifrac_with_log(
                    method=method,
                    sketch_size=sketch_size,
                    dart_seed=dart_seed,
                    output_path=dm_file,
                    log_path=log_file,
                )
            else:
                print(f"\nReusing existing log and distance matrix:")
                print(f"  log: {log_file}")
                print(f"  dm : {dm_file}")

                timing = parse_compute_time_from_log_file(log_file)

            row = {
                "sketch_size": sketch_size,
                "method": method,
                "replicate": replicate_index,
                "dartunifrac_seed": dart_seed,
                "distance_matrix": str(dm_file),
                "log_file": str(log_file),
            }

            row.update(timing)

            timing_rows.append(row)

            print(
                f"Method {method}, "
                f"sketch size {sketch_size}, "
                f"replicate {replicate_index}, "
                f"seed {dart_seed}: "
                f"compute time = {timing['compute_time_seconds']:.6f} s, "
                f"reported pairwise = {timing['reported_pairwise_ms']:.3f} ms"
            )

    timing_df = pd.DataFrame(timing_rows)

    timing_file = output_dir / "agp_compute_time_replicates.tsv"
    timing_df.to_csv(timing_file, sep="\t", index=False)

    print(f"\nSaved replicate-level timing results: {timing_file}")

    summary_df = summarize_timing_replicates(
        timing_df=timing_df,
        method=method,
    )

    summary_file = output_dir / "agp_compute_time_mean_sd.tsv"
    summary_df.to_csv(summary_file, sep="\t", index=False)

    print(f"Saved mean/SD timing summary: {summary_file}")

    plot_timing(
        summary_df=summary_df,
        output_dir=output_dir,
        method=method,
    )

    print("\nTiming summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
