#!/usr/bin/env python3
import argparse
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Font/theme per your snippet ----
mpl.rcParams.update({
    "font.family"     : "sans-serif",
    "font.sans-serif" : ["Helvetica"],   # system fallback handled automatically
    "font.size"       : 20,              # <--- main control for larger text
    "axes.titlesize"  : 20,              # axes title size
    "axes.labelsize"  : 20,              # x/y label size
    "xtick.labelsize" : 20,              # x tick label size
    "ytick.labelsize" : 20,              # y tick label size
    "legend.fontsize" : 20,
    "text.color"      : "black",
    "axes.labelcolor" : "black",
    "axes.edgecolor"  : "black",
    "xtick.color"     : "black",
    "ytick.color"     : "black",
    "axes.facecolor"  : "white",
    "figure.facecolor": "white",
    "axes.grid"       : False,
    "grid.color"      : "0.7",
    "grid.linestyle"  : "--",
    "grid.linewidth"  : 0.1,
})

def load_bench_df(path: str | None) -> pd.DataFrame:
    expected = ["samples", "DartUniFrac", "unifrac-binaries", "Striped_UniFrac"]
    if path and Path(path).exists():
        # auto-detect delimiter (tabs, commas, spaces)
        df = pd.read_csv(path, sep=None, engine="python")
    else:
        raise FileNotFoundError(f"Input file not found: {path}")

    df.columns = [c.strip() for c in df.columns]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"Input table missing required columns: {missing}")
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=expected).reset_index(drop=True)

def plot_time_on_x(df: pd.DataFrame, out_pdf: str) -> None:
    from matplotlib.ticker import ScalarFormatter

    methods = ["DartUniFrac", "unifrac-binaries", "Striped_UniFrac"]
    markers = ["o", "s", "^"]

    # Colors for the three categories
    time_color   = "#1f77b4"  # blue
    mem_color    = "#d62728"  # red
    striped_color = "#2ca02c" # green

    color_map = {
        "DartUniFrac": time_color,
        "unifrac-binaries": mem_color,
        "Striped_UniFrac": striped_color,
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    # x = samples (linear, scientific labels), y = time (log)
    for method, mk in zip(methods, markers):
        x = df["samples"].to_numpy()
        y = df[method].to_numpy()
        order = np.argsort(y)
        ax.plot(
            x[order],
            y[order],
            marker=mk,
            linewidth=1.8,
            markersize=8,
            label=method,
            color=color_map.get(method, "#000000"),
        )

    # y is log-scale time
    ax.set_yscale("log")

    # scientific notation on x-axis
    sci = ScalarFormatter(useMathText=True)
    sci.set_scientific(True)
    sci.set_powerlimits((0, 0))  # always scientific
    ax.xaxis.set_major_formatter(sci)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)

    ax.set_xlabel("Number of Samples")
    ax.set_ylabel("Time (s)")
    ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(
        description="Plot: samples on X, time on Y (log); one curve per method."
    )
    ap.add_argument(
        "-i", "--input", type=str, default=None,
        help="Path to TSV/CSV with columns: samples, DartUniFrac, unifrac-binaries, Striped_UniFrac"
    )
    ap.add_argument(
        "-o", "--output", type=str, default="dartunifrac_benchmark_time_x.pdf",
        help="Output PDF filename"
    )
    args = ap.parse_args()

    df = load_bench_df(args.input)
    plot_time_on_x(df, args.output)

if __name__ == "__main__":
    main()