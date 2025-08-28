#!/usr/bin/env python3
import sys
import argparse
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Helvetica']

def main():
    p = argparse.ArgumentParser(description='Plot DartMH vs ERS runtime vs sparsity for different sketch sizes.')
    p.add_argument('data', nargs='?', help='Input CSV/TSV; if omitted, read from stdin.')
    p.add_argument('--out', default='dartmh_ers_runtime.png', help='Output PNG filename.')
    args = p.parse_args()

    # Read data
    if args.data:
        df = pd.read_csv(args.data, sep=None, engine='python')
    else:
        df = pd.read_csv(sys.stdin, sep=None, engine='python')

    # Normalize and verify columns
    df = df.rename(columns={c: c.strip() for c in df.columns})
    required = {'Sparsity', 'sketch', 'DartMH', 'ERS'}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {missing}")

    # Types
    df['Sparsity'] = df['Sparsity'].astype(float)
    df['sketch']   = df['sketch'].astype(int)
    df['DartMH']   = df['DartMH'].astype(float)
    df['ERS']      = df['ERS'].astype(float)

    ks = sorted(df['sketch'].unique())
    cmap = plt.get_cmap('tab10')  # palette for k colors

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot lines: same color per k, different marker per algorithm
    for i, k in enumerate(ks):
        color = cmap(i % 10)
        sub = df[df['sketch'] == k].sort_values('Sparsity')

        ax.plot(sub['Sparsity'], sub['DartMH'],
                marker='o', linestyle='-', color=color, linewidth=1.8,
                label=None)
        ax.plot(sub['Sparsity'], sub['ERS'],
                marker='^', linestyle='-', color=color, linewidth=1.8,
                label=None)

    ax.set_xscale('log')
    ax.set_xlabel('Sparsity (fraction nonzeros)')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('DartMH vs ERS — runtime vs sparsity by sketch size')

    # Build legends: shapes for algorithms, colors for k
    alg_handles = [
        Line2D([0], [0], color='black', marker='o', linestyle='-', label='DartMH'),
        Line2D([0], [0], color='black', marker='^', linestyle='-', label='ERS'),
    ]
    k_handles = [Line2D([0], [0], color=cmap(i % 10), lw=2, label=f'k={k}') for i, k in enumerate(ks)]

    leg1 = ax.legend(handles=alg_handles, title='Algorithm', loc='upper left', fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=k_handles, title='Sketch size', ncol=1, fontsize=8)

    fig.tight_layout()
    fig.savefig(args.out, dpi=600)

    if sys.stdout.isatty():
        plt.show()

if __name__ == '__main__':
    main()