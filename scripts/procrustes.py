#!/usr/bin/env python3
"""
PCoA + Procrustes comparison for two distance matrices.

Requirements:
  pip install numpy pandas scipy scikit-bio matplotlib

Usage:
  python pcoa_procrustes.py \
      --dm1 GWMC_methodA.tsv \
      --dm2 GWMC_methodB.tsv \
      --axes 3 \
      --permutations 999 \
      --out out/gwmc_A_vs_B \
      --plot
"""

import argparse
import sys
from pathlib import Path
import matplotlib as mpl
import numpy as np
import pandas as pd
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt

mpl.rcParams.update({
    # font
    "font.family"     : "sans-serif",
    "font.sans-serif" : ["Helvetica"],   # fall-back handled automatically
    "font.size"       : 20,              # <--- main control for larger text
    "axes.titlesize"  : 20,              # axes title size
    "axes.labelsize"  : 20,              # x/y label size
    "xtick.labelsize" : 20,              # x tick label size
    "ytick.labelsize" : 20,              # y tick label size
    "legend.fontsize" : 20, 
    "text.color"      : "black",
    # axes & ticks
    "axes.labelcolor" : "black",
    "axes.edgecolor"  : "black",
    "xtick.color"     : "black",
    "ytick.color"     : "black",
    "axes.facecolor"  : "white",
    "figure.facecolor": "white",
    # grid (light grey, thin, dashed - similar to ggplot2::theme_bw)
    "axes.grid"       : False,
    "grid.color"      : "0.7",
    "grid.linestyle"  : "--",
    "grid.linewidth"  : 0.1,
})


def read_distance_matrix(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    if df.shape[0] != df.shape[1]:
        sys.exit(f"[ERROR] {tsv_path} is not square (shape={df.shape}).")
    # basic sanity checks
    if not np.allclose(df.values, df.values.T, atol=1e-8):
        print(f"[WARN] {tsv_path} is not perfectly symmetric (tolerance 1e-8).", file=sys.stderr)
    if not np.allclose(np.diag(df.values), 0.0, atol=1e-12):
        print(f"[WARN] Diagonal of {tsv_path} is not all zeros.", file=sys.stderr)
    return df


def run_pcoa(df: pd.DataFrame):
    dm = DistanceMatrix(df.values.astype(float), ids=list(df.index))
    ord_res = pcoa(dm)
    # Negative eigenvalues are possible for non-Euclidean distances
    eigvals = ord_res.eigvals
    n_neg = int((eigvals < 0).sum())
    if n_neg > 0:
        print(f"[INFO] PCoA on {df.shape[0]} samples has {n_neg} negative eigenvalues.", file=sys.stderr)
    return ord_res  # OrdinationResults


def intersect_and_extract(ord1, ord2, k: int):
    """Return matched coordinate matrices (n x k) and shared IDs in the order of ord1."""
    ids1 = list(ord1.samples.index)
    ids2 = list(ord2.samples.index)
    common = [i for i in ids1 if i in ids2]
    if len(common) < 3:
        sys.exit("[ERROR] Less than 3 shared samples between the two matrices; Procrustes not meaningful.")
    if len(common) < len(ids1) or len(common) < len(ids2):
        missing1 = set(ids1) - set(common)
        missing2 = set(ids2) - set(common)
        if missing1:
            print(f"[INFO] Dropping {len(missing1)} samples only in matrix 1.", file=sys.stderr)
        if missing2:
            print(f"[INFO] Dropping {len(missing2)} samples only in matrix 2.", file=sys.stderr)

    # Coordinates for all axes available
    coords1_full = ord1.samples.loc[common].to_numpy(dtype=float)
    coords2_full = ord2.samples.loc[common].to_numpy(dtype=float)

    max_k = min(coords1_full.shape[1], coords2_full.shape[1])
    if k is None:
        k_use = min(3, max_k)  # default to 3D if available
    else:
        k_use = min(k, max_k)
    if k_use < 2:
        print(f"[INFO] Only {k_use} axis available; using {k_use}.", file=sys.stderr)

    coords1 = coords1_full[:, :k_use]
    coords2 = coords2_full[:, :k_use]
    return coords1, coords2, common, k_use


def procrustes_with_perm(coords1, coords2, n_perm=0, seed=0):
    """
    Run SciPy Procrustes and (optionally) a permutation test by shuffling rows of coords2.
    Returns dict with keys: disparity, pval, mtx1, mtx2, resid
    """
    # SciPy procrustes centers & scales both sets internally
    mtx1, mtx2, disparity = procrustes(coords1, coords2)

    # Per-sample residual distances after alignment
    resid = np.linalg.norm(mtx1 - mtx2, axis=1)

    pval = np.nan
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        perm_d = 0
        for _ in range(n_perm):
            perm = rng.permutation(mtx2.shape[0])
            _, _, d = procrustes(coords1, coords2[perm, :])
            if d <= disparity + 1e-15:
                perm_d += 1
        pval = (perm_d + 1) / (n_perm + 1)  # conservative
    return {
        "disparity": float(disparity),
        "pval": float(pval) if not np.isnan(pval) else np.nan,
        "mtx1": mtx1,
        "mtx2": mtx2,
        "resid": resid,
    }


def orthogonal_map(coords1, coords2):
    """
    Compute an explicit orthogonal map (rotation/reflection) that best maps coords2 -> coords1
    after the same centering/scaling as scipy.spatial.procrustes.

    Returns: mean1, scale1, mean2, scale2, R  so that:
      X1_std ≈ X2_std @ R,
    where X*_std = (X* - mean*) / scale*
    """
    # Center & scale as in scipy.spatial.procrustes
    c1 = coords1 - coords1.mean(axis=0, keepdims=True)
    c2 = coords2 - coords2.mean(axis=0, keepdims=True)
    norm1 = np.linalg.norm(c1)
    norm2 = np.linalg.norm(c2)
    if norm1 == 0 or norm2 == 0:
        raise ValueError("Degenerate configuration with zero norm.")
    X1 = c1 / norm1
    X2 = c2 / norm2
    R, _ = orthogonal_procrustes(X2, X1)
    return coords1.mean(axis=0), norm1, coords2.mean(axis=0), norm2, R


def make_plot(mtx1, mtx2, ids, out_png):
    if mtx1.shape[1] < 2:
        print("[INFO] Plot requires at least 2 axes; skipping.", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)
    ax.scatter(mtx1[:, 0], mtx1[:, 1], label="DartuniFrac", s=30, marker="D", alpha=0.5)
    ax.scatter(mtx2[:, 0], mtx2[:, 1], label="UniFrac", s=30, marker="s", alpha=0.5)
    # segments to visualize residuals per sample
    for i in range(mtx1.shape[0]):
        ax.plot([mtx1[i, 0], mtx2[i, 0]], [mtx1[i, 1], mtx2[i, 1]], linewidth=0.6)
    ax.set_xlabel("Axis 1 (aligned)")
    ax.set_ylabel("Axis 2 (aligned)")
    ax.set_title("Procrustes-aligned PCoA coordinates")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Compare two distance matrices via PCoA + Procrustes.")
    ap.add_argument("--dm1", required=True, help="TSV square distance matrix (method 1)")
    ap.add_argument("--dm2", required=True, help="TSV square distance matrix (method 2)")
    ap.add_argument("--axes", type=int, default=None, help="Number of PCoA axes to use (default: min(3, available))")
    ap.add_argument("--permutations", type=int, default=0, help="Permutation test count (e.g., 999). 0 disables.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for permutations")
    ap.add_argument("--out", required=True, help="Output prefix (directory/filename prefix)")
    ap.add_argument("--plot", action="store_true", help="Save a 2D diagnostic plot of aligned coords")
    args = ap.parse_args()

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # Read DMs
    dm1 = read_distance_matrix(args.dm1)
    dm2 = read_distance_matrix(args.dm2)

    # PCoA on both
    ord1 = run_pcoa(dm1)
    ord2 = run_pcoa(dm2)

    # Variance explained (report first few for context)
    def var_str(ord_res, k):
        pe = ord_res.proportion_explained.iloc[:k] * 100
        return ", ".join([f"Axis{i+1}: {pe.iloc[i]:.2f}%" for i in range(len(pe))])

    # Get matched coords and chosen axis count
    coords1, coords2, shared_ids, k_use = intersect_and_extract(ord1, ord2, args.axes)

    print(f"[INFO] Using {len(shared_ids)} shared samples and {k_use} PCoA axis/axes.")
    print(f"[INFO] Method 1 variance explained (first {k_use}): {var_str(ord1, k_use)}")
    print(f"[INFO] Method 2 variance explained (first {k_use}): {var_str(ord2, k_use)}")

    # Procrustes and optional permutation test
    res = procrustes_with_perm(coords1, coords2, n_perm=args.permutations, seed=args.seed)
    disparity = res["disparity"]
    r2 = 1.0 - disparity  # often reported as Procrustes R^2 (heuristic with SciPy's normalization)

    print("\n=== Procrustes results ===")
    print(f"Disparity (M^2): {disparity:.6f}")
    print(f"Procrustes R^2 ≈ 1 - M^2: {r2:.6f}")
    if args.permutations > 0:
        print(f"Permutation test (n={args.permutations}) p-value: {res['pval']:.6f}")
    print("==========================\n")

    # Save aligned coordinates and residuals
    m1_df = pd.DataFrame(res["mtx1"], index=shared_ids, columns=[f"A{i+1}" for i in range(res["mtx1"].shape[1])])
    m2_df = pd.DataFrame(res["mtx2"], index=shared_ids, columns=[f"A{i+1}" for i in range(res["mtx2"].shape[1])])
    resid_df = pd.DataFrame({"sample_id": shared_ids, "residual_distance": res["resid"]})

    m1_df.to_csv(f"{out_prefix}.aligned_method1.tsv", sep="\t")
    m2_df.to_csv(f"{out_prefix}.aligned_method2.tsv", sep="\t")
    resid_df.to_csv(f"{out_prefix}.residuals.tsv", sep="\t", index=False)

    # the explicit orthogonal map for reproducibility/debugging
    try:
        mean1, scale1, mean2, scale2, R = orthogonal_map(coords1, coords2)
        np.savetxt(f"{out_prefix}.R.txt", R, fmt="%.8f")
        np.savetxt(f"{out_prefix}.mean1.txt", mean1[None, :], fmt="%.8f")
        np.savetxt(f"{out_prefix}.mean2.txt", mean2[None, :], fmt="%.8f")
        with open(f"{out_prefix}.scales.txt", "w") as fh:
            fh.write(f"scale1={scale1:.8f}\nscale2={scale2:.8f}\n")
    except Exception as e:
        print(f"[WARN] Could not compute/export explicit orthogonal map: {e}", file=sys.stderr)

    # plot (first two axes)
    if args.plot:
        png = f"{out_prefix}.aligned_plot.png"
        make_plot(res["mtx1"], res["mtx2"], shared_ids, png)
        print(f"[INFO] Saved diagnostic plot to {png}")

    # text summary
    with open(f"{out_prefix}.summary.txt", "w") as fh:
        fh.write(f"Shared samples: {len(shared_ids)}\n")
        fh.write(f"Axes used: {k_use}\n")
        fh.write(f"Disparity (M^2): {disparity:.6f}\n")
        fh.write(f"Procrustes R^2 ≈ 1 - M^2: {r2:.6f}\n")
        if args.permutations > 0:
            fh.write(f"Permutation p-value (n={args.permutations}): {res['pval']:.6f}\n")

    print(f"[DONE] Aligned coords and residuals written with prefix: {out_prefix}")


if __name__ == "__main__":
    main()
