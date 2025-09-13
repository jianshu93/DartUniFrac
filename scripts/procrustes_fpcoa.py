#!/usr/bin/env python3
"""
Compare a custom (approximate) PCoA against a scikit-bio PCoA from the same distance matrix
using a Procrustes test (SciPy). Supports optional 2D/3D plots and permutation p-value.

Requirements:
  pip install numpy pandas scipy scikit-bio matplotlib

Example:
  python fast_vs_skbio_procrustes.py \
      --dist GWMC_test_ers.tsv \
      --approx-pcoa data/pcoa.tsv \
      --axes 3 \
      --permutations 999 \
      --out out/fast_vs_skbio \
      --plot --plot3d
"""

import argparse
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes
import matplotlib as mpl

mpl.rcParams.update({
    # font
    "font.family"     : "sans-serif",
    "font.sans-serif" : ["Helvetica"],   # fall-back handled automatically
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


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def read_distance_matrix(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    if df.shape[0] != df.shape[1]:
        sys.exit(f"[ERROR] {tsv_path} is not square (shape={df.shape}).")
    if not np.allclose(df.values, df.values.T, atol=1e-8):
        print(f"[WARN] {tsv_path} not perfectly symmetric (tol=1e-8).", file=sys.stderr)
    if not np.allclose(np.diag(df.values), 0.0, atol=1e-12):
        print(f"[WARN] Diagonal of {tsv_path} not all zeros.", file=sys.stderr)
    return df


def parse_approx_pcoa_table(pcoa_tsv_path: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Parse your approximate PCoA TSV that contains:
      - Block 1: header 'PC1..' and per-sample coordinates
      - (optional blank line)
      - Block 2: header 'PC1..' and a 'proportion_explained' row

    Returns (coords_df, prop_explained or None)
       coords_df: index = sample IDs, columns=['PC1','PC2',...], float
    """
    text = Path(pcoa_tsv_path).read_text()
    # Split on one or more blank lines to isolate the first table block
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not parts:
        sys.exit(f"[ERROR] Could not parse {pcoa_tsv_path}: empty file?")
    # Block 1 = coordinates
    coords_df = pd.read_csv(io.StringIO(parts[0]), sep="\t", index_col=0)
    # Clean column names just in case
    coords_df.columns = [c.strip() for c in coords_df.columns]
    # Convert to numeric (robust to stray strings)
    coords_df = coords_df.apply(pd.to_numeric, errors="coerce")
    if coords_df.isna().any().any():
        nbad = int(coords_df.isna().sum().sum())
        print(f"[WARN] {nbad} NA values in coordinates after parsing; rows with NA will be dropped.",
              file=sys.stderr)
        coords_df = coords_df.dropna(axis=0, how="any")

    # Block 2 (optional) = proportion_explained row
    prop = None
    if len(parts) >= 2:
        # Expect a header row + a 'proportion_explained' data row
        pe_df = pd.read_csv(io.StringIO(parts[1]), sep="\t", index_col=0)
        if "proportion_explained" in pe_df.index:
            prop = pe_df.loc["proportion_explained"].astype(float)
            prop.index = [i.strip() for i in prop.index]
    return coords_df, prop


# ──────────────────────────────────────────────────────────────────────────────
# PCoA + Procrustes
# ──────────────────────────────────────────────────────────────────────────────

def run_skbio_pcoa(dm_df: pd.DataFrame):
    dm = DistanceMatrix(dm_df.values.astype(float), ids=list(dm_df.index))
    ord_res = pcoa(dm)
    eigvals = ord_res.eigvals
    n_neg = int((eigvals < 0).sum())
    if n_neg > 0:
        print(f"[INFO] PCoA has {n_neg} negative eigenvalue(s); distance may be non-Euclidean.",
              file=sys.stderr)
    return ord_res


def match_and_slice_axes(approx_df: pd.DataFrame, skbio_ord, k: int | None):
    ids_approx = list(approx_df.index)
    ids_skbio  = list(skbio_ord.samples.index)
    common = [i for i in ids_approx if i in ids_skbio]
    if len(common) < 3:
        sys.exit("[ERROR] Fewer than 3 shared samples; Procrustes not meaningful.")

    # Align on shared IDs and ensure consistent ordering
    A_full = approx_df.loc[common].to_numpy(dtype=float)
    B_full = skbio_ord.samples.loc[common].to_numpy(dtype=float)

    max_k = min(A_full.shape[1], B_full.shape[1])
    k_use = min(3, max_k) if k is None else min(k, max_k)
    if k_use < 2:
        print(f"[INFO] Only {k_use} axis available; Procrustes will use {k_use}.", file=sys.stderr)

    A = A_full[:, :k_use]
    B = B_full[:, :k_use]
    return A, B, common, k_use


def procrustes_with_permutations(A, B, n_perm=0, seed=0):
    """
    SciPy Procrustes (centers/scales internally) + optional permutation p-value.
    Returns dict with disparity (M^2), pval, aligned matrices, and per-sample residuals.
    """
    mtx1, mtx2, disparity = procrustes(A, B)
    resid = np.linalg.norm(mtx1 - mtx2, axis=1)

    pval = np.nan
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        count = 0
        for _ in range(n_perm):
            perm = rng.permutation(mtx2.shape[0])
            _, _, d = procrustes(A, B[perm, :])
            if d <= disparity + 1e-15:
                count += 1
        pval = (count + 1) / (n_perm + 1)  # conservative

    return {"disparity": float(disparity), "pval": float(pval) if not np.isnan(pval) else np.nan,
            "mtx1": mtx1, "mtx2": mtx2, "resid": resid}


def orthogonal_map(A, B):
    """
    Explicit orthogonal map after centering & scaling (like scipy.spatial.procrustes):
      Let A_std ≈ B_std @ R
    Returns (meanA, scaleA, meanB, scaleB, R)
    """
    Ac = A - A.mean(axis=0, keepdims=True)
    Bc = B - B.mean(axis=0, keepdims=True)
    na = np.linalg.norm(Ac)
    nb = np.linalg.norm(Bc)
    if na == 0 or nb == 0:
        raise ValueError("Degenerate configuration with zero norm.")
    A_std = Ac / na
    B_std = Bc / nb
    R, _ = orthogonal_procrustes(B_std, A_std)
    return A.mean(axis=0), na, B.mean(axis=0), nb, R


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot2d(m1, m2, ids, out_png):
    if m1.shape[1] < 2:
        print("[INFO] 2D plot skipped; need ≥2 axes.", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(7, 6), dpi=600)
    ax.scatter(m1[:, 0], m1[:, 1], label="fpcoa (aligned)", s=30)
    ax.scatter(m2[:, 0], m2[:, 1], label="scikit-bio (aligned)", s=30, marker="x")
    for i in range(m1.shape[0]):
        ax.plot([m1[i, 0], m2[i, 0]], [m1[i, 1], m2[i, 1]], linewidth=0.6)
    ax.set_xlabel("Axis 1 (aligned)")
    ax.set_ylabel("Axis 2 (aligned)")
    ax.set_title("Procrustes-aligned PCoA (2D)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def _set_equal_3d(ax):
    lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    rng = np.abs(lims[:, 1] - lims[:, 0])
    ctr = lims.mean(axis=1)
    rad = 0.5 * max(rng)
    ax.set_xlim3d([ctr[0] - rad, ctr[0] + rad])
    ax.set_ylim3d([ctr[1] - rad, ctr[1] + rad])
    ax.set_zlim3d([ctr[2] - rad, ctr[2] + rad])


def plot3d(m1, m2, ids, out_png):
    if m1.shape[1] < 3:
        print("[INFO] 3D plot skipped; need ≥3 axes.", file=sys.stderr)
        return
    fig = plt.figure(figsize=(8, 7), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(m1[:, 0], m1[:, 1], m1[:, 2], label="Approx (aligned)", s=25)
    ax.scatter(m2[:, 0], m2[:, 1], m2[:, 2], label="scikit-bio (aligned)", s=25, marker="x")
    for i in range(m1.shape[0]):
        ax.plot([m1[i, 0], m2[i, 0]],
                [m1[i, 1], m2[i, 1]],
                [m1[i, 2], m2[i, 2]], linewidth=0.5)
    ax.set_xlabel("Axis 1 (aligned)")
    ax.set_ylabel("Axis 2 (aligned)")
    ax.set_zlabel("Axis 3 (aligned)")
    ax.set_title("Procrustes-aligned PCoA (3D)")
    ax.legend(loc="best")
    _set_equal_3d(ax)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Procrustes test: custom PCoA vs scikit-bio PCoA from the same distance.")
    ap.add_argument("--dist", required=True, help="TSV square distance matrix used by both methods")
    ap.add_argument("--approx-pcoa", required=True, help="Your approximate PCoA TSV (coords + optional proportion_explained)")
    ap.add_argument("--axes", type=int, default=None, help="Number of axes to compare (default: min(3, available))")
    ap.add_argument("--permutations", type=int, default=0, help="Permutation test count (e.g., 999). 0 disables.")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for permutations")
    ap.add_argument("--out", required=True, help="Output prefix")
    ap.add_argument("--plot", action="store_true", help="Save 2D plot of aligned coordinates")
    ap.add_argument("--plot3d", action="store_true", help="Save 3D plot of aligned coordinates")
    args = ap.parse_args()

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    # 1) Read inputs
    dm = read_distance_matrix(args.dist)
    approx_coords, approx_prop = parse_approx_pcoa_table(args.approx_pcoa)

    # 2) Run scikit-bio PCoA on the same distance
    ord_skbio = run_skbio_pcoa(dm)

    # 3) Match IDs and slice axes
    A, B, shared_ids, k_use = match_and_slice_axes(approx_coords, ord_skbio, args.axes)
    print(f"[INFO] Shared samples: {len(shared_ids)}  |  Axes used: {k_use}")

    # Explain variance context
    def var_str_from_series(s: pd.Series | None, k: int):
        if s is None:
            return "N/A (not provided)"
        vals = [float(s.get(f"PC{i+1}", np.nan)) * 100 for i in range(k)]
        return ", ".join([f"Axis{i+1}: {v:.2f}%" for i, v in enumerate(vals)])
    def var_str_skbio(ord_res, k):
        pe = ord_res.proportion_explained.iloc[:k] * 100
        return ", ".join([f"Axis{i+1}: {pe.iloc[i]:.2f}%" for i in range(len(pe))])

    print(f"[INFO] Approx variance explained (first {k_use}): {var_str_from_series(approx_prop, k_use)}")
    print(f"[INFO] scikit-bio variance explained (first {k_use}): {var_str_skbio(ord_skbio, k_use)}")

    # 4) Procrustes test (+ permutations)
    res = procrustes_with_permutations(A, B, n_perm=args.permutations, seed=args.seed)
    disparity = res["disparity"]
    r2 = 1.0 - disparity  # heuristic with SciPy normalization

    print("\n=== Procrustes ===")
    print(f"Disparity (M^2): {disparity:.6f}")
    print(f"Procrustes R^2 ≈ 1 - M^2: {r2:.6f}")
    if args.permutations > 0:
        print(f"Permutation p-value (n={args.permutations}): {res['pval']:.6f}")
    print("==================\n")

    # 5) Save aligned coordinates & residuals
    m1_df = pd.DataFrame(res["mtx1"], index=shared_ids, columns=[f"A{i+1}" for i in range(res["mtx1"].shape[1])])
    m2_df = pd.DataFrame(res["mtx2"], index=shared_ids, columns=[f"A{i+1}" for i in range(res["mtx2"].shape[1])])
    resid_df = pd.DataFrame({"sample_id": shared_ids, "residual_distance": res["resid"]})
    m1_df.to_csv(f"{outp}.aligned_approx.tsv", sep="\t")
    m2_df.to_csv(f"{outp}.aligned_skbio.tsv", sep="\t")
    resid_df.to_csv(f"{outp}.residuals.tsv", sep="\t", index=False)

    # 6) Export explicit orthogonal map (optional debugging/repro)
    try:
        meanA, scaleA, meanB, scaleB, R = orthogonal_map(A, B)
        np.savetxt(f"{outp}.R.txt", R, fmt="%.8f")
        np.savetxt(f"{outp}.mean_approx.txt", meanA[None, :], fmt="%.8f")
        np.savetxt(f"{outp}.mean_skbio.txt", meanB[None, :], fmt="%.8f")
        with open(f"{outp}.scales.txt", "w") as fh:
            fh.write(f"scale_approx={scaleA:.8f}\nscale_skbio={scaleB:.8f}\n")
    except Exception as e:
        print(f"[WARN] Could not compute/export orthogonal map: {e}", file=sys.stderr)

    # 7) Plots
    if args.plot:
        png2d = f"{outp}.aligned_2d.png"
        plot2d(res["mtx1"], res["mtx2"], shared_ids, png2d)
        print(f"[INFO] Saved 2D plot: {png2d}")
    if args.plot3d:
        png3d = f"{outp}.aligned_3d.png"
        plot3d(res["mtx1"], res["mtx2"], shared_ids, png3d)
        print(f"[INFO] Saved 3D plot: {png3d}")

    # 8) Summary
    with open(f"{outp}.summary.txt", "w") as fh:
        fh.write(f"Shared samples: {len(shared_ids)}\n")
        fh.write(f"Axes used: {k_use}\n")
        fh.write(f"Disparity (M^2): {disparity:.6f}\n")
        fh.write(f"Procrustes R^2 ≈ 1 - M^2: {r2:.6f}\n")
        if args.permutations > 0:
            fh.write(f"Permutation p-value (n={args.permutations}): {res['pval']:.6f}\n")

    print(f"[DONE] Outputs written with prefix: {outp}")


if __name__ == "__main__":
    main()
