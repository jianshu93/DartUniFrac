import sys
import time
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

# ---------------------------------------------------------------------
# Command-line arguments: input DM, output coordinates
# ---------------------------------------------------------------------
if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <input_dm_tsv> <output_pcoa_tsv>", file=sys.stderr)
    sys.exit(1)

DIST_FILE = sys.argv[1]
OUT_FILE  = sys.argv[2]

t_start = time.time()
print(f"[INFO] Starting PCoA pipeline on {DIST_FILE}")

# ---------------------------------------------------------------------
# 1) Read the distance matrix
# ---------------------------------------------------------------------
t_read_start = time.time()
dist_df = pd.read_csv(DIST_FILE, sep="\t", index_col=0)
t_read_end = time.time()
print(f"[INFO] Loaded distance matrix with shape {dist_df.shape} in "
      f"{t_read_end - t_read_start:.2f} s")

# sanity checks
if dist_df.shape[0] != dist_df.shape[1]:
    sys.exit("Error: distance matrix must be square!")

if not np.allclose(dist_df, dist_df.T, atol=1e-12):
    print("[WARN] Input matrix is not perfectly symmetric.")
if not np.allclose(np.diag(dist_df), 0.0, atol=1e-12):
    print("[WARN] Diagonal values are not all zero.")

# ---------------------------------------------------------------------
# 2) Convert to a scikit-bio DistanceMatrix object 
# ---------------------------------------------------------------------
dm = DistanceMatrix(dist_df.values, ids=dist_df.index)

# ---------------------------------------------------------------------
# 3) Run fast PCoA (randomized SVD) by default
# ---------------------------------------------------------------------
t_pcoa_start = time.time()
try:
    # fast PCoA: randomized SVD, first two dimensions only
    ordination = pcoa(dm, method="fsvd", dimensions=2)
    method_used = "fsvd"
except TypeError:
    # fallback for older scikit-bio without 'method' kwarg
    print("[WARN] pcoa(method='fsvd') not supported, falling back to exact method.")
    ordination = pcoa(dm)
    method_used = "eigh"
t_pcoa_end = time.time()
print(f"[INFO] PCoA (method='{method_used}') completed in "
      f"{t_pcoa_end - t_pcoa_start:.2f} s")

# ---------------------------------------------------------------------
# 4) Extract first two axes (same as before)
# ---------------------------------------------------------------------
coords_df = ordination.samples.iloc[:, :2]
coords_df.columns = ["PCo1", "PCo2"]

# report variance explained (same as before)
var_exp = ordination.proportion_explained.iloc[:2] * 100
print(f"\nPCo1 explains {var_exp.iloc[0]:.2f}% variance")
print(f"PCo2 explains {var_exp.iloc[1]:.2f}% variance\n")
print(coords_df.head(), "\n")

# ---------------------------------------------------------------------
# 5) Save results (same as before, but filename from CLI)
# ---------------------------------------------------------------------
coords_df.to_csv(OUT_FILE, sep="\t")
print(f"[INFO] Full coordinates written to: {OUT_FILE}")

t_end = time.time()
print(f"[INFO] Total wall-clock time: {t_end - t_start:.2f} s")