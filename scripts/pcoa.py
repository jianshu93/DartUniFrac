import sys
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa

# ── 1. read the distance matrix ────────────────────────────────────────────────
DIST_FILE = "GWMC_test_ers.tsv"
dist_df = pd.read_csv(DIST_FILE, sep="\t", index_col=0)

# ── 2. sanity checks ───────────────────────────────────────────────────────────
if dist_df.shape[0] != dist_df.shape[1]:
    sys.exit("Error: distance matrix must be square!")

if not np.allclose(dist_df, dist_df.T, atol=1e-12):
    print("Warning: input matrix is not perfectly symmetric.")
if not np.allclose(np.diag(dist_df), 0.0, atol=1e-12):
    print("Warning: diagonal values are not all zero.")

# ── 3. convert to a scikit-bio DistanceMatrix object ───────────────────────────
dm = DistanceMatrix(dist_df.values, ids=dist_df.index)

# ── 4. run PCoA ────────────────────────────────────────────────────────────────
ordination = pcoa(dm)                       # returns an OrdinationResults object
coords_df  = ordination.samples.iloc[:, :2]  # first two axes only
coords_df.columns = ["PCo1", "PCo2"]

# ── 5. report variance explained ──────────────────────────────────────────────
var_exp = ordination.proportion_explained.iloc[:2] * 100
print(f"\nPCo1 explains {var_exp.iloc[0]:.2f}% variance")
print(f"PCo2 explains {var_exp.iloc[1]:.2f}% variance\n")
print(coords_df.head(), "\n")

# ── 6. save results ───────────────────────────────────────────────────────────
OUT_FILE = "GWMC_ers_pco.tsv"
coords_df.to_csv(OUT_FILE, sep="\t")
print(f"Full coordinates written to: {OUT_FILE}")
