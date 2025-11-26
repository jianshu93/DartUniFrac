#!/usr/bin/env python3
"""
Jackknifed UniFrac using DartUniFrac + UPGMA or NJ consensus (outside QIIME).

Features
- --auto-min-depth: depth = min library size (keeps all samples)
- Exact without-replacement rarefaction (NumPy hypergeometric)
- Writes replicate BIOMs as **HDF5** so dartunifrac can open them
- Tree method selectable: UPGMA (default) or Neighbor-Joining (--tree-method nj)

Outputs (in --outdir):
  depths.tsv · samples_used.txt · rep_XXX/dist.tsv · rep_XXX/<method>.nwk
  mean_distance.tsv · consensus_with_support.nwk

Requires: biom-format, h5py, numpy, pandas, scipy, scikit-bio
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Set, FrozenSet

import numpy as np
import pandas as pd
import h5py

from biom import Table, load_table
from skbio import TreeNode
from skbio.stats.distance import DistanceMatrix
from skbio.tree import nj

from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import average as upgma_average

def compute_sample_totals(tbl: Table) -> Dict[str, int]:
    return {sid: int(tbl.data(sid, axis='sample').sum())
            for sid in tbl.ids(axis='sample')}

def load_and_filter_table(biom_fp: Path, depth: int, totals: Dict[str, int]) -> Table:
    tbl = load_table(str(biom_fp))
    keep = [sid for sid, tot in totals.items() if tot >= depth]
    if not keep:
        raise ValueError("No samples meet the requested depth.")
    return tbl.filter(keep, axis='sample', inplace=False)

def rarefy_counts_without_replacement(counts: np.ndarray, depth: int,
                                      rng: np.random.Generator) -> np.ndarray:
    total = int(counts.sum())
    if depth > total:
        raise ValueError(f"depth {depth} exceeds library size {total}")
    out = np.zeros_like(counts, dtype=np.int64)
    remaining = depth
    remaining_total = total
    for i in range(len(counts) - 1):
        ngood = int(counts[i])
        if remaining == 0 or ngood == 0:
            x = 0
        else:
            nbad = remaining_total - ngood
            x = int(rng.hypergeometric(ngood, nbad, remaining))
        out[i] = x
        remaining -= x
        remaining_total -= ngood
    out[-1] = remaining
    return out

def rarefy_table(tbl: Table, depth: int, rng: np.random.Generator) -> Table:
    obs_ids = tbl.ids(axis='observation')
    samp_ids = tbl.ids(axis='sample')
    cols = []
    for sid in samp_ids:
        vec = np.asarray(tbl.data(sid, axis='sample')).ravel().astype(np.int64)
        cols.append(rarefy_counts_without_replacement(vec, depth, rng))
    mat = np.column_stack(cols)  # obs x samples
    return Table(mat, obs_ids, samp_ids)

def write_biom_hdf5(tbl: Table, path: Path) -> None:
    """Write BIOM **HDF5** v2.1 so dartunifrac can open it."""
    with h5py.File(str(path), "w") as h5f:
        tbl.to_hdf5(h5f, generated_by="jackknife-dartunifrac")


def read_dm_tsv(tsv_fp: Path, expected_ids: List[str]) -> np.ndarray:
    df = pd.read_csv(tsv_fp, sep="\t", header=0, index_col=0)
    dm_ids = df.index.tolist()
    if set(dm_ids) != set(expected_ids):
        raise ValueError(
            "Replicate DM sample set mismatch.\n"
            f"Expected {len(expected_ids)} IDs, got {len(dm_ids)}.\n"
            f"Missing: {set(expected_ids)-set(dm_ids)}; Extra: {set(dm_ids)-set(expected_ids)}"
        )
    df = df.reindex(index=expected_ids, columns=expected_ids)
    return df.values.astype(float)

def dm_to_tree(D: np.ndarray, sample_ids: List[str], method: str = "upgma") -> TreeNode:
    """Build a tree from a square distance matrix."""
    if method == "upgma":
        Z = upgma_average(squareform(D, checks=False))
        return TreeNode.from_linkage_matrix(Z, id_list=sample_ids)
    elif method == "nj":
        dm = DistanceMatrix(D, ids=sample_ids)
        return nj(dm)
    else:
        raise ValueError("unknown tree method (use 'upgma' or 'nj')")

def _canonical_split(tipset: Set[str], fullset: Set[str]) -> FrozenSet[str]:
    other = fullset - tipset
    return frozenset(tipset if len(tipset) <= len(other) else other)

def tree_splits(t: TreeNode) -> Set[FrozenSet[str]]:
    leaves = {n.name for n in t.tips()}
    splits: Set[FrozenSet[str]] = set()
    for n in t.non_tips():
        tipnames = {x.name for x in n.tips()}
        if 0 < len(tipnames) < len(leaves):
            splits.add(_canonical_split(tipnames, leaves))
    return splits

def annotate_support(base: TreeNode,
                     split_counts: Dict[FrozenSet[str], int],
                     nrep: int) -> TreeNode:
    """Annotate internal nodes with jackknife support in [0,1] as plain numbers, e.g. 0.720"""
    full = {x.name for x in base.tips()}
    for n in base.non_tips():
        tipnames = {x.name for x in n.tips()}
        if 0 < len(tipnames) < len(full):
            key = _canonical_split(tipnames, full)
            sup = split_counts.get(key, 0) / float(nrep)  # 0–1
            n.name = f"{sup:.3f}"  # e.g., 0.720 — no brackets, no quotes
    return base
    
def run_dartunifrac(dart_bin: str,
                    tree_fp: Path,
                    biom_fp: Path,
                    weighted: bool,
                    metric: str,
                    sketch_size: int,
                    out_tsv: Path,
                    extra: Optional[List[str]] = None) -> None:
    cmd = [dart_bin, "-t", str(tree_fp), "-b", str(biom_fp)]
    if weighted:
        cmd.append("--weighted")
    if metric:
        cmd += ["-m", metric]
    if sketch_size:
        cmd += ["-s", str(sketch_size)]
    cmd += ["-o", str(out_tsv)]
    if extra:
        cmd += list(extra)
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser(
        description="Jackknifed UniFrac with DartUniFrac + UPGMA or NJ consensus."
    )
    ap.add_argument("--table", "-b", required=True, type=Path,
                    help="BIOM v2.1 table (.biom)")
    ap.add_argument("--tree", "-t", required=True, type=Path,
                    help="Rooted Newick tree with feature IDs as tips")

    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--depth", "-d", type=int,
                       help="Rarefaction depth (reads/sample). Use this OR --auto-min-depth.")
    group.add_argument("--auto-min-depth", action="store_true",
                       help="Use the minimum total reads across samples so no samples are dropped.")

    ap.add_argument("--iterations", "-n", default=50, type=int,
                    help="Number of jackknife replicates (default: 50)")
    ap.add_argument("--weighted", action="store_true",
                    help="Use weighted UniFrac (adds --weighted to DartUniFrac)")
    ap.add_argument("--metric", "-m", default="dmh",
                    help="DartUniFrac metric/mode (e.g., dmh)")
    ap.add_argument("--sketch-size", "-s", default=2048, type=int,
                    help="Sketch size for DartUniFrac (-s)")
    ap.add_argument("--tree-method", choices=["upgma", "nj"], default="upgma",
                    help="Tree method for each replicate and consensus (default: upgma)")
    ap.add_argument("--dartunifrac-bin", default="dartunifrac",
                    help="Path to DartUniFrac binary (default: in PATH)")
    ap.add_argument("--outdir", "-o", default=Path("jackknife_out"), type=Path,
                    help="Output directory (default: ./jackknife_out)")
    ap.add_argument("--seed", default=42, type=int,
                    help="RNG seed for reproducible subsampling (default: 42)")
    ap.add_argument("--extra", default="", type=str,
                    help="Extra args to pass through to DartUniFrac, quoted as a string "
                         "(e.g. --extra='--threads 16')")
    args = ap.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Load once to compute totals and (optionally) auto-pick depth
    full_tbl = load_table(str(args.table))
    totals = compute_sample_totals(full_tbl)
    (pd.DataFrame.from_dict(totals, orient="index", columns=["total_reads"])
       .sort_values("total_reads")
       .to_csv(outdir / "depths.tsv", sep="\t", header=True))

    if args.auto_min_depth:
        chosen_depth = min(totals.values()) if totals else 0
        if chosen_depth <= 0:
            raise ValueError("At least one sample has zero reads (or no data). "
                             "Cannot rarefy. Filter zero-read samples or provide a positive --depth.")
        print(f"[info] Auto-picked rarefaction depth = {chosen_depth} "
              f"(minimum across {len(totals)} samples)")
        args.depth = chosen_depth
    else:
        if args.depth is None or args.depth <= 0:
            raise ValueError("--depth must be > 0")
        chosen_depth = args.depth
    print(f"[info] Rarefaction depth = {chosen_depth}")
    # Filter to samples meeting chosen depth
    tbl = load_and_filter_table(args.table, chosen_depth, totals)
    sample_ids = list(tbl.ids(axis='sample'))
    (outdir / "samples_used.txt").write_text("\n".join(sample_ids))

    # Prepare RNG and accumulators
    rng = np.random.default_rng(args.seed)
    split_counts: Dict[FrozenSet[str], int] = defaultdict(int)
    D_sum: Optional[np.ndarray] = None

    extra_list = shlex.split(args.extra) if args.extra else None

    # Replicates
    for r in range(1, args.iterations + 1):
        rep_dir = outdir / f"rep_{r:03d}"
        rep_dir.mkdir(exist_ok=True)

        # Rarefy & write HDF5 BIOM
        rare_tbl = rarefy_table(tbl, chosen_depth, rng)
        rep_biom = rep_dir / "rarefied.biom"
        write_biom_hdf5(rare_tbl, rep_biom)

        # Compute UniFrac DM via DartUniFrac
        rep_dm = rep_dir / "dist.tsv"
        run_dartunifrac(
            dart_bin=args.dartunifrac_bin,
            tree_fp=args.tree,
            biom_fp=rep_biom,
            weighted=args.weighted,
            metric=args.metric,
            sketch_size=args.sketch_size,
            out_tsv=rep_dm,
            extra=extra_list
        )

        # Build tree for this replicate
        D = read_dm_tsv(rep_dm, sample_ids)
        tree = dm_to_tree(D, sample_ids, args.tree_method)
        tree.write(rep_dir / f"{args.tree_method}.nwk")

        # Accumulate for mean DM and jackknife support
        D_sum = D if D_sum is None else (D_sum + D)
        for sp in tree_splits(tree):
            split_counts[sp] += 1

    # Mean distance across replicates
    D_mean = D_sum / float(args.iterations)
    pd.DataFrame(D_mean, index=sample_ids, columns=sample_ids).to_csv(
        outdir / "mean_distance.tsv", sep="\t", float_format="%.10g"
    )

    # Consensus tree (built from mean DM) with jackknife support labels
    ref_tree = dm_to_tree(D_mean, sample_ids, args.tree_method)
    ref_tree = annotate_support(ref_tree, split_counts, args.iterations)
    ref_tree.write(outdir / "consensus_with_support.nwk")

    print(f"[OK] Wrote:\n"
          f"  - {outdir/'depths.tsv'}\n"
          f"  - {outdir/'samples_used.txt'}\n"
          f"  - {outdir/'mean_distance.tsv'}\n"
          f"  - {outdir/'consensus_with_support.nwk'}\n"
          f"  - {outdir}/rep_XXX/dist.tsv and {outdir}/rep_XXX/{args.tree_method}.nwk per replicate")


if __name__ == "__main__":
    main()
