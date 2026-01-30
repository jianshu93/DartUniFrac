#!/usr/bin/env python3
import argparse
from pathlib import Path

# install unifrac first
# conda create --name unifrac -c conda-forge -c bioconda unifrac
# conda activate unifrac
# pip install unifrac iow

# running ssu
# time unifrac-binaries -t 2024.09.phylogeny.asv.nwk -i merged-gg2.2024.09.even500.biom -m weighted_normalized_fp32 --format hdf5 -o gg2_2024_merged_weighted_ssu_new.hf5 --pcoa 10 --n-substeps 24

# python ./extract_pcoa_ssu.py gg2_2024_merged_weighted_ssu_new.hf5 -o gg2_2024_merged_weighted_ssu_pcoa.txt


# use qiime2 to visualize results

# qiime tools import --type 'PCoAResults' --input-path gg2_2024_merged_weighted_ssu_pcoa.txt --output-path pcoa.qza

# qiime emperor plot --i-pcoa pcoa.qza --m-metadata-file metadata.tsv --o-visualization pcoa.qzv

# upload pcoa.qzv to https://view.qiime2.org

import unifrac

def main():
    ap = argparse.ArgumentParser(description="Extract UniFrac HDF5 PCoA and write QIIME2-compatible ordination.txt")
    ap.add_argument("h5file", help="UniFrac/unifrac-binaries output .h5/.hdf5 containing PCoA")
    ap.add_argument("-o", "--out", default="ordination.txt", help="Output ordination.txt path")
    args = ap.parse_args()

    h5 = Path(args.h5file)
    if not h5.exists():
        raise FileNotFoundError(h5)

    # returns skbio.stats.ordination.OrdinationResults
    ord_res = unifrac.h5pcoa(str(h5))

    out = Path(args.out)
    # Write in scikit-bio 'ordination' format (the same style QIIME2 exports)
    ord_res.write(str(out), format="ordination")

    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
