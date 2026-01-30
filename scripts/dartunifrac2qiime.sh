qiime tools import --input-path ./GWMC_unifrac_dmh.tsv --type 'DistanceMatrix' --output-path ./GWMC_unifrac_dmh.qza
qiime diversity pcoa --i-distance-matrix ./GWMC_unifrac_dmh.qza --o-pcoa ./GWMC_unifrac_dmh_pcoa.qza
qiime emperor plot --i-pcoa ./GWMC_unifrac_dmh_pcoa.qza --m-metadata-file ./GWMC_metadata_origin.tsv --o-visualization GWMC_unifrac_dmh_emperor.qzv


### or if you run with --pcoa option
qiime tools import \
  --type 'PCoAResults' \
  --input-path ordination.txt \
  --output-path GWMC_unifrac_dmh_pcoa.qza


qiime emperor plot --i-pcoa ./GWMC_unifrac_dmh_pcoa.qza --m-metadata-file ./GWMC_metadata_origin.tsv --o-visualization GWMC_unifrac_dmh_emperor.qzv


### visualize it at https://view.qiime2.org by uploading GWMC_unifrac_dmh_emperor.qzv
