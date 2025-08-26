import sys
import pandas as pd
import numpy as np
import skbio
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa


### GWMC
dm=skbio.io.read('GWMC_unifrac_binary_dist.tsv',into=skbio.stats.distance.DistanceMatrix)
dm1=skbio.io.read('GWMC_dart_dist.tsv',into=skbio.stats.distance.DistanceMatrix)

dm._data = dm._data.astype(np.float32)
dm1._data = dm1._data.astype(np.float32)

m=skbio.stats.distance.mantel(dm1,dm)
print("mantel r is:", m)



### ASVs_test
dm=skbio.io.read('ASVs_unifrac_truth_dist.tsv',into=skbio.stats.distance.DistanceMatrix)
dm1=skbio.io.read('ASVs_dmh.tsv',into=skbio.stats.distance.DistanceMatrix)

dm._data = dm._data.astype(np.float32)
dm1._data = dm1._data.astype(np.float32)

m=skbio.stats.distance.mantel(dm1,dm)
print("mantel r is:", m)




###GMT
dm=skbio.io.read('GMTOLsong_dist_c++.tsv',into=skbio.stats.distance.DistanceMatrix)
dm1=skbio.io.read('GMTOLsong_dist_dart.tsv',into=skbio.stats.distance.DistanceMatrix)

dm._data = dm._data.astype(np.float32)
dm1._data = dm1._data.astype(np.float32)

m=skbio.stats.distance.mantel(dm1,dm)
print("mantel r is:", m)

