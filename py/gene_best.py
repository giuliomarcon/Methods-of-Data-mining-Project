import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import normalized_mutual_info_score

gene_df = pd.read_csv('genedata.csv', index_col = 0)

if len(gene_df.columns) == 7001:
    true_labels = gene_df['class']
    gene_df = gene_df.drop('class',axis=1)

gene = gene_df.to_numpy()

def perform_spectral_nn(data):
	spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors',n_neighbors=5,n_jobs=3).fit(data)
	labels = spectral.labels_
	return labels

spectral = perform_spectral_nn(gene)

with open('gene_solution.txt','w') as file:
    for label in spectral:
        file.write(str(label)+'\n')
