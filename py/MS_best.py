import pandas as pd
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import SpectralClustering

ms_df = pd.read_csv('msdata.csv', index_col = 0)

if len(ms_df.columns) == 5001:
    true_labels = ms_df['class']
    ms_df = ms_df.drop('class',axis=1)

ms = ms_df.to_numpy()

mas = MaxAbsScaler()
ms_mas = mas.fit_transform(ms)

def euclidean_similarity(data):
    return 1/(1+euclidean_distances(data, data))

ms_ed_mas = euclidean_similarity(ms_mas)


def perform_spectral_pc(data):
    spectral = SpectralClustering(n_clusters=5, affinity='precomputed',n_jobs=3).fit(data)
    labels = spectral.labels_
    return labels

spectral_pc_ed_mas = perform_spectral_pc(ms_ed_mas)

with open('ms_solution.txt','w') as file:
    for label in spectral_pc_ed_mas:
        file.write(str(label)+'\n')
