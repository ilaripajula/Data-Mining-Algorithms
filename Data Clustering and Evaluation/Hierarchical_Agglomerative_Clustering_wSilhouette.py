from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from sklearn.metrics import silhouette_score,pairwise_distances
from sklearn import preprocessing
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# TASK USES NBA player stats from 2013. 480x28 Matrix of Data.
#Import Data and standardize and center to mean.
data = pd.read_csv("nba2013.csv")
index = data.set_index('player').index
N_data = data.drop(["player","pos","bref_team_id","season"],axis=1).fillna(0).to_numpy()
n,d = np.shape(N_data)

#UNCOMMENT to model with standardized data. 
#N_data = preprocessing.scale(N_data)

# Shuffling rows before doing computatation effects clustering change. 

BOOLEAN_SHUFFLE_ROWS = False

def shuffle():
    if BOOLEAN_SHUFFLE_ROWS:
        np.random.shuffle(N_data)
        
#Clustering calculations using skllearn
shuffle()
ZS = linkage(N_data,method = "single")
shuffle()
ZC = linkage(N_data,method = "complete")
shuffle()
ZA = linkage(N_data,method = "average")
shuffle()
ZCe= linkage(N_data,method = "centroid")

#PLOT SILHOUETTE
distances = pairwise_distances(N_data)
x = list(range(2,11))

sil = plt.figure(0,figsize=(20, 7))

y_single = [silhouette_score(distances,fcluster(ZS,k,criterion = "maxclust"),metric="euclidean") for k in x]
s = plt.plot(x,y_single, label = "single")

y_complete = [silhouette_score(distances,fcluster(ZC,k,criterion = "maxclust"),metric="euclidean") for k in x]
c = plt.plot(x,y_complete,label = "complete")

y_average = [silhouette_score(distances,fcluster(ZA,k,criterion = "maxclust"),metric="euclidean") for k in x]
a = plt.plot(x,y_average,label  ="average")

y_centroid = [silhouette_score(distances,fcluster(ZCe,k,criterion = "maxclust"),metric="euclidean") for k in x]
ce = plt.plot(x,y_centroid,label  ="centroid")

plt.legend(loc='best')

#PLOT DENDROGRAMS
single = plt.figure(1,figsize=(20, 7))
dendrogram(ZS,
            orientation='top',
            distance_sort='single',labels = index)

complete = plt.figure(2,figsize=(20, 7))
dendrogram(ZC,
            orientation='top',
            distance_sort='complete',labels = index)

average = plt.figure('AVG',figsize=(20, 7),)
dendrogram(ZA,
            orientation='top',
            distance_sort='average',labels = index)

centroid = plt.figure('C',figsize=(20, 7),)
dendrogram(ZCe,
            orientation='top',
            distance_sort='centroid',labels = index)

plt.show()