import numpy as np
import pandas as pd
from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score,davies_bouldin_score,normalized_mutual_info_score


data = pd.read_csv("spiral.txt",sep = "\t").to_numpy()
n,d = np.shape(data)


# =============================================================================
# Gkern creates a gaussian kernel matrix given row-wise data and a sigma value(float)
# =============================================================================
def gkern(dat,sig):
    K = np.empty((n,n))
    for i in range(n):
        for j in range(n):
            K[i,j] = np.exp(-((np.linalg.norm(dat[i,:]-dat[j,:])**2))/(2*(sig**2)))
    return K

# =============================================================================
# Kmeans function
# =============================================================================
def K_means(k,iters,dat):
#     Returns:
#     clusters -> dict(k,C) of clusters
#     means    -> np.array((k,d)) of means
    
    #Initialize K random points by sampling the data.
    delta = 0
    N,D = np.shape(dat)
    indicies = np.random.randint(0,N,k)
    means = dat[indicies,:]

    #Repeat for set number of iterations to converge.
    while iters > delta:
        clusters = defaultdict(list)
        #Assign each point to closest centroid.   
        for point in dat:
            cluster = 0
            mins = float('inf')
            k = 0
            for centroid in means:
                D = np.linalg.norm(point-centroid)
                if D < mins:
                    mins = D
                    cluster = k
                k = k + 1
            clusters[cluster].append(point)           
        #Update the new centroid (mean) of each cluster.
        for i in range(k):
            means[i,:] = np.mean(np.array(clusters[i]),axis=0)
        delta +=1
    
    return (clusters,means)

# =============================================================================
# Unnormalized Spectral Clustering Function (utilizes Kmeans above)
# =============================================================================
def Spectral_Clustering(k):
#     Returns:
#     clusters -> dict(k,C) of clusters
    W = gkern(data[:,0:2],1.0)
    D = np.diag(np.sum(W, axis=1))
    #UNORMALIZED SPECTRAL CLUSTERING
    L = D-W
    e_values,e_vectors = np.linalg.eig(L)
    y = e_vectors[:, e_values.argsort()]
    y = y[:,0:k]
    clusters,_ = K_means(k,30,y)
    
    return clusters


# PERFORM K_MEANS
k = 3
(clusters,means) = K_means(k,30,data[:,0:2])
#Label data from dict
k_data = np.zeros((1,3))
for i in range(k):
    c = np.array(clusters[i])
    labels = np.ones((len(c),1))*i
    k_data = np.append(k_data,np.concatenate((c,labels),axis=1),axis = 0)
k_data = k_data[1:n+1,:]


# PERFORM SPECTRAL CLUSTERING
a = Spectral_Clustering(k)
#Label data from dict
s_data = np.zeros((1,3))
ind = 0
for i in range(k):
    c = data[ind:ind+len(a[i]),0:2]
    ind = ind+len(a[i])
    labels = np.ones((len(c),1))*i
    s_data = np.append(s_data,np.concatenate((c,labels),axis=1),axis=0)
s_data = s_data[1:n+1,:]



# PLOT ORGINAL LABLES, K_MEANS CLUSTERING, AND SPECTRAL CLUSTERING
O = plt.figure(0,figsize=(20, 7))
ax1 = O.add_subplot(1, 1, 1) # nrows, ncols, index
ax1.set_facecolor("salmon")
OS = plt.scatter(data[:,0].T,data[:,1].T, c = data[:,2].T)

K = plt.figure(1,figsize=(20, 7))
ax2 = K.add_subplot(1, 1, 1) # nrows, ncols, index
ax2.set_facecolor("salmon")
KS = plt.scatter(k_data[:,0].T,k_data[:,1].T, c = k_data[:,2].T)

S = plt.figure(2,figsize=(20, 7))
ax3 = S.add_subplot(1, 1, 1) # nrows, ncols, index
ax3.set_facecolor("salmon")
SS = plt.scatter(s_data[:,0].T,s_data[:,1].T, c = s_data[:,2].T)
plt.show()

#Calculate index scores
kmeans_sil = silhouette_score(k_data[:,0:2], k_data[:,2])
spectral_sil = silhouette_score(s_data[:,0:2], s_data[:,2])

kmeans_bouldin = davies_bouldin_score(k_data[:,0:2], k_data[:,2])
spectral_bouldin = davies_bouldin_score(s_data[:,0:2], s_data[:,2])

kmeans_normal = normalized_mutual_info_score(data[:,2], k_data[:,2])
spectral_normal = normalized_mutual_info_score(data[:,2], s_data[:,2])

print("Silhouette")
print(kmeans_sil)
print(spectral_sil)

print("Daives-Bouldin")
print(kmeans_bouldin)
print(spectral_bouldin)

print("Normalized Mutual Information")
print(kmeans_normal)
print(spectral_normal)
