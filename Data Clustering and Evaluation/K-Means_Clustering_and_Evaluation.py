import pandas as pd
import math
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt

# TASK USES NBA player stats from 2013. 480x28 Matrix of Data.
pd.set_option('display.max_rows', None)
r_data = pd.read_csv("nba2013.csv")
#Preprocess data by removing non-numerical data and filling NaN values.
data = r_data.drop(["player","pos","bref_team_id","season"],axis=1).fillna(0).to_numpy()
n,d = np.shape(data)


def K_means(k,iters):
# =============================================================================
# Performs K-means Clustering.
#     Returns:
#     clusters -> dict(k,C) of clusters
#     means    -> np.array((k,d)) of means
# =============================================================================
    
    #Initialize K random points by sampling the data.
    delta = 0
    indicies = np.random.randint(0,n,k)
    means = data[indicies,:]

    #Repeat for set number of iterations to converge.
    while iters > delta:
        clusters = defaultdict(list)
        #Assign each point to closest centroid.   
        for point in data:
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

    
def CH(k,clusters,means):
# =============================================================================
#     Returns Calinski Harabasz Index
# =============================================================================
    m = np.mean(data,axis=0)
    #Inter cluster variance W
    W = 0
    for i in range(k):
        sum2 = 0
        for x in clusters[i]:
            sum2 = sum2 + math.pow(np.linalg.norm(x-means[i]),2)
        W = W + sum2
    #Intracluster Variance B
    B = 0
    for j in range(k):
        Ci = np.linalg.norm(clusters[j])
        B = B + Ci * math.pow(np.linalg.norm(m-means[j]),2)
    
    return ((n-k)*B) / ((k-1)*W)


def DaivesB(k,clusters,means):
# =============================================================================
#     Returns Daives Bouldin Index
# =============================================================================
    db = []
    for i in range(k):
        for j in range(k):
            if j != i:
                Si = np.var(np.array(clusters[i]),axis=0)
                Sj = np.var(np.array(clusters[j]),axis=0)
                db.append((Si-Sj)/np.linalg.norm(means[i]-means[j])/k)
        
    
    return np.max(db)
    
# Set k and iteration amount, converges relatively fast.
k = 5
iters = 30

x = range(2,11)
y_sil = []
y_ch = []
y_db= []

#Perform K-means and Evaluation for k in range x.

for k in x:
    clusters,means = K_means(k,iters)
    y_ch.append(CH(k,clusters,means))
    y_db.append(DaivesB(k,clusters,means))
    
print("CALINSKI HARABASZ VALUES, K = 2,5,10")
print(y_ch[0],y_ch[3],y_ch[8])
print("DAIVES BOULDIN, K = 2,5,10")
print(y_db[0],y_db[3],y_db[8])

#Plot the values

ch = plt.figure(1,figsize=(20, 7))
c = plt.plot(x,y_ch,label = "Calinski Harabazs Index")

ch = plt.figure(2,figsize=(20, 7))
c = plt.plot(x,y_db,label = "Daives Bouldin Index")

plt.legend(loc = "best")
plt.show()




