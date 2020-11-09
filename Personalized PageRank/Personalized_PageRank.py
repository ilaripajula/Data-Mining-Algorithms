import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

# =============================================================================
# Script performs personalized pagerank on dataset ('jester-800-10.csv') of 10 jokes rated 
# by 800 users. The data is binary (1 if joke is given positive review,0 otherwise). It
# outputs a reccomendation vector for a specific user derived from the ratings of the K-nearest
# neighbours.
# 
# This script file inlcudes 3 functions:
#     
#     pagerank -> performs personalized pagerank algorithm (power iteration method)
#                 returns personalization vector
#                 
#     preprocess -> performs preprocessing for the specific data 'jester-800-10.csv'
# 
#     user_user_reccomendation -> calculates the reccomendation vector for a user
#     
# =============================================================================

# =============================================================================
# FUNCTIONS
# =============================================================================

def pagerank(P,D, s, iters: int = 1000, d: float = 0.85):
    """
    Parameters
    ----------
    P : NxN np.array
        Probability array.
    D : NxN np.array
        Outgoing diagonals array.
    s : Nx1 np.array
        Personalized leap probability (initial node).
    iters : # of iterations
    d : alpha (discount) value

    Returns
    -------
    Nx1 np.array
        Ranked nodes.
    """
    v = s
    W = (P.T*d) @ np.linalg.inv(D.astype('float')) #D already inverse
    
    for _ in range(iters):
        v = W @ v + s*(1-d)
    return v/v.sum()

def preprocess(data,s):
    r1 = data[:,2:12]
    N = r1.shape[0]
    d = 0.1
    # This is the same matrix as iterating through 10 jokes and connecting user
    # that have liked similar jokes in an undirected fashion(bi-directed in this case).
    s = s @ r1.T
    s = normalize(s.reshape(1,-1),axis=1,norm ='l1')[0]
    P = r1 @ r1.T
    P = P - np.diag(P)
    link_P = normalize(P,axis=1,norm ='l1') * (1-d)
    leap_P = np.zeros((N,N)) + (d/N)
    # Outdegree of a user is the diagonal matrix of the degree.
    D = np.diag(data[:,12]) # Inverse D
    A = link_P + leap_P
    return A,D,s,d

def user_user_reccomendation(x,NN):
    """
    Parameters
    ----------
    x : np.array
        The rating vector of the input user.
    NN : KxD np.array
        The rating rowwise rating vectors of the K nearest neighbours.

    Returns
    -------
    x_rating : np.array
        The reccomendation vector.
        
    """
    xl = np.count_nonzero(x)
    x_rating = []
    for j in range(10):
        num = 0
        for y in NN:
                bit_and = np.count_nonzero(np.bitwise_and(y,x))
                num +=(y[j]-(1/bit_and)*np.count_nonzero(y))
        x_rating.append(num/10 + (1/bit_and)*xl)
            
    return x_rating

# =============================================================================
# CALLS
# =============================================================================
data = pd.read_csv("jester-800-10.csv").to_numpy()
input_data = pd.read_csv("test-800-10.csv").to_numpy()

# Column 0 = ID
# Columns 2-11 Jokes 
# Column 12 number of positives (degree)

# =============================================================================
# EXAMPLE 1
# 
# =============================================================================
ID = 15314
S = input_data[np.argwhere(input_data[:,0] == ID)[0,0],2:12]

A,D,s,d = preprocess(data,S)

v = pagerank(A,D,s,1000,0.85)
K = 10
NN = data[v.argsort()[-K:],2:12]

print("ID: " + str(ID) + " Ratings: " +  str(S))
print("K-NEAREST NEIGHBOURS:")
for i in v.argsort()[-K:]:
    print("ID: " + str(data[i,0]) + " Ratings: " +  str(data[i,2:12]))

print("RECCOMEND")
print(np.around(user_user_reccomendation(S,NN),2))
print('')

# =============================================================================
# EXAMPLE 2
# 
# =============================================================================
print('INPUT -> RECCOMENDATIONS FOR USERS IN test_800_10.csv ')
K = 10
for user in input_data:
    S = user[2:12]
    A,D,s,d = preprocess(data,S)
    ID = user[0]
    v = pagerank(A,D,s,1000,0.85)
    NN = data[v.argsort()[-K:],2:12]
    reccomend = user_user_reccomendation(S, NN)
    print("ID: " + str(ID) + " Ratings: " +  str(S) + ' Recommend: ' + str(np.around(reccomend,2)))




