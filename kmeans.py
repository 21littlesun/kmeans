import numpy as np
from numpy.linalg import norm
import random
from scipy.spatial import distance

# This file implements the standard kmeans algorithm.
# Author: Lexie Sun
# Time: Feb 4, 2020 
def kmeans(X:np.ndarray, k:int, centroids=None, tolerance=1e-2):
    # select k unique points from X as initial centroids m(t=0)_1...k for clusters C(t=0)_1...k
    if centroids == None:
    	#standard kmeans algorithm.
    	n = X.shape[0] # n is the num of records in X.
    	p = 1
    else: n,p = X.shape
    # initially, randomly pick k points from X as the centroids.
    sample_index = np.array(random.sample(range(n), k)) 
    # print(sample_index)
    centroids = X[sample_index] # centroids is a k*p matrix. 
    # put the K points we picked into the clusters.
    clusters = [[i] for i in sample_index] # clusters should be a list of lists
    # for each point/record x in X, find the closet centroid to x.
    centroids_new = np.zeros((k, p))
    c_distance = norm(centroids_new-centroids)
    while c_distance > tolerance:
        distance = compute_distance(X, centroids, k, n)
        clusters = assign_centroid(X, distance, n, k)
        centroids_new = recompute_centroids(X, clusters, k, p)
        c_distance = norm(centroids_new-centroids)
        centroids = centroids_new
    return centroids, clusters


def compute_distance(X, centroids, k, n):
    distance = np.zeros((n, k))
    for i in range(k):
        distance[:,i] = norm(X-centroids[i,:], axis=1)
    return distance


def assign_centroid(X, distance, n, k):
	index = np.argmin(distance, axis=1) # value in 'index' is 0,....,k-1
	clusters = []
	for i in range(k):
		clusters.append(list(np.where(index == i)[0])) #append every x(index of x, actually) to its cluster.
	return clusters

def recompute_centroids(X, clusters, k, p):
    centroids_new = np.zeros((k, p))
    for i in range(k):
        index = clusters[i]
        centroids_new[i] = np.sum(X[index],axis=0)/len(clusters[i])
    return centroids_new


def likely_confusion_matrix(y, clusters):
    truth_F = set(np.where(y==0)[0])
    truth_T = set(np.where(y==1)[0])
    pred_F = set(clusters[0])
    pred_T = set(clusters[1])
    FF = len(truth_F.intersection(pred_F))
    FT = len(truth_F.intersection(pred_T))
    TF = len(truth_T.intersection(pred_F))
    TT = len(truth_T.intersection(pred_T))
    print('        pred F  pred T')
    print('Truth')
    print('F          '+str(FF)+'      '+str(FT))
    print('T          '+str(TF).rjust(3,' ')+'     '+str(TT))
    print('clustering accuracy: ', (FF+TT)/len(y))


def reassign_grey(X, centroids, clusters):
    # change X inplace, return None
    # change all the values into one of the values in the centroids.
    k,_ = centroids.shape
    row, col = X.shape
    X = X.reshape(-1,1)
    for i in range(k):
        X[clusters[i]] = i
    cluster_no = list(X[:,0].astype(np.uint8))
    X = centroids[cluster_no]
    return X.reshape(row, col)


def reassign_colors(X, centroids, clusters):
    k,_ = centroids.shape # the num of centroids
    row, col,_ = X.shape
    X = np.array([X[:,:,i].flatten() for i in range(X.shape[2])]).T
    for i in range(k):
        X[clusters[i]] = i
    cluster_no = list(X[:,0].astype(np.uint8))
    X = centroids[cluster_no]
    return X.reshape(row, col, 3)


def similarity_matrix(X):
    # compute the similarity of observation i and observation j.
    n,_ = X.shape
    similarity_X = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            similarity_X[i][j] = norm(X[i]-X[j])
    return similarity_X




