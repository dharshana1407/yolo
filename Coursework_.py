from scipy.io import loadmat
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.neighbors import NearestNeighbors as NN
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
import time
import scipy as sp
from scipy.spatial import distance
from sklearn.decomposition import PCA
import pandas as pd
from numpy import linalg as LA

import scipy.io as spio
from numpy import linalg as LA
from sklearn.preprocessing import normalize
import seaborn as sn

train_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')
print(len(train_idxs))
camera = train_idxs['camId'];
camera = camera.flatten()
print(camera.shape)    

labels = train_idxs['labels']
labels = labels.flatten()
print(labels.shape)

query_id = train_idxs['query_idx']
query_id= query_id.flatten()
print(query_id.shape)

gallery_id = train_idxs['gallery_idx']
gallery_id = gallery_id.flatten()
print(gallery_id.shape)

training_data = train_idxs['train_idx']
training_data = training_data.flatten()
print(training_data.shape)

import json
with open('feature_data.json', 'r') as f:
    features = json.load(f) 
features = np.asarray(features)
print(features[training_data - 1].shape)

rank_list = ([])

total_accuracy = 0
topone = 0
topfive = 0
topten = 0

start_time = time.time()

#PCA
total_train = 0
for i in range(len(training_data)):
    total_train = total_train + features[training_data[i] - 1]

avg_train = total_train/len(training_data)

A = np.zeros_like(features)

for i in range(len(training_data)):
    A[i] = features[training_data[i] - 1] - avg_train
print("shape of A", A.shape)
A_t = np.transpose(A)

S = np.dot(A_t, A)/len(training_data)
print("shape of S", S.shape)

#find eigenvalues and eigenvectors
eigvals, eigvecs = np.linalg.eigh(S)
print("no. of eigvals", len(eigvals))
rank = np.linalg.matrix_rank(S, tol = None, hermitian = True) # 363
print("rank", rank)

zero_eigvals = len(eigvals) - rank

non_zero_eigvals = eigvals
non_zero_eigvecs = eigvecs

u = normalize(np.dot(A, non_zero_eigvecs), axis = 0)
print("shape of u", u.shape)

M_pca = rank - 2047 #reducing this value reduces dimensions...I think- currently set to M_pca = 1

#FIND PROJECTION W

w_t = np.array([[]])
w_t = np.dot(np.transpose(A), u[:,0:M_pca])
w_t = np.reshape(w_t,(2048,M_pca))
w = np.transpose(w_t)
print("shape of w", w.shape)
'''
#RECONSTRUCTION USING W
Zk = np.array([[]])

for j in range(2048):
    z = np.zeros(14096, dtype = int)
    for i in range(M_pca):
        w_u = w[i,j]*u[:, i]
        z = (np.add(z, w_u))
    Zk = np.append(Zk,z)
print("shape of Zk", Zk.shape)

Zk = np.reshape(Zk,(14096, 2048)) + avg_train
print(Zk)
'''
#MAHALANOBIS

df = pd.DataFrame(features[training_data - 1])
covmx = df.cov()
invcovmx = sp.linalg.inv(covmx)
min_mahala = np.zeros((len(training_data),2))
for i in range(len(training_data)):
    j = 0

    for j in range(len(training_data)):

        if(labels[training_data[i] - 1] != labels[training_data[j] - 1]):

        #cov = np.cov(features[training_data[i]],features[training_data[j]])
        #invcov = sp.linalg.inv(cov)
            current_mahala = distance.mahalanobis(features[training_data[i] - 1], features[training_data[j] - 1], covmx)
            if(current_mahala < min_mahala[i, 0]):
                min_mahala[i,0] = current_mahala
                min_mahala[i,0] = j
    print(i)
#todo for mahalanobis- use distance to find closest 




'''
K MEANS ATTEMPT
km = KMeans(n_clusters=2, random_state=0).fit(features[gallery_id-1])

km = np.asarray(km.labels_)

km_1 = (gallery_id-1)[np.where(km == 0)]
km_2 = (gallery_id-1)[np.where(km == 1)]

km_1 = np.asarray(km_1)
km_2 = np.asarray(km_2)


print(len(gallery_id-1))
print("km_1:", km_1.shape)
print("km_2:", km_2.shape)
'''

#KNN FUNCTION

def knn(gallery_id, query_id, labels, camera):
    distance_metric = 'cosine'
    classifier = NN(algorithm = 'auto', n_neighbors=10, metric= distance_metric)

    for j in range(len(query_id)):
        print(j)
        rank_list = ([])
        k = 0

        for i in range(len(gallery_id)):
            if(not((labels[gallery_id[i]-1] == labels[query_id[j]-1]) and (camera[gallery_id[i]-1] == camera[query_id[j]-1]))):
                rank_list.append(gallery_id[i]-1)
        
        rank_list = np.asarray(rank_list)

        classifier.fit(features[rank_list])
        ind = classifier.kneighbors(X=features[query_id[j]-1].reshape(1,-1), n_neighbors=10, return_distance=False)

        if labels[query_id[j]-1] == labels[rank_list[ind[0,0]]]:
            topone = topone + 1

        if labels[query_id[j]-1] in labels[rank_list[ind[0,0:5]]]:
            topfive = topfive + 1

        if labels[query_id[j]-1] in labels[rank_list[ind[0,:]]]:
            topten = topten + 1

        #total_accuracy = total_accuracy + metrics.accuracy_score(labels[query_id[j]-1].reshape(1,-1), y_pred)
        #print(j)
        #rank_list[0:k,j] = new_gallery

end_time = time.time()

print("Using distance metric", distance_metric)    
print("time taken is ", end_time - start_time)
print("final accuracy for top 1: ", topone/len(query_id))
print("final accuracy for top 5: ", topfive/len(query_id))
print("final accuracy for top 10: ", topten/len(query_id))
#print("final accuracy for top 1: ", total_accuracy/len(query_id))


