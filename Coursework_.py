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
from sklearn.preprocessing import normalize
import seaborn as sn
import itertools
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score


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
<<<<<<< HEAD

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
=======

c = 0
j = 0

maximum = np.amax(labels[gallery_id-1]) #1463
counter = np.zeros(maximum, dtype= int)

for j, i in itertools.product(range(maximum), range(len(gallery_id))):
    if(labels[gallery_id[i]-1]==j+1):
        counter[j] = counter[j] + 1
print(counter)
print("sum", np.sum(counter))
### Sorting gallery class wise for LMKNN

sorted_class = ([[]])
sorted_class = np.sort(labels[gallery_id-1])
print(sorted_class)
print(sorted_class.shape)

gallery_class = ([[]])
'''
for i in range(len(counter)-1):
    if(not(counter[i] == counter[i+1])):
        c = c + 1
c = c + 1

print(c)
'''

class_labels = ([])
j = 0
for i in range(maximum):
    #print(i)
    if(counter[i] > 0):
        l = 0
        for k in range(len(gallery_id)):
            if(labels[gallery_id[k]-1] == i+1 and l < counter[i]):
                #print("*")
                gallery_class = np.append(gallery_class, features[labels[gallery_id[k]-1]])
                #class_mean[j] = class_mean[j] + features[labels[gallery_id[k]-1]]
                #class_labels = np.append(class_labels, i+1)
                l = l + 1    
        if(l>0):
            c = c + 1
            print(c)
            #class_mean[:,j] = class_mean[:,j]/counter[i]
        #j = j + 1    

class_mean = np.zeros((c,2048))
k = 0
l = 0 
for i in range(maximum):
    if(counter[i]>0):
        for j in range(counter[i]):
            class_mean[k,:] = class_mean[k,:] + features[labels[gallery_id[l]-1]]
            l = l + 1
        k = k + 1
'''
for i in range(len(sorted_class)-1):
    if(not(sorted_class[i]==sorted_class[i+1])):
        class_mean[k,:] = class_mean[k,:]
'''
print(l)
gallery_class = gallery_class.reshape(5328,2048)
print(gallery_class)
print(gallery_class.shape)

print(class_mean)
print(class_mean.shape)

print(class_labels)
print(len(class_labels))


start_time = time.time()

classifier = NN(n_neighbors=10, metric='euclidean')

for j in range(len(query_id)):
    rank_list = ([])
    #k = 0
    #counter = 0
    for i in range(len(gallery_id)):
        if(not((labels[gallery_id[i]-1] == labels[query_id[j]-1]) and (camera[gallery_id[i]-1] == camera[query_id[j]-1]))):
            rank_list.append(gallery_id[i]-1)
            #k = k + 1
    rank_list = np.asarray(rank_list)
    classifier.fit(features[rank_list])
    ind = classifier.kneighbors(X=features[query_id[j]-1].reshape(1,-1), n_neighbors=10, return_distance=False)

    if labels[query_id[j]-1] == labels[rank_list[ind[0,0]]]:
        topone = topone + 1  
    if labels[query_id[j]-1] in labels[rank_list[ind[0,0:5]]]:
        topfive = topfive + 1
    if labels[query_id[j]-1] in labels[rank_list[ind[0,:]]]:
        topten = topten + 1

    print(j, ":", counter[j])
    #rank_list[0:k,j] = new_gallery
print("final accuracy for top 1: ", topone/len(query_id))
print("final accuracy for top 5: ", topfive/len(query_id))
print("final accuracy for top 10: ", topten/len(query_id))

end_time = time.time()
print("Computation Time: %s seconds" % (end_time - start_time))

#KMEANSS
'''
kmeans = KMeans(n_clusters=5, random_state=0).fit(features[gallery_id-1])
pred_id = kmeans.predict(features[query_id-1])
print(pred_id)
print(pred_id.shape)

k_means = KMeans(n_clusters=5,random_state=None)
 
k_means.fit(features[gallery_id-1])
y_predict = k_means.labels_
 
nmi = normalized_mutual_info_score(labels[gallery_id-1], y_predict)

if len(labels[gallery_id-1]) != len(y_predict):
    print("zxhcgs")
    exit(0)

label1 = np.unique(labels[gallery_id-1])
n_class1 = len(label1)
 
label2 = np.unique(y_predict)
n_class2 = len(label2)

n_class = max(n_class1, n_class2)
G = np.zeros((n_class, n_class))

for i in range(0, n_class1):
    for j in range(0, n_class2):
        ss = labels[gallery_id-1] == label1[i]
        tt = y_predict== label2[j]
        G[i, j] = np.count_nonzero(ss & tt)

A = la.linear_assignment(-G)

new_l2 = np.zeros(y_predict.shape)
for i in range(0, n_class2):
    new_l2[y_predict == label2[A[i][1]]] = label1[A[i][0]]
y_permuted_predict = new_l2.astype(int)

acc = accuracy_score(labels[gallery_id-1], y_permuted_predict)
print(acc)


classifier = NN(n_neighbors=10, metric='euclidean')
start_time = time.time()
for j in range(len(query_id)):
>>>>>>> 5bf07cbff39a343b5828d9b0de8efe6bce9d7108

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
end_time = time.time()
print("Computation Time: %s seconds" % (end_time - start_time))
'''


