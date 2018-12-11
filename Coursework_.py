from scipy.io import loadmat
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.neighbors import NearestNeighbors as NN
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  
import time
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
import sklearn.utils.linear_assignment_ as la
import itertools
from sklearn.decomposition import PCA

train_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')
print(len(train_idxs))

camera = train_idxs['camId']
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

import json
with open('feature_data.json', 'r') as f:
    features = json.load(f) 
features = np.asarray(features)
print(features.shape)

rank_list = ([])
'''
gallery_length = len(gallery_id)
query_length = len(query_id)
rank_list = np.zeros((gallery_length, query_length))
'''

total_accuracy = 0
topone = 0
topfive = 0
topten = 0

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

    rank_list = ([])
    k = 0

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
        print(topfive)

    if labels[query_id[j]-1] in labels[rank_list[ind[0,:]]]:
        topten = topten + 1

    #total_accuracy = total_accuracy + metrics.accuracy_score(labels[query_id[j]-1].reshape(1,-1), y_pred)
    print(j)
    #rank_list[0:k,j] = new_gallery
    
print("final accuracy for top 1: ", topone/len(query_id))
print("final accuracy for top 1: ", topfive/len(query_id))
print("final accuracy for top 1: ", topten/len(query_id))
#print("final accuracy for top 1: ", total_accuracy/len(query_id))
end_time = time.time()
print("Computation Time: %s seconds" % (end_time - start_time))
'''


