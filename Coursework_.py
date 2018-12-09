from scipy.io import loadmat
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.neighbors import NearestNeighbors as NN
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix  

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
classifier = NN(n_neighbors=10, metric='euclidean')

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


