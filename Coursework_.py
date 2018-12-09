from scipy.io import loadmat
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.neighbors import KNeighborsClassifier as KNN
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

gallery_length = len(gallery_id)
query_length = len(query_id)
rank_list = np.zeros((gallery_length, query_length))


total_accuracy = 0

for j in range(len(query_id)):

    new_gallery = ([]) 
    k = 0

    for i in range(len(gallery_id)):
            if(not((labels[gallery_id[i]-1] == labels[query_id[j]-1]) and (camera[gallery_id[i]-1] == camera[query_id[j]-1]))):
                new_gallery.append(gallery_id[i]-1)
                k = k + 1
    
    new_gallery = np.asarray(new_gallery)
    classifier = KNN(n_neighbors=1)
    classifier.fit(features[new_gallery], labels[new_gallery].ravel())
    
    reshaped_query = features[query_id[j]-1].reshape(1,-1)
    y_pred = classifier.predict(reshaped_query)
    total_accuracy = total_accuracy + metrics.accuracy_score(labels[query_id[j]-1].reshape(1,-1), y_pred)
    rank_list[0:k,j] = new_gallery
    print(j)
print("final accuracy: ", total_accuracy/len(query_id))

