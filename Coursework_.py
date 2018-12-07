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

new_gallery = ([])
total_accuracy = 0

for j in range(len(query_id)):

    new_gallery = ([])

    for i in range(len(gallery_id)):
            if(not((labels[gallery_id[i]] == labels[query_id[j]]) and (camera[gallery_id[i]] == camera[query_id[j]]))):
                new_gallery.append(gallery_id[i])
    
    new_gallery = np.asarray(new_gallery)
    classifier = KNN(n_neighbors=1)
    classifier.fit(features[new_gallery-1], labels[new_gallery-1].ravel())
    reshaped_query = np.reshape(query_id[j], 1,-1)
    y_pred = classifier.predict(features[reshaped_query-1])
    total_accuracy = total_accuracy + metrics.accuracy_score(labels[reshaped_query-1], y_pred)
    print(j)

print("final accuracy: ", total_accuracy/len(query_id))


'''
for i in range(len(gallery_id)):
    in_query = False
    for j in range(len(query_id)):
        if((labels[gallery_id[i]] == labels[query_id[j]]) and (camera[gallery_id[i]] == camera[query_id[j]])):
            in_query = True
    if(in_query == False): 
        new_gallery.append(gallery_id[i])
    
new_gallery = np.asarray(new_gallery)
print(new_gallery.shape)
accuracy = 0
classifier = KNN(n_neighbors=1)
pred = ([[]])
for i in range (len(query_id)):
    classifier.fit(features[new_gallery-1], labels[new_gallery-1].ravel())
    query = np.asarray(features[query_id[i]-1])
    y_pred = classifier.predict(query.reshape(1,-1))
    pred = np.append(pred, y_pred)
    print(pred.shape)
    #accuracy = accuracy + metrics.accuracy_score(labels[query_id[i]-1],y_pred)

print(y_pred.shape)
print(pred.shape)
print("Accuracy:",metrics.accuracy_score(labels[query_id-1],pred))

'''
