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

for i in range(len(gallery_id)):
    for j in range(len(query_id)):
        if((labels[gallery_id[i]] == labels[query_id[j]]) and (camera[gallery_id[i]] == camera[query_id[j]])):
            new_gallery.append(gallery_id[i])
    
new_gallery = np.asarray(new_gallery)
print(new_gallery.shape)

classifier = KNN(n_neighbors=1)


classifier.fit(features[new_gallery-1], labels[new_gallery-1].ravel())
y_pred = classifier.predict(features[query_id-1])

print(y_pred.shape)
print("Accuracy:",metrics.accuracy_score(labels[query_id-1], y_pred))

