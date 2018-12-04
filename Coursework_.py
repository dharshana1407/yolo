from scipy.io import loadmat
from sklearn.model_selection import train_test_split
train_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')
#print(train_idxs)
camera = train_idxs['camId']
#print(camera.ravel())
#print(camera.shape)    
#yollo
import json
with open('feature_data.json', 'r') as f:
    features = json.load(f)

labels = train_idxs['labels'].flatten()
camId = train_idxs['camId'].flatten()
gallery_idx = train_idxs['gallery_idx'].flatten()
query_idx = train_idxs['query_idx'].flatten()
train_idx = train_idxs['train_idx'].flatten()

new_gallery = ([])

for i in range(len(gallery_idx)):

    for j in range(len(query_idx)):

        if((labels[gallery_idx[i]] == labels[query_idx[j]]) and (camId[gallery_idx[i]] == camId[query_idx[j]])):
            new_gallery.append(gallery_idx[i])

print(len(new_gallery))