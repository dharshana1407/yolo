from scipy.io import loadmat
train_idxs = loadmat('cuhk03_new_protocol_config_labeled.mat')
print(train_idxs)
camera = train_idxs['camId']
print(camera.ravel())
print(camera.shape)    
#yollo
import json
with open('feature_data.json', 'r') as f:
    features = json.load(f)