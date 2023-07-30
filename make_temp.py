# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 10:45:45 2023

@author: 24567
"""
import pandas as pd
import numpy as np
import pickle
    
file = r'.\feature.xlsx'
frame = pd.read_excel(file)
data = np.array(frame.values)
label_idx = [10]
feature_idx = [2,3,4,5,6,7,8,9]
image_idx = [1]

data_label = np.squeeze(data[:,label_idx])
data_feature = np.squeeze(data[:,feature_idx])
data_image = np.squeeze(data[:,image_idx])

for i in range(0,10):
    print(data_image[i])
    print(data_feature[i])
    print(data_label[i])

with open('./temp/label.pkl', 'wb') as f:
    pickle.dump(data_label, f)
with open('./temp/feature.pkl', 'wb') as f:
    pickle.dump(data_feature, f)
with open('./temp/image.pkl', 'wb') as f:
    pickle.dump(data_image, f)
