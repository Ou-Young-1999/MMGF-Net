# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 15:37:54 2023

@author: 24567
"""
import pandas as pd
import numpy as np
from torchvision import transforms
from PIL import Image
import random
    
file = r'.\ODIR-5K_Training_Annotations.xlsx'
frame = pd.read_excel(file)
data = np.array(frame.values)
label_idx = [7,8,9,10,11,12,13,14]
word_idx = [5,6]
path_idx = [3,4]
num_idx = [0]

data_label = data[:,label_idx]
data_word = data[:,word_idx]
data_path = data[:,path_idx]
data_num = data[:,num_idx]

#转换标签
data_transform_label = []
for i in data_label:
    if i[0] == 1:
        data_transform_label.append(0)
    else:
        data_transform_label.append(1)

#转换关键词
data_transform_word = []
for i in data_word:
    keyword_number = [0,0,0,0,0,0,0,0]
    keyword_description = ['normal','retinopathy','glaucoma','cataract',\
                           'age-related macular degeneration','hypertensive','myopia']
    for j in range(0,7):
        if keyword_description[j] in i[0]:
            keyword_number[j]=1
        if keyword_description[j] in i[1]:
            keyword_number[j]=1
    if sum(keyword_number)==0:
        keyword_number[7]=1
    data_transform_word.append(keyword_number)

#合并图像
feature = []
tran = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    ])
for i in range(len(data_label)):
    num = data_num[i][0]
    #image1 = Image.open('./ODIR-5K_Training_Dataset/' + data_path[i][0])
    #image1 = tran(image1)
    #image2 = Image.open('./ODIR-5K_Training_Dataset/' + data_path[i][1])
    #image2 = tran(image2)
    #image = Image.blend(image1,image2,alpha=0.5)
    image_path = str(num)+'.jpg'
    #image.save('./ODIR/'+image_path)
    
    sample = [num,image_path]
    for j in data_transform_word[i]:
        rand = random.gauss(0, 0.5)
        sample.append(j+rand)
    sample.append(data_transform_label[i])
    feature.append(sample)
    #print(num)
    

#写入excel
columns = ['id','image','normal','retinopathy','glaucoma','cataract',\
           'age-related macular degeneration','hypertensive','myopia','other','label']
feature = pd.DataFrame(feature,columns=columns)
feature.to_excel('feature.xlsx',index=False)
print('over')
