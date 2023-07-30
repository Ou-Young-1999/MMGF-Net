import torch.utils.data as data
from PIL import Image
import os
import os.path
import pickle
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

def split(data, flag, fold):
    train = []
    valid = []
    test = []
    n = 0
    for i in data:
        if n == 0:
            test.append(i)
        elif n == fold:
            valid.append(i)
        else:
            train.append(i)
        if n == 4:
            n = 0
        else:
            n = n+1

    if flag == 'train':
        return train
    if flag == 'valid':
        return valid
    if flag == 'test':
        return test

def make_dataset(dir, flag, fold):
    
    imagesPath = []
    labelsPath = []
    clinicalsPath = []
    
    with open(os.path.join(dir, 'image.pkl'), 'rb') as f:
        imagesPath = pickle.load(f)
        imagesPath = split(imagesPath, flag, fold)
    with open(os.path.join(dir, 'label.pkl'), 'rb') as f:
        labelsPath = pickle.load(f)
        labelsPath = split(labelsPath, flag, fold)
    with open(os.path.join(dir, 'feature.pkl'), 'rb') as f:
        clinicalsPath = pickle.load(f)
        clinicalsPath = split(clinicalsPath, flag, fold)
    if flag =='train':
        imageDouble = []
        labelDouble = []
        clinicalDouble = []
        for i in range(len(labelsPath)):
            if labelsPath[i] == 0:
                imageDouble.append(imagesPath[i])
                labelDouble.append(labelsPath[i])
                clinicalDouble.append(clinicalsPath[i])
                imageDouble.append(imagesPath[i])
                labelDouble.append(labelsPath[i])
                clinicalDouble.append(clinicalsPath[i])
            else:
                imageDouble.append(imagesPath[i])
                labelDouble.append(labelsPath[i])
                clinicalDouble.append(clinicalsPath[i])
        return imageDouble,labelDouble,clinicalDouble
    
    return imagesPath, labelsPath, clinicalsPath

class myDataset(data.Dataset):

    def __init__(self, root, transform_x=None,  flag='train',fold=1):

        image, label, clinical = make_dataset(root, flag, fold)
        self.root = root
        self.transform = transform_x
        self.imagePath = image
        self.labelPath = label
        self.clinicalPath = clinical

    def __getitem__(self, index):

        name = self.imagePath[index]
        label = self.labelPath[index]
        clinical = self.clinicalPath[index]

        clinical = clinical.astype(float)
        cli = torch.tensor(clinical)
        cli = cli.type(torch.FloatTensor)
        
        image = Image.open('../ODIR/' + name)
        image = self.transform(image)
        
        return image, cli, int(label)

    def __len__(self):

        return len(self.imagePath)
