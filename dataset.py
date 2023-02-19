import torch.utils.data as data
from PIL import Image
import os
import os.path
import pickle
import torch

def split(data, flag):
    trainlen = int(0.8 * len(data))
    validlen = int(0.1 * len(data))
    train = data[:trainlen]
    valid = data[trainlen:(validlen + trainlen)]
    test = data[(validlen + trainlen):]

    if flag == 'train':
        return train

    if flag == 'valid':
        return valid

    if flag == 'test':
        return test


def make_dataset(dir, flag):
    
    imagesPath = []
    labelsPath = []
    clinicalsPath = []

    with open(os.path.join(dir, 'double22_img.pkl'), 'rb') as f:
        imagesPathDouble = pickle.load(f)
        imagesPathDouble = split(imagesPathDouble, flag)
    with open(os.path.join(dir, 'double22_label.pkl'), 'rb') as f:
        labelsPathDouble = pickle.load(f)
        labelsPathDouble = split(labelsPathDouble, flag)
    with open(os.path.join(dir, 'double22_clinical.pkl'), 'rb') as f:
        clinicalsPathDouble = pickle.load(f)
        clinicalsPathDouble = split(clinicalsPathDouble, flag)
    if flag == 'train':
        for i in range(len(imagesPathDouble)):
            if int(labelsPathDouble[i]) == 1:
                for j in range(1, 3):
                    imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                    labelsPath.append(labelsPathDouble[i])
                    clinicalsPath.append(clinicalsPathDouble[i])
                    imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                    labelsPath.append(labelsPathDouble[i])
                    clinicalsPath.append(clinicalsPathDouble[i])
            else:
                for j in range(1, 3):
                    imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                    labelsPath.append(labelsPathDouble[i])
                    clinicalsPath.append(clinicalsPathDouble[i])
    else:
        for i in range(len(imagesPathDouble)):
            for j in range(1, 3):
                imagesPath.append(imagesPathDouble[i] + '-{}'.format(j))
                labelsPath.append(labelsPathDouble[i])
                clinicalsPath.append(clinicalsPathDouble[i])
    
    return imagesPath, labelsPath, clinicalsPath

class myDataset(data.Dataset):

    def __init__(self, root, transform_x=None, flag='train'):

        image, label, clinical = make_dataset(root, flag)
        self.root = root
        self.transform = transform_x
        self.imagePath = image
        self.labelPath = label
        self.clinicalPath = clinical

    def __getitem__(self, index):

        name = self.imagePath[index]
        name = name.replace('\\', '/')
        label = self.labelPath[index]
        clinical = self.clinicalPath[index]

        cli = torch.tensor(clinical, dtype=torch.float64)
        cli = cli.type(torch.FloatTensor)
        
        image3 = Image.open(name + '-d3.jpg').convert('L')
        image3 = self.transform(image3)
        
        sample = torch.cat((image3, image3, image3), 0)
        return sample, cli, int(label)

    def __len__(self):

        return len(self.imagePath)