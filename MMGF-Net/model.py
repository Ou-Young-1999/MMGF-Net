import torch.nn as nn
import torch
import numpy as np
from resnet import resnet34
from GAT import GAT
from corGAT import corGAT
import torch.nn.functional as F

class fc(nn.Module):
    def __init__(self):
        super(fc, self).__init__()
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(197*256, 72),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(72, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )

    def forward(self, inp):
        out = self.fc(inp)
        return out

class ae(nn.Module):
    def __init__(self):
        super(ae, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(22, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, inp):
        enc = self.encoder(inp)
        dec = self.decoder(enc)
        return enc, dec

class ChannelAttention(nn.Module):  # Channel Attention Module
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):  # Spatial Attention Module
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

class ModalAttention(nn.Module):  
    def __init__(self, in_planes):
        super(ModalAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // 8, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // 8, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = max_out + avg_out

        img = self.avg(out[:,0:196,:])
        cli = out[:,196,:].unsqueeze(1)
        both = torch.cat([img, cli],dim=1)
        img = self.softmax(both)[:,0,:].unsqueeze(1)
        cli = self.softmax(both)[:,1,:].unsqueeze(1)
        img = img.repeat(1,196,1)
        node = torch.cat([img,cli],dim=1)

        return node



class Classifier(nn.Module):
    def __init__(self, device):
        super(Classifier, self).__init__()

        self.encoder_img = resnet34()
        self.conGAT = GAT(n_feat=256, n_hid=256, n_class=256, dropout=0.5, alpha=0.2, n_heads=8)
        self.corGAT = corGAT(n_feat=256, n_hid=256, n_class=256, dropout=0.5, alpha=0.2, n_heads=8)
        self.encoder_cli = ae()
        self.fc = fc()
        self.channel = ChannelAttention(256)  # Channel Attention Module
        self.node = ModalAttention(197)  # Modal Attention Module
        self.spatial = SpatialAttention()  # Spatial Attention Module
        self.device = device

    def forward(self, img, cli):
        batch_size = img.size(0)

        out1, x1 = self.encoder_img(img)

        CBAM_Cout = self.channel(x1)
        x1 = x1 * CBAM_Cout
        CBAM_Sout = self.spatial(x1)
        x1 = x1 * CBAM_Sout

        x1 = x1.view(batch_size, x1.size(1), -1)
        x1 = x1.permute(0, 2, 1)
        
        x2, out2 = self.encoder_cli(cli)
        x2 = torch.unsqueeze(x2, 1) 

        x = torch.cat([x1,x2],dim=1)

        node = self.node(x)
        x = x * node

        n = x.size(1)
        adj = np.ones(n * n).reshape(n, n)

        adj = torch.tensor(adj, dtype=torch.float32).to(self.device)

        con = self.conGAT(x, adj)
        cor = self.corGAT(x, adj)
        out3 = con + cor + x
        out3 = self.fc(out3.view(batch_size, -1))

        return out1, out2, out3,node
