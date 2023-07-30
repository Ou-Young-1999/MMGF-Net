import torch
import numpy as np
import random
from tqdm import tqdm
from torchvision.transforms import autoaugment, transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Classifier
from dataset import myDataset
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from WarmUpLR import WarmupLR 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter('./tensorboard/MBGF-Net')
model_path1 = './model/acc.ckpt'
model_path2 = './model/loss.ckpt'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(3407)

normalize = transforms.Normalize(mean=[0.556],
                                 std=[0.063])
test_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #normalize
])
train_tfm = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #normalize
])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy('imagenet')),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    #normalize
 ])

train_set = myDataset(root='../temp/', transform_x=train_tfm, flag='train', fold = 1)
valid_set = myDataset(root='../temp/', transform_x=test_tfm, flag='valid', fold = 1)

print('train data:{}'.format(len(train_set)))
print('valid data:{}'.format(len(valid_set)))

train_loader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True,
)
valid_loader = DataLoader(
    valid_set,
    batch_size=8,
    shuffle=True,
)

classifier = Classifier(device)
model = classifier.to(device)

model.encoder_img.load_state_dict(torch.load('resnet34-pre.pth'), strict=False)
in_channel = model.encoder_img.fc.in_features
model.encoder_img.fc = nn.Linear(in_channel, 2)
model.encoder_img.to(device)
'''
optimizer = torch.optim.Adam([
                    {'params': model.encoder_img.parameters(),'lr': 1e-5,'weight_decay': 1e-4},
                    {'params': model.encoder_cli.parameters(),'lr': 5e-5,'weight_decay': 1e-4},
                    {'params': model.conGAT.parameters(), 'lr': 1e-5,'weight_decay': 1e-4},
                    {'params': model.corGAT.parameters(), 'lr': 1e-5,'weight_decay': 1e-4},
                    {'params': model.fc.parameters(), 'lr': 1e-5,'weight_decay': 1e-4},
                    {'params': model.channel.parameters(), 'lr': 1e-5,'weight_decay': 1e-4},
                    {'params': model.node.parameters(), 'lr': 1e-5,'weight_decay': 1e-4},
                    {'params': model.spatial.parameters(), 'lr': 1e-5,'weight_decay': 1e-4}
                    ])
'''
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay = 1e-4)
num_epoch = 50
best_acc = 0.0
best_loss = 10000.0
step = 0

stale = 0
patience = 200

class_weight = torch.FloatTensor([1, 1]).to(device)

for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    train_acc_img = 0.0
    train_acc_cli = 0.0

    val_acc = 0.0
    val_loss = 0.0
    val_acc_img = 0.0
    val_acc_cli = 0.0

    num_examples = 0

    # training
    model.train()
    train_loader = tqdm(train_loader)
    
    for i, batch in enumerate(train_loader):
        images, clinicals, labels = batch
        images = images.to(device)
        clinicals = clinicals.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        out_img, out_cli, outputs = model(images, clinicals)
 
        loss1 = F.cross_entropy(outputs, labels, weight=class_weight)
        loss2 = F.cross_entropy(out_img, labels, weight=class_weight)
        loss3 = F.cross_entropy(out_cli, labels, weight=class_weight)
        #loss = 0.1*loss1 + 0.6*loss2 + 0.3*loss3
        loss = 0.4*loss1+0.3*loss2+0.3*loss3
        loss.backward()
        
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)       
        optimizer.step()
        step = 1 + step
        
        num_examples+=len(labels)

        _, train_pred = torch.max(outputs, 1)
        _, train_pred_img = torch.max(out_img, 1)
        _, train_pred_cli = torch.max(out_cli, 1)
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_acc_img += (train_pred_img.detach() == labels.detach()).sum().item()
        train_acc_cli += (train_pred_cli.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    
    accuracy_train=train_acc/num_examples
    accuracy_train_img=train_acc_img/num_examples
    accuracy_train_cli=train_acc_cli/num_examples
    writer.add_scalar('Loss/train', train_loss / len(train_loader), epoch)
    writer.add_scalar('Acc/train', accuracy_train, epoch)
    writer.add_scalar('Acc_img/train', accuracy_train_img, epoch)
    writer.add_scalar('Acc_cli/train', accuracy_train_cli, epoch)

    # validation
    model.eval()

    num_correct = 0
    num_examples = 0
    num_tp = 0
    num_tn = 0
    num_ap = 0
    num_an = 0
    num_prep = 0
    num_pren = 0

    num_correct_img = 0
    num_examples_img = 0
    num_tp_img = 0
    num_tn_img = 0
    num_ap_img = 0
    num_an_img = 0
    num_prep_img = 0
    num_pren_img = 0

    num_correct_cli = 0
    num_examples_cli = 0
    num_tp_cli = 0
    num_tn_cli = 0
    num_ap_cli = 0
    num_an_cli = 0
    num_prep_cli = 0
    num_pren_cli = 0

    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            images, clinicals, labels = batch
            images = images.to(device)
            clinicals = clinicals.to(device)
            labels = labels.to(device)

            out_img, out_cli, outputs = model(images, clinicals)
 
            loss1 = F.cross_entropy(outputs, labels, weight=class_weight)
            loss2 = F.cross_entropy(out_img, labels, weight=class_weight)
            loss3 = F.cross_entropy(out_cli, labels, weight=class_weight)
            #loss = 0.1*loss1 + 0.6*loss2 + 0.3*loss3
            loss = 0.4*loss1+0.3*loss2+0.3*loss3

            _, val_pred = torch.max(outputs, 1)
            _, val_pred_img = torch.max(out_img, 1)
            _, val_pred_cli = torch.max(out_cli, 1)
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item()
            val_acc_img += (val_pred_img.cpu() == labels.cpu()).sum().item()
            val_acc_cli += (val_pred_cli.cpu() == labels.cpu()).sum().item()
            val_loss += loss.item()

            correct = sum(targetsi == predicti for targetsi, predicti in zip(labels, val_pred))
            tp = sum(targetsi and predicti for targetsi, predicti in zip(labels, val_pred)) 
            ap = sum(labels)
            prep = sum(val_pred) 
            tn = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(labels, val_pred)) 
            an = len(labels) - sum(labels) 
            pren = len(labels) - sum(val_pred) 
            num_correct += correct 
            num_examples += len(labels)  
            num_tp += tp  
            num_tn += tn  
            num_ap += ap  
            num_an += an  
            num_prep += prep
            num_pren += pren

            correct_img = sum(targetsi == predicti for targetsi, predicti in zip(labels, val_pred_img))
            tp_img = sum(targetsi and predicti for targetsi, predicti in zip(labels, val_pred_img))
            ap_img = sum(labels)  
            prep_img = sum(val_pred_img)  
            tn_img = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(labels, val_pred_img)) 
            an_img = len(labels) - sum(labels) 
            pren_img = len(labels) - sum(val_pred_img) 
            num_correct_img += correct_img 
            num_examples_img += len(labels) 
            num_tp_img += tp_img 
            num_tn_img += tn_img 
            num_ap_img += ap_img  
            num_an_img += an_img  
            num_prep_img += prep_img
            num_pren_img += pren_img

            correct_cli = sum(targetsi == predicti for targetsi, predicti in zip(labels, val_pred_cli))
            tp_cli = sum(targetsi and predicti for targetsi, predicti in zip(labels, val_pred_cli))  
            ap_cli = sum(labels)  
            prep_cli = sum(val_pred_cli) 
            tn_cli = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(labels, val_pred_cli))  
            an_cli = len(labels) - sum(labels)  
            pren_cli = len(labels) - sum(val_pred_cli) 
            num_correct_cli += correct_cli  
            num_examples_cli += len(labels)  
            num_tp_cli += tp_cli  
            num_tn_cli += tn_cli  
            num_ap_cli += ap_cli  
            num_an_cli += an_cli  
            num_prep_cli += prep_cli
            num_pren_cli += pren_cli

        accuracy_valid=val_acc/num_examples
        accuracy_valid_img=val_acc_img/num_examples
        accuracy_valid_cli=val_acc_cli/num_examples
        writer.add_scalar('Loss/valid', val_loss / len(valid_loader), epoch)
        writer.add_scalar('Acc/valid', accuracy_valid, epoch)
        writer.add_scalar('Acc_img/valid', accuracy_valid_img, epoch)
        writer.add_scalar('Acc_cli/valid', accuracy_valid_cli, epoch)

        yRecall = num_tp / (num_ap+ 1e-8) 
        yPrecision = num_tp / (num_prep+ 1e-8)
        yF1 = 2 * yRecall * yPrecision / (yRecall + yPrecision + 1e-8)
        nRecall = num_tn / (num_an + 1e-8)
        nPrecision = num_tn / (num_pren+ 1e-8)
        nF1 = 2 * nRecall * nPrecision / (nRecall + nPrecision + 1e-8)

        yRecall_img = num_tp_img / (num_ap_img+ 1e-8)  
        yPrecision_img = num_tp_img / (num_prep_img+ 1e-8)
        yF1_img = 2 * yRecall_img * yPrecision_img / (yRecall_img + yPrecision_img + 1e-8)
        nRecall_img = num_tn_img / (num_an_img + 1e-8) 
        nPrecision_img = num_tn_img / (num_pren_img+ 1e-8)
        nF1_img = 2 * nRecall_img * nPrecision_img / (nRecall_img + nPrecision_img + 1e-8)

        yRecall_cli = num_tp_cli / (num_ap_cli+ 1e-8)  
        yPrecision_cli = num_tp_cli / (num_prep_cli+ 1e-8)
        yF1_cli = 2 * yRecall_cli * yPrecision_cli / (yRecall_cli + yPrecision_cli + 1e-8)
        nRecall_cli = num_tn_cli / (num_an_cli + 1e-8) 
        nPrecision_cli = num_tn_cli / (num_pren_cli+ 1e-8)
        nF1_cli = 2 * nRecall_cli * nPrecision_cli / (nRecall_cli + nPrecision_cli + 1e-8)

        print('[{:03d}/{:03d}] Train Acc: {:3.6f}/{:3.6f}/{:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f}/{:3.6f}/{:3.6f} loss: {:3.6f}'.format(
            epoch + 1, num_epoch, accuracy_train, accuracy_train_img, accuracy_train_cli, train_loss / len(train_loader),
            accuracy_valid, accuracy_valid_img, accuracy_valid_cli, val_loss / len(valid_loader)
        ))

        print('All—P-recall={:.4f},P-precision={:.4f},F1={:.4f} | N-recall={:.4f},N-precision={:.4f},F1={:.4f}'.format(
            yRecall, yPrecision, yF1, nRecall, nPrecision, nF1
        ))
        print('img-P-recall={:.4f},P-precision={:.4f},F1={:.4f} | N-recall={:.4f},N-precision={:.4f},F1={:.4f}'.format(
            yRecall_img, yPrecision_img, yF1_img, nRecall_img, nPrecision_img, nF1_img
        ))
        print('cli-P-recall={:.4f},P-precision={:.4f},F1={:.4f} | N-recall={:.4f},N-precision={:.4f},F1={:.4f}'.format(
            yRecall_cli, yPrecision_cli, yF1_cli, nRecall_cli, nPrecision_cli, nF1_cli
        ))

        if val_acc > best_acc:
            best_acc = val_acc
            stale = 0
            torch.save(model.state_dict(), model_path1)
            print('saving best_acc model with acc {:.3f}'.format(best_acc / num_examples))
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
        
        if val_loss < best_loss:
            best_loss = val_loss
            stale = 0
            torch.save(model.state_dict(), model_path2)
            print('saving best_loss model with acc {:.3f}'.format(val_acc / num_examples))
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break
