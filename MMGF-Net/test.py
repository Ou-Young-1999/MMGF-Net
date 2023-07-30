import os
import sys
import numpy

import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm

from dataset import myDataset
from model import Classifier


def main():
    device = torch.device("cpu")
    print("using {} device.".format(device))
    
    normalize = transforms.Normalize(mean=[0.556],
                                 std=[0.063])
    test_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Resize((224,224)),
        transforms.ToTensor(),
    ])
    
    test_set = myDataset(root='../temp/', transform_x=test_tfm, flag='test', fold = 1)
    #test_set = myTestDataset(root='../Data/temp', transform_x=test_tfm, flag='test', catagory='single')

    batch_size = 1
    nw = 1  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    test_loader = torch.utils.data.DataLoader(test_set,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    test_num = len(test_set)
    print("using {} images for test.".format(test_num))
    
    net = Classifier(device)
    num = 0
    for child in net.children():
        if num == 0:
            in_channel = child.fc.in_features#迁移学习
            child.fc = nn.Linear(in_channel, 2)#修改最后一层全连接层
            child.to(device)
            #print(child)
            break
        num = num+1
    model_weight_path = "./model/acc.ckpt"
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # change fc layer structure

    net.to(device)
    
    # test
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    acc_img = 0.0
    acc_cli = 0.0
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
    
    pre = []
    out = []
    label = []
    node = []

    pre_img = []
    output_img = []

    pre_cli = []
    output_cli = []
    
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for test_data in test_bar:
            test_images, _, test_labels = test_data
            out_img, out_cli, outputs, nodes = net(test_images.to(device),_)
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, test_labels.to(device)).sum().item()

            predict_y_img = torch.max(out_img, dim=1)[1]
            acc_img += torch.eq(predict_y_img, test_labels.to(device)).sum().item()

            predict_y_cli = torch.max(out_cli, dim=1)[1]
            acc_cli += torch.eq(predict_y_cli, test_labels.to(device)).sum().item()
            
            soft = softmax(outputs)
            pre.append(predict_y)
            label.append(test_labels)
            out.append(soft)
            node.append(nodes)

            soft_img = softmax(out_img)
            pre_img.append(predict_y_img)
            output_img.append(soft_img)

            soft_cli = softmax(out_cli)
            pre_cli.append(predict_y_cli)
            output_cli.append(soft_cli)
            
            correct = sum(targetsi == predicti for targetsi, predicti in zip(test_labels, predict_y))
            tp = sum(targetsi and predicti for targetsi, predicti in zip(test_labels, predict_y))  # 真阳
            ap = sum(test_labels)  # 实阳
            prep = sum(predict_y)  # 预测是阳
            tn = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(test_labels, predict_y))  # 真阴
            an = len(test_labels) - sum(test_labels)  # 实阴
            pren = len(test_labels) - sum(predict_y)  # 预测是阴
            num_correct += correct  # 总准确数目
            num_examples += len(test_labels)  # 总样本量
            num_tp += tp  # 总真+
            num_tn += tn  # 总真-
            num_ap += ap  # 总实+
            num_an += an  # 总实—
            num_prep += prep
            num_pren += pren

            correct_img = sum(targetsi == predicti for targetsi, predicti in zip(test_labels, predict_y_img))
            tp_img = sum(targetsi and predicti for targetsi, predicti in zip(test_labels, predict_y_img))  # 真阳
            ap_img = sum(test_labels)  # 实阳
            prep_img = sum(predict_y_img)  # 预测是阳
            tn_img = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(test_labels, predict_y_img))  # 真阴
            an_img = len(test_labels) - sum(test_labels)  # 实阴
            pren_img = len(test_labels) - sum(predict_y_img)  # 预测是阴
            num_correct_img += correct_img  # 总准确数目
            num_examples_img += len(test_labels)  # 总样本量
            num_tp_img += tp_img  # 总真+
            num_tn_img += tn_img  # 总真-
            num_ap_img += ap_img  # 总实+
            num_an_img += an_img  # 总实—
            num_prep_img += prep_img
            num_pren_img += pren_img

            correct_cli = sum(targetsi == predicti for targetsi, predicti in zip(test_labels, predict_y_cli))
            tp_cli = sum(targetsi and predicti for targetsi, predicti in zip(test_labels, predict_y_cli))  # 真阳
            ap_cli = sum(test_labels)  # 实阳
            prep_cli = sum(predict_y_cli)  # 预测是阳
            tn_cli = sum(1 - (targetsi or predicti) for targetsi, predicti in zip(test_labels, predict_y_cli))  # 真阴
            an_cli = len(test_labels) - sum(test_labels)  # 实阴
            pren_cli = len(test_labels) - sum(predict_y_cli)  # 预测是阴
            num_correct_cli += correct_cli  # 总准确数目
            num_examples_cli += len(test_labels)  # 总样本量
            num_tp_cli += tp_cli  # 总真+
            num_tn_cli += tn_cli  # 总真-
            num_ap_cli += ap_cli  # 总实+
            num_an_cli += an_cli  # 总实—
            num_prep_cli += prep_cli
            num_pren_cli += pren_cli

    val_accurate = acc / test_num
    val_accurate_img = acc_img / test_num
    val_accurate_cli = acc_cli / test_num

    yRecall = num_tp / (num_ap+ 1e-8)  # 召回率
    yPrecision = num_tp / (num_prep+ 1e-8)
    yF1 = 2 * yRecall * yPrecision / (yRecall + yPrecision + 1e-8)
    nRecall = num_tn / (num_an + 1e-8) # 召回率
    nPrecision = num_tn / (num_pren+ 1e-8)
    nF1 = 2 * nRecall * nPrecision / (nRecall + nPrecision + 1e-8)

    yRecall_img = num_tp_img / (num_ap_img+ 1e-8)  # 召回率
    yPrecision_img = num_tp_img / (num_prep_img+ 1e-8)
    yF1_img = 2 * yRecall_img * yPrecision_img / (yRecall_img + yPrecision_img + 1e-8)
    nRecall_img = num_tn_img / (num_an_img + 1e-8) # 召回率
    nPrecision_img = num_tn_img / (num_pren_img+ 1e-8)
    nF1_img = 2 * nRecall_img * nPrecision_img / (nRecall_img + nPrecision_img + 1e-8)

    yRecall_cli = num_tp_cli / (num_ap_cli+ 1e-8)  # 召回率
    yPrecision_cli = num_tp_cli / (num_prep_cli+ 1e-8)
    yF1_cli = 2 * yRecall_cli * yPrecision_cli / (yRecall_cli + yPrecision_cli + 1e-8)
    nRecall_cli = num_tn_cli / (num_an_cli + 1e-8) # 召回率
    nPrecision_cli = num_tn_cli / (num_pren_cli+ 1e-8)
    nF1_cli = 2 * nRecall_cli * nPrecision_cli / (nRecall_cli + nPrecision_cli + 1e-8)
    print('Test Acc: %.3f' %(val_accurate))
    print('Image Acc: %.3f' %(val_accurate_img))
    print('Clinical Acc: %.3f' %(val_accurate_cli))
    print('阳性召回率={:.2f},阳性准确度={:.2f},阳性F1={:.2f} | 阴性召回率={:.2f},阴性准确度={:.2f},阴性F1={:.2f},'.format(
        yRecall, yPrecision, yF1, nRecall, nPrecision, nF1))  
    print('阳性召回率={:.2f},阳性准确度={:.2f},阳性F1={:.2f} | 阴性召回率={:.2f},阴性准确度={:.2f},阴性F1={:.2f},'.format(
        yRecall_img, yPrecision_img, yF1_img, nRecall_img, nPrecision_img, nF1_img))  
    print('阳性召回率={:.2f},阳性准确度={:.2f},阳性F1={:.2f} | 阴性召回率={:.2f},阴性准确度={:.2f},阴性F1={:.2f},'.format(
        yRecall_cli, yPrecision_cli, yF1_cli, nRecall_cli, nPrecision_cli, nF1_cli))  

    with open("./result/fusion.txt", 'w', encoding="utf-8") as f:        
        f.write('Test Acc: %.3f\n' %(val_accurate))
        f.write('阳性召回率={:.2f},阳性准确度={:.2f},yF1={:.2f} | 阴性召回率={:.2f},阴性准确度={:.2f},阴性F1={:.2f},\n'.format(
            yRecall, yPrecision, yF1, nRecall, nPrecision, nF1))
        f.write('label   pre   probability'+'\n')
        for i in range(len(label)):
            f.write(str(label[i])+'   ')
            f.write(str(pre[i])+'   ')
            f.write(str(out[i])+'\n')
          
    with open("./result/image.txt", 'w', encoding="utf-8") as f:        
        f.write('Test Acc: %.3f\n' %(val_accurate_img))
        f.write('阳性召回率={:.2f},阳性准确度={:.2f},yF1={:.2f} | 阴性召回率={:.2f},阴性准确度={:.2f},阴性F1={:.2f},\n'.format(
            yRecall_img, yPrecision_img, yF1_img, nRecall_img, nPrecision_img, nF1_img))
        f.write('label   pre   probability'+'\n')
        for i in range(len(label)):
            f.write(str(label[i])+'   ')
            f.write(str(pre_img[i])+'   ')
            f.write(str(output_img[i])+'\n')

    with open("./result/clinical.txt", 'w', encoding="utf-8") as f:        
        f.write('Test Acc: %.3f\n' %(val_accurate_cli))
        f.write('阳性召回率={:.2f},阳性准确度={:.2f},yF1={:.2f} | 阴性召回率={:.2f},阴性准确度={:.2f},阴性F1={:.2f},\n'.format(
            yRecall_cli, yPrecision_cli, yF1_cli, nRecall_cli, nPrecision_cli, nF1_cli))
        f.write('label   pre   probability'+'\n')
        for i in range(len(label)):
            f.write(str(label[i])+'   ')
            f.write(str(pre_cli[i])+'   ')
            f.write(str(output_cli[i])+'\n')

    with open("./result/node.txt", 'w', encoding="utf-8") as f: 
        for i in range(len(node)):
            f.write(str(node[i].numpy()[:,0,:])+'   '+str(node[i].numpy()[:,196,:])+'\n')
    
if __name__ == '__main__':
    main()
