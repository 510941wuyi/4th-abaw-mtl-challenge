import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tkinter import _flatten

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from util.DatasetDefinition import Test_DataSet
from util.ModelDefinition4all import BasicNet as net4all
from util.ModelDefinition4va import BasicNet as net4va
from util.ModelDefinition4exp import BasicNet as net4exp
from util.ModelDefinition4au import BasicNet as net4au

def model_predict1(model, data_loader, device):
    model.eval()
    y_true, output_label = [], []
    sigmoid = torch.nn.Sigmoid()
    start_index = 0
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(data_loader):
            X_batch = X_batch.to(device)
            # out = model.forward_feature(X_batch)
            # output_feature.append(out.to('cpu'))
            out = model(X_batch)
            out = sigmoid(out)
            output_label.append(out.to('cpu'))
            y_true.append(y_batch)

    y_true, output_label = torch.cat(y_true), torch.cat(output_label)
    return y_true, output_label

def metric_acc(output, y_true):
    output = output.detach().numpy()
    y_true = y_true.detach().numpy()
    # y_pred = output.argmax(axis=1)
    # acc = accuracy_score(y_true, y_pred)
    # f1 = f1_score(y_true, y_pred, average='macro')
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    f1 = f1_score(y_true, output, average='macro')
    
    return f1, output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = '/data/wuyi/MTL/test'
valid_set_label_path = os.path.join(root, 'MTL_Challenge_test_set_release.txt')
img_path = os.path.join(root, 'cropped_aligned')

valid_set_label = open(valid_set_label_path, 'r').readlines()[1:]

valid_list = []
for val_str in valid_set_label:
    components = val_str.strip('\n')
    valid_list.append(components)
print(len(valid_list))

valid_set = Test_DataSet(valid_list, 'valid', img_path)
valid_loader = DataLoader(valid_set, batch_size=128, shuffle=False,num_workers = 8)

train_option = {
    'criterion': nn.CrossEntropyLoss().to(device),
    'criterion_cpu': nn.CrossEntropyLoss(),
    'opt_class': optim.Adam,
    'weight_decay': 1e-4,
}

model4all = net4all(output_size=[2, 8, 12]).to(device)
model4all.load_state_dict(torch.load('/data/wuyi/MTL/output_dir5/best_model_pretrain.pkl'))

model4va = net4va(output_size=2).to(device)
model4va.load_state_dict(torch.load('/data/wuyi/MTL/output_dir4/best_model_pretrain.pkl'))

model4exp = net4exp(output_size=8).to(device)
model4exp.load_state_dict(torch.load('/data/wuyi/MTL/output_dir0/best_model_pretrain.pkl'))

model4au = net4au(output_size=12).to(device)
model4au.load_state_dict(torch.load('/data/wuyi/MTL/output_dir2/best_model_focalloss2.pkl'))

model4all.eval()
model4va.eval()
model4exp.eval()
model4au.eval()

img_names, va, exp, au = [], [], [], []
sigmoid = torch.nn.Sigmoid()
with torch.no_grad():
    for i, (img_name, X_batch) in enumerate(valid_loader):
        X_batch = X_batch.to(device)
        va1, exp1, au1 = model4all(X_batch)
        va2, exp2, au2 = model4va(X_batch), model4exp(X_batch), model4au(X_batch)
        va12 = 0.6 * va1 + 0.4 * va2
        exp12 = 0.4 * exp1 + 0.6 * exp2
        au12 = 0.4 * au1 + 0.6 * au2
        img_names.append(img_name)
        va12 = torch.tanh(va12)
        va.append(va12.to('cpu'))
        exp.append(exp12.to('cpu'))
        au12 = sigmoid(au12)
        au.append(au12.to('cpu'))

va, exp, au = torch.cat(va), torch.cat(exp), torch.cat(au)
img_names = list(_flatten(img_names))
va, exp, au = va.detach().numpy(), exp.detach().numpy(), au.detach().numpy()
exp = exp.argmax(axis=1)
au[au > 0.5] = 1
au[au <= 0.5] = 0

num = len(img_names)
with open('test5.txt', 'w') as f:
    f.write('image,valence,arousal,expression,aus\n')
    for i in range(num):
        tmp=''
        for j in range(12):
            tmp = tmp + str(int(au[i][j])) + ','
        tmp = tmp[:-1]
        f.write('{},{},{},{},{}\n'.format(img_names[i], str(va[i][0]), str(va[i][1]), str(exp[i]), tmp))
