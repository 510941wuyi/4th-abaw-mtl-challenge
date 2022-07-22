import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from util.DatasetDefinition import MTL_DataSet, build_dataset
from util.ModelDefinition import BasicNet
from util.HelperFunction import model_fit

import warnings
warnings.filterwarnings('ignore')
torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

root = '/data/wuyi/MTL'
train_set_label_path = os.path.join(root, 'training_set_annotations.txt')
valid_set_label_path = os.path.join(root, 'validation_set_annotations.txt')
img_path = os.path.join(root, 'cropped_aligned')

train_set_label = open(train_set_label_path, 'r').readlines()[1:]
valid_set_label = open(valid_set_label_path, 'r').readlines()[1:]

train_list = []
for train_str in train_set_label:
    components = train_str.strip('\n').split(',')
    flag = True
    if float(components[1]) == -5:
        flag = False
    if components[3] == '-1':
        flag = False
    for i in range(4, 16):
        if components[i] == '-1':
            flag = False
            break
    if flag == True:
        train_list.append(components)

valid_list = []
for val_str in valid_set_label:
    components = val_str.strip('\n').split(',')
    flag = True
    if float(components[1]) == -5:
        flag = False
    if components[3] == '-1':
        flag = False
    for i in range(4, 16):
        if components[i] == '-1':
            flag = False
            break
    if flag == True:
        valid_list.append(components)

print(len(train_list), len(valid_list))

train_set = MTL_DataSet(train_list, 'train', img_path)
valid_set = MTL_DataSet(valid_list, 'valid', img_path)

# root = '/amax/wuyi/LSD'

# train_set = build_dataset(is_train=True, root=root)
# valid_set = build_dataset(is_train=False, root=root)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True,num_workers = 8)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False,num_workers = 8)

data = {
    'train': train_loader,
    'valid': valid_loader
}

train_option = {
    'criterion': nn.CrossEntropyLoss().to(device),
    'criterion_cpu': nn.CrossEntropyLoss(),
    'opt_class': optim.Adam,
    'weight_decay': 1e-4,
}

model = BasicNet(output_size=[2, 8, 12]).to(device)
save_path = os.path.join(root, 'output_dir5', 'best_model_pretrain.pkl')
model_fit(model, 5e-5, 30, data, train_option, device, save_path, print_interval=500)
