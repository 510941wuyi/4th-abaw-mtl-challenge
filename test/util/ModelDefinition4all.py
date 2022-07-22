from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import *

class BasicNet(nn.Module):

    NUM_CAGETORIES_list = [2, 8, 12]
    
    def __init__(self, output_size=NUM_CAGETORIES_list):
        super(BasicNet, self).__init__()
        self.pretrained_resnet = resnet50(pretrained=True)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', self.pretrained_resnet.conv1),
            ('bn1', self.pretrained_resnet.bn1),
            ('relu', self.pretrained_resnet.relu),
            ('maxpool', self.pretrained_resnet.maxpool),

            ('layer1', self.pretrained_resnet.layer1),
            ('layer2', self.pretrained_resnet.layer2),
            ('layer3', self.pretrained_resnet.layer3),
            ('layer4', self.pretrained_resnet.layer4),

            ('avgpool', nn.AdaptiveAvgPool2d((1, 1))) # 参数(1, 1)表示希望输出的feature map空间方向上的维度为1×1
        ]))

        # dict = torch.load('/data/wuyi/MultiPIE/5view/model/exp/best_model0.pkl')
        # layers = OrderedDict()
        # for k, v in dict.items():
        #     k = k[18:]
        #     if k == "fc.weight":
        #         break
        #     layers[k] = v
        # self.features.load_state_dict(layers)

        # hidden_size_list = []
        cur_size = self.pretrained_resnet.fc.in_features
        layers_va = []
        layers_exp = []
        layers_au = []
        
        # for hidden_size in hidden_size_list:
        #     layers.append(nn.Linear(cur_size, hidden_size, bias=False))
        #     layers.append(nn.BatchNorm1d(hidden_size))
        #     layers.append(nn.ReLU(True))
        #     cur_size = hidden_size
        
        layers_va.append(nn.Linear(cur_size, output_size[0]))
        layers_exp.append(nn.Linear(cur_size, output_size[1]))
        layers_au.append(nn.Linear(cur_size, output_size[2]))
        self.classifier_va = nn.Sequential(*layers_va)
        self.classifier_exp = nn.Sequential(*layers_exp)
        self.classifier_au = nn.Sequential(*layers_au)

        # self.pretrained_vgg = vgg19(pretrained=True)
        # self.features = nn.Sequential(OrderedDict([
        #     ('features', self.pretrained_vgg.features),
        #     ('avgpool', nn.AdaptiveAvgPool2d((7, 7)))
        # ]))

        # hidden_size_list = [4096, 4096]
        # cur_size = 25088
        # layers = []

        # for hidden_size in hidden_size_list:
        #     layers.append(nn.Linear(cur_size, hidden_size, bias=True))
        #     layers.append(nn.ReLU(inplace=True))
        #     layers.append(nn.Dropout(p=0.5, inplace=False))
        #     cur_size = hidden_size
        
        # layers.append(nn.Linear(cur_size, output_size))
        # self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        output_va = self.classifier_va(x)
        output_exp = self.classifier_exp(x)
        output_au = self.classifier_au(x)
        return output_va, output_exp, output_au
    
    def forward_feature(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

    # 参考：https://pytorch.org/docs/master/notes/autograd.html
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True