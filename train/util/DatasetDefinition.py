# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image
from pathlib import Path

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from timm.data import create_transform
from timm.data import transforms as tfs
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def build_dataset(is_train, root):
    transform = build_transform(is_train)

    root = os.path.join(root, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=128,
            is_training=True,
            color_jitter=None,
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    # if args.input_size <= 224:
    #     crop_pct = 224 / 256
    # else:
    #     crop_pct = 1.0
    # size = int(args.input_size / crop_pct)
    # t.append(
    #     transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    # )
    # t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.Resize(128))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class MTL_DataSet(Dataset):
    def __init__(self, lt, dataset_str, img_dir):
        self.lt = lt
        self.dataset_str = dataset_str
        self.img_dir = Path(img_dir)

        if self.dataset_str in ['train']:
            self.preprocess = transforms.Compose([
                tfs.RandomResizedCropAndInterpolation(size=(112, 112), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=PIL.Image.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        elif dataset_str in ['valid', 'test']:
            self.preprocess = transforms.Compose([
                # transforms.Resize(size=256, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.Resize(size=112, interpolation=PIL.Image.BICUBIC, max_size=None, antialias=None),
                transforms.ToTensor(),
                transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
            ])
        else:
            print('Not Existing dataset_str type')
            exit(0)
    
    def __getitem__(self, index):
        img_name = self.lt[index][0]
        img_path = self.img_dir / img_name
        with open(img_path, 'rb') as f:
            X = Image.open(f)
            X = self.preprocess(X)
        y_va = [float(self.lt[index][i]) for i in range(1,3)]
        y_va = torch.tensor(y_va, dtype=torch.float)
        y_exp = int(self.lt[index][3])
        y_au = [int(self.lt[index][i]) for i in range(4,16)]
        y_au = torch.tensor(y_au, dtype=torch.float)

        return X, y_va, y_exp, y_au

    def __len__(self):

        return len(self.lt)
        
