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

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch
from timm.data import create_transform
from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import random_split, Dataset


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DATA_SET_MEAN = (0.56101511, 0.57580587, 0.543732820)
DATA_SET_STD = (0.24505545, 0.2083447,  0.22679123)

# ----------------------- Build Transform --------------------------
def build_transform(is_train, args):
    mean = DATA_SET_MEAN
    std = DATA_SET_STD

    if is_train == 'train':
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
    else:
        t = []
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
        size = int(args.input_size / crop_pct)
        t.append(
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC), 
        )
        t.append(transforms.CenterCrop(args.input_size))
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
        transform = transforms.Compose(t)
    return transform

# ----------------------- Build Trwnsform Pretrain --------------------

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
DATA_SET_MEAN = (0.5010, 0.5117, 0.4860)
DATA_SET_STD = (0.1585, 0.1577, 0.1681)

def build_transform_pretrain(args):
    mean = DATA_SET_MEAN
    std = DATA_SET_STD


    # Update the train transformation with the new specification
    transform = transforms.Compose([
        transforms.CenterCrop(args.input_size),
        transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
 

    return transform

# ----------------------- Build Classes --------------------------
class MAEDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print('idx', idx)
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_dir, row['image'])
        image = Image.open(image_path).convert('RGB')
        raw_img = image.copy()

        if self.transform:
            image = self.transform(image)

        image = image.to(torch.float)
        labels = row['label']
        labels = torch.tensor(labels, dtype=torch.float)
        return image, labels


#-------------------------- Finetuning -------------------------
def build_dataset_new(args):
    csv_file = os.path.join(args.data_path, 'regression_dataset.csv')
    image_dir = os.path.join(args.data_path, 'VLM_images')
    
    # Read the Excel file using openpyxl engine
    data = pd.read_csv(csv_file)

    # Split the dataset
    train_size = int(0.8 * len(data))
    val_size = int(0.15 * len(data))
    test_size = len(data) - train_size - val_size
    
    data_train, data_val, data_test = random_split(data.index.tolist(), [train_size, val_size, test_size])
    # import pdb
    # pdb.set_trace
    transform_train = build_transform(is_train='train', args=args)
    transform_eval = build_transform(is_train='eval', args=args)
    
    dataset_train = MAEDataset(data.iloc[data_train.indices], image_dir, transform=transform_train)
    dataset_val = MAEDataset(data.iloc[data_val.indices], image_dir, transform=transform_eval)
    dataset_test = MAEDataset(data.iloc[data_test.indices], image_dir, transform=transform_eval)
    
    return dataset_train, dataset_val, dataset_test

#------------------------------ Pretraining -------------------------------------
def build_dataset_pretrain(args):
    csv_file = os.path.join(args.data_path, 'regression_dataset.csv')
    image_dir = os.path.join(args.data_path, 'VLM_images')
    
    # Read the Excel file using openpyxl engine
    data = pd.read_csv(csv_file)

    # Split the dataset
    train_size = int(1 * len(data))
    val_size = int(0 * len(data))
    test_size = len(data) - train_size - val_size
    
    data_train, data_val, data_test = random_split(data.index.tolist(), [train_size, val_size, test_size])
    # import pdb
    # pdb.set_trace
    transform_train = build_transform_pretrain(args=args)
    transform_eval = build_transform_pretrain(args=args)
    
    dataset_train = MAEDataset(data.iloc[data_train.indices], image_dir, transform=transform_train)
    dataset_val = MAEDataset(data.iloc[data_val.indices], image_dir, transform=transform_eval)
    dataset_test = MAEDataset(data.iloc[data_test.indices], image_dir, transform=transform_eval)
    
    return dataset_train, dataset_val, dataset_test




#----------------------- Old functions --------------------------
# def build_dataset(is_train, args):
#     transform = build_transform(is_train, args)

#     root = os.path.join(args.data_path, 'train' if is_train else 'val')
#     dataset = datasets.ImageFolder(root, transform=transform)

#     print(dataset)

#     return dataset


# def build_transform(is_train, args):
#     mean = IMAGENET_DEFAULT_MEAN
#     std = IMAGENET_DEFAULT_STD
#     # train transform
#     if is_train:
#         # this should always dispatch to transforms_imagenet_train
#         transform = create_transform(
#             input_size=args.input_size,
#             is_training=True,
#             color_jitter=args.color_jitter,
#             auto_augment=args.aa,
#             interpolation='bicubic',
#             re_prob=args.reprob,
#             re_mode=args.remode,
#             re_count=args.recount,
#             mean=mean,
#             std=std,
#         )
#         return transform

#     # eval transform
#     t = []
#     if args.input_size <= 224:
#         crop_pct = 224 / 256
#     else:
#         crop_pct = 1.0
#     size = int(args.input_size / crop_pct)
#     t.append(
#         transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
#     )
#     t.append(transforms.CenterCrop(args.input_size))

#     t.append(transforms.ToTensor())
#     t.append(transforms.Normalize(mean, std))
#     return transforms.Compose(t)
