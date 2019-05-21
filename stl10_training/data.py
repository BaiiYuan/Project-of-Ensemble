import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from IPython import embed

import numpy as np

def get_train_val_split(batch_size, num_workers=1, val_split=0.2):
    ''' Returns training and validation data loaders.
    batch_size (int): Batch size.
    num_workers (int): How many threads to use to load data. For deterministic behavior, set seed or set to 1.
    val_split (float): What proportion of the STL10 train set to use as validation
    '''
    data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
    ])

    dataset = datasets.stl10.STL10(root="data", split="train", transform=data_transform, download=True)

    # Train / Val Split
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_split * num_train))
    split2 = int(np.floor(val_split * 0.5 * num_train))
    np.random.shuffle(indices)

    # Use first portion as train, second as val
    train_idx, val_idx, val_idx2 = indices[split:], indices[split2:split], indices[:split2]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    val_sampler2 = SubsetRandomSampler(val_idx2)

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
    val_loader2 = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler2)

    return train_loader, val_loader, val_loader2

def get_test(batch_size, num_workers=1):
    ''' Returns testing data loaders.
    batch_size (int): Batch size.
    num_workers (int): How many threads to use to load data. For deterministic behavior, set seed or set to 1.
    '''
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.stl10.STL10(root="data", split="test", transform=data_transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1)

    return test_loader