import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from IPython import embed

import numpy as np

def get_train_val_split(batch_size, model_num, num_workers=1, val_split=0.2):
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
    num_train = len(dataset) # 4000
    indices = list(range(num_train))
    np.random.shuffle(indices)

    train_idx, ensemble_idx = indices[:4000], indices[4000:]
    ensemble_train_idx, ensemble_test_idx = ensemble_idx[:800], ensemble_idx[800:]
    train_idx = [train_idx[i*800:(i+1)*800] for i in range(5)]
    val_idx = train_idx[model_num]
    del train_idx[model_num]
    train_idx = np.array(train_idx).reshape(-1).tolist()

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    ensemble_train_sampler = SubsetRandomSampler(ensemble_train_idx)
    ensemble_val_sampler = SubsetRandomSampler(ensemble_test_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=val_sampler)
    ensemble_train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=ensemble_train_sampler)
    ensemble_val_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, sampler=ensemble_val_sampler)

    return train_loader, val_loader, ensemble_train_loader, ensemble_val_loader

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