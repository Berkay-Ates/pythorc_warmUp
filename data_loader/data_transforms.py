from typing import Any
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

# gradient computation etc. not efficient for whole data set
# -> divide dataset into small batches

"""
# training loop
for epoch in range(num_epochs):
    # loop over all batches
    for i in range(total_batches):
        batch_x, batch_y = ...
"""

# epoch = one forward and backward pass of ALL training samples
# batch_size = number of training samples used in one forward/backward pass
# number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes
# e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch

# --> DataLoader can do the batch computation for us

# Implement a custom Dataset:
# inherit Dataset
# implement __init__ , __getitem__ , and __len__


class WineDataset(Dataset):
    def __init__(self, transform=None):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt("./wine.csv", delimiter=",", dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = xy[:, 1:]  # size [n_samples, n_features]
        self.y_data = xy[:, [0]]  # size [n_samples, 1]
        self.transform = transform

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, target = sample
        inputs *= self.factor
        return inputs, target


dataset = WineDataset(transform=ToTensor())

first_data = dataset[0]
features, labes = first_data
print(features)
print(f"type {type(features)}, type {type(labes)}")


composed = torchvision.transforms.Compose([ToTensor(), MulTransform(factor=2)])
dataset = WineDataset(transform=composed)

first_data = dataset[0]
features, labes = first_data
print(features)
print(f"type {type(features)}, type {type(labes)}")
