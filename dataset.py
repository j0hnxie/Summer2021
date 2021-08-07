import os
import pandas as pd
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
import pathlib

class CustomDataset(Dataset):
    def __init__(self,
                 inputs: list,
                 targets: list,
                 transform=None
                 ):
        self.inputs = inputs
        self.targets = targets
        self.transform = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_ID = self.inputs[index]
        target_ID = self.targets[index]

        x, y = io.imread(input_ID), io.imread(target_ID)

        if self.transform is not None:
            x = self.transform(x)
            y = self.transform(y)

        # print(type(x))
        # print(type(y))

        # x = torch.from_numpy(x).type(self.inputs_dtype)
        # y = torch.from_numpy(y).type(self.targets_dtype)

        return x, y

test_data = pd.read_csv("./archive/metadata.csv")
test_data = test_data[(test_data['split'] == 'train')]

inputs = os.listdir('/Users/johnxie/Documents/Summer2021/archive/train')
targets = os.listdir('/Users/johnxie/Documents/Summer2021/archive/mask') 
inputs.sort()
targets.sort()

for i in range(len(inputs)):
    inputs[i] = '/Users/johnxie/Documents/Summer2021/archive/train/' + inputs[i]
    targets[i] = '/Users/johnxie/Documents/Summer2021/archive/mask/' + targets[i]

dataset = CustomDataset(inputs=inputs, targets=targets, transform=ToTensor())
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)


train_features, train_labels = next(iter(dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

plt.figure(1)
img = train_features[0].squeeze()
img = img.permute(1, 2, 0)
plt.imshow(img, cmap="gray")
plt.figure(2)
mask = train_labels[0].squeeze()
mask = mask.permute(1, 2, 0)
plt.imshow(mask, cmap="gray")
plt.show()
