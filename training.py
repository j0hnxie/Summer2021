import numpy as np
import torch
from unet import UNet
from dataset import CustomDataset
from trainer import Trainer
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor
from clDice import soft_cldice
import os

def load_data(batch_size = 8):
    inputs = os.listdir('/Users/johnxie/Documents/Summer2021/data/train/train_sample')
    targets = os.listdir('/Users/johnxie/Documents/Summer2021/data/mask/mask_sample') 
    inputs.sort()
    targets.sort()

    for i in range(len(inputs)):
        inputs[i] = '/Users/johnxie/Documents/Summer2021/data/train/train_sample/' + inputs[i]
        targets[i] = '/Users/johnxie/Documents/Summer2021/data/mask/mask_sample/' + targets[i]

    dataset = CustomDataset(inputs=inputs, targets=targets, transform=ToTensor())
    train_size = int(.8 * len(dataset))
    validation_size = len(dataset) - train_size
    training_dataset, validation_dataset = random_split(dataset, [train_size, validation_size])
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=8, shuffle=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=8, shuffle=True)

    return training_dataloader, validation_dataloader

def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def model(classes):
    model = UNet(classes)
    return model

def loss():
    criterion = torch.nn.CrossEntropyLoss()
    return criterion

def optimizer(model):
    optimizer = torch.optim.Adam(model.parameters())
    return optimizer

def training(model, device, loss_fn, optimizer, training_data, validation_data, iterations):
    trainer = Trainer(model=model,
                    device=device,
                    criterion=loss_fn,
                    optimizer=optimizer,
                    training_DataLoader=training_data,
                    validation_DataLoader=validation_data,
                    lr_scheduler=None,
                    epochs=iterations,
                    epoch=0,
                    notebook=True)
    return trainer

# start training
training_data, validation_data = load_data()
device = device()
loss_fn = loss()
unet = model(2)

# from torchinfo import summary
# summary(unet, input_size=(64, 1, 1000, 1000))
# print(unet)

optimizer = optimizer(unet)
trainer = training(unet, device, loss_fn, optimizer, training_data, validation_data, 10)
training_losses, validation_losses, lr_rates = trainer.run_trainer()

from visual import plot_training

fig = plot_training(
    training_losses,
    validation_losses,
    lr_rates,
    gaussian=True,
    sigma=1,
    figsize=(10, 4),
)