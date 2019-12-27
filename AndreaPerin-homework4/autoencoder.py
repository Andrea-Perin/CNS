#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:14:12 2019

@author: andrea
"""

import os

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import random

import torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


#from tqdm import tqdm


#%% GPU/CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using a GPU!")
else:
    device = torch.device('cpu')
    print('No GPU for you!')

#%% Custom MNIST dataset
class from_matlab(Dataset):
    """A custom MNIST dataset."""

    def __init__(self, matfile, transform=None):
        """
        Args:
            matfile (string): Name of the file from which to get the data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        train_data=sio.loadmat(matfile)
        self.x_train = train_data['input_images']
        self.y_train = train_data['output_labels'].astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.x_train[idx].reshape((28,28))
        label = self.y_train[idx][0]
        sample = [image, label]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample
    
    def classes(self):
        unq,cnts=np.unique(self.y_train, return_counts=True)
        return unq,cnts

### Rotation transform
class Rotate(object):
    """Rotate the input images."""

    def __call__(self, sample):
        sample= np.rot90(np.fliplr(sample))
        return sample

#%% Instantiation of the dataset
### filename
matfile = '../datasets/MNIST.mat'
### transformations
transform = transforms.Compose([
    Rotate(),
    transforms.ToTensor(),
])
### instantiation
plain_dataset=from_matlab(matfile,
                          transform=transform)

#%% Dataset composition
labs, cnts = plain_dataset.classes()
cnts/sum(cnts) 

#%% Plot some samples
plt.close('all')
fig, axs = plt.subplots(5, 5, figsize=(8,8))
for ax in axs.flatten():
    img, label = random.choice(plain_dataset)
    ax.imshow(img.squeeze().numpy(), cmap='gist_gray')
    ax.set_title('Label: %d' % label)
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()


#%% Creating test-train split




#%% Define the network architecture
    
class Autoencoder(nn.Module):
    
    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

### Initialize the network
encoded_space_dim = 6
net = Autoencoder(encoded_space_dim=encoded_space_dim)
#%%

### Some examples
# Take an input image (remember to add the batch dimension)
img = plain_dataset[0][0].unsqueeze(0)
print('Original image shape:', img.shape)
# Encode the image
img_enc = net.encode(img)
print('Encoded image shape:', img_enc.shape)
# Decode the image
dec_img = net.decode(img_enc)
print('Decoded image shape:', dec_img.shape)








