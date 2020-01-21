#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:14:12 2019

@author: andrea
"""

import os
import random
from collections import deque

import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
import MNIST_funcs as autoenc

#%% GPU/CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using a GPU!")
else:
    device = torch.device('cpu')
    print('No GPU for you!')

#%% Defining a VAE model
class VAE(nn.Module):

    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim
        
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_mean = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim)
        )
        self.encoder_var = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, z_dim)
        )
                
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(z_dim, 64),
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

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        mean = self.encoder_mean(x)
        logvar = self.encoder_var(x)
        return mean, logvar
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1,784), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

#%% Instantiate the model
model = VAE(z_dim = 2)
model.to(device)
loss_fn = loss_function
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#%% Instantiate the dataset
### filename
matfile = 'MNIST.mat'
### transformations
transform = transforms.Compose([
    autoenc.Rotate(),
    transforms.ToTensor(),
])
dataset = autoenc.from_matlab(matfile, transform=transform)
train_loader, test_loader, valid_loader, _, test_samp, _ = autoenc.ttv_split(dataset=dataset,
                                                                             reuse=True)  

### Training parameters
num_epochs=100
patience=5
early_stopping = autoenc.EarlyStopping(patience=patience)
train_loss = []
valid_loss = []
for epoch in range(1,num_epochs+1):
    ### Training
    model.train()
    avg_train_loss = []
    for idx,(sample_batch,_) in enumerate(train_loader):
        # Extract data and move tensors to the selected device
        image_batch = sample_batch.to(device)
        # Forward pass
        reconstructed_batch, mu, logvar = model(image_batch)
        loss = loss_fn(reconstructed_batch, image_batch, mu, logvar)
        avg_train_loss.append(loss.data.cpu().numpy())
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss.append(np.mean(avg_train_loss))
    ### Validation
    model.eval()
    with torch.no_grad(): # No need to track the gradients
        val_loss = 0
        for idx,(sample_batch,_) in enumerate(valid_loader):
            # Extract data and move tensors to the selected device
            image_batch = sample_batch.to(device)
            # Forward pass
            reconstructed_batch, mu, logvar = model(image_batch)
            val_loss += loss_function(reconstructed_batch, image_batch, mu, logvar).item()
        # Evaluate global loss
        valid_loss.append(val_loss/len(valid_loader.dataset))
    epoch_len = len(str(num_epochs))
    batch_info = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                    f'train_loss: {train_loss[-1]:.5f} ' +
                    f'valid_loss: {valid_loss[-1]:.5f}')
    print(batch_info)
    # earlystop check
    early_stopping(valid_loss[-1],model)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    # load the last checkpoint with the best model, then return it with the
    # average losses
    model.load_state_dict(torch.load('net_params.pth'))
os.rename('net_params.pth','VAE_params.pth')

### Testing
model.eval()
with torch.no_grad(): # No need to track the gradients
    test_loss = 0
    for idx,(sample_batch,_) in enumerate(test_loader):
        # Extract data and move tensors to the selected device
        image_batch = sample_batch.to(device)
        # Forward pass
        reconstructed_batch, mu, logvar = model(image_batch)
        test_loss += loss_function(reconstructed_batch, image_batch, mu, logvar).item()
    # Evaluate global loss
    test_loss/=len(test_loader.dataset)
print("The test loss is: ",test_loss)

#%% Plot samples
import random
### Get the encoded representation of the test samples
encoded_samples = []
for idx, (data,label) in enumerate(test_loader):
    # Encode image
    model.eval()
    with torch.no_grad():
        for (samp,lab) in zip(data,label):
            mu, logvar = model.encode(samp.unsqueeze(0).to(device))
            encoded_img = model.reparameterize(mu, logvar)
            # Append to list
            encoded_samples.append((encoded_img.cpu().numpy().squeeze(), 
                                    lab.item()))

### Visualize encoded space
color_map = {
        0: '#1f77b4',
        1: '#ff7f0e',
        2: '#2ca02c',
        3: '#d62728',
        4: '#9467bd',
        5: '#8c564b',
        6: '#e377c2',
        7: '#7f7f7f',
        8: '#bcbd22',
        9: '#17becf'
        }
    

# Plot just 1k points
encoded_samples_reduced = random.sample(encoded_samples, 1000)
plt.figure(figsize=(12,10))
for enc_sample, label in encoded_samples_reduced:
    plt.plot(enc_sample[0], enc_sample[1], marker='.', color=color_map[label])
# Plot the points from which samples are drawn
x_samples = [0 for i in range(40)]
y_samples = [-1.5+0.1*i for i in range(40)]
plt.scatter(x_samples,y_samples, color = 'k', marker='*')
plt.grid(True)
plt.legend([plt.Line2D([0], [0], ls='', marker='.', color=c, label=l) for 
            l, c in color_map.items()], color_map.keys())
plt.tight_layout()
plt.savefig('VAE_map.pdf')
plt.show()   

### Actual plotting
num_samples = 40
z_dim = 2
if z_dim == 2:
    #%% Generate samples, moving smoothly from a sample
    enc_img = torch.tensor([0.0, -1.5]).float().unsqueeze(0).to(device)
    shift = (torch.tensor([0.0, 0.1])).to(device)
    model.eval()
    with torch.no_grad():
        for idx in range(num_samples):
            enc_img += shift
            img = model.decode(enc_img)
            # Plot the image
            fig,ax = plt.subplots(figsize=(4,4))
            ax.imshow(img.squeeze().cpu().numpy(), cmap='gist_gray')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig('VAE_gen_'+str(idx)+'.pdf')!
            plt.show()




