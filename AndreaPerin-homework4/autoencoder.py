#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 21:14:12 2019

@author: andrea
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
#from tqdm import tqdm
import MNIST_autoenc as autoenc

#%% GPU/CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using a GPU!")
else:
    device = torch.device('cpu')
    print('No GPU for you!')
    
#%% Creating test-train-validation split
### filename
matfile = '../datasets/MNIST.mat'
### transformations
transform = transforms.Compose([
    autoenc.Rotate(),
    transforms.ToTensor(),
])
dataset = autoenc.from_matlab(matfile, transform=transform)
train_loader, test_loader, valid_loader, _, test_samp, _ = autoenc.ttv_split(dataset=dataset,
                                                                             reuse=True)  

#%% Actual training
enc_space_dims = [i for i in range(2,7)]
performance = []
### Define a loss function
loss_fn = torch.nn.MSELoss()    
### Loop over encoding dimensions
for enc_dim in enc_space_dims:    
    ### Define a network
    net = autoenc.Autoencoder(encoded_space_dim=enc_dim)    
    ### Define an optimizer
    lr = 1e-3 
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    ### send to device
    net.to(device)
    ### training
    num_epochs=100
    patience=5
    net, train_log, valid_log = autoenc.train_network(model=net,
                                                      num_epochs=num_epochs,
                                                      train_loader=train_loader,
                                                      valid_loader=valid_loader,
                                                      loss_fn=loss_fn,
                                                      optimizer=optim,
                                                      device=device,
                                                      patience=patience)
    os.rename('net_params.pth','net_params_'+str(enc_dim)+'.pth')
    ### plot the losses
    plt.close('all')
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(train_log, label="Training loss", color='#4b0082')
    ax.plot(valid_log, label="Validation loss", color='#778899')
    ax.set_title('Loss progression', fontsize=20)
    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.legend()
    plt.savefig('learning_'+str(enc_dim)+'.pdf')
    ### evaluating the reconstruction performance on test set
    performance.append([autoenc.val_epoch(net=net,
                                          dataloader=test_loader,
                                          loss_fn=loss_fn,
                                          optimizer=optim,
                                          device=device)])
### Plot performance on test set
plt.close('all')
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(enc_space_dims, performance, label="Test loss", color='#4b0082')
ax.set_title('Test loss vs. encoding dimension', fontsize=20)
ax.set_xlabel("Encoding dimension", fontsize=20)
ax.set_ylabel("Loss", fontsize=20)
ax.legend()
plt.savefig('performance.pdf')

#%% Performance on modified datasets
### Take the best model
best_net_idx = enc_space_dims[np.argmin(performance)]
best_net = autoenc.Autoencoder(encoded_space_dim=best_net_idx)
best_net.load_state_dict(torch.load('net_params_'+str(best_net_idx)+'.pth'))
### create noisy dataset and evaluate
noise_transform = transforms.Compose([autoenc.Rotate(),
                                            transforms.ToTensor(),
                                            autoenc.GaussNoise(mean=0.0,
                                                               std=0.1),
                                            ])
noisy_dataset = autoenc.from_matlab(matfile, transform=noise_transform)
noise_loader = torch.utils.data.DataLoader(noisy_dataset, 
                                           batch_size=512,
                                           sampler=test_samp,
                                           pin_memory=True)
noise_performance = autoenc.val_epoch(net=net,
                                      dataloader=noise_loader,
                                      loss_fn=loss_fn,
                                      optimizer=optim,
                                      device=device)
### create occluded dataset and evaluate
corrupted_transform = transforms.Compose([autoenc.Rotate(),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(p=1.0,
                                                                   scale=(0.02, 0.1), 
                                                                   ratio=(0.3, 3.3), 
                                                                   value=0),
                                          ])
corrupted_dataset = autoenc.from_matlab(matfile, transform=corrupted_transform)
corrupt_loader = torch.utils.data.DataLoader(corrupted_dataset,
                                             batch_size=512,
                                             sampler=test_samp,
                                             pin_memory=True)
corrupt_performance = autoenc.val_epoch(net=net,
                                        dataloader=corrupt_loader,
                                        loss_fn=loss_fn,
                                        optimizer=optim,
                                        device=device)
### Print performances on screen
print("The performance on noisy samples is: ",noise_performance)
print("The performance on occluded samples is: ",corrupt_performance)






