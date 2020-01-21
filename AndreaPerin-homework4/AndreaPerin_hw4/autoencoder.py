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
import MNIST_funcs as autoenc

#%% GPU/CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using a GPU!")
else:
    device = torch.device('cpu')
    print('No GPU for you!')
print()

#%% Creating test-train-validation split
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

#%% Performance on various datasets
### Take the best model
#best_net_idx = enc_space_dims[np.argmin(performance)]
best_net_idx=6
loss_fn = torch.nn.MSELoss()    
lr=1e-3
best_net = autoenc.Autoencoder(encoded_space_dim=best_net_idx)
optim = torch.optim.Adam(best_net.parameters(), lr=lr, weight_decay=1e-5)
best_net.load_state_dict(torch.load('net_params_'+str(best_net_idx)+'.pth'))
best_net.to(device)
### On standard test dataset
test_performance = autoenc.val_epoch(net=best_net,
                                      dataloader=test_loader,
                                      loss_fn=loss_fn,
                                      device=device)
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
noise_performance = autoenc.val_epoch(net=best_net,
                                      dataloader=noise_loader,
                                      loss_fn=loss_fn,
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
corrupt_performance = autoenc.val_epoch(net=best_net,
                                        dataloader=corrupt_loader,
                                        loss_fn=loss_fn,
                                        device=device)
### Print performances on screen
print("The performance on standard test samples is: ",test_performance.item())
print("The performance on noisy samples is: ",noise_performance.item())
print("The performance on occluded samples is: ",corrupt_performance.item())

### Performance against various noise levels
mean = np.linspace(0,2,15)
std = np.linspace(0,2,15)
all_perfs = []
for m,s in zip(mean,std):
    noise_transform = transforms.Compose([autoenc.Rotate(),
                                                transforms.ToTensor(),
                                                autoenc.GaussNoise(mean=m,
                                                                std=s),
                                                ])
    noisy_dataset = autoenc.from_matlab(matfile, transform=noise_transform)
    noise_loader = torch.utils.data.DataLoader(noisy_dataset, 
                                            batch_size=512,
                                            sampler=test_samp,
                                            pin_memory=True)
    noise_performance = autoenc.val_epoch(net=best_net,
                                        dataloader=noise_loader,
                                        loss_fn=loss_fn,
                                        device=device)
    all_perfs.append(noise_performance)
### Plotting
fig, ax = plt.subplots(figsize=(10,8))
ax.tick_params(axis='both', which='major', labelsize=15)
ax.plot(mean, all_perfs, color='#4b0082')
ax.set_title('Test loss vs. noise level', fontsize=20)
ax.set_xlabel('Mean and variance', fontsize=20)
ax.set_ylabel('Loss', fontsize=20)
plt.grid()
plt.tight_layout()
plt.savefig('noise_perf.pdf')
plt.show()

### Sampling function
def sample_around(centroid, model, mean=0.0, var=1.0):
    '''
    Returns a sample drawn around the centroid.
    PARAMETERS:
        - centroid:     the centroid in the encoding space, around which to sample
                        according to a multivariate normal distribution.
        - model:        the autoencoder that generates the encodings.
        - mean:         the mean of the gaussian.
        - var:          the variance of the gaussian.
    '''
    device = centroid.device
    centroid = centroid + mean + var*torch.randn(model.enc_dim).to(device)
    return model.decode(centroid)

# Get centroids
centroids = [autoenc.find_centroid(idx, best_net, train_loader, device) for idx in range(10)]

#%% Plot samples drawn around centroids
sigma = np.linspace(0, 10, num=5)
mean = np.zeros_like(sigma)
for m,s in zip(mean,sigma):
    imgs = []
    for c in centroids:
        img=sample_around(c,best_net, mean=m, var=s)
        imgs.append(img.cpu().detach().numpy().squeeze())
    plt.close('all')
    fig, axs = plt.subplots(2, 5, figsize=(8,3))
    fig.suptitle('Mean: '+str(m)+', var: '+str(s), fontsize=20, y=1.04)
    for idx,ax in enumerate(axs.flatten()):
        ax.imshow(imgs[idx], cmap='gist_gray')
        ax.set_title('Sampled: '+str(idx))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig('sampled_var_'+str(s)+'.pdf')
    plt.show()
    print()

### Draw samples from 2-dimensional autoencoder
net_idx=2
loss_fn = torch.nn.MSELoss()    
lr=1e-3
net = autoenc.Autoencoder(encoded_space_dim=net_idx)
optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
net.load_state_dict(torch.load('net_params_'+str(net_idx)+'.pth'))
net.to(device)
# Plot points
import random
### Get the encoded representation of the test samples
encoded_samples = []
for idx, (data,label) in enumerate(train_loader):
    # Encode image
    net.eval()
    with torch.no_grad():
        for (samp,lab) in zip(data,label):
            enc_img = net.encode(samp.unsqueeze(0).to(device))
            encoded_samples.append((enc_img.cpu().numpy().squeeze(), 
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
x_samples = [5-0.2*i for i in range(40)]
y_samples = [5-0.2*i for i in range(40)]
plt.scatter(x_samples,y_samples, color = 'k', marker='*')
plt.grid(True)
plt.legend([plt.Line2D([0], [0], ls='', marker='.', color=c, label=l) for 
            l, c in color_map.items()], color_map.keys())
plt.tight_layout()
plt.savefig('AE_map.pdf')
plt.show()   

### Plot images
points = np.vstack((x_samples,y_samples)).T
points = torch.from_numpy(points).to(device).double()
net = net.double()
net.eval()
with torch.no_grad():
    for idx,pt in enumerate(points):
        img = net.decode(pt)
        # Plot the image
        fig,ax = plt.subplots(figsize=(4,4))
        ax.imshow(img.squeeze().cpu().numpy(), cmap='gist_gray')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.savefig('AE_gen_'+str(idx)+'.pdf')
        plt.show()



