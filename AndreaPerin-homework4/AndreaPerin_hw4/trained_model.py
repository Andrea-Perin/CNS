#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:48:40 2020

@author: andrea
"""

import argparse
import torch
import MNIST_autoenc as MNIST
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms


#%%
### Parse input parameters
parser = argparse.ArgumentParser(description='MNIST Auto-Encoder')
parser.add_argument('--enc_dim', type=int, default=6, choices=[2,3,4,5,6], 
                    help='Encoding dimension of the AE (default: 6). Possible'+
                    ' choices: 2,3,4,5,6.')
args = parser.parse_args()
enc_dim = args.enc_dim

### Create model and load the weights
PATH = 'weights/net_params_'+str(enc_dim)+'.pth'
model = MNIST.Autoencoder(encoded_space_dim = enc_dim)
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()
loss_fn = torch.nn.MSELoss() 

### Load the MNIST file over which to perform stuff
MATFILE = 'MNIST.mat'
transf = transforms.Compose([MNIST.Rotate(),
                             transforms.ToTensor(),        
        ])
dataset = MNIST.from_matlab(MATFILE, 
                            transform=transf)
data_loader = torch.utils.data.DataLoader(dataset,
                                          batch_size=512)
performance = MNIST.val_epoch(net=model,
                              dataloader=data_loader,
                              loss_fn=loss_fn,
                              device=torch.device('cpu'))
print(performance)    
