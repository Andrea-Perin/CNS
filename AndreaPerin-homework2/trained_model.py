#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:14:48 2019

@author: andrea
"""


#%% IMPORTING THE NECESSARY LIBRARIES

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import scipy.io as sio

np.random.seed(43)

#%% DEFINING A SIMPLE NETWORK

### Define the network class
class Net(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No, dropout):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.bn1 = nn.BatchNorm1d(Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.bn2 = nn.BatchNorm1d(Nh2)
        self.fc3 = nn.Linear(Nh2, No)


    def forward(self, X, add_out=False):
        X = F.relu(self.fc1(X))
        X = self.dropout(X)
        X = F.relu(self.fc2(self.bn1(X)))
        X = self.dropout(X)
        X = F.softmax(self.fc3(self.bn2(X)), dim=-1)

        if (add_out):
            return X, np.argmax(X)
        return X

#%% LOADING THE MODEL


bestnet = Net(784,360,187,10,0.5782)

if torch.cuda.is_available():
    bestnet.load_state_dict(torch.load('bnet.pkl'))
else:
    bestnet.load_state_dict(torch.load('bnet.pkl',
                            map_location=lambda storage, loc: storage))

#%% IMPORTING THE DATASET

data = sio.loadmat('MNIST.mat')
X_test=data['input_images'].astype(np.float32)
y_test=data['output_labels'].astype(np.int64)

#%% LOOKING AT THE PERFORMANCE ON THE TEST SET

pred_label = bestnet(torch.tensor(X_test)).float().detach().numpy()
pred_label = np.argmax(pred_label, axis=1)

errors = y_test.squeeze()!=pred_label
print(1-errors.mean())
