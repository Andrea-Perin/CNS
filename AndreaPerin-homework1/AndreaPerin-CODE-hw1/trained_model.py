#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 22:11:59 2019

@author: andrea
"""

#%%
from my_network import My_net
import numpy as np
import os
#%% GETTING THE RELEVANT INFO

# importing the weights; from the weights, the architecture is inferred
filenamew='weights.npy'
filenamep='net_params.txt'
loaded_file = np.load(filenamew)
layers = [loaded_file[0].shape[1]-1]
for i in loaded_file:
    layers.append(i.shape[0])

with open(filenamep) as f:
    fl = f.readlines()
    act = fl[0][:-1]
    reg = fl[1][:-1]
    lmbd = fl[2][:-1]


#%% CREATING THE NET
    
my_net = My_net(layers, act, reg, lmbd)
my_net.load_weights(filenamew, verbose=False)

#%% IMPORTING THE DATA

# Loading the training and test sets
folder = os.getcwd()+"/"
train_fname = "training_set.txt"
test_fname = "test_set.txt"

train_set = np.loadtxt(folder+train_fname, delimiter=',')
test_set = np.loadtxt(folder+test_fname, delimiter=',')

#%% COMPUTING MSE ON TEST SET AND MISSING SET

sqerr = lambda x,y : (x-y)**2/2

x_test = test_set[:,0]
y_test = test_set[:,1]

test_mse = np.mean([sqerr(my_net.predict(x),y) for x,y in zip(x_test,y_test)])
print(test_mse)

# UNCOMMENT THE FOLLOWING TO INCLUDE THE MISSING DATA

#missing_fname = ""
#missing_set = np.loadtxt(folder+missing_fname, delimiter=',')
#x_miss = missing_set[:,0]
#y_miss = missing_set[:,1]
#miss_mse = np.sum([sqerr(my_net.predict(x),y) for x,y in zip(x_miss,y_miss)])
#print(miss_mse)