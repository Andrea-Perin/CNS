#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:07:58 2019

@author: andrea
"""

#%% IMPORTING My_net
import my_network
import os
import matplotlib.pyplot as plt
import numpy as np
#%% IMPORTING DATA
    
# Loading the training and test sets
folder = os.getcwd()+"/"
train_fname = "training_set.txt"
test_fname = "test_set.txt"

train_set = np.loadtxt(folder+train_fname, delimiter=',')
test_set = np.loadtxt(folder+test_fname, delimiter=',')

# Showing both sets
alpha=0.5
plt.close('all')
plt.scatter(train_set[:,0],train_set[:,1], label='Training set', alpha=alpha)
plt.scatter(test_set[:,0],test_set[:,1], label='Test set', alpha=alpha)
plt.legend()
plt.show()

#%% RANDOM SEARCH (early stopping, kfold cross validation assumed)
    
num_trials = 5
# layermin and layermax must have the same length; also, for all i,
# layermin[i]<=layermax[i]
layermin=[1,60,60,1]
layermax=[1,200,200,1]
# the min and max learning rates
lr_min = 0.00001
lr_max = 0.001
# the min and max lambda coefficient for regularization
lamb_min = 0.01
lamb_max = 0.25
# num epochs in the training; do not set this to a large value, otherwise
# the search amy end up being too long (even with early stopping on!)
num_epochs = 150
# the set of activation functions. Possible values: 'sigmoid','tanh','ReLU',
# 'leaky', 'ELU'
activations = [ 'sigmoid','ReLU' ,'ELU' ]
# the set of possible regularizations. Possible values: 'no', 'L1', 'L2'
regularization = [ 'L2', 'no' ]

# creating the training and test sets
x_train = train_set[:,0]
y_train = train_set[:,1]
x_test = test_set[:,0]
y_test = test_set[:,1]

# creating the best parameters dictionary
params = my_network.random_search(x_train, y_train,x_test,y_test,num_trials,
                                  lr_min,lr_max,layermin,layermax,lamb_min,
                                  lamb_max,activations,regularization,
                                  num_epochs)
#%% DISPLAYING THE BEST PARAMETERS FOUND
print(params)

#%% TRAINING A NETWORK WITH THOSE PARAMETERS

np.random.seed(13)
# early stopping may be deactivated in order to guarantee a more thorough 
# training, at the risk of some overfitting if no regularization is present
early_stop = False
# validation fraction to use for early stopping
valid_frac = 0.1
num_epochs = 3000
best_net=my_network.train_by_dict(x_train, y_train,x_test,y_test,valid_frac,
                       num_epochs,params,early_stopping=early_stop)

#%% PLOTTING RESULTING PREDICTION

x_min = min(np.amin(x_train),np.amin(x_test))
x_max = max(np.amax(x_train),np.amax(x_test))
resolution = 1000
xs=np.linspace(x_min,x_max,resolution)
ys=np.array([best_net.predict(x) for x in xs])

plt.close('all')
plt.figure(figsize=(8,6))
plt.scatter(x_train, y_train,color='b',s=1,label='Train data points')
plt.scatter(x_test, y_test, color='r',s=1,label='Test data points')
plt.plot(xs, ys, color='g', ls='--',label='Network prediction (trained)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%% PLOTTING ACTIVATION OF LAST LAYER FOR SOME POINTS

num_check = 5
x_check = np.linspace(x_min,x_max,num_check)
for point in x_check:
    best_net.plot_activation(point)
    
#%% EVENTUALLY SAVING THE WEIGHTS and the info about the net

filenamew = 'weights'
filenamep = 'net_params.txt'
best_net.save_net(filenamew=filenamew,filenamep=filenamep)
