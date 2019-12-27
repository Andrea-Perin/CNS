#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 17:11:37 2019

@author: andrea
"""

#%% IF SKORCH IS TO BE INSTALLED
#!pip install -U skorch


#%% IMPORTING THE NECESSARY LIBRARIES

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler

import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

from skorch import NeuralNetClassifier
from scipy.stats import randint,uniform

np.random.seed(43)

#%% LOADING DATA IN A NAIVE WAY, FOR USING SKORCH

data = sio.loadmat('MNIST.mat')
Xvals=data['input_images'].astype(np.float32)
Yvals=data['output_labels'].astype(np.int64)

# creating a train test split; all classes are balanced, so simple random set
datalen=Xvals.shape[0]
test_frac=0.1
num_test = int(datalen*test_frac)

#creating the indices; these will also be used later in the dataloaders
indices = np.array([1*(i<num_test) for i in range(datalen)])
np.random.shuffle(indices)

# splitting: 0->train, 1->test
X_train, X_test = Xvals[indices==0], Xvals[indices==1]
y_train, y_test = Yvals[indices==0], Yvals[indices==1]

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

#%% WRAPPING STUFF INSIDE SKORCH FOR BETTER RANDOM SEARCH

from skorch.callbacks import EarlyStopping
my_early= EarlyStopping(monitor='valid_loss', patience=3)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


net = NeuralNetClassifier(
    module=Net,
    module__Ni=784,
    module__No=10,
    
    batch_size = 6000,
    max_epochs=100,
    verbose = 1,
    
    lr = 1e-3, 

    optimizer=torch.optim.Adam,
    device=device,
    callbacks=[my_early]
)

#%% DEFINING THE SET OF PARAMS TO RANDOM

tune_params = {
    'module__Nh1': randint(20, 500),
    'module__Nh2': randint(10, 250),
    'module__dropout': uniform(),
    'optimizer__weight_decay': uniform(scale=1e-3)
}

#%% ACTUALLY PERFORMING THE RANDOM SEARCH

rscv = RandomizedSearchCV(net, tune_params, n_iter=5, cv=3, verbose=1)
rscv.fit(X_train, y_train.squeeze())

#%% TAKING THE BEST ESTIMATOR FROM THE RANDOM SEARCH; MODIFY IF NEEDED AND RETRAIN IT

best_net=rscv.best_estimator_
# increasing the number of epochs to train more
best_net.max_epochs=200
# the first hidden layer is set to be the double of the second
# this has proven to be a good choice in the past (still, this is minor:
# the distribution from which Nh1 is taken has already an exepcted value 
# that is two times Nh2)
#best_net.module__Nh1=2*best_net.module__Nh2
# setting a higher patience value for the early stopping
best_net.callbacks[0].patience = 10


best_net.fit(X_train,y_train.squeeze())

#%% SAVING THE PYTORCH MODEL ON FILE

bnet = best_net.module_
torch.save(bnet.state_dict(), "bnet.pkl")

#%% LOADING THE MODEL

bestnet = Net(784,360,187,10,0.5782)
bestnet.load_state_dict(torch.load('bnet.pkl'))

#%% LOOKING AT THE PERFORMANCE ON THE TEST SET

pred_label = bestnet(torch.tensor(X_test)).float().detach().numpy()
pred_label = np.argmax(pred_label, axis=1)

errors = y_test.squeeze()!=pred_label
print('mean accuracy',1-errors.mean())

#%% SHOWING RECEPTIVE FIELD OF A NEURON IN THE FIRST LAYER
neuron_idx = 4

w1=bestnet.fc1.weight.cpu().detach().numpy()
rec_field1=w1[neuron_idx].reshape(28,28)
plt.imshow(rec_field1)


#%% SHOWING RECEPTIVE FIELD OF A NEURON IN THE SECOND LAYER

neuron_idx = 4

w2 = bestnet.fc2.weight.cpu().detach().numpy()
rec_field2 = np.sum([w1[i,:]*w2[neuron_idx,i] for i in range(w2.shape[1])], axis=0)
plt.imshow(rec_field2.reshape(28,28))


#%% SHOWING RECEPTIVE FIELD OF A NEURON IN THE THIRD LAYER

w3 = bestnet.fc3.weight.cpu().detach().numpy()

wh2 = np.array([np.sum([w1[j,:]*w2[i,j] for j in range(w1.shape[0])], axis=0)
                for i in range(w2.shape[0])])

for neuron_idx in range(10):

    rec_field3 = np.sum([wh2[i,:]*w3[neuron_idx,i] for i in 
                        range(w3.shape[1])], axis=0)
    print('\nNeuron index: ',neuron_idx)
    plt.imshow(rec_field3.reshape(28,28))
    plt.show()


#%% TESTING RESULTS ON SOME EXAMPLES

def showpred(model,n):
    model.eval()
    # Get the data from the test set
    x = X_test[n]
    # Get output of network and prediction
    acts = model(torch.from_numpy(x).unsqueeze(0)).detach().numpy().squeeze()
    prediction = np.argmax(acts)
    # Print the prediction of the network
    print('Network output: ')
    print(acts)
    print('Network prediction: ')
    print(prediction)
    print('Actual label: ',y_test[n])
    
    # Draw the image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(np.rot90(np.fliplr(x.reshape(28,28))), cmap='Greys',aspect="auto")
    xs=np.arange(10)
    tics = [str(i) for i in range(10)]
    plt.bar(xs, acts, align='center')
    plt.xticks(xs, tics)
    plt.show()

# getting some indices
err_indices = np.nonzero(y_test.squeeze()!=pred_label)

for id,i in enumerate(err_indices[0]):
    if id<10:
        showpred(bestnet,i)

#%% TRYING TO MAXIMIZE THE OUTPUT OF A NEURON: CURRENTLY NOT WORKING!
    # FOR REASONS I DO NOT UNDERSTAND, THE GRADIENT OF img IS None, EVEN
    #THOUGH require_grad=True and retain_grad() WERE PASSED!

# setting the trained model to evaluation mode
mynet=bestnet
mynet.to('cpu')
mynet.eval()


# defining a class to save hooks at a certain layer
class SaveFeatures():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.features = torch.tensor(output,requires_grad=True).cuda()
    def close(self):
        self.hook.remove()


layer_idx=5 #not 3 since also biases are present
neuron_idx=3
optimization_steps=10



# saving features of neuron in layer_idx
activations = SaveFeatures(list(mynet.children())[layer_idx])


# the image to maximize
img = torch.randn((1,28*28), requires_grad=True)
img.retain_grad()
# an optimizer for the image
optimizer = torch.optim.Adam([img], lr=100)



for n in range(optimization_steps):
    optimizer.zero_grad()
    mynet(img)#.unsqueeze(0))
    # defining the loss
    loss = -activations.features[0, neuron_idx]
    loss.backward()
    optimizer.step()

activations.close()

plt.imshow(img.detach().numpy().reshape(28,28))









