#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 13:56:34 2019

@author: andrea
"""

#%% IMPORTING NECESSARY STUFF

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler
from pytorchtools import EarlyStopping


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

#%% IF A GPU CAN BE USED

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using a GPU!")
else:
    device = torch.device('cpu')
    print('No GPU for you!')
    
#%% USING PYTORCH UTILITIES

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
        self.y_train = train_data['output_labels']
        self.transform = transform

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.x_train[idx]#.reshape(28,28)
        label = self.y_train[idx]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def classes(self):
        unq,cnts=np.unique(self.y_train, return_counts=True)
        return unq,cnts
#%% LOADING THE PLAIN DATASET
        
plain_dataset = from_matlab(matfile='MNIST.mat')

fig = plt.figure()

for i in range(len(plain_dataset)):
    sample = plain_dataset[i]

    print(i, sample['image'].shape, sample['label'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    ax.imshow(sample['image'].reshape(28,28))
    
    if i == 3:
        plt.show()
        break
    
#%% CREATING A SIMPLE FEED-FORWARD NETWORK

class Net(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No, act=nn.functional.relu):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)   
        self.dropout = nn.Dropout(0.5)
        self.act = act
        
        
    def forward(self, x, additional_out=False):
        
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        
        if additional_out:
            return out, x
        
        return out


#%% TAKING A LOOK AT THE DATA COMPOSITION; ARE THERE ANY BLATANT UNBALANCES?

labs, cnts = plain_dataset.classes()
cnts/sum(cnts) #no significant unbalances;  all are around 0.1


#%% TRAINING THE NET; DEFINING A FUNCTION 

def train_network(mynet, train_loader, validation_loader, optimizer,
                  patience=20, device=torch.device("cuda"),
                  loss_fn = nn.CrossEntropyLoss()):
    
    # setting the device
    if torch.cuda.is_available():
        device = torch.device("cuda")    
    else:
        device = torch.device('cpu')
    
    #passing to device
    mynet.to(device)
    
    #setting an early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    #setting the logs
    avg_train_loss_log = []
    avg_valid_loss_log = []
    num_epochs = 10
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch: ', epoch+1)

        ############
        # TRAINING #
        ############
        mynet.train()        
        train_loss_log = []
        for i, data in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            inputs=data['image'].float().to(device)
            labels=data['label'].long().squeeze(1).to(device)
            outputs=mynet(inputs).squeeze()
            # loss computation, adding to the log
            loss = loss_fn(outputs, labels)
            train_loss_log.append(loss)
            # backward pass
            loss.backward()
            # optimization step
            optimizer.step()
            
        ##############
        # VALIDATION #
        ##############
        mynet.eval() #evaluation mode
        validation_loss_log = []
        with torch.set_grad_enabled(False):
            for j, vdata in enumerate(validation_loader, 0):               
                # forward pass
                vinputs=vdata['image'].float().to(device)
                vlabels=vdata['label'].long().squeeze(1).to(device)
                voutputs=mynet(vinputs)
                # computing the validation loss, addinng to the log
                vloss=loss_fn(voutputs,vlabels)
                validation_loss_log.append(vloss)
        
        #computing the averages of the losses
        avg_train_loss_log.append(np.mean(train_loss_log))
        avg_valid_loss_log.append(np.mean(validation_loss_log))
        # printing some info on screen
        print('\t Train. loss ():', avg_train_loss_log[-1])
        print('\t Valid. loss ():', avg_valid_loss_log[-1])
        
        # using the eraly stopping mechanism
        early_stopping(avg_valid_loss_log[-1], mynet)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        mynet.load_state_dict(torch.load('checkpoint.pt'))

    return mynet,avg_train_loss_log,avg_valid_loss_log 
    

#%% CREATING A RANDOM SEARCH MECHANISM
    
# creating the folds
def kfold_mask(leng, k):
    mask = np.arange(leng)
    np.random.shuffle(mask)
    mask %= k
    return mask

# resetting the weights
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


# performing kfold cross validation
def kfold_validation(dataset, model, mask, numfold, optimizer):
        
    # batch parameters
    btc_params = {'batch_size':len(dataset)//50,
                  'num_workers':2}
            
    avg_valid=0
    
    for fold in range(numfold):
        
        # resetting the model each time
        model.apply(weight_reset)
        
        # creating the training and validation splits
        train_indices, val_indices = mask[mask==fold], mask[mask!=fold]
        
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        
        # creating appropriate loaders for training/validation
        train_loader = torch.utils.data.DataLoader(plain_dataset,
                                                   **btc_params,
                                                   sampler=train_sampler)
        validation_loader=torch.utils.data.DataLoader(plain_dataset,
                                                      **btc_params,
                                                      sampler=valid_sampler)
        
        # training and getting the lowest value in the validation set
        _,_,whole_avg=train_network(model, train_loader, validation_loader,
                                    optimizer)
        avg_valid += min(whole_avg) 
        
    avg_valid = np.mean(avg_valid)
    
    return avg_valid

#%% CREATING A RANDOM SEARCH

def random_search(dataset, num_trials, 
                  weight_decays, # the lambdas in the L2 regularization
                  lr, #the learning rates
                  layermin, layermax, # min and max size of each layer
                  activation_function, # possible act funcs choices
                  early_stopping=True, num_folds=5):
    
    # defining the folds
    mask = kfold_mask(len(dataset),k=num_folds)

    # intializing the log of parameters
    results=np.zeros(num_trials)
    # creating the grid of hyperparams
    lrs=[np.random.choice(lr) for i in range(num_trials)]
    arch = [[np.random.random_integers(i,j) for i,j in 
             zip(layermin,layermax)] for k in range(num_trials)]
    act_func=[np.random.choice(activation_function) for i in range(num_trials)]
    reg=[np.random.choice(weight_decays) for i in range(num_trials)]
    
    
    for n in range(num_trials):
        
        print("Trial number: ",n)
        
        #defining the architecture
        Ni=28*28
        h1=arch[n][0]        
        h2=arch[n][1]
        No=10 
        
        # defining network and optimizer
        mynet = Net(Ni,h1,h2,No,act_func[n])
        optimizer = torch.optim.Adam(mynet.parameters(), 
                                     lr=lrs[n], weight_decay=reg[n])
        
        # performing kfold cross validation, storing on results
        results[n] = kfold_validation(dataset,mynet,mask,num_folds,optimizer)

    # the chosen combination will be the one which better performs on valid
    best_index = np.argmin(results)

    mynet = Net(Ni,h1,h2,No,act_func[best_index])
    optimizer = torch.optim.Adam(mynet.parameters(), 
                                     lr=lrs[n], weight_decay=reg[best_index])
    
    # returning the network and the optimizer
    return mynet,optimizer    

#%% USING THE BEST NETWORK RETURNED BY THE RANDOM SEARCH
    
numtry=10
w_decays = [1e-1,1e-2,1e-3]
l_rates = [1e-2, 4e-2, 1e-3, 7e-3]
layermin=[100,50]
layermax=[700,500]
act=[nn.functional.relu,nn.functional.sigmoid]


bnet,opt = random_search(plain_dataset,numtry,
                         w_decays, l_rates,
                         layermin, layermax, act)
    
#%% SETTING THE TRAIN AND VALIDATION

# batch parameters
btc_params = {'batch_size':len(plain_dataset)//10,
              'num_workers':2}
        
validation_split = .1
shuffle_dataset = True
random_seed= 43

# Creating data indices for training and validation splits:
dataset_size = len(plain_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(plain_dataset,
                                           **btc_params,
                                           sampler=train_sampler)

validation_loader=torch.utils.data.DataLoader(plain_dataset,
                                              **btc_params,
                                              sampler=valid_sampler)

#%% ACTUAL TRAINING

bestnet = train_network(bnet, train_loader, validation_loader, opt)

#%% MEASURING PERFORMANCE ON TEST SET

#%% INVESTIGATING THE ACTIVATIONS OF THE HIDDEN NEURONS

ideal_img = torch.randn((28*28), requires_grad=True)































#%% CREATING SOME USEFUL TRANSFORMATIONS
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        return {'image': torch.from_numpy(image),
                'label': torch.from_numpy(label)}
        
        
class Rotate(object):
    """Rotate the input images."""

    def __call__(self, sample):
        image = sample['image']
        image = np.rot90(np.fliplr(image.reshape(28,28)))#.copy()
        return {'image': image,
                'label': sample['label']}
    
#%% CREATING A TRANSFORMED DATASET
        
rot_data=from_matlab('MNIST.mat',transform=transforms.Compose([Rotate(),
                                                            ToTensor()]))
    
#%% VISUALIZING SOME OF THE TRAINING DATA 
fig = plt.figure()

for i in range(len(rot_data)):
    sample = rot_data[i]

    print(i, sample['image'].shape, sample['label'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    ax.imshow(sample['image'])
    
    if i == 3:
        plt.show()
        break    
            
#%%CREATING A SIMPLE CONVOLUTIONAL NETWORK

class CNNet(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)   
        self.dropout = nn.Dropout(0.5)
        self.act = nn.functional.relu
        
        
    def forward(self, x, additional_out=False):
        
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        x = self.act(self.fc2(x))
        x = self.dropout(x)
        out = self.fc3(x)
        
        if additional_out:
            return out, x
        
        return out

            
    
#%%
        

        
            
            
            
            
            