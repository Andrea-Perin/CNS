# -*- coding: utf-8 -*-

"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Dr. Matteo Gadaleta

Lab. 03 - Introduction to PyTorch (part 2)
 
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


#%% Training and test data

# Set random seed
np.random.seed(3)

### Define a simple quadratic model
# y = a + b * x + c * x^2
# a = -1.45, b = 1.12, c = 2.3

beta_true = [-1.45, 1.12, 2.3]
def poly_model(x, beta):
    """
    INPUT
        x: x vector
        beta: polynomial parameters
    """
    pol_order = len(beta)
    x_matrix = np.array([x**i for i in range(pol_order)]).transpose()
    y_true = np.matmul(x_matrix, beta)
    return y_true

def class_sep(x, y, beta):
    """
    INPUT
        x: x vector
        y: y vector
        beta: polynomial parameters
    """
    pol_order = len(beta)
    x_matrix = np.array([x**i for i in range(pol_order)]).transpose()
    y_true = np.matmul(x_matrix, beta)
    return y_true > y
    
### Generate 20 train points
num_train_points = 40
x_train = np.random.rand(num_train_points)
y_train = 2.5*np.random.rand(num_train_points)-1.5
class_train = class_sep(x_train, y_train, beta_true)
y_noise = np.random.randn(len(y_train)) * 0.4
y_train = y_train + y_noise

### Generate 20 test points
num_test_points = 10
x_test = np.random.rand(num_test_points)
y_test = 2.5*np.random.rand(num_test_points)-1.5
class_test = class_sep(x_test, y_test, beta_true)
y_noise = np.random.randn(len(y_test)) * 0.4
y_test = y_test + y_noise

### Plot
plt.close('all')
plt.figure(figsize=(12,8))
x_highres = np.linspace(0,1,1000)
plt.plot(x_highres, poly_model(x_highres, beta_true), color='b', ls='--', label='True data model')
plt.plot(x_train[np.nonzero(class_train)], y_train[np.nonzero(class_train)], color='r', ls='', marker='.', label='Train data points - class 0')
plt.plot(x_train[np.where(class_train==0)], y_train[np.where(class_train==0)], color='b', ls='', marker='.', label='Train data points - class 1')
plt.plot(x_test[np.nonzero(class_test)], y_test[np.nonzero(class_test)], color='y', ls='', marker='.', label='Train data points - class 0')
plt.plot(x_test[np.where(class_test==0)], y_test[np.where(class_test==0)], color='g', ls='', marker='.', label='Train data points - class 1')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


#%% Neural Network

### Define the network class
class Net(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features=Ni, out_features=Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)
        
        self.act = nn.Sigmoid()
        
    def forward(self, x, additional_out=False):
        
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        out = self.fc3(x)
        
        if additional_out:
            return out, x
        
        return out

### Initialize the network
Ni = 2
Nh1 = 24
Nh2 = 12
No = 2
net = Net(Ni, Nh1, Nh2, No)

### Define the loss function (the most used are already implemented in pytorch, see the doc!)
loss_fn = nn.CrossEntropyLoss()


### Define an optimizer
lr = 1e-2
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=5e-4)

### Training
train_loss_log = []
test_loss_log = []
conc_out = torch.Tensor().float()
conc_label = torch.Tensor().long()
num_epochs = 1#200
for num_epoch in range(num_epochs):
    
    print('Epoch', num_epoch + 1)
    # Training
    net.train() # Training mode (e.g. enable dropout)
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    conc_out = torch.Tensor().float()
    conc_label = torch.Tensor().long()
    for i in range(0, num_train_points):
        input_train = torch.tensor([x_train[i],y_train[i]]).float().view(-1, 2)
        label_train = torch.tensor(class_train[i]).long().view(-1, 1).squeeze(1)
        # Forward pass
        out = net(input_train)
        conc_out = torch.cat([conc_out, out])
        conc_label = torch.cat([conc_label, label_train])
        
    # Evaluate loss
    loss = loss_fn(conc_out, conc_label)
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()
    # Print loss
    print('\t Training loss ():', float(loss.data))
        
    # Validation
    net.eval() # Evaluation mode (e.g. disable dropout)
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().long()
        for i in range(0, num_test_points):
            # Get input and output arrays
            input_test = torch.tensor([x_test[i],y_test[i]]).float().view(-1, 2)
            label_test = torch.tensor(class_test[i]).long().view(-1, 1).squeeze(1)
            # Forward pass
            out = net(input_test)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out])
            conc_label = torch.cat([conc_label, label_test])
        # Evaluate global loss
        test_loss = loss_fn(conc_out, conc_label)
        # Print loss
        print('\t Validation loss:', float(test_loss.data))
        
    # Log
    train_loss_log.append(float(loss.data))
    test_loss_log.append(float(test_loss.data))

softmax = nn.functional.softmax(conc_out, dim=1).squeeze().numpy()
errors = conc_label-(conc_out[:,0]<conc_out[:,1]).long()
print('Class probabilities (softmax):\n ', softmax)
print('Real classes: ', conc_label)
print('Errors: ', errors)
        
# Plot losses
plt.close('all')
plt.figure(figsize=(12,8))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(test_loss_log, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

print(np.where(errors==1))
print(np.where(errors==-1))


### Plot
plt.close('all')
plt.figure(figsize=(12,8))
x_highres = np.linspace(0,1,1000)
plt.plot(x_highres, poly_model(x_highres, beta_true), color='b', ls='--', label='True data model')
plt.plot(x_test[np.intersect1d(np.where(errors==0), np.where(conc_label==0))], y_test[np.intersect1d(np.where(errors==0), np.where(conc_label==0))], color='r', ls='', marker='.', label='Test data points - class 0 (correct)')
plt.plot(x_test[np.intersect1d(np.where(errors==0), np.where(conc_label==1))], y_test[np.intersect1d(np.where(errors==0), np.where(conc_label==1))], color='b', ls='', marker='.', label='Test data points - class 1 (correct)')
plt.plot(x_test[np.where(errors==-1)], y_test[np.where(errors==-1)], color='y', ls='', marker='.', label='Test data points - class 0 (misclassified)')
plt.plot(x_test[np.where(errors==1)], y_test[np.where(errors==1)], color='g', ls='', marker='.', label='Test data points - class 1 (misclassified)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()







            
