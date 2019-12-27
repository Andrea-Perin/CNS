# -*- coding: utf-8 -*-

"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Dr. Matteo Gadaleta

Lab. 03 - Introduction to PyTorch (part 1)
 
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

#%% Tensors

x = torch.tensor([1, 2, 3])
print(x)
print(x.type())

x = torch.tensor([1, 2, 3]).float()
print(x)
print(x.type())

x = torch.rand(4, 5)
print(x)

x = torch.zeros(4, 5).long()
print(x)

x = torch.ones(4, 5).double()
print(x)


#%% Numpy bridge

# Define a numpy array
np_x = np.array([1,2,3,4,5], dtype=np.float32)
# Convert to torch tensor
torch_x = torch.from_numpy(np_x)
# Go back to numpy
np_x2 = torch_x.numpy()


#%% Basic operations

a = torch.rand(3, 4)
b = torch.rand(3, 4)

c = torch.add(a, b)
print(c)


#%% Operations on GPU

# Check if a cuda GPU is available
if torch.cuda.is_available():
    # Define the device (here you can select which GPU to use if more than 1)
    device = torch.device("cuda")
    # Move previous a and b tensors to the GPU
    a = a.to(device)
    b = b.to(device)
    # The operation on a and b will be executed on GPU
    c = a + b
    # Move the result tensor back to CPU
    c = c.cpu()
    print(c)
    
    
#%% Autograd
    
# Define operations
x = torch.tensor([2.0], requires_grad=True).float()
y = torch.tensor([3.0], requires_grad=True).float()
z = 3 * x**2 + y**3
# Backward
z.backward()
# Print gradients
print('dz/dx evaluated in %f: %f' % (x, x.grad)) # dz/dx = 6 * x = 12
print('dz/dy evaluated in %f: %f' % (y, y.grad)) # dz/dy = 3 * y^2 = 27


# Define operations
x = torch.tensor([2.0], requires_grad=True).float()
y = 3 * x**2
z = 2 * y**2
# Backward
z.backward()
# Print gradients
print('dz/dx evaluated in %f: %f' % (x, x.grad)) 
# dz/dx =
# = (dz/dy) * (dy/dx) = 
# = (4*y) * (6*x)  = 
# = (4*3*x^2) * (6*x) =
# = 72 * x^3
print(72 * 2**3)



#%%

x = torch.linspace(-10, 10, 1000, requires_grad=True)
y = torch.sigmoid(x)
y.sum().backward()

plt.plot(x.data.numpy(), y.data.numpy())
plt.plot(x.data.numpy(), x.grad.numpy())


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
        noise: enable noisy sampling
    """
    pol_order = len(beta)
    x_matrix = np.array([x**i for i in range(pol_order)]).transpose()
    y_true = np.matmul(x_matrix, beta)
    return y_true
    
### Generate 20 train points
num_train_points = 10
x_train = np.random.rand(num_train_points)
y_train = poly_model(x_train, beta_true)
noise = np.random.randn(len(y_train)) * 0.2
y_train = y_train + noise

### Generate 20 test points
num_test_points = 10
x_test = np.random.rand(num_test_points)
y_test = poly_model(x_test, beta_true)
noise = np.random.randn(len(y_test)) * 0.2
y_test = y_test + noise

### Plot
plt.close('all')
plt.figure(figsize=(12,8))
x_highres = np.linspace(0,1,1000)
plt.plot(x_highres, poly_model(x_highres, beta_true), color='b', ls='--', label='True data model')
plt.plot(x_train, y_train, color='r', ls='', marker='.', label='Train data points')
plt.plot(x_test, y_test, color='g', ls='', marker='.', label='Test data points')
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
Ni = 1
Nh1 = 128
Nh2 = 256
No = 1
net = Net(Ni, Nh1, Nh2, No)

### Define the loss function (the most used are already implemented in pytorch, see the doc!)
loss_fn = nn.MSELoss()

#%% Optimization (manual update)

### Convert to tensor object
x_train = torch.tensor(x_train).float().view(-1, 1)
y_train = torch.tensor(y_train).float().view(-1, 1)
x_test = torch.tensor(x_test).float().view(-1, 1)
y_test = torch.tensor(y_test).float().view(-1, 1)

# Check tensor shapes
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# Learning rate
lr = 0.01
num_epochs = 10000

# Update
train_loss_log = []
test_loss_log = []
net.train()
for num_ep in range(num_epochs):
    # IMPORTANT! zeroes the gradient buffers of all parameters
    net.zero_grad()     
    # Forward pass
    out = net(x_train)
    # Evaluate loss
    loss = loss_fn(out, y_train)
    # Backward pass
    loss.backward()
    # Update
    for p in net.parameters():
        p.data.sub_(p.grad.data * lr)
        
    # Test network
    with torch.no_grad(): # Avoid tracking the gradients (much faster!)
        out = net(x_test)
        test_loss = loss_fn(out, y_test)
    
    # Log
    train_loss_log.append(float(loss.data))
    test_loss_log.append(float(test_loss.data))
    print('Epoch %d - lr: %.5f - Train loss: %.5f - Test loss: %.5f' % (num_ep + 1, lr, float(loss.data), float(test_loss.data)))


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


#%% Optimization (better)

### Reinitialize the network
net = Net(Ni, Nh1, Nh2, No)

### Define the optimization algorithm
optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-3)

# Learning rate
lr = 0.01
num_epochs = 3000

# Update
train_loss_log = []
test_loss_log = []
net.train()
for num_ep in range(num_epochs):
    # IMPORTANT! zeroes the gradient buffers of all parameters
    optimizer.zero_grad()     
    # Forward pass
    out = net(x_train)
    # Evaluate loss
    loss = loss_fn(out, y_train)
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()
        
    # Test network
    with torch.no_grad(): # Avoid tracking the gradients (much faster!)
        out = net(x_test)
        test_loss = loss_fn(out, y_test)
    
    # Log
    train_loss_log.append(float(loss.data))
    test_loss_log.append(float(test_loss.data))
    print('Epoch %d - lr: %.5f - Train loss: %.5f - Test loss: %.5f' % (num_ep + 1, lr, float(loss.data), float(test_loss.data)))

# Plot losses
plt.figure(figsize=(12,8))
plt.semilogy(train_loss_log, label='Train loss')
plt.semilogy(test_loss_log, label='Test loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
    
    
#%% FORWARD PASS (after training)

# Define the input tensor
x_highres = np.linspace(0, 1, 1000)
x_highres = torch.tensor(x_highres).float().view(-1, 1)

# Evaluate the output
net.eval()
with torch.no_grad():
    net_output = net(x_highres)

### Plot
plt.close('all')
plt.figure(figsize=(12,8))
plt.plot(x_highres.flatten().numpy(), poly_model(x_highres.flatten().numpy(), beta_true), color='b', ls='--', label='True data model')
plt.plot(x_train.flatten().numpy(), y_train.flatten().numpy(), color='r', ls='', marker='.', label='Train data points')
plt.plot(x_highres.flatten().numpy(), net_output.flatten().numpy(), color='g', ls='--', label='Network output (trained weights)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
    
    
#%% Accessing network parameters

# First hidden layer
h1_w = net.fc1.weight.data.numpy()
h1_b = net.fc1.bias.data.numpy()

# Second hidden layer
h2_w = net.fc2.weight.data.numpy()
h2_b = net.fc2.bias.data.numpy()

# Output layer
out_w = net.fc3.weight.data.numpy()
out_b = net.fc3.bias.data.numpy()

# Weights histogram
fig, axs = plt.subplots(3, 1, figsize=(12,8))
axs[0].hist(h1_w.flatten(), 50)
axs[0].set_title('First hidden layer weights')
axs[1].hist(h2_w.flatten(), 50)
axs[1].set_title('Second hidden layer weights')
axs[2].hist(out_w.flatten(), 50)
axs[2].set_title('Output layer weights')
[ax.grid() for ax in axs]
plt.tight_layout()
plt.show()


#%% Save network parameters

### Save the network state
# The state dictionary includes all the parameters of the network
net_state_dict = net.state_dict()
# Save the state dict to a file
torch.save(net_state_dict, 'net_parameters.torch')

### Reload the network state
# First initialize the network (if not already done)
net = Net(Ni, Nh1, Nh2, No) 
# Load the state dict previously saved
net_state_dict = torch.load('net_parameters.torch')
# Update the network parameters
net.load_state_dict(net_state_dict)


#%% Analyze actiovations

net.eval()
with torch.no_grad():
    x1 = torch.tensor([0.1]).float()
    y1, z1 = net(x1, additional_out=True)
    x2 = torch.tensor([0.9]).float()
    y2, z2 = net.forward(x2, additional_out=True)
    x3 = torch.tensor([2.5]).float()
    y3, z3 = net.forward(x3, additional_out=True)


fig, axs = plt.subplots(3, 1, figsize=(12,6))
axs[0].stem(z1.numpy())
axs[0].set_title('Last layer activations for input x=%.2f' % x1)
axs[1].stem(z2.numpy())
axs[1].set_title('Last layer activations for input x=%.2f' % x2)
axs[2].stem(z3.numpy())
axs[2].set_title('Last layer activations for input x=%.2f' % x3)
plt.tight_layout()
plt.show()


