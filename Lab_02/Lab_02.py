# -*- coding: utf-8 -*-

"""
NEURAL NETWORKS AND DEEP LEARNING

ICT FOR LIFE AND HEALTH - Department of Information Engineering
PHYSICS OF DATA - Department of Physics and Astronomy
COGNITIVE NEUROSCIENCE AND CLINICAL NEUROPSYCHOLOGY - Department of Psychology

A.A. 2019/20 (6 CFU)
Dr. Alberto Testolin, Dr. Federico Chiariotti

Author: Dr. Matteo Gadaleta

Lab. 02 - Linear regression with artificial neurons
 
"""


#%% Define the true model and generate some noisy samples

import numpy as np
import matplotlib.pyplot as plt
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
print(x_train.shape)
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
plt.plot(x_highres, poly_model(x_highres, beta_true), color='b', ls='--', 
         label='True data model')
plt.plot(x_train, y_train, color='r', ls='', marker='.', label='Train data points')
plt.plot(x_test, y_test, color='g', ls='', marker='.', label='Test data points')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


#%% Activation function

# Define activation function
from scipy.special import expit
act = expit
# 1st derivative
act_der = lambda x: act(x) * (1 - act(x))

# Plot activation function
x_plot = np.linspace(-5,5,1000)
y_act = act(x_plot)
y_act_der = act_der(x_plot)

plt.close('all')
plt.plot(x_plot, y_act, label='Sigmoid function')
plt.plot(x_plot, y_act_der, label='Sigmoid 1st derivative')
plt.legend()
plt.grid()
plt.show()


#%% Network class

class Network():
    
    def __init__(self, Ni, Nh1, Nh2, No):
            
        ### WEIGHT INITIALIZATION (Xavier)
        # Initialize hidden weights and biases (layer 1)
        Wh1 = (np.random.rand(Nh1, Ni) - 0.5) * np.sqrt(12 / (Nh1 + Ni))
        Bh1 = np.zeros([Nh1, 1])
        self.WBh1 = np.concatenate([Wh1, Bh1], 1) # Weight matrix including biases
        # Initialize hidden weights and biases (layer 2)
        Wh2 = (np.random.rand(Nh2, Nh1) - 0.5) * np.sqrt(12 / (Nh2 + Nh1))
        Bh2 = np.zeros([Nh2, 1])
        self.WBh2 = np.concatenate([Wh2, Bh2], 1) # Weight matrix including biases
        # Initialize output weights and biases
        Wo = (np.random.rand(No, Nh2) - 0.5) * np.sqrt(12 / (No + Nh2))
        Bo = np.zeros([No, 1])
        self.WBo = np.concatenate([Wo, Bo], 1) # Weight matrix including biases
        
        ### ACTIVATION FUNCTION
        self.act = expit
        self.act_der = lambda x: act(x) * (1 - act(x))
        
    def forward(self, x, additional_out=False):
        
        # Convert to numpy array
        x = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(x, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        if additional_out:
            return Y.squeeze(), Z2
        
        return Y.squeeze()
        
    def update(self, x, label, lr):
        
        # Convert to numpy array
        X = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(X, 1)
        print(self.WBh1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        # Evaluate the derivative terms
        D1 = Y - label
        D2 = Z2
        D3 = self.WBo[:,:-1]
        D4 = self.act_der(H2)
        D5 = Z1
        D6 = self.WBh2[:,:-1]
        D7 = self.act_der(H1)
        D8 = X
        
        # Layer Error
        Eo = D1
        Eh2 = np.matmul(Eo, D3) * D4
        Eh1 = np.matmul(Eh2, D6) * D7
        
        
        # Derivative for weight matrices
        dWBo = np.matmul(Eo.reshape(-1,1), D2.reshape(1,-1))
        dWBh2 = np.matmul(Eh2.reshape(-1,1), D5.reshape(1,-1))
        dWBh1 = np.matmul(Eh1.reshape(-1,1), D8.reshape(1,-1))
        
        # Update the weights
        self.WBh1 -= lr * dWBh1
        self.WBh2 -= lr * dWBh2
        self.WBo -= lr * dWBo
        
        # Evaluate loss function
        loss = (Y - label)**2/2
        
        return loss
    
    def plot_weights(self):
    
        fig, axs = plt.subplots(3, 1, figsize=(12,6))
        axs[0].hist(self.WBh1.flatten(), 20)
        axs[1].hist(self.WBh2.flatten(), 50)
        axs[2].hist(self.WBo.flatten(), 20)
        plt.legend()
        plt.grid()
        plt.show()

### PARAMETERS
Ni = 1 # Number of inputs
Nh1 = 100# Number of hidden neurons (layer 1)
Nh2 = 100# Number of hidden neurons (layer 2)
No = 1 # Number of outputs

### Initialize network
net = Network(Ni, Nh1, Nh2, No)

# Access the class members
print('1st hidden layer weigth matrix shape:', net.WBh1.shape)
print('2nd hidden layer weigth matrix shape:', net.WBh2.shape)
print('Output layer weigth matrix shape:', net.WBo.shape)

# Plot weights
plt.close('all')
net.plot_weights()

#%% FORWARD PASS (before training)

# Define the x vector
x_highres = np.linspace(0, 1, 1000)

# Evaluate the output for each input (this can be done as a batch, 
#but for now let's do 1 input at a time)
initial_net_output = []
for x in x_highres:
    net_out = net.forward(x)
    initial_net_output.append(net_out)
initial_net_output = np.array(initial_net_output)
    
# Or in just 1 line of pythonic code!!
initial_net_output = np.array([net.forward(x) for x in x_highres])

### Plot
plt.close('all')
plt.figure(figsize=(12,8))
x_highres = np.linspace(0,1,1000)
plt.plot(x_highres, poly_model(x_highres, beta_true), color='b', ls='--', 
         label='True data model')
plt.plot(x_train, y_train, color='r', ls='', marker='.', label='Train data points')
plt.plot(x_highres, initial_net_output, color='g', ls='--', 
         label='Network output (random weights)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

#%% TRAINING

num_epochs = 1000
lr = 0.2
en_decay = True 
lr_final = 0.01
lr_decay = (lr_final / lr)**(1 / num_epochs)

train_loss_log = []
test_loss_log = []
for num_ep in range(num_epochs):
    # Learning rate decay
    if en_decay:
        lr *= lr_decay
    # Train single epoch (sample by sample, no batch for now)
    train_loss_vec = [net.update(x, y, lr) for x, y in zip(x_train, y_train)]
    avg_train_loss = np.mean(train_loss_vec)
    # Test network
    y_test_est = np.array([net.forward(x) for x in x_test])
    avg_test_loss = np.mean((y_test_est - y_test)**2/2)
    # Log
    train_loss_log.append(avg_train_loss)
    test_loss_log.append(avg_test_loss)
    print('Epoch %d - lr: %.5f - Train loss: %.5f - Test loss: %.5f' 
          % (num_ep + 1, lr, avg_train_loss, avg_test_loss))

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

# Plot weights after training
net.plot_weights()


#%% FORWARD PASS (after training)

net_output = np.array([net.forward(x) for x in x_highres])

### Plot
plt.close('all')
plt.figure(figsize=(12,8))
x_highres = np.linspace(0,1,1000)
plt.plot(x_highres, poly_model(x_highres, beta_true), color='b', ls='--', 
         label='True data model')
plt.plot(x_train, y_train, color='r', ls='', marker='.', label='Train data points')
plt.plot(x_highres, net_output, color='g', ls='--', 
         label='Network output (trained weights)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()


#%% Analyze actiovations

x1 = 0.1
y1, z1 = net.forward(x1, additional_out=True)
x2 = 0.9
y2, z2 = net.forward(x2, additional_out=True)
x3 = 2.5
y3, z3 = net.forward(x3, additional_out=True)


fig, axs = plt.subplots(3, 1, figsize=(12,6))
axs[0].stem(z1)
axs[0].set_title('Last layer activations for input x=%.2f' % x1)
axs[1].stem(z2)
axs[1].set_title('Last layer activations for input x=%.2f' % x2)
axs[2].stem(z3)
axs[2].set_title('Last layer activations for input x=%.2f' % x3)
plt.tight_layout()
plt.show()


