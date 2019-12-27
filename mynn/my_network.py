#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 15:09:44 2019

@author: andrea
"""

#%% LIBRARIES
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import itertools

#%% My_net CLASS

class My_net():
    
    def __init__(self, layers, activation, regularization = 'no', lmbd=0.5):
        # INITIALIZER FOR A NETWORK    
        # Weight initialization (Glorot-Xavier)
        weights = []
        for i in range(len(layers)-1):
            weights.append((np.random.rand(layers[i+1],layers[i])-0.5) * 
                           np.sqrt(12/(layers[i+1]+layers[i])))
        # Creating biases
        biases = []
        for i in range(1,len(layers)):
            biases.append(np.zeros([layers[i],1]))
        # Joining biases and weights
        self.W=[np.concatenate([weights[i],biases[i]],1) for i 
                in range(len(weights))]
        ### DEFINING ACTIVATION FUNCTIONS
        activations = {'sigmoid':lambda x: expit(x),
                       'ReLU':lambda x: x*(x>=0),
                       'tanh':np.tanh,
                       'leaky':lambda x: x*(x>=0)+0.1*x*(x<0),
                       'ELU':lambda x: x*(x>=0)+0.5*(np.exp(x)-1)}
        
        activations_der = {'sigmoid':lambda x: expit(x)*(1-expit(x)),
                           'ReLU':lambda x: 1*(x>=0),
                           'tanh':lambda x: 1/(np.cosh(x))**2,
                           'leaky':lambda x: 1*(x>=0)+0.1*(x<0),
                           'ELU':lambda x: 1*(x>=0)+0.5*np.exp(x)}
        # Choosing the activation function
        self.act_name = activation
        self.act = activations[activation]
        self.act_der = activations_der[activation]
        # Choosing the regularization
        self.reg = regularization
        self.lmbd = lmbd
        
    def predict(self, x_in, add_out=False):
        # GIVEN A POINT, RETURNS THE PREDICTION OF THE NET
        tmp_res = x_in
        for i in range(len(self.W)-1):
            tmp_res = np.append(tmp_res,1)
            tmp_res = self.act(np.matmul(self.W[i],tmp_res))
        tmp_res = np.append(tmp_res,1)
        if (add_out):
            last_out = tmp_res
        tmp_res = self.act(np.matmul(self.W[-1],tmp_res))
        if (add_out):
            return tmp_res.squeeze(), last_out
        return tmp_res.squeeze()
    
    def update(self, x_in, y_true, l_r):
        # GIVEN A SINGLE INPUT AND THE LEARN RATE, PERFORMS BACKPROPAGATION
        
        # appending the needed 1
        X = np.append(x_in,1)
        # the components of the chain derivatives
        H = []
        Z = [X] 
        for i in range(len(self.W)-1):
            H.append(np.matmul(self.W[i],Z[-1]))
            Z.append(np.append(self.act(H[-1]),1))
        H.append(np.matmul(self.W[-1],Z[-1]))
        # the common factor dl/dy_p
        y_pred = self.predict(x_in)
        dl = y_pred-y_true
        E=[np.array([dl])]
        # building the dW coming from the chain rule. Additional terms for reg
        # are added after.
        for i in range(len(self.W)-1):
            E.append(np.matmul(E[-1],self.W[-1-i][:,:-1])*self.act_der(H[-2-i]))
            
        dW=[]
        for i in range(len(self.W)):
            dW.append(np.matmul(E[-1-i].reshape(-1,1), Z[i].reshape(1,-1)))
        
        if (self.reg=='no'):
            # updating the weights
            self.W = [ ( self.W[i]-l_r*dW[i] ) for i in range(len(dW))]
            loss = ((y_pred-y_true)**2)/2
            return loss
        
        if (self.reg=='L2'):
            # L2 reg
            dW=[(dW[i]+2*self.lmbd*np.absolute(self.W[i])) for i in range(len(dW))]
            # updating the weights
            self.W = [ ( self.W[i]-l_r*dW[i] ) for i in range(len(dW))]
            loss = ((y_pred-y_true)**2)/2 + self.lmbd*sum(
                    [np.sum(np.square(i)) for i in self.W])
            return loss
        
        if (self.reg=='L1'):
            # L1 reg
            dW = [(dW[i]+self.lmbd*(np.sign(self.W[i]))) for i in range(len(dW))]
            # updating the weights
            self.W = [ ( self.W[i]-l_r*dW[i] ) for i in range(len(dW))]
            loss = ((y_pred-y_true)**2)/2+self.lmbd*sum(
                    [np.sum(np.absolute(i))for i in self.W])
            return loss
        
        #if (self.reg=='elastic'):
            # L1-L2 mixed reg
            
    def train(self,x_train,y_train,x_test,y_test,lr,
              num_epochs=1000,en_decay=True,lr_final=0.0001):
        
        lossf = lambda x,y: (x-y)**2/2
        # training the network using early stopping          
        lr_decay = (lr_final / lr)**(1 / num_epochs)
        train_loss_log = np.zeros(num_epochs)
        test_loss_log = np.zeros(num_epochs)            
        
        # actual training
        for i in range(num_epochs):
            # Learning rate decay
            if en_decay:
                lr *= lr_decay
            train_loss_log[i] = np.mean([self.update(x,y,lr) 
                                    for x,y in zip(x_train,y_train)])
            test_loss_log[i] = np.mean([lossf(self.predict(x),y)
                          for x,y in zip(x_test,y_test)])
            print('Epoch %d - lr: %.5f - Train loss: %.5f - Test loss: %.5f' 
          % (i + 1, lr, train_loss_log[i], test_loss_log[i]))
        return train_loss_log,test_loss_log
        
    
    def train_valid(self,x_train,y_train,x_valid,y_valid,x_test,y_test,lr,
              num_epochs=1000,en_decay=True,lr_final=0.0001,
              early_stopping=True, es_wind=15):
        
        lossf = lambda x,y: (x-y)**2/2
        # training the network using early stopping          
        lr_decay = (lr_final / lr)**(1 / num_epochs)
        valid_loss_log = np.zeros(num_epochs)
        train_loss_log = np.zeros(num_epochs)
        test_loss_log = np.zeros(num_epochs)
        # actual training
        for i in range(num_epochs):
            # Learning rate decay
            if en_decay:
                lr *= lr_decay
                
            train_loss_log[i] = np.mean([self.update(x,y,lr) 
                                    for x,y in zip(x_train,y_train)])
            valid_loss_log[i] = np.mean([lossf(self.predict(x),y)
                          for x,y in zip(x_valid,y_valid)])
            test_loss_log[i] = np.mean([lossf(self.predict(x),y)
                          for x,y in zip(x_test,y_test)])
            if (early_stopping and i>es_wind):
                if (np.mean(valid_loss_log[i-es_wind:i])<valid_loss_log[i]):
                    print("Early stop")
                    break
            print('Epoch %d - lr: %.5f - Train loss: %.5f - Test loss: %.5f' 
          % (i + 1, lr, train_loss_log[i], test_loss_log[i]))
        return train_loss_log,valid_loss_log,test_loss_log

    def plot_weights(self):
        fig, axs = plt.subplots(len(self.W), 1, figsize=(8,6))
        for i in range(len(self.W)):
            axs[i].hist(self.W[i].flatten(), 20)
            plt.grid()
        plt.legend()
        plt.grid()
        plt.savefig('weights.pdf')
        plt.show()
    
    def save_weights(self, filename='weights.npz'):
        np.save(filename, self.W)
        print("Weights saved on file: "+filename)
        
    def load_weights(self, filename='weights.npz'):
        loaded_file = np.load(filename)
        self.W = [i for i in loaded_file]
        print("Weights loaded from file: ", filename)
        
    def plot_activation(self, x_input):
        fig, axs = plt.subplots(1, 1, figsize=(8,6))
        _,lastZ = self.predict(x_input,add_out=True)
        axs.stem(lastZ)
        axs.set_title('Last layer activations for input x=%.2f' % x_input)
        plt.tight_layout()
        plt.savefig('activation'+str(x_input)+'.pdf')
        plt.show()

    def save_net(self, filenamew='weights.npz',filenamep='settings.txt'):
        self.save_weights(filenamew)
        with open(filenamep,'w+') as file:
            file.write(str(self.act_name))
            file.write('\n')
            file.write(str(self.reg))
            file.write('\n')
            file.write(str(self.lmbd))
    
#%% UTILITIES FOR CROSS VALIDATION, GRID SEARCH AND RANDOM SEARCH
def kfold_mask(training_set, k):
    mask = np.arange(training_set.shape[0])
    np.random.shuffle(mask)
    mask %= k
    return mask

def grid_search(x_train, y_train, x_test, y_test, l_rate, num_epochs, 
            layers, activation_function, regularization, lmbd, lr_final=None,
            early_stopping=True, num_folds=5, en_decay=True):
   
    # defining the folds
    tr_set = np.concatenate([x_train,y_train])
    tr_set = tr_set.reshape(tr_set.shape[0]//2,2)
    mask = kfold_mask(training_set=tr_set,k=num_folds)
    
    # looping over the params; creating a combination
    combs=list(itertools.product(layers,activation_function,regularization,
                                 lmbd,l_rate,num_epochs))
    combsize=sum(1 for x in combs)
    results = np.zeros(combsize)
    for idx, comb in enumerate(combs):
        print("Current evaluation: ",idx+1," of ",combsize,'\n')
        # looping over the folds
        for i in range(num_folds):
            print("Right now: ",i+1," fold")
            # creating a network with those specifications:
            # comb[0] -> layers
            # comb[1] -> activation function
            # comb[2] -> regularization
            # comb[3] -> lmbd
            net = My_net(comb[0], comb[1], comb[2], comb[3])
            
            # setting the validation and training sets
            X_train = x_train[mask!=i]        
            Y_train = y_train[mask!=i]
            X_valid = x_train[mask==i]
            Y_valid = y_train[mask==i]
            
            # training the network with num_epochs, l_rate
            es_window = 15
            _,_,res=net.train_valid(X_train,Y_train,X_valid,Y_valid,
                   x_test,y_test,comb[4],comb[5],lr_final=comb[4]/comb[5],
                   en_decay=en_decay, early_stopping=early_stopping)
            results[idx] += np.mean(res[-es_window:])
            
    # creating a human-readable dict with the best params
    best_params = combs[np.argmin(results)]
    best_params_dict = {'layers': best_params[0],
                        'activation_function': best_params[1],
                        'regularization': best_params[2],
                        'lambda': best_params[3],
                        'learning_rate': best_params[4],
                        'num_epochs': best_params[5]}
    return best_params_dict
        
def random_search(x_train, y_train, x_test, y_test, num_trials, 
                  l_rate_min, l_rate_max, # min and max learning rates
                  layermin, layermax, # min and max size of each layer
                  lmbd_min, lmbd_max, # min and max lambda for regularization
                  activation_function, # possible act funcs choices
                  regularization, # possible regularization choices
                  num_epochs,
                  lr_final=None, early_stopping=True, num_folds=5, 
                  en_decay=True):
    
    # defining the folds
    tr_set = np.concatenate([x_train,y_train])
    tr_set = tr_set.reshape(tr_set.shape[0]//2,2)
    mask = kfold_mask(training_set=tr_set,k=num_folds)

    # intializing the log of parameters
    results=np.zeros(num_trials)
    l_rates=np.random.uniform(l_rate_min,l_rate_max,num_trials)
    lamb = np.random.uniform(lmbd_min,lmbd_max,num_trials)
    arch = [[np.random.random_integers(i,j) for i,j in 
             zip(layermin,layermax)] for k in range(num_trials)]
    act_func=[np.random.choice(activation_function) for i in range(num_trials)]
    reg=[np.random.choice(regularization) for i in range(num_trials)]
    
    
    for n in range(num_trials):
        print("Trial number: ",n)
        
        for i in range(num_folds):
            print("Right now: ",i+1," fold")
            
            # creating a network with those specifications
            net = My_net(arch[n],act_func[n],reg[n],lamb[n])
        
            # setting the validation and training sets
            X_train = x_train[mask!=i]        
            Y_train = y_train[mask!=i]
            X_valid = x_train[mask==i]
            Y_valid = y_train[mask==i]
            
            # training the network with num_epochs, l_rate
            es_window = 15
            _,_,res=net.train_valid(X_train,Y_train,X_valid,Y_valid,
                   x_test,y_test, l_rates[n], num_epochs, 
                   lr_final=l_rates[n]/num_epochs, en_decay=en_decay, 
                   early_stopping=early_stopping)
            results[n] += np.mean(res[-es_window:])
    
    best_index = np.argmin(results)
    best_params_dict = {'layers': arch[best_index],
                        'activation_function': act_func[best_index],
                        'regularization': reg[best_index],
                        'lambda': lamb[best_index],
                        'learning_rate': l_rates[best_index] }
    return best_params_dict    

#%% TRAINING A NET BY PROVIDING A DICTIONARY

def train_by_dict(x_train, y_train, x_test, y_test, valid_fraction, n_epochs,
                      dictionary, early_stopping=True, en_decay=True):
    
    lr = dictionary['learning_rate']
    layers = dictionary['layers']
    activation_function = dictionary['activation_function']
    regularization = dictionary['regularization']
    lamb = dictionary['lambda']
    lr_final = lr/n_epochs
    
    best_net = My_net(layers,activation_function,regularization,lamb)
    best_net = My_net(layers,activation_function,regularization,lamb)
    
    if (early_stopping):
        #setting the validation up
        num_valid = int(valid_fraction*x_train.shape[0])
        mask = np.zeros(x_train.shape[0])
        mask[:num_valid]=1
        np.random.shuffle(mask)
        X_train = x_train[mask==0]
        Y_train = y_train[mask==0]
        X_valid = x_train[mask==1]
        Y_valid = y_train[mask==1]     
        
        #training
        _,_,res=best_net.train_valid(X_train,Y_train,X_valid,Y_valid,
                   x_test,y_test, lr, n_epochs, 
                   lr_final=lr/n_epochs, en_decay=en_decay, 
                   early_stopping=early_stopping)
    else:
        best_net.train(x_train,y_train,x_test,y_test,lr,
                       n_epochs,lr_final=lr_final)
        
    return best_net