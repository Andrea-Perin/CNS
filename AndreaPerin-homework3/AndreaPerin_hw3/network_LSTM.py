#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:27:22 2019

@author: andrea
"""

from torch import nn
import torch


class Network(nn.Module):
    
    def __init__(self, input_size, hidden_units, layers_num, dropout_prob=0):
        # Call the parent init function (required!)
        super().__init__()
        # Define recurrent layer
        self.rnn = nn.LSTM(input_size=input_size, 
                           hidden_size=hidden_units,
                           num_layers=layers_num,
                           dropout=dropout_prob,
                           batch_first=True)
        # Define output layer
        self.out = nn.Linear(hidden_units, input_size)
        
        
    def forward(self, x, state=None):
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Linear layer
        x = self.out(x)
        return x, rnn_state
    
    
    def angle_loss(self, predictions, labels):
        '''
        Angle loss:
            Given a vector representation of two words, the loss is computed
            as
            
                al(in,out)= 1-[dot_prod(in,out)/(norm(in)*norm(out))],
            
            for each output, as to make full use of the vector representation.
            If a batch is composed of n sentences, the output of this loss will
            be in [0, 1] (it should be in [0,2n], but it is normalized so 
            smaller batches are comparable to large ones).
        '''
        #for each element in a batch, computes the p2 norm
        norm_in=torch.norm(predictions, p=2, dim=1)
        norm_out=torch.norm(labels, p=2, dim=1) 
        #dot product, batchwise
        dotprod=torch.sum(predictions*labels, dim=1)
        #1-cosine as loss, summed over batches
        total_loss=labels.shape[0]-torch.sum(dotprod/(norm_in*norm_out))
        return total_loss/(2*labels.shape[0])

        
    def distance_loss(self, predictions, labels):
        '''
        Distance loss:
            Given a vector representation of two words, the loss is computed
            as
            
                dl(in,out)= sum_i^n [dist(in_i,out_i)]/n,
        '''
        num_words = predictions.shape[0]
        dists = torch.zeros(num_words)
        for i in range(num_words):
            dists[i]=torch.dist(predictions[i],labels[i])        
        return torch.sum(dists)/num_words




def train_batch(net, batch, loss_fn, optimizer):
    
    ### Prepare network input and labels
    # Get the labels (the last word of each sequence)
    #structure of batch: [batch element,word,element of word]
    labels = batch[:, -1, :]
    # Remove the labels from the input tensor
    net_input = batch[:, :-1, :]
    
    ### Forward pass
    # Eventually clear previous recorded gradients
    optimizer.zero_grad()
    # Forward pass
    net_out, _ = net(net_input)
    
    
    ### Update network
    # Evaluate loss only for last output
    loss = loss_fn(net_out[:, -1, :], labels)
    # Backward pass
    loss.backward()
    # Update
    optimizer.step()
    # Return average batch loss
    return float(loss.data)


if __name__ == '__main__':
    
#%% Initialize network
    #importing stuff
    from dataset_converted import WildeDatasetConverted, DrawWords
    from torch.utils.data import DataLoader

    # Initialize dataset
    filepath = 'wilde.txt'
    dictionary='embedding.pt'
    drawer = DrawWords(15)
    dataset = WildeDatasetConverted(filepath, dictionary, drawer)    
    
    #setting sizes
    input_size = dataset[0].shape[1]
    hidden_units = 512
    layers_num = 2
    dropout_prob = 0.3
    net = Network(input_size, hidden_units, layers_num, dropout_prob)
    
    #creating dataloader
    dataloader = DataLoader(dataset, batch_size=52, shuffle=True)
    
    for btc in dataloader:
        batch=btc
        
        
    #%% Test the network output

    out, rnn_state = net(batch)
    print("Shapes and states")
    print(out.shape)
    print(rnn_state[0].shape)
    print(rnn_state[1].shape)
    print(rnn_state[0])
        
    #%% Test network update
    
    optimizer = torch.optim.RMSprop(net.parameters())
    loss_fn = nn.MSELoss()
    
    train_batch(net, batch, loss_fn, optimizer)
        
