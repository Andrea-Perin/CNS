#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 22:02:45 2019

@author: andrea
"""

#%% TRYING TO IMPLEMENT A NETWORK FOR WORD2VEC

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import WildeDataset
import time
import numpy as np

#%%

class w2v(nn.Module):

    def __init__(self,vocabulary_size,embedding_dimension,sparse_grad=False):
        super(w2v, self).__init__()
        # the embedding layers are just that-an embedding
        self.embed_in = nn.Embedding(vocabulary_size, embedding_dimension, 
                                     sparse=sparse_grad)
        self.embed_out = nn.Embedding(vocabulary_size, embedding_dimension, 
                                      sparse=sparse_grad)
        # some random initialization        
        self.embed_in.weight.data.uniform_(-1, 1)
        self.embed_out.weight.data.uniform_(-1, 1)


    def neg_samp_loss(self, in_idx, pos_out_idx, neg_out_idxs):
        '''
        Negative sampling loss:
            in this case, a different loss is used because only a subset
            of the neurons is trained. In particular, for each in_idx,
            we compute the embedding of the central word provided by the 
            network. We then do the same for the context word. We then follow
            the definition of the negative sampling loss, which is something
            like
                Loss=-log(sigma(v'_wo v_wi)) + 
                    +\sum_i^k E_{wi~P(w)}[log(sigma(v'_wi v_wi))]
            
            Note that the weights are not matmultiplied, also the second term
            is produced by stuff drawn from the distribution P(w) of the
            words in the corpus (this is built from the frequencies of the 
            words in the corpus). 
        '''
        emb_in = self.embed_in(in_idx)
        emb_out = self.embed_out(pos_out_idx)
        
        pos_loss = torch.mul(emb_in, emb_out) 
        pos_loss = torch.sum(pos_loss, dim=1)
        pos_loss = F.logsigmoid(pos_loss)
        neg_emb_out = self.embed_out(neg_out_idxs)
        neg_loss = torch.bmm(-neg_emb_out, emb_in.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(neg_loss)
        neg_loss = torch.sum(neg_loss, dim=-1)
        
        total_loss = torch.mean(pos_loss + neg_loss)
        
        return -total_loss       
           
    def forward(self, indices):
        return self.embed_in(indices)
        
    
#%% ACTUAL CREATION OF EMBEDDINGS
        
if __name__ == '__main__':
    
    # using the dataset that is developed in dataset.py
    filename = 'wilde.txt'
    a = WildeDataset(filename)
    # the length of the dataset
    datalen = len(a.vocab)
    # the embedding dimension: how many features per word?
    embdim = 256    
    
    word2vec=w2v(datalen,embdim)
    
    if torch.cuda.is_available():
        word2vec.to(torch.device("cuda"))
        
    # HELPER FUNCTION FOR EXTRACTING NEGATIVE SAMPLES
    def get_negative_samples(batch_size, n_samples):    
        neg_samples = np.random.choice(len(a.vocab), size=(batch_size, n_samples), 
                                   replace=False, p=a.word_freq_neg_samp)
        if torch.cuda.is_available():
            return torch.LongTensor(neg_samples).cuda()
        return torch.LongTensor(neg_samples)

    #setting the optimizer
    optimizer = optim.Adam(word2vec.parameters(), lr=0.003)
            
    #creating the dataloader
    dataloader = DataLoader(a, batch_size=128,
                            shuffle=True, num_workers=4)

    #%% ACTUAL TRAINING AND CREATION OF EMBEDDINGS
    
    n_epochs = 30
    n_neg_samples = 10
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        loss_values = []
        
        start_t = time.time()
        
        for btc in dataloader:
            
            optimizer.zero_grad() # zero the parameter gradients
            
            inputs, labels = btc['center'].cuda(), btc['context'].cuda()
            
            loss = word2vec.neg_samp_loss(inputs, labels, 
                                     get_negative_samples(len(inputs),n_neg_samples))
            loss.backward()
            
            optimizer.step()
            loss_values.append(loss.item())
            
        ellapsed_t = time.time() - start_t
        #if epoch % 1 == 0:
        print("{}/{}\tLoss: {}\tElapsed time: {}".format(epoch + 1, n_epochs, 
                                                          np.mean(loss_values), 
                                                          ellapsed_t))
    print('Done')
    
#%% SOME TESTING
    vocabulary = list(a.vocab)
    
    def print_word_and_embedding(indx):
      maxlen=len(list(a.vocab))
      if (indx>maxlen):
        print("Index too high; maximum value is ", maxlen-1)
      else:
        print(a.idx_to_word[indx])
        print(word2vec.embed_in.weight.data[indx])
    
    print_word_and_embedding(0)
    
#%% SAVING THE WEIGHTS
    
    torch.save(word2vec.embed_in.weight.data, 'embedding.pt')