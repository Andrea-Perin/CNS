#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:05:50 2019

@author: andrea
"""

#%% TRYING TO IMPLEMENT WORD2VEC

# -*- coding: utf-8 -*-

import torch
import re
import string
from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader
from functools import reduce
from torchvision import transforms

#%%

class WildeDataset(Dataset):
    
    def __init__(self, filepath, transform=None, windsize=5, translate=False):
        
        ### Load data
        text = open(filepath, 'r').read()
        
        ### Preprocess data
        
        
        # Remove the first and the last part 
        text = re.split('\n{10}', text)[1]
        # Remove spaces after a new line
        text = re.sub('\n[ ]+', '\n', text)
        # Lower case
        text = text.lower()
        #removing punctuation
        text=text.translate(text.maketrans("","", string.punctuation))
        #removing excess newlines
        text = re.sub('\n+', ' ', text)
        text = re.sub(' +', ' ', text)

        ### creating the actual dataset
        
        # creating the corpus
        text=re.sub("chapter [0-9]+", '', text)
        corpus = re.split(' ', text)
        #removing empty strings
        corpus = list(filter(None, corpus))
        # creating the dictionaries to count occurrences                
        vocab_count = Counter()
        vocab_count.update(corpus)
        #vocab_cnt = Counter({w:c for w,c in vocab_cnt.items() if c > 2})
        #saving words and word counts in an appropriate dictionary
        word_idx = dict()
        word_count = list()
        vocab=set()
        for idx, (word, count) in enumerate(vocab_count.most_common()):
            word_idx[word]=idx
            word_count.append(count)
            vocab.add(word)
        word_count=np.array(word_count)
        #creating an array with the frequencies of words
        word_frequency = word_count/np.sum(word_count)
        #generating frequencies for negative sampling (as in the paper)
        wordfreq_negsamp = word_frequency**(3/4)
        wordfreq_negsamp = wordfreq_negsamp/np.sum(wordfreq_negsamp) 
        # the probability of dropping words from the dictionary
        drop_prob = 1-np.sqrt(0.00001/word_frequency)
        # creating the end corpus
        train_corpus = [w for w in corpus if w in vocab_count.keys() and 
                        np.random.rand() > drop_prob[word_idx[w]]]
        # creating the pairs of words
        window_size = windsize
        
        #generating the actual dataset (pairs of centers and contexts)
        self.dataset = list()
        for i, w in enumerate(train_corpus):
            window_start = max(i - window_size, 0)
            window_end = min(i + window_size, len(train_corpus))
            for c in train_corpus[window_start:window_end]:
                if c != w:
                    self.dataset.append((word_idx[w], word_idx[c]))
        #converting to tensor, setting transforms
        self.dataset=torch.LongTensor(self.dataset)
        self.transform = transform
        self.translate = translate
        #creating the reverse dictionary (from index to word)
        self.idx_to_word = {v: k for k, v in word_idx.items()}  
        #including the word frequence fir the negative sampling
        self.word_freq_neg_samp = torch.from_numpy(wordfreq_negsamp)
        self.vocab=vocab
        self.word_idx = word_idx
        
        
    def __len__(self):
        return len(self.dataset)
        
    
    def __getitem__(self, idx):
        # Get sonnet text
        pair = self.dataset[idx]
        # Encode with numbers
        center, context = pair[0], pair[1]
        # Create sample
        sample = {'center': center, 'context': context}
        # Transform (if defined)
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
    def to_human(self, idx):
        '''
        To get the actual words from the pair at index idx
        '''
        pair = self.dataset[idx]
        # Encode with numbers
        center, context = pair[0], pair[1]
        sample = {'center': self.idx_to_word[center.item()], 
                  'context': self.idx_to_word[context.item()]}
        return sample            

#%% CHECKING HOW THIS WORKS
        
if __name__ == '__main__':
    
    # Initialize dataset
    filepath = 'wilde.txt'
    dataset = WildeDataset(filepath, translate=True)
    
    #%% Test sampling
    sample = dataset[0]
    
    print('##############')
    print('##############')
    print('CENTER WORD')
    print('##############')
    print(sample['center'])
    
    print('##############')
    print('##############')
    print('CONTEXT WORD')
    print('##############')
    print(sample['context'])

    
    #%% Test dataloader
    dataloader = DataLoader(dataset, batch_size=52, shuffle=True)
    
    for batch_sample in dataloader:
        batch = batch_sample['center']
        print(batch.shape)
    
