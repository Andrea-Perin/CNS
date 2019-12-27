#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:19:49 2019

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
from torchvision import transforms

#%%

class WildeDatasetConverted(Dataset):
    
    def __init__(self, filepath, dictionary, transform=None, thresh=5):
        
        ### Load data
        text = open(filepath, 'r').read()
        
        ### Preprocess data
        # Remove the first and the last part 
        text = re.split('\n{10}', text)[1]
        # Remove spaces after a new line
        text = re.sub('\n[ ]+', '\n', text)
        # Lower case
        text = text.lower()
        #removing excess newlines
        text = re.sub('\n+', ' ', text)
        text = re.sub(' +', ' ', text)
        #then also creating a list of all words
        text=re.sub("chapter [0-9]+", '', text)
        #removing punctuation that (realistically) does not end sentences
        useless_punctuation = string.punctuation.translate(
                string.punctuation.maketrans('','', '!.?'))
        text=text.translate(text.maketrans(useless_punctuation, 
                                           ' '*len(useless_punctuation)))
        text = re.sub(' +', ' ', text)
        #split on ., ! and ?
        sentence_list=re.split('[!.?]',text)
        # every sentence becomes a list of words
        for idx,sent in enumerate(sentence_list):
            #removing empty strings
            contents=sent.translate(sent.maketrans("","", string.punctuation))
            contents = re.split(' ', contents)
            contents = list(filter(None, contents))
            sentence_list[idx]=contents
        # creating the dictionaries to count occurrences                
        vocab_count = Counter()
        text=text.translate(text.maketrans("","", string.punctuation))
        text=re.split(' ',text)
        vocab_count.update(text)
        #saving words and word counts in an appropriate dictionary
        word_idx = dict()
        for idx, (word, count) in enumerate(vocab_count.most_common()):
            word_idx[word]=idx
        #going from word to vector     
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        w2v=torch.load(dictionary, map_location=device)                               
        #turning words into embeddings
        self.sents = list()
        for idx,sent in enumerate(sentence_list):
            wordlist=list()
            for widx,word in enumerate(sent):
                wordlist.append(w2v[word_idx[word]])
            if (len(wordlist)>=thresh):
                self.sents.append(torch.stack(wordlist))
        #saving transforms
        self.transform=transform
        #saving translation for reference
        self.word_index = word_idx
                

    def __len__(self):
        return len(self.sents)
        
    
    def __getitem__(self, idx):
        # Get sonnet text
        sent = self.sents[idx]
        # Transform (if defined)
        if self.transform:
            sent = self.transform(sent)
        return sent
 
    
    
#DRAWING WORDS CLASS
class DrawWords():

    def __init__(self, num_words):
        self.num_words = min(num_words,5)
        
    def __call__(self, sample):
        # Randomly choose an index
        tot_words = len(sample)
        if (tot_words-self.num_words==0):
            start_idx=0
        else:
            start_idx = np.random.randint(0, tot_words - self.num_words)
        end_idx = start_idx + self.num_words
        return sample[start_idx:end_idx]

    
#%% CHECKING HOW THIS WORKS
        
if __name__ == '__main__':
    
    # Initialize dataset
    filepath = 'wilde.txt'
    dictionary='embedding.pt'
    numwords=10
    trans = transforms.Compose([DrawWords(numwords)])
    dataset = WildeDatasetConverted(filepath, dictionary, trans, thresh=5)
    
    #%% Test sampling
    print('Dataset length: ', len(dataset))    
    print('##############')
    print('##############')
    print('SINGLE SENTENCE SHAPE')
    print('##############')
    print(dataset[0].shape)

    #%% Test dataloader
    dataloader = DataLoader(dataset, batch_size=1000, shuffle=True)
    print('##############')
    print('##############')
    print('BATCHES SHAPE')
    print('##############')

    for batch_sample in dataloader:
        batch = batch_sample
        print(batch.shape)

    
