#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 21:27:11 2019

@author: andrea
"""

# -*- coding: utf-8 -*-

import json
import torch
import argparse
from network_LSTM import Network
from pathlib import Path
import numpy as np


##############################
##############################
## PARAMETERS
##############################
parser = argparse.ArgumentParser(description='Generate a chapter starting from a given text')

parser.add_argument('--seed', type=str, default='the', help='Initial text of the chapter')
parser.add_argument('--model_dir',   type=str, default='wilde_model', help='Network model directory')
parser.add_argument('--gen_mode',   type=str, default='argmax', choices=['argmax','softmax'], help='Generation mode')
parser.add_argument('--length',   type=int, default=300, help='Number of words to print')

##############################
##############################
##############################

#levenshtein distance: useful for later!
def LevDist(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]

##############################
##############################
##############################
#%%

if __name__ == '__main__':
    
    ### Parse input arguments
    args = parser.parse_args()
    
    #%% Load training parameters
    model_dir = Path(args.model_dir)
    print ('Loading model from: %s' % model_dir)
    training_args = json.load(open(model_dir / 'training_args.json'))
      
    #%% Load encoder dictionary; then create inverse dictionary
    word_to_number = json.load(open(model_dir / 'word_encoding.json'))
    number_to_word = {v: k for k, v in word_to_number.items()}
    encodings = torch.load('embedding.pt', map_location='cpu')
    norms = {word: torch.norm(encodings[word_to_number[word]]) for 
             word in word_to_number.keys()}
    
    #prediction function: probabilistic
    def pred_word(vec):
        vec=vec.squeeze(0)
        norm_vec = torch.norm(vec)
        dists = []
        idxs = []
        for key,val in number_to_word.items():
            dists.append(torch.sum(vec*encodings[key])/(norm_vec*norms[val]))
            idxs.append(key)
        dists=np.exp(dists)/sum(np.exp(dists))
        out_idx = np.random.choice(idxs, p=dists)
        return out_idx
       
    #prediction function: deterministic
    def pred_word_m(vec):
        vec=vec.squeeze(0)
        norm_vec = torch.norm(vec)
        dists = []
        idxs = []
        for key,val in number_to_word.items():
            dists.append(torch.sum(vec*encodings[key])/(norm_vec*norms[val]))
            idxs.append(key)
        out_idx = np.argmax(dists)
        return idxs[out_idx]

    #%% Initialize network
    net = Network(input_size=256, 
                  hidden_units=training_args['hidden_units'], 
                  layers_num=training_args['layers_num'])
        
    #%% Choose generation mode
    if (args.gen_mode == 'argmax'):
        predict=pred_word_m     
    if (args.gen_mode == 'softmax'):
        predict=pred_word

    #%% Load network trained parameters
    net.load_state_dict(torch.load(model_dir / 'net_params.pth', map_location='cpu'))
    net.eval() # Evaluation mode (e.g. disable dropout)

    #%% Find initial state of the RNN
    with torch.no_grad():
        # Encode seed
        seed_words = args.chapter_seed.split(' ')
        # are all the words in the dictionary? if not, find a substitute...
        for idx,word in enumerate(seed_words):
            if (word in word_to_number.keys()):
                seed_words[idx] = word_to_number[word]
            else:
                distances=[LevDist(word,i) for i in word_to_number.keys()]
                seed_words[idx] = word_to_number[number_to_word[min(distances)]]
        
        seed = torch.stack([encodings[i] for i in seed_words])
        # Add batch axis
        seed = seed.unsqueeze(0)
        # Forward pass
        net_out, net_state = net(seed)
        # Get the most probable last output index
        next_word = predict(net_out[:, -1, :])
        # Print the seed words
        print(args.chapter_seed, end='', flush=True)
        print(" ", end='', flush=True)
        print(number_to_word[next_word], end='', flush=True)
        print(" ", end='', flush=True)

    #%% Generate chapter
    max_word = int(args.length)
    tot_word_count = 0
    while True:
        with torch.no_grad(): # No need to track the gradients
            # The new network input is the encoding of the last chosen word
            net_input = encodings[next_word]
            net_input = net_input.unsqueeze(0)
            net_input = net_input.unsqueeze(0)
            # Forward pass
            net_out, net_state = net(net_input, net_state)
            # Get the most probable word index
            next_word = predict(net_out)
            # Decode the letter
            print(number_to_word[next_word]+' ', end='', flush=True)
            # Count total letters
            tot_word_count += 1
            if tot_word_count > max_word:
                break
        
        
        
        
        
        
        
        
        
        
        
        
