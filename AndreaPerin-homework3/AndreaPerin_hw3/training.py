#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 16:03:52 2019

@author: andrea
"""

# -*- coding: utf-8 -*-

import torch
import json
from torch import optim
from dataset_converted import WildeDatasetConverted, DrawWords
from network_LSTM import Network, train_batch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from time import time


if __name__ == '__main__':
        
    #%% Check device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('Selected device:', device)

    #%% Setting parameters
    args={
        "datasetpath": "wilde.txt",
        "dictionary": "embedding.pt",
        "sentence_len": 3,
        "hidden_units": 512,
        "layers_num": 2,
        "dropout_prob": 0.3,
        "batchsize": 500,
        "num_epochs": 1,
        "out_dir": "wilde_model"}
    
    #%% Create dataset
    corpus=args['datasetpath']
    dictionary=args['dictionary']
    sentence_len=args['sentence_len']
    trans = transforms.Compose([DrawWords(sentence_len)])
    
    dataset = WildeDatasetConverted(filepath=corpus,
                                    dictionary=dictionary,
                                    transform=trans)
    
    #%% Initialize network
    input_size = dataset[0].shape[1] #depends on the embedding dimension of word2vec
    hidden_units = args['hidden_units']
    layers_num = args['layers_num']
    dropout_prob = args['dropout_prob']
    net = Network(input_size=input_size, 
                  hidden_units=hidden_units, 
                  layers_num=layers_num, 
                  dropout_prob=dropout_prob)
    net.to(device)
    #select output files
    out_directory=args["out_dir"]
    
    #%% Train network
    
    # Define Dataloader
    batch_size=args['batchsize']
    dataloader = DataLoader(dataset, 
                            batch_size=batch_size, 
                            shuffle=True, 
                            num_workers=4)
    # Define optimizer
    optimizer = optim.Adam(net.parameters(), weight_decay=5e-4)
    # Define loss function
    loss_fn = net.angle_loss
    
    # Start training
    for epoch in range(args["num_epochs"]):
        print('##################################')
        print('## EPOCH %d' % (epoch + 1))
        print('##################################')
        # Iterate batches
        for batch_sample in dataloader:
            start=time()
            batch_sample.to(device)
            # Update network
            batch_loss = train_batch(net, batch_sample, loss_fn, optimizer)
            print('\t Training loss (single batch):', batch_loss, 
                  '\tElapsed time: ', time()-start)

    ### Save all needed parameters
    # Create output dir
    out_dir = Path(out_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Save network parameters
    torch.save(net.state_dict(), out_dir / 'net_params.pth')
    # Save training parameters
    with open(out_dir / 'training_args.json', 'w') as f:
        json.dump(args, f, indent=4)
    # Save encoder dictionary
    with open(out_dir / 'word_encoding.json', 'w') as f:
        json.dump(dataset.word_index, f, indent=4)
        
    
    
    
    