#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:09:44 2019

@author: andrea
"""

import os

import matplotlib.pyplot as plt
import numpy as np

import torch
from torch.utils.data import DataLoader,Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
#from tqdm import tqdm
import MNIST_autoenc as autoenc

#%% GPU/CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using a GPU!")
else:
    device = torch.device('cpu')
    print('No GPU for you!')
print()

#%% 
