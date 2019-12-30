from collections import deque

import scipy.io as sio
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler






#%% Custom MNIST dataset
class from_matlab(Dataset):
    """A custom MNIST dataset."""

    def __init__(self, matfile, transform=None):
        """
        Args:
            matfile (string): Name of the file from which to get the data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        train_data=sio.loadmat(matfile)
        self.x_train = train_data['input_images']
        self.y_train = train_data['output_labels'].astype(int)
        self.transform = transform

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.x_train[idx].reshape((28,28))
        label = self.y_train[idx][0]
        sample = [image, label]
        if self.transform:
            sample[0] = self.transform(sample[0])
        return sample
    
    def classes(self):
        unq,cnts=np.unique(self.y_train, return_counts=True)
        return unq,cnts

#%% Useful transformations
class Rotate(object):
    """Rotate the input images."""

    def __call__(self, sample):
        sample= np.rot90(np.fliplr(sample))
        return sample



class GaussNoise(object):
    """Add Gaussian noise with given mean and variance to the images."""
    
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
    
    def __call__(self, sample):
        noise = self.mean+self.std*torch.randn(sample.size())
        return sample+noise



class Occlude(object):
    """Add random occlusions to the images. Based on the paper at
    https://arxiv.org/pdf/1708.04896v1.pdf"""

    def __init__(self, p=0.5, s_min=0.02, s_max=0.04, r_min=0.3):
        self.p = p  # erasing probability
        self.s_min = s_min # lower bound for area ratio of occlusion rectangle
        self.s_max = s_max # upper bound for area ratio of occlusion rectangle
        self.r_min = r_min # lower bound for aspect ratio of occlusion rectangle
        self.r_max = 1./r_min # upper bound for aspect ratio of occlusion rectangle

    def __call__(self, tensor):
        if np.random.uniform() > self.p:
            return tensor # unmodified return
        else:
            # area of the rectangle
            Se = np.random.uniform(self.s_min,self.s_max) * tensor.nelement() 
            # aspect ratio of the rectangle
            re = np.random.uniform(self.r_min, self.r_max)
            # dimensions of the rectangle
            He, We = np.sqrt(Se * re), np.sqrt(Se/re)
            # width and height of image
            W, H = tensor.shape[1], tensor.shape[0]
            # location of the rectangle. It must be correct
            xe, ye = int(np.random.uniform(0,W-We)), int(np.random.uniform(0,H-He))
            # filling with random value
            tensor[ ye:ye + int(He), xe:xe + int(We)].fill_(np.random.uniform())
            return tensor


#%% Early-stopping utility
class EarlyStopping:
    """
    Early stops the training if validation loss 
    doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation 
                            loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation 
                            loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to 
                            qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} '+
                  '--> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'net_params.pth')
        self.val_loss_min = val_loss



#%% Define the network architecture
class Autoencoder(nn.Module):

    def __init__(self, encoded_space_dim):
        super().__init__()
        
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


#%% Network training functions

### Training function
def train_epoch(net, dataloader, loss_fn, optimizer, device):
    '''
    This function trains a model on a training set.
    Returns the average loss during an epoch on the training set.
    PARAMETERS:
    - net:             a Pytorch network
    - dataloader:    a Pytorch DataLoader for the training set
    - loss_fn:        the loss function to be used for the computation
    - optimizer:     the optimizer to use
    - device:      the device to be used
    '''
    # Training
    net.train()
    avg_train_loss = []
    for idx,sample_batch in enumerate(dataloader):
        # Extract data and move tensors to the selected device
        image_batch = sample_batch[0].to(device)
        # Forward pass
        output = net(image_batch)
        loss = loss_fn(output, image_batch)
        avg_train_loss.append(loss.data.cpu().numpy())
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return np.mean(avg_train_loss)


### Validation function
def val_epoch(net, dataloader, loss_fn, optimizer, device):
    '''
    This function evaluates the performance of a model on a validation set.
    Returns the loss on the validation set.
    PARAMETERS:
    - net:         a Pytorch network
    - dataloader:  a Pytorch DataLoader for the validation set
    - loss_fn:     the loss function to be used for the computation
    - optimizer:   the optimizer to use
    - device:      the device to be used
    '''
    net.eval()
    with torch.no_grad(): # No need to track the gradients
        conc_out = torch.Tensor().float()
        conc_label = torch.Tensor().float()
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            out = net(image_batch)
            # Concatenate with previous outputs
            conc_out = torch.cat([conc_out, out.cpu()])
            conc_label = torch.cat([conc_label, image_batch.cpu()]) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def train_network(model, num_epochs, train_loader, valid_loader, loss_fn, 
                  optimizer, device, patience=5):
    '''
        This function trains a model, using a validation set in addition
        to a training set. Returns a trained model, together with the logs 
        of the training loss and the validation loss.
        PARAMETERS:
        - model:         the Pytorch network to train
        - num_epochs:    the maximum number of epochs for which to train the model 
        - train_loader: the DataLoader for the training set
        - valid_loader: the DataLoader for the validation set
        - loss_fn:        the loss function to use for the training
        - optimizer:    the optimizer to use for the training
        - device:       the device to use
        - patience:     the patience parameter for the EarlyStopping
    '''
    # earlystopping class
    early_stopping = EarlyStopping(patience=patience)
    # logs for losses
    train_loss = deque()
    valid_loss = deque()
    
    for epoch in range(1,num_epochs+1):
        # training step
        train_loss.append(train_epoch(model, 
                                      train_loader, 
                                      loss_fn, 
                                      optimizer,
                                      device))
        # validation step
        valid_loss.append(val_epoch(model,
                                    valid_loader,
                                    loss_fn,
                                    optimizer,
                                    device))
        # print info on batch losses
        epoch_len = len(str(num_epochs))
        batch_info = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss[-1]:.5f} ' +
                     f'valid_loss: {valid_loss[-1]:.5f}')
        print(batch_info)
        # earlystop check
        early_stopping(valid_loss[-1],model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # load the last checkpoint with the best model, then return it with the
    # average losses
    model.load_state_dict(torch.load('net_params.pth'))
    return  model, train_loss, valid_loss




#%% Train-test-validation split
def ttv_split(dataset, batch_size=512, test_split=8.0/60, valid_split=2.0/60, 
              shuffle=True, random_seed=43, reuse=False):
    '''
        This function performs a train-test-validation split on a given 
        dataset. The split is performed by creating three different dataloaders,
        from which the train, test and validation samples can be drawn.
        PARAMETERS:
        - dataset:         a Pytorch Dataset object
        - batch_size:     the number of samples per batch for each of the loaders
        - test_split:    the fraction of the dataset to be used as test
        - valid_split:    the fraction of the dataset to be used as validation
        - shuffle:      a bool value, whether to shuffle the dataset before splitting it
        - random_seed:    the random seed to use for shuffling
        - reuse:          whether to return the subsamplers (so to reuse the splits later)
    '''    
    ### preliminary information
    dataset_size = len(dataset)
    batch_size = batch_size
    test_split = test_split
    valid_split = valid_split
    shuffle_dataset = shuffle
    random_seed= random_seed

    # Creating data indices for training and test splits:
    indices = list(range(dataset_size))
    split_test = int(np.floor( test_split * dataset_size))
    split_valid = int(np.floor( valid_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices = indices[split_test+split_valid:]
    test_indices = indices[split_valid:split_test+split_valid]
    valid_indices = indices[:split_valid]

    # Creating pytorch data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size, 
                                           sampler=train_sampler,
                                           pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=batch_size,
                                          sampler=test_sampler,
                                          pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset, 
                                           batch_size=batch_size,
                                           sampler=valid_sampler,
                                           pin_memory=True)
    if reuse:
        return train_loader,test_loader,valid_loader,train_sampler,test_sampler,valid_sampler
    return train_loader, test_loader, valid_loader


