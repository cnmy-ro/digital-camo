import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from VOC12Dataset import VOC12Dataset
from Model import FCN_VGG16

import utils.data_utils


# np.random.seed(0)
# torch.manual_seed(0)

# Config --

## Data config
DATA_CONFIG = { 'data dir' : "./data/",
                'batch size' : 4
              }

## Training config
TRAINING_CONFIG = {'epochs': 1,
                   'learning rate': 0.001
                   }


# Create the datasets and data loaders --
train_dataset = VOC12Dataset(DATA_CONFIG['data dir'], 'train')
val_dataset = VOC12Dataset(DATA_CONFIG['data dir'], 'val')

train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=True)


# Initialize the model --
fcn_model = FCN_VGG16(mode='fcn-32s')
param_list = [p.numel() for p in fcn_model.parameters() if p.requires_grad == True]
print("Trainable parameters:", param_list)

cross_entropy_loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(fcn_model.parameters(), 
                            lr=TRAINING_CONFIG['learning rate'])


# Training loop --
batch_size = DATA_CONFIG['batch size']
n_classes = 21
train_loss_list = []
val_loss_list = []

for e in range(1, TRAINING_CONFIG['epochs']+1):
    print("Starting epoch:", e,)

    # One pass over the training set
    print("Training ...")
    for train_batch in tqdm(train_loader):
        input_batch = train_batch['image data']  # Shape: (batch_size, 3, H, W)
        label_batch = train_batch['gt mask'].long() # Shape: (batch_size, H, W)
        
        # Normalize input images over the batch
        input_batch = utils.data_utils.normalize_intensities(input_batch, normalization='min-max')

        # Forward pass
        optimizer.zero_grad() # Clear any previous gradients
        pred_batch = fcn_model(input_batch)
        
        # Compute training loss
        train_loss = cross_entropy_loss(pred_batch, label_batch)
        train_loss_list.append(train_loss)
        
        # Back-propagation
        train_loss.backward()  # Compute gradients  
        optimizer.step() # Update model parameters


    # Validate
    print("Validating ...")
    for val_batch in tqdm(val_loader):
        input_batch = train_batch['image data']  # Shape: (batch_size, 3, H, W)
        label_batch = train_batch['gt mask'].long() # Shape: (batch_size, H, W)

        # Normalize input images over the batch
        input_batch = utils.data_utils.normalize_intensities(input_batch, normalization='min-max')

        # Forward pass
        optimizer.zero_grad() # Clear any previous gradients
        pred_batch = fcn_model(input_batch)

        # Compute validation loss
        val_loss = cross_entropy_loss(pred_batch, label_batch)
        val_loss_list.append(val_loss)