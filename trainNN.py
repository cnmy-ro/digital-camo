import numpy as np
import torch
from torch.utils.data import DataLoader

import PASCALData, Model


np.random.seed(0)
torch.manual_seed(0)


# Data config
DATA_DIR = "./Data/"
BATCH_SIZE = 4

# Training config
EPOCHS = 10
LR = 0.01

# Create the datasets and data loaders
train_dataset = PASCALData.VOC12Dataset(DATA_DIR, 'train', normalize=True)
val_dataset = PASCALData.VOC12Dataset(DATA_DIR, 'val', normalize=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

# sample_img = train_dataset[3]['image data'].permute(1,2,0).numpy()
# sample_mask = train_dataset[3]['gt mask'].numpy()

