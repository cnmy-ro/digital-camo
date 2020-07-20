import torch
from torch.utils.data import DataLoader

import PASCALData, Model

# Data config
DATA_DIR = "./Data/"
BATCH_SIZE = 4

# Create the datasets and data loaders
train_dataset = PASCALData.VOC12Dataset(DATA_DIR, 'train', normalize=True)
val_dataset = PASCALData.VOC12Dataset(DATA_DIR, 'val', normalize=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# sample = train_dataset[0]['image data'].permute(1,2,0).numpy()