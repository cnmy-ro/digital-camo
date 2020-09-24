import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from VOC12Dataset import VOC12Dataset
from Model import FCN_VGG16

import utils.data_utils


# Config
DATA_CONFIG = { 'data dir' : "./data",
                'batch size' : 1
              }

CHECKPOINT_DIR = "./model_checkpoints"
MODEL_PATH = f"{CHECKPOINT_DIR}/fcnvgg16_100.pt"
OUTPUT_DIR = "./results/inference_predictions"


# Create the datasets and data loaders ----------------------------------------
val_dataset = VOC12Dataset(DATA_CONFIG['data dir'], 'val')
val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=False)
print("Val loader length:", len(val_loader))


fcn_model = FCN_VGG16(mode='fcn-32s')
fcn_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
fcn_model.eval()

sample_count = 0
for val_batch in tqdm(val_loader):
    input_img = val_batch['image data'] # Shape: (batch_size, 3, H, W)
    label = val_batch['gt mask'].long() # Shape: (batch_size, H, W)

    # Normalize input images over the batch
    input_img = utils.data_utils.normalize_intensities(input_img, normalization='min-max')

    # Forward pass
    with torch.no_grad():  # Disable autograd engine
        pred = fcn_model(input_img)

    sample_count += 1

    pred_mask = pred.argmax(dim=1).squeeze().numpy()
    #print(pred_mask.shape)
    plt.imshow(pred_mask)
    plt.show()
    break
    #plt.imsave(f"{OUTPUT_DIR}/{sample_count}.png", pred_mask, cmap='tab20')