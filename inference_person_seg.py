import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score as jsc

from datasets.VOC12DatasetSSPerson import VOC12DatasetSSPerson
from models.FCNVGG16Binary import FCNVGG16Binary

import utils.datautils as datautils
import utils.preprocessing as preprocessing
import utils.metrics as metrics


# Config
DATA_CONFIG = { 'data dir' : "../../Datasets/PASCAL_VOC12_SS_Person",
                'batch size' : 32
              }

CHECKPOINT_DIR = "./model_checkpoints"
MODEL_PATH = f"{CHECKPOINT_DIR}/fcnvgg16_ep5_iou41.pt"
OUTPUT_DIR = "./results/inference_predictions"


# Create the datasets and data loaders ----------------------------------------
val_dataset = VOC12DatasetSSPerson(DATA_CONFIG['data dir'], 'val')
val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=False)
print("Val loader length:", len(val_loader))


fcn_model = FCNVGG16Binary(mode='fcn-32s').cuda()
fcn_model.load_state_dict(torch.load(MODEL_PATH))
fcn_model.eval()

softmax_fn = nn.Softmax(dim=1)

sample_count = 0
for val_batch in val_loader:
    input_img = val_batch['image data'].cuda() # Shape: (batch_size, 3, H, W)
    label = val_batch['label mask'].cuda() # Shape: (batch_size, H, W)
    print(torch.unique(label))

    # Normalize input images over the batch
    input_img = preprocessing.normalize_intensities(input_img, normalization='min-max')
    #print(input_img.max())
    # Forward pass
    with torch.no_grad():  # Disable autograd engine
        pred = fcn_model(input_img)
        pred = softmax_fn(pred)

    sample_count += 1
    
    #print(pred[0,1,:,:])
    pred_mask = pred.argmax(dim=1).squeeze()
    print("Label mask unique counts: ", np.unique(label.cpu().numpy(), return_counts=True))
    print("Pred mask unique counts: ", np.unique(pred_mask.cpu().numpy(), return_counts=True))
    print("Macro IoU:", jsc(pred_mask.cpu().numpy().reshape(-1), 
                           label.cpu().numpy().reshape(-1),
                           average='macro'))
    print("Normal binary IoU:", jsc(pred_mask.cpu().numpy().reshape(-1), 
                                    label.cpu().numpy().reshape(-1)))
    
    custom_iou = torch.sum(pred_mask*label).cpu().numpy() / np.sum(np.maximum(pred_mask.cpu().numpy(), label.cpu().numpy()))
    print("Custom normal IoU:", custom_iou)
    print()
    #plt.imshow(pred_mask)
    #plt.show()
    #break

    #plt.imsave(f"{OUTPUT_DIR}/{sample_count}.png", pred_mask, cmap='gray')

