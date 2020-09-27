import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.VOC12DatasetSSFull import VOC12DatasetSSFull
from models.FCNVGG16Full import FCNVGG16Full

# import utils.datautils as datautils
import utils.preprocessing as preprocessing
import utils.metrics as metrics



# Random seeds
np.random.seed(0)
torch.manual_seed(0)


# Config
DATA_CONFIG = { 'data dir' : "../../Datasets/PASCAL_VOC12/VOC2012",
                'batch size' : 32
              }

TRAINING_CONFIG = {'epochs': 100,
                   'learning rate': 0.01
                   }

CHECKPOINT_DIR = "./model_checkpoints"
OUTPUT_DIR = "./results"


def main():

    # Obtain GPU info
    is_available = torch.cuda.is_available()
    n_devices = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu0_name = torch.cuda.get_device_name(0)

    print("\n")
    print("GPU info --")
    print(f"Is available: {is_available}")
    print(f"Total CUDA devices: {n_devices}")
    print(f"Current device: {current_device}")
    print(f"GPU 0 name: {gpu0_name}")
    print("\n")

    # gpu_0 = torch.device('cuda:0') # Currently only 1 GPU is needed
    # gpu_1 = torch.device('cuda:1')


    # Create the datasets and data loaders ----------------------------------------
    train_dataset = VOC12DatasetSSFull(DATA_CONFIG['data dir'], 'train')
    val_dataset = VOC12DatasetSSFull(DATA_CONFIG['data dir'], 'val')

    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=True)


    # -----------------------------------------------------------
    # Initialize the model 
    fcn_model = FCNVGG16Full(mode='fcn-32s').cuda()
    param_list = [p.numel() for p in fcn_model.parameters() if p.requires_grad == True]
    print(f"Trainable parameters: {param_list}")

    # Define loss
    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction = 'mean') # 2-dimensional CE loss

    # Define optimizer
    optimizer = torch.optim.SGD(fcn_model.parameters(),
                                lr = TRAINING_CONFIG['learning rate'])


    # -----------------------------------------------------------
    # Training loop 

    batch_size = DATA_CONFIG['batch size']
    n_classes = 21

    epoch_train_loss_list = []
    epoch_train_iou_list = []

    epoch_val_loss_list = []
    epoch_val_iou_list = []

    for e in range(1, TRAINING_CONFIG['epochs']+1):
        print(f"\nStarting epoch:{e}")

        # One pass over the training set -- 
        print("Training ...")
        epoch_train_loss = 0

        fcn_model.train() # Set model in train mode

        for train_batch in tqdm(train_loader):
            input_batch = train_batch['image data'].cuda()  # Shape: (batch_size, 3, H, W)
            label_batch = train_batch['label mask'].long().cuda() # Shape: (batch_size, H, W)

            # Normalize input images over the batch
            input_batch = preprocessing.normalize_intensities(input_batch, normalization='min-max')

            # Forward pass
            optimizer.zero_grad() # Clear any previous gradients
            pred_batch = fcn_model(input_batch)

            # Compute training loss
            train_loss = cross_entropy_fn(pred_batch, label_batch)
            epoch_train_loss += train_loss.item()

            # Back-propagation
            train_loss.backward()  # Compute gradients
            optimizer.step() # Update model parameters


        epoch_train_loss /= len(train_loader)
        epoch_train_loss_list.append(epoch_train_loss)
        print(f"Train loss: {epoch_train_loss}")

        train_iou_score = metrics.iou_from_tensors(pred_batch, label_batch)
        epoch_train_iou_list.append(train_iou_score)
        print(f"Training IoU: {train_iou_score}")


        # Clear CUDA cache
        torch.cuda.empty_cache()


        # Validate --
        print("Validating ...")
        epoch_val_loss = 0

        optimizer.zero_grad() # Clear any previous gradients
        fcn_model.eval() # Set model in eval mode

        for val_batch in tqdm(val_loader):
            input_batch = val_batch['image data'].cuda() # Shape: (batch_size, 3, H, W)
            label_batch = val_batch['label mask'].long().cuda() # Shape: (batch_size, H, W)

            # Normalize input images over the batch
            input_batch = preprocessing.normalize_intensities(input_batch, normalization='min-max')

            # Forward pass
            with torch.no_grad():  # Disable autograd engine
                pred_batch = fcn_model(input_batch)
                # Compute validation loss
                val_loss = cross_entropy_fn(pred_batch, label_batch)

            epoch_val_loss += val_loss.item()


        # Clear CUDA cache
        torch.cuda.empty_cache()

        epoch_val_loss /= len(val_loader)
        epoch_val_loss_list.append(val_loss)
        print(f"Validation loss: {epoch_val_loss}")

        val_iou_score = metrics.iou_from_tensors(pred_batch, label_batch)
        epoch_val_iou_list.append(val_iou_score)
        print(f"Validation IoU: {val_iou_score}")

        if e % 50 == 0:  # Checkpoint every 50 epochs
            torch.save(fcn_model.state_dict(), f"{CHECKPOINT_DIR}/fcnvgg16_ep{e}_iou{round(val_iou_score*100)}.pt")

    # Write metrics into files
    np.savetxt(f"{OUTPUT_DIR}/training_losses.csv", np.array(epoch_train_loss_list))
    np.savetxt(f"{OUTPUT_DIR}/validation_losses.csv", np.array(epoch_val_loss_list))

    np.savetxt(f"{OUTPUT_DIR}/training_iou.csv", np.array(epoch_train_iou_list))
    np.savetxt(f"{OUTPUT_DIR}/validation_iou.csv", np.array(epoch_val_iou_list))



if __name__ == '__main__':
    main()