import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from datasets.VOC12DatasetSSPerson import VOC12DatasetSSPerson
from models.FCNVGG16Binary import FCNVGG16Binary

# import utils.datautils as datautils
import utils.preprocessing as preprocessing
import utils.metrics as metrics

# Random seeds
np.random.seed(0)
torch.manual_seed(0)


# Config
DATA_CONFIG = { 'data dir' : "../../Datasets/PASCAL_VOC12_SS_Person",
                'batch size' : 32
              }

TRAINING_CONFIG = {'epochs': 200,
                   'learning rate': 0.005
                   }

CHECKPOINT_DIR = "./model_checkpoints"
OUTPUT_DIR = "./results"


# Logging stuff
def get_logger(display_time=False):

    # create logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    if display_time:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def main():
    logger = get_logger()

    # Obtain GPU info
    is_available = torch.cuda.is_available()
    n_devices = torch.cuda.device_count()
    current_device = torch.cuda.current_device()
    gpu0_name = torch.cuda.get_device_name(0)

    logger.debug("")
    logger.debug("GPU info --")
    logger.debug(f"Is available: {is_available}")
    logger.debug(f"Total CUDA devices: {n_devices}")
    logger.debug(f"Current device: {current_device}")
    logger.debug(f"GPU 0 name: {gpu0_name}")
    logger.debug("")

    # Create the datasets and data loaders ----------------------------------------
    train_dataset = VOC12DatasetSSPerson(DATA_CONFIG['data dir'], 'train')
    val_dataset = VOC12DatasetSSPerson(DATA_CONFIG['data dir'], 'val')

    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['batch size'], shuffle=True)


    # Initialize the model ---------------------------------------------------------
    fcn_model = FCNVGG16Binary(mode='fcn-32s').cuda()
    param_list = [p.numel() for p in fcn_model.parameters() if p.requires_grad == True]
    logger.debug(f"Parameters: {param_list}")
    logger.debug(f"Total parameters: {sum(param_list)}")

    cross_entropy_fn = torch.nn.CrossEntropyLoss(reduction='mean') # 2-dimensional CE loss

    optimizer = torch.optim.Adam(fcn_model.parameters(),
                                 lr = TRAINING_CONFIG['learning rate'])


    # Training loop ----------------------------------------------------------
    batch_size = DATA_CONFIG['batch size']

    epoch_train_loss_list = []
    epoch_train_iou_list = []

    epoch_val_loss_list = []
    epoch_val_iou_list = []

    for e in range(1, TRAINING_CONFIG['epochs']+1):
        logger.debug("")
        logger.debug(f"Starting epoch:{e}")

        # One pass over the training set
        logger.debug("Training ...")
        epoch_train_loss = 0

        fcn_model.train() # Set model in train mode

        for train_batch in tqdm(train_loader):
            input_batch = train_batch['image data'].cuda()  # Shape: (batch_size, 3, H, W)
            label_batch = train_batch['label mask'].cuda()  # Shape: (batch_size, H, W)

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
        logger.debug(f"Train loss: {epoch_train_loss}")

        train_iou_score = metrics.iou_from_tensors(pred_batch, label_batch)
        epoch_train_iou_list.append(train_iou_score)
        logger.debug(f"Training IoU: {train_iou_score}")


        # Clear CUDA cache
        torch.cuda.empty_cache()


        # Validate --
        logger.debug("Validating ...")
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
        logger.debug(f"Validation loss: {epoch_val_loss}")

        val_iou_score = metrics.iou_from_tensors(pred_batch, label_batch)
        epoch_val_iou_list.append(val_iou_score)
        logger.debug(f"Validation IoU: {val_iou_score}")

        if e % 25 == 0:  # Checkpoint every 25 epochs
            torch.save(fcn_model.state_dict(), f"{CHECKPOINT_DIR}/fcnvgg16_ep{e}_iou{round(val_iou_score*100)}.pt")

    # Write metrics into files
    np.savetxt(f"{OUTPUT_DIR}/training_losses.csv", np.array(epoch_train_loss_list))
    np.savetxt(f"{OUTPUT_DIR}/validation_losses.csv", np.array(epoch_val_loss_list))

    np.savetxt(f"{OUTPUT_DIR}/training_iou.csv", np.array(epoch_train_iou_list))
    np.savetxt(f"{OUTPUT_DIR}/validation_iou.csv", np.array(epoch_val_iou_list))



if __name__ == '__main__':
    main()