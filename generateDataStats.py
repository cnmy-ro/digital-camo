'''
Generate channel-wise mean and std deviation of pixel values across the entire segmentation 'trainval' set
'''

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import PASCALData

DATA_DIR = "./Data/"
output_path = "./datastats-trainval-360x240.txt"


trainval_dataset = PASCALData.VOC12Dataset(DATA_DIR, mode='trainval', normalize=False)
trainval_loader = DataLoader(trainval_dataset, batch_size=1, shuffle=False)


# -------------------------------
print("\nCalculating Mean ...")
mean = np.array([0,0,0], dtype=np.float32)
for batch in tqdm(trainval_loader):
    img = batch['image data'].squeeze().permute(1,2,0).numpy()
    mean += np.mean(img, axis=(0,1))

mean /= len(trainval_loader)
print("\nChannel-wise Mean:", list(mean))


# -------------------------------
print("\nCalculating Standard Deviation ...")
stddev = np.array([0,0,0], dtype=np.float32)
for batch in tqdm(trainval_loader):
    img = batch['image data'].squeeze().permute(1,2,0).numpy()
    sq_diff = (img - mean) ** 2
    stddev += np.sum(sq_diff, axis=(0,1))

stddev /= img.shape[0] * img.shape[1] * len(trainval_loader) - 1
stddev = np.sqrt(stddev)
print("\nChannel-wise Standard Deviation:", list(stddev))


# -------------------------------
stats = np.stack([mean,stddev], axis=0)
np.savetxt(output_path, stats, delimiter=',')