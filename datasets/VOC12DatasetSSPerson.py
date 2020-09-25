import os, random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

"""
Class to represent the a subset of the PASCAL VOC12 semantic segmentation dataset that only contains
images with humans in them (i.e. having pixels correspionding to the 'Person' class).

"""

class VOC12DatasetSSPerson(Dataset):

    def __init__(self, data_dir, mode='train'):
        """
        - mode options: 'train', 'val'
        """
        self.data_dir = data_dir
        self.img_dir = f"{self.data_dir}/Images"
        self.labelmask_dir = f"{self.data_dir}/SegmentationClass"

        self.mode = mode
        self.person_label = 15

        self.img_filenames = sorted(os.listdir(self.img_dir))
        self.img_filenames = [filename.split(".")[0] for filename in self.img_filenames]
        random.shuffle(self.img_filenames) # Shuffle
        # Split
        split_idx = round( len(self.img_filenames) * 0.8 )    
        self.train_img_filenames = self.img_filenames[:split_idx]
        self.val_img_filenames = self.img_filenames[split_idx:]

        self.resize_dims = (320,256) # W x H  (for PIL)


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_img_filenames)
        elif self.mode == 'val':
            return len(self.val_img_filenames)


    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.img_filenames[idx]}.jpg"
        img = Image.open(img_path).resize(self.resize_dims, resample=Image.BILINEAR)
        img = torch.from_numpy(np.array(img)) # Transform PIL image of shape (H,W,C) to Tensor of shape (C,H,W)
        img = img.permute((2,0,1))            #

        labelmask_path = f"{self.labelmask_dir}/{self.img_filenames[idx]}.png"
        labelmask = Image.open(labelmask_path).resize(self.resize_dims, resample=Image.NEAREST)
        # print(labelmask.mode)  # Mode is 'P' 
        labelmask = torch.from_numpy(np.array(labelmask)) # Shape: (H,W)
        labelmask = labelmask == self.person_label # Convert the multi-label mask into a binary mask 
        labelmask = labelmask.long()

        # Create the dict of Tensors
        sample = {'image data': img,
                  'label mask': labelmask} 

        return sample



if __name__ == '__main__':

    dataset = VOC12DatasetSSPerson("../../Datasets/PASCAL_VOC12_SS_Person", "train")
    sample_img = dataset[0]['image data'].squeeze().permute(1,2,0).numpy()
    sample_labelmask = dataset[0]['label mask'].squeeze().numpy()
    print("Image shape:", sample_img.shape)
    print("Unique classes:", np.unique(sample_labelmask))