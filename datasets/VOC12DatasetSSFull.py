import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

"""
Class to represent the complete semantic segmentation dataset from PASCAL VOC12

"""

class VOC12DatasetSSFull(Dataset):

    def __init__(self, data_dir, mode='train'):
        """
        - mode options: 'train', 'val', 'trainval'
        """
        
        super().__init__()

        self.data_dir = data_dir
        self.img_dir = f"{self.data_dir}/JPEGImages"
        self.labelmask_dir = f"{self.data_dir}/SegmentationClass"

        self.mode = mode 

        img_filenames_path = f"{self.data_dir}/ImageSets/Segmentation/{self.mode}.txt"  # Depending on the mode, the path will change
        with open(img_filenames_path, 'r') as f:
            self.img_filenames = sorted(f.read().split('\n'))

        self.resize_dims = (320,256) # W x H  (for PIL)


    def __len__(self):
        return len(self.img_filenames)


    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.img_filenames[idx]}.jpg"
        img = Image.open(img_path).resize(self.resize_dims, resample=Image.BILINEAR)
        img = torch.from_numpy(np.array(img)) # Transform PIL image of shape (H,W,C) to Tensor of shape (C,H,W)
        img = img.permute((2,0,1))            #

        labelmask_path = f"{self.labelmask_dir}/{self.img_filenames[idx]}.png"
        labelmask = Image.open(labelmask_path).resize(self.resize_dims, resample=Image.NEAREST)
        # print(labelmask.mode)  # Mode is 'P' 
        labelmask = torch.from_numpy(np.array(labelmask)) # Shape: (H,W)
        labelmask[labelmask == 255] = 0 # Turn the "NULL" pixel label (255) to Background (0)

        # Create the dict of Tensors
        sample = {'image data': img,
                  'label mask': labelmask} 

        return sample




if __name__ == '__main__':

    dataset = VOC12DatasetSSFull("../../Datasets/PASCAL_VOC12/VOC2012", "train")
    sample_img = dataset[0]['image data'].squeeze().permute(1,2,0).numpy()
    print(sample_img.shape)