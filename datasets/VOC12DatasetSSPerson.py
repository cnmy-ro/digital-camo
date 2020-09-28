import os, random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF


"""
Class to represent the a subset of the PASCAL VOC12 semantic segmentation dataset that only contains
images with humans in them (i.e. having pixels correspionding to the 'Person' class).

"""

class VOC12DatasetSSPerson(Dataset):

    def __init__(self, data_dir, mode='train'):
        """
        - mode options: 'train', 'val', 'trainval'
        """
        self.data_dir = data_dir
        self.img_dir = f"{self.data_dir}/Images"
        self.labelmask_dir = f"{self.data_dir}/SegmentationClass"

        self.mode = mode
        self.person_label = 15

        trainval_img_filenames = sorted(os.listdir(self.img_dir))
        trainval_img_filenames = [filename.split(".")[0] for filename in trainval_img_filenames]
        random.shuffle(trainval_img_filenames) # Shuffle
        # Split
        split_idx = round( len(trainval_img_filenames) * 0.8 )    
        train_img_filenames = trainval_img_filenames[:split_idx]
        val_img_filenames = trainval_img_filenames[split_idx:]

        if self.mode == 'train':
            self.relevant_img_filenames = train_img_filenames
        elif self.mode == 'val':
            self.relevant_img_filenames = val_img_filenames
        elif self.mode == 'trainval':
            self.relevant_img_filenames = trainval_img_filenames

        self.resize_dims = (320,256) # W x H  (for PIL)


    def __len__(self):
        return len(self.relevant_img_filenames)


    def _augmentation_transform(self, image_pil, labelmask_pil):
        # Apply a transformation with 0.5 probability
        if random.random() > 0.5:
            choice = random.choice([1,2,3,4])
            
            if choice == 1: # Random rotation
                angle = random.randint(-30, 30)
                image_pil = TF.rotate(image_pil, angle)
                labelmask_pil = TF.rotate(labelmask_pil, angle)
            
            elif choice == 2: # Horizontal flip
                image_pil = TF.hflip(image_pil)
                labelmask_pil = TF.hflip(labelmask_pil) 
            
            elif choice == 3: # Brightness (only for input image) 
                brightness_factor = (np.random.random() * 0.2) - 0.1
                image_pil = TF.adjust_brightness(image_pil, brightness_factor=brightness_factor)
            
            elif choice == 4: # Contrast (only for input image)
                contrast_factor = (np.random.random() * 0.2) - 0.1
                image_pil = TF.adjust_contrast(image_pil, contrast_factor=contrast_factor)

        return image_pil, labelmask_pil


    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.relevant_img_filenames[idx]}.jpg"
        image_pil = Image.open(img_path).resize(self.resize_dims, resample=Image.BILINEAR)

        labelmask_path = f"{self.labelmask_dir}/{self.relevant_img_filenames[idx]}.png"
        labelmask_pil = Image.open(labelmask_path).resize(self.resize_dims, resample=Image.NEAREST)
        # print(labelmask_pil.mode)  # Mode is 'P'

        # Data augmentation
        if self.mode == 'train':
            image_pil, labelmask_pil = self._augmentation_transform(image_pil, labelmask_pil)

        image_tensor = torch.from_numpy(np.array(image_pil)) # Transform PIL image of shape (H,W,C) to Tensor of shape (C,H,W)
        image_tensor = image_tensor.permute((2,0,1))         #
          
        labelmask_tensor = torch.from_numpy(np.array(labelmask_pil)) # Shape: (H,W)
        labelmask_tensor = labelmask_tensor == self.person_label # Convert the multi-label mask into a binary mask 
        labelmask_tensor = labelmask_tensor.long()

        # Create the dict of Tensors
        sample = {'image data': image_tensor,
                  'label mask': labelmask_tensor} 

        return sample



if __name__ == '__main__':

    dataset = VOC12DatasetSSPerson("../../Datasets/PASCAL_VOC12_SS_Person", "trainval")
    print("No. of samples:", len(dataset))
    sample_img = dataset[0]['image data'].squeeze().permute(1,2,0).numpy()
    sample_labelmask = dataset[0]['label mask'].squeeze().numpy()
    print("Image shape:", sample_img.shape)
    print("Unique classes:", np.unique(sample_labelmask))