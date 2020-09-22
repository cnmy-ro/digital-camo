import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

'''

Class to represent the complete semantic segmentation dataset from PASCAL VOC12

'''

class VOC12Dataset(Dataset):

    def __init__(self, data_dir, mode='train'):
        '''
        - mode options: 'train', 'val', 'trainval'
        '''
        
        super(VOC12Dataset, self).__init__()

        self.data_dir = data_dir
        self.img_dir = self.data_dir + '/' + "VOC2012/JPEGImages"
        self.gt_mask_dir = self.data_dir + '/' + "VOC2012/SegmentationClass"

        self.mode = mode 

        img_filenames_path = self.data_dir+"VOC2012/ImageSets/Segmentation/{}.txt".format(self.mode)
        with open(img_filenames_path, 'r') as f:
            self.img_filenames = sorted(f.read().split('\n'))

        self.to_tensor = transforms.Compose([transforms.ToTensor()])

        self.resize_dims = (320,256) # W x H -- (480,360), (360,240)


    def __len__(self):
        return len(self.img_filenames)


    def __getitem__(self, idx):
        img_path = self.img_dir + '/' + self.img_filenames[idx] + ".jpg"
        img = Image.open(img_path).resize(self.resize_dims, resample=Image.BILINEAR) # Shape format: W x H X C
        img = self.to_tensor(img)  # Transform PIL image of shape (W,H,C) to Tensor of shape (C,H,W)

        gt_mask_path = self.gt_mask_dir + '/' + self.img_filenames[idx] + ".png"

        gt_mask = Image.open(gt_mask_path).resize(self.resize_dims, resample=Image.NEAREST)
        # print(gt_mask.mode)  # Mode is 'P' 

        gt_mask = torch.from_numpy(np.array(gt_mask))

        gt_mask[gt_mask == 255] = 0 # Turn the "NULL" pixel label (255) to Background (0)

        # Create the dict of Tensors
        sample = {'image data': img,
                  'gt mask': gt_mask} 

        return sample



##################################################

if __name__ == '__main__':

    dataset = VOC12Dataset("./data/", "train")
    sample_img = dataset[0]['image data'].squeeze().permute(1,2,0).numpy()