import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np



class VOC12Dataset(Dataset):

    def __init__(self, data_dir, mode='train', normalize=True, data_stats_path='./Data/datastats-trainval-360x240.txt'):
        super(VOC12Dataset, self).__init__()

        self.data_dir = data_dir
        self.img_dir = self.data_dir + "VOC2012/JPEGImages/"
        self.gt_mask_dir = self.data_dir + "VOC2012/SegmentationClass/"

        self.mode = mode # Modes: 'train', 'val', 'trainval'


        img_filenames_path = self.data_dir+"VOC2012/ImageSets/Segmentation/{}.txt".format(self.mode)
        with open(img_filenames_path, 'r') as f:
            self.img_filenames = sorted(f.read().split('\n'))

        if normalize:
            stats = np.loadtxt(data_stats_path, delimiter=',')
            mean, stddev = stats[0,:], stats[1,:]
            self.img_preprocess_transform = transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(mean, stddev)])
        else:
            self.img_preprocess_transform = transforms.Compose([transforms.ToTensor()])


        self.resize_dims = (360, 240) # W x H -- (480,360), (360,240)


    def __len__(self):
        return len(self.img_filenames)


    def __getitem__(self, idx):
        img_path = self.img_dir + self.img_filenames[idx] + ".jpg"
        img = Image.open(img_path).resize(self.resize_dims, resampling=Image.LINEAR) # Shape format: W x H
        img = self.img_preprocess_transform(img) # Shape format: C x H x W

        gt_mask_path = self.gt_mask_dir + self.img_filenames[idx] + ".png"

        gt_mask = Image.open(gt_mask_path).resize(self.resize_dims, resampling=Image.NEAREST)
        print(gt_mask.mode)

        gt_mask = torch.from_numpy(np.array(gt_mask))

        sample = {'image data': img,
                  'gt mask': gt_mask} # Dict of Tensors

        return sample



###############################################################################

if __name__ == '__main__':

    dataset = VOC12Dataset("./Data/", "train")
    sample_img = dataset[0]['image data'].squeeze().permute(1,2,0).numpy()