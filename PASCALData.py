from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np



class VOC12Dataset(Dataset):

    def __init__(self, data_dir, mode='train', normalize=True, data_stats_path='./datastats-trainval-480x360.txt'):
        super(VOC12Dataset, self).__init__()

        self.data_dir = data_dir
        self.mode = mode # Modes: 'train', 'val', 'trainval'

        img_filenames_path = self.data_dir+"VOC2012/ImageSets/Segmentation/{}.txt".format(self.mode)
        with open(img_filenames_path, 'r') as f:
            self.img_filenames = sorted(f.read().split('\n'))

        if normalize:
            stats = np.loadtxt(data_stats_path, delimiter=',')
            mean, stddev = stats[0,:], stats[1,:]
            self.preprocess_transform = transforms.Compose([transforms.ToTensor(),
                                                            transforms.Normalize(mean, stddev)])
        else:
            self.preprocess_transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.img_filenames)


    def __getitem__(self, idx):
        img_path = self.data_dir + "VOC2012/JPEGImages/" + self.img_filenames[idx] + ".jpg"
        img = Image.open(img_path).resize((480,360)) # Shape format: W x H
        img = self.preprocess_transform(img) # Shape format: C x H x W

        sample = {'image data': img} # Dict of Tensors

        return sample



if __name__ == '__main__':

    dataset = VOC12Dataset("./Data/", "train")
    sample_img = dataset[0]['image data'].squeeze().permute(1,2,0).numpy()