"""
Script to generate the class distribution of pixel values
"""
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

# from datasets.VOC12DatasetSSPerson import VOC12DatasetSSPerson


def main(label_dir):
    
    label_filenames = sorted(os.listdir(label_dir))
    
    classwise_pixel_counts = {0:0, 
                              1:0, 2:0, 3:0, 4:0, 5:0,
                              6:0, 7:0, 8:0, 9:0, 10:0,
                              11:0, 12:0, 13:0, 14:0, 15:0,
                              16:0, 17:0, 18:0, 19:0, 20:0,
                              255:0}

    person_class_id = 15

    resize_dims = (320,256) # W x H  (for PIL)

    for lf in tqdm(label_filenames):
        label_path = f"{label_dir}/{lf}"
        label_mask = Image.open(label_path).resize(resize_dims, resample=Image.NEAREST)
        label_mask = np.array(label_mask)
        
        unique_classes = np.unique(label_mask)
        for class_id in unique_classes:
            classwise_pixel_counts[class_id] += np.sum(label_mask == class_id)
    
    total_pixels = sum([count for count in classwise_pixel_counts.values()])
    print("Total pixels:", total_pixels)
    print("Classwise pixel counts:", classwise_pixel_counts)
    print("Person class pixel count:", classwise_pixel_counts[person_class_id])
    print("Not-person classes pixel count:", total_pixels - classwise_pixel_counts[person_class_id])

    

if __name__ == '__main__':

    label_dir = "../../../Datasets/PASCAL_VOC12_SS_Person/SegmentationClass"
    main(label_dir)