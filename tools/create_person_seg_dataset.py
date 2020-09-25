"""

"""

import os, shutil
from tqdm import tqdm

voc12_data_dir = "../../../Datasets/PASCAL_VOC12/VOC2012"
image_src_dir = f"{voc12_data_dir}/JPEGImages"
seg_src_dir = f"{voc12_data_dir}/SegmentationClass"

# Output paths
image_dest_dir = "../../../Datasets/PASCAL_VOC12_PersonSeg/Images"
seg_dest_dir = "../../../Datasets/PASCAL_VOC12_PersonSeg/SegmentationClass"


seg_trainval_images = []
with open(f"{voc12_data_dir}/ImageSets/Segmentation/trainval.txt", 'r') as f:
    seg_trainval_images = f.read().split("\n")

# print(seg_trainval_images)

seg_images_w_persons = []
with open(f"{voc12_data_dir}/ImageSets/Main/person_trainval.txt", 'r') as f:
    file_data = f.read().split("\n")
    for line in file_data:
        line = line.split()
        try:
            if line[0] in seg_trainval_images and line[1] == '1':
            
                seg_images_w_persons.append(line[0])
        except: continue

# print(seg_images_w_persons)
print("Total images found:", len(seg_images_w_persons))

print("Creating a data subset - copying files ...")
for img_name in tqdm(seg_images_w_persons):
    # Copy images
    shutil.copyfile(f"{image_src_dir}/{img_name}.jpg", f"{image_dest_dir}/{img_name}.jpg")
    # Copy segmentation masks
    shutil.copyfile(f"{seg_src_dir}/{img_name}.png", f"{seg_dest_dir}/{img_name}.png")
    