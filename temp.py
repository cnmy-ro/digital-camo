import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

filename = "./Data/VOC2012/SegmentationClass/2007_000039.png"

img = Image.open(filename).resize((320,240)) # filename is the png file in question
palette = img.getpalette()
img_array = np.array(img)
img_rgb = np.array(img.convert('RGB'))

print(img_array[150,150])
print(img_rgb[150,150])