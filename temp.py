import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

filename = "./Data/VOC2012/SegmentationClass/2007_000039.png"

img = Image.open(filename) # filename is the png file in question

img = np.array(img)

print(np.array(img)[170,250])

plt.imshow(np.array(img))
plt.show()
