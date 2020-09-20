import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

image_path = "./Data/VOC2012/SegmentationClass/2007_000039.png"

# PIL reads the label image as a grayscale
img_pil = Image.open(image_path).resize((320,240)) 
img_pil_np = np.array(img_pil)
print("img_pil_np shape:", img_pil_np.shape)
print("img_pil_np max:", img_pil_np.max())
plt.hist(img_pil_np, bins=256)
plt.show()
print()

img_pil_rgb_np = np.array(img_pil.convert('RGB'))
print("img_pil_rgb_np shape:", img_pil_rgb_np.shape)
print("img_pil_rgb_np max:", img_pil_rgb_np.max())
plt.hist(img_pil_rgb_np, bins=256)
plt.show()
print()

img_plt = plt.imread(image_path)
print("img_plt shape:", img_plt.shape)
print("img_plt max:", img_plt.max())
plt.hist(img_plt, bins=256)
plt.show()
print()

img_cv = cv2.imread(image_path)
print("img_cv shape:", img_cv.shape)
print("img_cv max:", img_cv.max())
plt.hist(img_cv, bins=256)
plt.show()