import numpy as np
import torch
import matplotlib.pyplot as plt

from VOC12Dataset import VOC12Dataset

voc12_dataset = VOC12Dataset(data_dir='./data/')
print(voc12_dataset[0].keys())


idx = np.random.randint(0,len(voc12_dataset))

X = voc12_dataset[idx]['image data']
X = X.numpy()
X = X.transpose((1,2,0))

Y = voc12_dataset[idx]['gt mask']
Y = Y.numpy()


fig, axs = plt.subplots(1,2)
axs[0].imshow(X)
axs[1].imshow(Y)
plt.show()