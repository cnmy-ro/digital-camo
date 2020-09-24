import numpy as np
import matplotlib.pyplot as plt

train_losses_file = "./results/training_losses.csv"
val_losses_file = "./results/validation_losses.csv"

train_iou_file = "./results/training_iou.csv"
val_iou_file = "./results/validation_iou.csv"


train_losses = np.loadtxt(train_losses_file, delimiter=',')
val_losses = np.loadtxt(val_losses_file, delimiter=',')

train_iou = np.loadtxt(train_iou_file, delimiter=',')
val_iou = np.loadtxt(val_iou_file, delimiter=',')

fig = plt.figure()

ax_loss = fig.subpadd_subplotlot()
ax_loss.plot(np.arange(1, train_losses.shape[0]+1), train_losses, 'r-', label='Train loss')
ax_loss.plot(np.arange(1, val_losses.shape[0]+1), val_losses, 'b-', label='Val loss')

ax_iou = ax_loss.twinx()
ax_iou.plot(np.arange(1, train_iou.shape[0]+1), train_iou, 'm-', label='Train IoU')
ax_iou.plot(np.arange(1, val_iou.shape[0]+1), val_iou, 'c-', label='Val IoU')

plt.show()