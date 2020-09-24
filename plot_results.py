import numpy as np
import matplotlib.pyplot as plt

train_losses_file = "./results/training_losses.csv"
val_losses_file = "./results/validation_losses.csv"

train_losses = np.loadtxt(train_losses_file, delimiter=',')
val_losses = np.loadtxt(val_losses_file, delimiter=',')

plt.plot(np.arange(1,101), train_losses, 'r-', label='training')
plt.plot(np.arange(1,101), val_losses, 'b-', label='validation')
plt.legend()

plt.show()