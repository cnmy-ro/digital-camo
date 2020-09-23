import numpy as np
import torch

'''
GPU stuff
'''

is_available = torch.cuda.is_available()
n_devices = torch.cuda.device_count()
current_device = torch.cuda.current_device()
gpu0_name = torch.cuda.get_device_name(0)

print("GPU info --")
print("Is available:", is_available)
print("Total CUDA devices:", n_devices)
print("Current device:", current_device)
print("GPU 0 name:", gpu0_name)
