import torch
import torchvision
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

print("Environment setup seems correct.")
