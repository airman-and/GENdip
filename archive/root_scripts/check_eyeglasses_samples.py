import pandas as pd
import os

# Load attributes
attr_path = '/root/workspace/andycho/GenDL-LatentControl/dataset/celebA/list_attr_celeba.csv'
df = pd.read_csv(attr_path)

# Filter for Eyeglasses = 1
eyeglasses_pos = df[df['Eyeglasses'] == 1]

# Get the first 10 image IDs
print("First 10 images with Eyeglasses used for vector extraction:")
print(eyeglasses_pos['image_id'].head(10).tolist())

# Check total count again
print(f"\nTotal Eyeglasses samples: {len(eyeglasses_pos)}")
