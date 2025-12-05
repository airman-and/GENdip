import os
import torchvision
from torchvision.datasets import CelebA

def download_celeba():
    # Define the path where we want the dataset
    # config.py expects: project_dir/dataset/celebA
    # torchvision downloads into 'root/celeba' usually.
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.join(base_path, 'dataset')
    
    print(f"Attempting to download CelebA to {dataset_root}...")
    
    try:
        os.makedirs(dataset_root, exist_ok=True)
        # download=True will download and extract
        dataset = CelebA(root=dataset_root, split='all', target_type='attr', download=True)
        print("Download and extraction successful!")
        
        # Check the structure
        print("Checking structure...")
        celeba_path = os.path.join(dataset_root, 'celeba')
        if os.path.exists(celeba_path):
            print(f"Found {celeba_path}")
            print(f"Contents: {os.listdir(celeba_path)}")
        else:
            print(f"Could not find 'celeba' folder in {dataset_root}")
            
    except Exception as e:
        print(f"Error downloading CelebA: {e}")
        print("Note: CelebA download often fails due to Google Drive quota limits.")

if __name__ == '__main__':
    download_celeba()
