import os
import requests
import sys
from tqdm import tqdm

def download_ffhq():
    # FFHQ model URL from official repository
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    
    # Define output directory and file path
    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, "checkpoints", "stylegan2_ada")
    output_path = os.path.join(output_dir, "ffhq.pkl")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading FFHQ checkpoint to {output_path}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        with open(output_path, 'wb') as f, tqdm(
            desc="ffhq.pkl",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
                
        print("\nDownload complete!")
        print(f"Saved to: {output_path}")
        
    except Exception as e:
        print(f"\nError downloading file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_ffhq()
