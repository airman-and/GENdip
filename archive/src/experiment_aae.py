import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from core.aae_model import Encoder, Decoder
from core.dataset import get_celeba_loader

def load_aae_model(device, latent_dim=128, image_size=128):
    encoder = Encoder(latent_dim=latent_dim, image_size=image_size).to(device)
    decoder = Decoder(latent_dim=latent_dim, image_size=image_size).to(device)
    
    # Try to load checkpoint
    checkpoint_path = config.aae_model_path
    if not os.path.exists(checkpoint_path):
        # Try to find any checkpoint in checkpoints dir
        checkpoint_dir = os.path.join(config.project_dir, 'checkpoints')
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                # Sort by epoch number
                try:
                    checkpoints.sort(key=lambda x: int(x.split('_')[2]))
                except:
                    checkpoints.sort()
                
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"[Experiment] Found alternative checkpoint: {checkpoint_path}")
            else:
                print(f"[Error] No checkpoints found at {checkpoint_dir}")
                return None, None
        else:
            print(f"[Error] Checkpoint directory not found: {checkpoint_dir}")
            return None, None

    print(f"[Experiment] Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        # Fallback if saved differently
        print("[Warning] Checkpoint format unrecognized, trying direct load...")
        try:
            encoder.load_state_dict(checkpoint) # Might fail if it's full model
        except:
            print("[Error] Could not load state dict")
            return None, None

    encoder.eval()
    decoder.eval()
    return encoder, decoder

def interpolate_points(p1, p2, n_steps=10):
    # Linear interpolation
    ratios = np.linspace(0, 1, n_steps)
    vectors = []
    for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
    return torch.stack(vectors)

def run_experiments():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Experiment] Running on {device}")
    
    # 1. Load Model
    encoder, decoder = load_aae_model(device, latent_dim=config.model_latent_dim, image_size=config.image_size)
    if encoder is None:
        return

    # 2. Load Data (for reconstruction and interpolation source)
    dataloader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=16,
        image_size=config.image_size,
        shuffle=True
    )
    
    # Get a batch
    try:
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)
    except Exception as e:
        print(f"[Error] Could not load data: {e}")
        return

    output_dir = os.path.join(config.output_path, 'aae_experiments')
    os.makedirs(output_dir, exist_ok=True)

    # --- Experiment 1: Reconstruction ---
    print("[Experiment] 1. Reconstruction")
    with torch.no_grad():
        z = encoder(real_images)
        recon_images = decoder(z)
        
        comparison = torch.cat([real_images[:8], recon_images[:8]])
        save_image(comparison, os.path.join(output_dir, 'reconstruction.png'), nrow=8, normalize=True)
    print(f"Saved reconstruction.png")

    # --- Experiment 2: Latent Space Interpolation ---
    print("[Experiment] 2. Interpolation")
    with torch.no_grad():
        # Take first two images
        img1 = real_images[0].unsqueeze(0)
        img2 = real_images[1].unsqueeze(0)
        
        z1 = encoder(img1)
        z2 = encoder(img2)
        
        # Interpolate
        z_interp = interpolate_points(z1, z2, n_steps=10)
        imgs_interp = decoder(z_interp)
        
        save_image(imgs_interp, os.path.join(output_dir, 'interpolation.png'), nrow=10, normalize=True)
    print(f"Saved interpolation.png")

    # --- Experiment 3: Random Sampling (Generative Capability) ---
    print("[Experiment] 3. Random Sampling")
    with torch.no_grad():
        # Sample from Gaussian Prior N(0, 1)
        z_random = torch.randn(64, config.model_latent_dim).to(device)
        imgs_random = decoder(z_random)
        
        save_image(imgs_random, os.path.join(output_dir, 'random_samples.png'), nrow=8, normalize=True)
    print(f"Saved random_samples.png")

    # --- Experiment 4: Latent Space Arithmetic (Attribute Control) ---
    print("[Experiment] 4. Attribute Control")
    # We need to calculate the average latent vector for specific attributes
    # Let's try to find 'Smiling' and 'Eyeglasses' vectors
    
    # Load a larger batch to find attributes
    attr_dataloader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1024, # Load enough samples
        image_size=config.image_size,
        shuffle=False,
        max_samples=500 # Limit to speed up
    )
    
    # Get attribute labels
    attr_df = attr_dataloader.dataset.attr_df
    # Optimize lookup by setting index
    if attr_df.index.name != 'image_id':
        attr_df = attr_df.set_index('image_id')
    
    print("Calculating attribute vectors...")
    z_pos_smile = []
    z_neg_smile = []
    z_pos_beard = [] # Goatee (염소 수염)
    z_neg_beard = []
    
    encoder.eval()
    with torch.no_grad():
        for images, filenames in attr_dataloader:
            images = images.to(device)
            z = encoder(images)
            
            for i, fname in enumerate(filenames):
                # Find row in attr_df (Optimized)
                try:
                    row = attr_df.loc[fname]
                except KeyError:
                    continue
                
                if row['Smiling'] == 1:
                    z_pos_smile.append(z[i])
                else:
                    z_neg_smile.append(z[i])
                    
                if row['Goatee'] == 1:
                    z_pos_beard.append(z[i])
                else:
                    z_neg_beard.append(z[i])
    
    if not z_pos_smile or not z_neg_smile:
        print("[Warning] Could not find enough Smiling samples")
        return

    # Calculate mean vectors
    z_mean_pos_smile = torch.stack(z_pos_smile).mean(dim=0)
    z_mean_neg_smile = torch.stack(z_neg_smile).mean(dim=0)
    smile_vector = z_mean_pos_smile - z_mean_neg_smile
    
    # Beard Vector: (Has Goatee) - (No Goatee)
    z_mean_pos_beard = torch.stack(z_pos_beard).mean(dim=0)
    z_mean_neg_beard = torch.stack(z_neg_beard).mean(dim=0)
    beard_vector = z_mean_pos_beard - z_mean_neg_beard
    
    # Apply to a sample image
    target_img = real_images[0].unsqueeze(0)
    z_target = encoder(target_img)
    
    # 1. Make Smile
    z_smile = z_target + smile_vector * 1.5
    img_smile = decoder(z_smile)
    
    # 2. Add Beard (Add Goatee vector)
    z_beard = z_target + beard_vector * 2.0 
    img_beard = decoder(z_beard)
    
    # 3. Smile + Beard
    z_both = z_target + smile_vector * 1.5 + beard_vector * 2.0
    img_both = decoder(z_both)
    
    # Save results
    res = torch.cat([target_img, img_smile, img_beard, img_both])
    save_image(res, os.path.join(output_dir, 'attribute_control_beard_direct.png'), nrow=4, normalize=True)
    print(f"Saved attribute_control_beard_direct.png (Original -> Smile -> Beard(Goatee) -> Both)")

    # --- Experiment 5: Batch Beard Application ---
    print("[Experiment] 5. Batch Beard Application")
    # Select first 8 images from the batch
    batch_imgs = real_images[:8]
    z_batch = encoder(batch_imgs)
    
    # Add Beard Vector
    z_batch_beard = z_batch + beard_vector * 2.0
    imgs_batch_beard = decoder(z_batch_beard)
    
    # Concatenate Original and Beard images
    batch_res = torch.cat([batch_imgs, imgs_batch_beard])
    save_image(batch_res, os.path.join(output_dir, 'batch_beard_direct_control.png'), nrow=8, normalize=True)
    print(f"Saved batch_beard_direct_control.png (Top: Original, Bottom: With Beard)")

if __name__ == '__main__':
    run_experiments()
