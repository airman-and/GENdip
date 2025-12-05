"""
StyleGAN2-ADA 실험 스크립트
AAE의 experiment_aae.py와 동일한 실험들을 수행
"""
import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image
from torch.utils.data import DataLoader

# StyleGAN2-ADA-PyTorch 경로 추가
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan2_ada_pytorch')
sys.path.insert(0, stylegan_path)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import pickle
import glob
from core.dataset import get_celeba_loader

# Projection을 위한 import
import dnnlib
import legacy
from projector import project

def load_stylegan_model(device, model_path=None):
    """StyleGAN2-ADA 모델 로드"""
    if model_path is None:
        # 체크포인트 자동 찾기
        possible_paths = [
            os.path.join(config.project_dir, 'checkpoints', 'stylegan2_ada', 'network-snapshot-*.pkl'),
            os.path.join(config.output_path, 'stylegan2_ada_training', 'network-snapshot-*.pkl'),
        ]
        
        for pattern in possible_paths:
            matches = glob.glob(pattern)
            if matches:
                model_path = max(matches, key=os.path.getmtime)
                break
    
    if model_path is None or not os.path.exists(model_path):
        print(f"[Error] StyleGAN2-ADA model not found")
        print(f"[Info] Please train first: python src/train_stylegan.py")
        return None
    
    print(f"[Experiment] Loading model from {model_path}")
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    return G

def project_image(G, image_tensor, device, num_steps=500):
    """
    이미지 텐서를 W space로 투영
    image_tensor: [C, H, W], range [0, 1]
    """
    # [0, 1] -> [0, 255]
    img_255 = (image_tensor * 255.0).clamp(0, 255).byte()
    
    try:
        w = project(G, img_255, num_steps=num_steps, device=device, verbose=False)
        return w
    except Exception as e:
        print(f"[Warning] Projection failed: {e}")
        return None

def interpolate_points_w(w1, w2, n_steps=10):
    """W space에서 보간"""
    ratios = np.linspace(0, 1, n_steps)
    vectors = []
    for ratio in ratios:
        w = (1.0 - ratio) * w1 + ratio * w2
        vectors.append(w)
    return torch.stack(vectors)

def run_experiments():
    """AAE와 동일한 실험들을 StyleGAN2-ADA로 수행"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Experiment] Running on {device}")
    
    # 1. Load Model
    G = load_stylegan_model(device)
    if G is None:
        return
    
    # 2. Load Data
    if not os.path.exists(config.celebA_image_path):
        print(f"[Error] Dataset not found at {config.celebA_image_path}")
        return
    
    dataloader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=16,
        image_size=G.img_resolution,
        shuffle=True
    )
    
    # Get a batch
    try:
        real_images, _ = next(iter(dataloader))
        real_images = real_images.to(device)
    except Exception as e:
        print(f"[Error] Could not load data: {e}")
        return
    
    output_dir = os.path.join(config.output_path, 'stylegan2_ada_experiments')
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Experiment 1: Reconstruction ---
    print("[Experiment] 1. Reconstruction (Projecting images to W space...)")
    print("[Info] This may take a while (30-60 seconds per image)...")
    
    recon_images = []
    projected_count = 0
    for i in range(min(8, len(real_images))):
        img = real_images[i]
        w = project_image(G, img, device, num_steps=500)
        if w is not None:
            with torch.no_grad():
                recon = G.synthesis(w, noise_mode='const')
                recon = (recon + 1.0) / 2.0
                recon = torch.clamp(recon, 0, 1)
                recon_images.append(recon)
                projected_count += 1
                print(f"  Projected {projected_count}/8 images...")
    
    if recon_images:
        comparison = torch.cat([real_images[:len(recon_images)], torch.cat(recon_images, dim=0)])
        save_image(comparison, os.path.join(output_dir, 'reconstruction.png'), 
                  nrow=len(recon_images), normalize=False)
        print(f"Saved reconstruction.png")
    else:
        print("[Warning] Could not reconstruct any images")
    
    # --- Experiment 2: Latent Space Interpolation ---
    print("[Experiment] 2. Interpolation")
    print("[Info] Projecting two images for interpolation...")
    
    img1 = real_images[0]
    img2 = real_images[1]
    
    w1 = project_image(G, img1, device, num_steps=500)
    w2 = project_image(G, img2, device, num_steps=500)
    
    if w1 is not None and w2 is not None:
        # W space에서 보간
        w_interp = interpolate_points_w(w1, w2, n_steps=10)
        
        with torch.no_grad():
            imgs_interp = []
            for w in w_interp:
                img = G.synthesis(w.unsqueeze(0), noise_mode='const')
                img = (img + 1.0) / 2.0
                img = torch.clamp(img, 0, 1)
                imgs_interp.append(img)
            
            imgs_interp = torch.cat(imgs_interp, dim=0)
            save_image(imgs_interp, os.path.join(output_dir, 'interpolation.png'), 
                      nrow=10, normalize=False)
        print(f"Saved interpolation.png")
    else:
        print("[Warning] Could not project images for interpolation")
    
    # --- Experiment 3: Random Sampling ---
    print("[Experiment] 3. Random Sampling")
    with torch.no_grad():
        z_random = torch.randn(64, G.z_dim).to(device)
        imgs_random = G(z_random, None, truncation_psi=1.0)
        imgs_random = (imgs_random + 1.0) / 2.0
        imgs_random = torch.clamp(imgs_random, 0, 1)
        
        save_image(imgs_random, os.path.join(output_dir, 'random_samples.png'), 
                  nrow=8, normalize=False)
    print(f"Saved random_samples.png")
    
    # --- Experiment 4: Attribute Control ---
    print("[Experiment] 4. Attribute Control")
    print("[Info] This will take a while (projecting many images)...")
    
    # Load larger batch for attribute calculation
    attr_dataloader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,  # 한 번에 하나씩 projection
        image_size=G.img_resolution,
        shuffle=False,
        max_samples=100  # 속도 향상을 위해 줄임
    )
    
    # Get attribute labels
    attr_df = attr_dataloader.dataset.attr_df
    if attr_df.index.name != 'image_id':
        attr_df = attr_df.set_index('image_id')
    
    print("Calculating attribute vectors in W space...")
    w_pos_smile = []
    w_neg_smile = []
    w_pos_beard = []
    w_neg_beard = []
    
    count = 0
    for images, filenames in attr_dataloader:
        if len(w_pos_smile) >= 20 and len(w_neg_smile) >= 20 and \
           len(w_pos_beard) >= 20 and len(w_neg_beard) >= 20:
            break
        
        for i, fname in enumerate(filenames):
            if len(w_pos_smile) >= 20 and len(w_neg_smile) >= 20 and \
               len(w_pos_beard) >= 20 and len(w_neg_beard) >= 20:
                break
            
            try:
                row = attr_df.loc[fname]
            except KeyError:
                continue
            
            img = images[i].to(device)
            w = project_image(G, img, device, num_steps=300)  # 속도 향상
            
            if w is None:
                continue
            
            w_mean = w.mean(dim=1, keepdim=True)  # [1, 1, w_dim]
            count += 1
            
            if count % 10 == 0:
                print(f"  Processed {count} images...")
            
            if row['Smiling'] == 1 and len(w_pos_smile) < 20:
                w_pos_smile.append(w_mean.cpu())
            elif row['Smiling'] == -1 and len(w_neg_smile) < 20:
                w_neg_smile.append(w_mean.cpu())
            
            if row['Goatee'] == 1 and len(w_pos_beard) < 20:
                w_pos_beard.append(w_mean.cpu())
            elif row['Goatee'] == -1 and len(w_neg_beard) < 20:
                w_neg_beard.append(w_mean.cpu())
    
    if not w_pos_smile or not w_neg_smile:
        print("[Warning] Could not find enough samples for attribute vectors")
        return
    
    # Calculate mean vectors
    w_mean_pos_smile = torch.cat(w_pos_smile, dim=0).mean(dim=0).to(device)  # [1, w_dim]
    w_mean_neg_smile = torch.cat(w_neg_smile, dim=0).mean(dim=0).to(device)
    smile_vector = (w_mean_pos_smile - w_mean_neg_smile).squeeze(0)  # [w_dim]
    
    w_mean_pos_beard = torch.cat(w_pos_beard, dim=0).mean(dim=0).to(device)
    w_mean_neg_beard = torch.cat(w_neg_beard, dim=0).mean(dim=0).to(device)
    beard_vector = (w_mean_pos_beard - w_mean_neg_beard).squeeze(0)
    
    # Apply to a sample image
    print("Projecting target image...")
    target_img = real_images[0].unsqueeze(0)
    w_target = project_image(G, target_img.squeeze(0), device, num_steps=500)
    
    if w_target is not None:
        w_target_mean = w_target.mean(dim=1, keepdim=True)  # [1, 1, w_dim]
        
        with torch.no_grad():
            # 1. Make Smile
            w_smile = w_target_mean.clone()
            w_smile = w_smile + smile_vector.unsqueeze(0).unsqueeze(0) * 1.5
            w_smile = w_smile.repeat(1, G.mapping.num_ws, 1)
            img_smile = G.synthesis(w_smile, noise_mode='const')
            img_smile = (img_smile + 1.0) / 2.0
            img_smile = torch.clamp(img_smile, 0, 1)
            
            # 2. Add Beard
            w_beard = w_target_mean.clone()
            w_beard = w_beard + beard_vector.unsqueeze(0).unsqueeze(0) * 2.0
            w_beard = w_beard.repeat(1, G.mapping.num_ws, 1)
            img_beard = G.synthesis(w_beard, noise_mode='const')
            img_beard = (img_beard + 1.0) / 2.0
            img_beard = torch.clamp(img_beard, 0, 1)
            
            # 3. Smile + Beard
            w_both = w_target_mean.clone()
            w_both = w_both + smile_vector.unsqueeze(0).unsqueeze(0) * 1.5 + \
                     beard_vector.unsqueeze(0).unsqueeze(0) * 2.0
            w_both = w_both.repeat(1, G.mapping.num_ws, 1)
            img_both = G.synthesis(w_both, noise_mode='const')
            img_both = (img_both + 1.0) / 2.0
            img_both = torch.clamp(img_both, 0, 1)
        
        # Save results
        res = torch.cat([target_img, img_smile, img_beard, img_both])
        save_image(res, os.path.join(output_dir, 'attribute_control_beard_direct.png'), 
                  nrow=4, normalize=False)
        print(f"Saved attribute_control_beard_direct.png (Original -> Smile -> Beard(Goatee) -> Both)")
    
    # --- Experiment 5: Batch Beard Application ---
    print("[Experiment] 5. Batch Beard Application")
    print("[Info] Projecting batch images...")
    
    batch_imgs = real_images[:8]
    batch_w = []
    
    for img in batch_imgs:
        w = project_image(G, img, device, num_steps=300)
        if w is not None:
            batch_w.append(w.mean(dim=1, keepdim=True))
    
    if batch_w and beard_vector is not None:
        with torch.no_grad():
            imgs_batch_beard = []
            for w in batch_w:
                w_beard = w.clone().to(device)
                w_beard = w_beard + beard_vector.unsqueeze(0).unsqueeze(0) * 2.0
                w_beard = w_beard.repeat(1, G.mapping.num_ws, 1)
                img_beard = G.synthesis(w_beard, noise_mode='const')
                img_beard = (img_beard + 1.0) / 2.0
                img_beard = torch.clamp(img_beard, 0, 1)
                imgs_batch_beard.append(img_beard)
            
            if imgs_batch_beard:
                batch_res = torch.cat([batch_imgs[:len(imgs_batch_beard)], 
                                      torch.cat(imgs_batch_beard, dim=0)])
                save_image(batch_res, os.path.join(output_dir, 'batch_beard_direct_control.png'), 
                          nrow=8, normalize=False)
                print(f"Saved batch_beard_direct_control.png (Top: Original, Bottom: With Beard)")
    else:
        print("[Warning] Could not complete batch beard application")

if __name__ == '__main__':
    run_experiments()

