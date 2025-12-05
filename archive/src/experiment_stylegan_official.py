"""
StyleGAN2-ADA 공식 구현을 사용한 실험 스크립트
"""
import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image

# StyleGAN2-ADA-PyTorch 경로 추가
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan2_ada_pytorch')
sys.path.insert(0, stylegan_path)

import config
import pickle

def load_stylegan_model(model_path, device='cuda'):
    """StyleGAN2-ADA 모델 로드"""
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    return G

def generate_images(G, num_images=16, truncation_psi=1.0, device='cuda'):
    """랜덤 이미지 생성"""
    z = torch.randn([num_images, G.z_dim]).to(device)
    c = None  # class labels (not used for unconditional)
    
    with torch.no_grad():
        img = G(z, c, truncation_psi=truncation_psi)
        # [-1, 1] -> [0, 1]
        img = (img + 1.0) / 2.0
        img = torch.clamp(img, 0, 1)
    
    return img

def style_mixing(G, row_seeds, col_seeds, truncation_psi=1.0, device='cuda'):
    """Style mixing 실험"""
    all_seeds = sorted(set(row_seeds + col_seeds))
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_z = torch.from_numpy(all_z).to(device)
    
    all_w = G.mapping(all_z, None, truncation_psi=truncation_psi)
    w_avg = G.mapping.w_avg
    
    w_dict = {seed: w for seed, w in zip(all_seeds, all_w)}
    
    images = []
    for row_seed in row_seeds:
        w_row = w_dict[row_seed].unsqueeze(0).repeat(len(col_seeds), 1, 1)
        for col_seed in col_seeds:
            w_col = w_dict[col_seed]
            # Style mixing: early layers from row, later layers from col
            w_mixed = w_row[0].clone()
            w_mixed[8:] = w_col[8:]  # Mix after layer 8
            
            with torch.no_grad():
                img = G.synthesis(w_mixed.unsqueeze(0), noise_mode='const')
                img = (img + 1.0) / 2.0
                img = torch.clamp(img, 0, 1)
                images.append(img)
    
    return torch.cat(images, dim=0)

def interpolate_latents(G, z1, z2, num_steps=10, truncation_psi=1.0, device='cuda'):
    """Latent space 보간"""
    z1 = torch.from_numpy(z1).unsqueeze(0).to(device) if isinstance(z1, np.ndarray) else z1
    z2 = torch.from_numpy(z2).unsqueeze(0).to(device) if isinstance(z2, np.ndarray) else z2
    
    w1 = G.mapping(z1, None, truncation_psi=truncation_psi)
    w2 = G.mapping(z2, None, truncation_psi=truncation_psi)
    
    alphas = np.linspace(0, 1, num_steps)
    images = []
    
    for alpha in alphas:
        w = (1 - alpha) * w1 + alpha * w2
        with torch.no_grad():
            img = G.synthesis(w, noise_mode='const')
            img = (img + 1.0) / 2.0
            img = torch.clamp(img, 0, 1)
            images.append(img)
    
    return torch.cat(images, dim=0)

def run_experiments():
    """StyleGAN2-ADA 실험 실행"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Experiment] Running on {device}")
    
    # 모델 경로 찾기
    model_path = None
    possible_paths = [
        os.path.join(config.project_dir, 'checkpoints', 'stylegan2_ada', 'network-snapshot-*.pkl'),
        os.path.join(config.output_path, 'stylegan2_ada_training', '*.pkl'),
    ]
    
    # 실제로는 glob으로 찾아야 함
    import glob
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            # 가장 최신 것 선택
            model_path = max(matches, key=os.path.getmtime)
            break
    
    if model_path is None:
        print("[Error] StyleGAN2-ADA 모델을 찾을 수 없습니다.")
        print("[Info] 사전 학습된 모델을 다운로드하거나 학습을 먼저 실행하세요.")
        print("[Info] 예: python train_stylegan_official.py --dataset=dataset/celebA.zip")
        return
    
    print(f"[Experiment] Loading model from {model_path}")
    G = load_stylegan_model(model_path, device)
    
    output_dir = os.path.join(config.output_path, 'stylegan2_ada_experiments')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 랜덤 샘플링
    print("[Experiment] 1. Random Sampling")
    images = generate_images(G, num_images=16, truncation_psi=1.0, device=device)
    save_image(images, os.path.join(output_dir, 'random_samples.png'), nrow=4, normalize=False)
    print("Saved random_samples.png")
    
    # 2. Truncation trick 비교
    print("[Experiment] 2. Truncation Trick Comparison")
    truncation_values = [1.0, 0.7, 0.5, 0.3]
    all_images = []
    for trunc in truncation_values:
        imgs = generate_images(G, num_images=4, truncation_psi=trunc, device=device)
        all_images.append(imgs)
    save_image(torch.cat(all_images, dim=0), os.path.join(output_dir, 'truncation_comparison.png'), 
               nrow=4, normalize=False)
    print("Saved truncation_comparison.png")
    
    # 3. Style Mixing
    print("[Experiment] 3. Style Mixing")
    row_seeds = [85, 100, 75, 458]
    col_seeds = [55, 821, 1789, 293]
    mixed_images = style_mixing(G, row_seeds, col_seeds, device=device)
    save_image(mixed_images, os.path.join(output_dir, 'style_mixing.png'), 
               nrow=len(col_seeds), normalize=False)
    print("Saved style_mixing.png")
    
    # 4. Latent Interpolation
    print("[Experiment] 4. Latent Interpolation")
    z1 = np.random.RandomState(42).randn(G.z_dim)
    z2 = np.random.RandomState(123).randn(G.z_dim)
    interp_images = interpolate_latents(G, z1, z2, num_steps=10, device=device)
    save_image(interp_images, os.path.join(output_dir, 'interpolation.png'), 
               nrow=10, normalize=False)
    print("Saved interpolation.png")
    
    print("[Experiment] All experiments completed!")

if __name__ == '__main__':
    run_experiments()

