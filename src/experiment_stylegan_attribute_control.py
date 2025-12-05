"""
StyleGAN2-ADA에서 AAE처럼 속성 벡터를 사용한 이미지 편집
실제 이미지를 projection한 후 속성 벡터를 적용하여 편집
"""
import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as transforms

# StyleGAN2-ADA-PyTorch 경로 추가
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan2_ada_pytorch')
sys.path.insert(0, stylegan_path)

import config
import pickle
import dnnlib
import legacy
from projector import project

def load_stylegan_model(model_path, device='cuda'):
    """StyleGAN2-ADA 모델 로드"""
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    return G

def project_image_to_w(G, image_path, device='cuda', num_steps=1000):
    """
    실제 이미지를 W space로 투영
    
    Args:
        G: StyleGAN2-ADA Generator
        image_path: 이미지 경로
        device: device
        num_steps: projection 단계 수
    
    Returns:
        w: W space 벡터 [1, num_ws, w_dim]
    """
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('RGB')
    img = img.resize((G.img_resolution, G.img_resolution), Image.LANCZOS)
    img = np.array(img).transpose(2, 0, 1)  # HWC -> CHW
    img = torch.from_numpy(img).float().to(device)  # [C, H, W], [0, 255]
    
    # Projection 수행
    print(f"[Projection] Projecting image to W space (this may take a while)...")
    w = project(
        G,
        target=img,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    
    return w

def extract_attribute_vector_w(G, dataloader, attr_name, device='cuda', num_samples=50):
    """
    StyleGAN2-ADA에서 속성 벡터 추출 (W space)
    
    AAE와 동일한 방식이지만 W space에서 수행
    """
    print(f"[Attribute] Extracting '{attr_name}' vector in W space...")
    print(f"[Attribute] This will project {num_samples * 2} images (may take a while)...")
    
    w_pos = []
    w_neg = []
    
    G.eval()
    count_pos = 0
    count_neg = 0
    
    with torch.no_grad():
        for images, filenames in dataloader:
            if count_pos >= num_samples and count_neg >= num_samples:
                break
                
            for i, fname in enumerate(filenames):
                if count_pos >= num_samples and count_neg >= num_samples:
                    break
                
                # 속성 확인
                try:
                    row = dataloader.dataset.attr_df.loc[fname]
                    has_attr = row[attr_name] == 1
                except:
                    continue
                
                # 이미지를 W space로 투영
                img = images[i].unsqueeze(0).to(device)
                img = (img * 255.0).clamp(0, 255).byte()  # [0, 1] -> [0, 255]
                img = img.squeeze(0)  # [C, H, W]
                
                try:
                    w = project(G, img, num_steps=500, device=device, verbose=False)
                    w_mean = w.mean(dim=1, keepdim=True)  # [1, 1, w_dim]
                    
                    if has_attr and count_pos < num_samples:
                        w_pos.append(w_mean.cpu())
                        count_pos += 1
                    elif not has_attr and count_neg < num_samples:
                        w_neg.append(w_mean.cpu())
                        count_neg += 1
                except Exception as e:
                    print(f"[Warning] Failed to project {fname}: {e}")
                    continue
    
    if not w_pos or not w_neg:
        print(f"[Error] Could not find enough samples (pos: {len(w_pos)}, neg: {len(w_neg)})")
        return None
    
    # 평균 계산
    w_pos_mean = torch.cat(w_pos, dim=0).mean(dim=0)  # [1, w_dim]
    w_neg_mean = torch.cat(w_neg, dim=0).mean(dim=0)  # [1, w_dim]
    attr_vector = (w_pos_mean - w_neg_mean).to(device)  # [1, w_dim]
    
    print(f"[Attribute] Extracted vector from {len(w_pos)} positive and {len(w_neg)} negative samples")
    
    return attr_vector

def manipulate_w_space(G, w, attr_vector, scale=1.0, device='cuda'):
    """
    W space에서 속성 벡터를 적용하여 이미지 편집
    
    Args:
        G: Generator
        w: W space 벡터 [1, num_ws, w_dim]
        attr_vector: 속성 벡터 [1, w_dim]
        scale: 속성 벡터 스케일
        device: device
    
    Returns:
        edited_image: 편집된 이미지
    """
    # W space에 속성 벡터 추가
    w_edited = w.clone()
    w_edited = w_edited + attr_vector.unsqueeze(0) * scale  # [1, num_ws, w_dim]
    
    # 이미지 생성
    with torch.no_grad():
        img = G.synthesis(w_edited, noise_mode='const')
        img = (img + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        img = torch.clamp(img, 0, 1)
    
    return img

def run_attribute_control_experiment():
    """AAE와 동일한 방식의 속성 제어 실험"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Experiment] Running on {device}")
    
    # 모델 로드
    import glob
    model_path = None
    possible_paths = [
        os.path.join(config.project_dir, 'checkpoints', 'stylegan2_ada', 'network-snapshot-*.pkl'),
        os.path.join(config.output_path, 'stylegan2_ada_training', '*.pkl'),
    ]
    
    for pattern in possible_paths:
        matches = glob.glob(pattern)
        if matches:
            model_path = max(matches, key=os.path.getmtime)
            break
    
    if model_path is None:
        print("[Error] StyleGAN2-ADA 모델을 찾을 수 없습니다.")
        return
    
    print(f"[Experiment] Loading model from {model_path}")
    G = load_stylegan_model(model_path, device)
    
    # 데이터 로드
    from core.dataset import get_celeba_loader
    
    if not os.path.exists(config.celebA_image_path):
        print(f"[Error] Dataset not found")
        return
    
    # 속성 벡터 추출을 위한 데이터로더
    attr_dataloader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        shuffle=False,
        max_samples=100  # 속성 벡터 계산용
    )
    
    # 속성 벡터 추출 (시간이 오래 걸릴 수 있음)
    print("\n[Experiment] Extracting attribute vectors...")
    smile_vector = extract_attribute_vector_w(
        G, attr_dataloader, 'Smiling', device=device, num_samples=20
    )
    
    if smile_vector is None:
        print("[Error] Failed to extract attribute vector")
        return
    
    # 테스트 이미지 로드 및 projection
    test_loader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        shuffle=True,
        max_samples=8
    )
    
    print("\n[Experiment] Projecting test images to W space...")
    test_images = []
    test_w = []
    
    for i, (images, filenames) in enumerate(test_loader):
        if i >= 8:
            break
        
        img = images[0].to(device)
        img_255 = (img * 255.0).clamp(0, 255).byte().squeeze(0)  # [C, H, W]
        
        try:
            w = project(G, img_255, num_steps=500, device=device, verbose=False)
            test_images.append(img)
            test_w.append(w)
            print(f"  Projected image {i+1}/8")
        except Exception as e:
            print(f"  [Warning] Failed to project image {i+1}: {e}")
            continue
    
    if not test_w:
        print("[Error] No images were successfully projected")
        return
    
    # 속성 적용
    print("\n[Experiment] Applying attribute vectors...")
    results = []
    
    for img, w in zip(test_images, test_w):
        # 원본
        results.append(img)
        
        # 속성 적용 (다양한 스케일)
        for scale in [0.5, 1.0, 1.5]:
            edited = manipulate_w_space(G, w, smile_vector, scale=scale, device=device)
            results.append(edited)
    
    # 결과 저장
    output_dir = os.path.join(config.output_path, 'stylegan2_ada_attribute_control')
    os.makedirs(output_dir, exist_ok=True)
    
    results_tensor = torch.cat(results, dim=0)
    save_image(
        results_tensor,
        os.path.join(output_dir, 'smile_control.png'),
        nrow=4,  # 원본 + 3개 스케일
        normalize=False
    )
    
    print(f"\n[Experiment] Results saved to {output_dir}/smile_control.png")
    print("[Experiment] Each row: Original -> Scale 0.5 -> Scale 1.0 -> Scale 1.5")

if __name__ == '__main__':
    run_attribute_control_experiment()

