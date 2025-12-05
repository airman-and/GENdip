"""
StyleGAN2-ADA 메인 스크립트 (GPU 최적화 버전)
멀티프로세싱으로 여러 GPU를 활용하여 병렬 projection 수행
"""
import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image
import glob
from multiprocessing import Pool, Manager
from functools import partial
import time

# StyleGAN2-ADA-PyTorch 경로 추가
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan2_ada_pytorch')
sys.path.insert(0, stylegan_path)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import pickle
from core.dataset import get_celeba_loader
from projector import project

def load_stylegan_model(model_path, device):
    """StyleGAN2-ADA 모델 로드"""
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    return G

def project_single_image(args):
    """
    단일 이미지를 projection하는 헬퍼 함수 (멀티프로세싱용)
    
    Args:
        args: (model_path, image_tensor, num_steps, gpu_id) 튜플
    
    Returns:
        w_mean: W space 벡터의 평균
    """
    model_path, image_tensor, num_steps, gpu_id = args
    
    # 각 프로세스에서 GPU 설정
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)
    
    try:
        # 모델 로드 (각 프로세스에서 독립적으로)
        with open(model_path, 'rb') as f:
            G = pickle.load(f)['G_ema'].to(device)
        G.eval()
        
        # 이미지 전처리
        img_255 = (image_tensor * 255.0).clamp(0, 255).byte()
        
        # Projection 수행
        w = project(G, img_255, num_steps=num_steps, device=device, verbose=False)
        
        if w is not None:
            w_mean = w.mean(dim=1, keepdim=True).cpu()
            return w_mean.numpy()
        else:
            return None
    except Exception as e:
        print(f"[Error] GPU {gpu_id} projection failed: {e}", flush=True)
        return None

def project_images_parallel(model_path, images, num_steps=300, num_gpus=None):
    """
    여러 이미지를 병렬로 projection
    
    Args:
        model_path: 모델 경로
        images: 이미지 텐서 리스트 [N, C, H, W]
        num_steps: projection 단계 수
        num_gpus: 사용할 GPU 개수 (None이면 자동 감지)
    
    Returns:
        w_means: W space 벡터 리스트
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    num_gpus = min(num_gpus, len(images), torch.cuda.device_count())
    
    print(f"[Parallel] Using {num_gpus} GPU(s) for parallel projection")
    print(f"[Parallel] Processing {len(images)} images...")
    
    # 이미지를 CPU로 이동하고 numpy로 변환
    images_cpu = [img.cpu() for img in images]
    
    # 각 이미지에 GPU ID 할당 (라운드로빈)
    tasks = []
    for i, img in enumerate(images_cpu):
        gpu_id = i % num_gpus
        tasks.append((model_path, img, num_steps, gpu_id))
    
    # 병렬 처리
    start_time = time.time()
    with Pool(processes=num_gpus) as pool:
        results = pool.map(project_single_image, tasks)
    
    elapsed_time = time.time() - start_time
    print(f"[Parallel] Completed {len([r for r in results if r is not None])}/{len(images)} projections in {elapsed_time:.1f}s")
    
    # 결과를 텐서로 변환
    w_means = []
    for r in results:
        if r is not None:
            w_means.append(torch.from_numpy(r))
    
    return w_means

def extract_attribute_vector_w_optimized(G, model_path, dataloader_pos, dataloader_neg, attr_name, device, num_samples=50, num_gpus=None):
    """
    GPU 최적화된 속성 벡터 추출
    
    Args:
        G: Generator (참고용, 실제로는 model_path 사용)
        model_path: 모델 파일 경로
        dataloader_pos: 양성 샘플 데이터로더
        dataloader_neg: 음성 샘플 데이터로더
        attr_name: 속성 이름
        device: 메인 디바이스
        num_samples: 샘플 수
        num_gpus: 사용할 GPU 개수
    """
    print(f"[Attribute] Extracting '{attr_name}' vector in W space (GPU optimized)...")
    
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    
    # 양성 샘플 수집
    print(f"[Info] Collecting positive samples...")
    pos_images = []
    for images, _ in dataloader_pos:
        if len(pos_images) >= num_samples:
            break
        pos_images.append(images[0].to('cpu'))  # CPU로 이동 (멀티프로세싱용)
        if len(pos_images) >= num_samples:
            break
    
    # 음성 샘플 수집
    print(f"[Info] Collecting negative samples...")
    neg_images = []
    for images, _ in dataloader_neg:
        if len(neg_images) >= num_samples:
            break
        neg_images.append(images[0].to('cpu'))
        if len(neg_images) >= num_samples:
            break
    
    print(f"[Info] Collected {len(pos_images)} positive and {len(neg_images)} negative samples")
    print(f"[Info] Starting parallel projection on {num_gpus} GPU(s)...")
    
    # 병렬 projection
    w_pos = project_images_parallel(model_path, pos_images, num_steps=300, num_gpus=num_gpus)
    w_neg = project_images_parallel(model_path, neg_images, num_steps=300, num_gpus=num_gpus)
    
    if not w_pos or not w_neg:
        print(f"[Error] Could not find enough samples (pos: {len(w_pos)}, neg: {len(w_neg)})")
        return None
    
    # 속성 벡터 계산
    w_pos_mean = torch.cat(w_pos, dim=0).mean(dim=0).squeeze(0).to(device)
    w_neg_mean = torch.cat(w_neg, dim=0).mean(dim=0).squeeze(0).to(device)
    attr_vector = (w_pos_mean - w_neg_mean).to(device)
    
    print(f"[Attribute] Extracted vector from {len(w_pos)} positive and {len(w_neg)} negative samples")
    return attr_vector

def manipulate_image_w(G, image, attribute_vector, scale=3.0, device='cuda'):
    """
    StyleGAN2-ADA에서 이미지 편집
    AAE의 manipulate_image와 동일한 인터페이스
    """
    # 이미지를 W space로 투영
    img_255 = (image * 255.0).clamp(0, 255).byte()
    w = project(G, img_255, num_steps=500, device=device, verbose=False)
    
    if w is None:
        return None
    
    w_mean = w.mean(dim=1, keepdim=True)  # [1, 1, w_dim]
    
    # 속성 벡터 적용
    w_modified = w_mean.clone()
    w_modified = w_modified + attribute_vector.unsqueeze(0).unsqueeze(0) * scale
    w_modified = w_modified.repeat(1, G.mapping.num_ws, 1)
    
    # 이미지 생성
    with torch.no_grad():
        img_modified = G.synthesis(w_modified, noise_mode='const')
        img_modified = (img_modified + 1.0) / 2.0
        img_modified = torch.clamp(img_modified, 0, 1)
    
    return img_modified

def main():
    """GPU 최적화된 메인 함수"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    print(f"[Main] Device: {device}")
    print(f"[Main] Available GPUs: {num_gpus}")
    
    # 모델 경로 찾기
    model_path = None
    possible_paths = [
        os.path.join(config.project_dir, 'checkpoints', 'stylegan2_ada', 'network-snapshot-*.pkl'),
        os.path.join(config.output_path, 'stylegan2_ada_training', '**', 'network-snapshot-*.pkl'),
        os.path.join(config.output_path, 'stylegan2_ada_training', 'network-snapshot-*.pkl'),
    ]
    
    for pattern in possible_paths:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            model_path = max(matches, key=os.path.getmtime)
            break
    
    if model_path is None:
        print(f"[Main] StyleGAN2-ADA model not found")
        print(f"[Main] Please train first: python src/train_stylegan.py")
        return
    
    # Load model (메인 프로세스용)
    print(f"[Main] Loading StyleGAN2-ADA model from {model_path}")
    G = load_stylegan_model(model_path, device)
    print("[Main] Model loaded successfully")
    
    # Check dataset
    if not os.path.exists(config.celebA_image_path):
        print(f"[Main] Dataset not found at {config.celebA_image_path}")
        return
    
    # Extract attribute vector (GPU 최적화 버전)
    print("\n[Main] Extracting 'Smiling' attribute vector (GPU optimized)...")
    
    # Load data for attribute extraction
    attr_loader_pos = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        filter_attr="Smiling",
        filter_value=1,
        shuffle=False,
        max_samples=50
    )
    
    attr_loader_neg = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        filter_attr="Smiling",
        filter_value=-1,
        shuffle=False,
        max_samples=50
    )
    
    # 병렬 속성 벡터 추출
    smiling_vector = extract_attribute_vector_w_optimized(
        G=G,
        model_path=model_path,
        dataloader_pos=attr_loader_pos,
        dataloader_neg=attr_loader_neg,
        attr_name="Smiling",
        device=device,
        num_samples=50,
        num_gpus=num_gpus
    )
    
    if smiling_vector is None:
        print("[Error] Could not extract attribute vector")
        return
    
    # Load test images
    print("\n[Main] Loading test images...")
    test_loader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=8,
        image_size=G.img_resolution,
        filter_attr="Smiling",
        filter_value=-1,
        shuffle=True,
        max_samples=8
    )
    
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    
    # Manipulate images (테스트 이미지는 적으므로 순차 처리)
    print("\n[Main] Manipulating test images...")
    results = []
    
    # Original images
    results.append(test_images)
    
    # Edited images with different scales
    for scale in [1.0, 2.0, 3.0]:
        edited_batch = []
        for img in test_images:
            edited = manipulate_image_w(G, img, smiling_vector, scale=scale, device=device)
            if edited is not None:
                edited_batch.append(edited)
        
        if edited_batch:
            results.append(torch.cat(edited_batch, dim=0))
    
    # Save results
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = torch.cat(results, dim=0)
    output_path_file = os.path.join(output_dir, 'stylegan_latent_control_smiling_optimized.png')
    
    save_image(
        all_results,
        output_path_file,
        nrow=8,
        normalize=False
    )
    
    print(f"\n[Main] Results saved to {output_path_file}")
    print("[Main] Rows from top to bottom:")
    print("  - Row 1: Original images (no smiling)")
    print("  - Row 2: Scale 1.0 (slight smile)")
    print("  - Row 3: Scale 2.0 (moderate smile)")
    print("  - Row 4: Scale 3.0 (strong smile)")

if __name__ == '__main__':
    main()

