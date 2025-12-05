"""
StyleGAN2-ADA 메인 스크립트
AAE의 main.py와 동일한 기능 제공
멀티프로세싱으로 두 GPU를 동시에 활용
"""
import os
import sys
import torch
import numpy as np
from torchvision.utils import save_image
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import multiprocessing as mp

# StyleGAN2-ADA-PyTorch 경로 추가
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan2_ada_pytorch')
sys.path.insert(0, stylegan_path)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
import pickle
from core.dataset import get_celeba_loader
from projector import project

def project_on_gpu(args):
    """특정 GPU에서 이미지 projection (멀티프로세싱용)"""
    img_tensor, gpu_id, model_path, num_steps = args
    try:
        # 각 프로세스에서 CUDA 초기화
        torch.cuda.set_device(gpu_id)
        current_device = f'cuda:{gpu_id}'
        
        # 모델 로드 (프로세스별로 독립적)
        with open(model_path, 'rb') as f:
            G_local = pickle.load(f)['G_ema'].to(current_device)
        G_local.eval()
        
        img = img_tensor.to(current_device)
        img_255 = (img * 255.0).clamp(0, 255).byte()
        w = project(G_local, img_255, num_steps=num_steps, device=current_device, verbose=False)
        
        if w is not None:
            return (w.mean(dim=1, keepdim=True).cpu().numpy(), gpu_id)
        return (None, gpu_id)
    except Exception as e:
        print(f"[Warning] GPU {gpu_id} projection failed: {e}", flush=True)
        return (None, gpu_id)

def manipulate_on_gpu(args):
    """특정 GPU에서 이미지 manipulation (멀티프로세싱용)"""
    img_cpu, scale, gpu_id, img_idx, scale_idx, model_path_arg, attr_vector_np, proj_steps = args
    try:
        torch.cuda.set_device(gpu_id)
        current_device = f'cuda:{gpu_id}'
        
        # 모델 로드 (각 프로세스에서 독립적으로)
        with open(model_path_arg, 'rb') as f:
            G_local = pickle.load(f)['G_ema'].to(current_device)
        G_local.eval()
        
        img = img_cpu.to(current_device)
        attr_vector_local = torch.from_numpy(attr_vector_np).to(current_device)
        
        edited = manipulate_image_w(
            G_local, img, attr_vector_local,
            scale=scale,
            device=current_device,
            normalize_attr=True,
            truncation_psi=0.7,
            projection_steps=proj_steps
        )
        if edited is not None:
            return (edited.cpu(), img_idx, scale_idx)
        return (None, img_idx, scale_idx)
    except Exception as e:
        print(f"[Warning] GPU {gpu_id} manipulation failed (img {img_idx}, scale {scale}): {e}", flush=True)
        return (None, img_idx, scale_idx)

def load_stylegan_model(model_path, device):
    """StyleGAN2-ADA 모델 로드"""
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    G.eval()
    return G

def extract_attribute_vector_w(G, dataloader, attr_name, device, num_samples=50):
    """
    StyleGAN2-ADA에서 속성 벡터 추출 (W space)
    AAE의 extract_attribute_vector와 동일한 인터페이스
    """
    print(f"[Attribute] Extracting '{attr_name}' vector in W space...")
    print(f"[Info] This will project {num_samples * 2} images (may take a while)...")
    
    w_pos = []
    w_neg = []
    
    attr_df = dataloader.dataset.attr_df
    if attr_df.index.name != 'image_id':
        attr_df = attr_df.set_index('image_id')
    
    count = 0
    for images, filenames in dataloader:
        if len(w_pos) >= num_samples and len(w_neg) >= num_samples:
            break
        
        for i, fname in enumerate(filenames):
            if len(w_pos) >= num_samples and len(w_neg) >= num_samples:
                break
            
            try:
                row = attr_df.loc[fname]
                has_attr = row[attr_name] == 1
            except:
                continue
            
            img = images[i].to(device)
            img_255 = (img * 255.0).clamp(0, 255).byte()
            
            try:
                w = project(G, img_255, num_steps=300, device=device, verbose=False)
                if w is not None:
                    w_mean = w.mean(dim=1, keepdim=True).cpu()  # [1, 1, w_dim]
                    
                    if has_attr and len(w_pos) < num_samples:
                        w_pos.append(w_mean)
                    elif not has_attr and len(w_neg) < num_samples:
                        w_neg.append(w_mean)
                    
                    count += 1
                    if count % 10 == 0:
                        print(f"  Processed {count} images...")
            except Exception as e:
                continue
    
    if not w_pos or not w_neg:
        print(f"[Error] Could not find enough samples (pos: {len(w_pos)}, neg: {len(w_neg)})")
        return None
    
    w_pos_mean = torch.cat(w_pos, dim=0).mean(dim=0).squeeze(0)  # [w_dim]
    w_neg_mean = torch.cat(w_neg, dim=0).mean(dim=0).squeeze(0)
    attr_vector = (w_pos_mean - w_neg_mean).to(device)
    
    print(f"[Attribute] Extracted vector from {len(w_pos)} positive and {len(w_neg)} negative samples")
    return attr_vector

def manipulate_image_w(G, image, attribute_vector, scale=3.0, device='cuda', normalize_attr=True, truncation_psi=None, projection_steps=500):
    """
    StyleGAN2-ADA에서 이미지 편집
    AAE의 manipulate_image와 동일한 인터페이스
    
    Args:
        G: StyleGAN generator
        image: Input image tensor [C, H, W]
        attribute_vector: Attribute direction vector [w_dim]
        scale: Scaling factor for attribute vector
        device: torch device
        normalize_attr: If True, normalize attribute vector to prevent excessive movement
        truncation_psi: Truncation trick parameter (None to disable, typically 0.5-0.7)
        projection_steps: Number of steps for image projection to W space
    """
    # 이미지를 W space로 투영
    img_255 = (image * 255.0).clamp(0, 255).byte()
    w = project(G, img_255, num_steps=projection_steps, device=device, verbose=False)
    
    if w is None:
        return None
    
    # w의 shape: [num_steps, num_ws, w_dim]
    # num_steps 차원에서 평균을 내어 [num_ws, w_dim]으로 만들기
    w_mean = w.mean(dim=0)  # [num_ws, w_dim]
    
    # 속성 벡터 정규화 (품질 저하 방지)
    if normalize_attr:
        # Attribute vector의 크기를 제한하여 과도한 이동 방지
        attr_norm = torch.norm(attribute_vector)
        if attr_norm > 0:
            # 정규화: attribute vector의 크기를 1.0으로 제한
            # 이렇게 하면 scale 값이 실제 이동 거리를 더 잘 제어함
            attribute_vector = attribute_vector / attr_norm
    
    # 속성 벡터 적용
    # w_mean을 [1, num_ws, w_dim] 형태로 만들기
    w_mean = w_mean.unsqueeze(0)  # [1, num_ws, w_dim]
    
    # 속성 벡터를 [1, num_ws, w_dim] 형태로 확장하여 더하기
    # attribute_vector는 [w_dim] 형태
    attr_expanded = attribute_vector.unsqueeze(0).unsqueeze(0)  # [1, 1, w_dim]
    attr_expanded = attr_expanded.repeat(1, G.mapping.num_ws, 1)  # [1, num_ws, w_dim]
    
    w_modified = w_mean + attr_expanded * scale  # [1, num_ws, w_dim]
    
    # Truncation trick 적용 (선택적, 품질 향상)
    if truncation_psi is not None:
        # W space의 평균을 계산 (간단한 근사: 원본 w_mean의 평균 사용)
        w_avg = w_mean.mean(dim=1, keepdim=True)  # [1, 1, w_dim]
        w_avg = w_avg.repeat(1, G.mapping.num_ws, 1)  # [1, num_ws, w_dim]
        # Truncation: w_avg에 가깝게 제한
        w_modified = w_avg + truncation_psi * (w_modified - w_avg)
    
    # 이미지 생성
    with torch.no_grad():
        img_modified = G.synthesis(w_modified, noise_mode='const')
        img_modified = (img_modified + 1.0) / 2.0
        img_modified = torch.clamp(img_modified, 0, 1)
    
    # Shape 확인 및 정규화
    # G.synthesis는 [B, C, H, W] 형태를 반환 (B는 보통 1)
    if len(img_modified.shape) == 4:
        if img_modified.shape[0] == 1:
            img_modified = img_modified.squeeze(0)  # [C, H, W]
        else:
            # 배치가 여러 개인 경우 첫 번째만 사용
            img_modified = img_modified[0]  # [C, H, W]
    elif len(img_modified.shape) != 3:
        print(f"[Warning] Unexpected shape in manipulate_image_w: {img_modified.shape}")
    
    # 최종 shape 확인: [C, H, W] 형태여야 함
    assert len(img_modified.shape) == 3, f"Expected [C, H, W], got {img_modified.shape}"
    
    return img_modified

def main(attr_name="Smiling", num_samples=50, num_test_images=8, projection_steps_attr=300, projection_steps_test=500, model_path=None):
    """
    AAE의 main()과 동일한 기능
    
    Args:
        attr_name: 실험할 속성 이름 (예: "Smiling", "Eyeglasses", "Male", "Young", etc.)
        num_samples: 속성 벡터 추출에 사용할 샘플 수 (양성 + 음성 = num_samples * 2)
        num_test_images: 테스트할 이미지 개수
        projection_steps_attr: 속성 추출 시 projection steps
        projection_steps_test: 테스트 이미지 편집 시 projection steps
        model_path: 사용할 모델 체크포인트 경로 (None일 경우 자동 검색)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Main] Device: {device}")
    print(f"[Main] Target attribute: {attr_name}")
    
    # 모델 경로 찾기
    if model_path is None:
        possible_paths = [
            os.path.join(config.project_dir, 'checkpoints', 'stylegan2_ada', '*.pkl'),
            os.path.join(config.project_dir, 'checkpoints', 'stylegan2_ada', 'network-snapshot-*.pkl'),
            os.path.join(config.output_path, 'stylegan2_ada_training', '**', 'network-snapshot-*.pkl'),
            os.path.join(config.output_path, 'stylegan2_ada_training', 'network-snapshot-*.pkl'),
        ]
        
        for pattern in possible_paths:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                model_path = max(matches, key=os.path.getmtime)
                break
    else:
        if not os.path.exists(model_path):
            print(f"[Error] Specified model path does not exist: {model_path}")
            return
    
    if model_path is None:
        print(f"[Main] StyleGAN2-ADA model not found")
        print(f"[Main] Please train first: python src/train_stylegan.py")
        return
    
    # Load model
    print(f"[Main] Loading StyleGAN2-ADA model from {model_path}")
    G = load_stylegan_model(model_path, device)
    print("[Main] Model loaded successfully")
    
    # Check dataset
    if not os.path.exists(config.celebA_image_path):
        print(f"[Main] Dataset not found at {config.celebA_image_path}")
        return
    
    # Extract attribute vector
    print(f"\n[Main] Extracting '{attr_name}' attribute vector...")
    print("[Info] This will take a while (projecting images)...")
    
    # Load data for attribute extraction
    attr_loader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        filter_attr=attr_name,
        filter_value=1,
        shuffle=False,
        max_samples=num_samples
    )
    
    attr_loader_neg = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=1,
        image_size=G.img_resolution,
        filter_attr=attr_name,
        filter_value=-1,
        shuffle=False,
        max_samples=num_samples
    )
    
    # GPU 개수 확인 및 최적화
    # num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_gpus = 1 # Force single GPU to avoid model reloading overhead in current MP implementation
    print(f"[GPU] Using {num_gpus} GPU(s) for parallel processing (Forced single-GPU for stability)")
    
    # GPU 0과 GPU 1에 모델 로드 (멀티 GPU 활용)
    G_gpus = {}
    if num_gpus >= 2:
        print(f"[GPU] Loading models on GPU 0 and GPU 1...")
        G_gpus[0] = load_stylegan_model(model_path, 'cuda:0')
        G_gpus[1] = load_stylegan_model(model_path, 'cuda:1')
    else:
        G_gpus[0] = G
    
    # 속성 벡터 추출 (멀티프로세싱으로 병렬 처리)
    print(f"[Info] Projecting images with attribute (parallel processing, {num_samples} samples)...")
    
    # 이미지 수집
    pos_images_list = []
    for idx, (images, _) in enumerate(attr_loader):
        if len(pos_images_list) >= num_samples:
            break
        pos_images_list.append(images[0].cpu())
    
    # 멀티프로세싱으로 병렬 projection
    w_pos = []
    total_pos = len(pos_images_list)
    
    # 멀티프로세싱으로 병렬 처리
    if num_gpus >= 2:
        # 작업 준비 (각 이미지에 GPU 할당)
        tasks = []
        for i, img in enumerate(pos_images_list):
            gpu_id = i % num_gpus
            tasks.append((img, gpu_id, model_path, projection_steps_attr))
        
        # 멀티프로세싱 풀 생성
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.map(project_on_gpu, tasks)
        
        # 결과 수집
        for result, gpu_id in results:
            if result is not None:
                w_pos.append(torch.from_numpy(result))
                if len(w_pos) % 5 == 0:
                    print(f"[Progress] Positive: {len(w_pos)}/{total_pos} images projected", flush=True)
    else:
        # 단일 GPU인 경우 순차 처리 (Optimized: Reuse G)
        for i, img in enumerate(pos_images_list):
            # result, _ = project_on_gpu((img, 0, model_path, projection_steps_attr))
            try:
                img = img.to(device)
                img_255 = (img * 255.0).clamp(0, 255).byte()
                w = project(G, img_255, num_steps=projection_steps_attr, device=device, verbose=False)
                if w is not None:
                    result = w.mean(dim=1, keepdim=True).cpu().numpy()
                    w_pos.append(torch.from_numpy(result))
                    if len(w_pos) % 1 == 0:
                         print(f"[Progress] Positive: {len(w_pos)}/{total_pos} images projected", flush=True)
            except Exception as e:
                print(f"[Warning] Projection failed: {e}")
    
    print(f"[Info] Projecting images without attribute (parallel processing)...")
    
    # 음성 샘플 이미지 수집
    neg_images_list = []
    for idx, (images, _) in enumerate(attr_loader_neg):
        if len(neg_images_list) >= num_samples:
            break
        neg_images_list.append(images[0].cpu())
    
    # 멀티프로세싱으로 병렬 projection
    w_neg = []
    total_neg = len(neg_images_list)
    
    # 멀티프로세싱으로 병렬 처리
    if num_gpus >= 2:
        # 작업 준비 (각 이미지에 GPU 할당)
        tasks = []
        for i, img in enumerate(neg_images_list):
            gpu_id = i % num_gpus
            tasks.append((img, gpu_id, model_path, projection_steps_attr))
        
        # 멀티프로세싱 풀 생성
        with mp.Pool(processes=num_gpus) as pool:
            results = pool.map(project_on_gpu, tasks)
        
        # 결과 수집
        for result, gpu_id in results:
            if result is not None:
                w_neg.append(torch.from_numpy(result))
                if len(w_neg) % 5 == 0:
                    print(f"[Progress] Negative: {len(w_neg)}/{total_neg} images projected", flush=True)
    else:
        # 단일 GPU인 경우 순차 처리 (Optimized: Reuse G)
        for i, img in enumerate(neg_images_list):
            # result, _ = project_on_gpu((img, 0, model_path, projection_steps_attr))
            try:
                img = img.to(device)
                img_255 = (img * 255.0).clamp(0, 255).byte()
                w = project(G, img_255, num_steps=projection_steps_attr, device=device, verbose=False)
                if w is not None:
                    result = w.mean(dim=1, keepdim=True).cpu().numpy()
                    w_neg.append(torch.from_numpy(result))
                    if len(w_neg) % 1 == 0:
                        print(f"[Progress] Negative: {len(w_neg)}/{total_neg} images projected", flush=True)
            except Exception as e:
                print(f"[Warning] Projection failed: {e}")
    
    if not w_pos or not w_neg:
        print("[Error] Could not extract attribute vector")
        return
    
    w_pos_mean = torch.cat(w_pos, dim=0).mean(dim=0).squeeze(0).to(device)
    w_neg_mean = torch.cat(w_neg, dim=0).mean(dim=0).squeeze(0).to(device)
    attribute_vector = w_pos_mean - w_neg_mean
    
    # Load test images (속성이 없는 이미지들을 선택)
    print(f"\n[Main] Loading test images ({num_test_images} images)...")
    test_loader = get_celeba_loader(
        celebA_image_path=config.celebA_image_path,
        celebA_attr_path=config.celebA_attr_path,
        batch_size=num_test_images,
        image_size=G.img_resolution,
        filter_attr=attr_name,
        filter_value=-1,
        shuffle=True,
        max_samples=num_test_images
    )
    
    test_images, _ = next(iter(test_loader))
    test_images = test_images.to(device)
    
    # Manipulate images
    print("\n[Main] Manipulating images (projecting and editing)...")
    print("[Info] This will take a while...")
    results = []
    
    # Original images - shape 확인
    print(f"[Debug] test_images shape: {test_images.shape}")
    assert len(test_images.shape) == 4, f"test_images should be [B, C, H, W], got {test_images.shape}"
    results.append(test_images.cpu())
    
    # Edited images with different scales
    # 더 작은 scale 범위 사용 (품질 저하 방지)
    # 원래: [1.0, 2.0, 3.0] -> 복구
    
    # GPU 병렬 처리를 위한 작업 준비
    scales = [1.0, 2.0, 3.0]
    manipulation_tasks = []
    for scale_idx, scale in enumerate(scales):
        for img_idx, img in enumerate(test_images):
            # img는 [C, H, W] 형태
            gpu_id = (scale_idx * len(test_images) + img_idx) % num_gpus if num_gpus >= 2 else 0
            manipulation_tasks.append((img.cpu(), scale, gpu_id, img_idx, scale_idx, model_path, attribute_vector.cpu().numpy(), projection_steps_test))
    
    print(f"[GPU] Processing {len(manipulation_tasks)} image manipulations with {num_gpus} GPU(s)...")
    
    # 병렬 처리 실행
    if num_gpus >= 2:
        mp.set_start_method('spawn', force=True)
        with mp.Pool(processes=num_gpus) as pool:
            manipulation_results = pool.map(manipulate_on_gpu, manipulation_tasks)
    else:
        manipulation_results = [manipulate_on_gpu(task) for task in manipulation_tasks]
    
    # 결과를 scale별로 그룹화
    edited_by_scale = {scale: [] for scale in scales}
    for edited, img_idx, scale_idx in manipulation_results:
        if edited is not None:
            edited_by_scale[scales[scale_idx]].append((edited, img_idx))
    
    # 각 scale별로 배치 생성
    for scale in scales:
        if edited_by_scale[scale]:
            # img_idx 순서대로 정렬
            edited_by_scale[scale].sort(key=lambda x: x[1])
            edited_batch = [edited for edited, _ in edited_by_scale[scale]]
            
            if edited_batch:
                batch_tensor = torch.stack(edited_batch, dim=0)  # [B, C, H, W]
                print(f"[Debug] Scale {scale}: batch_tensor shape = {batch_tensor.shape}")
                assert len(batch_tensor.shape) == 4, f"batch_tensor should be [B, C, H, W], got {batch_tensor.shape}"
                results.append(batch_tensor)
    
    # Save results
    output_dir = config.output_path
    os.makedirs(output_dir, exist_ok=True)
    
    # 개별 이미지 저장 디렉토리 생성 (속성 이름 기반)
    attr_name_safe = attr_name.lower().replace(' ', '_')
    individual_dir = os.path.join(output_dir, f'stylegan_latent_control_{attr_name_safe}_individual')
    os.makedirs(individual_dir, exist_ok=True)
    
    # 각 원본 이미지에 대해 개별 파일로 저장
    print(f"\n[Main] Saving individual images...")
    
    # 원본 이미지들
    original_images = results[0]  # [8, C, H, W]
    
    # 각 원본 이미지에 대해 원본 + 3가지 스케일 버전을 하나의 파일로 저장
    for img_idx in range(len(original_images)):
        # 원본 이미지
        original = original_images[img_idx].cpu()  # [C, H, W]
        
        # 각 스케일별 편집된 이미지
        edited_images = []
        for scale_idx, scale in enumerate([1.0, 2.0, 3.0]):
            if scale_idx + 1 < len(results):
                edited_batch = results[scale_idx + 1]  # [8, C, H, W]
                if img_idx < len(edited_batch):
                    edited_images.append(edited_batch[img_idx])  # [C, H, W]
        
        # 원본 + 편집된 이미지들을 하나의 이미지로 합치기 (가로로)
        images_to_save = [original] + edited_images  # 총 4개 (원본 + 3개 스케일)
        
        # 배치로 변환
        batch_images = torch.stack(images_to_save, dim=0)  # [4, C, H, W]
        
        # 개별 파일로 저장
        output_path_individual = os.path.join(individual_dir, f'image_{img_idx+1:02d}_comparison.png')
        save_image(
            batch_images,
            output_path_individual,
            nrow=4,  # 가로로 4개 (원본 + 3개 스케일)
            normalize=False,
            padding=2
        )
        
        print(f"  Saved: {os.path.basename(output_path_individual)}")
    
    # 전체 결과도 저장 (기존 방식)
    # results의 각 요소가 [B, C, H, W] 형태인지 확인
    print(f"\n[Debug] Checking results shapes:")
    for i, r in enumerate(results):
        print(f"  results[{i}] shape: {r.shape}")
    
    # 모든 결과를 배치 차원으로 연결
    all_results = torch.cat(results, dim=0)  # [32, C, H, W] 형태여야 함
    
    # Shape 확인 및 디버깅
    print(f"\n[Debug] all_results shape: {all_results.shape}")
    print(f"[Debug] Expected: [32, 3, 128, 128] for 32 images")
    
    if len(all_results.shape) != 4:
        print(f"[Error] Unexpected tensor shape: {all_results.shape}")
        print("[Error] Expected [B, C, H, W] format")
        return
    
    # 실제 이미지 개수 확인
    actual_num_images = all_results.shape[0]
    if actual_num_images != 32:
        print(f"[Warning] Expected 32 images, but got {actual_num_images} images")
        print(f"[Warning] This might cause issues with nrow=8")
    
    attr_name_safe = attr_name.lower().replace(' ', '_')
    output_path_file = os.path.join(output_dir, f'stylegan_latent_control_{attr_name_safe}.png')
    
    save_image(
        all_results,
        output_path_file,
        nrow=8,  # 가로로 8개씩 배치
        normalize=False,
        padding=2
    )
    
    print(f"\n[Main] Full results saved to {output_path_file}")
    print(f"[Main] Total images: {actual_num_images} (Expected: 32)")
    
    print(f"\n[Main] Individual images saved to {individual_dir}")
    print(f"[Main] Each file contains (attribute: {attr_name}):")
    print(f"  - Left: Original image (no {attr_name.lower()})")
    print(f"  - 2nd: Scale 1.0 (slight {attr_name.lower()})")
    print(f"  - 3rd: Scale 2.0 (moderate {attr_name.lower()})")
    print(f"  - Right: Scale 3.0 (strong {attr_name.lower()})")
    print("\n[Info] Quality improvements applied:")
    print("  - Attribute vector normalization: Enabled")
    print("  - Truncation trick (psi=0.7): Enabled")
    print("  - Scale range: [1.0, 2.0, 3.0]")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='StyleGAN2-ADA Attribute Control Experiment')
    parser.add_argument(
        '--attr', 
        type=str, 
        default='Smiling',
        help='Attribute name to experiment with (e.g., Smiling, Eyeglasses, Male, Young, Blond_Hair, etc.)'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of samples for attribute vector extraction (default: 50, increase for better quality)'
    )
    parser.add_argument(
        '--num-test-images',
        type=int,
        default=8,
        help='Number of test images to edit (default: 8)'
    )
    parser.add_argument(
        '--projection-steps-attr',
        type=int,
        default=300,
        help='Number of projection steps for attribute extraction (default: 300)'
    )
    parser.add_argument(
        '--projection-steps-test',
        type=int,
        default=500,
        help='Number of projection steps for test image editing (default: 500)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to the StyleGAN2-ADA model checkpoint (optional)'
    )
    args = parser.parse_args()
    main(
        attr_name=args.attr,
        num_samples=args.num_samples,
        num_test_images=args.num_test_images,
        projection_steps_attr=args.projection_steps_attr,
        projection_steps_test=args.projection_steps_test,
        model_path=args.model
    )

