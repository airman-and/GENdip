"""
StyleGAN2-ADA 학습 스크립트
AAE의 train_aae.py와 동일한 인터페이스로 작성
GPU 활용 최대화 버전
백그라운드 실행 및 상세 로그 지원
"""
import os
import sys
import subprocess
import torch
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_optimal_batch_size(image_size, num_gpus, gpu_memory_gb):
    """
    GPU 메모리에 따라 최적 배치 크기 계산
    
    Args:
        image_size: 이미지 크기
        num_gpus: GPU 개수
        gpu_memory_gb: GPU 메모리 (GB)
    
    Returns:
        optimal_batch_size: 최적 배치 크기
    """
    # 이미지 크기별 기본 배치 크기 (단일 GPU 기준)
    base_batch_sizes = {
        64: 64,
        128: 32,
        256: 16,
        512: 8,
        1024: 4
    }
    
    # 기본값 (가장 가까운 크기)
    base_batch = base_batch_sizes.get(image_size, 32)
    
    # GPU 메모리에 따라 조정 (더 공격적인 설정)
    if gpu_memory_gb >= 48:  # 초고사양 GPU (RTX 8000 50GB, A100 80GB 등)
        multiplier = 4.0  # 50GB 메모리면 배치 크기를 크게 늘릴 수 있음
    elif gpu_memory_gb >= 40:  # 고사양 GPU (RTX 8000 48GB, A100 40GB 등)
        multiplier = 3.0
    elif gpu_memory_gb >= 20:  # 중고사양 GPU (RTX 3090, V100 등)
        multiplier = 2.0
    else:  # 일반 GPU
        multiplier = 1.0
    
    # GPU 개수에 따라 조정
    optimal_batch = int(base_batch * multiplier * num_gpus)
    
    # 최소값 보장
    optimal_batch = max(optimal_batch, num_gpus)
    
    # 최대값 제한 (메모리 오버플로우 방지)
    # 128x128 이미지 기준으로 대략 50GB 메모리면 배치 128 정도까지 가능
    max_batch_by_resolution = {
        64: 256,
        128: 128,
        256: 64,
        512: 32,
        1024: 16
    }
    max_batch = max_batch_by_resolution.get(image_size, 128) * num_gpus
    optimal_batch = min(optimal_batch, max_batch)
    
    return optimal_batch

def train_stylegan(
    num_epochs=30,
    batch_size=None,  # None이면 자동 계산
    image_size=128,
    learning_rate=0.002,
    device='cuda',
    resume=None,
    auto_optimize=True,  # GPU 최적화 자동 수행
    background=False,  # 백그라운드 실행
    log_file=None,  # 로그 파일 경로 (None이면 자동 생성)
    kimg=None  # 직접 kimg 지정 (None이면 num_epochs * 10으로 계산)
):
    """
    StyleGAN2-ADA 학습 함수
    AAE의 train_aae와 유사한 인터페이스
    GPU 활용 최대화 버전
    
    Args:
        num_epochs: 학습 에포크 수 (StyleGAN은 kimg 사용)
        batch_size: 배치 크기 (None이면 자동 계산)
        image_size: 이미지 크기
        learning_rate: 학습률
        device: 디바이스
        resume: 재개할 체크포인트 경로
        auto_optimize: GPU 최적화 자동 수행 여부
    """
    # GPU 정보 확인
    if device == 'cuda' and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"[GPU Info] Found {num_gpus} GPU(s)")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.2f} GB")
        
        # GPU 개수는 2의 거듭제곱만 가능 (StyleGAN 제약)
        if num_gpus > 1:
            # 가장 가까운 2의 거듭제곱으로 조정
            import math
            optimal_gpus = 2 ** int(math.log2(num_gpus))
            if optimal_gpus != num_gpus:
                print(f"[Warning] Adjusting GPU count from {num_gpus} to {optimal_gpus} (must be power of 2)")
                num_gpus = optimal_gpus
    else:
        num_gpus = 0
        gpu_memory_gb = 0
        print("[Warning] CUDA not available, using CPU")
    
    # StyleGAN2-ADA-PyTorch 경로
    stylegan_path = os.path.join(config.project_dir, 'stylegan2_ada_pytorch')
    train_script = os.path.join(stylegan_path, 'train.py')
    
    if not os.path.exists(train_script):
        print(f"[Error] StyleGAN2-ADA not found at {stylegan_path}")
        print(f"[Info] Please clone: git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git")
        return
    
    # 데이터셋 경로 확인 (ZIP 형식 필요)
    dataset_zip = os.path.join(config.project_dir, 'dataset', 'celebA.zip')
    
    if not os.path.exists(dataset_zip):
        print(f"[Warning] Dataset ZIP not found at {dataset_zip}")
        print(f"[Info] Converting CelebA to ZIP format...")
        print(f"[Info] Run: python {os.path.join(stylegan_path, 'dataset_tool.py')} \\")
        print(f"    --source={config.celebA_image_path} \\")
        print(f"    --dest={dataset_zip} \\")
        print(f"    --resolution={image_size}x{image_size}")
        return
    
    # 출력 디렉토리
    outdir = os.path.join(config.output_path, 'stylegan2_ada_training')
    os.makedirs(outdir, exist_ok=True)
    
    # 로그 파일 설정
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(outdir, f'training_{timestamp}.log')
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 배치 크기 자동 최적화
    if batch_size is None and auto_optimize:
        batch_size = get_optimal_batch_size(image_size, num_gpus, gpu_memory_gb)
        print(f"[Optimization] Auto-calculated batch size: {batch_size}")
    
    if batch_size is None:
        batch_size = 32  # 기본값
    
    # kimg 계산 (StyleGAN은 kimg 단위 사용)
    # 직접 kimg가 지정되지 않으면 num_epochs * 10으로 계산
    if kimg is None:
        kimg = num_epochs * 10
    
    # Workers 수 최적화 (CPU 코어 수에 따라)
    import multiprocessing
    num_workers = min(multiprocessing.cpu_count(), 8)  # 최대 8개
    
    # 학습 명령 구성
    cmd = [
        'python', train_script,
        '--outdir', outdir,
        '--cfg', 'auto',  # auto가 GPU 개수에 따라 자동 최적화
        '--data', dataset_zip,
        '--gpus', str(num_gpus) if num_gpus > 0 else '0',
        '--batch', str(batch_size),
        '--gamma', '10.0',
        '--kimg', str(kimg),
        '--snap', '50',
        '--workers', str(num_workers),  # DataLoader workers 최적화
        '--aug', 'noaug',  # Augmentation 비활성화 (PyTorch 2.8 호환성 문제 해결)
    ]
    # TF32는 환경 변수로 설정 (PyTorch 기본값)
    # CUDA에서 TF32는 기본적으로 활성화되어 있음
    
    # Mixed precision은 기본적으로 활성화되어 있음 (fp32=False가 기본)
    # 명시적으로 비활성화하려면 --fp32 추가
    
    if resume:
        cmd.extend(['--resume', resume])
    
    # 가상환경 확인
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
    python_path = sys.executable
    
    print(f"\n[Training] Starting StyleGAN2-ADA training")
    print(f"[Training] Conda Environment: {conda_env}")
    print(f"[Training] Python: {python_path}")
    print(f"[Training] Device: {device}")
    print(f"[Training] GPUs: {num_gpus}")
    print(f"[Training] Batch size: {batch_size}")
    print(f"[Training] Image size: {image_size}")
    print(f"[Training] Workers: {num_workers}")
    print(f"[Training] kimg: {kimg} (approximately {num_epochs} epochs)")
    print(f"[Training] TF32: Enabled")
    print(f"[Training] Log file: {log_file}")
    print(f"[Training] Background: {background}")
    print(f"[Training] Command: {' '.join(cmd)}\n")
    
    # 로그 파일에 시작 정보 기록
    with open(log_file, 'w') as f:
        f.write(f"StyleGAN2-ADA Training Log\n")
        f.write(f"{'='*60}\n")
        f.write(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Conda Environment: {conda_env}\n")
        f.write(f"Python: {python_path}\n")
        f.write(f"GPUs: {num_gpus}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Image Size: {image_size}\n")
        f.write(f"Workers: {num_workers}\n")
        f.write(f"kimg: {kimg}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'='*60}\n\n")
    
    # 학습 실행
    if background:
        # 백그라운드 실행 (nohup 사용)
        import signal
        log_handle = open(log_file, 'a')
        # nohup을 명시적으로 사용하여 더 안전하게 실행
        nohup_cmd = ['nohup'] + cmd
        process = subprocess.Popen(
            nohup_cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid  # 새로운 프로세스 그룹 생성
        )
        
        # PID 저장
        pid_file = log_file.replace('.log', '.pid')
        with open(pid_file, 'w') as f:
            f.write(str(process.pid))
        
        print(f"[Training] Process started in background")
        print(f"[Training] PID: {process.pid}")
        print(f"[Training] PID file: {pid_file}")
        print(f"[Training] Log file: {log_file}")
        print(f"[Training] Monitor with: tail -f {log_file}")
        print(f"[Training] Check process: ps -p {process.pid}")
        print(f"[Training] Stop process: kill {process.pid}")
        
        return process.pid
    else:
        # 포그라운드 실행 (로그도 파일에 기록)
        log_handle = open(log_file, 'a')
        try:
            process = subprocess.run(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True
            )
            log_handle.write(f"\n{'='*60}\n")
            log_handle.write(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_handle.write(f"Exit Code: {process.returncode}\n")
            log_handle.write(f"{'='*60}\n")
        finally:
            log_handle.close()
        
        return process.returncode


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='StyleGAN2-ADA Training')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size (None for auto)')
    parser.add_argument('--image-size', type=int, default=128, help='Image size')
    parser.add_argument('--lr', type=float, default=0.002, help='Learning rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--background', action='store_true', help='Run in background')
    parser.add_argument('--log', type=str, default=None, help='Log file path')
    parser.add_argument('--no-auto-optimize', action='store_true', help='Disable auto optimization')
    parser.add_argument('--kimg', type=int, default=None, help='Target kimg (overrides epochs calculation)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_stylegan(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        learning_rate=args.lr,
        device=device,
        resume=args.resume,
        auto_optimize=not args.no_auto_optimize,
        background=args.background,
        log_file=args.log,
        kimg=args.kimg
    )
