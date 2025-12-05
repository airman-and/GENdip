"""
StyleGAN2-ADA 공식 구현을 사용한 학습 스크립트
공식 레포지토리: https://github.com/NVlabs/stylegan2-ada-pytorch
"""
import os
import sys

# StyleGAN2-ADA-PyTorch 경로 추가
stylegan_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stylegan2_ada_pytorch')
sys.path.insert(0, stylegan_path)

import subprocess
import config

def train_stylegan_official(
    dataset_path=None,
    outdir=None,
    cfg='auto',
    gpus=1,
    batch=32,
    gamma=10.0,
    kimg=25000,
    snap=50,
    resume=None
):
    """
    StyleGAN2-ADA 공식 구현을 사용한 학습
    
    Args:
        dataset_path: 데이터셋 경로 (ZIP 파일 또는 폴더)
        outdir: 출력 디렉토리
        cfg: 설정 ('auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar')
        gpus: 사용할 GPU 개수
        batch: 배치 크기
        gamma: R1 regularization weight
        kimg: 학습할 이미지 수 (1000 단위)
        snap: 체크포인트 저장 주기 (tick 단위)
        resume: 재개할 체크포인트 경로
    """
    if dataset_path is None:
        # CelebA 데이터셋을 ZIP 형식으로 변환 필요
        dataset_path = config.celebA_image_path
        print(f"[Warning] CelebA를 ZIP 형식으로 변환해야 합니다.")
        print(f"[Info] dataset_tool.py를 사용하여 변환하세요:")
        print(f"  python {os.path.join(stylegan_path, 'dataset_tool.py')} \\")
        print(f"    --source={config.celebA_image_path} \\")
        print(f"    --dest={os.path.join(config.project_dir, 'dataset', 'celebA.zip')}")
        return
    
    if outdir is None:
        outdir = os.path.join(config.output_path, 'stylegan2_ada_training')
    
    # train.py 실행
    cmd = [
        'python', os.path.join(stylegan_path, 'train.py'),
        '--outdir', outdir,
        '--cfg', cfg,
        '--data', dataset_path,
        '--gpus', str(gpus),
        '--batch', str(batch),
        '--gamma', str(gamma),
        '--kimg', str(kimg),
        '--snap', str(snap),
    ]
    
    if resume:
        cmd.extend(['--resume', resume])
    
    print(f"[Training] 실행 명령: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='StyleGAN2-ADA 학습')
    parser.add_argument('--dataset', type=str, default=None, help='데이터셋 경로 (ZIP 파일)')
    parser.add_argument('--outdir', type=str, default=None, help='출력 디렉토리')
    parser.add_argument('--cfg', type=str, default='auto', help='설정')
    parser.add_argument('--gpus', type=int, default=1, help='GPU 개수')
    parser.add_argument('--batch', type=int, default=32, help='배치 크기')
    parser.add_argument('--gamma', type=float, default=10.0, help='R1 regularization weight')
    parser.add_argument('--kimg', type=int, default=25000, help='학습할 이미지 수 (1000 단위)')
    parser.add_argument('--snap', type=int, default=50, help='체크포인트 저장 주기')
    parser.add_argument('--resume', type=str, default=None, help='재개할 체크포인트')
    
    args = parser.parse_args()
    
    train_stylegan_official(
        dataset_path=args.dataset,
        outdir=args.outdir,
        cfg=args.cfg,
        gpus=args.gpus,
        batch=args.batch,
        gamma=args.gamma,
        kimg=args.kimg,
        snap=args.snap,
        resume=args.resume
    )

