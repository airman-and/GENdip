# StyleGAN2-ADA 사용 가이드

공식 StyleGAN2-ADA-PyTorch 레포지토리를 사용하여 학습 및 실험을 진행할 수 있습니다.

## 레포지토리 위치

- **StyleGAN (TensorFlow)**: `stylegan_official/`
- **StyleGAN2-ADA (PyTorch)**: `stylegan2_ada_pytorch/` ⭐ **권장**

## 1. 데이터셋 준비

StyleGAN2-ADA는 ZIP 형식의 데이터셋을 사용합니다. CelebA를 변환하려면:

```bash
cd /root/workspace/andycho/GenDL-LatentControl
conda activate latent-control

# CelebA 이미지를 ZIP 형식으로 변환
python stylegan2_ada_pytorch/dataset_tool.py \
    --source=dataset/celebA/img_align_celeba/img_align_celeba \
    --dest=dataset/celebA.zip \
    --resolution=128x128
```

## 2. 학습 실행

### 방법 1: 래퍼 스크립트 사용 (권장)

```bash
python src/train_stylegan_official.py \
    --dataset=dataset/celebA.zip \
    --outdir=output/stylegan2_ada_training \
    --cfg=auto \
    --gpus=1 \
    --batch=32 \
    --kimg=25000
```

### 방법 2: 공식 스크립트 직접 사용

```bash
python stylegan2_ada_pytorch/train.py \
    --outdir=output/stylegan2_ada_training \
    --cfg=auto \
    --data=dataset/celebA.zip \
    --gpus=1 \
    --batch=32 \
    --gamma=10.0 \
    --kimg=25000 \
    --snap=50
```

### 주요 파라미터

- `--cfg`: 설정 (`auto`, `stylegan2`, `paper256`, `paper512`, `paper1024`, `cifar`)
- `--gpus`: 사용할 GPU 개수
- `--batch`: 배치 크기
- `--gamma`: R1 regularization weight (기본값: 10.0)
- `--kimg`: 학습할 이미지 수 (1000 단위, 예: 25000 = 25M 이미지)
- `--snap`: 체크포인트 저장 주기 (tick 단위)

## 3. 사전 학습된 모델 사용

공식 사전 학습된 모델을 다운로드하여 사용할 수 있습니다:

```python
# Python에서 직접 사용
import pickle
import torch

# FFHQ 모델 다운로드 및 로드
with open('ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

# 이미지 생성
z = torch.randn([1, G.z_dim]).cuda()
img = G(z, None)  # NCHW, float32, dynamic range [-1, +1]
```

또는 URL로 직접 사용:

```bash
python stylegan2_ada_pytorch/generate.py \
    --outdir=out \
    --seeds=0-35 \
    --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
```

## 4. 실험 실행

학습된 모델로 다양한 실험을 수행할 수 있습니다:

```bash
python src/experiment_stylegan_official.py
```

실험 내용:
- 랜덤 샘플링
- Truncation trick 비교
- Style mixing
- Latent space 보간

## 5. 이미지 투영 (Projection)

실제 이미지를 latent space로 투영:

```bash
python stylegan2_ada_pytorch/projector.py \
    --outdir=out \
    --target=target_image.png \
    --network=checkpoints/network-snapshot-*.pkl
```

## 6. Style Mixing

두 이미지의 스타일을 혼합:

```bash
python stylegan2_ada_pytorch/style_mixing.py \
    --outdir=out \
    --rows=85,100,75,458 \
    --cols=55,821,1789,293 \
    --network=checkpoints/network-snapshot-*.pkl
```

## 참고 자료

- 공식 레포지토리: https://github.com/NVlabs/stylegan2-ada-pytorch
- 논문: https://arxiv.org/abs/2006.06676
- 사전 학습된 모델: https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/

## 주의사항

1. **데이터셋 형식**: StyleGAN2-ADA는 ZIP 형식의 데이터셋을 요구합니다. `dataset_tool.py`로 변환하세요.
2. **GPU 메모리**: 최소 12GB GPU 메모리가 필요합니다.
3. **학습 시간**: CelebA 128x128 기준으로 약 수일 소요될 수 있습니다.
4. **체크포인트**: `network-snapshot-*.pkl` 형식으로 저장됩니다.

