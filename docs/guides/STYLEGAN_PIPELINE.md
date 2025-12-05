# StyleGAN2-ADA 파이프라인 가이드

AAE와 동일한 파이프라인으로 StyleGAN2-ADA를 사용할 수 있습니다.

## 파일 구조

### AAE 파이프라인
- `src/train_aae.py` - 학습
- `src/experiment_aae.py` - 실험
- `src/main.py` - 속성 제어

### StyleGAN2-ADA 파이프라인 (동일한 구조)
- `src/train_stylegan.py` - 학습
- `src/experiment_stylegan.py` - 실험
- `src/main_stylegan.py` - 속성 제어

## 1. 데이터셋 준비

StyleGAN2-ADA는 ZIP 형식의 데이터셋이 필요합니다:

```bash
cd /root/workspace/andycho/GenDL-LatentControl
conda activate latent-control

# CelebA를 ZIP 형식으로 변환
python stylegan2_ada_pytorch/dataset_tool.py \
    --source=dataset/celebA/img_align_celeba/img_align_celeba \
    --dest=dataset/celebA.zip \
    --resolution=128x128
```

## 2. 학습

### AAE 학습
```bash
python src/train_aae.py
```

### StyleGAN2-ADA 학습 (동일한 방식)
```bash
python src/train_stylegan.py
```

또는 파라미터 조정:
```python
# train_stylegan.py에서 수정
train_stylegan(
    num_epochs=30,
    batch_size=32,
    image_size=128,
    learning_rate=0.002,
    device='cuda'
)
```

## 3. 실험 실행

### AAE 실험
```bash
python src/experiment_aae.py
```

실험 내용:
1. Reconstruction (재구성)
2. Interpolation (보간)
3. Random Sampling (랜덤 샘플링)
4. Attribute Control (속성 제어)
5. Batch Attribute Application (배치 속성 적용)

### StyleGAN2-ADA 실험 (동일한 실험들)
```bash
python src/experiment_stylegan.py
```

**주의:** StyleGAN2-ADA는 projection이 필요해서 시간이 오래 걸립니다:
- 재구성: 이미지당 30-60초
- 보간: 2개 이미지 projection 필요
- 속성 제어: 많은 이미지 projection 필요 (수십 분 소요)

## 4. 속성 제어 (메인 스크립트)

### AAE 속성 제어
```bash
python src/main.py
```

### StyleGAN2-ADA 속성 제어 (동일한 기능)
```bash
python src/main_stylegan.py
```

**기능:**
- 'Smiling' 속성 벡터 추출
- 테스트 이미지에 속성 적용
- 다양한 스케일로 속성 강도 조절

## 5. 비교표

| 작업 | AAE | StyleGAN2-ADA |
|------|-----|---------------|
| **학습** | `train_aae.py` | `train_stylegan.py` |
| **실험** | `experiment_aae.py` | `experiment_stylegan.py` |
| **속성 제어** | `main.py` | `main_stylegan.py` |
| **실행 시간** | 빠름 (< 1분) | 느림 (수십 분) |
| **이미지 품질** | 보통 | ⭐ 매우 높음 |

## 6. 출력 파일

### AAE 출력
- `output/aae_experiments/`
  - `reconstruction.png`
  - `interpolation.png`
  - `random_samples.png`
  - `attribute_control_beard_direct.png`
  - `batch_beard_direct_control.png`
- `output/latent_control_smiling.png`

### StyleGAN2-ADA 출력 (동일한 구조)
- `output/stylegan2_ada_experiments/`
  - `reconstruction.png`
  - `interpolation.png`
  - `random_samples.png`
  - `attribute_control_beard_direct.png`
  - `batch_beard_direct_control.png`
- `output/stylegan_latent_control_smiling.png`

## 7. 주의사항

### StyleGAN2-ADA의 특별한 점

1. **Projection 시간**
   - 실제 이미지를 사용하는 실험은 projection이 필요
   - 이미지당 30-60초 소요
   - 속성 벡터 계산 시 많은 이미지 projection 필요

2. **랜덤 샘플링은 빠름**
   - 랜덤 벡터 기반 실험은 즉시 가능
   - AAE와 동일한 속도

3. **이미지 품질**
   - StyleGAN2-ADA가 훨씬 높은 품질
   - 더 현실적인 이미지 생성

## 8. 빠른 시작

```bash
# 1. 데이터셋 변환 (한 번만)
python stylegan2_ada_pytorch/dataset_tool.py \
    --source=dataset/celebA/img_align_celeba/img_align_celeba \
    --dest=dataset/celebA.zip \
    --resolution=128x128

# 2. 학습
python src/train_stylegan.py

# 3. 실험 (랜덤 샘플링은 빠름)
python src/experiment_stylegan.py

# 4. 속성 제어
python src/main_stylegan.py
```

## 9. 코드 구조 비교

### AAE
```python
# 학습
encoder = Encoder(...)
decoder = Decoder(...)
z = encoder(image)  # 즉시
recon = decoder(z)

# 속성 제어
z = encoder(image)
z_modified = z + attr_vector
img_edited = decoder(z_modified)
```

### StyleGAN2-ADA
```python
# 학습
G = StyleGANGenerator(...)
img = G(z)

# 속성 제어
w = project(G, image)  # ⚠️ 시간 소요
w_modified = w + attr_vector
img_edited = G.synthesis(w_modified)
```

## 10. 결론

AAE와 **완전히 동일한 파이프라인**으로 StyleGAN2-ADA를 사용할 수 있습니다:
- ✅ 동일한 파일 구조
- ✅ 동일한 실험 종류
- ✅ 동일한 인터페이스
- ⚠️ 다만 실행 시간이 더 오래 걸림 (projection 때문)

