# AAE vs StyleGAN2-ADA 실험 방법 비교

## 1. 모델 구조 차이

### AAE (Adversarial Autoencoder)
- **Encoder-Decoder 구조**: 이미지를 latent space로 인코딩하고 다시 디코딩
- **Latent Space**: Z space (128차원)
- **역방향 가능**: 실제 이미지 → Latent code → 재구성 이미지
- **속성 제어**: Latent code에 속성 벡터를 더하는 방식

### StyleGAN2-ADA
- **Generator-only 구조**: Latent code에서 직접 이미지 생성
- **Latent Space**: 
  - Z space (512차원) → Mapping Network → W space (512차원, 18개 레이어)
- **역방향 불가능**: 실제 이미지를 latent로 변환하려면 별도의 projection 필요
- **속성 제어**: W space에서 스타일 조작

## 2. 실험 방법 비교

### AAE 실험 (`experiment_aae.py`)

#### ✅ 가능한 실험

1. **재구성 (Reconstruction)**
   ```python
   z = encoder(real_images)
   recon_images = decoder(z)
   ```
   - 실제 이미지를 인코딩 후 디코딩하여 재구성 품질 확인
   - **장점**: 실제 이미지와 비교 가능

2. **Latent Space 보간 (Interpolation)**
   ```python
   z1 = encoder(img1)
   z2 = encoder(img2)
   z_interp = interpolate(z1, z2)
   imgs_interp = decoder(z_interp)
   ```
   - 두 실제 이미지 간 보간
   - **장점**: 실제 이미지 기반 보간 가능

3. **랜덤 샘플링**
   ```python
   z_random = torch.randn(64, 128)
   imgs_random = decoder(z_random)
   ```
   - Gaussian prior에서 샘플링하여 생성

4. **속성 제어 (Attribute Control)** ⭐ **핵심 차이점**
   ```python
   # 속성 벡터 계산
   z_pos = [encoder(img) for img in images_with_attribute]
   z_neg = [encoder(img) for img in images_without_attribute]
   attr_vector = mean(z_pos) - mean(z_neg)
   
   # 속성 적용
   z_target = encoder(target_image)
   z_modified = z_target + attr_vector * scale
   img_modified = decoder(z_modified)
   ```
   - **장점**: 실제 이미지에 속성을 직접 적용 가능
   - **장점**: 속성 벡터를 데이터에서 자동 계산
   - **장점**: 실제 이미지 기반 속성 편집

5. **배치 속성 적용**
   - 여러 이미지에 동일한 속성 벡터 일괄 적용

#### ❌ 불가능한 실험

- Style Mixing (W space가 없음)
- Truncation Trick (Z space에서 직접 샘플링)

---

### StyleGAN2-ADA 실험 (`experiment_stylegan_official.py`)

#### ✅ 가능한 실험

1. **랜덤 샘플링**
   ```python
   z = torch.randn([16, 512])
   img = G(z, None, truncation_psi=1.0)
   ```
   - Z space에서 샘플링하여 생성

2. **Truncation Trick 비교** ⭐ **StyleGAN 특화**
   ```python
   for trunc in [1.0, 0.7, 0.5, 0.3]:
       img = G(z, None, truncation_psi=trunc)
   ```
   - **장점**: 이미지 품질과 다양성 트레이드오프 제어
   - **장점**: 더 현실적인 이미지 생성 가능

3. **Style Mixing** ⭐ **StyleGAN 특화**
   ```python
   w1 = G.mapping(z1)
   w2 = G.mapping(z2)
   w_mixed = w1.clone()
   w_mixed[8:] = w2[8:]  # 레이어별 스타일 혼합
   img = G.synthesis(w_mixed)
   ```
   - **장점**: 레이어별로 다른 스타일 혼합 가능
   - **장점**: 고해상도/저해상도 특징 분리 제어

4. **Latent Space 보간**
   ```python
   w1 = G.mapping(z1)
   w2 = G.mapping(z2)
   w_interp = (1-alpha) * w1 + alpha * w2
   img = G.synthesis(w_interp)
   ```
   - W space에서 보간 (더 부드러운 결과)

#### ❌ 어려운 실험

1. **재구성 (Reconstruction)**
   - Encoder가 없어서 실제 이미지를 재구성하기 어려움
   - **해결책**: `projector.py`를 사용하여 이미지를 W space로 투영 (시간 소요)

2. **속성 제어 (Attribute Control)**
   - 실제 이미지에 직접 속성을 적용하기 어려움
   - **해결책**: 
     - 먼저 이미지를 W space로 투영
     - 속성 벡터 계산 (별도 학습 필요)
     - W space에서 조작
   - **단점**: AAE보다 복잡하고 시간 소요

3. **실제 이미지 기반 실험**
   - 모든 실험이 생성된 이미지 기반
   - 실제 이미지를 사용하려면 projection 필요

---

## 3. 주요 차이점 요약

| 특징 | AAE | StyleGAN2-ADA |
|------|-----|---------------|
| **Encoder** | ✅ 있음 | ❌ 없음 (별도 projection 필요) |
| **재구성** | ✅ 쉬움 | ❌ 어려움 (projection 필요) |
| **실제 이미지 편집** | ✅ 직접 가능 | ❌ projection 후 가능 |
| **속성 제어** | ✅ 쉬움 (벡터 계산) | ❌ 복잡 (별도 학습) |
| **이미지 품질** | 보통 | ⭐ 매우 높음 |
| **Style Mixing** | ❌ 불가능 | ✅ 가능 |
| **Truncation Trick** | ❌ 없음 | ✅ 있음 |
| **W Space** | ❌ 없음 | ✅ 있음 (18 레이어) |
| **Latent 차원** | 128 | 512 (Z), 512×18 (W) |

## 4. 각 방법의 장단점

### AAE 장점
1. ✅ **실제 이미지 편집 용이**: Encoder가 있어서 실제 이미지를 바로 편집 가능
2. ✅ **속성 제어 간단**: 데이터에서 속성 벡터를 자동 계산
3. ✅ **재구성 가능**: 실제 이미지 재구성 품질 확인 가능
4. ✅ **해석 가능성**: Encoder-Decoder 구조로 이해하기 쉬움

### AAE 단점
1. ❌ **이미지 품질**: StyleGAN보다 낮을 수 있음
2. ❌ **Style Mixing 불가**: 레이어별 스타일 제어 불가
3. ❌ **제한된 Latent Space**: Z space만 사용

### StyleGAN2-ADA 장점
1. ✅ **높은 이미지 품질**: 최고 수준의 생성 품질
2. ✅ **Style Mixing**: 레이어별 세밀한 제어
3. ✅ **Truncation Trick**: 품질-다양성 트레이드오프
4. ✅ **W Space**: 더 풍부한 latent representation

### StyleGAN2-ADA 단점
1. ❌ **실제 이미지 편집 복잡**: Projection 필요
2. ❌ **속성 제어 어려움**: 별도 학습 또는 수동 조작 필요
3. ❌ **재구성 어려움**: Encoder 없음
4. ❌ **학습 시간**: 더 오래 걸림

## 5. 사용 시나리오

### AAE를 사용하는 경우
- ✅ 실제 이미지 속성 편집 (미소 추가, 수염 추가 등)
- ✅ 이미지 재구성 품질 확인
- ✅ 빠른 프로토타이핑
- ✅ 해석 가능한 모델 필요

### StyleGAN2-ADA를 사용하는 경우
- ✅ 고품질 이미지 생성
- ✅ Style Mixing 실험
- ✅ 최신 GAN 기술 연구
- ✅ 사전 학습된 모델 활용

## 6. 결론

**AAE**는 **실제 이미지 편집과 속성 제어**에 특화되어 있고, **StyleGAN2-ADA**는 **고품질 생성과 스타일 조작**에 특화되어 있습니다. 

프로젝트 목표에 따라 선택:
- **속성 제어 실험** → AAE 권장
- **고품질 생성 및 Style Mixing** → StyleGAN2-ADA 권장

