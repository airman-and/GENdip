# StyleGAN2-ADA vs AAE: 벡터 수정 및 속성 제어 비교

## 핵심 답변: **둘 다 벡터 수정 가능합니다!**

StyleGAN2-ADA도 AAE/VAE처럼 벡터를 수정해서 이미지를 만들 수 있습니다. 다만 **방식과 편의성**에 차이가 있습니다.

## 1. 벡터 수정 방식 비교

### AAE/VAE 방식
```python
# 1. 실제 이미지 → Encoder → Z space (즉시)
z = encoder(real_image)  # 매우 빠름

# 2. Z space에서 벡터 수정
z_modified = z + attr_vector * scale

# 3. Decoder → 편집된 이미지
edited_image = decoder(z_modified)
```

**특징:**
- ✅ **즉시 가능**: Encoder가 있어서 실제 이미지를 바로 Z space로 변환
- ✅ **빠름**: Forward pass 한 번으로 끝
- ✅ **간단**: 코드가 직관적

### StyleGAN2-ADA 방식
```python
# 방법 1: 랜덤 샘플링한 벡터 수정 (즉시 가능)
z = torch.randn([1, 512])
w = G.mapping(z)  # Z → W space
w_modified = w + attr_vector * scale
edited_image = G.synthesis(w_modified)

# 방법 2: 실제 이미지 사용 (projection 필요)
w = project(G, real_image)  # ⚠️ 시간 소요 (수백~수천 step)
w_modified = w + attr_vector * scale
edited_image = G.synthesis(w_modified)
```

**특징:**
- ✅ **랜덤 벡터는 즉시 가능**: Z space에서 샘플링 후 바로 수정
- ⚠️ **실제 이미지는 projection 필요**: 최적화 과정으로 시간 소요
- ✅ **W space 사용**: 더 풍부한 표현력

## 2. 속성 벡터 계산 비교

### AAE 방식
```python
# 데이터에서 속성 벡터 자동 계산
z_pos = [encoder(img) for img in images_with_attr]
z_neg = [encoder(img) for img in images_without_attr]
attr_vector = mean(z_pos) - mean(z_neg)  # 매우 빠름
```

### StyleGAN2-ADA 방식
```python
# 데이터에서 속성 벡터 계산 (projection 필요)
w_pos = [project(G, img) for img in images_with_attr]  # ⚠️ 각 이미지마다 projection
w_neg = [project(G, img) for img in images_without_attr]
attr_vector = mean(w_pos) - mean(w_neg)  # 시간 소요
```

## 3. 실제 사용 시나리오

### 시나리오 1: 랜덤 이미지 생성 및 편집

**AAE:**
```python
z = torch.randn([1, 128])
img = decoder(z)  # 생성
img_edited = decoder(z + attr_vector)  # 편집
```

**StyleGAN2-ADA:**
```python
z = torch.randn([1, 512])
w = G.mapping(z)
img = G.synthesis(w)  # 생성
img_edited = G.synthesis(w + attr_vector)  # 편집
```

**결론:** 둘 다 동일하게 가능 ✅

### 시나리오 2: 실제 이미지 편집

**AAE:**
```python
z = encoder(real_image)  # 즉시
img_edited = decoder(z + attr_vector)  # 즉시
# 총 시간: < 1초
```

**StyleGAN2-ADA:**
```python
w = project(G, real_image)  # ⚠️ 30초~수분 소요
img_edited = G.synthesis(w + attr_vector)  # 즉시
# 총 시간: 30초~수분
```

**결론:** AAE가 훨씬 빠름 ⚠️

## 4. 코드 예시

### AAE 속성 제어 (기존 코드)
```python
# experiment_aae.py에서
z_target = encoder(target_img)
z_smile = z_target + smile_vector * 1.5
img_smile = decoder(z_smile)
```

### StyleGAN2-ADA 속성 제어 (새로 추가한 코드)
```python
# experiment_stylegan_attribute_control.py에서
w = project(G, target_img)  # projection 필요
w_smile = w + smile_vector * 1.5
img_smile = G.synthesis(w_smile)
```

## 5. 정확한 차이점 요약

| 항목 | AAE/VAE | StyleGAN2-ADA |
|------|---------|---------------|
| **랜덤 벡터 수정** | ✅ 가능 | ✅ 가능 |
| **실제 이미지 편집** | ✅ 가능 (즉시) | ✅ 가능 (projection 필요) |
| **속성 벡터 계산** | ✅ 빠름 (Encoder 사용) | ⚠️ 느림 (Projection 필요) |
| **편집 속도** | ✅ 빠름 (< 1초) | ⚠️ 느림 (30초~수분) |
| **이미지 품질** | 보통 | ⭐ 매우 높음 |
| **W space** | ❌ 없음 | ✅ 있음 (더 풍부) |

## 6. 결론

**StyleGAN2-ADA도 벡터 수정이 가능합니다!**

다만:
- ✅ **랜덤 벡터 기반 편집**: AAE와 동일하게 즉시 가능
- ⚠️ **실제 이미지 편집**: Projection이 필요해서 시간이 걸림
- ✅ **이미지 품질**: StyleGAN2-ADA가 더 높음

**사용 권장:**
- **빠른 프로토타이핑, 실제 이미지 편집** → AAE
- **고품질 생성, 랜덤 벡터 편집** → StyleGAN2-ADA

## 7. 추가된 스크립트

`experiment_stylegan_attribute_control.py`를 추가했습니다. 이 스크립트는 AAE와 동일한 방식으로 StyleGAN2-ADA에서 속성 제어를 수행합니다.

```bash
python src/experiment_stylegan_attribute_control.py
```

**주의:** Projection 과정이 시간이 오래 걸릴 수 있습니다 (이미지당 30초~수분).

