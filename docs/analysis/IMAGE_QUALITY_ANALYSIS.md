# 이미지 품질 저하 원인 분석

## 문제 현상
Scale 값이 증가할수록 (1.0 → 2.0 → 3.0) 이미지가 점점 흉측해지는 현상

## 주요 원인

### 1. **Latent Space에서 너무 멀리 이동**
- StyleGAN의 W space는 학습된 데이터 분포 주변에서만 현실적인 이미지를 생성
- Attribute vector를 큰 scale로 곱하면, 원본 이미지의 latent code에서 너무 멀리 떨어진 영역으로 이동
- 이 영역은 모델이 학습하지 않은 비현실적인 영역일 가능성이 높음

### 2. **Attribute Vector의 크기 문제**
- 현재 코드: `w_modified = w_mean + attr_expanded * scale`
- Scale이 3.0일 때, attribute vector가 3배로 확대되어 latent space에서 큰 이동 발생
- Attribute vector 자체의 크기가 클 경우, 작은 scale에서도 왜곡 발생 가능

### 3. **W Space의 모든 레이어에 동일하게 적용**
- 현재 코드는 모든 레이어(num_ws)에 동일한 attribute vector를 적용
- 하지만 StyleGAN의 각 레이어는 다른 수준의 특징을 제어 (저해상도 → 고해상도)
- 모든 레이어에 동일하게 적용하면 일관성 없는 변화 발생 가능

### 4. **Truncation Trick 미사용**
- StyleGAN에서는 truncation trick을 사용하여 latent code를 평균에 가깝게 제한
- 이를 통해 더 현실적인 이미지 생성 가능
- 현재 코드에서는 truncation을 사용하지 않아 비현실적인 영역으로 이동 가능

## 개선 방안

### 방안 1: Attribute Vector 정규화
```python
# Attribute vector의 크기를 정규화하여 일정 범위로 제한
attr_norm = torch.norm(attribute_vector)
if attr_norm > threshold:
    attribute_vector = attribute_vector / attr_norm * threshold
```

### 방안 2: Truncation Trick 적용
```python
# W space의 평균을 계산하고, 편차를 제한
w_avg = compute_w_avg(G, num_samples=10000)
w_modified = w_avg + truncation_psi * (w_modified - w_avg)
```

### 방안 3: 레이어별 다른 Scale 적용
```python
# 저해상도 레이어에는 작은 scale, 고해상도 레이어에는 큰 scale
# 또는 특정 레이어에만 적용
for i in range(G.mapping.num_ws):
    if i < num_layers_to_modify:
        w_modified[0, i] += attribute_vector * scale
```

### 방안 4: Interpolation 사용
```python
# 원본과 수정된 latent code 사이를 interpolation
w_modified = (1 - alpha) * w_mean + alpha * (w_mean + attr_expanded * scale)
```

### 방안 5: Scale 범위 조정
- 현재: [1.0, 2.0, 3.0]
- 제안: [0.5, 1.0, 1.5] 또는 [0.3, 0.6, 1.0]
- 더 작은 scale로 시작하여 점진적으로 증가

## 권장 해결책

**즉시 적용 가능한 방법:**
1. Scale 범위를 줄임: [0.5, 1.0, 1.5] 또는 [0.3, 0.6, 0.9]
2. Attribute vector 정규화 추가
3. Truncation trick 적용 (w_avg 계산 필요)

**더 나은 품질을 위한 방법:**
1. 레이어별 다른 scale 적용
2. Interpolation 기반 접근
3. 더 많은 샘플로 attribute vector 추출 (현재 50개 → 100개 이상)

