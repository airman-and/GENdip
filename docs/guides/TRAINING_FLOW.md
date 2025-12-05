# StyleGAN2-ADA 학습 작업 흐름 (Flow)

## 전체 작업 흐름 다이어그램

```
[시작]
  │
  ├─→ [1단계] 데이터셋 변환 (dataset_tool.py)
  │     │
  │     ├─ CelebA 이미지 폴더 읽기
  │     ├─ 128x128로 리사이즈
  │     ├─ ZIP 형식으로 압축
  │     └─ dataset/celebA.zip 생성
  │           │
  │           └─→ [완료 대기] wait_and_train.sh
  │                 │
  │                 ├─ 30초마다 ZIP 파일 확인
  │                 └─ ZIP 파일 발견 시 → [2단계]
  │
  ├─→ [2단계] 학습 시작 (train_stylegan.py)
  │     │
  │     ├─ GPU 정보 확인
  │     │   ├─ GPU 개수 감지 (2개)
  │     │   ├─ GPU 메모리 확인 (50GB)
  │     │   └─ 최적 배치 크기 계산 (256)
  │     │
  │     ├─ 학습 설정
  │     │   ├─ kimg 계산 (300)
  │     │   ├─ Workers 수 계산
  │     │   └─ 로그 파일 생성
  │     │
  │     ├─ 백그라운드 실행
  │     │   ├─ subprocess.Popen() 실행
  │     │   ├─ PID 파일 저장
  │     │   └─ 로그 파일에 모든 출력 기록
  │     │
  │     └─→ [3단계] StyleGAN2-ADA 학습
  │
  └─→ [3단계] StyleGAN2-ADA 실제 학습 (train.py)
        │
        ├─ [초기화]
        │   ├─ Generator 생성
        │   ├─ Discriminator 생성
        │   ├─ Optimizer 설정
        │   └─ DataLoader 설정
        │
        ├─ [학습 루프] (300 kimg = 300,000 이미지)
        │   │
        │   ├─ [각 iteration]
        │   │   ├─ 배치 데이터 로드 (256 images)
        │   │   ├─ Discriminator 학습
        │   │   │   ├─ Real images → D(real)
        │   │   │   ├─ Fake images → D(fake)
        │   │   │   ├─ Gradient penalty 계산
        │   │   │   └─ D loss backward
        │   │   │
        │   │   └─ Generator 학습
        │   │       ├─ Random z → Generator → Fake images
        │   │       ├─ D(fake) 계산
        │   │       └─ G loss backward
        │   │
        │   ├─ [50 ticks마다]
        │   │   ├─ 체크포인트 저장
        │   │   │   └─ network-snapshot-XXXXXX.pkl
        │   │   └─ 샘플 이미지 저장
        │   │
        │   └─ [진행 상황 로깅]
        │       ├─ JSON 로그 (training_options.json)
        │       ├─ 메트릭 로그 (metric-*.json)
        │       └─ 텍스트 로그 (training.log)
        │
        └─ [완료]
            ├─ 최종 체크포인트 저장
            ├─ 로그 파일 종료 기록
            └─ 프로세스 종료
```

## 단계별 상세 설명

### 1단계: 데이터셋 변환

**프로세스**: `dataset_tool.py` (백그라운드)

```bash
python stylegan2_ada_pytorch/dataset_tool.py \
    --source=dataset/celebA/img_align_celeba/img_align_celeba \
    --dest=dataset/celebA.zip \
    --resolution=128x128
```

**작업 내용**:
1. CelebA 이미지 폴더 스캔 (~200k 이미지)
2. 각 이미지를 128x128로 리사이즈
3. ZIP 형식으로 압축
4. `dataset/celebA.zip` 생성

**소요 시간**: 10-30분
**로그**: `output/dataset_conversion.log`

---

### 2단계: 학습 준비 및 시작

**프로세스**: `wait_and_train.sh` → `train_stylegan.py`

#### 2-1. 대기 프로세스 (`wait_and_train.sh`)

```bash
while [ ! -f "dataset/celebA.zip" ]; do
    sleep 30  # 30초마다 확인
done
```

**작업 내용**:
- 30초마다 `dataset/celebA.zip` 파일 존재 확인
- 파일이 생성되면 `train_stylegan.py` 실행

#### 2-2. 학습 스크립트 (`train_stylegan.py`)

**작업 내용**:

1. **GPU 최적화**
   ```python
   - GPU 개수 감지: 2개
   - GPU 메모리 확인: 50GB
   - 배치 크기 자동 계산: 256 (128 per GPU)
   ```

2. **학습 설정**
   ```python
   - kimg: 300 (30 epochs × 10)
   - Workers: CPU 코어 수에 따라
   - 로그 파일: training_YYYYMMDD_HHMMSS.log
   ```

3. **백그라운드 실행**
   ```python
   subprocess.Popen(
       cmd,
       stdout=log_file,  # 모든 출력을 로그에 기록
       stderr=subprocess.STDOUT,
       preexec_fn=os.setsid  # 새로운 프로세스 그룹
   )
   ```

4. **PID 저장**
   - `training_YYYYMMDD_HHMMSS.pid` 파일에 프로세스 ID 저장

---

### 3단계: StyleGAN2-ADA 실제 학습

**프로세스**: `stylegan2_ada_pytorch/train.py` (백그라운드)

#### 3-1. 초기화

```
- Generator (G) 생성
- Discriminator (D) 생성
- Optimizer 설정 (Adam)
- DataLoader 설정 (배치 256, workers 8)
```

#### 3-2. 학습 루프 (300 kimg)

**각 iteration (배치 256)**:

```
1. 데이터 로드
   └─ 256개 이미지 배치 로드

2. Discriminator 학습
   ├─ Real images → D(real) → Real loss
   ├─ Fake images (G(z)) → D(fake) → Fake loss
   ├─ Gradient penalty 계산
   └─ D_loss = -real_loss + fake_loss + gradient_penalty

3. Generator 학습
   ├─ Random z (512-dim) 샘플링
   ├─ z → Mapping Network → w (512-dim, 18 layers)
   ├─ w → Synthesis Network → Fake images
   ├─ D(fake) → G_loss
   └─ G_loss backward

4. EMA 업데이트
   └─ G_ema = 0.999 × G_ema + 0.001 × G
```

**50 ticks마다 (약 50-60분)**:

```
1. 체크포인트 저장
   └─ network-snapshot-XXXXXX.pkl
       ├─ G (현재 Generator)
       ├─ D (현재 Discriminator)
       └─ G_ema (EMA Generator) ⭐ 사용 권장

2. 샘플 이미지 생성
   └─ output/stylegan2_ada_training/images/...

3. 메트릭 계산 (선택적)
   └─ FID, IS 등
```

#### 3-3. 완료

```
- 최종 체크포인트 저장
- 로그 파일 종료 기록
- 프로세스 종료
```

---

## 파일 생성 흐름

### 데이터셋 변환 중
```
dataset/celebA.zip (생성 중...)
output/dataset_conversion.log (진행 상황)
```

### 학습 시작 후
```
output/stylegan2_ada_training/
├── training_YYYYMMDD_HHMMSS.log (전체 로그)
├── training_YYYYMMDD_HHMMSS.pid (프로세스 ID)
├── training_options.json (학습 설정)
├── network-snapshot-000050.pkl (체크포인트)
├── network-snapshot-000100.pkl
├── network-snapshot-000150.pkl
├── ...
└── network-snapshot-000300.pkl (최종)
```

---

## 프로세스 관계도

```
[터미널]
  │
  ├─→ [nohup] wait_and_train.sh (PID: XXXX)
  │     │
  │     └─→ [대기] 30초마다 ZIP 파일 확인
  │           │
  │           └─→ [실행] train_stylegan.py (PID: YYYY)
  │                 │
  │                 └─→ [백그라운드] train.py (PID: ZZZZ)
  │                       │
  │                       └─→ [학습 루프] 300 kimg
  │
  └─→ [백그라운드] dataset_tool.py (PID: AAAA)
        │
        └─→ [변환] CelebA → ZIP
```

---

## 모니터링 포인트

### 1. 데이터셋 변환 모니터링
```bash
# 진행 상황 확인
tail -f output/dataset_conversion.log

# 파일 크기 확인 (증가 중이면 진행 중)
ls -lh dataset/celebA.zip
```

### 2. 학습 시작 확인
```bash
# PID 파일 생성 확인
ls -lh output/stylegan2_ada_training/training_*.pid

# 로그 파일 확인
tail -f output/stylegan2_ada_training/training_*.log
```

### 3. 학습 진행 확인
```bash
# 체크포인트 생성 확인
ls -lt output/stylegan2_ada_training/network-snapshot-*.pkl

# GPU 사용률
watch -n 1 nvidia-smi

# 프로세스 확인
ps -p $(cat output/stylegan2_ada_training/training_*.pid)
```

---

## 예상 타임라인

```
T+0분    : 데이터셋 변환 시작
T+10-30분: 데이터셋 변환 완료 → 학습 시작
T+10-30분: 초기 체크포인트 (50 ticks)
T+60-90분: 두 번째 체크포인트 (100 ticks)
T+110-150분: 세 번째 체크포인트 (150 ticks)
...
T+240-360분: 최종 체크포인트 (300 kimg) → 완료
```

---

## 요약

1. **데이터셋 변환** (10-30분) → ZIP 파일 생성
2. **학습 준비** (즉시) → GPU 최적화, 설정 완료
3. **학습 시작** (자동) → 백그라운드 실행
4. **학습 진행** (4-6시간) → 50 ticks마다 체크포인트 저장
5. **학습 완료** → 최종 모델 저장

**모든 과정이 자동화되어 있어서, 한 번 시작하면 완료까지 자동으로 진행됩니다!**

