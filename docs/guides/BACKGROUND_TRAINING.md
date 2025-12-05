# 백그라운드 학습 가이드

## 가상환경

현재 사용 중인 가상환경: **latent-control**

```bash
conda activate latent-control
```

## 백그라운드 실행 방법

### 방법 1: Python 스크립트 직접 사용

```bash
cd /root/workspace/andycho/GenDL-LatentControl
conda activate latent-control

# 백그라운드 실행
python src/train_stylegan.py --epochs 30 --background
```

### 방법 2: 쉘 스크립트 사용 (권장)

```bash
./src/train_stylegan_bg.sh
```

### 방법 3: nohup 사용

```bash
conda activate latent-control
nohup python src/train_stylegan.py --epochs 30 --background > training.log 2>&1 &
```

## 로그 파일

### 자동 생성
- 기본 위치: `output/stylegan2_ada_training/training_YYYYMMDD_HHMMSS.log`
- 커스텀 경로: `--log path/to/logfile.log`

### 로그 내용
- 시작/종료 시간
- GPU 정보
- 학습 설정
- 모든 출력 (stdout, stderr)

### 로그 모니터링

```bash
# 실시간 로그 확인
tail -f output/stylegan2_ada_training/training_*.log

# 마지막 100줄 확인
tail -n 100 output/stylegan2_ada_training/training_*.log

# 특정 키워드 검색
grep "kimg" output/stylegan2_ada_training/training_*.log
```

## 프로세스 관리

### PID 확인

```bash
# PID 파일에서 확인
cat output/stylegan2_ada_training/training_*.pid

# 또는 프로세스 검색
ps aux | grep train_stylegan
```

### 프로세스 상태 확인

```bash
# GPU 사용률 확인
watch -n 1 nvidia-smi

# 프로세스 상태
ps -p $(cat output/stylegan2_ada_training/training_*.pid)
```

### 프로세스 중지

```bash
# PID 파일에서 PID 읽어서 종료
kill $(cat output/stylegan2_ada_training/training_*.pid)

# 또는 프로세스 검색해서 종료
pkill -f train_stylegan.py
```

## 학습 진행 상황 확인

### StyleGAN2-ADA 자체 로그

```bash
# 학습 진행 상황 (JSON 형식)
cat output/stylegan2_ada_training/training_options.json

# 메트릭 (FID 등)
cat output/stylegan2_ada_training/metric-*.json
```

### 체크포인트 확인

```bash
# 체크포인트 목록
ls -lh output/stylegan2_ada_training/network-snapshot-*.pkl

# 최신 체크포인트
ls -t output/stylegan2_ada_training/network-snapshot-*.pkl | head -1
```

## 예제 사용법

### 빠른 테스트 (15 epochs, 백그라운드)

```bash
python src/train_stylegan.py \
    --epochs 15 \
    --background \
    --log output/stylegan2_ada_training/test.log
```

### 전체 학습 (30 epochs, 백그라운드)

```bash
python src/train_stylegan.py \
    --epochs 30 \
    --background \
    --log output/stylegan2_ada_training/full_training.log
```

### 커스텀 배치 크기

```bash
python src/train_stylegan.py \
    --epochs 30 \
    --batch-size 128 \
    --background
```

## 문제 해결

### 로그 파일이 생성되지 않음
- 출력 디렉토리 권한 확인
- 디스크 공간 확인

### 프로세스가 죽음
- 로그 파일에서 에러 확인
- GPU 메모리 부족 확인 (OOM)
- `nvidia-smi`로 GPU 상태 확인

### 학습이 너무 느림
- GPU 사용률 확인 (`nvidia-smi`)
- 배치 크기 확인
- Workers 수 확인

## 체크리스트

학습 시작 전:
- [ ] 가상환경 활성화 확인
- [ ] 데이터셋 ZIP 파일 존재 확인
- [ ] GPU 사용 가능 확인
- [ ] 디스크 공간 충분한지 확인
- [ ] 로그 파일 경로 확인

학습 중:
- [ ] 로그 파일 모니터링
- [ ] GPU 사용률 확인
- [ ] 체크포인트 저장 확인

