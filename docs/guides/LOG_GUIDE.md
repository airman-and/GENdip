# 로그 확인 가이드

## 로그 파일 위치

### 1. 데이터셋 변환 로그
**경로**: `output/dataset_conversion.log`

**내용**:
- 데이터셋 변환 진행 상황
- 에러 메시지 (있는 경우)

**확인 방법**:
```bash
# 실시간 확인
tail -f output/dataset_conversion.log

# 마지막 50줄
tail -n 50 output/dataset_conversion.log

# 전체 내용
cat output/dataset_conversion.log
```

---

### 2. 학습 로그 (메인)
**경로**: `output/stylegan2_ada_training/training_YYYYMMDD_HHMMSS.log`

**내용**:
- 학습 시작/종료 시간
- GPU 정보
- 학습 설정
- 모든 학습 출력 (stdout, stderr)
- StyleGAN2-ADA의 모든 출력

**확인 방법**:
```bash
# 실시간 확인 (가장 많이 사용)
tail -f output/stylegan2_ada_training/training_*.log

# 또는 최신 파일만
tail -f $(ls -t output/stylegan2_ada_training/training_*.log | head -1)

# 마지막 100줄
tail -n 100 output/stylegan2_ada_training/training_*.log

# 특정 키워드 검색
grep "kimg" output/stylegan2_ada_training/training_*.log
grep "error" output/stylegan2_ada_training/training_*.log -i
```

---

### 3. StyleGAN2-ADA 학습 설정 (JSON)
**경로**: `output/stylegan2_ada_training/training_options.json`

**내용**:
- 학습 설정 (배치 크기, kimg, gamma 등)
- JSON 형식

**확인 방법**:
```bash
# 전체 내용
cat output/stylegan2_ada_training/training_options.json

# 예쁘게 보기 (jq 필요)
cat output/stylegan2_ada_training/training_options.json | jq .

# Python으로 읽기
python -m json.tool output/stylegan2_ada_training/training_options.json
```

---

### 4. 메트릭 로그 (JSON)
**경로**: `output/stylegan2_ada_training/metric-*.json`

**내용**:
- FID, IS 등 메트릭 값
- 학습 진행에 따른 메트릭 변화

**확인 방법**:
```bash
# 모든 메트릭 파일
ls -lh output/stylegan2_ada_training/metric-*.json

# 최신 메트릭 확인
cat $(ls -t output/stylegan2_ada_training/metric-*.json | head -1) | jq .
```

---

## 빠른 확인 명령어

### 현재 진행 상황 한눈에 보기
```bash
cd /root/workspace/andycho/GenDL-LatentControl

echo "=== 데이터셋 변환 ==="
tail -n 5 output/dataset_conversion.log 2>/dev/null || echo "로그 없음"

echo ""
echo "=== 학습 로그 (최신 10줄) ==="
tail -n 10 output/stylegan2_ada_training/training_*.log 2>/dev/null || echo "학습 아직 시작 안됨"

echo ""
echo "=== 체크포인트 ==="
ls -lt output/stylegan2_ada_training/network-snapshot-*.pkl 2>/dev/null | head -3 || echo "체크포인트 없음"
```

---

## 실시간 모니터링

### 방법 1: tail -f (가장 많이 사용)
```bash
# 데이터셋 변환 모니터링
tail -f output/dataset_conversion.log

# 학습 모니터링
tail -f output/stylegan2_ada_training/training_*.log
```

### 방법 2: watch로 주기적 확인
```bash
# 5초마다 최신 20줄 확인
watch -n 5 'tail -n 20 output/stylegan2_ada_training/training_*.log 2>/dev/null || echo "학습 대기 중..."'
```

### 방법 3: 멀티 로그 모니터링
```bash
# 여러 로그를 동시에 보기
tail -f output/dataset_conversion.log output/stylegan2_ada_training/training_*.log
```

---

## 로그에서 중요한 정보 찾기

### 학습 진행 상황
```bash
# kimg 진행 상황
grep "kimg" output/stylegan2_ada_training/training_*.log | tail -5

# 체크포인트 저장 확인
grep "snapshot" output/stylegan2_ada_training/training_*.log | tail -5
```

### 에러 확인
```bash
# 에러 메시지 찾기
grep -i "error" output/stylegan2_ada_training/training_*.log
grep -i "exception" output/stylegan2_ada_training/training_*.log
grep -i "failed" output/stylegan2_ada_training/training_*.log
```

### GPU 사용률
```bash
# GPU 관련 정보
grep -i "gpu" output/stylegan2_ada_training/training_*.log
```

---

## 로그 파일 크기 확인

```bash
# 모든 로그 파일 크기
ls -lh output/dataset_conversion.log output/stylegan2_ada_training/*.log 2>/dev/null

# 로그 파일이 너무 크면
du -sh output/stylegan2_ada_training/*.log
```

---

## 로그 정리

로그 파일이 너무 커지면:

```bash
# 오래된 로그 백업
mkdir -p output/logs_backup
mv output/stylegan2_ada_training/training_*.log output/logs_backup/

# 또는 압축
gzip output/stylegan2_ada_training/training_*.log
```

---

## 요약: 가장 많이 사용하는 명령어

```bash
# 1. 학습 로그 실시간 확인 (가장 중요!)
tail -f output/stylegan2_ada_training/training_*.log

# 2. 데이터셋 변환 확인
tail -f output/dataset_conversion.log

# 3. 최신 체크포인트 확인
ls -lt output/stylegan2_ada_training/network-snapshot-*.pkl | head -1

# 4. GPU 사용률
watch -n 1 nvidia-smi
```

