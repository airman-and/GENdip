# 문서 및 스크립트 가이드

이 폴더는 프로젝트의 모든 문서와 스크립트를 체계적으로 정리한 디렉토리입니다.

## 폴더 구조

```
docs/
├── guides/          # 사용 가이드 및 튜토리얼
├── analysis/        # 실험 분석 및 비교 문서
└── scripts/         # 실행 스크립트
```

## 📚 가이드 문서 (guides/)

### 기본 가이드
- **LOG_GUIDE.md** - 로그 파일 확인 및 모니터링 가이드
- **TRAINING_FLOW.md** - 학습 프로세스 전체 흐름
- **BACKGROUND_TRAINING.md** - 백그라운드 학습 방법
- **STYLEGAN_USAGE.md** - StyleGAN 사용법
- **STYLEGAN_PIPELINE.md** - StyleGAN 파이프라인 설명

### 설정 및 최적화
- **GPU_OPTIMIZATION.md** - GPU 최적화 방법
- **TRAINING_TIME_ESTIMATE.md** - 학습 시간 예측
- **SCALE_EXPLANATION.md** - Scale 파라미터 설명

## 📊 분석 문서 (analysis/)

- **EXPERIMENT_COMPARISON.md** - 실험 결과 비교
- **IMAGE_QUALITY_ANALYSIS.md** - 이미지 품질 분석
- **STYLEGAN_VS_AAE_ATTRIBUTE.md** - StyleGAN vs AAE 속성 제어 비교
- **analysis_resnet_vs_cnn.md** - ResNet vs CNN 분석

## 🔧 스크립트 (scripts/)

### 실험 실행 스크립트
- **run_8hour_experiment.sh** - 8시간 실험 실행
- **run_8hour_high_quality.sh** - 고품질 8시간 실험
- **run_failed_attributes.sh** - 실패한 속성 재실행
- **start_multiple_experiments.sh** - 다중 실험 시작

### 유틸리티 스크립트
- **STATUS_CHECK.sh** - 실험 진행 상황 확인

## 빠른 참조

### 실험 시작하기
```bash
# 기본 실험
./docs/scripts/run_8hour_experiment.sh

# 고품질 실험
./docs/scripts/run_8hour_high_quality.sh

# 상태 확인
./docs/scripts/STATUS_CHECK.sh
```

### 문서 읽기
```bash
# 학습 가이드
cat docs/guides/TRAINING_FLOW.md

# 로그 확인 방법
cat docs/guides/LOG_GUIDE.md

# 실험 결과 비교
cat docs/analysis/EXPERIMENT_COMPARISON.md
```

## 메인 README

프로젝트 루트의 `README.md`를 참고하세요.




