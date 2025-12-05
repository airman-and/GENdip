#!/bin/bash
# 현재 작업 상태 확인 스크립트

cd /root/workspace/andycho/GenDL-LatentControl

echo "=========================================="
echo "StyleGAN2-ADA 학습 상태 확인"
echo "=========================================="
echo ""

# 1. 데이터셋 변환
echo "[1단계] 데이터셋 변환:"
if ps aux | grep "dataset_tool.py" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "dataset_tool.py" | grep -v grep | awk '{print $2}')
    echo "  ✅ 실행 중 (PID: $PID)"
    if [ -f "output/dataset_conversion.log" ]; then
        echo "  📝 최신 로그:"
        tail -n 3 output/dataset_conversion.log | sed 's/^/    /'
    fi
else
    echo "  ❌ 실행 중이지 않음"
    if [ -f "dataset/celebA.zip" ]; then
        echo "  ✅ ZIP 파일은 존재함 ($(du -h dataset/celebA.zip | cut -f1))"
    else
        echo "  ⚠️  ZIP 파일도 없음 - 변환 필요"
    fi
fi
echo ""

# 2. 학습 대기
echo "[2단계] 학습 대기 프로세스:"
if ps aux | grep wait_and_train | grep -v grep > /dev/null; then
    PID=$(ps aux | grep wait_and_train | grep -v grep | awk '{print $2}')
    echo "  ✅ 실행 중 (PID: $PID)"
    echo "  ⏳ 30초마다 ZIP 파일 확인 중"
else
    echo "  ❌ 실행 중이지 않음"
fi
echo ""

# 3. 학습
echo "[3단계] 학습:"
if ps aux | grep "train.py" | grep stylegan | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "train.py" | grep stylegan | grep -v grep | awk '{print $2}')
    echo "  ✅ 실행 중 (PID: $PID)"
    if ls output/stylegan2_ada_training/training*.log 1>/dev/null 2>&1; then
        LOG_FILE=$(ls -t output/stylegan2_ada_training/training*.log | head -1)
        echo "  📝 최신 로그: $LOG_FILE"
        tail -n 3 "$LOG_FILE" | sed 's/^/    /'
    fi
else
    echo "  ⏳ 아직 시작 안됨 (데이터셋 대기 중)"
fi
echo ""

# 4. GPU 상태
echo "[GPU 상태]"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s (%s): %s%% 사용 | 메모리 %s/%s MB\n", $1, $2, $3, $4, $5}'
echo ""

# 5. 체크포인트
echo "[체크포인트]"
if ls output/stylegan2_ada_training/network-snapshot-*.pkl 1>/dev/null 2>&1; then
    echo "  ✅ 체크포인트 발견:"
    ls -lth output/stylegan2_ada_training/network-snapshot-*.pkl | head -3 | \
        awk '{printf "    %s (%s) - %s\n", $9, $5, $6, $7, $8}'
else
    echo "  ⏳ 아직 생성 안됨"
fi
echo ""

# 6. 모니터링 명령어
echo "=========================================="
echo "모니터링 명령어"
echo "=========================================="
echo "데이터셋 변환: tail -f output/dataset_conversion.log"
echo "학습 로그: tail -f output/stylegan2_ada_training/training_*.log"
echo "GPU: watch -n 1 nvidia-smi"
echo "=========================================="

