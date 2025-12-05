#!/bin/bash
# 진행 상황 모니터링 스크립트

cd /root/workspace/andycho/GenDL-LatentControl

echo "=========================================="
echo "StyleGAN2-ADA 진행 상황 모니터링"
echo "=========================================="
echo ""

# 이미지 변환 진행률
if [ -d "dataset/celebA/img_align_celeba_square_128" ]; then
    COUNT=$(find dataset/celebA/img_align_celeba_square_128 -name "*.jpg" 2>/dev/null | wc -l)
    TOTAL=202599  # CelebA 총 이미지 수
    PERCENT=$(echo "scale=1; $COUNT * 100 / $TOTAL" | bc)
    echo "[1] 이미지 정사각형 변환:"
    echo "  진행: $COUNT / $TOTAL ($PERCENT%)"
    
    if ps aux | grep prepare_celeba_square | grep -v grep > /dev/null; then
        echo "  상태: ✅ 진행 중"
    else
        echo "  상태: ✅ 완료"
    fi
else
    echo "[1] 이미지 정사각형 변환:"
    echo "  진행: 0 / $TOTAL (0%)"
    echo "  상태: ⏳ 시작 안됨"
fi
echo ""

# ZIP 파일
echo "[2] ZIP 파일:"
if [ -f "dataset/celebA.zip" ]; then
    SIZE=$(du -h dataset/celebA.zip | cut -f1)
    SIZE_BYTES=$(stat -c%s dataset/celebA.zip 2>/dev/null || echo 0)
    if [ "$SIZE_BYTES" -gt 1000000 ]; then
        echo "  상태: ✅ 정상 ($SIZE)"
    else
        echo "  상태: ⚠️  비정상 ($SIZE) - 재생성 필요"
    fi
else
    echo "  상태: ⏳ 없음"
fi
echo ""

# 학습
echo "[3] 학습:"
if ps aux | grep "train.py" | grep stylegan | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "train.py" | grep stylegan | grep -v grep | awk '{print $2}')
    echo "  상태: ✅ 실행 중 (PID: $PID)"
    if ls output/stylegan2_ada_training/training*.log 1>/dev/null 2>&1; then
        LOG_FILE=$(ls -t output/stylegan2_ada_training/training*.log | head -1)
        echo "  로그: $LOG_FILE"
        echo "  최신:"
        tail -n 2 "$LOG_FILE" | sed 's/^/    /'
    fi
elif ps aux | grep wait_and_train | grep -v grep > /dev/null; then
    echo "  상태: ⏳ 대기 중 (ZIP 파일 완료 대기)"
else
    echo "  상태: ❌ 대기 프로세스 없음"
fi
echo ""

# GPU
echo "[4] GPU 사용률:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s%% | 메모리 %s/%s MB\n", $1, $2, $3, $4}'
echo ""

echo "=========================================="

