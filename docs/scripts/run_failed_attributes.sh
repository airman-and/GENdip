#!/bin/bash
# 실패한 9개 속성 재실행

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "실패한 속성 재실행 시작"
echo "============================================================"
echo ""

# 실패한 속성 목록
FAILED_ATTRIBUTES=(
    "Eyeglasses"
    "Male"
    "Young"
    "Blond_Hair"
    "Wearing_Hat"
    "Wearing_Lipstick"
    "Mustache"
    "Bald"
    "Attractive"
)

echo "실험 설정:"
echo "  - 속성 개수: ${#FAILED_ATTRIBUTES[@]}개"
echo "  - 샘플 수: 100개 (속성당, 고품질)"
echo "  - 테스트 이미지: 16개 (속성당)"
echo ""

# 각 속성에 대해 백그라운드로 실행
for attr in "${FAILED_ATTRIBUTES[@]}"; do
    echo "시작: $attr (재실행)"
    nohup python src/main_stylegan.py \
        --attr "$attr" \
        --num-samples 100 \
        --num-test-images 16 \
        --projection-steps-attr 300 \
        --projection-steps-test 500 \
        > "output/stylegan_${attr}_retry_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    sleep 1  # 시스템 부하 방지를 위한 대기
done

echo ""
echo "============================================================"
echo "모든 실험이 백그라운드로 시작되었습니다!"
echo "============================================================"
echo ""
echo "진행 상황 확인:"
echo "  python src/check_experiment_status.py"
echo "  nvidia-smi"
echo ""
