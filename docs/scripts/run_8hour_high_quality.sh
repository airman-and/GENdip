#!/bin/bash
# 8시간 고품질 실험 (더 많은 샘플과 테스트 이미지)

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "8시간 고품질 실험 시작"
echo "============================================================"
echo ""

# 실험할 속성 목록 (10개 속성, 각각 더 많은 샘플 사용)
ATTRIBUTES=(
    "Smiling"
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
echo "  - 속성 개수: ${#ATTRIBUTES[@]}개"
echo "  - 샘플 수: 100개 (속성당, 고품질)"
echo "  - 테스트 이미지: 16개 (속성당)"
echo "  - 예상 시간: 약 8시간"
echo ""

# 각 속성에 대해 백그라운드로 실행
for attr in "${ATTRIBUTES[@]}"; do
    echo "시작: $attr (고품질 모드)"
    nohup python src/main_stylegan.py \
        --attr "$attr" \
        --num-samples 100 \
        --num-test-images 16 \
        --projection-steps-attr 300 \
        --projection-steps-test 500 \
        > "output/stylegan_${attr}_hq_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    sleep 2  # 시스템 부하 방지를 위한 최소 대기
done

echo ""
echo "============================================================"
echo "모든 고품질 실험이 백그라운드로 시작되었습니다!"
echo "============================================================"
echo ""
echo "진행 상황 확인:"
echo "  python src/check_experiment_status.py"
echo ""

