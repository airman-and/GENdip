#!/bin/bash
# 8시간 정도 걸릴 실험을 실행하는 스크립트

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "8시간 실험 시작"
echo "============================================================"
echo ""

# 실험할 속성 목록 (14개 속성, 각각 약 34분)
ATTRIBUTES=(
    "Smiling"
    "Eyeglasses"
    "Male"
    "Young"
    "Blond_Hair"
    "Black_Hair"
    "Wearing_Hat"
    "Wearing_Lipstick"
    "Mustache"
    "Goatee"
    "Bald"
    "Attractive"
    "Heavy_Makeup"
    "Bangs"
)

echo "실험 설정:"
echo "  - 속성 개수: ${#ATTRIBUTES[@]}개"
echo "  - 샘플 수: 50개 (속성당)"
echo "  - 테스트 이미지: 8개 (속성당)"
echo "  - 예상 시간: 약 8시간"
echo ""

# 각 속성에 대해 백그라운드로 실행
for attr in "${ATTRIBUTES[@]}"; do
    echo "시작: $attr"
    nohup python src/main_stylegan.py \
        --attr "$attr" \
        --num-samples 50 \
        --num-test-images 8 \
        > "output/stylegan_${attr}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    sleep 10  # 시스템 부하 방지를 위한 대기
done

echo ""
echo "============================================================"
echo "모든 실험이 백그라운드로 시작되었습니다!"
echo "============================================================"
echo ""
echo "진행 상황 확인:"
echo "  python src/check_experiment_status.py"
echo ""
echo "또는:"
echo "  ps aux | grep main_stylegan"
echo "  nvidia-smi"
echo ""

