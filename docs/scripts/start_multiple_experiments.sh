#!/bin/bash
# 여러 속성 실험을 백그라운드로 시작하는 스크립트

cd "$(dirname "$0")/../.."

echo "============================================================"
echo "여러 속성 실험 시작"
echo "============================================================"
echo ""

# 실험할 속성 목록 (8시간 내 완료 가능한 개수로 제한)
ATTRIBUTES=(
    "Eyeglasses"
    "Male"
    "Young"
    "Blond_Hair"
    "Wearing_Hat"
    "Wearing_Lipstick"
    "Mustache"
    "Bald"
    "Attractive"
    "Heavy_Makeup"
)

echo "실험할 속성: ${#ATTRIBUTES[@]}개"
for attr in "${ATTRIBUTES[@]}"; do
    echo "  - $attr"
done
echo ""

# 각 속성에 대해 백그라운드로 실행
for attr in "${ATTRIBUTES[@]}"; do
    echo "시작: $attr"
    nohup python src/main_stylegan.py --attr "$attr" > "output/stylegan_${attr}_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
    sleep 5  # 시스템 부하 방지를 위한 대기
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
echo "  tail -f output/stylegan_*.log"
echo ""
