#!/bin/bash
# 완전한 파이프라인: 이미지 변환 → ZIP 변환 → 학습 시작

cd /root/workspace/andycho/GenDL-LatentControl

source /root/anaconda3/etc/profile.d/conda.sh
conda activate latent-control

echo "=========================================="
echo "StyleGAN2-ADA 완전 자동화 파이프라인"
echo "=========================================="
echo ""

# 1. 이미지 정사각형 변환 확인/실행
SQUARE_DIR="dataset/celebA/img_align_celeba_square_128"
SOURCE_DIR="dataset/celebA/img_align_celeba/img_align_celeba"

if [ ! -d "$SQUARE_DIR" ] || [ $(ls $SQUARE_DIR/*.jpg 2>/dev/null | wc -l) -lt 100000 ]; then
    echo "[Step 1] 이미지를 정사각형으로 변환..."
    echo "이 작업은 시간이 걸립니다 (약 30분-1시간)..."
    
    python src/prepare_celeba_square.py
    
    if [ $? -ne 0 ]; then
        echo "[Error] 이미지 변환 실패"
        exit 1
    fi
    echo "[Step 1] ✅ 완료"
else
    echo "[Step 1] ✅ 정사각형 이미지 이미 존재"
fi

echo ""

# 2. ZIP 파일 변환
if [ ! -f "dataset/celebA.zip" ] || [ $(stat -f%z dataset/celebA.zip 2>/dev/null || stat -c%s dataset/celebA.zip 2>/dev/null) -lt 1000000 ]; then
    echo "[Step 2] ZIP 파일로 변환..."
    
    python stylegan2_ada_pytorch/dataset_tool.py \
        --source=$SQUARE_DIR \
        --dest=dataset/celebA.zip \
        > output/dataset_conversion.log 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[Error] ZIP 변환 실패"
        tail -20 output/dataset_conversion.log
        exit 1
    fi
    echo "[Step 2] ✅ 완료"
else
    echo "[Step 2] ✅ ZIP 파일 이미 존재"
fi

echo ""

# 3. 학습 시작
echo "[Step 3] 학습 시작..."
python src/train_stylegan.py \
    --epochs 30 \
    --background \
    --log output/stylegan2_ada_training/training.log

echo ""
echo "=========================================="
echo "파이프라인 완료!"
echo "=========================================="
echo "모니터링: tail -f output/stylegan2_ada_training/training.log"
echo "=========================================="

