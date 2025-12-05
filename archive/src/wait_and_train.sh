#!/bin/bash
# 데이터셋 변환 완료를 기다린 후 학습 시작

cd /root/workspace/andycho/GenDL-LatentControl

source /root/anaconda3/etc/profile.d/conda.sh
conda activate latent-control

echo "=========================================="
echo "Waiting for dataset conversion..."
echo "=========================================="

# 정사각형 이미지 변환 완료 대기
SQUARE_DIR="dataset/celebA/img_align_celeba_square_128"
EXPECTED_COUNT=200000

echo "Waiting for square image conversion..."
while [ $(find $SQUARE_DIR -name "*.jpg" 2>/dev/null | wc -l) -lt $EXPECTED_COUNT ]; do
    COUNT=$(find $SQUARE_DIR -name "*.jpg" 2>/dev/null | wc -l)
    echo "⏳ Square images: $COUNT / $EXPECTED_COUNT (checking every 60 seconds)"
    sleep 60
done

echo "✅ Square image conversion completed!"
echo ""

# ZIP 파일 변환 (정사각형 이미지 사용)
if [ ! -f "dataset/celebA.zip" ] || [ $(stat -c%s dataset/celebA.zip 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo "Converting square images to ZIP..."
    python stylegan2_ada_pytorch/dataset_tool.py \
        --source=$SQUARE_DIR \
        --dest=dataset/celebA.zip \
        > output/dataset_conversion.log 2>&1
    
    if [ $? -ne 0 ]; then
        echo "[Error] ZIP conversion failed"
        tail -20 output/dataset_conversion.log
        exit 1
    fi
    echo "✅ ZIP conversion completed!"
fi

echo ""

# 학습 시작
echo "=========================================="
echo "Starting StyleGAN2-ADA training..."
echo "=========================================="

python src/train_stylegan.py \
    --epochs 30 \
    --background \
    --log output/stylegan2_ada_training/training.log

echo ""
echo "=========================================="
echo "Training started in background!"
echo "=========================================="
echo "Monitor: tail -f output/stylegan2_ada_training/training.log"
echo "Check PID: cat output/stylegan2_ada_training/training.pid"
echo "GPU: watch -n 1 nvidia-smi"
echo "=========================================="

