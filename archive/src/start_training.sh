#!/bin/bash
# StyleGAN2-ADA 학습 자동 시작 스크립트
# 데이터셋 변환부터 학습까지 자동으로 수행

cd /root/workspace/andycho/GenDL-LatentControl

# 가상환경 활성화
source /root/anaconda3/etc/profile.d/conda.sh
conda activate latent-control

echo "=========================================="
echo "StyleGAN2-ADA Training Setup"
echo "=========================================="

# 1. 데이터셋 ZIP 파일 확인
if [ ! -f "dataset/celebA.zip" ]; then
    echo "[Step 1] Converting CelebA to ZIP format..."
    echo "This may take a while (10-30 minutes)..."
    
    python stylegan2_ada_pytorch/dataset_tool.py \
        --source=dataset/celebA/img_align_celeba/img_align_celeba \
        --dest=dataset/celebA.zip \
        --resolution=128x128
    
    if [ $? -eq 0 ]; then
        echo "[Step 1] Dataset conversion completed!"
    else
        echo "[Error] Dataset conversion failed!"
        exit 1
    fi
else
    echo "[Step 1] Dataset ZIP file already exists!"
fi

# 2. 학습 시작
echo ""
echo "[Step 2] Starting StyleGAN2-ADA training..."
echo "Training will run in background"
echo ""

python src/train_stylegan.py \
    --epochs 30 \
    --background \
    --log output/stylegan2_ada_training/training.log

echo ""
echo "=========================================="
echo "Training started!"
echo "=========================================="
echo "Monitor: tail -f output/stylegan2_ada_training/training.log"
echo "Check PID: cat output/stylegan2_ada_training/training.pid"
echo "GPU usage: watch -n 1 nvidia-smi"
echo "=========================================="

