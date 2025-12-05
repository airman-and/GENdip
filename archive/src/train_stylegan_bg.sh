#!/bin/bash
# StyleGAN2-ADA 백그라운드 학습 실행 스크립트

cd /root/workspace/andycho/GenDL-LatentControl

# 가상환경 활성화
source /root/anaconda3/etc/profile.d/conda.sh
conda activate latent-control

# 백그라운드 실행
python src/train_stylegan.py \
    --epochs 30 \
    --background \
    --log output/stylegan2_ada_training/training.log

echo "Training started in background"
echo "Monitor: tail -f output/stylegan2_ada_training/training.log"
echo "Check PID: cat output/stylegan2_ada_training/training.pid"

