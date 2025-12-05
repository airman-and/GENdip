#!/bin/bash
mkdir -p output/logs

echo "[$(date)] Starting Male experiment..."
python src/main_stylegan.py --attr Male --num-samples 200 --num-test-images 8 > output/logs/stylegan_male.log 2>&1
echo "[$(date)] Male experiment finished."

echo "[$(date)] Starting Young experiment..."
python src/main_stylegan.py --attr Young --num-samples 200 --num-test-images 8 > output/logs/stylegan_young.log 2>&1
echo "[$(date)] Young experiment finished."

echo "[$(date)] Starting Blond_Hair experiment..."
python src/main_stylegan.py --attr Blond_Hair --num-samples 200 --num-test-images 8 > output/logs/stylegan_blond.log 2>&1
echo "[$(date)] Blond_Hair experiment finished."
